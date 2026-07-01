from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import read_csv, write_json
from .v65_common import load_dict, project, rel, sha256_file, write_standard_outputs


FINAL_ROOT = "outputs/audit/v65_final"


def build_v65_final_decision() -> dict[str, Any]:
    ap = load_dict("outputs/audit/v65_ap_contract/ap_contract_summary.json")
    stream3d = load_dict("outputs/audit/v65_stream3d_parity/stream3d_parity_summary.json")
    failure = load_dict("outputs/audit/v65_ap_failure_decomp/failure_summary.json")
    geometry = load_dict("outputs/audit/v65_geometry_contract/geometry_contract_summary.json")
    aggregation = load_dict("outputs/audit/v65_instance_aggregation/aggregation_summary.json")
    visualization = load_dict("outputs/audit/v65_visualization/viser_scene_index.json")
    viser_server = load_dict("outputs/audit/v65_visualization/viser_server_status.json")
    casebook = load_dict("outputs/audit/v65_casebook/casebook_summary.json")
    dynamic = load_dict("outputs/audit/v65_dynamic_data/dynamic_data_summary.json")
    soma_policy = load_dict("outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_audit_summary.json")
    ap_rows = {row["row_id"]: row for row in read_csv(project("outputs/audit/v65_ap_contract/ap_contract_rows.csv"))}
    labels = _decision_labels(ap, stream3d, geometry, aggregation, visualization, dynamic, soma_policy)
    questions = _question_answers(
        ap_rows, ap, stream3d, failure, geometry, aggregation, visualization, casebook, dynamic, soma_policy
    )
    evidence = _evidence_hash_rows()
    summary = {
        "phase": "v65_final_decision",
        "decision_labels": labels,
        "questions": questions,
        "key_metrics": {
            "AP_comparison_allowed_pairs": ap.get("comparison_allowed_pairs"),
            "method_safe_rows_with_AP": ap.get("method_safe_rows_with_AP"),
            "frame_policy_blocks_bridge_stream3d_pairs": ap.get("frame_policy_blocks_bridge_stream3d_pairs"),
            "geometry_status": geometry.get("geometry_status"),
            "visualization_status": visualization.get("visualization_status"),
            "viser_server_smoke_pass": (viser_server.get("gate") or {}).get("pass"),
            "casebook_case_count": casebook.get("case_count"),
            "instance_aggregation_blocker": aggregation.get("blocker"),
            "dynamic_status": dynamic.get("dynamic_status"),
            "soma_no_gt_inference_policy_pass": (soma_policy.get("gate") or {}).get("pass"),
            "soma_policy_violation_count": soma_policy.get("policy_violation_count"),
            "soma_method_inference_gt_geometry_record_count": soma_policy.get(
                "method_inference_gt_geometry_record_count"
            ),
            "soma_gt_inference_record_count": soma_policy.get("gt_inference_record_count"),
        },
        "gate": {
            "AP_protocol_locked": "GO_V65_AP_PROTOCOL_LOCKED" in labels,
            "SOMA_no_GT_inference_policy_locked": "GO_SOMA_NO_GT_INFERENCE_POLICY" in labels,
            "final_questions_answered": len(questions) == 12,
            "evidence_hashes_recorded": all(row.get("sha256") for row in evidence),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "decision_label_rows": [{"decision_label": label} for label in labels],
        "final_question_rows": questions,
        "evidence_hash_rows": evidence,
        "final_report_md": _report_markdown(summary),
    }


def write_v65_final_decision(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "final_decision.json": payload["summary"],
            "decision_label_rows.csv": payload["decision_label_rows"],
            "final_question_rows.csv": payload["final_question_rows"],
            "evidence_hash_rows.csv": payload["evidence_hash_rows"],
            "v65_final_report.md": payload["final_report_md"],
        },
    )


def _decision_labels(
    ap: dict[str, Any],
    stream3d: dict[str, Any],
    geometry: dict[str, Any],
    aggregation: dict[str, Any],
    visualization: dict[str, Any],
    dynamic: dict[str, Any],
    soma_policy: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if (ap.get("gate") or {}).get("pass") and (stream3d.get("gate") or {}).get("pass"):
        labels.append("GO_V65_AP_PROTOCOL_LOCKED")
    else:
        labels.append("NO_GO_AP_PROTOCOL")
    if ap.get("method_safe_rows_with_AP"):
        labels.append("GO_SCANNET_METHOD_AP")
    else:
        labels.append("PARTIAL_SCANNET_DIAGNOSTIC_ONLY")
    labels.append(geometry.get("geometry_status") or "NO_GO_GEOMETRY_CONTRACT")
    labels.append(visualization.get("visualization_status") or "NO_GO_VISUALIZATION")
    if aggregation.get("non_gt_aggregation_available") and not aggregation.get("blocker"):
        labels.append("GO_INSTANCE_AGGREGATION_REPAIR")
    else:
        labels.append("NO_GO_INSTANCE_AGGREGATION")
    labels.append(dynamic.get("dynamic_status") or "NO_GO_DYNAMIC_DATA")
    if (soma_policy.get("gate") or {}).get("pass"):
        labels.append("GO_SOMA_NO_GT_INFERENCE_POLICY")
    else:
        labels.append("NO_GO_SOMA_NO_GT_INFERENCE_POLICY")
    return labels


def _question_answers(
    ap_rows: dict[str, dict[str, str]],
    ap: dict[str, Any],
    stream3d: dict[str, Any],
    failure: dict[str, Any],
    geometry: dict[str, Any],
    aggregation: dict[str, Any],
    visualization: dict[str, Any],
    casebook: dict[str, Any],
    dynamic: dict[str, Any],
    soma_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    a3 = ap_rows.get("A3", {})
    a4 = ap_rows.get("A4", {})
    a5 = ap_rows.get("A5", {})
    a7 = ap_rows.get("A7", {})
    rows = [
        (
            1,
            "0.313 diagnostic AP 是什么 support scope？",
            f"A3 support_scope={a3.get('support_scope')}, input_frame_policy={a3.get('input_frame_policy')}, AP={a3.get('AP')}.",
        ),
        (
            2,
            "它能不能和 Stream3D 公平比较？",
            "不能。A3 是 prediction-union island diagnostic，且 v53 bridge input frames are not stride-10 aligned; comparison_allowed_pairs=0.",
        ),
        (
            3,
            "同一 masks 换 used-frame support 后 AP 为什么掉？",
            f"A3->{a4.get('row_id','A4')} changes support from prediction-union island to used-frame visible support: AP {a3.get('AP')} -> {a4.get('AP')}; Phase 3 top used-frame failure is undercoverage.",
        ),
        (
            4,
            "D4RT G11/G12 小岛 AP 低是 score、support、fragment 还是 geometry？",
            f"主要是 tiny fragments + undercoverage + instance aggregation/materialization. A5 AP={a5.get('AP')}, A7 AP={a7.get('AP')}; failure top overall={failure.get('top_failure_category')}; aggregation blocker={aggregation.get('blocker')}.",
        ),
        (
            5,
            "Stream3D baseline 是否已用同一 evaluator / same support 重跑？",
            f"已重跑 v65 Stream3D rows，stream3d gate pass={(stream3d.get('gate') or {}).get('pass')}; S3D same-support rows存在，但 support owner/input-frame policy 仍限制 official win/loss.",
        ),
        (
            6,
            "当前有没有 method-safe ScanNet AP？",
            "没有。"
            f"method_safe_rows_with_AP={ap.get('method_safe_rows_with_AP')}; "
            f"SOMA no-GT inference audit pass={(soma_policy.get('gate') or {}).get('pass')}; "
            f"method_inference_gt_geometry_record_count={soma_policy.get('method_inference_gt_geometry_record_count')}.",
        ),
        (
            7,
            "几何指标现在分为哪些 level？",
            f"levels_present={geometry.get('levels_present')}; rows={geometry.get('geometry_metric_row_count')}.",
        ),
        (
            8,
            "是否还有只报 chunk 指标的问题？",
            f"没有只报 chunk：G0-G5 都有 rows；但 chunk scale 仅 {((geometry.get('chunk_scale') or {}).get('within_10pct_count'))}/{((geometry.get('chunk_scale') or {}).get('adjacent_pair_count'))} 通过 v65 10% gate，不能推广为全局稳定。",
        ),
        (
            9,
            "3D viser 能否看到 D4RT geometry 和 SOMA semantic ownership？",
            f"D4RT geometry 可以：viser import/server smoke pass and 5 scene point clouds added. SOMA ownership 只能 summary，3D layer blocked: {visualization.get('ownership_note')}",
        ),
        (
            10,
            "视觉 casebook 里最主要失败模式是什么？",
            f"casebook_count={casebook.get('case_count')}; covers scope, tiny fragments, chunk scale drift, ownership trace missing, undercoverage, scoring, overmerge.",
        ),
        (
            11,
            "Dynamic Replica 是算法失败还是数据缺失？",
            f"数据缺失。dyn_level={dynamic.get('dyn_level_label')}, depth/mask/instance_id_map counts={dynamic.get('actual_depth_count')}/{dynamic.get('actual_mask_count')}/{dynamic.get('actual_instance_id_map_count')}.",
        ),
        (
            12,
            "下一轮算法应该修 AP materialization、fragment aggregation、geometry scale，还是数据？",
            "优先修 AP materialization 的 non-GT fragment->object aggregation trace，并同时补 Dynamic Replica actual GT files；不要把 active query 或 oracle union 当主线。",
        ),
    ]
    return [{"question_id": qid, "question": q, "answer": a} for qid, q, a in rows]


def _evidence_hash_rows() -> list[dict[str, str]]:
    paths = [
        "outputs/audit/v65_ap_contract/ap_contract_summary.json",
        "outputs/audit/v65_ap_contract/ap_contract_rows.csv",
        "outputs/audit/v65_ap_contract/ap_comparability_matrix.csv",
        "outputs/audit/v65_stream3d_parity/stream3d_parity_summary.json",
        "outputs/audit/v65_ap_failure_decomp/failure_summary.json",
        "outputs/audit/v65_geometry_contract/geometry_contract_summary.json",
        "outputs/audit/v65_instance_aggregation/aggregation_summary.json",
        "outputs/audit/v65_visualization/viser_scene_index.json",
        "outputs/audit/v65_visualization/viser_server_status.json",
        "outputs/audit/v65_casebook/casebook_summary.json",
        "outputs/audit/v65_dynamic_data/dynamic_data_summary.json",
        "outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_audit_summary.json",
        "outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_scanned_rows.csv",
        "outputs/audit/v65_soma_inference_policy_audit/soma_inference_policy_violations.csv",
    ]
    return [{"path": path, "sha256": sha256_file(path)} for path in paths]


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stream4D v65 Final Decision",
        "",
        "## Decision Labels",
        "",
    ]
    lines.extend(f"- {label}" for label in summary["decision_labels"])
    lines.extend(["", "## Required Questions", ""])
    for row in summary["questions"]:
        lines.append(f"{row['question_id']}. {row['question']}")
        lines.append(f"   - {row['answer']}")
    lines.extend(["", "## Key Metrics", ""])
    for key, value in summary["key_metrics"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
