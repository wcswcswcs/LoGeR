#!/usr/bin/env python3
"""Build ACL2 v108TF Stage1 LingBot action surface contract audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage1_action_surface_contract"
V107TF_STAGE1 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention/stage1_cache_operation_instrumentation"

FILES = {
    "method": ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
    "stream": ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream.py",
    "window": ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream_window.py",
    "attention": ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py",
    "aggregator": ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def find_line(path: Path, pattern: str) -> int:
    if not path.exists():
        return -1
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return idx
    return -1


def loci_rows() -> list[dict[str, Any]]:
    method = FILES["method"]
    stream = FILES["stream"]
    attention = FILES["attention"]
    aggregator = FILES["aggregator"]
    window = FILES["window"]
    return [
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "A",
            "operation_type": "anchor_scale_frame_initialization",
            "file_path": rel(method),
            "function_or_class": "LingbotMapMethod._run_inference",
            "line_hint": find_line(method, "_resolve_keyframe_interval"),
            "existing_control_knob": "_keyframe_interval/_auto_keyframe_threshold/_num_scale_frames/_force_non_keyframe_indices",
            "new_hook_needed": "replace-keyframe/protect-stable-keyframe require new hook; skip/snap can use existing force_non_keyframe",
            "risk_level": "medium",
            "no_op_parity_strategy": "empty force list and trace enabled must match no-trace outputs on KITTI 00/02 96F",
            "full_sequence_action_fidelity_strategy": "action rows must show selected base keyframes changed final_is_keyframe and skip_append",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "A",
            "operation_type": "anchor_scale_frame_initialization",
            "file_path": rel(stream),
            "function_or_class": "GCTStream.inference_streaming",
            "line_hint": find_line(stream, "base_is_keyframe ="),
            "existing_control_knob": "base_is_keyframe plus forced_non_keyframe controls actual cache persistence",
            "new_hook_needed": "scale-frame replacement cannot be done with current wrapper only",
            "risk_level": "medium",
            "no_op_parity_strategy": "no forced frames",
            "full_sequence_action_fidelity_strategy": "expected base keyframe count vs observed forced_non_keyframe rows",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "B",
            "operation_type": "cache_append_write_control",
            "file_path": rel(stream),
            "function_or_class": "GCTStream._set_skip_append/_set_context_only_append",
            "line_hint": find_line(stream, "def _set_skip_append"),
            "existing_control_knob": "force_non_keyframe_indices/context_only_special/anchor_special_only",
            "new_hook_needed": "reduced append strength or token-subset append beyond special-only needs new hook",
            "risk_level": "low",
            "no_op_parity_strategy": "empty force list and no special mode",
            "full_sequence_action_fidelity_strategy": "cache_append/local_reference_separation rows and action jsonl must agree",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "B",
            "operation_type": "cache_append_write_control",
            "file_path": rel(attention),
            "function_or_class": "Attention streaming KV append branch",
            "line_hint": find_line(attention, "skip_append = bool"),
            "existing_control_knob": "_skip_append/_context_only_append flags",
            "new_hook_needed": "semantic token-specific patch append policy",
            "risk_level": "low",
            "no_op_parity_strategy": "v107TF operation trace parity rows",
            "full_sequence_action_fidelity_strategy": "patch cache_append token delta vs no_action",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "C",
            "operation_type": "retention_eviction_budget_keep_drop",
            "file_path": rel(attention),
            "function_or_class": "Attention._apply_kv_cache_eviction/_v107_emit_eviction_rows",
            "line_hint": find_line(attention, "def _apply_kv_cache_eviction"),
            "existing_control_knob": "_kv_cache_sliding_window/_kv_cache_scale_frames config only",
            "new_hook_needed": "semantic+recency eviction ordering requires new hook",
            "risk_level": "high",
            "no_op_parity_strategy": "config unchanged; v107TF observed eviction/budget rows",
            "full_sequence_action_fidelity_strategy": "expected retention/eviction/budget_keep/drop deltas",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "C",
            "operation_type": "retention_eviction_budget_keep_drop",
            "file_path": rel(aggregator),
            "function_or_class": "StreamAggregator constructor/cache manager setup",
            "line_hint": find_line(aggregator, "kv_cache_sliding_window"),
            "existing_control_knob": "sliding-window and scale-frame cache sizes",
            "new_hook_needed": "frame-level semantic keep/drop policy",
            "risk_level": "high",
            "no_op_parity_strategy": "unchanged cache size",
            "full_sequence_action_fidelity_strategy": "cache source frames before/after eviction",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "D",
            "operation_type": "trajectory_memory_write_retention",
            "file_path": rel(attention),
            "function_or_class": "_v107_emit_trajectory rows and headlocal store token masking",
            "line_hint": find_line(attention, "operation_type=\"trajectory_write\""),
            "existing_control_knob": "headlocal modes can zero trajectory-related token ranges for selected heads",
            "new_hook_needed": "independent trajectory memory write/retention gate needs new hook",
            "risk_level": "high",
            "no_op_parity_strategy": "no headlocal action map",
            "full_sequence_action_fidelity_strategy": "trajectory_write rows must change; otherwise surface_not_controllable",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "E",
            "operation_type": "local_preserve_reference_trajectory_block",
            "file_path": rel(attention),
            "function_or_class": "Attention._headlocal_store_tokens",
            "line_hint": find_line(attention, "def _headlocal_store_tokens"),
            "existing_control_knob": "stage4_head_action_map with v106_anchor_reference_block/v106_trajectory_write_block/v106_reference_trajectory_block/v106_context_only_with_local_preserve",
            "new_hook_needed": "clean path-specific anchor update block without patch harm may need deeper hook",
            "risk_level": "high",
            "no_op_parity_strategy": "empty headlocal action map",
            "full_sequence_action_fidelity_strategy": "action rows plus headlocal_action_enabled and trace token deltas",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "F",
            "operation_type": "special_token_camera_register_trajectory_routing",
            "file_path": rel(stream),
            "function_or_class": "GCTStream._set_context_only_append",
            "line_hint": find_line(stream, "def _set_context_only_append"),
            "existing_control_knob": "context_only_special/anchor_special_only",
            "new_hook_needed": "camera-token-only or trajectory-token-only subset routing needs new hook",
            "risk_level": "medium",
            "no_op_parity_strategy": "no context-only forced frames",
            "full_sequence_action_fidelity_strategy": "special token cache rows and patch-token untouched control",
        },
        {
            "schema": "acl2_v108tf_stage1_action_surface_code_locus_v1",
            "surface_id": "F",
            "operation_type": "special_token_camera_register_trajectory_routing",
            "file_path": rel(attention),
            "function_or_class": "context_only_append special cache branch",
            "line_hint": find_line(attention, "pending_special_k"),
            "existing_control_knob": "_context_only_append/_context_only_special_mode",
            "new_hook_needed": "separate camera/register/trajectory subsets",
            "risk_level": "medium",
            "no_op_parity_strategy": "empty context-only list",
            "full_sequence_action_fidelity_strategy": "context_only_operation rows and special-prefix cache growth",
        },
    ]


def feasibility_rows(loci: list[dict[str, Any]]) -> list[dict[str, Any]]:
    surfaces = {
        "A": ("anchor_scale_frame_initialization", "implementable_now_partial", "skip/snap high-risk base keyframes via force_non_keyframe; replacement/protection need hook"),
        "B": ("cache_append_write_control", "implementable_now", "force_non_keyframe/context_only/anchor_only directly change cache append/write"),
        "C": ("retention_eviction_budget_keep_drop", "trace_visible_new_hook_needed", "eviction/retention visible but semantic eviction ordering not exposed"),
        "D": ("trajectory_memory_write_retention", "trace_visible_partial_headlocal_only", "trajectory_write visible but independent write gate not exposed"),
        "E": ("local_preserve_reference_trajectory_block", "implementable_high_risk", "headlocal modes exist but prior v105 relaxed use harmed good; must be guarded"),
        "F": ("special_token_camera_register_trajectory_routing", "implementable_now_partial", "special-only/scale-only supported; token-subset variants need hook"),
    }
    rows: list[dict[str, Any]] = []
    for surface_id, (operation, status, note) in surfaces.items():
        loci_count = sum(1 for row in loci if row["surface_id"] == surface_id)
        rows.append(
            {
                "schema": "acl2_v108tf_stage1_surface_feasibility_row_v1",
                "surface_id": surface_id,
                "operation_type": operation,
                "code_locus_count": loci_count,
                "implementation_status": status,
                "has_existing_runtime_knob": status in {"implementable_now", "implementable_now_partial", "implementable_high_risk"},
                "new_hook_needed": "new_hook_needed" in status or "partial" in status or "high_risk" in status,
                "full_sequence_pilot_allowed": surface_id in {"A", "B", "E", "F"},
                "stage1_contract_pass": loci_count > 0 and surface_id in {"A", "B", "E", "F"},
                "note": note,
            }
        )
    return rows


def noop_parity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = read_csv(V107TF_STAGE1 / "operation_trace_parity_rows.csv")
    for row in source:
        if row.get("seq") not in {"00", "02"}:
            continue
        if not str(row.get("target_id", "")).endswith(("w0049", "w0060")):
            continue
        rows.append(
            {
                "schema": "acl2_v108tf_stage1_noop_parity_row_v1",
                "source_schema": row.get("schema", ""),
                "source_artifact": "v107tf_operation_trace_parity_rows",
                "surface_scope": "all_surfaces_noop_trace_parity_precondition",
                "seq": row.get("seq", ""),
                "target_id": row.get("target_id", ""),
                "trace_frame_count": "96",
                "pose_max_abs_diff": row.get("pose_max_abs_diff", ""),
                "depth_max_abs_diff": row.get("depth_max_abs_diff", ""),
                "intrinsics_max_abs_diff": row.get("intrinsics_max_abs_diff", ""),
                "confidence_max_abs_diff": row.get("confidence_max_abs_diff", ""),
                "operation_row_count": row.get("operation_row_count", ""),
                "observed_operation_types": row.get("observed_operation_types", ""),
                "trace_error_row_count": row.get("trace_error_row_count", ""),
                "noop_parity_pass": str(row.get("parity_pass", "")).lower() == "true",
                "note": "Existing v107TF 96F no-action trace parity evidence for Stage1 no-op contract; surface-specific action fidelity still required before geometry evaluation.",
            }
        )
    return rows


def contract_md(loci: list[dict[str, Any]], feasibility: list[dict[str, Any]], parity: list[dict[str, Any]]) -> str:
    lines = [
        "# v108TF Stage1 Action Surface Contract",
        "",
        "This audit maps LingBot memory operation surfaces to real code paths. It does not claim v108 method success.",
        "",
        "## Surface Summary",
        "",
    ]
    for row in feasibility:
        lines.extend(
            [
                f"### Surface {row['surface_id']}: {row['operation_type']}",
                "",
                f"- implementation_status: `{row['implementation_status']}`",
                f"- existing_runtime_knob: `{row['has_existing_runtime_knob']}`",
                f"- full_sequence_pilot_allowed: `{row['full_sequence_pilot_allowed']}`",
                f"- note: {row['note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## No-op Parity Evidence",
            "",
            f"- parity rows used: `{len(parity)}`",
            "- scope: KITTI 00 and 02 96F no-action trace parity from v107TF operation instrumentation.",
            "- important boundary: this proves tracing/no-op parity only; every non-noop candidate still needs action fidelity.",
            "",
            "## Code Loci",
            "",
        ]
    )
    for row in loci:
        lines.append(
            f"- Surface {row['surface_id']} `{row['operation_type']}`: "
            f"{row['file_path']}:{row['line_hint']} `{row['function_or_class']}`; "
            f"knob={row['existing_control_knob']}; hook={row['new_hook_needed']}"
        )
    lines.append("")
    return "\n".join(lines)


def surface_not_controllable_report(feasibility: list[dict[str, Any]]) -> str:
    lines = [
        "# v108TF Surface Not Fully Controllable Report",
        "",
        "No surface is entirely absent from code, but some surfaces are trace-visible rather than fully controllable through current wrapper knobs.",
        "",
    ]
    for row in feasibility:
        if row["implementation_status"] in {"trace_visible_new_hook_needed", "trace_visible_partial_headlocal_only"}:
            lines.extend(
                [
                    f"## Surface {row['surface_id']}: {row['operation_type']}",
                    "",
                    f"- searched files: `{', '.join(rel(path) for path in FILES.values())}`",
                    f"- status: `{row['implementation_status']}`",
                    f"- why no full hook exists: {row['note']}",
                    "- wrapper-level alternative: use keyframe/cache schedule controls only if action fidelity shows the intended operation changes.",
                    "- cannot claim method from trace visibility alone.",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing_files = [rel(path) for path in FILES.values() if not path.exists()]
    loci = loci_rows()
    feasibility = feasibility_rows(loci)
    parity = noop_parity_rows()
    implementable_pass_count = sum(1 for row in feasibility if row["stage1_contract_pass"])
    parity_pass = len({row["seq"] for row in parity if row["noop_parity_pass"]}) >= 2 and all(row["noop_parity_pass"] for row in parity)
    stage1_pass = (not missing_files) and implementable_pass_count >= 4 and parity_pass

    write_csv(OUT / "action_surface_code_loci.csv", loci)
    write_csv(OUT / "action_surface_implementation_feasibility.csv", feasibility)
    write_csv(OUT / "action_surface_noop_parity_rows.csv", parity)
    (OUT / "action_surface_contract.md").write_text(contract_md(loci, feasibility, parity), encoding="utf-8")
    (OUT / "surface_not_controllable_report.md").write_text(surface_not_controllable_report(feasibility), encoding="utf-8")
    summary = {
        "schema": "acl2_v108tf_stage1_action_surface_contract_summary_v1",
        "stage1_pass": stage1_pass,
        "missing_required_files": missing_files,
        "surface_count": 6,
        "code_locus_count": len(loci),
        "implementation_feasibility_rows": len(feasibility),
        "stage1_contract_pass_surface_count": implementable_pass_count,
        "noop_parity_row_count": len(parity),
        "noop_parity_seqs": sorted({row["seq"] for row in parity if row["noop_parity_pass"]}),
        "noop_parity_pass": parity_pass,
        "surfaces_full_sequence_pilot_allowed": [
            row["surface_id"] for row in feasibility if row["full_sequence_pilot_allowed"]
        ],
        "surfaces_new_hook_needed_before_claim": [
            row["surface_id"] for row in feasibility if row["implementation_status"].startswith("trace_visible")
        ],
        "outputs": {
            "action_surface_code_loci": rel(OUT / "action_surface_code_loci.csv"),
            "action_surface_contract": rel(OUT / "action_surface_contract.md"),
            "action_surface_noop_parity_rows": rel(OUT / "action_surface_noop_parity_rows.csv"),
            "action_surface_implementation_feasibility": rel(OUT / "action_surface_implementation_feasibility.csv"),
            "surface_not_controllable_report": rel(OUT / "surface_not_controllable_report.md"),
            "stage1_summary": rel(OUT / "stage1_summary.json"),
        },
    }
    write_json(OUT / "stage1_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
