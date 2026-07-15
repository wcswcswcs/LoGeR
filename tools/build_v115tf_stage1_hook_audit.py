#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
STAGE1 = RESULT_ROOT / "stage1_hook_audit"
REPORTS = STAGE1 / "source_span_audit_reports"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def line_of(path: Path, needle: str) -> int | str:
    if not path.exists():
        return ""
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if needle in line:
            return idx
    return ""


def hs_hook_rows() -> list[dict[str, Any]]:
    attention = ROOT / "third_party/HorizonStream/horizonstream/runtime/layers/attention.py"
    runtime = ROOT / "third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py"
    model = ROOT / "third_party/HorizonStream/horizonstream/runtime/models/horizonstream.py"
    return [
        {
            "surface": "HS-LA local attention logits/probabilities",
            "status": "blocked_for_full_attention_map",
            "file": rel(attention),
            "line_hint": line_of(attention, "F.scaled_dot_product_attention"),
            "default_off": True,
            "action_available": False,
            "blocker": "Default fused SDPA path returns head output only, not attention logits/probability rows.",
            "fail_forward": "Use HS-HG local pose-query head-output gate; materialize manual attention only as future selected-layer repair.",
        },
        {
            "surface": "HS-HG local pose-query head output",
            "status": "implemented",
            "file": rel(runtime),
            "line_hint": line_of(runtime, "def apply_local_head_output_action"),
            "call_file": rel(attention),
            "call_line_hint": line_of(attention, "_apply_semantic_head_gate"),
            "default_off": "action must start with HS_HG",
            "action_available": True,
            "blocker": "",
            "fail_forward": "Run 00/02 pilot and compare against internal-only/semantic-only/same-magnitude controls if geometry gate passes.",
        },
        {
            "surface": "HS-GQ GLA/state update",
            "status": "implemented_smoke_pass_default_full_oom_reduced_config_no_geometry_gate",
            "file": rel(runtime),
            "line_hint": line_of(runtime, "def apply_gq_state_delta_action"),
            "call_file": rel(model),
            "call_line_hint": line_of(model, "recurrent_state = apply_gq_state_delta_action"),
            "default_off": "action must start with HS_GQ1/HS_GQ2/HS_GQ3/HS_GQ4/HS_GQ5",
            "action_available": True,
            "blocker": "GQ1/GQ3/GQ4 default full promotion remains OOM on 22GB GPUs; the only completed full GQ1 00/02 run uses chunk_block_num=1, gq_layer_filter=23, trace/audit disabled, and does not pass the geometry gate.",
            "fail_forward": "Controls are only required for a geometry-pass candidate; current GQ branch is classified as default OOM-blocked plus config-specific no-geometry-effect.",
        },
        {
            "surface": "MRT readout / predicted metric scale",
            "status": "existing_safety_layer_only",
            "file": rel(ROOT / "third_party/HorizonStream/horizonstream/models/horizonstream.py"),
            "line_hint": line_of(ROOT / "third_party/HorizonStream/horizonstream/models/horizonstream.py", "apply_mrt_scale_action"),
            "default_off": True,
            "action_available": True,
            "blocker": "Cannot be counted as v115 primary semantic method.",
            "fail_forward": "Only compose after a base HS-HG/HS-GQ candidate passes.",
        },
    ]


def lingbot_rows() -> list[dict[str, Any]]:
    sources = [
        ("A2_anchor_source_span", V112 / "stage1_hook_traceability_audit/A2_ANCHOR_SOURCE_SPAN_BLOCKED.md"),
        ("L2_local_query_type", V112 / "stage1_hook_traceability_audit/QUERY_TYPE_INDEX_BLOCKED.md"),
        ("T7_trajectory_retrieval", V112 / "stage1_hook_traceability_audit/T5_RETRIEVAL_HOOK_BLOCKED.md"),
    ]
    rows: list[dict[str, Any]] = []
    for surface, path in sources:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        rows.append(
            {
                "surface": surface,
                "status": "audit_only_blocked" if "BLOCKED" in path.name or "blocked" in text.lower() else "audit_only_exists",
                "file": rel(path),
                "exists": path.exists(),
                "non_b1_h1_schedule_surface": True,
                "default_off_parity": "not_run_horizonstream_scope",
                "summary": " ".join(text.splitlines()[:4])[:600] if text else "",
                "v115_implication": "Satisfies non-B1/H1 hook audit minimum, but no LingBot runtime action claim is made.",
            }
        )
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    hs_rows = hs_hook_rows()
    lb_rows = lingbot_rows()
    write_csv(STAGE1 / "hs_hook_loci.csv", hs_rows)
    write_csv(STAGE1 / "lingbot_hook_loci.csv", lb_rows)
    write_csv(
        STAGE1 / "lingbot_noop_parity_rows.csv",
        [
            {
                "surface": row["surface"],
                "status": row["status"],
                "parity_status": "not_run_horizonstream_scope",
                "reason": "v115 execution is HorizonStream-tagged; LingBot minimum is hook audit, not runtime action.",
            }
            for row in lb_rows
        ],
    )
    write_text(
        REPORTS / "HS_LOCAL_ATTENTION_LOGIT_HOOK_BLOCKED.md",
        """# HS Local Attention Logit Hook Blocked

`Attention.forward` uses `F.scaled_dot_product_attention(...)` on the default path and does not materialize attention logits or probability maps. A v115 HS-LA claim would need a selected-layer manual attention path or another exact logit hook with default-off parity.

Fail-forward used here: implement HS-HG at the local pose-query per-head output, where the tensor is visible before projection.
""",
    )
    write_text(
        REPORTS / "HS_HG_HEAD_OUTPUT_HOOK_IMPLEMENTED.md",
        """# HS-HG Head Output Hook Implemented

Implemented function: `semantic_runtime.apply_local_head_output_action`.

Runtime call: `attention._apply_semantic_head_gate` immediately before `transpose(...).reshape(...)` and projection.

Scope guard: action must start with `HS_HG`, the KV-cache must match local pose read, and `kv_new` must be the empty sentinel used by `_process_causal_frame_attention` for pose-token reads.
""",
    )
    write_text(
        REPORTS / "LINGBOT_NON_B1H1_HOOK_AUDIT.md",
        """# LingBot Non-B1/H1 Hook Audit

Audited A2 anchor source-span, L2/query-type, and trajectory retrieval reports from v112. They remain blocked/audit-only for this HorizonStream run; no LingBot runtime action or metric is claimed in v115.
""",
    )
    write_text(
        STAGE1 / "action_fidelity_schema.md",
        """# Action Fidelity Schema

Primary v115 HS-HG action artifact: `hs_hg_action_gate_rows.csv`.

Required columns include `action`, `control`, `scope`, `num_local_rows`, `num_heads`, `gate_mean`, `gate_std`, `gate_row_mean_mean`, `gate_row_mean_std`, `changed_head_fraction_abs_gt_1e_4`, `semantic_risk_mean`, `semantic_stable_mean`, and `internal_head_q_std`.

Interpretation: `gate_row_mean_mean ~= 1` and very small `gate_row_mean_std` are required evidence that the head gate is row-mean neutral rather than a repeat of v114 value magnitude scaling.
""",
    )
    pose_parity = read_json(STAGE1 / "hs_noop_trace_parity_summary.json")
    hook_parity = read_json(STAGE1 / "hs_hook_level_noop_parity.json")
    smoke_gate = read_json(STAGE1 / "hs_hg_smoke_gate_summary.json")
    strict_pose_pass = bool(pose_parity.get("pass"))
    hook_pass = bool(hook_parity.get("pass"))
    hg_smoke_pass = bool(smoke_gate.get("pass_action_fidelity"))
    summary = {
        "schema": "acl2_v115tf_stage1_hook_audit_summary_v1",
        "hs_hook_row_count": len(hs_rows),
        "lingbot_hook_row_count": len(lb_rows),
        "hs_la_attention_logit_status": "blocked",
        "hs_hg_status": "implemented",
        "hs_gq_status": "implemented_smoke_pass_default_full_oom_reduced_config_no_geometry_gate",
        "strict_pose_noop_parity_pass": strict_pose_pass,
        "hook_level_noop_parity_pass": hook_pass,
        "hs_hg_action_fidelity_smoke_pass": hg_smoke_pass,
        "stage1_status": "partial_pose_parity_blocker" if not strict_pose_pass else "pass",
        "blocker": "" if strict_pose_pass else "Pose-level smoke diff is ~3e-5 while hook-level no-op parity is 0.0; do not claim full strict parity.",
        "outputs": {
            "hs_hook_loci": rel(STAGE1 / "hs_hook_loci.csv"),
            "lingbot_hook_loci": rel(STAGE1 / "lingbot_hook_loci.csv"),
            "hs_noop_trace_parity_summary": rel(STAGE1 / "hs_noop_trace_parity_summary.json"),
            "hs_hook_level_noop_parity": rel(STAGE1 / "hs_hook_level_noop_parity.json"),
            "hs_hg_smoke_gate_summary": rel(STAGE1 / "hs_hg_smoke_gate_summary.json"),
            "hs_gq_decision_summary": rel(RESULT_ROOT / "diagnostics/stage_hs_gq_decision_summary.json"),
        },
    }
    write_json(STAGE1 / "stage1_hook_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
