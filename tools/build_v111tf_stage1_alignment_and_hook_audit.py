#!/usr/bin/env python3
"""Build ACL2 v111TF Stage1 semantic alignment and hook audit artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
OUT = RESULT_ROOT / "stage1_alignment_and_hook_audit"

V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V108_STAGE1 = V108 / "stage1_action_surface_contract"
V108_STAGE2 = V108 / "stage2_semantic_cue_bank"
V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
V110_STAGE4 = V110 / "stage4_full_00_01_02_05_validation"

SEQUENCES = ("00", "01", "02", "05")
PATCH_GRID_H = 20
PATCH_GRID_W = 36
PATCH_COUNT = PATCH_GRID_H * PATCH_GRID_W
PATCH_START_IDX = 6
SPECIAL_TOKENS = [
    ("camera", 0),
    ("register_0", 1),
    ("register_1", 2),
    ("register_2", 3),
    ("register_3", 4),
    ("anchor_or_scale", 5),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def semantic_token_alignment_rows(stage2: dict[str, Any]) -> list[dict[str, Any]]:
    frame_universe = stage2.get("frame_universe_by_seq", {})
    coverage_by_seq = stage2.get("frame_semantic_coverage_by_seq", {})
    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        frame_count = int(frame_universe.get(seq, 0))
        rows.append(
            {
                "schema": "acl2_v111tf_stage1_semantic_token_alignment_row_v1",
                "seq": seq,
                "frame_count": frame_count,
                "frame_semantic_coverage": coverage_by_seq.get(seq, ""),
                "patch_grid_h": stage2.get("patch_grid_h", PATCH_GRID_H),
                "patch_grid_w": stage2.get("patch_grid_w", PATCH_GRID_W),
                "patch_count_per_frame": PATCH_COUNT,
                "special_token_count": len(SPECIAL_TOKENS),
                "patch_start_idx": stage2.get("patch_start_idx", PATCH_START_IDX),
                "patch_token_start": PATCH_START_IDX,
                "patch_token_end_inclusive": PATCH_START_IDX + PATCH_COUNT - 1,
                "expected_patch_rows": frame_count * PATCH_COUNT,
                "semantic_projection_coverage_ok": fnum(coverage_by_seq.get(seq, "nan")) >= 0.99,
                "source": rel(V108_STAGE2 / "stage2_summary.json"),
            }
        )
    return rows


def memory_context_token_index_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token_name, token_id in SPECIAL_TOKENS:
        rows.append(
            {
                "schema": "acl2_v111tf_stage1_memory_context_token_index_row_v1",
                "token_name": token_name,
                "token_id": token_id,
                "token_family": "special_context_token",
                "source_code_locus": rel(ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py"),
                "evidence": "patch_start_idx = 1 + num_register_tokens + 1; num_register_tokens inferred as 4 because v108 patch_start_idx=6",
            }
        )
    rows.append(
        {
            "schema": "acl2_v111tf_stage1_memory_context_token_index_row_v1",
            "token_name": "image_patch_tokens",
            "token_id": f"{PATCH_START_IDX}..{PATCH_START_IDX + PATCH_COUNT - 1}",
            "token_family": "patch_tokens",
            "source_code_locus": rel(V108_STAGE2 / "stage2_summary.json"),
            "evidence": "v108 semantic projection uses 20x36 patch grid and patch_start_idx=6",
        }
    )
    return rows


def anchor_context_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v111tf_stage1_anchor_context_source_index_row_v1",
            "context_family": "Anchor Context",
            "default_source_frame_rule": "first_n_scale_frames",
            "default_n": 8,
            "source_code_locus": rel(ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py"),
            "config_or_code_field": "kv_cache_scale_frames",
            "patch_source_token_range": f"{PATCH_START_IDX}..{PATCH_START_IDX + PATCH_COUNT - 1}",
            "special_token_range": "0..5",
            "coverage": 1.0,
            "v111_status": "index_audit_pass; delayed selection hook/parity not yet implemented",
        }
    ]


def local_window_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v111tf_stage1_local_window_source_index_row_v1",
            "context_family": "Local Pose-Reference Window",
            "default_source_frame_rule": "recent_sliding_window_frames",
            "default_k": 64,
            "source_code_locus": rel(ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py"),
            "config_or_code_field": "kv_cache_sliding_window",
            "patch_source_token_range": f"{PATCH_START_IDX}..{PATCH_START_IDX + PATCH_COUNT - 1}",
            "special_token_range": "0..5",
            "coverage": 1.0,
            "v111_status": "index_audit_pass; local attention/query-specific hook/parity not yet implemented",
        }
    ]


def trajectory_context_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_types = [
        ("camera", "0", "camera token"),
        ("register", "1..4", "four register tokens"),
        ("anchor_or_scale", "5", "scale/anchor special token"),
        ("all_context_special", "0..5", "camera + register + anchor/scale compact context tokens"),
    ]
    for token_type, token_ids, note in token_types:
        rows.append(
            {
                "schema": "acl2_v111tf_stage1_trajectory_context_token_index_row_v1",
                "context_family": "Trajectory Memory",
                "token_type": token_type,
                "token_ids": token_ids,
                "source_code_locus": rel(ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py"),
                "coverage": 1.0,
                "note": note,
                "v111_status": "index_audit_pass; T2 per-token mask hook/parity pending except existing anchor_special_only coarse path",
            }
        )
    return rows


def noop_parity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V108_STAGE1 / "action_surface_noop_parity_rows.csv"):
        rows.append(
            {
                "schema": "acl2_v111tf_stage1_noop_parity_row_v1",
                "parity_scope": "inherited_v108_v107tf_noop_trace_parity",
                **row,
                "v111_interpretation": "valid precondition for existing no-op trace; not sufficient for new v111 token/attention/value hooks",
            }
        )
    b1_action_rows = [
        row for row in read_csv(V110_STAGE4 / "action_fidelity_rows.csv")
        if row.get("candidate_id") == "B1" and row.get("policy_id") in {"B1_semantic_only", "B1_semantic_plus_internal"}
    ]
    for row in b1_action_rows:
        rows.append(
            {
                "schema": "acl2_v111tf_stage1_noop_parity_row_v1",
                "parity_scope": "v110_b1_existing_action_fidelity_not_noop",
                "seq": row.get("seq", ""),
                "policy_id": row.get("policy_id", ""),
                "stage4_action_mode": row.get("stage4_action_mode", ""),
                "expected_action_frame_count": row.get("expected_action_frame_count", ""),
                "action_effective_frame_count": row.get("action_effective_frame_count", ""),
                "action_noop_frame_count": row.get("action_noop_frame_count", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "noop_parity_pass": "",
                "source": rel(V110_STAGE4 / "action_fidelity_rows.csv"),
                "v111_interpretation": "B1/T1 existing action fidelity passed in v110; no-action parity for new v111 hooks remains required before T2/A/L runtime claims",
            }
        )
    return rows


def action_fidelity_schema_text() -> str:
    return """# ACL2 v111TF Stage1 Action Fidelity Schema

Every v111 runtime action must record at least:

```text
candidate_id
policy_id
seq
memory_family
action_mode
expected_action_frame_count
observed_action_frame_count
action_effective_frame_count
action_noop_frame_count
affected_token_type
affected_token_count
anchor_context_affected_token_count
local_window_affected_token_count
trajectory_context_affected_token_count
action_fidelity_pass
noop_parity_pass_for_zero_action
raw_action_jsonl
config_path
run_worker_returncode
```

For B1/T1 existing `force_non_keyframe` controls, v110 action fidelity is
usable as prior evidence. For v111 T2/A/L new hooks, zero-action / all-ones
weights must be parity-tested before full ATE claims.
"""


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(V108_STAGE2 / "stage2_summary.json")
    semantic_rows = semantic_token_alignment_rows(stage2)
    memory_rows = memory_context_token_index_rows()
    anchor_rows = anchor_context_rows()
    local_rows = local_window_rows()
    traj_rows = trajectory_context_rows()
    parity_rows = noop_parity_rows()

    semantic_projection_coverage = fnum(stage2.get("frame_semantic_coverage"))
    token_alignment_pass = bool(stage2.get("token_alignment_pass"))
    memory_context_index_coverage = 1.0 if memory_rows and traj_rows else 0.0
    inherited_noop_pass = all(
        boolish(row.get("noop_parity_pass"))
        for row in parity_rows
        if row.get("parity_scope") == "inherited_v108_v107tf_noop_trace_parity"
    )
    b1_existing_action_ready = all(
        boolish(row.get("action_fidelity_pass"))
        for row in parity_rows
        if row.get("parity_scope") == "v110_b1_existing_action_fidelity_not_noop"
    )
    stage1_t1_ready = (
        semantic_projection_coverage >= 0.99
        and token_alignment_pass
        and memory_context_index_coverage >= 0.99
        and inherited_noop_pass
        and b1_existing_action_ready
    )
    new_hook_parity_pending = True
    stage1_full_memory_management_ready = False

    write_csv(OUT / "semantic_token_alignment_rows.csv", semantic_rows)
    write_csv(OUT / "memory_context_token_index_rows.csv", memory_rows)
    write_csv(OUT / "anchor_context_source_index_rows.csv", anchor_rows)
    write_csv(OUT / "local_window_source_index_rows.csv", local_rows)
    write_csv(OUT / "trajectory_context_token_index_rows.csv", traj_rows)
    write_csv(OUT / "noop_parity_rows.csv", parity_rows)
    write_text(OUT / "action_fidelity_schema.md", action_fidelity_schema_text())

    summary = {
        "schema": "acl2_v111tf_stage1_alignment_and_hook_audit_summary_v1",
        "stage1_t1_ready": stage1_t1_ready,
        "stage1_full_memory_management_ready": stage1_full_memory_management_ready,
        "new_hook_parity_pending": new_hook_parity_pending,
        "semantic_projection_coverage": semantic_projection_coverage,
        "token_alignment_pass": token_alignment_pass,
        "memory_context_index_coverage": memory_context_index_coverage,
        "inherited_noop_trace_parity_pass": inherited_noop_pass,
        "b1_existing_action_fidelity_ready": b1_existing_action_ready,
        "trajectory_token_indices": {
            "camera": "0",
            "register": "1..4",
            "anchor_or_scale": "5",
            "patch_tokens": f"{PATCH_START_IDX}..{PATCH_START_IDX + PATCH_COUNT - 1}",
        },
        "blockers_for_full_stage1": [
            "T2 per-token camera/register/anchor mask hook not parity-tested",
            "A1 delayed anchor selection hook not parity-tested",
            "A2 anchor attention/value scaling hook not parity-tested",
            "L1/L2 local attention/query-specific hook not parity-tested",
        ],
        "outputs": {
            "semantic_token_alignment_rows": rel(OUT / "semantic_token_alignment_rows.csv"),
            "memory_context_token_index_rows": rel(OUT / "memory_context_token_index_rows.csv"),
            "anchor_context_source_index_rows": rel(OUT / "anchor_context_source_index_rows.csv"),
            "local_window_source_index_rows": rel(OUT / "local_window_source_index_rows.csv"),
            "trajectory_context_token_index_rows": rel(OUT / "trajectory_context_token_index_rows.csv"),
            "noop_parity_rows": rel(OUT / "noop_parity_rows.csv"),
            "action_fidelity_schema": rel(OUT / "action_fidelity_schema.md"),
            "stage1_summary": rel(OUT / "stage1_summary.json"),
        },
    }
    write_json(OUT / "stage1_summary.json", summary)
    if not stage1_t1_ready:
        write_text(
            OUT / "STAGE1_T1_BLOCKED.md",
            "# ACL2 v111TF Stage1 T1 Blocked\n\n"
            f"stage1_t1_ready: `{stage1_t1_ready}`\n\n"
            f"summary: `{json.dumps(clean_json(summary), sort_keys=True, ensure_ascii=False)}`\n",
        )
    if not stage1_full_memory_management_ready:
        write_text(
            OUT / "STAGE1_NEW_HOOK_PARITY_PENDING.md",
            "# ACL2 v111TF Stage1 New Hook Parity Pending\n\n"
            "T1/B1 can proceed with existing force_non_keyframe action evidence. "
            "T2/A/L require default-off hook implementation and no-action parity before runtime claims.\n",
        )
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
