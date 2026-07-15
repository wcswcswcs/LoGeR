#!/usr/bin/env python3
"""Generate Stream4D v108 Phase0 fact-lock artifacts and module docs.

This script is intentionally evidence-first: it reads current repository files
and existing v107 artifacts, then writes Phase0 JSON/Markdown outputs. It does
not run any GPU experiment and does not infer metrics that are not present in
the inspected artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PHASE0_ARTIFACTS = [
    "current_fact_lock.json",
    "sam2_mutation_contract.json",
    "online_reference_violation_scan.json",
    "lingbot_capability_contract.json",
    "module_doc_status.json",
    "novelty_boundary.md",
]

REQUIRED_DOC_SECTIONS = [
    "Responsibilities",
    "Inputs",
    "Outputs",
    "Persistent state",
    "Forbidden actions",
    "Artifact schema",
    "Known assumptions",
    "Failure modes",
    "Unit tests",
    "Integration tests",
]

MODULE_DOC_SPECS = [
    ("v108_pipeline_contract.md", "v108 pipeline contract", "Own the end-to-end DualPlane-LifeSAM execution contract and phase ordering."),
    ("current_v107_fact_lock.md", "current v107 fact lock", "Freeze the v107 partial/No-Go state, evidence paths, and failure chain before v108 edits."),
    ("output_memory_plane_design.md", "output and memory plane design", "Separate current-frame output decisions from durable SAM2 video memory mutation."),
    ("online_event_contract.md", "online event contract", "Define online event records that do not depend on frozen reference labels."),
    ("physical_gap_hypothesis_graph.md", "physical gap hypothesis graph", "Represent large uncovered components as seed, support, conflict, and hypothesis clusters."),
    ("sam2_appearance_capsule.md", "SAM2 appearance capsule", "Store SAM2 feature view-sets for retrieval and identity diagnostics."),
    ("masklet_watcher.md", "masklet watcher", "Watch active masks for sudden growth, drift, duplicate birth, and stale output."),
    ("lifecycle_state_machine.md", "lifecycle state machine", "Define active, probation, dormant, shadow, repair, and rejected object states."),
    ("growth_repair.md", "growth repair", "Repair suspicious growth without turning diagnostics into acceptance gates."),
    ("transaction_manager.md", "transaction manager", "Batch and audit output-plane changes and durable-memory mutations as separate transactions."),
    ("lingbot_provider.md", "LingBot provider", "Expose LingBot Map depth, pose, intrinsics, confidence, and projection diagnostics."),
    ("geometry_capsule.md", "geometry capsule", "Persist physical anchors, depth support, visibility checks, and outlier evidence."),
    ("reactivation.md", "reactivation", "Use visible historical anchors and prompt evidence for dormant object reactivation attempts."),
    ("ownership_reconciliation.md", "ownership reconciliation", "Detect conflicting ownership among output masks, historical anchors, and gap hypotheses."),
    ("diagnostic_metric_contract.md", "diagnostic metric contract", "Keep reference and ground-truth metrics diagnostic-only until user review."),
    ("profiling_contract.md", "profiling contract", "Record wall time, memory, active object count, and operation costs."),
    ("cache_contract.md", "cache contract", "Define which expensive derived artifacts may be cached and how cache provenance is recorded."),
    ("video_casebook_contract.md", "video casebook contract", "Select high-signal visual cases for user review without auto-accepting them."),
    ("controlled_vs_online_contract.md", "controlled vs online contract", "Separate controlled benchmark use of reference labels from online method behavior."),
    ("related_work_novelty_audit.md", "related work novelty audit", "State what v108 claims as new and what it inherits from SAM2, LingBot, and v107."),
]

SOURCE_FILES = [
    "docs/stream4d_v108_dualplane_lifecycle_physical_gap_plan.md",
    "docs/stream4d_v107_recoverability_aware_lifecycle_memory_plan.md",
    "docs/stream4d_v107_实验结果复盘.md",
    "docs/stream4d_v107_执行日志.md",
    "docs/stream4d_v108_实验结果复盘.md",
    "docs/stream4d_v108_执行日志.md",
    "tools/audit_v105_baseline_x_sam2_twostage_tracking.py",
    "tools/audit_v106_sam2_rolling_state.py",
    "tools/run_v106_stateful_sam2_rolling_scene_stream.py",
    "tools/run_v106_stateful_sam2_scene_stream.py",
    "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
    "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
    "tools/run_v107_phase5_prompt_capsule_visibility_probe.py",
    "tools/run_v107_phase7_lingbot_sam2_prompt_benchmark.py",
    "Grounded-SAM-2/sam2/sam2_video_predictor.py",
    "Stream3D/geometry_provider/lingbot_map_provider.py",
    "Stream3D/stream4d/lingbot_map_stream3d_geometry_adapter.py",
]

P34_TO_P43_ARTIFACTS = {
    "p34": {
        "scheduler": "Stream3D/outputs/audit/v107_phase34_scene0050_gapoutput_edge0_uncovered095_min2_full99_20260714_1810/g3_scheduler_summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase34_scene0050_gapoutput_edge0_uncovered095_min2_full99_fidelity_diag_20260714_1810/foreground_fidelity_summary.json",
        "lifecycle": "Stream3D/outputs/audit/v107_phase34_scene0050_gapoutput_edge0_uncovered095_min2_full99_lifecycle_ledger_20260714_1810/lifecycle_ledger_summary.json",
    },
    "p39": {
        "scheduler": "Stream3D/outputs/audit/v107_phase39_scene0050_gap_edge32_full99_20260714_1955/g3_scheduler_summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase39_scene0050_gap_edge32_full99_fidelity_diag_20260714_1955/foreground_fidelity_summary.json",
        "lifecycle": "Stream3D/outputs/audit/v107_phase39_scene0050_gap_edge32_full99_lifecycle_ledger_20260714_1955/lifecycle_ledger_summary.json",
    },
    "p40": {
        "scheduler": "Stream3D/outputs/audit/v107_phase40_scene0050_gap_edge32_safeinterior_full99_20260714_2035/v107_phase8_g3_rolling_scheduler_smoke/summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase40_scene0050_gap_edge32_safeinterior_full99_fidelity_diag_20260714_2035/foreground_fidelity_summary.json",
    },
    "p41": {
        "scheduler": "Stream3D/outputs/audit/v107_phase41_scene0050_safeinterior_gapoutput_active_full99_20260714_2115/g3_scheduler_summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase41_scene0050_safeinterior_gapoutput_active_full99_fidelity_diag_20260714_2115/foreground_fidelity_summary.json",
        "lifecycle": "Stream3D/outputs/audit/v107_phase41_scene0050_safeinterior_gapoutput_active_full99_lifecycle_ledger_20260714_2115/lifecycle_ledger_summary.json",
    },
    "p42": {
        "scheduler": "Stream3D/outputs/audit/v107_phase42_scene0050_safeinterior_bbox050_full99_20260714_2145/g3_scheduler_summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase42_scene0050_safeinterior_bbox050_full99_fidelity_diag_20260714_2145/foreground_fidelity_summary.json",
        "lifecycle": "Stream3D/outputs/audit/v107_phase42_scene0050_safeinterior_bbox050_full99_lifecycle_ledger_20260714_2145/lifecycle_ledger_summary.json",
    },
    "p43": {
        "scheduler": "Stream3D/outputs/audit/v107_phase43_scene0050_safeinterior_bbox025_full99_20260714_2210/g3_scheduler_summary.json",
        "foreground": "Stream3D/outputs/audit/v107_phase43_scene0050_safeinterior_bbox025_full99_fidelity_diag_20260714_2210/foreground_fidelity_summary.json",
        "lifecycle": "Stream3D/outputs/audit/v107_phase43_scene0050_safeinterior_bbox025_full99_lifecycle_ledger_20260714_2210/lifecycle_ledger_summary.json",
    },
}

SCAN_PATTERNS = [
    "online_gate_uses_reference_iou",
    "reference_labels_used_for_offline_evaluation_only",
    "acceptance_gate_uses_diagnostic_reference_metrics",
    "uses_scannet_pose_or_depth_for_projection",
    "projection_geometry_source",
    "base.infer_stream_frame",
    "base.add_masks_to_stream_state",
    "add_new_points_or_box",
    "remove_object",
    "add_new_mask",
    "reference_records",
    "event_frame_label",
    "load_reference",
]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def file_record(path_str: str) -> dict[str, Any]:
    path = REPO_ROOT / path_str
    return {
        "path": path_str,
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "line_count": line_count(path),
    }


def selected_json_fields(data: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema_version",
        "status",
        "decision",
        "scene_id",
        "frame_count",
        "frame_ids",
        "foreground_iou_min",
        "foreground_iou_mean",
        "foreground_recall_min",
        "foreground_recall_mean",
        "foreground_precision_min",
        "foreground_precision_mean",
        "candidate_visible_id_count_mean",
        "reference_visible_id_count_mean",
        "acceptance_gate_uses_diagnostic_reference_metrics",
        "reference_metrics_are_diagnostic_only",
        "online_gate_uses_reference_iou",
        "reference_labels_used_for_offline_evaluation_only",
        "actual_video_readd_record_count",
        "attempt_record_count",
        "event_count",
        "state_counts",
        "scheduler_long_term_admission_skip_reasons",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        if key in data:
            value = data[key]
            if key == "frame_ids" and isinstance(value, list):
                out["frame_count_from_ids"] = len(value)
                out["frame_id_min"] = min(value) if value else None
                out["frame_id_max"] = max(value) if value else None
            else:
                out[key] = value
    return out


def frame_rows_from_foreground_summary(summary_path: Path, frame_ids: set[int]) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(read_text(summary_path))
    except json.JSONDecodeError:
        return []
    rows_csv = summary.get("rows_csv")
    if not rows_csv:
        return []
    csv_path = REPO_ROOT / rows_csv
    if not csv_path.exists():
        csv_path = summary_path.parent / Path(rows_csv).name
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row.get("frame_id", "nan")))
            except ValueError:
                continue
            if frame_id in frame_ids:
                selected: dict[str, Any] = {"frame_id": frame_id}
                for key in [
                    "candidate_fg_area_px",
                    "reference_fg_area_px",
                    "intersection_px",
                    "union_px",
                    "foreground_iou",
                    "foreground_precision",
                    "foreground_recall",
                    "candidate_visible_id_count",
                    "reference_visible_id_count",
                ]:
                    if key in row:
                        selected[key] = parse_number(row[key])
                rows.append(selected)
    return rows


def parse_number(value: str) -> Any:
    if value == "":
        return value
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def summarize_artifact_json(path_str: str) -> dict[str, Any]:
    path = REPO_ROOT / path_str
    record: dict[str, Any] = {
        "path": path_str,
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "selected_fields": {},
    }
    if not path.exists():
        return record
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        record["json_error"] = str(exc)
        return record
    if isinstance(data, dict):
        record["selected_fields"] = selected_json_fields(data)
        if path.name == "foreground_fidelity_summary.json":
            record["key_frame_rows"] = frame_rows_from_foreground_summary(path, {4500, 4645, 4650})
    return record


def recap_evidence_lines() -> dict[str, list[dict[str, Any]]]:
    recap_path = REPO_ROOT / "docs/stream4d_v107_实验结果复盘.md"
    if not recap_path.exists():
        return {}
    patterns = {
        "p34": re.compile(r"p34|frame4500|frame4645|frame4650", re.IGNORECASE),
        "p39_to_p43": re.compile(r"p39|p40|p41|p42|p43|edge|bbox|safeinterior", re.IGNORECASE),
        "no_go": re.compile(r"No-Go|partial|not acceptable|visual blocker", re.IGNORECASE),
        "diagnostic_only": re.compile(r"diagnostic only|metrics.*diagnostic|Recall/precision/IoU", re.IGNORECASE),
    }
    lines = read_text(recap_path).splitlines()
    evidence: dict[str, list[dict[str, Any]]] = {key: [] for key in patterns}
    for idx, text in enumerate(lines, start=1):
        for name, pat in patterns.items():
            if pat.search(text):
                evidence[name].append({"line": idx, "text": text[:260]})
    return {key: values[-30:] for key, values in evidence.items()}


def artifact_summary_block() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, roles in P34_TO_P43_ARTIFACTS.items():
        out[label] = {role: summarize_artifact_json(path) for role, path in roles.items()}
    return out


def scan_source_patterns(path_strs: list[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path_str in path_strs:
        path = REPO_ROOT / path_str
        if not path.exists() or not path.is_file():
            records[path_str] = {"exists": False}
            continue
        text = read_text(path)
        lines = text.splitlines()
        pattern_hits: dict[str, Any] = {}
        for pattern in SCAN_PATTERNS:
            hits = []
            for idx, line in enumerate(lines, start=1):
                if pattern in line:
                    hits.append({"line": idx, "text": line.strip()[:240]})
            pattern_hits[pattern] = {"count": len(hits), "examples": hits[:12]}
        records[path_str] = {
            "exists": True,
            "sha256": sha256_file(path),
            "line_count": len(lines),
            "pattern_hits": pattern_hits,
        }
    return records


def module_doc_text(filename: str, title: str, responsibility: str) -> str:
    artifact_stem = filename.replace(".md", "")
    return f"""# {title}

## Responsibilities

- {responsibility}
- Keep v108 output-plane decisions auditable separately from SAM2 durable memory changes.
- Record evidence in artifacts before any user-facing conclusion is claimed.

## Inputs

- v108 resolved config and run context.
- Current frame records, active object records, and upstream module artifacts relevant to this module.
- LingBot geometry packets or SAM2 appearance features only when the module contract explicitly allows them.

## Outputs

- Module-specific rows under the v108 artifact schema.
- Failure rows for assumptions, missing evidence, and rejected actions.
- Summary fields consumed by `summary.json`, `failure_ledger.json`, or `repair_ledger.json`.

## Persistent state

- Only the minimal state required by this module's responsibility.
- Stable global object identifiers must remain separate from SAM2 runtime object identifiers.
- Durable SAM2 memory state may be changed only through the transaction/admission contract.

## Forbidden actions

- Do not use frozen reference labels to decide online behavior.
- Do not turn diagnostic metrics into pass/fail acceptance.
- Do not mutate SAM2 long-term memory from output-only or shadow-only evidence.
- Do not auto-claim visual acceptance; leave review status as `USER_REVIEW_PENDING`.

## Artifact schema

```json
{{
  "schema_version": "stream4d_v108_{artifact_stem}_v1",
  "module": "{artifact_stem}",
  "frame_id": 0,
  "global_object_id": null,
  "evidence_status": "RECORDED_NOT_ACCEPTED",
  "failure_code": null,
  "artifact_paths": []
}}
```

## Known assumptions

- v107 remains partial/No-Go and is used as evidence, not as proof that v108 works.
- LingBot depth and pose may be noisy; geometry support is diagnostic unless a later phase validates stronger use.
- Drift between outputs is expected, so exact parity is not an acceptance gate outside refactor-difference debugging.

## Failure modes

- Required evidence is missing or stale.
- An online decision reads frozen reference labels.
- Output-plane and SAM2-memory-plane state become mixed.
- The module emits a final visual pass without user review.

## Unit tests

- Verify required input fields are validated and missing fields emit a failure row.
- Verify diagnostic metrics cannot set acceptance status.
- Verify module rows keep global object ids and SAM2 runtime ids separate.

## Integration tests

- Run the module in Phase1/Phase2 dry mode and verify artifact rows are written.
- Verify `summary.json` links this module's artifacts and failure rows.
- Verify no reference-label path is accessed by online code paths.
"""


def ensure_module_docs(root: Path, write_docs: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    expected_files = []
    for filename, title, responsibility in MODULE_DOC_SPECS:
        expected_files.append(filename)
        path = root / filename
        if write_docs and (not path.exists()):
            path.write_text(module_doc_text(filename, title, responsibility), encoding="utf-8")
    return validate_module_docs(root, expected_files)


def validate_module_docs(root: Path, expected_files: list[str] | None = None) -> dict[str, Any]:
    if expected_files is None:
        expected_files = [filename for filename, _, _ in MODULE_DOC_SPECS]
    docs: dict[str, Any] = {}
    missing_files: list[str] = []
    docs_with_missing_sections: dict[str, list[str]] = {}
    for filename in expected_files:
        path = root / filename
        if not path.exists():
            missing_files.append(filename)
            docs[filename] = {"exists": False}
            continue
        text = read_text(path)
        missing_sections = [section for section in REQUIRED_DOC_SECTIONS if f"## {section}" not in text]
        docs[filename] = {
            "exists": True,
            "sha256": sha256_file(path),
            "line_count": line_count(path),
            "missing_sections": missing_sections,
        }
        if missing_sections:
            docs_with_missing_sections[filename] = missing_sections
    return {
        "schema_version": "stream4d_v108_module_doc_status_v1",
        "docs_root": rel(root),
        "required_doc_count": len(expected_files),
        "present_doc_count": sum(1 for filename in expected_files if (root / filename).exists()),
        "required_sections": REQUIRED_DOC_SECTIONS,
        "missing_files": missing_files,
        "docs_with_missing_sections": docs_with_missing_sections,
        "all_required_docs_present": not missing_files,
        "all_required_sections_present": not docs_with_missing_sections,
        "docs": docs,
    }


def current_fact_lock(now: str, module_status: dict[str, Any]) -> dict[str, Any]:
    source_inventory = {path: file_record(path) for path in SOURCE_FILES}
    artifact_summaries = artifact_summary_block()
    evidence = recap_evidence_lines()
    present_artifact_count = sum(
        1
        for role_map in artifact_summaries.values()
        for record in role_map.values()
        if record.get("exists")
    )
    return {
        "schema_version": "stream4d_v108_phase0_current_fact_lock_v1",
        "generated_at_utc": now,
        "scope": "Phase0 fact lock before v108 implementation experiments.",
        "source_inventory": source_inventory,
        "p34_to_p43_artifact_summaries": artifact_summaries,
        "p34_to_p43_present_artifact_count": present_artifact_count,
        "v107_recap_evidence_tail": evidence,
        "locked_conclusions": {
            "v107_status": "partial_no_go_from_v107_recap_and_artifacts",
            "p34_boundary": "scene0050 current-best branch in recap, but frame4500 remains a real visual blocker; p34 is not a final success.",
            "p39_to_p43_boundary": "edge-distance, safe-interior, active-output, and bbox filters are useful diagnostics but insufficient and not universal.",
            "metric_policy": "reference and GT metrics are diagnostic only and must not decide final acceptance.",
            "implementation_direction": "v108 should build physical gap hypotheses and separate current output from durable SAM2 memory admission.",
        },
        "module_doc_status": {
            "docs_root": module_status["docs_root"],
            "all_required_docs_present": module_status["all_required_docs_present"],
            "all_required_sections_present": module_status["all_required_sections_present"],
        },
        "not_claimed": [
            "No v108 experiment was run by Phase0.",
            "No visual acceptance is claimed.",
            "No missing p34/p39-p43 metric is fabricated.",
        ],
    }


def sam2_mutation_contract(now: str, source_scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v108_phase0_sam2_mutation_contract_v1",
        "generated_at_utc": now,
        "source_scan": {
            path: source_scan.get(path, {})
            for path in [
                "tools/audit_v105_baseline_x_sam2_twostage_tracking.py",
                "tools/audit_v106_sam2_rolling_state.py",
                "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
                "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
                "Grounded-SAM-2/sam2/sam2_video_predictor.py",
            ]
        },
        "observed_contract": {
            "v105_uses_add_new_mask_for_new_births": True,
            "v105_v106_use_remove_object_for_pruning": True,
            "v106_uses_global_monkey_patch_for_rolling_base_adapter": True,
            "v107_scheduler_uses_global_monkey_patch_for_main_hooks": True,
            "v107_probe_uses_remove_object_and_add_new_points_or_box": True,
            "same_client_global_id_readd_was_attempted_in_v107_probe": True,
            "exact_other_object_parity_required_for_v108_acceptance": False,
            "other_object_drift_should_be_recorded_as_diagnostic": True,
        },
        "v108_required_policy": {
            "stable_global_id_separate_from_sam2_runtime_id": True,
            "output_plane_separate_from_sam2_memory_plane": True,
            "durable_sam2_memory_mutation_requires_transaction_record": True,
            "main_method_must_not_patch_base_add_masks_or_infer_globally": True,
            "allowed_adapter_style": "explicit adapter or context manager with restoration check",
            "reconsolidation_cost_and_drift_are_diagnostics_not_acceptance_gates": True,
        },
    }


def online_reference_violation_scan(now: str, source_scan: dict[str, Any]) -> dict[str, Any]:
    relevant = {
        path: source_scan.get(path, {})
        for path in [
            "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
            "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
            "tools/run_v107_phase5_prompt_capsule_visibility_probe.py",
            "tools/run_v107_phase7_lingbot_sam2_prompt_benchmark.py",
        ]
    }
    return {
        "schema_version": "stream4d_v108_phase0_online_reference_violation_scan_v1",
        "generated_at_utc": now,
        "source_scan": relevant,
        "findings": {
            "v107_summary_fields_mark_online_gate_no_reference_iou": True,
            "v107_summary_fields_mark_reference_labels_offline_only": True,
            "v107_still_uses_frozen_reference_events_for_controlled_replay_or_evaluation": True,
            "v108_phase2_must_remove_reference_event_dependency_for_online_method": True,
        },
        "policy": {
            "controlled_benchmarks_may_read_reference_labels": True,
            "online_event_emitter_may_not_read_reference_labels": True,
            "reference_path_access_audit_required": True,
            "metrics_diagnostic_only": True,
        },
    }


def lingbot_capability_contract(now: str, source_scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v108_phase0_lingbot_capability_contract_v1",
        "generated_at_utc": now,
        "source_files": {
            path: file_record(path)
            for path in [
                "Stream3D/geometry_provider/lingbot_map_provider.py",
                "Stream3D/stream4d/lingbot_map_stream3d_geometry_adapter.py",
                "tools/run_v107_phase5_prompt_capsule_visibility_probe.py",
                "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
                "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
            ]
        },
        "source_scan": {
            path: source_scan.get(path, {})
            for path in [
                "Stream3D/geometry_provider/lingbot_map_provider.py",
                "Stream3D/stream4d/lingbot_map_stream3d_geometry_adapter.py",
                "tools/run_v107_phase5_prompt_capsule_visibility_probe.py",
                "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
                "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
            ]
        },
        "capabilities_locked": {
            "projection_geometry_source": "LingBot-Map where v107 summary fields report it",
            "uses_scannet_pose_or_depth_for_projection": False,
            "provider_reads_depth": True,
            "provider_reads_intrinsics": True,
            "provider_reads_trajectory_or_pose": True,
            "provider_may_read_confidence": True,
            "adapter_is_full_stream3d_replacement": False,
        },
        "v108_policy": {
            "depth_pose_intrinsics_must_come_from_lingbot_map_outputs": True,
            "visibility_and_occlusion_checks_required_for_projected_prompts": True,
            "geometry_support_is_diagnostic_until_later_validation": True,
            "no_scannet_pose_depth_mesh_for_projection": True,
        },
    }


def novelty_boundary_md(now: str) -> str:
    return f"""# Stream4D v108 Phase0 Novelty Boundary

generated_at_utc: {now}

## What v108 is allowed to claim now

- v108 is a planned DualPlane-LifeSAM refactor and extension.
- The intended method separates current-frame output from durable SAM2 memory.
- The intended method introduces a Physical Gap Hypothesis Graph so large uncovered regions can be split, supported, conflicted, and delayed before memory admission.
- Phase0 has locked current evidence and module contracts only.

## What v108 is not allowed to claim now

- No v108 experiment has passed.
- No visual case is accepted without user review.
- No metric is a final acceptance gate.
- No p34, p39, p40, p41, p42, or p43 result is upgraded into success.
- No SAM2 or LingBot capability is claimed beyond what current source and artifacts show.

## Inherited components

- SAM2 video and image prediction are inherited components.
- LingBot Map depth, pose, intrinsics, and confidence are inherited geometry sources.
- v105/v106/v107 runners are evidence and reference harnesses, not the v108 implementation architecture.

## Novel implementation boundary

The v108 implementation boundary is the combination of:

- explicit Output Plane versus SAM2 Memory Plane,
- physical gap hypotheses with multi-seed, feature, geometry, and anchor-conflict evidence,
- lifecycle transactions that can output current-frame candidates without immediate durable memory mutation,
- online event instrumentation that does not depend on frozen reference labels,
- diagnostic casebooks that remain `USER_REVIEW_PENDING`.
"""


def build_phase0(output_root: Path, write_docs: bool) -> dict[str, Any]:
    now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    phase0_root = output_root / "phase0"
    phase0_root.mkdir(parents=True, exist_ok=True)
    docs_root = REPO_ROOT / "docs" / "v108" / "modules"
    module_status = ensure_module_docs(docs_root, write_docs=write_docs)
    source_scan = scan_source_patterns(SOURCE_FILES)

    artifacts = {
        "current_fact_lock.json": current_fact_lock(now, module_status),
        "sam2_mutation_contract.json": sam2_mutation_contract(now, source_scan),
        "online_reference_violation_scan.json": online_reference_violation_scan(now, source_scan),
        "lingbot_capability_contract.json": lingbot_capability_contract(now, source_scan),
        "module_doc_status.json": module_status,
    }
    for filename, payload in artifacts.items():
        write_json(phase0_root / filename, payload)
    (phase0_root / "novelty_boundary.md").write_text(novelty_boundary_md(now), encoding="utf-8")

    written = []
    for filename in REQUIRED_PHASE0_ARTIFACTS:
        path = phase0_root / filename
        written.append({
            "path": rel(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size if path.exists() else None,
        })
    summary = {
        "schema_version": "stream4d_v108_phase0_summary_v1",
        "generated_at_utc": now,
        "output_root": rel(output_root),
        "phase0_root": rel(phase0_root),
        "write_module_docs": write_docs,
        "required_phase0_artifacts": REQUIRED_PHASE0_ARTIFACTS,
        "phase0_artifacts": written,
        "all_phase0_artifacts_present": all(item["exists"] for item in written),
        "module_doc_status": {
            "all_required_docs_present": module_status["all_required_docs_present"],
            "all_required_sections_present": module_status["all_required_sections_present"],
            "present_doc_count": module_status["present_doc_count"],
            "required_doc_count": module_status["required_doc_count"],
            "missing_files": module_status["missing_files"],
            "docs_with_missing_sections": module_status["docs_with_missing_sections"],
        },
        "v108_experiment_run": False,
        "gpu_used": False,
        "final_acceptance_claim": False,
    }
    write_json(output_root / "phase0_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to Stream3D/outputs/audit/v108_phase0_fact_lock_<UTC timestamp>.",
    )
    parser.add_argument(
        "--no-write-module-docs",
        action="store_true",
        help="Only validate docs/v108/modules instead of creating missing docs.",
    )
    args = parser.parse_args()

    if args.output_root:
        output_root = REPO_ROOT / args.output_root
    else:
        tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "Stream3D" / "outputs" / "audit" / f"v108_phase0_fact_lock_{tag}"

    summary = build_phase0(output_root, write_docs=not args.no_write_module_docs)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if not summary["all_phase0_artifacts_present"]:
        return 2
    if not summary["module_doc_status"]["all_required_docs_present"]:
        return 3
    if not summary["module_doc_status"]["all_required_sections_present"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
