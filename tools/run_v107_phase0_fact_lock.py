#!/usr/bin/env python3
"""Build Stream4D v107 Phase0 fact-lock and gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

V107_PLAN = ROOT / "docs/stream4d_v107_recoverability_aware_lifecycle_memory_plan.md"
V106_RECAP = ROOT / "docs/stream4d_v106_实验结果复盘.md"
CURRENT_BEST_CONFIG = (
    ROOT
    / "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"
)
CURRENT_BEST_SUMMARY = (
    ROOT
    / "Stream3D/outputs/audit/"
    / "v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505"
    / "v106_stateful_sam2_rolling_scene_stream/summary.json"
)
LINGBOT_REAL_SUMMARY = (
    ROOT
    / "Stream3D/outputs/audit/v106_phase8_lingbot_real_stream_contract_20260712_1850_scene0050_c2"
    / "lingbot_stream_contract_summary.json"
)
LINGBOT_PACKETS = (
    ROOT
    / "Stream3D/outputs/audit/v106_phase8_lingbot_real_stream_contract_20260712_1850_scene0050_c2"
    / "lingbot_frame_geometry_packets.json"
)
LINGBOT_GATE = (
    ROOT
    / "Stream3D/outputs/audit/v106_phase8_lingbot_shadow_gate_20260712_1853/phase8/gate_summary.json"
)
LINGBOT_PARITY = (
    ROOT
    / "Stream3D/outputs/audit/v106_phase8_lingbot_shadow_gate_20260712_1853/phase8/"
    / "lingbot_shadow_parity_summary.json"
)

REQUIRED_MODULE_DOCS = [
    "v107_pipeline_contract.md",
    "current_v106_fact_lock.md",
    "sam2_object_memory_semantics.md",
    "lifecycle_state_machine.md",
    "identity_capsule_design.md",
    "probation_manager_design.md",
    "masklet_watcher_design.md",
    "transactional_memory_design.md",
    "active_set_scheduler_design.md",
    "lingbot_provider_design.md",
    "geometry_capsule_design.md",
    "reactivation_prompt_design.md",
    "reference_metric_design.md",
    "lifecycle_metric_design.md",
    "profiling_contract.md",
    "cache_contract.md",
    "video_casebook_contract.md",
    "related_work_novelty_audit.md",
]
REQUIRED_SECTIONS = [
    "Responsibilities",
    "Inputs",
    "Outputs",
    "Persistent State",
    "Forbidden Actions",
    "Failure Modes",
    "Artifact Schema",
    "Unit Tests",
    "Integration Tests",
]

RELATED_WORK = [
    {
        "method": "SAM 2",
        "source_url": "https://arxiv.org/abs/2408.00714",
        "source_type": "paper",
        "phase0_takeaway": (
            "SAM 2 is the promptable image/video segmentation foundation model with streaming memory; "
            "v107 must treat its video inference_state as the expensive active runtime, not as a stable "
            "long-horizon identity registry."
        ),
        "v107_boundary": (
            "Do not claim a new SAM2 architecture. v107 wraps and audits runtime object-slot lifecycle, "
            "identity virtualization, and reactivation around SAM2."
        ),
    },
    {
        "method": "SAM2Long",
        "source_url": "https://arxiv.org/abs/2410.16268",
        "source_type": "paper",
        "phase0_takeaway": (
            "SAM2Long targets long-video SAM2 robustness with a training-free memory tree/pathway search "
            "for segmentation uncertainty and error accumulation."
        ),
        "v107_boundary": (
            "Do not claim to be the first long-video SAM2 memory method. v107's narrower claim must be "
            "object lifecycle virtualization with recoverable inactive capsules in this Stream4D system."
        ),
    },
    {
        "method": "Efficient-SAM2",
        "source_url": "https://arxiv.org/abs/2602.08224",
        "source_type": "paper",
        "phase0_takeaway": (
            "Efficient-SAM2 accelerates SAM2 by object-aware sparse visual encoding and sparse memory retrieval, "
            "reporting speedup with a small accuracy drop."
        ),
        "v107_boundary": (
            "Do not claim generic SAM2 acceleration. v107 may report speed only if lifecycle scheduling keeps "
            "or improves frozen-frame mask fidelity."
        ),
    },
    {
        "method": "DAM4SAM / distractor-aware memory for SAM2",
        "source_url": "https://arxiv.org/abs/2411.17576",
        "source_type": "paper",
        "phase0_takeaway": (
            "DAM4SAM adds recent-appearance and distractor-resolving memory for visual object tracking with SAM2."
        ),
        "v107_boundary": (
            "Do not claim distractor-aware SAM2 memory novelty. v107 should use cannot-link/sibling relations "
            "as lifecycle safeguards, not as a replacement for a proven distractor memory module."
        ),
    },
    {
        "method": "XMem",
        "source_url": "https://arxiv.org/abs/2207.07115",
        "source_type": "paper",
        "phase0_takeaway": (
            "XMem uses sensory, working, and long-term stores for long-term video object segmentation."
        ),
        "v107_boundary": (
            "Do not claim the first multi-store video memory. v107's distinct claim is coupling SAM2 runtime "
            "slot budgeting to persistent identity capsules and geometry-assisted reactivation gates."
        ),
    },
    {
        "method": "Cutie",
        "source_url": "https://arxiv.org/abs/2310.12982",
        "source_type": "paper",
        "phase0_takeaway": (
            "Cutie improves VOS with object-level memory reading and object representations to reduce matching noise."
        ),
        "v107_boundary": (
            "Do not claim first object-level VOS memory. v107 should be positioned as a systems lifecycle layer "
            "around SAM2, not a new object transformer."
        ),
    },
    {
        "method": "BoT-SORT",
        "source_url": "https://arxiv.org/abs/2206.14651",
        "source_type": "paper",
        "phase0_takeaway": (
            "BoT-SORT combines motion, appearance, camera-motion compensation, and Kalman state for robust MOT IDs."
        ),
        "v107_boundary": (
            "Do not claim first recoverable identity tracking. v107's identity capsules must be evaluated as "
            "SAM2 objectlet lifecycle state, not as a general MOT replacement."
        ),
    },
    {
        "method": "LingBot-Map / Geometric Context Transformer",
        "source_url": "https://arxiv.org/abs/2604.14141",
        "source_type": "paper",
        "phase0_takeaway": (
            "LingBot-Map is a streaming 3D reconstruction model with anchor context, pose-reference window, "
            "and trajectory memory for compact long-sequence geometry state."
        ),
        "v107_boundary": (
            "Do not claim LingBot proves segmentation or identity improvement. In v107 it is a geometry packet "
            "provider until Phase6/7 prove capsule and reactivation value."
        ),
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - artifact should record parse failure
        return {"_parse_error": f"{type(exc).__name__}: {exc}"}


def module_docs_audit() -> dict[str, Any]:
    module_dir = ROOT / "docs/v107/modules"
    files = sorted(p.name for p in module_dir.glob("*.md"))
    missing_files = [name for name in REQUIRED_MODULE_DOCS if name not in files]
    missing_sections: dict[str, list[str]] = {}
    for name in REQUIRED_MODULE_DOCS:
        path = module_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        missing = [section for section in REQUIRED_SECTIONS if section not in text]
        if missing:
            missing_sections[name] = missing
    return {
        "module_dir": rel(module_dir),
        "required_file_count": len(REQUIRED_MODULE_DOCS),
        "found_file_count": len(files),
        "files": files,
        "missing_files": missing_files,
        "required_sections": REQUIRED_SECTIONS,
        "missing_sections": missing_sections,
        "passes": len(files) >= len(REQUIRED_MODULE_DOCS) and not missing_files and not missing_sections,
    }


def current_best_fact_lock() -> dict[str, Any]:
    summary = read_json(CURRENT_BEST_SUMMARY)
    config = parse_config(CURRENT_BEST_CONFIG)
    rolling = summary.get("v106_sam2_rolling_state", {})
    frame_ids = summary.get("frame_ids", [])
    metrics = {
        "scene_id": summary.get("scene_id"),
        "frame_count": summary.get("frame_count"),
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "frame_stride_observed": (
            int(frame_ids[1]) - int(frame_ids[0]) if isinstance(frame_ids, list) and len(frame_ids) >= 2 else None
        ),
        "wrapper_wall_time_sec": summary.get("wrapper_wall_time_sec"),
        "wrapper_total_with_v106_visual_export_wall_time_sec": summary.get(
            "wrapper_total_with_v106_visual_export_wall_time_sec"
        ),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "total_tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "total_gap_segmentation_runtime_sec": summary.get("total_gap_segmentation_runtime_sec"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "mean_visible_id_count": summary.get("mean_visible_id_count"),
        "mean_foreground_ratio": summary.get("mean_foreground_ratio"),
        "stream_add_masks_call_count": rolling.get("stream_add_masks_call_count"),
        "stream_add_masks_input_mask_count": rolling.get("stream_add_masks_input_mask_count"),
        "stream_add_masks_admitted_mask_count": rolling.get("stream_add_masks_admitted_mask_count"),
        "stream_add_masks_skipped_mask_count": rolling.get("stream_add_masks_skipped_mask_count"),
        "stream_add_masks_runtime_sec": rolling.get("stream_add_masks_runtime_sec"),
        "reconsolidate_call_count": rolling.get("reconsolidate_call_count"),
        "reconsolidate_runtime_sec": rolling.get("reconsolidate_runtime_sec"),
        "reconsolidate_frame_output_count_sum": rolling.get("reconsolidate_frame_output_count_sum"),
        "final_noncond_stream_frame_count": summary.get("final_noncond_stream_frame_count"),
        "empty_propagation_frames": summary.get("empty_propagation_frames"),
    }
    required_metric_keys = [
        "scene_id",
        "frame_count",
        "wrapper_wall_time_sec",
        "total_runtime_sec",
        "total_tracking_runtime_sec",
        "peak_cuda_memory_mb",
        "stream_add_masks_admitted_mask_count",
        "reconsolidate_call_count",
    ]
    return {
        "schema_version": "stream4d_v107_current_v106_fact_lock_v1",
        "sources": {
            "v107_plan": {"path": rel(V107_PLAN), "sha256": sha256_file(V107_PLAN)},
            "v106_recap": {"path": rel(V106_RECAP), "sha256": sha256_file(V106_RECAP)},
            "current_best_config": {"path": rel(CURRENT_BEST_CONFIG), "sha256": sha256_file(CURRENT_BEST_CONFIG)},
            "current_best_summary": {"path": rel(CURRENT_BEST_SUMMARY), "sha256": sha256_file(CURRENT_BEST_SUMMARY)},
        },
        "current_best_label": "v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty",
        "current_best_config_summary": {
            "sam2": config.get("sam2"),
            "stream": config.get("stream"),
            "birth": config.get("birth"),
            "gap": config.get("gap"),
            "v106_runtime_optimization": summary.get("v106_runtime_optimization"),
        },
        "metrics": metrics,
        "missing_required_metrics": [key for key in required_metric_keys if metrics.get(key) is None],
        "interpretation": (
            "v106 current best establishes rolling SAM2 inference_state as the baseline. The expensive terms "
            "to attack in v107 are active object count, post-start admissions, reconsolidation, and long-horizon "
            "state growth, not file-level chunk stitching."
        ),
    }


def cost_breakdown(fact_lock: dict[str, Any]) -> dict[str, Any]:
    metrics = fact_lock["metrics"]
    total = metrics.get("total_runtime_sec") or 0.0

    def frac(value: Any) -> float | None:
        if total <= 0 or value is None:
            return None
        return float(value) / float(total)

    return {
        "schema_version": "stream4d_v107_current_cost_breakdown_v1",
        "basis": "v106 current-best summary.json actual metrics",
        "summary_source": fact_lock["sources"]["current_best_summary"],
        "wall_time_sec": {
            "wrapper_wall_time_sec": metrics.get("wrapper_wall_time_sec"),
            "wrapper_total_with_visual_sec": metrics.get("wrapper_total_with_v106_visual_export_wall_time_sec"),
            "total_runtime_sec": metrics.get("total_runtime_sec"),
        },
        "runtime_components": {
            "tracking_sec": metrics.get("total_tracking_runtime_sec"),
            "tracking_fraction_of_total": frac(metrics.get("total_tracking_runtime_sec")),
            "gap_segmentation_sec": metrics.get("total_gap_segmentation_runtime_sec"),
            "gap_segmentation_fraction_of_total": frac(metrics.get("total_gap_segmentation_runtime_sec")),
            "reconsolidate_sec": metrics.get("reconsolidate_runtime_sec"),
            "reconsolidate_fraction_of_total": frac(metrics.get("reconsolidate_runtime_sec")),
            "stream_add_masks_sec": metrics.get("stream_add_masks_runtime_sec"),
            "stream_add_masks_fraction_of_total": frac(metrics.get("stream_add_masks_runtime_sec")),
        },
        "lifecycle_pressure_signals": {
            "mean_visible_id_count": metrics.get("mean_visible_id_count"),
            "mean_foreground_ratio": metrics.get("mean_foreground_ratio"),
            "stream_add_masks_admitted_mask_count": metrics.get("stream_add_masks_admitted_mask_count"),
            "stream_add_masks_skipped_mask_count": metrics.get("stream_add_masks_skipped_mask_count"),
            "reconsolidate_call_count": metrics.get("reconsolidate_call_count"),
            "peak_cuda_memory_mb": metrics.get("peak_cuda_memory_mb"),
            "final_noncond_stream_frame_count": metrics.get("final_noncond_stream_frame_count"),
        },
        "phase0_decision": (
            "Speed is not a valid success claim by itself. v107 may optimize these costs only under frozen "
            "mask/identity parity and visual-review gates."
        ),
    }


def lingbot_contract() -> dict[str, Any]:
    real = read_json(LINGBOT_REAL_SUMMARY)
    packets = read_json(LINGBOT_PACKETS)
    gate = read_json(LINGBOT_GATE)
    parity = read_json(LINGBOT_PARITY)
    packet_rows = packets.get("rows", []) if isinstance(packets, dict) else []
    return {
        "schema_version": "stream4d_v107_lingbot_capability_contract_v1",
        "sources": {
            "real_stream_summary": {"path": rel(LINGBOT_REAL_SUMMARY), "sha256": sha256_file(LINGBOT_REAL_SUMMARY)},
            "packet_records": {"path": rel(LINGBOT_PACKETS), "sha256": sha256_file(LINGBOT_PACKETS)},
            "shadow_gate": {"path": rel(LINGBOT_GATE), "sha256": sha256_file(LINGBOT_GATE)},
            "shadow_parity": {"path": rel(LINGBOT_PARITY), "sha256": sha256_file(LINGBOT_PARITY)},
        },
        "provider": real.get("provider"),
        "provider_root": real.get("provider_root"),
        "provider_code_sha256": real.get("provider_code_sha256"),
        "checkpoint_sha256": (real.get("model_info") or {}).get("checkpoint_sha256"),
        "model_class": (real.get("model_info") or {}).get("model_class"),
        "parameter_count": (real.get("model_info") or {}).get("parameter_count"),
        "evidence": {
            "real_streaming_api_audit_pass": real.get("real_streaming_api_audit_pass"),
            "provider_construction_smoke_pass": real.get("provider_construction_smoke_pass"),
            "provider_forward_smoke_pass": real.get("provider_forward_smoke_pass"),
            "contract_artifacts_complete": real.get("contract_artifacts_complete"),
            "packet_record_count": len(packet_rows),
            "summary_frame_count": real.get("frame_count"),
            "packet_schema_version": packets.get("schema_version") if isinstance(packets, dict) else None,
            "frame_ids": real.get("frame_ids"),
            "forward_runtime_sec": real.get("forward_runtime_sec"),
            "peak_memory_bytes": real.get("peak_memory_bytes"),
            "output_keys": real.get("output_keys"),
            "output_shape_summary": real.get("output_shape_summary"),
            "overlap_repush_count": real.get("overlap_repush_count"),
            "label_sha_identical": gate.get("label_sha_identical"),
            "affects_main_labels": gate.get("affects_main_labels"),
            "shadow_gate_passes": gate.get("passes"),
            "shadow_parity_passes": parity.get("passes"),
        },
        "capability_level": "shadow_geometry_packet_provider_only",
        "allowed_before_phase6": [
            "read packet records as offline evidence",
            "audit depth/pose packet completeness",
            "design capsule fields and prompt templates",
        ],
        "forbidden_before_phase6": [
            "write SAM2 masks",
            "veto pixel gaps",
            "assign final identity",
            "claim geometry-assisted reactivation improves segmentation",
            "alter scheduler admission or demotion decisions",
        ],
        "phase6_entry_requirement": (
            "A new v107 capability test must prove object-specific geometry capsules can be extracted, "
            "projected, and sanity-checked without changing main labels before LingBot influences scheduler logic."
        ),
    }


def related_work_markdown() -> str:
    lines = [
        "# Stream4D v107 Phase0 Related-Work Method Matrix",
        "",
        "This matrix records the Phase0 novelty boundary. It is not a paper claim by itself.",
        "",
        "| Method | Source | Phase0 takeaway | v107 boundary |",
        "| --- | --- | --- | --- |",
    ]
    for item in RELATED_WORK:
        lines.append(
            f"| {item['method']} | {item['source_url']} | "
            f"{item['phase0_takeaway']} | {item['v107_boundary']} |"
        )
    lines.extend(
        [
            "",
            "Phase0 novelty lock:",
            "",
            "- Allowed narrow claim: a recoverability-aware lifecycle layer around rolling SAM2 that separates stable "
            "global objectlet identity from expensive active SAM2 runtime slots, with explicit transaction and "
            "reactivation gates.",
            "- Forbidden broad claim: first long-video object memory, first object-level VOS memory, first SAM2 memory "
            "compression, first MOT re-identification, or proven geometry-assisted segmentation improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def novelty_markdown() -> str:
    return """# Stream4D v107 Phase0 Novelty Boundary

## Permitted Claim Candidates

- Recoverability-aware object lifecycle virtualization for the local rolling-SAM2 Stream4D stack.
- Stable global objectlet identity capsules decoupled from SAM2 runtime object slots.
- Transactional admission/demotion/reactivation gates that require parity evidence before live scheduler use.
- LingBot geometry as a shadow packet provider and future capsule support, only after capability gates.

## Not Permitted Without Later Evidence

- First or best long-video SAM2 memory method.
- First object-level video object segmentation memory.
- First or best SAM2 acceleration method.
- Geometry improves SAM2 masks or identity recovery.
- Human-accepted visual quality.
- General dynamic-object permanence beyond the ScanNet/static-object assumption.

## Phase0 Decision

Phase0 can approve Phase1 reference instrumentation and Phase2 SAM2 memory microbenchmarking.
It cannot approve live lifecycle scheduling, active demotion, geometry reactivation, or full-scene
performance claims.
"""


def forbidden_claims() -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v107_forbidden_claims_v1",
        "claims": [
            {
                "claim": "v107 improves segmentation quality",
                "allowed_now": False,
                "unlock_condition": "Phase2/Phase4/Phase10 frozen parity and final metrics beat controls.",
            },
            {
                "claim": "v107 is faster",
                "allowed_now": False,
                "unlock_condition": "Same-frame fidelity gates pass and wall/runtime comparisons are artifact-backed.",
            },
            {
                "claim": "LingBot geometry reactivates dormant objects",
                "allowed_now": False,
                "unlock_condition": "Phase6 capability and Phase7 geometry reactivation gates pass.",
            },
            {
                "claim": "visual quality accepted by user",
                "allowed_now": False,
                "unlock_condition": "Only user can mark USER_VISUAL_ACCEPTED; Codex may mark USER_VISUAL_REVIEW_PENDING.",
            },
            {
                "claim": "SAM2 direct post-start object add is safe",
                "allowed_now": False,
                "unlock_condition": "Source audit says direct add is rejected; v106 workaround requires Phase2 parity.",
            },
            {
                "claim": "same numeric re-add is production-safe",
                "allowed_now": False,
                "unlock_condition": "Synthetic smoke can be positive, but production safety requires Phase2 A0-A6.",
            },
        ],
    }


def gate_summary(
    fact_lock: dict[str, Any],
    module_audit: dict[str, Any],
    lingbot: dict[str, Any],
    sam2_contract: dict[str, Any],
    sam2_smoke: dict[str, Any],
) -> dict[str, Any]:
    source = sam2_contract.get("source_contract", {})
    smoke_iou = sam2_smoke.get("existing_object_frame2_iou_after_add") or {}
    low_iou = {
        key: value
        for key, value in smoke_iou.items()
        if value is not None and float(value) < 0.99
    }
    checks = [
        {
            "name": "module_docs_complete",
            "passes": bool(module_audit.get("passes")),
            "evidence": {"found": module_audit.get("found_file_count"), "required": module_audit.get("required_file_count")},
        },
        {
            "name": "v106_current_best_metrics_locked",
            "passes": not fact_lock.get("missing_required_metrics"),
            "evidence": fact_lock.get("metrics"),
        },
        {
            "name": "sam2_source_contract_locked",
            "passes": source.get("public_api_direct_new_id_after_tracking_allowed") is False
            and source.get("remove_object_reindexes_runtime_object_indices") is True
            and source.get("stable_global_id_mapping_required") is True,
            "evidence": source,
        },
        {
            "name": "sam2_gpu_smoke_executed",
            "passes": sam2_smoke.get("direct_new_id_after_tracking_allowed") is False
            and sam2_smoke.get("v106_post_start_add_success") is True
            and sam2_smoke.get("same_numeric_id_readd_success_with_v106_workaround") is True
            and sam2_smoke.get("final_infer_error") is None,
            "evidence": {
                "direct_error": sam2_smoke.get("direct_new_id_after_tracking_error"),
                "post_start_add_runtime_sec": sam2_smoke.get("v106_post_start_add_runtime_sec"),
                "remove_runtime_sec": sam2_smoke.get("remove_runtime_sec"),
                "readd_runtime_sec": sam2_smoke.get("same_numeric_id_readd_runtime_sec"),
                "peak_cuda_memory_mb": sam2_smoke.get("peak_cuda_memory_mb"),
            },
        },
        {
            "name": "lingbot_shadow_capability_locked",
            "passes": lingbot["evidence"].get("shadow_gate_passes") is True
            and lingbot["evidence"].get("label_sha_identical") is True
            and lingbot["evidence"].get("affects_main_labels") is False
            and lingbot["evidence"].get("packet_record_count") == lingbot["evidence"].get("summary_frame_count"),
            "evidence": lingbot["evidence"],
        },
        {
            "name": "related_work_and_forbidden_claims_written",
            "passes": True,
            "evidence": {"related_work_count": len(RELATED_WORK)},
        },
    ]
    phase0_passes = all(bool(check["passes"]) for check in checks)
    return {
        "schema_version": "stream4d_v107_phase0_gate_summary_v1",
        "created_unix_time": time.time(),
        "checks": checks,
        "warnings": [
            {
                "name": "synthetic_add_changes_existing_masks",
                "evidence": low_iou,
                "interpretation": (
                    "The synthetic GPU smoke proves mechanism executability, not fidelity safety. Existing-object "
                    "mask IoU after post-start add is below 0.99, so Phase1/Phase2 parity is mandatory before "
                    "live lifecycle scheduler mutation."
                ),
            }
        ]
        if low_iou
        else [],
        "phase0_passes": phase0_passes,
        "phase1_reference_instrumentation_allowed": bool(phase0_passes),
        "phase2_sam2_memory_microbenchmark_allowed": bool(phase0_passes),
        "live_lifecycle_mutation_allowed": False,
        "geometry_reactivation_allowed": False,
        "human_visual_status": "USER_VISUAL_REVIEW_NOT_RUN",
        "decision": (
            "PASS_PHASE0_WITH_WARNINGS" if phase0_passes else "NO_GO_PHASE0_FACT_LOCK_INCOMPLETE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam2-contract", required=True)
    parser.add_argument("--sam2-smoke", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    phase0 = output_root / "phase0"
    phase0.mkdir(parents=True, exist_ok=True)

    sam2_contract_path = Path(args.sam2_contract)
    if not sam2_contract_path.is_absolute():
        sam2_contract_path = ROOT / sam2_contract_path
    sam2_smoke_path = Path(args.sam2_smoke)
    if not sam2_smoke_path.is_absolute():
        sam2_smoke_path = ROOT / sam2_smoke_path

    fact_lock = current_best_fact_lock()
    module_audit = module_docs_audit()
    current_cost = cost_breakdown(fact_lock)
    lingbot = lingbot_contract()
    sam2_contract = read_json(sam2_contract_path)
    sam2_smoke = read_json(sam2_smoke_path)

    fact_lock["module_docs_audit"] = module_audit
    fact_lock["sam2_memory_api_contract"] = {
        "path": rel(sam2_contract_path),
        "sha256": sha256_file(sam2_contract_path),
    }
    fact_lock["sam2_memory_gpu_smoke"] = {
        "path": rel(sam2_smoke_path),
        "sha256": sha256_file(sam2_smoke_path),
    }

    gate = gate_summary(fact_lock, module_audit, lingbot, sam2_contract, sam2_smoke)

    write_json(phase0 / "fact_lock.json", fact_lock)
    write_json(phase0 / "current_cost_breakdown.json", current_cost)
    write_json(phase0 / "lingbot_capability_contract.json", lingbot)
    write_text(phase0 / "related_work_method_matrix.md", related_work_markdown())
    write_text(phase0 / "novelty_boundary.md", novelty_markdown())
    write_json(phase0 / "forbidden_claims.json", forbidden_claims())
    write_json(phase0 / "phase0_gate_summary.json", gate)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase0_fact_lock_run_summary_v1",
            "output_root": rel(output_root),
            "phase0_gate_summary": rel(phase0 / "phase0_gate_summary.json"),
            "decision": gate["decision"],
            "phase0_passes": gate["phase0_passes"],
            "live_lifecycle_mutation_allowed": gate["live_lifecycle_mutation_allowed"],
        },
    )
    print(json.dumps({"output_root": str(output_root), "decision": gate["decision"]}, sort_keys=True))
    return 0 if gate["phase0_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
