from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Stream3D.stream4d_v108.artifacts import ArtifactWriter, sha256_file
from Stream3D.stream4d_v108.phase1_parity import compare_label_dirs, summarize_parity


@dataclass(frozen=True)
class RollingBaselineCase:
    case_name: str
    scene_id: str
    frame_start: int
    frame_stride: int
    frame_count: int
    gpu: str
    config: str
    output_root: Path
    reference_label_dir: Path | None = None
    model_dtype: str = "bfloat16"
    label_only_visual_export: bool = True
    compact_visual_video: bool = True


@dataclass(frozen=True)
class FrameStateCount:
    frame_id: int
    chunk_frame_index: int
    object_id_count: int
    visible_id_count: int
    foreground_ratio: float


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _summary_path(output_root: Path) -> Path:
    return output_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"


def _label_dir(output_root: Path) -> Path:
    return output_root / "v106_stateful_sam2_rolling_scene_stream" / "labels"


def _frame_state_counts(summary: dict[str, Any]) -> list[FrameStateCount]:
    rows: list[FrameStateCount] = []
    for record in summary.get("records", []):
        rows.append(
            FrameStateCount(
                frame_id=int(record["frame_id"]),
                chunk_frame_index=int(record.get("chunk_frame_index", -1)),
                object_id_count=int(record.get("object_id_count", 0)),
                visible_id_count=int(record.get("visible_id_count", 0)),
                foreground_ratio=float(record.get("foreground_ratio", 0.0)),
            )
        )
    return rows


def _state_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rolling_state = summary.get("v106_sam2_rolling_state", {})
    records = summary.get("records", [])
    frame_ids = [int(row["frame_id"]) for row in records if "frame_id" in row]
    object_counts = [int(row.get("object_id_count", 0)) for row in records]
    visible_counts = [int(row.get("visible_id_count", 0)) for row in records]
    return {
        "schema_version": "stream4d_v108_phase1_runtime_state_summary_v1",
        "frame_count": int(len(frame_ids)),
        "frame_id_min": min(frame_ids) if frame_ids else None,
        "frame_id_max": max(frame_ids) if frame_ids else None,
        "object_id_count_min": min(object_counts) if object_counts else None,
        "object_id_count_max": max(object_counts) if object_counts else None,
        "visible_id_count_min": min(visible_counts) if visible_counts else None,
        "visible_id_count_max": max(visible_counts) if visible_counts else None,
        "total_object_id_count": summary.get("total_object_id_count"),
        "final_active_stream_object_count": summary.get("final_active_stream_object_count"),
        "final_noncond_stream_frame_count": summary.get("final_noncond_stream_frame_count"),
        "mean_visible_id_count": summary.get("mean_visible_id_count"),
        "mean_foreground_ratio": summary.get("mean_foreground_ratio"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "wrapper_wall_time_sec": summary.get("wrapper_wall_time_sec"),
        "wrapper_total_with_visual_export_wall_time_sec": summary.get(
            "wrapper_total_with_v106_visual_export_wall_time_sec"
        ),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "total_tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "total_gap_segmentation_runtime_sec": summary.get("total_gap_segmentation_runtime_sec"),
        "sam2_add_frame_call_count": rolling_state.get("add_frame_call_count"),
        "sam2_stream_add_masks_call_count": rolling_state.get("stream_add_masks_call_count"),
        "sam2_stream_add_masks_input_mask_count": rolling_state.get("stream_add_masks_input_mask_count"),
        "sam2_stream_add_masks_admitted_mask_count": rolling_state.get("stream_add_masks_admitted_mask_count"),
        "sam2_stream_add_masks_skipped_mask_count": rolling_state.get("stream_add_masks_skipped_mask_count"),
        "sam2_reconsolidate_call_count": rolling_state.get("reconsolidate_call_count"),
        "sam2_rolling_prune_call_count": rolling_state.get("rolling_prune_call_count"),
        "local_global_mapping_status": "not_exposed_by_v106_summary",
        "local_global_mapping_note": (
            "v106 summary exposes per-frame object/visible ID counts, but not a durable per-object "
            "local-to-global mapping table. Phase2 instrumentation must expose this explicitly."
        ),
    }


def build_v106_argv(case: RollingBaselineCase) -> list[str]:
    argv = [
        "--config",
        case.config,
        "--scene-id",
        case.scene_id,
        "--frame-start",
        str(case.frame_start),
        "--frame-stride",
        str(case.frame_stride),
        "--frame-count",
        str(case.frame_count),
        "--output-root",
        case.output_root.as_posix(),
        "--gpu",
        str(case.gpu),
        "--model-dtype",
        case.model_dtype,
    ]
    if case.label_only_visual_export:
        argv.append("--label-only-visual-export")
    if case.compact_visual_video:
        argv.append("--compact-visual-video")
    return argv


def run_phase1_refactor_candidate(case: RollingBaselineCase, repo_root: Path) -> dict[str, Any]:
    """Run the frozen rolling engine through a v108-owned artifact wrapper.

    This is a Phase1 compatibility step: it proves that the v108 package can
    own orchestration and diagnostics while the mature v106 SAM2 rolling core is
    still reused as the output-preserving engine.
    """

    from tools.run_v106_stateful_sam2_rolling_scene_stream import main as run_v106_main

    output_root = Path(case.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(output_root)
    argv = build_v106_argv(case)

    started = time.time()
    exit_code = int(run_v106_main(argv))
    wrapper_elapsed_sec = float(time.time() - started)
    if exit_code != 0:
        failure = {
            "schema_version": "stream4d_v108_phase1_refactor_candidate_summary_v1",
            "case_name": case.case_name,
            "scene_id": case.scene_id,
            "v108_candidate_run": True,
            "core_engine_exit_code": exit_code,
            "wrapper_elapsed_sec": wrapper_elapsed_sec,
            "status": "FAILED_CORE_ENGINE",
            "v108_refactor_scope": "orchestration_and_artifact_layer",
            "core_engine": "tools.run_v106_stateful_sam2_rolling_scene_stream.main",
            "v108_inner_sam2_engine_refactored": False,
            "command_argv": argv,
        }
        writer.write_json(
            "phase1_refactor_candidate_summary.json",
            failure,
            "stream4d_v108_phase1_refactor_candidate_summary_v1",
        )
        return failure

    summary_path = _summary_path(output_root)
    candidate_label_dir = _label_dir(output_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    label_count = len(list(candidate_label_dir.glob("frame_*.png")))

    state_rows = _frame_state_counts(summary)
    state_summary = _state_summary(summary)
    writer.write_csv(
        "phase1_frame_state_counts.csv",
        state_rows,
        "stream4d_v108_phase1_frame_state_counts_v1",
    )
    writer.write_json(
        "phase1_runtime_state_summary.json",
        state_summary,
        "stream4d_v108_phase1_runtime_state_summary_v1",
    )

    parity_summary: dict[str, Any] | None = None
    if case.reference_label_dir is not None:
        rows = compare_label_dirs(candidate_label_dir, case.reference_label_dir)
        writer.write_csv(
            "phase1_label_parity_rows.csv",
            rows,
            "stream4d_v108_phase1_label_parity_rows_v1",
        )
        parity_summary = summarize_parity(rows)
        parity_summary.update(
            {
                "candidate_label_dir": _repo_relative(candidate_label_dir, repo_root),
                "reference_label_dir": _repo_relative(case.reference_label_dir, repo_root),
                "diagnostic_only": True,
                "acceptance_gate": False,
            }
        )

    result = {
        "schema_version": "stream4d_v108_phase1_refactor_candidate_summary_v1",
        "case_name": case.case_name,
        "scene_id": case.scene_id,
        "frame_start": int(case.frame_start),
        "frame_stride": int(case.frame_stride),
        "frame_count": int(case.frame_count),
        "gpu": str(case.gpu),
        "config": case.config,
        "v108_candidate_run": True,
        "v108_refactor_scope": "orchestration_and_artifact_layer",
        "core_engine": "tools.run_v106_stateful_sam2_rolling_scene_stream.main",
        "core_engine_summary_schema": summary.get("schema_version"),
        "v108_inner_sam2_engine_refactored": False,
        "reason_inner_engine_not_refactored": (
            "Phase1 compatibility candidate reuses the frozen v106 rolling SAM2 engine to preserve output; "
            "further extraction is still required before claiming the inner engine is fully refactored."
        ),
        "core_engine_exit_code": exit_code,
        "command_argv": argv,
        "wrapper_elapsed_sec": wrapper_elapsed_sec,
        "candidate_summary_path": _repo_relative(summary_path, repo_root),
        "candidate_summary_sha256": sha256_file(summary_path),
        "candidate_label_dir": _repo_relative(candidate_label_dir, repo_root),
        "candidate_label_count": int(label_count),
        "reference_label_dir": _repo_relative(case.reference_label_dir, repo_root)
        if case.reference_label_dir is not None
        else "",
        "parity_summary": parity_summary,
        "runtime_state_summary": state_summary,
        "artifact_manifest": writer.manifest(),
        "status": "OK" if label_count == int(case.frame_count) else "LABEL_COUNT_MISMATCH",
    }
    writer.write_json(
        "phase1_refactor_candidate_summary.json",
        result,
        "stream4d_v108_phase1_refactor_candidate_summary_v1",
    )
    return result
