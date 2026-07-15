#!/usr/bin/env python3
"""Build a v106 Phase9 three-consecutive-chunk H4 handoff chain artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v106.artifacts import read_json, sha256_file, write_json  # noqa: E402
from Stream3D.stream4d_v106.config import Phase9SceneLoopSmokeConfig  # noqa: E402
from Stream3D.stream4d_v106.phase9_scene_loop import H2_VARIANT, H4_VARIANT, run_phase9_scene_loop_smoke  # noqa: E402


SUPPORTED_HANDOFF_REPLAY_VARIANTS = {H2_VARIANT, H4_VARIANT}


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _rel(path: str | Path) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _summary_path_from_loop_summary(summary: dict[str, Any]) -> Path:
    replay_summary = str(summary.get("handoff_replay_summary") or "")
    if not replay_summary:
        raise RuntimeError(f"missing handoff_replay_summary in boundary {summary.get('boundary_name')}")
    path = _resolve(replay_summary)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _labels_dir_from_replay_summary(summary_path: Path) -> Path:
    labels_dir = summary_path.parent / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)
    return labels_dir


def _metric_record(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    metrics = summary.get("handoff_overlap_metrics") or {}
    aggregate = metrics.get("aggregate") or {}
    replay = summary.get("handoff_replay") or {}
    drift_filter = summary.get("handoff_drift_filter") or {}
    phase5 = summary.get("phase5_repair_birth_defer") or {}
    residual = summary.get("residual_repair") or {}
    replay_summary_path = _resolve(summary.get("handoff_replay_summary") or "")
    replay_summary = read_json(replay_summary_path) if replay_summary_path.exists() else {}

    return {
        "boundary": summary.get("boundary_name"),
        "from_chunk_index": summary.get("from_chunk_index"),
        "to_chunk_index": summary.get("to_chunk_index"),
        "passes": bool(summary.get("passes", False)),
        "summary_path": _rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "replay_executed": bool(replay.get("executed", False)),
        "replay_runtime_sec": replay.get("runtime_sec"),
        "replay_summary": _rel(replay_summary_path) if replay_summary_path.exists() else "",
        "replay_summary_sha256": sha256_file(replay_summary_path) if replay_summary_path.exists() else None,
        "active_birth_records": summary.get("active_birth_records"),
        "active_birth_records_sha256": summary.get("active_birth_records_sha256"),
        "phase5_repair_birth_defer_enabled": bool(phase5.get("enabled", False)),
        "handoff_drift_filter_enabled": bool(drift_filter.get("enabled", False)),
        "handoff_drift_filter_dropped_obj_ids": drift_filter.get("dropped_obj_ids"),
        "handoff_drift_filter_audit": drift_filter.get("audit_path"),
        "handoff_drift_filter_audit_sha256": drift_filter.get("audit_sha256"),
        "residual_repair_enabled": bool(residual.get("enabled", False)),
        "residual_birth_bank_returncode": (residual.get("birth_bank") or {}).get("returncode"),
        "residual_birth_bank_runtime_sec": (residual.get("birth_bank") or {}).get("runtime_sec"),
        "residual_replay_returncode": (residual.get("replay") or {}).get("returncode"),
        "residual_replay_runtime_sec": (residual.get("replay") or {}).get("runtime_sec"),
        "replay_total_runtime_sec": replay_summary.get("total_runtime_sec"),
        "replay_tracking_runtime_sec": replay_summary.get("total_tracking_runtime_sec"),
        "replay_peak_cuda_memory_mb": replay_summary.get("peak_cuda_memory_mb"),
        "replay_birth_record_count": replay_summary.get("birth_record_count"),
        "reference_total_runtime_sec": replay_summary.get("reference_total_runtime_sec"),
        "runtime_ratio_vs_reference": replay_summary.get("runtime_ratio_vs_reference"),
        "min_foreground_iou": replay_summary.get("min_foreground_iou"),
        "mean_foreground_iou": replay_summary.get("mean_foreground_iou"),
        "min_CCOC": aggregate.get("min_CCOC"),
        "min_HIR": aggregate.get("min_HIR"),
        "min_HCR": aggregate.get("min_HCR"),
        "min_OPC": aggregate.get("min_OPC"),
        "max_CFR": aggregate.get("max_CFR"),
        "max_CMR": aggregate.get("max_CMR"),
        "max_BFMR": aggregate.get("max_BFMR"),
        "mean_CCOC": aggregate.get("mean_CCOC"),
        "mean_HCR": aggregate.get("mean_HCR"),
    }


def _run_boundary(
    *,
    scene_id: str,
    output_dir: Path,
    c0_summary: str,
    c0_labels_dir: str,
    reference_c1_summary: str,
    c0_chunk_index: int,
    c1_chunk_index: int,
    c0_frame_start: int,
    c1_frame_start: int,
    frame_stride: int,
    frame_count: int,
    overlap: int,
    replay_gpu: str,
    residual_gpu: str,
    sam2_replay_config: str,
    video_gpu_hot_window: int,
    handoff_replay_variant: str,
    phase5_min_area: int,
    phase5_overlap_birth_keep_min_area: int,
    phase5_selected_chunk_indices: str,
    phase5_min_birth_mask_area: int,
    phase5_min_output_mask_area: int,
    phase5_min_output_component_area: int,
    phase5_repair_birth_defer_enabled: bool = False,
    phase5_reference_birth_records: str = "",
    handoff_drift_filter_enabled: bool = False,
    handoff_drift_growth_threshold: float = 1.35,
    handoff_drift_min_probe_area: int = 20000,
    handoff_drift_probe_frame_count: int = 2,
    residual_repair_enabled: bool = False,
    residual_selected_chunk_indices: str = "",
) -> dict[str, Any]:
    if phase5_repair_birth_defer_enabled and not str(phase5_reference_birth_records).strip():
        raise ValueError("phase5_reference_birth_records is required when Phase5 repair/birth/defer is enabled")
    config = Phase9SceneLoopSmokeConfig(
        enabled=True,
        scene_id=scene_id,
        c0_summary=c0_summary,
        c0_labels_dir=c0_labels_dir,
        c0_chunk_index=int(c0_chunk_index),
        c1_chunk_index=int(c1_chunk_index),
        c0_frame_start=int(c0_frame_start),
        c1_frame_start=int(c1_frame_start),
        frame_stride=int(frame_stride),
        frame_count=int(frame_count),
        overlap=int(overlap),
        reference_c1_summary=reference_c1_summary,
        handoff_replay_variant=str(handoff_replay_variant),
        execute_handoff_replay=True,
        replay_gpu=str(replay_gpu),
        residual_gpu=str(residual_gpu or replay_gpu),
        sam2_replay_config=sam2_replay_config,
        video_gpu_hot_window=int(video_gpu_hot_window),
        reuse_video_state_template=True,
        phase5_repair_birth_defer_enabled=bool(phase5_repair_birth_defer_enabled),
        phase5_reference_birth_records=str(phase5_reference_birth_records),
        phase5_variant="R1_repair_vs_birth",
        phase5_min_area=int(phase5_min_area),
        phase5_overlap_birth_keep_min_area=int(phase5_overlap_birth_keep_min_area),
        phase5_selected_chunk_indices=str(phase5_selected_chunk_indices),
        phase5_repair_overlap_coeff=0.55,
        phase5_duplicate_suppress_overlap_coeff=0.95,
        phase5_duplicate_suppress_area_ratio_min=0.90,
        phase5_birth_max_overlap_coeff=0.25,
        phase5_min_persistence_frames=1,
        phase5_large_persistent_area=4096,
        phase5_defer_new_births_until_non_overlap=True,
        phase5_min_birth_mask_area=int(phase5_min_birth_mask_area),
        phase5_min_output_mask_area=int(phase5_min_output_mask_area),
        phase5_min_output_component_area=int(phase5_min_output_component_area),
        handoff_drift_filter_enabled=bool(handoff_drift_filter_enabled),
        handoff_drift_growth_threshold=float(handoff_drift_growth_threshold),
        handoff_drift_min_probe_area=int(handoff_drift_min_probe_area),
        handoff_drift_probe_frame_count=int(handoff_drift_probe_frame_count),
        residual_repair_enabled=bool(residual_repair_enabled),
        residual_selected_chunk_indices=str(residual_selected_chunk_indices),
        residual_input_role=(
            f"v106_phase9_three_chunk_{str(handoff_replay_variant).lower()}_c{int(c0_chunk_index)}_c{int(c1_chunk_index)}"
            "_residual_gap_temporal_age_gate"
        ),
        residual_source=(
            f"v106_phase9_three_chunk_{str(handoff_replay_variant).lower()}_c{int(c0_chunk_index)}_c{int(c1_chunk_index)}"
            "_temporal_age_gate_residual_gap"
        ),
    )
    return run_phase9_scene_loop_smoke(REPO_ROOT, config, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--c0-labels-dir", required=True)
    parser.add_argument("--c1-reference-summary", required=True)
    parser.add_argument("--c2-reference-summary", required=True)
    parser.add_argument("--chunk-indices", default="0,1,2")
    parser.add_argument("--frame-starts", required=True, help="Comma-separated original frame starts for C0,C1,C2.")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--replay-gpu", default="6")
    parser.add_argument("--residual-gpu", default="")
    parser.add_argument(
        "--sam2-replay-config",
        default="configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml",
    )
    parser.add_argument("--video-gpu-hot-window", type=int, default=8)
    parser.add_argument("--phase5-min-area", type=int, default=16)
    parser.add_argument(
        "--phase5-overlap-birth-keep-min-area",
        type=int,
        default=0,
        help=(
            "Keep birth_new masks inside overlap frames when their area is at least this value; "
            "0 preserves the previous behavior of deferring overlap births to the first non-overlap frame."
        ),
    )
    parser.add_argument("--phase5-min-birth-mask-area", type=int, default=0)
    parser.add_argument("--phase5-min-output-mask-area", type=int, default=0)
    parser.add_argument("--phase5-min-output-component-area", type=int, default=0)
    parser.add_argument("--phase5-selected-chunk-indices-first-boundary", default="")
    parser.add_argument("--phase5-selected-chunk-indices-second-boundary", default="")
    parser.add_argument(
        "--handoff-replay-variant",
        choices=sorted(SUPPORTED_HANDOFF_REPLAY_VARIANTS),
        default=H4_VARIANT,
    )
    parser.add_argument("--phase5-repair-birth-defer-first-boundary", action="store_true", default=False)
    parser.add_argument("--phase5-repair-birth-defer-second-boundary", action="store_true", default=False)
    parser.add_argument("--c1-reference-birth-records", default="")
    parser.add_argument("--c2-reference-birth-records", default="")
    parser.add_argument("--handoff-drift-filter-first-boundary", action="store_true", default=False)
    parser.add_argument("--handoff-drift-filter-second-boundary", action="store_true", default=False)
    parser.add_argument("--handoff-drift-growth-threshold", type=float, default=1.35)
    parser.add_argument("--handoff-drift-min-probe-area", type=int, default=20000)
    parser.add_argument("--handoff-drift-probe-frame-count", type=int, default=2)
    parser.add_argument("--residual-repair-first-boundary", action="store_true", default=False)
    parser.add_argument("--residual-repair-second-boundary", action="store_true", default=False)
    parser.add_argument("--residual-selected-chunk-indices", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunk_indices = [int(v) for v in str(args.chunk_indices).split(",") if v.strip()]
    frame_starts = [int(v) for v in str(args.frame_starts).split(",") if v.strip()]
    if len(chunk_indices) != 3:
        raise ValueError(f"--chunk-indices must contain exactly 3 integers, got {chunk_indices}")
    if len(frame_starts) != 3:
        raise ValueError(f"--frame-starts must contain exactly 3 integers, got {frame_starts}")

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    first_dir = output_root / f"scene_loop_boundary_c{chunk_indices[0]}_c{chunk_indices[1]}"
    first = _run_boundary(
        scene_id=args.scene_id,
        output_dir=first_dir,
        c0_summary=args.c0_summary,
        c0_labels_dir=args.c0_labels_dir,
        reference_c1_summary=args.c1_reference_summary,
        c0_chunk_index=chunk_indices[0],
        c1_chunk_index=chunk_indices[1],
        c0_frame_start=frame_starts[0],
        c1_frame_start=frame_starts[1],
        frame_stride=args.frame_stride,
        frame_count=args.frame_count,
        overlap=args.overlap,
        replay_gpu=args.replay_gpu,
        residual_gpu=args.residual_gpu,
        sam2_replay_config=args.sam2_replay_config,
        video_gpu_hot_window=args.video_gpu_hot_window,
        handoff_replay_variant=args.handoff_replay_variant,
        phase5_min_area=args.phase5_min_area,
        phase5_overlap_birth_keep_min_area=args.phase5_overlap_birth_keep_min_area,
        phase5_selected_chunk_indices=args.phase5_selected_chunk_indices_first_boundary,
        phase5_min_birth_mask_area=args.phase5_min_birth_mask_area,
        phase5_min_output_mask_area=args.phase5_min_output_mask_area,
        phase5_min_output_component_area=args.phase5_min_output_component_area,
        phase5_repair_birth_defer_enabled=bool(args.phase5_repair_birth_defer_first_boundary),
        phase5_reference_birth_records=args.c1_reference_birth_records,
        handoff_drift_filter_enabled=bool(args.handoff_drift_filter_first_boundary),
        handoff_drift_growth_threshold=float(args.handoff_drift_growth_threshold),
        handoff_drift_min_probe_area=int(args.handoff_drift_min_probe_area),
        handoff_drift_probe_frame_count=int(args.handoff_drift_probe_frame_count),
        residual_repair_enabled=bool(args.residual_repair_first_boundary),
        residual_selected_chunk_indices=args.residual_selected_chunk_indices,
    )

    first_summary_path = first_dir / "scene_loop_smoke_summary.json"
    first_replay_summary = _summary_path_from_loop_summary(first)
    first_replay_labels = _labels_dir_from_replay_summary(first_replay_summary)

    second_dir = output_root / f"scene_loop_boundary_c{chunk_indices[1]}_c{chunk_indices[2]}"
    second = _run_boundary(
        scene_id=args.scene_id,
        output_dir=second_dir,
        c0_summary=_rel(first_replay_summary),
        c0_labels_dir=_rel(first_replay_labels),
        reference_c1_summary=args.c2_reference_summary,
        c0_chunk_index=chunk_indices[1],
        c1_chunk_index=chunk_indices[2],
        c0_frame_start=frame_starts[1],
        c1_frame_start=frame_starts[2],
        frame_stride=args.frame_stride,
        frame_count=args.frame_count,
        overlap=args.overlap,
        replay_gpu=args.replay_gpu,
        residual_gpu=args.residual_gpu,
        sam2_replay_config=args.sam2_replay_config,
        video_gpu_hot_window=args.video_gpu_hot_window,
        handoff_replay_variant=args.handoff_replay_variant,
        phase5_min_area=args.phase5_min_area,
        phase5_overlap_birth_keep_min_area=args.phase5_overlap_birth_keep_min_area,
        phase5_selected_chunk_indices=args.phase5_selected_chunk_indices_second_boundary,
        phase5_min_birth_mask_area=args.phase5_min_birth_mask_area,
        phase5_min_output_mask_area=args.phase5_min_output_mask_area,
        phase5_min_output_component_area=args.phase5_min_output_component_area,
        phase5_repair_birth_defer_enabled=bool(args.phase5_repair_birth_defer_second_boundary),
        phase5_reference_birth_records=args.c2_reference_birth_records,
        handoff_drift_filter_enabled=bool(args.handoff_drift_filter_second_boundary),
        handoff_drift_growth_threshold=float(args.handoff_drift_growth_threshold),
        handoff_drift_min_probe_area=int(args.handoff_drift_min_probe_area),
        handoff_drift_probe_frame_count=int(args.handoff_drift_probe_frame_count),
        residual_repair_enabled=bool(args.residual_repair_second_boundary),
        residual_selected_chunk_indices=args.residual_selected_chunk_indices,
    )
    second_summary_path = second_dir / "scene_loop_smoke_summary.json"

    boundaries = [_metric_record(first_summary_path), _metric_record(second_summary_path)]
    verification = {
        "schema_version": "stream4d_v106_phase9_three_chunk_h4_chain_v2",
        "scope": "single_scene_three_consecutive_chunk_chain_smoke_not_full_dev",
        "scene_id": args.scene_id,
        "chunk_indices": chunk_indices,
        "frame_starts": frame_starts,
        "frame_stride": int(args.frame_stride),
        "frame_count_per_chunk": int(args.frame_count),
        "overlap": int(args.overlap),
        "handoff_replay_variant": str(args.handoff_replay_variant),
        "all_boundaries_pass": all(bool(row.get("passes")) for row in boundaries),
        "dev_promotion_scope_complete": False,
        "dev_promotion_scope_reason": "single scene only; Phase16.3 requires at least two scenes and additional full-dev metrics",
        "boundaries": boundaries,
        "first_boundary_summary": _rel(first_summary_path),
        "second_boundary_summary": _rel(second_summary_path),
        "inputs": {
            "c0_summary": _rel(args.c0_summary),
            "c0_labels_dir": _rel(args.c0_labels_dir),
            "c1_reference_summary": _rel(args.c1_reference_summary),
            "c2_reference_summary": _rel(args.c2_reference_summary),
            "c1_reference_birth_records": _rel(args.c1_reference_birth_records) if args.c1_reference_birth_records else "",
            "c2_reference_birth_records": _rel(args.c2_reference_birth_records) if args.c2_reference_birth_records else "",
            "sam2_replay_config": _rel(args.sam2_replay_config),
        },
        "repair_controls": {
            "phase5_repair_birth_defer_first_boundary": bool(args.phase5_repair_birth_defer_first_boundary),
            "phase5_repair_birth_defer_second_boundary": bool(args.phase5_repair_birth_defer_second_boundary),
            "phase5_min_area": int(args.phase5_min_area),
            "phase5_overlap_birth_keep_min_area": int(args.phase5_overlap_birth_keep_min_area),
            "phase5_selected_chunk_indices_first_boundary": str(args.phase5_selected_chunk_indices_first_boundary),
            "phase5_selected_chunk_indices_second_boundary": str(args.phase5_selected_chunk_indices_second_boundary),
            "phase5_min_birth_mask_area": int(args.phase5_min_birth_mask_area),
            "phase5_min_output_mask_area": int(args.phase5_min_output_mask_area),
            "phase5_min_output_component_area": int(args.phase5_min_output_component_area),
            "handoff_drift_filter_first_boundary": bool(args.handoff_drift_filter_first_boundary),
            "handoff_drift_filter_second_boundary": bool(args.handoff_drift_filter_second_boundary),
            "handoff_drift_growth_threshold": float(args.handoff_drift_growth_threshold),
            "handoff_drift_min_probe_area": int(args.handoff_drift_min_probe_area),
            "handoff_drift_probe_frame_count": int(args.handoff_drift_probe_frame_count),
            "residual_repair_first_boundary": bool(args.residual_repair_first_boundary),
            "residual_repair_second_boundary": bool(args.residual_repair_second_boundary),
            "residual_selected_chunk_indices": str(args.residual_selected_chunk_indices),
            "residual_gpu": str(args.residual_gpu or args.replay_gpu),
        },
    }
    out_path = output_root / "three_chunk_chain_verification.json"
    write_json(out_path, verification)
    print(json.dumps({"chain_verification": _rel(out_path), "sha256": sha256_file(out_path), **verification}, indent=2, sort_keys=True))
    return 0 if verification["all_boundaries_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
