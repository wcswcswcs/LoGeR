from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .artifacts import file_record, read_json, sha256_file, write_json
from .chunk_schedule import build_sequential_windows


def _split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"_read_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_payload": payload}


def _path_record(repo_root: Path, raw_path: str, required: bool = True) -> Dict[str, Any]:
    path = _resolve(repo_root, raw_path)
    rel_path = _rel(repo_root, path)
    return file_record(repo_root, rel_path, required=required)


def _nested_get(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_b4_row(repo_root: Path, summary_path: str) -> Dict[str, Any]:
    path = _resolve(repo_root, summary_path)
    summary = _read_json_if_exists(path)
    return {
        "variant": "B4",
        "status": "present" if summary else "missing",
        "evidence_path": _rel(repo_root, path),
        "scene_id": summary.get("scene_id"),
        "frame_count": summary.get("frame_count"),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "coverage": None,
        "asa": None,
        "ttp": None,
        "max_cfr": None,
        "max_cmr": None,
        "scope": "single_chunk_artifact_replay",
        "full_dev_evidence": False,
    }


def _extract_b6_row(repo_root: Path, summary_path: str, gate_path: str) -> Dict[str, Any]:
    summary_file = _resolve(repo_root, summary_path)
    gate_file = _resolve(repo_root, gate_path)
    summary = _read_json_if_exists(summary_file)
    gate = _read_json_if_exists(gate_file)
    metrics = _nested_get(gate, ("variant_metrics", "R1_repair_vs_birth"))
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "variant": "B6",
        "status": "present" if summary and gate else "missing",
        "evidence_path": _rel(repo_root, summary_file),
        "gate_path": _rel(repo_root, gate_file),
        "scene_id": summary.get("scene_id"),
        "frame_count": summary.get("frame_count"),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "coverage": metrics.get("coverage"),
        "asa": metrics.get("asa"),
        "ttp": metrics.get("ttp"),
        "max_cfr": metrics.get("max_cfr"),
        "max_cmr": metrics.get("max_cmr"),
        "scope": "scene0050_single_chunk_exact_streaming_gate",
        "full_dev_evidence": False,
        "chunk_gate_pass": bool(gate.get("passes", False)),
    }


def _extract_b7_row(repo_root: Path, decision_path: str) -> Dict[str, Any]:
    path = _resolve(repo_root, decision_path)
    decision = _read_json_if_exists(path)
    best = decision.get("best_coverage_variant")
    if not isinstance(best, dict):
        best = {}
    exact = decision.get("exact_r1")
    if not isinstance(exact, dict):
        exact = {}
    return {
        "variant": "B7",
        "status": decision.get("status", "missing") if decision else "missing",
        "evidence_path": _rel(repo_root, path),
        "scene_id": decision.get("scene_id"),
        "frame_count": decision.get("frame_count"),
        "total_runtime_sec": best.get("runtime_sec"),
        "tracking_runtime_sec": None,
        "peak_cuda_memory_mb": best.get("peak_mb"),
        "coverage": best.get("coverage"),
        "asa": best.get("asa"),
        "ttp": best.get("ttp"),
        "max_cfr": best.get("max_cfr"),
        "max_cmr": best.get("max_cmr"),
        "exact_b6_runtime_sec": exact.get("total_runtime_sec"),
        "scope": "scene0050_single_chunk_specgap_probe",
        "full_dev_evidence": False,
        "promotion_pass": decision.get("status") not in {"NO_GO", "missing", None},
    }


def _scan_scene_path_evidence(repo_root: Path, required_scenes: List[str]) -> Dict[str, Any]:
    audit_root = repo_root / "Stream3D" / "outputs" / "audit"
    scene_records: Dict[str, Any] = {}
    if not audit_root.exists():
        for scene in required_scenes:
            scene_records[scene] = {"path_match_count": 0, "sample_paths": []}
        return scene_records
    all_paths = [path for path in audit_root.rglob("*") if path.is_file() and "v106" in str(path)]
    for scene in required_scenes:
        matches = [_rel(repo_root, path) for path in all_paths if scene in str(path)]
        scene_records[scene] = {
            "path_match_count": len(matches),
            "sample_paths": sorted(matches)[:20],
            "has_any_v106_evidence": bool(matches),
        }
    return scene_records


def _scan_video_index(repo_root: Path) -> List[Dict[str, Any]]:
    audit_root = repo_root / "Stream3D" / "outputs" / "audit"
    if not audit_root.exists():
        return []
    suffixes = {".mp4", ".png", ".jpg", ".jpeg"}
    rows: List[Dict[str, Any]] = []
    for path in sorted(audit_root.rglob("*")):
        if not path.is_file() or "v106" not in str(path):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rows.append(
            {
                "path": _rel(repo_root, path),
                "byte_size": path.stat().st_size,
                "suffix": path.suffix.lower(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _selected_frame_ids(repo_root: Path, scene_id: str, stride: int) -> List[int]:
    color_dir = repo_root / "Stream3D" / "data" / "scannet" / "processed" / scene_id / "color"
    files = sorted(color_dir.glob("*.jpg"), key=lambda path: int(path.stem) if path.stem.isdigit() else 10**15)
    if not files:
        files = sorted(color_dir.glob("*.png"), key=lambda path: int(path.stem) if path.stem.isdigit() else 10**15)
    ids = [int(path.stem) for path in files if path.stem.isdigit()]
    return ids[:: max(int(stride), 1)]


def _chunk_schedule(frame_ids: List[int], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not frame_ids:
        return records
    step = int(chunk_size) - int(overlap)
    for window in build_sequential_windows("phase9_manifest_scene", len(frame_ids), int(chunk_size), int(overlap)):
        start = int(window.start_frame)
        end = int(window.end_frame_exclusive)
        input_count = end - start
        input_overlap_count = min(int(window.overlap), input_count)
        output_start_local = input_overlap_count if int(window.chunk_index) > 0 else 0
        output_start = start + output_start_local
        frame_records = []
        for chunk_frame_index, scene_sampled_index in enumerate(range(start, end)):
            frame_records.append(
                {
                    "chunk_frame_index": int(chunk_frame_index),
                    "scene_sampled_index": int(scene_sampled_index),
                    "original_frame_id": int(frame_ids[scene_sampled_index]),
                    "is_overlap_input": bool(int(window.chunk_index) > 0 and chunk_frame_index < input_overlap_count),
                    "final_label_owner": bool(chunk_frame_index >= output_start_local),
                }
            )
        output_scene_sampled_indices = [
            int(row["scene_sampled_index"])
            for row in frame_records
            if bool(row["final_label_owner"])
        ]
        output_frame_ids = [int(frame_ids[index]) for index in output_scene_sampled_indices]
        records.append(
            {
                "chunk_index": int(window.chunk_index),
                "start_index": start,
                "end_index_exclusive": end,
                "frame_start": int(frame_ids[start]),
                "frame_end": int(frame_ids[end - 1]),
                "frame_count": int(input_count),
                "input_overlap_count": int(input_overlap_count),
                "output_start_chunk_frame_index": int(output_start_local),
                "output_start_index": int(output_start),
                "output_end_index_exclusive": end,
                "output_frame_count": int(len(output_frame_ids)),
                "output_frame_start": int(output_frame_ids[0]) if output_frame_ids else None,
                "output_frame_end": int(output_frame_ids[-1]) if output_frame_ids else None,
                "step": int(step),
                "configured_overlap": int(overlap),
                "overlap_frame_ids": [
                    int(frame_ids[row["scene_sampled_index"]])
                    for row in frame_records
                    if bool(row["is_overlap_input"])
                ],
                "output_scene_sampled_indices": output_scene_sampled_indices,
                "output_frame_ids": output_frame_ids,
                "frame_records": frame_records,
            }
        )
    return records


def _ownership_audit(frame_ids: List[int], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_indices = [
        int(scene_sampled_index)
        for chunk in chunks
        for scene_sampled_index in chunk.get("output_scene_sampled_indices", [])
    ]
    counts = Counter(output_indices)
    duplicate_indices = [int(index) for index, count in sorted(counts.items()) if count > 1]
    selected_indices = list(range(len(frame_ids)))
    missing_indices = [index for index in selected_indices if index not in counts]
    return {
        "selected_frame_count": int(len(selected_indices)),
        "output_frame_count": int(len(output_indices)),
        "unique_output_frame_count": int(len(counts)),
        "duplicate_output_scene_sampled_indices": duplicate_indices,
        "missing_output_scene_sampled_indices": missing_indices,
        "duplicate_output_frame_ids": [int(frame_ids[index]) for index in duplicate_indices],
        "missing_output_frame_ids": [int(frame_ids[index]) for index in missing_indices],
        "covers_selected_frames_once": bool(
            len(output_indices) == len(selected_indices)
            and len(counts) == len(selected_indices)
            and not duplicate_indices
            and not missing_indices
        ),
    }


def _extract_preflight_records(repo_root: Path, raw_paths_csv: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_path in _split_csv(raw_paths_csv):
        path = _resolve(repo_root, raw_path)
        summary = _read_json_if_exists(path)
        if not summary:
            records.append(
                {
                    "path": _rel(repo_root, path),
                    "exists": False,
                    "full_dev_evidence": False,
                    "status": "missing",
                }
            )
            continue
        video_path = Path(str(summary.get("video_path", "")))
        if not video_path.is_absolute():
            video_path = repo_root / video_path
        sheet_paths = []
        for sheet in summary.get("sheet_paths", []) if isinstance(summary.get("sheet_paths"), list) else []:
            sheet_path = Path(str(sheet))
            if not sheet_path.is_absolute():
                sheet_path = repo_root / sheet_path
            sheet_paths.append(sheet_path)
        records.append(
            {
                "path": _rel(repo_root, path),
                "exists": True,
                "status": "present",
                "scene_id": summary.get("scene_id"),
                "frame_count": summary.get("frame_count"),
                "frame_start": summary.get("frame_ids", [None])[0] if isinstance(summary.get("frame_ids"), list) else None,
                "birth_record_count": summary.get("birth_record_count"),
                "anchor_group_count": summary.get("anchor_group_count"),
                "total_runtime_sec": summary.get("total_runtime_sec"),
                "total_tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
                "reference_total_runtime_sec": summary.get("reference_total_runtime_sec"),
                "runtime_ratio_vs_reference": summary.get("runtime_ratio_vs_reference"),
                "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
                "all_labels_exact_equal": summary.get("all_labels_exact_equal"),
                "exact_label_frame_count": summary.get("exact_label_frame_count"),
                "min_foreground_iou": summary.get("min_foreground_iou"),
                "video_path": _rel(repo_root, video_path) if video_path else None,
                "video_exists": bool(video_path.exists()) if video_path else False,
                "sheet_count": len(sheet_paths),
                "all_sheets_exist": bool(sheet_paths) and all(path.exists() for path in sheet_paths),
                "full_dev_evidence": False,
                "scope": "single_chunk_preflight",
            }
        )
    return records


def _extract_dev_chain_records(repo_root: Path, raw_paths_csv: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_path in _split_csv(raw_paths_csv):
        path = _resolve(repo_root, raw_path)
        payload = _read_json_if_exists(path)
        if not payload:
            records.append(
                {
                    "path": _rel(repo_root, path),
                    "exists": False,
                    "status": "missing",
                    "full_dev_evidence": False,
                }
            )
            continue
        boundaries = payload.get("boundaries", [])
        if not isinstance(boundaries, list):
            boundaries = []
        records.append(
            {
                "path": _rel(repo_root, path),
                "exists": True,
                "status": "present",
                "sha256": sha256_file(path),
                "scope": payload.get("scope"),
                "scene_id": payload.get("scene_id"),
                "chunk_indices": payload.get("chunk_indices"),
                "frame_starts": payload.get("frame_starts"),
                "frame_stride": payload.get("frame_stride"),
                "frame_count_per_chunk": payload.get("frame_count_per_chunk"),
                "overlap": payload.get("overlap"),
                "boundary_count": len(boundaries),
                "boundary_names": [row.get("boundary") for row in boundaries if isinstance(row, dict)],
                "all_boundaries_pass": bool(payload.get("all_boundaries_pass", False)),
                "dev_promotion_scope_complete": bool(payload.get("dev_promotion_scope_complete", False)),
                "dev_promotion_scope_reason": payload.get("dev_promotion_scope_reason"),
                "min_boundary_CCOC": min(
                    (
                        float(row["min_CCOC"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("min_CCOC") is not None
                    ),
                    default=None,
                ),
                "min_boundary_HIR": min(
                    (
                        float(row["min_HIR"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("min_HIR") is not None
                    ),
                    default=None,
                ),
                "min_boundary_HCR": min(
                    (
                        float(row["min_HCR"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("min_HCR") is not None
                    ),
                    default=None,
                ),
                "max_boundary_CFR": max(
                    (
                        float(row["max_CFR"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("max_CFR") is not None
                    ),
                    default=None,
                ),
                "max_boundary_CMR": max(
                    (
                        float(row["max_CMR"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("max_CMR") is not None
                    ),
                    default=None,
                ),
                "max_boundary_BFMR": max(
                    (
                        float(row["max_BFMR"])
                        for row in boundaries
                        if isinstance(row, dict) and row.get("max_BFMR") is not None
                    ),
                    default=None,
                ),
                "full_dev_evidence": False,
            }
        )
    return records


def _extract_coverage_diagnostic_records(repo_root: Path, raw_paths_csv: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_path in _split_csv(raw_paths_csv):
        path = _resolve(repo_root, raw_path)
        payload = _read_json_if_exists(path)
        if not payload:
            records.append(
                {
                    "path": _rel(repo_root, path),
                    "exists": False,
                    "status": "missing",
                    "passes": False,
                }
            )
            continue
        scene_summaries = payload.get("scene_summaries", [])
        if not isinstance(scene_summaries, list):
            scene_summaries = []
        worst_cases = []
        for scene in scene_summaries:
            if not isinstance(scene, dict):
                continue
            for case in scene.get("worst_cases", [])[:3]:
                if isinstance(case, dict):
                    worst_cases.append(
                        {
                            "scene_id": scene.get("scene_id"),
                            "boundary": case.get("boundary"),
                            "frame_id": case.get("frame_id"),
                            "foreground_union_iou": case.get("foreground_union_iou"),
                            "coverage_recall_vs_reference": case.get("coverage_recall_vs_reference"),
                            "missing_reference_foreground_fraction": case.get("missing_reference_foreground_fraction"),
                            "pred_visible_id_count": case.get("pred_visible_id_count"),
                            "reference_visible_id_count": case.get("reference_visible_id_count"),
                            "diff_image_path": case.get("diff_image_path"),
                        }
                    )
        records.append(
            {
                "path": _rel(repo_root, path),
                "exists": True,
                "status": "present",
                "sha256": sha256_file(path),
                "schema_version": payload.get("schema_version"),
                "scene_count": payload.get("scene_count"),
                "scenes": payload.get("scenes"),
                "all_input_boundaries_pass": bool(payload.get("all_input_boundaries_pass", False)),
                "all_pass_foreground_union_iou_098": bool(payload.get("all_pass_foreground_union_iou_098", False)),
                "all_pass_coverage_recall_099": bool(payload.get("all_pass_coverage_recall_099", False)),
                "passes": bool(
                    payload.get("all_input_boundaries_pass", False)
                    and payload.get("all_pass_foreground_union_iou_098", False)
                    and payload.get("all_pass_coverage_recall_099", False)
                ),
                "worst_cases": worst_cases,
            }
        )
    return records


def _build_execution_manifest(
    repo_root: Path,
    config: Any,
    phase_config: Any,
    required_scenes: List[str],
    required_variants: List[str],
    preflight_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scene_schedules = []
    for scene_id in required_scenes:
        frame_ids = _selected_frame_ids(repo_root, scene_id, config.data.frame_stride)
        chunks = _chunk_schedule(frame_ids, config.data.chunk_size, config.data.overlap)
        ownership_audit = _ownership_audit(frame_ids, chunks)
        scene_schedules.append(
            {
                "scene_id": scene_id,
                "raw_color_frame_count": len(
                    list((repo_root / "Stream3D" / "data" / "scannet" / "processed" / scene_id / "color").glob("*.jpg"))
                )
                or len(
                    list((repo_root / "Stream3D" / "data" / "scannet" / "processed" / scene_id / "color").glob("*.png"))
                ),
                "frame_stride": config.data.frame_stride,
                "selected_frame_count": len(frame_ids),
                "chunk_size": config.data.chunk_size,
                "overlap": config.data.overlap,
                "step": int(config.data.chunk_size) - int(config.data.overlap),
                "chunk_count": len(chunks),
                "first_frame_id": frame_ids[0] if frame_ids else None,
                "last_frame_id": frame_ids[-1] if frame_ids else None,
                "chunks": chunks,
                "final_label_ownership": ownership_audit,
                "same_scene_chunk_parallelism": config.run.same_scene_chunk_parallelism,
            }
        )
    command_records = [
        {
            "name": "phase9_scene0011_chunk0_b4_replay_preflight",
            "status": "executed",
            "purpose": "Validate scene0011 exact replay path before full-dev.",
            "command": [
                "CUDA_VISIBLE_DEVICES=6",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "PYTHONUNBUFFERED=1",
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
                "tools/build_v105_phase5_frozen_birth_replay.py",
                "--config",
                "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml",
                "--scene-id",
                "scene0011_00",
                "--birth-records",
                "Stream3D/outputs/audit/v105_baseline_x_gapadaptive_scene0011_chunk0_f32_20260712/birth_bank/birth_records.json",
                "--reference-summary",
                "Stream3D/outputs/audit/v105_baseline_x_gapadaptive_scene0011_chunk0_f32_20260712/baseline_x_gapadaptive_sam2/summary.json",
                "--output-root",
                "Stream3D/outputs/audit/v106_phase9_scene0011_chunk0_b4_replay_preflight_20260712_1917",
                "--frame-start",
                "0",
                "--frame-stride",
                "5",
                "--frame-count",
                "32",
                "--anchor-birth-priority",
                "--use-video-feature-bank",
                "--video-feature-bank-storage-device",
                "cuda",
                "--video-gpu-hot-window",
                "8",
                "--reuse-video-state-template",
                "--duplicate-window-frames",
                "3",
                "--duplicate-overlap-threshold",
                "0.55",
            ],
            "evidence": preflight_records,
            "full_dev_evidence": False,
        },
        {
            "name": "full_dev_v106_scene_loop",
            "status": "not_implemented",
            "purpose": "Required to produce B4/B6 full-dev evidence with inherited masks/global IDs/lifecycle across every chunk.",
            "required_properties": [
                "cache_mode=write_only_verified_no_read",
                "cross_run_cache_read_count=0",
                "same-scene chunks sequential",
                "complete videos and boundary casebook",
                "identity metrics and MV_AP diagnostics across all required scenes",
            ],
            "blocked_by": "Current v106 runner is an artifact/gate harness; it does not yet execute a fresh full-scene sequential SAM2 scene loop.",
            "full_dev_evidence": False,
        },
        {
            "name": "b0_b1_independent_baseline_comparators",
            "status": "not_implemented_in_v106",
            "purpose": "Required Phase9 comparison rows for independent chunks and posthoc one-to-one relabel.",
            "v105_related_runner": "tools/run_v105_fullscene_multichunk_repair.py",
            "warning": "The v105 runner is chunk-windowed and explicitly does not claim continuous scene-level identity, so it can support B0/B1 comparator audits but cannot prove v106 method success.",
            "full_dev_evidence": False,
        },
    ]
    return {
        "schema_version": "stream4d_v106_phase9_full_dev_execution_manifest_v1",
        "required_scenes": required_scenes,
        "required_variants": required_variants,
        "scene_schedules": scene_schedules,
        "command_records": command_records,
        "gpu_plan": {
            "available_gpus": ["6", "7"],
            "allowed_parallelism": "different scenes may run in parallel; same-scene chunks must remain sequential",
            "same_scene_chunk_parallelism": config.run.same_scene_chunk_parallelism,
        },
        "honesty_boundary": "This manifest is preflight/run planning plus executed chunk evidence; it is not a completed full-dev cold run.",
        "phase9_config": {
            "required_cache_mode": phase_config.required_cache_mode,
            "preflight_chunk_summaries_csv": phase_config.preflight_chunk_summaries_csv,
        },
    }


def _requirement(name: str, passes: bool, actual: Any, expected: Any, evidence: Any = None) -> Dict[str, Any]:
    return {
        "name": name,
        "passes": bool(passes),
        "actual": actual,
        "expected": expected,
        "evidence": evidence,
    }


def run_phase9_full_dev(repo_root: Path, config: Any, phase_config: Any, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_scenes = _split_csv(phase_config.required_scenes_csv)
    required_variants = _split_csv(phase_config.required_variants_csv)
    scene_path_evidence = _scan_scene_path_evidence(repo_root, required_scenes)
    video_rows = _scan_video_index(repo_root)
    preflight_records = _extract_preflight_records(repo_root, phase_config.preflight_chunk_summaries_csv)
    dev_chain_records = _extract_dev_chain_records(repo_root, phase_config.dev_chain_summaries_csv)
    coverage_diagnostic_records = _extract_coverage_diagnostic_records(
        repo_root, getattr(phase_config, "coverage_diagnostics_csv", "")
    )
    execution_manifest = _build_execution_manifest(
        repo_root, config, phase_config, required_scenes, required_variants, preflight_records
    )

    b0_row = {
        "variant": "B0",
        "status": "missing",
        "scope": "full_dev_required",
        "full_dev_evidence": False,
        "reason": "No v106 B0 full-dev baseline evidence path is configured or present.",
    }
    b1_row = {
        "variant": "B1",
        "status": "missing",
        "scope": "full_dev_required",
        "full_dev_evidence": False,
        "reason": "No v106 B1 full-dev baseline evidence path is configured or present.",
    }
    b4_row = _extract_b4_row(repo_root, phase_config.phase2_b4_summary)
    b6_row = _extract_b6_row(repo_root, phase_config.phase5_b6_replay_summary, phase_config.phase5_b6_gate_records)
    b7_row = _extract_b7_row(repo_root, phase_config.phase7_final_decision)
    variant_rows = [b0_row, b1_row, b4_row, b6_row, b7_row]

    integrated = _read_json_if_exists(_resolve(repo_root, phase_config.integrated_phase0_6_run_summary))
    phase8_gate = _read_json_if_exists(_resolve(repo_root, phase_config.phase8_gate_summary))

    variant_status = {row["variant"]: row.get("status") for row in variant_rows}
    missing_variants = [name for name in required_variants if variant_status.get(name) in {None, "missing"}]
    scene_missing_any_evidence = [
        scene for scene, record in scene_path_evidence.items() if not bool(record.get("has_any_v106_evidence"))
    ]
    scenes_with_preflight = {
        str(record.get("scene_id"))
        for record in preflight_records
        if record.get("exists") and record.get("scene_id")
    }
    scene_missing_preflight_or_any = [
        scene
        for scene in required_scenes
        if scene not in scenes_with_preflight and not bool(scene_path_evidence.get(scene, {}).get("has_any_v106_evidence"))
    ]
    present_dev_chain_records = [row for row in dev_chain_records if bool(row.get("exists"))]
    dev_chain_scenes = sorted({str(row.get("scene_id")) for row in present_dev_chain_records if row.get("scene_id")})
    dev_chain_required_scene_coverage = {
        scene: scene in set(dev_chain_scenes)
        for scene in required_scenes
    }
    dev_chain_records_have_three_chunks = bool(present_dev_chain_records) and all(
        int(row.get("boundary_count") or 0) >= 2
        and isinstance(row.get("chunk_indices"), list)
        and len(row.get("chunk_indices") or []) >= 3
        for row in present_dev_chain_records
    )
    dev_chain_all_present_records_pass = bool(present_dev_chain_records) and all(
        bool(row.get("all_boundaries_pass")) for row in present_dev_chain_records
    )
    dev_chain_required_scenes_covered = bool(required_scenes) and all(dev_chain_required_scene_coverage.values())
    dev_chain_aggregate_scope_complete = bool(
        dev_chain_all_present_records_pass
        and dev_chain_records_have_three_chunks
        and dev_chain_required_scenes_covered
    )
    coverage_diagnostics_present = bool(coverage_diagnostic_records) and all(
        bool(row.get("exists")) for row in coverage_diagnostic_records
    )
    coverage_diagnostics_pass = coverage_diagnostics_present and all(
        bool(row.get("passes")) for row in coverage_diagnostic_records
    )
    has_full_dev_artifact = any("full_dev" in row.get("evidence_path", "") for row in variant_rows)
    all_variant_full_dev = all(bool(row.get("full_dev_evidence")) for row in variant_rows)
    cache_policy_pass = config.run.cache_mode == phase_config.required_cache_mode
    cache_read_count_evidence = None
    cache_read_count_zero = False
    mv_ap_diagnostics_complete = False
    complete_videos = False
    complete_identity_metrics = False
    latency_vram_complete = False

    requirements = [
        _requirement(
            "cache_mode_write_only_verified_no_read",
            cache_policy_pass,
            config.run.cache_mode,
            phase_config.required_cache_mode,
            "Config must be switched before final dev/holdout cold runs.",
        ),
        _requirement(
            "cache_read_count_zero",
            cache_read_count_zero,
            cache_read_count_evidence,
            0,
            "No authoritative full-dev cache_read_count artifact exists in current v106 evidence.",
        ),
        _requirement(
            "same_scene_chunks_sequential",
            bool(config.run.scene_stream_sequential and config.run.same_scene_chunk_parallelism == 1),
            {
                "scene_stream_sequential": config.run.scene_stream_sequential,
                "same_scene_chunk_parallelism": config.run.same_scene_chunk_parallelism,
            },
            {"scene_stream_sequential": True, "same_scene_chunk_parallelism": 1},
            "Config-level contract.",
        ),
        _requirement(
            "required_dev_scenes_have_any_v106_preflight",
            not scene_missing_preflight_or_any,
            {"scene_path_evidence": scene_path_evidence, "preflight_scenes": sorted(scenes_with_preflight)},
            required_scenes,
            "Preflight/chunk evidence is useful but not full-dev evidence.",
        ),
        _requirement(
            "required_dev_scenes_have_three_chunk_chain",
            dev_chain_aggregate_scope_complete,
            {
                "scene_coverage": dev_chain_required_scene_coverage,
                "chain_scenes": dev_chain_scenes,
                "record_count": len(dev_chain_records),
                "present_record_count": len(present_dev_chain_records),
                "all_present_records_pass": dev_chain_all_present_records_pass,
                "records_have_at_least_two_boundaries": dev_chain_records_have_three_chunks,
            },
            {
                "required_scenes": required_scenes,
                "min_consecutive_chunks_per_scene": 3,
                "all_boundaries_pass": True,
            },
            "This verifies the three-consecutive-chunk chain part of Phase16.3, not full-dev videos/metrics.",
        ),
        _requirement(
            "required_dev_scenes_have_full_dev_evidence",
            False,
            {"scene_path_evidence": scene_path_evidence, "preflight_scenes": sorted(scenes_with_preflight)},
            required_scenes,
            "No required scene currently has v106 full-dev cold-run evidence.",
        ),
        _requirement(
            "complete_videos",
            complete_videos,
            {"v106_video_or_sheet_artifact_count": len(video_rows)},
            "Each chunk exports one 32-frame MP4 and four 8-frame sheets plus boundary videos.",
            "Current scan is artifact presence only; it found no complete full-dev video index.",
        ),
        _requirement(
            "complete_identity_metrics",
            complete_identity_metrics,
            {
                "chunk_b6_gate_pass": b6_row.get("chunk_gate_pass"),
                "variant_rows_scope": [r.get("scope") for r in variant_rows],
                "coverage_diagnostics_present": coverage_diagnostics_present,
                "coverage_diagnostics_pass": coverage_diagnostics_pass,
                "coverage_diagnostic_records": coverage_diagnostic_records,
            },
            "Full-dev identity metrics for all required scenes and variants.",
            "Current evidence is chunk-scoped, not full-dev scoped; coverage diagnostics expose foreground parity failures.",
        ),
        _requirement(
            "latency_vram_complete",
            latency_vram_complete,
            {
                "integrated_phase0_6_summary_exists": bool(integrated),
                "chunk_variant_rows_with_runtime": [
                    row.get("variant") for row in variant_rows if _float_or_none(row.get("total_runtime_sec")) is not None
                ],
            },
            "Latency/VRAM table complete for B0/B1/B4/B6/B7 across required scenes.",
            "Current evidence lacks B0/B1 and full-dev scene0011 rows.",
        ),
        _requirement(
            "mv_ap_diagnostics_complete",
            mv_ap_diagnostics_complete,
            None,
            "MV_AP_window and MV_AP_scene diagnostics for frozen dev.",
            "No v106 full-dev MV_AP diagnostic artifact is configured or present.",
        ),
        _requirement(
            "b0_b1_b4_b6_b7_comparison_complete",
            not missing_variants and all_variant_full_dev,
            {"variant_status": variant_status, "all_variant_full_dev": all_variant_full_dev},
            required_variants,
            "B4/B6/B7 chunk evidence exists, but B0/B1 and full-dev scoped rows are missing.",
        ),
        _requirement(
            "lingbot_shadow_remains_no_effect",
            bool(phase8_gate.get("passes", False)),
            phase8_gate,
            "Phase8 shadow gate passes.",
            _rel(repo_root, _resolve(repo_root, phase_config.phase8_gate_summary)),
        ),
    ]

    missing_evidence = [
        {
            "requirement": record["name"],
            "actual": record["actual"],
            "expected": record["expected"],
            "evidence": record["evidence"],
        }
        for record in requirements
        if not record["passes"]
    ]
    b6_full_dev_pass = False
    b7_promotion_pass = bool(b7_row.get("promotion_pass"))
    freeze_decision = {
        "schema_version": "stream4d_v106_phase9_freeze_decision_v1",
        "status": "not_frozen",
        "frozen_variant": None,
        "candidate_from_chunk_evidence_only": "B6" if b6_row.get("chunk_gate_pass") and not b7_promotion_pass else None,
        "b6_full_dev_identity_quality_pass": b6_full_dev_pass,
        "b6_chunk_gate_pass": bool(b6_row.get("chunk_gate_pass")),
        "b7_promotion_pass": b7_promotion_pass,
        "b7_status": b7_row.get("status"),
        "reason": (
            "Phase9 full-dev evidence is incomplete. B7 is NO_GO in current evidence; "
            "B6 has only scene0050 32-frame chunk evidence, not full-dev evidence across required scenes."
        ),
    }
    summary = {
        "schema_version": "stream4d_v106_phase9_full_dev_summary_v1",
        "status": "NO_GO_INCOMPLETE_FULL_DEV_EVIDENCE",
        "passes": False,
        "config": {
            "run": asdict(config.run),
            "data": asdict(config.data),
            "phase9": asdict(phase_config),
        },
        "required_scenes": required_scenes,
        "required_variants": required_variants,
        "requirement_evidence_records_json": _rel(repo_root, output_dir / "requirement_evidence_records.json"),
        "missing_evidence_records_json": _rel(repo_root, output_dir / "missing_evidence_records.json"),
        "variant_comparison_table_json": _rel(repo_root, output_dir / "variant_comparison_table.json"),
        "latency_memory_table_json": _rel(repo_root, output_dir / "latency_memory_table.json"),
        "video_index_records_json": _rel(repo_root, output_dir / "video_index_records.json"),
        "preflight_chunk_records_json": _rel(repo_root, output_dir / "preflight_chunk_records.json"),
        "dev_chain_records_json": _rel(repo_root, output_dir / "dev_chain_records.json"),
        "coverage_diagnostic_records_json": _rel(repo_root, output_dir / "coverage_diagnostic_records.json"),
        "full_dev_execution_manifest_json": _rel(repo_root, output_dir / "full_dev_execution_manifest.json"),
        "freeze_decision_json": _rel(repo_root, output_dir / "freeze_decision.json"),
        "full_dev_artifact_present": has_full_dev_artifact,
        "missing_evidence_count": len(missing_evidence),
        "dev_chain_evidence": {
            "record_count": len(dev_chain_records),
            "present_record_count": len(present_dev_chain_records),
            "all_present_records_pass": dev_chain_all_present_records_pass,
            "records_have_at_least_two_boundaries": dev_chain_records_have_three_chunks,
            "required_scene_coverage": dev_chain_required_scene_coverage,
            "required_scenes_covered": dev_chain_required_scenes_covered,
            "scope_complete": dev_chain_aggregate_scope_complete,
            "scope_note": "Aggregate covers the three-consecutive-chunk chain requirement only; full-dev evidence requirements are evaluated separately.",
            "scenes": dev_chain_scenes,
        },
        "coverage_diagnostic_evidence": {
            "record_count": len(coverage_diagnostic_records),
            "all_present": coverage_diagnostics_present,
            "all_pass": coverage_diagnostics_pass,
            "records": coverage_diagnostic_records,
        },
        "scene_path_evidence": scene_path_evidence,
        "notes": [
            "This phase is an evidence audit, not a substitute for a real cold full-dev run.",
            "No missing metric is synthesized. Missing full-dev rows remain blocker evidence.",
        ],
    }
    latency_rows = [
        {
            "variant": row.get("variant"),
            "scope": row.get("scope"),
            "total_runtime_sec": row.get("total_runtime_sec"),
            "tracking_runtime_sec": row.get("tracking_runtime_sec"),
            "peak_cuda_memory_mb": row.get("peak_cuda_memory_mb"),
            "full_dev_evidence": row.get("full_dev_evidence"),
            "evidence_path": row.get("evidence_path"),
        }
        for row in variant_rows
    ]

    write_json(output_dir / "requirement_evidence_records.json", requirements)
    write_json(output_dir / "missing_evidence_records.json", missing_evidence)
    write_json(output_dir / "variant_comparison_table.json", variant_rows)
    write_json(output_dir / "latency_memory_table.json", latency_rows)
    write_json(output_dir / "video_index_records.json", video_rows)
    write_json(output_dir / "preflight_chunk_records.json", preflight_records)
    write_json(output_dir / "dev_chain_records.json", dev_chain_records)
    write_json(output_dir / "coverage_diagnostic_records.json", coverage_diagnostic_records)
    write_json(output_dir / "full_dev_execution_manifest.json", execution_manifest)
    write_json(output_dir / "freeze_decision.json", freeze_decision)
    write_json(output_dir / "input_file_records.json", {
        "phase2_b4_summary": _path_record(repo_root, phase_config.phase2_b4_summary),
        "phase5_b6_replay_summary": _path_record(repo_root, phase_config.phase5_b6_replay_summary),
        "phase5_b6_gate_records": _path_record(repo_root, phase_config.phase5_b6_gate_records),
        "phase7_final_decision": _path_record(repo_root, phase_config.phase7_final_decision),
        "phase8_gate_summary": _path_record(repo_root, phase_config.phase8_gate_summary),
    })
    write_json(output_dir / "phase9_full_dev_summary.json", summary)
    return summary
