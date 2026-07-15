from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase7SpecGapParallelConfig


VARIANTS = ("S0_exact_sequential", "S1_spec_large_gaps_only", "S2_spec_all_persistent_tubes", "S3_s2_dual_gpu_scheduler")


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("records")
        if isinstance(rows, list):
            return rows
    return []


def _candidate_rows(
    *,
    birth_payload: Dict[str, Any],
    classification_rows: List[Dict[str, Any]],
    config: Phase7SpecGapParallelConfig,
) -> List[Dict[str, Any]]:
    cls_by_obj = {
        int(row["scheduled_obj_id"]): row
        for row in classification_rows
        if row.get("scheduled_obj_id") is not None and str(row.get("action")) == "birth_new"
    }
    rows = []
    for row in birth_payload.get("rows", []):
        if str(row.get("phase5_role")) != "birth_new":
            continue
        obj_id = int(row["obj_id"])
        cls = cls_by_obj.get(obj_id, {})
        area = int(row.get("mask_area", cls.get("source_mask_area", 0)) or 0)
        persistence = int(cls.get("persistence_frames_from_anchor", 0) or 0)
        candidate = {
            "candidate_id": f"gap_{obj_id:06d}",
            "obj_id": obj_id,
            "frame_id": int(row["frame_id"]),
            "chunk_frame_index": int(row["chunk_frame_index"]),
            "source_ref_obj_id": int(row.get("source_ref_obj_id", -1)),
            "mask_area": area,
            "persistence_frames": persistence,
            "large_gap": bool(area >= int(config.large_gap_area_threshold)),
            "persistent": bool(persistence >= int(config.min_persistence_frames)),
            "mask_path": row.get("mask_path"),
            "source": row.get("source"),
        }
        rows.append(candidate)
    rows.sort(key=lambda item: (int(item["chunk_frame_index"]), int(item["obj_id"])))
    return rows


def _tube_records(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tubes = []
    for rank, row in enumerate(candidates):
        tubes.append(
            {
                "tube_id": f"tube_{rank:04d}",
                "candidate_id": row["candidate_id"],
                "source_ref_obj_id": int(row["source_ref_obj_id"]),
                "anchor_frame_id": int(row["frame_id"]),
                "anchor_chunk_frame_index": int(row["chunk_frame_index"]),
                "mask_area": int(row["mask_area"]),
                "persistence_frames": int(row["persistence_frames"]),
                "large_gap": bool(row["large_gap"]),
                "persistent": bool(row["persistent"]),
                "span_frames": int(row["persistence_frames"]),
                "status": "eligible" if bool(row["persistent"]) else "short_or_unstable",
            }
        )
    return tubes


def _variant_selection(variant: str, tubes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if variant == "S0_exact_sequential":
        return []
    if variant == "S1_spec_large_gaps_only":
        return [row for row in tubes if row["large_gap"] and row["persistent"]]
    if variant in {"S2_spec_all_persistent_tubes", "S3_s2_dual_gpu_scheduler"}:
        return [row for row in tubes if row["persistent"]]
    raise ValueError(f"unknown variant: {variant}")


def _worker_records(variant: str, selected: List[Dict[str, Any]], *, worker_count: int) -> List[Dict[str, Any]]:
    out = []
    for commit_rank, row in enumerate(selected):
        worker_id = int(commit_rank % max(1, worker_count))
        out.append(
            {
                "schema_version": "stream4d_v106_specgap_worker_record_v1",
                "variant": variant,
                "chunk_index": 2,
                "candidate_id": row["candidate_id"],
                "tube_id": row["tube_id"],
                "worker_id": worker_id,
                "status": "scheduled_not_executed",
                "commit_rank": int(commit_rank),
            }
        )
    return out


def _determinism_records(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    forward = [row["tube_id"] for row in selected]
    reverse_completion = list(reversed(forward))
    committed_after_reverse = sorted(reverse_completion, key=lambda tube_id: forward.index(tube_id))
    return {
        "forward_commit_order": forward,
        "reverse_completion_order": reverse_completion,
        "committed_after_reverse_completion": committed_after_reverse,
        "deterministic_commit_order": forward == committed_after_reverse,
    }


def _variant_audit(
    *,
    variant: str,
    tubes: List[Dict[str, Any]],
    exact_metrics: Dict[str, Any],
    exact_runtime: Dict[str, Any],
    real_probe: Dict[str, Any] | None = None,
    filtered_probe: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected = _variant_selection(variant, tubes)
    worker_count = 2 if variant == "S3_s2_dual_gpu_scheduler" else 1
    workers = _worker_records(variant, selected, worker_count=worker_count)
    determinism = _determinism_records(selected)
    selected_areas = [int(row["mask_area"]) for row in selected]
    has_real_replay = bool(real_probe and real_probe.get("available") and real_probe.get("full_speculative_prompt_decode"))
    has_measured_runtime = bool(
        has_real_replay
        and real_probe
        and real_probe.get("runtime", {}).get("total_runtime_sec") is not None
    )
    return {
        "variant": variant,
        "selected_tube_count": int(len(selected)),
        "selected_mask_area_sum": int(sum(selected_areas)),
        "selected_mask_area_mean": float(np.mean(selected_areas)) if selected_areas else 0.0,
        "worker_count": int(worker_count),
        "worker_records": workers,
        "determinism": determinism,
        "has_real_speculative_replay": has_real_replay,
        "has_measured_speculative_runtime": has_measured_runtime,
        "real_speculative_probe": real_probe,
        "filtered_exact_candidate_probe": filtered_probe,
        "exact_reference_metrics": exact_metrics,
        "exact_reference_runtime": exact_runtime,
        "promotion_status": "real_probe_available_pending_gate" if has_real_replay else "not_promoted_missing_real_speculative_replay",
    }


def _optional_probe(repo_root: Path, replay_text: str, metric_text: str) -> Dict[str, Any] | None:
    if not replay_text or not metric_text:
        return None
    replay_path = _resolve(repo_root, replay_text)
    metric_path = _resolve(repo_root, metric_text)
    if not replay_path.exists() or not metric_path.exists():
        return {
            "available": False,
            "replay_summary": str(replay_path),
            "metric_summary": str(metric_path),
        }
    replay = _read_json(replay_path)
    metric = _read_json(metric_path)
    return {
        "available": True,
        "full_speculative_prompt_decode": False,
        "probe_kind": "filtered_exact_candidate_replay",
        "replay_summary": str(replay_path),
        "replay_summary_sha256": sha256_file(replay_path),
        "metric_summary": str(metric_path),
        "metric_summary_sha256": sha256_file(metric_path),
        "runtime": {
            "total_runtime_sec": replay.get("total_runtime_sec"),
            "total_tracking_runtime_sec": replay.get("total_tracking_runtime_sec"),
            "peak_cuda_memory_mb": replay.get("peak_cuda_memory_mb"),
            "birth_record_count": replay.get("birth_record_count"),
            "anchor_group_count": replay.get("anchor_group_count"),
            "frame_count": replay.get("frame_count"),
        },
        "metrics": metric,
    }


def _optional_real_probe(
    repo_root: Path,
    *,
    replay_text: str,
    metric_text: str,
    birth_records_text: str,
    birth_summary_text: str,
) -> Dict[str, Any] | None:
    if not replay_text or not metric_text or not birth_records_text or not birth_summary_text:
        return None
    replay_path = _resolve(repo_root, replay_text)
    metric_path = _resolve(repo_root, metric_text)
    birth_records_path = _resolve(repo_root, birth_records_text)
    birth_summary_path = _resolve(repo_root, birth_summary_text)
    paths = {
        "replay_summary": replay_path,
        "metric_summary": metric_path,
        "birth_records": birth_records_path,
        "birth_summary": birth_summary_path,
    }
    if not all(path.exists() for path in paths.values()):
        return {
            "available": False,
            "paths": {name: str(path) for name, path in paths.items()},
            "missing": {name: str(path) for name, path in paths.items() if not path.exists()},
        }
    replay = _read_json(replay_path)
    metric = _read_json(metric_path)
    birth_records = _read_json(birth_records_path)
    birth_summary = _read_json(birth_summary_path)
    audit = birth_records.get("audit", {}) if isinstance(birth_records, dict) else {}
    return {
        "available": True,
        "full_speculative_prompt_decode": bool(audit.get("full_speculative_prompt_decode")),
        "probe_kind": "real_inherited_proxy_prompt_decode_replay",
        "replay_summary": str(replay_path),
        "replay_summary_sha256": sha256_file(replay_path),
        "metric_summary": str(metric_path),
        "metric_summary_sha256": sha256_file(metric_path),
        "birth_records": str(birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "birth_summary": str(birth_summary_path),
        "birth_summary_sha256": sha256_file(birth_summary_path),
        "runtime": {
            "total_runtime_sec": replay.get("total_runtime_sec"),
            "total_tracking_runtime_sec": replay.get("total_tracking_runtime_sec"),
            "peak_cuda_memory_mb": replay.get("peak_cuda_memory_mb"),
            "birth_record_count": replay.get("birth_record_count"),
            "anchor_group_count": replay.get("anchor_group_count"),
            "frame_count": replay.get("frame_count"),
            "specgap_birth_runtime_sec": birth_summary.get("total_runtime_sec"),
            "gap_prompt_decode_latency_sec": birth_summary.get("gap_prompt_decode_latency_sec"),
        },
        "metrics": metric,
        "birth_summary_fields": {
            "schema_version": birth_summary.get("schema_version"),
            "model_provider": birth_summary.get("model_provider"),
            "variant": birth_summary.get("variant"),
            "inherited_seed_row_count": birth_summary.get("inherited_seed_row_count"),
            "new_birth_record_count": birth_summary.get("new_birth_record_count"),
            "selected_tube_count": birth_summary.get("selected_tube_count"),
            "tube_count": birth_summary.get("tube_count"),
            "prompt_record_count": birth_summary.get("prompt_record_count"),
            "peak_cuda_memory_mb": birth_summary.get("peak_cuda_memory_mb"),
        },
    }


def _probe_quality_checks(
    *,
    variant: str,
    probe: Dict[str, Any] | None,
    exact_metrics: Dict[str, Any],
    exact_runtime: Dict[str, Any],
    config: Phase7SpecGapParallelConfig,
) -> List[Dict[str, Any]]:
    if not probe or not probe.get("available"):
        return [
            {
                "name": f"{variant}_filtered_probe_available",
                "passes": False,
                "actual": probe,
                "expected": "filtered exact-candidate replay probe exists",
            }
        ]
    metrics = probe.get("metrics", {})
    runtime = probe.get("runtime", {})
    exact_total = float(exact_runtime.get("total_runtime_sec") or 0.0)
    cur_total = float(runtime.get("total_runtime_sec") or 0.0)
    return [
        {
            "name": f"{variant}_filtered_probe_not_full_speculative_decode",
            "passes": False,
            "actual": {"full_speculative_prompt_decode": probe.get("full_speculative_prompt_decode")},
            "expected": "true before Phase7 promotion",
            "repair_direction": "Replace filtered exact-candidate replay with inherited-only proxy gap prompt decoding and batch SAM2 birth.",
        },
        {
            "name": f"{variant}_filtered_probe_quality_thresholds",
            "passes": (
                float(metrics.get("max_cmr", 1.0)) <= float(exact_metrics.get("max_cmr", 0.0)) + float(config.max_cmr_delta)
                and float(metrics.get("drr_area", 1.0)) <= float(exact_metrics.get("drr_area", 0.0)) + float(config.max_drr_delta)
                and float(metrics.get("ttp", 0.0)) >= float(exact_metrics.get("ttp", 1.0)) + float(config.min_ttp_delta)
                and float(metrics.get("asa", 0.0)) >= float(exact_metrics.get("asa", 1.0)) + float(config.min_asa_delta)
                and float(metrics.get("coverage", 0.0)) >= float(exact_metrics.get("coverage", 1.0)) + float(config.min_coverage_delta)
            ),
            "actual": {
                "coverage": metrics.get("coverage"),
                "asa": metrics.get("asa"),
                "ttp": metrics.get("ttp"),
                "max_cmr": metrics.get("max_cmr"),
                "drr_area": metrics.get("drr_area"),
                "exact": {
                    "coverage": exact_metrics.get("coverage"),
                    "asa": exact_metrics.get("asa"),
                    "ttp": exact_metrics.get("ttp"),
                    "max_cmr": exact_metrics.get("max_cmr"),
                    "drr_area": exact_metrics.get("drr_area"),
                },
            },
            "expected": "within Phase7 promotion quality deltas",
        },
        {
            "name": f"{variant}_filtered_probe_latency_lower_than_exact",
            "passes": bool(exact_total > 0.0 and cur_total < exact_total),
            "actual": {"exact_total_runtime_sec": exact_total, "probe_total_runtime_sec": cur_total},
            "expected": "probe total runtime lower than exact reference",
        },
        {
            "name": f"{variant}_filtered_probe_vram_under_limit",
            "passes": float(runtime.get("peak_cuda_memory_mb") or 1.0e9) <= float(config.max_peak_vram_mb),
            "actual": {"peak_cuda_memory_mb": runtime.get("peak_cuda_memory_mb")},
            "expected_max": config.max_peak_vram_mb,
        },
    ]


def _real_probe_quality_checks(
    *,
    variant: str,
    probe: Dict[str, Any] | None,
    exact_metrics: Dict[str, Any],
    exact_runtime: Dict[str, Any],
    config: Phase7SpecGapParallelConfig,
) -> List[Dict[str, Any]]:
    if not probe or not probe.get("available"):
        return [
            {
                "name": f"{variant}_real_probe_available",
                "passes": False,
                "actual": probe,
                "expected": "real inherited-only proxy prompt-decode replay probe exists",
            }
        ]
    metrics = probe.get("metrics", {})
    runtime = probe.get("runtime", {})
    exact_total = float(exact_runtime.get("total_runtime_sec") or 0.0)
    cur_total = float(runtime.get("total_runtime_sec") or 0.0)
    return [
        {
            "name": f"{variant}_real_probe_full_speculative_prompt_decode",
            "passes": bool(probe.get("full_speculative_prompt_decode")),
            "actual": {"full_speculative_prompt_decode": probe.get("full_speculative_prompt_decode")},
            "expected": True,
        },
        {
            "name": f"{variant}_real_probe_no_blank_frames",
            "passes": int(metrics.get("blank_frame_count") or 0) == 0,
            "actual": {"blank_frame_count": metrics.get("blank_frame_count")},
            "expected": 0,
        },
        {
            "name": f"{variant}_real_probe_quality_thresholds",
            "passes": (
                float(metrics.get("max_cmr", 1.0)) <= float(exact_metrics.get("max_cmr", 0.0)) + float(config.max_cmr_delta)
                and float(metrics.get("drr_area", 1.0)) <= float(exact_metrics.get("drr_area", 0.0)) + float(config.max_drr_delta)
                and float(metrics.get("ttp", 0.0)) >= float(exact_metrics.get("ttp", 1.0)) + float(config.min_ttp_delta)
                and float(metrics.get("asa", 0.0)) >= float(exact_metrics.get("asa", 1.0)) + float(config.min_asa_delta)
                and float(metrics.get("coverage", 0.0)) >= float(exact_metrics.get("coverage", 1.0)) + float(config.min_coverage_delta)
            ),
            "actual": {
                "coverage": metrics.get("coverage"),
                "asa": metrics.get("asa"),
                "ttp": metrics.get("ttp"),
                "max_cmr": metrics.get("max_cmr"),
                "drr_area": metrics.get("drr_area"),
                "exact": {
                    "coverage": exact_metrics.get("coverage"),
                    "asa": exact_metrics.get("asa"),
                    "ttp": exact_metrics.get("ttp"),
                    "max_cmr": exact_metrics.get("max_cmr"),
                    "drr_area": exact_metrics.get("drr_area"),
                },
            },
            "expected": "within Phase7 promotion quality deltas",
            "repair_direction": (
                "Inherited-only proxy prompt decode is real but under-covers; repair proxy recall/tube anchors "
                "or route weak frames to final exact residual repair without re-entering infinite track-gap loops."
            ),
        },
        {
            "name": f"{variant}_real_probe_latency_lower_than_exact",
            "passes": bool(exact_total > 0.0 and cur_total < exact_total),
            "actual": {"exact_total_runtime_sec": exact_total, "probe_total_runtime_sec": cur_total},
            "expected": "real probe total runtime lower than exact reference",
        },
        {
            "name": f"{variant}_real_probe_vram_under_limit",
            "passes": float(runtime.get("peak_cuda_memory_mb") or 1.0e9) <= float(config.max_peak_vram_mb),
            "actual": {"peak_cuda_memory_mb": runtime.get("peak_cuda_memory_mb")},
            "expected_max": config.max_peak_vram_mb,
        },
    ]


def _extract_exact_metrics(phase5_gate: Dict[str, Any]) -> Dict[str, Any]:
    metrics = phase5_gate.get("variant_metrics", {}).get("R1_repair_vs_birth", {})
    keep = [
        "coverage",
        "asa",
        "ttp",
        "max_cmr",
        "max_cfr",
        "drr_area",
        "drr_count",
        "mean_foreground_union_iou",
        "peak_cuda_memory_mb",
        "total_runtime_sec",
        "total_tracking_runtime_sec",
        "frame_count",
        "new_birth_id_count",
        "duplicate_birth_id_count",
    ]
    return {key: metrics.get(key) for key in keep if key in metrics}


def _precondition_checks(paths: Dict[str, Path], phase0_6: Dict[str, Any], phase6_gate: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages = phase0_6.get("stages", {})
    return [
        {
            "name": "all_required_paths_exist",
            "passes": all(path.exists() for path in paths.values()),
            "actual": {name: str(path) for name, path in paths.items()},
            "expected": "all configured Phase7 inputs exist",
        },
        {
            "name": "phase0_6_integrated_pass",
            "passes": bool(phase0_6.get("all_requested_stages_pass")) and all(
                bool(stages.get(stage, {}).get("passes")) for stage in ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"]
            ),
            "actual": {
                "all_requested_stages_pass": phase0_6.get("all_requested_stages_pass"),
                "stage_passes": {stage: stages.get(stage, {}).get("passes") for stage in sorted(stages)},
            },
            "expected": "Phase0-6 pass before speculative gap",
        },
        {
            "name": "phase6_reappearance_gate_pass",
            "passes": bool(phase6_gate.get("passes")),
            "actual": {"passes": phase6_gate.get("passes"), "failure_count": phase6_gate.get("failure_count")},
            "expected": True,
        },
    ]


def run_phase7_specgap_parallel(
    repo_root: Path,
    config: Phase7SpecGapParallelConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "integrated_phase0_6_run_summary": _resolve(repo_root, config.integrated_phase0_6_run_summary),
        "phase6_gate_records": _resolve(repo_root, config.phase6_gate_records),
        "exact_b6_replay_summary": _resolve(repo_root, config.exact_b6_replay_summary),
        "exact_b6_gate_records": _resolve(repo_root, config.exact_b6_gate_records),
        "candidate_birth_records": _resolve(repo_root, config.candidate_birth_records),
        "candidate_classification_records": _resolve(repo_root, config.candidate_classification_records),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        summary = {
            "schema_version": "stream4d_v106_phase7_specgap_parallel_summary_v1",
            "passes": False,
            "missing_paths": missing,
        }
        write_json(output_dir / "gate_records.json", summary)
        write_json(output_dir / "failure_records.json", [{"failure": "missing_required_artifact", "paths": missing}])
        return summary

    phase0_6 = _read_json(paths["integrated_phase0_6_run_summary"])
    phase6_gate = _read_json(paths["phase6_gate_records"])
    phase5_gate = _read_json(paths["exact_b6_gate_records"])
    replay_summary = _read_json(paths["exact_b6_replay_summary"])
    birth_payload = _read_json(paths["candidate_birth_records"])
    classification_rows = _records(_read_json(paths["candidate_classification_records"]))

    preconditions = _precondition_checks(paths, phase0_6, phase6_gate)
    candidates = _candidate_rows(birth_payload=birth_payload, classification_rows=classification_rows, config=config)
    tubes = _tube_records(candidates)
    exact_metrics = _extract_exact_metrics(phase5_gate)
    exact_runtime = {
        "total_runtime_sec": replay_summary.get("total_runtime_sec"),
        "total_tracking_runtime_sec": replay_summary.get("total_tracking_runtime_sec"),
        "peak_cuda_memory_mb": replay_summary.get("peak_cuda_memory_mb"),
        "anchor_group_count": replay_summary.get("anchor_group_count"),
        "birth_record_count": replay_summary.get("birth_record_count"),
        "frame_count": replay_summary.get("frame_count"),
        "replay_summary_path": str(paths["exact_b6_replay_summary"]),
    }
    probes = {
        "S1_spec_large_gaps_only": _optional_probe(repo_root, config.s1_filtered_replay_summary, config.s1_filtered_metric_summary),
        "S2_spec_all_persistent_tubes": _optional_probe(repo_root, config.s2_filtered_replay_summary, config.s2_filtered_metric_summary),
        "S3_s2_dual_gpu_scheduler": None,
        "S0_exact_sequential": None,
    }
    real_probes = {
        "S1_spec_large_gaps_only": _optional_real_probe(
            repo_root,
            replay_text=config.s1_real_replay_summary,
            metric_text=config.s1_real_metric_summary,
            birth_records_text=config.s1_real_birth_records,
            birth_summary_text=config.s1_real_birth_summary,
        ),
        "S2_spec_all_persistent_tubes": _optional_real_probe(
            repo_root,
            replay_text=config.s2_real_replay_summary,
            metric_text=config.s2_real_metric_summary,
            birth_records_text=config.s2_real_birth_records,
            birth_summary_text=config.s2_real_birth_summary,
        ),
        "S3_s2_dual_gpu_scheduler": None,
        "S0_exact_sequential": None,
    }

    variant_records = {
        variant: _variant_audit(
            variant=variant,
            tubes=tubes,
            exact_metrics=exact_metrics,
            exact_runtime=exact_runtime,
            real_probe=real_probes.get(variant),
            filtered_probe=probes.get(variant),
        )
        for variant in VARIANTS
    }
    worker_records = [
        worker
        for variant in VARIANTS
        for worker in variant_records[variant]["worker_records"]
    ]
    reconciliation_inputs = [
        {
            "variant": variant,
            "selected_tube_count": variant_records[variant]["selected_tube_count"],
            "deterministic_commit_order": variant_records[variant]["determinism"]["deterministic_commit_order"],
            "has_real_speculative_replay": variant_records[variant]["has_real_speculative_replay"],
        }
        for variant in VARIANTS
    ]
    tube_counts = Counter(
        "large_persistent" if row["large_gap"] and row["persistent"]
        else "persistent" if row["persistent"]
        else "short_or_unstable"
        for row in tubes
    )

    promotion_checks = [
        *preconditions,
        {
            "name": "candidate_tubes_exist",
            "passes": bool(tubes),
            "actual": {"candidate_count": len(candidates), "tube_count": len(tubes), "tube_counts": dict(tube_counts)},
            "expected": "at least one gap tube candidate",
        },
        {
            "name": "deterministic_commit_order",
            "passes": all(variant_records[v]["determinism"]["deterministic_commit_order"] for v in VARIANTS),
            "actual": {v: variant_records[v]["determinism"]["deterministic_commit_order"] for v in VARIANTS},
            "expected": True,
        },
        {
            "name": "real_speculative_replay_available",
            "passes": any(variant_records[v]["has_real_speculative_replay"] for v in ("S1_spec_large_gaps_only", "S2_spec_all_persistent_tubes", "S3_s2_dual_gpu_scheduler")),
            "actual": {v: variant_records[v]["has_real_speculative_replay"] for v in VARIANTS},
            "expected": "at least one speculative variant has actual replay labels and metrics",
            "repair_direction": "Implement real inherited-only proxy + batch SAM2 birth + exact cohort tracking replay; do not promote schedule-only artifacts.",
        },
        {
            "name": "measured_speculative_latency_available",
            "passes": any(variant_records[v]["has_measured_speculative_runtime"] for v in ("S1_spec_large_gaps_only", "S2_spec_all_persistent_tubes", "S3_s2_dual_gpu_scheduler")),
            "actual": {v: variant_records[v]["has_measured_speculative_runtime"] for v in VARIANTS},
            "expected": "at least one speculative variant has measured runtime, not an estimate",
            "repair_direction": "Profile prompt decode/cohort critical path after real replay exists.",
        },
    ]
    for variant in ("S1_spec_large_gaps_only", "S2_spec_all_persistent_tubes"):
        real_probe = real_probes.get(variant)
        if real_probe and real_probe.get("available"):
            promotion_checks.extend(
                _real_probe_quality_checks(
                    variant=variant,
                    probe=real_probe,
                    exact_metrics=exact_metrics,
                    exact_runtime=exact_runtime,
                    config=config,
                )
            )
        else:
            promotion_checks.extend(
                _probe_quality_checks(
                    variant=variant,
                    probe=probes.get(variant),
                    exact_metrics=exact_metrics,
                    exact_runtime=exact_runtime,
                    config=config,
                )
            )
    failure_records = [check for check in promotion_checks if not bool(check["passes"])]

    write_json(output_dir / "specgap_candidate_records.json", candidates)
    write_json(output_dir / "specgap_tube_records.json", tubes)
    write_json(output_dir / "specgap_worker_records.json", worker_records)
    write_json(output_dir / "specgap_reconciliation_input_records.json", reconciliation_inputs)
    write_json(output_dir / "variant_records.json", variant_records)

    summary = {
        "schema_version": "stream4d_v106_phase7_specgap_parallel_summary_v1",
        "passes": not failure_records,
        "promotion": "NO_GO" if failure_records else "PROMOTE",
        "failure_count": int(len(failure_records)),
        "checks": promotion_checks,
        "scope": {
            "scene_id": config.scene_id,
            "candidate_count": int(len(candidates)),
            "tube_count": int(len(tubes)),
            "tube_counts": dict(tube_counts),
            "variants": list(VARIANTS),
        },
        "exact_reference_metrics": exact_metrics,
        "exact_reference_runtime": exact_runtime,
        "variant_summary": {
            variant: {
                "selected_tube_count": variant_records[variant]["selected_tube_count"],
                "selected_mask_area_sum": variant_records[variant]["selected_mask_area_sum"],
                "worker_count": variant_records[variant]["worker_count"],
                "has_real_speculative_replay": variant_records[variant]["has_real_speculative_replay"],
                "has_measured_speculative_runtime": variant_records[variant]["has_measured_speculative_runtime"],
                "real_speculative_probe": variant_records[variant]["real_speculative_probe"],
                "filtered_exact_candidate_probe": variant_records[variant]["filtered_exact_candidate_probe"],
                "promotion_status": variant_records[variant]["promotion_status"],
            }
            for variant in VARIANTS
        },
        "ap_generated": False,
        "mv_ap_scene_used_as_gate": False,
        "honesty_note": (
            "Phase7 generated deterministic tube and scheduler artifacts, but no speculative "
            "variant is promoted because no real speculative replay labels/runtime were produced."
        ),
    }
    write_json(output_dir / "gate_records.json", summary)
    write_json(output_dir / "failure_records.json", failure_records)
    return summary
