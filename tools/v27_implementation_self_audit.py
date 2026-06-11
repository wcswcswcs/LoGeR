#!/usr/bin/env python3
"""ACL2 v27 causal semantic role router hard-gate audit.

The audit is intentionally strict: it checks that predicted video-masklet fine
labels reach runtime as token labels and that path-specific role streams are
present.  It never treats predicted semantics as GT semantics and never starts
candidate rollouts itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


PHASE0_CANDIDATES = [
    "V27_P0_00_H9_REFERENCE",
    "V27_P0_01_CAUSAL_LOADED_BUT_IGNORED",
    "V27_P0_02_CAUSAL_PASS_THROUGH_CONSUMED",
    "V27_P0_03_CAUSAL_DEBUG_ONLY_ALL_PATHS",
    "V27_P0_04_CAUSAL_FRAME_GLOBAL_SMOKE",
    "V27_P0_05_CAUSAL_SWA_SMOKE",
    "V27_P0_06_CAUSAL_TTT_SMOKE",
]


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    text = (text or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _selected_cache_chunks(chunks: Sequence[int], horizons: Sequence[int]) -> Set[int]:
    needed: Set[int] = set()
    for chunk in chunks:
        for horizon in horizons:
            for idx in range(int(chunk), int(chunk) + int(horizon) + 1):
                needed.add(idx)
    return needed


def _load_cache_index(cache_dir: Path) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for row in _read_jsonl(cache_dir / "cache_index.jsonl"):
        try:
            out[int(row.get("chunk_idx"))] = row  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return out


def _load_semantic_rows(cache_dir: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in _read_csv(cache_dir / "semantic_audit" / "per_chunk_semantic.csv"):
        try:
            out[int(row.get("chunk_idx", ""))] = row
        except ValueError:
            continue
    return out


def _load_fine_labels(cache_dir: Path, chunks: Set[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in _read_csv(cache_dir / "semantic_audit" / "label_counts_by_chunk.csv"):
        try:
            idx = int(row.get("chunk_idx", ""))
        except ValueError:
            continue
        if chunks and idx not in chunks:
            continue
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        rows.append({
            "chunk_idx": idx,
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "fine_label": label,
            "count": row.get("count", "0"),
            "semantic_source": "video_masklet_frontend_cache",
            "uses_gt_semantic": False,
        })
    return rows


def _read_pose_file(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(part) for part in line.split()])
    return rows


def _pose_diff_metrics(candidate_path: Path, reference_path: Path) -> Dict[str, object]:
    candidate = _read_pose_file(candidate_path)
    reference = _read_pose_file(reference_path)
    if not candidate or not reference:
        return {
            "pose_compare_available": False,
            "matched_rows": 0,
            "max_translation_abs_diff": float("inf"),
            "max_pose_abs_diff": float("inf"),
            "shape_mismatch": bool(len(candidate) != len(reference)),
        }
    matched = min(len(candidate), len(reference))
    max_translation = 0.0
    max_pose = 0.0
    for cand_row, ref_row in zip(candidate[:matched], reference[:matched]):
        width = min(len(cand_row), len(ref_row))
        for col in range(1, min(width, 4)):
            max_translation = max(max_translation, abs(cand_row[col] - ref_row[col]))
        for col in range(1, min(width, 8)):
            max_pose = max(max_pose, abs(cand_row[col] - ref_row[col]))
    return {
        "pose_compare_available": True,
        "matched_rows": matched,
        "max_translation_abs_diff": max_translation,
        "max_pose_abs_diff": max_pose,
        "shape_mismatch": bool(len(candidate) != len(reference)),
    }


def _phase0_run_dir(results_root: Path, prefix: str, candidate: str, chunk: int, horizon: int) -> Path:
    name = f"{prefix}_{candidate}_chunk{chunk}_h{horizon}_globalgate_H9parent_SWKS3"
    return results_root / "rollouts" / name


def _phase0_noop_gate(results_root: Path, prefix: str, chunk: int, horizon: int) -> Dict[str, object]:
    reference = _phase0_run_dir(results_root, prefix, "V27_P0_00_H9_REFERENCE", chunk, horizon) / "01.txt"
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for candidate in PHASE0_CANDIDATES[:4]:
        pose_path = _phase0_run_dir(results_root, prefix, candidate, chunk, horizon) / "01.txt"
        metrics = _pose_diff_metrics(pose_path, reference)
        row = {
            "candidate_id": candidate,
            "reference_candidate_id": "V27_P0_00_H9_REFERENCE",
            "candidate_pose_path": str(pose_path),
            "reference_pose_path": str(reference),
            **metrics,
        }
        rows.append(row)
        if (
            not metrics["pose_compare_available"]
            or metrics["shape_mismatch"]
            or metrics["max_translation_abs_diff"] != 0.0
            or metrics["max_pose_abs_diff"] != 0.0
        ):
            failures.append(row)
    return {
        "noop_parity_gate_pass": bool(len(rows) == 4 and not failures),
        "noop_rows_checked": len(rows),
        "noop_failures": failures,
        "noop_rows": rows,
    }


def _scan_runtime(results_root: Path, prefix: str, chunk: int, horizon: int) -> Dict[str, object]:
    fine_projection_rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []
    context_rows: List[Dict[str, object]] = []
    runtime_fine = False
    path_specific = False
    path_flags = {"frame": False, "global": False, "swa": False, "ttt": False}
    role_stream_nonempty = {"R_frame": False, "R_global": False, "R_swa": False, "R_ttt": False}
    empty_source_events = 0
    conflict_available = False
    scale_available = False
    condition_levels: List[Dict[str, object]] = []

    for candidate in PHASE0_CANDIDATES:
        run_dir = _phase0_run_dir(results_root, prefix, candidate, chunk, horizon)
        for row in _read_jsonl(run_dir / "semantic_group_summary.jsonl"):
            counts = row.get("fine_label_name_counts", {})
            if isinstance(counts, dict):
                for label, count in counts.items():
                    fine_projection_rows.append({
                        "candidate_id": candidate,
                        "chunk_idx": row.get("chunk_idx"),
                        "start_frame": row.get("start_frame"),
                        "end_frame": row.get("end_frame"),
                        "fine_label": label,
                        "token_count": count,
                    })
        for row in _read_jsonl(run_dir / "semantic_role_summary.jsonl"):
            path_role_counts = row.get("path_role_counts", {})
            if isinstance(path_role_counts, dict):
                for stream_name, counts in path_role_counts.items():
                    nonempty = bool(counts)
                    if stream_name == "R_frame_tok":
                        role_stream_nonempty["R_frame"] |= nonempty
                    elif stream_name == "R_global_tok":
                        role_stream_nonempty["R_global"] |= nonempty
                    elif stream_name == "R_swa_tok":
                        role_stream_nonempty["R_swa"] |= nonempty
                    elif stream_name == "R_ttt_tok":
                        role_stream_nonempty["R_ttt"] |= nonempty
                    path_rows.append({
                        "candidate_id": candidate,
                        "chunk_idx": row.get("chunk_idx"),
                        "role_stream": stream_name,
                        "role_counts": json.dumps(counts, sort_keys=True),
                    })
            path_specific |= bool(row.get("path_specific_role_streams_available", False))
        for row in _read_jsonl(run_dir / "semantic_memory_path_summary.jsonl"):
            runtime_fine |= bool(row.get("runtime_fine_role_policy_available", False))
            path_specific |= bool(row.get("path_specific_role_streams_available", False))
            path_flags["frame"] |= bool(row.get("frame_semantic_source_consumed", False))
            path_flags["global"] |= bool(row.get("chunk_global_semantic_source_consumed", False))
            path_flags["swa"] |= bool(row.get("swa_semantic_source_consumed", False))
            path_flags["ttt"] |= bool(row.get("ttt_semantic_role_consumed", False))
            conflict_available |= bool(row.get("condition_signal_conflict_available", False))
            scale_available |= bool(row.get("condition_signal_scale_risk_available", False))
            condition_levels.append({
                "candidate_id": candidate,
                "chunk_idx": row.get("chunk_idx"),
                "condition_signal_conflict_available": row.get("condition_signal_conflict_available"),
                "condition_signal_scale_risk_available": row.get("condition_signal_scale_risk_available"),
                "condition_signal_conflict_level": row.get("condition_signal_conflict_level"),
                "condition_signal_scale_risk_level": row.get("condition_signal_scale_risk_level"),
                "condition_signal_conflict_source": row.get("condition_signal_conflict_source"),
                "condition_signal_scale_risk_source": row.get("condition_signal_scale_risk_source"),
                "condition_signal_conflict_value": row.get("condition_signal_conflict_value"),
                "condition_signal_scale_risk_value": row.get("condition_signal_scale_risk_value"),
            })
        for row in _read_jsonl(run_dir / "context_skip_summary.jsonl"):
            context_rows.append({"candidate_id": candidate, **row})
            empty_source_events += int(row.get("num_context_empty_source_events", 0) or 0)

    return {
        "fine_projection_rows": fine_projection_rows,
        "path_rows": path_rows,
        "context_rows": context_rows,
        "runtime_fine_role_policy_available": runtime_fine,
        "path_specific_role_streams_available": path_specific,
        "role_stream_nonempty": role_stream_nonempty,
        "path_consumption_flags": path_flags,
        "context_empty_source_events": empty_source_events,
        "condition_signal_conflict_available": conflict_available,
        "condition_signal_scale_risk_available": scale_available,
        "condition_levels": condition_levels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v27_semanticprior_causalrolerouter_allmemory_parallel")
    parser.add_argument("--stage-c-cache-dir", default="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    parser.add_argument("--phase0-prefix", default="V27_P0_SMOKE_R1")
    parser.add_argument("--phase0-chunk", type=int, default=10)
    parser.add_argument("--phase0-horizon", type=int, default=3)
    args = parser.parse_args()

    results = Path(args.results_root).resolve()
    out_dir = results / "implementation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.stage_c_cache_dir).resolve()
    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    needed_chunks = _selected_cache_chunks(chunks, horizons)
    cache_index = _load_cache_index(cache_dir)
    semantic_rows = _load_semantic_rows(cache_dir)

    cache_rows: List[Dict[str, object]] = []
    coverage_values: List[float] = []
    visible_values: List[float] = []
    masklet_counts: List[int] = []
    hits = 0
    for idx in sorted(needed_chunks):
        index_row = cache_index.get(idx)
        sem = semantic_rows.get(idx, {})
        masklet_path = ""
        manifest_path = ""
        file_hit = False
        if index_row is not None:
            chunk_name = str(index_row.get("chunk"))
            masklet_path = str(cache_dir / chunk_name / "masklet.pt")
            manifest_path = str(cache_dir / chunk_name / "manifest.json")
            file_hit = Path(masklet_path).exists() and Path(manifest_path).exists()
        if file_hit:
            hits += 1
        coverage = _safe_float(sem.get("coverage_mean"))
        visible = _safe_float(sem.get("visible_masklet_frame_frac"))
        n_masklets = int(float(sem.get("num_masklets", index_row.get("num_masklets", 0) if index_row else 0) or 0))
        if math.isfinite(coverage):
            coverage_values.append(coverage)
        if math.isfinite(visible):
            visible_values.append(visible)
        masklet_counts.append(n_masklets)
        cache_rows.append({
            "chunk_idx": idx,
            "cache_index_hit": index_row is not None,
            "masklet_file_hit": file_hit,
            "masklet_path": masklet_path,
            "manifest_path": manifest_path,
            "num_masklets": n_masklets,
            "coverage_mean": coverage if math.isfinite(coverage) else "",
            "visible_masklet_frame_frac": visible if math.isfinite(visible) else "",
            "masklet_labels": sem.get("masklet_labels", ""),
        })

    focus_weight = 0.0
    focus_cov_weighted = 0.0
    for row in semantic_rows.values():
        overlap = _safe_float(row.get("focus_overlap"))
        cov = _safe_float(row.get("coverage_mean"))
        if math.isfinite(overlap) and math.isfinite(cov) and overlap > 0:
            focus_weight += overlap
            focus_cov_weighted += overlap * cov
    focus_coverage = focus_cov_weighted / focus_weight if focus_weight > 0 else float("nan")

    fine_rows = _load_fine_labels(cache_dir, needed_chunks)
    fine_labels = sorted({str(row["fine_label"]) for row in fine_rows})
    cache_hit_rate = hits / max(1, len(needed_chunks))
    chunks_with_masklets_ratio = sum(1 for c in masklet_counts if c > 0) / max(1, len(masklet_counts))
    mean_coverage = sum(coverage_values) / max(1, len(coverage_values))
    mean_visible = sum(visible_values) / max(1, len(visible_values))
    mean_masklets = sum(masklet_counts) / max(1, len(masklet_counts))
    cache_gate = bool(
        cache_hit_rate >= 0.98
        and chunks_with_masklets_ratio >= 0.95
        and len(fine_labels) >= 6
        and mean_coverage >= 0.80
        and math.isfinite(focus_coverage)
        and focus_coverage >= 0.80
    )

    noop = _phase0_noop_gate(results, args.phase0_prefix, args.phase0_chunk, args.phase0_horizon)
    runtime = _scan_runtime(results, args.phase0_prefix, args.phase0_chunk, args.phase0_horizon)
    role_stream_gate = all(bool(v) for v in runtime["role_stream_nonempty"].values())
    path_consumption_gate = all(bool(v) for v in runtime["path_consumption_flags"].values())
    runtime_gate = bool(
        runtime["runtime_fine_role_policy_available"]
        and runtime["path_specific_role_streams_available"]
        and role_stream_gate
        and path_consumption_gate
        and bool(runtime["condition_signal_conflict_available"])
        and bool(runtime["condition_signal_scale_risk_available"])
        and int(runtime["context_empty_source_events"]) == 0
    )
    all_gate = bool(cache_gate and noop["noop_parity_gate_pass"] and runtime_gate)

    _write_csv(out_dir / "stage_c_cache_hit_audit.csv", cache_rows)
    _write_csv(out_dir / "fine_label_coverage_by_chunk.csv", fine_rows)
    _write_csv(out_dir / "fine_label_token_projection.csv", runtime["fine_projection_rows"])
    _write_csv(out_dir / "fine_label_coverage_by_path.csv", runtime["path_rows"])
    _write_csv(out_dir / "noop_parity_metrics.csv", noop["noop_rows"])
    _write_jsonl(out_dir / "context_skip_summary.jsonl", runtime["context_rows"])
    _write_csv(out_dir / "semantic_role_router_audit.csv", runtime["condition_levels"])
    _write_csv(out_dir / "per_token_condition_summary.csv", runtime["condition_levels"])
    _write_jsonl(out_dir / "path_consumption_summary.jsonl", [{
        "path_consumption_flags": runtime["path_consumption_flags"],
        "role_stream_nonempty": runtime["role_stream_nonempty"],
        "runtime_fine_role_policy_available": runtime["runtime_fine_role_policy_available"],
        "path_specific_role_streams_available": runtime["path_specific_role_streams_available"],
        "context_empty_source_events": runtime["context_empty_source_events"],
    }])
    _write_jsonl(out_dir / "swa_semantic_cache_summary.jsonl", [
        row for row in runtime["path_rows"] if row.get("role_stream") == "R_swa_tok"
    ])
    _write_jsonl(out_dir / "ttt_semantic_write_summary.jsonl", [
        row for row in runtime["path_rows"] if row.get("role_stream") == "R_ttt_tok"
    ])

    failures: List[Dict[str, object]] = []
    if not cache_gate:
        failures.append({"gate": "cache_quality", "failure": "cache_or_fine_label_gate_failed"})
    if not noop["noop_parity_gate_pass"]:
        failures.append({"gate": "noop_parity", "failure": "direct_pose_compare_failed", "detail": noop["noop_failures"]})
    if not runtime_gate:
        failures.append({
            "gate": "runtime_fine_path_roles",
            "failure": "fine_path_role_runtime_gate_failed",
            "detail": {
                "runtime_fine_role_policy_available": runtime["runtime_fine_role_policy_available"],
                "path_specific_role_streams_available": runtime["path_specific_role_streams_available"],
                "role_stream_nonempty": runtime["role_stream_nonempty"],
                "path_consumption_flags": runtime["path_consumption_flags"],
                "condition_signal_conflict_available": runtime["condition_signal_conflict_available"],
                "condition_signal_scale_risk_available": runtime["condition_signal_scale_risk_available"],
                "context_empty_source_events": runtime["context_empty_source_events"],
            },
        })
    _write_jsonl(out_dir / "codex_self_check_failures.jsonl", failures)

    summary: Dict[str, object] = {
        "phase": "v27_phase0_causal_role_router_audit",
        "uses_gt_semantic": False,
        "uses_video_masklet_semantic": True,
        "semantic_source": "video_masklet_frontend_cache",
        "cache_hit_rate": cache_hit_rate,
        "chunks_with_masklets_ratio": chunks_with_masklets_ratio,
        "needed_chunk_count": len(needed_chunks),
        "mean_masklets_per_chunk": mean_masklets,
        "mean_coverage": mean_coverage,
        "mean_visible_masklet_frame_frac": mean_visible,
        "focus_coverage_200_300": focus_coverage,
        "fine_label_count": len(fine_labels),
        "fine_labels": fine_labels,
        "cache_quality_gate_pass": cache_gate,
        **{k: v for k, v in noop.items() if k != "noop_rows"},
        "runtime_fine_role_policy_available": runtime["runtime_fine_role_policy_available"],
        "path_specific_role_streams_available": runtime["path_specific_role_streams_available"],
        "role_stream_nonempty": runtime["role_stream_nonempty"],
        "path_consumption_flags": runtime["path_consumption_flags"],
        "condition_signal_conflict_available": runtime["condition_signal_conflict_available"],
        "condition_signal_scale_risk_available": runtime["condition_signal_scale_risk_available"],
        "context_empty_source_events": runtime["context_empty_source_events"],
        "runtime_gate_pass": runtime_gate,
        "all_gate_pass": all_gate,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
    }
    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# ACL2 v27 Causal Role Router Phase-0 Audit",
        "",
        f"all_gate_pass = `{str(all_gate).lower()}`",
        f"cache_quality_gate_pass = `{str(cache_gate).lower()}`",
        f"runtime_gate_pass = `{str(runtime_gate).lower()}`",
        f"noop_parity_gate_pass = `{str(noop['noop_parity_gate_pass']).lower()}`",
        f"cache_hit_rate = `{cache_hit_rate:.10f}`",
        f"chunks_with_masklets_ratio = `{chunks_with_masklets_ratio:.10f}`",
        f"mean_coverage = `{mean_coverage:.10f}`",
        f"focus_coverage_200_300 = `{focus_coverage:.10f}`",
        f"fine_label_count = `{len(fine_labels)}`",
        f"runtime_fine_role_policy_available = `{str(runtime['runtime_fine_role_policy_available']).lower()}`",
        f"path_specific_role_streams_available = `{str(runtime['path_specific_role_streams_available']).lower()}`",
        f"condition_signal_conflict_available = `{str(runtime['condition_signal_conflict_available']).lower()}`",
        f"condition_signal_scale_risk_available = `{str(runtime['condition_signal_scale_risk_available']).lower()}`",
        "",
        "Predicted video-masklet semantics are used; this is not a GT semantic experiment.",
    ]
    (out_dir / "codex_self_check_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if all_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
