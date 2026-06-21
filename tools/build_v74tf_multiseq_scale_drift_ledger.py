#!/usr/bin/env python3
"""Phase 1 multi-sequence scale-drift ledger for ACL2 v74-TF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_acl2_v67_segments import _chunk_rows
from diagnose_v73_semantic_explains_geometry_cues import _radio_features, _semantic_features
from kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _umeyama_sim3
from v73_semantic_memory_common import TARGET_CHUNKS, load_json, read_csv, safe_float, utc_now, write_csv, write_json
from v74tf_common import (
    GT_ROOT,
    PREPROCESS_ROOT,
    REPORT_ROOT,
    V73_REPORT_ROOT,
    add_j_scale,
    discover_v74tf_prefix_run,
    discover_radio_dirs,
    parse_seqs,
    read_stage_summary,
    stage_cache_dir,
    stage_chunk_dirs,
    summarize_metric_coverage,
    target_chunks_for_seq,
)


JOIN_FEATURE_KEYS = (
    "raw_overlap_residual_rmse",
    "raw_overlap_residual_mean",
    "D_geo_mean_patch",
    "global_k_layer5_gram_motion",
    "global_k_layer7_gram_motion",
    "semantic_available",
    "semantic_confidence_mean",
    "semantic_nonvoid_ratio",
    "stable_structure_ratio",
    "dynamic_thing_ratio",
    "lowtrust_stuff_ratio",
    "road_context_ratio",
    "sky_context_ratio",
    "radio_available",
    "radio_static_mean",
    "radio_dynamic_mean",
    "radio_lowtrust_mean",
    "radio_boundary_mean",
    "radio_interior_mean",
    "radio_temporal_stability_mean",
)


def _load_v73_01_rows(v73_report_root: Path) -> list[dict[str, Any]]:
    ledger_payload = load_json(v73_report_root / "phase1_scale_drift_ledger" / "scale_drift_ledger.json") or {}
    ledger_rows = ledger_payload.get("rows", []) if isinstance(ledger_payload, dict) else []
    feature_rows = read_csv(v73_report_root / "phase3_semantic_explanation" / "semantic_geometry_features_by_chunk.csv")
    features_by_chunk: dict[int, dict[str, Any]] = {}
    for row in feature_rows:
        try:
            features_by_chunk[int(row.get("chunk_id", -1))] = row
        except (TypeError, ValueError):
            continue
    out: list[dict[str, Any]] = []
    for row in ledger_rows:
        try:
            chunk = int(row.get("chunk_id", -1))
        except (TypeError, ValueError):
            continue
        merged = {"seq": "01", **row}
        feats = features_by_chunk.get(chunk, {})
        for key in JOIN_FEATURE_KEYS:
            if key in feats:
                merged[key] = feats.get(key)
        merged["metric_source"] = "v73_h35_trace_ledger_joined_with_semantic_features"
        merged["metric_coverage_status"] = "full_or_partial_real_metrics"
        out.append(merged)
    return out


def _rmse(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def _global_chunk_ate(
    frames: np.ndarray,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    start: int,
    end: int,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> float | None:
    mask = (frames >= int(start)) & (frames < int(end))
    if int(mask.sum()) < 1:
        return None
    eval_frames = frames[mask]
    aligned = (scale * (rot @ pred_pos[mask].T)).T + trans
    errors = np.linalg.norm(aligned - gt_pos[eval_frames], axis=1)
    return _rmse(errors)


def _boundary_translation_delta(frames: np.ndarray, pred_pos: np.ndarray, gt_pos: np.ndarray, start: int) -> float | None:
    if int(start) <= 0:
        return None
    prev_idx = np.flatnonzero(frames == int(start) - 1)
    cur_idx = np.flatnonzero(frames == int(start))
    if prev_idx.size == 0 or cur_idx.size == 0 or int(start) >= gt_pos.shape[0]:
        return None
    pred_step = float(np.linalg.norm(pred_pos[int(cur_idx[0])] - pred_pos[int(prev_idx[0])]))
    gt_step = float(np.linalg.norm(gt_pos[int(start)] - gt_pos[int(start) - 1]))
    delta = abs(pred_step - gt_step)
    return float(delta) if math.isfinite(delta) else None


def _prefix_trajectory_rows(preprocess_root: Path, seq: str, run_dir: Path, max_chunks: int) -> list[dict[str, Any]]:
    seq = str(seq).zfill(2)
    gt_path = GT_ROOT / f"{seq}.txt"
    pred_path = run_dir / f"{seq}.txt"
    _, _, gt_pos = _load_kitti_gt(gt_path)
    frames, _, pred_pos = _load_tum_prediction(pred_path, gt_pos.shape[0])
    valid = (frames >= 0) & (frames < gt_pos.shape[0])
    frames = frames[valid]
    pred_pos = pred_pos[valid]
    if frames.shape[0] < 3:
        return []
    scale, rot, trans = _umeyama_sim3(pred_pos, gt_pos[frames], with_scale=True)
    raw_rows = _chunk_rows(
        f"{seq}_v74tf_prefix",
        frames,
        pred_pos,
        gt_pos,
        chunk_size=32,
        overlap=3,
        head_len=10,
    )
    semantic_stage_dir = stage_cache_dir(preprocess_root, seq)
    radio_dirs = discover_radio_dirs(preprocess_root, seq)
    radio_dir = radio_dirs[0] if radio_dirs else Path("__missing_radio_sidecar__")
    available_chunks = [int(row.get("chunk_idx")) for row in raw_rows if row.get("chunk_idx") is not None]
    selected = set(target_chunks_for_seq(seq, available_chunks, max_chunks=max_chunks))
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        chunk = int(row.get("chunk_idx"))
        start = int(row.get("start"))
        end = int(row.get("end"))
        local_ate = row.get("whole_ate_rmse_m")
        global_ate = _global_chunk_ate(frames, pred_pos, gt_pos, start, end, scale, rot, trans)
        semantic_features = _semantic_features(semantic_stage_dir, chunk)
        radio_features = _radio_features(radio_dir, chunk)
        out.append(
            {
                "seq": seq,
                "chunk_id": chunk,
                "is_target_chunk": chunk in selected,
                "frame_start": start,
                "frame_end": end,
                "local_sim3_ate": local_ate,
                "global_chunk_ate": global_ate,
                "oracle_gap": float(global_ate - local_ate)
                if global_ate is not None and local_ate is not None
                else None,
                "head_to_tail": row.get("head_to_tail_ate_rmse_m"),
                "future_after_overlap": row.get("overlap_to_future_ate_rmse_m"),
                "scale_cv": row.get("scale_cv_head_mid_tail"),
                "boundary_jump": _boundary_translation_delta(frames, pred_pos, gt_pos, start),
                "raw_overlap_residual": None,
                "raw_overlap_residual_rmse": None,
                "reset_relative_index": chunk % 5,
                "Y_short": None,
                "Y_mid": None,
                "Y_long": None,
                "Y_scale_drift": None,
                "low_observability_score": None,
                "metric_source": f"v74tf_h35_prefix_trajectory_recomputed:{run_dir}",
                "metric_coverage_status": "prefix_real_trajectory_metrics_no_overlap_residual",
                "prefix_frame_count": int(frames.shape[0]),
                "prefix_frame_min": int(frames.min()),
                "prefix_frame_max": int(frames.max()),
                **semantic_features,
                **radio_features,
            }
        )
    return out


def _cache_only_rows(preprocess_root: Path, seq: str, max_chunks: int) -> list[dict[str, Any]]:
    stage_dir = stage_cache_dir(preprocess_root, seq)
    summary = read_stage_summary(preprocess_root, seq)
    chunks: list[tuple[int, int | None, int | None]] = []
    for chunk_dir in stage_chunk_dirs(stage_dir):
        manifest = load_json(chunk_dir / "manifest.json") or {}
        if not isinstance(manifest, dict):
            continue
        try:
            chunk_id = int(manifest.get("chunk_idx", chunk_dir.name.split("_")[1]))
        except (TypeError, ValueError, IndexError):
            continue
        start = manifest.get("start_frame")
        end = manifest.get("end_frame")
        chunks.append((chunk_id, int(start) if start is not None else None, int(end) if end is not None else None))
    if not chunks:
        n = int(summary.get("num_chunks") or 0)
        stride = int(summary.get("chunk_size") or 32) - int(summary.get("chunk_overlap") or 3)
        chunks = [(i, i * stride, i * stride + int(summary.get("chunk_size") or 32)) for i in range(n)]
    selected = set(target_chunks_for_seq(seq, [chunk for chunk, _s, _e in chunks], max_chunks=max_chunks))
    out: list[dict[str, Any]] = []
    for chunk, start, end in sorted(chunks):
        is_target = chunk in selected
        out.append(
            {
                "seq": seq,
                "chunk_id": chunk,
                "is_target_chunk": is_target,
                "frame_start": start,
                "frame_end": end,
                "local_sim3_ate": None,
                "global_chunk_ate": None,
                "oracle_gap": None,
                "head_to_tail": None,
                "future_after_overlap": None,
                "scale_cv": None,
                "boundary_jump": None,
                "raw_overlap_residual": None,
                "raw_overlap_residual_rmse": None,
                "reset_relative_index": chunk % 5,
                "Y_short": None,
                "Y_mid": None,
                "Y_long": None,
                "Y_scale_drift": None,
                "low_observability_score": None,
                "metric_source": "stage_c_cache_manifest_only",
                "metric_coverage_status": "blocked_missing_baseline_geometry_or_overlap_artifacts",
            }
        )
    return out


def _add_lowobs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        context_vals = [
            safe_float(row.get("road_context_ratio")),
            safe_float(row.get("sky_context_ratio")),
            safe_float(row.get("lowtrust_stuff_ratio")),
        ]
        stable_vals = [safe_float(row.get("stable_structure_ratio")), safe_float(row.get("radio_static_mean"))]
        if any(v is not None for v in context_vals + stable_vals):
            context = sum(v for v in context_vals if v is not None)
            stable = sum(v for v in stable_vals if v is not None)
            row["low_observability_score"] = float(context - stable)
        elif "low_observability_score" not in row:
            row["low_observability_score"] = None
        if "raw_overlap_residual" not in row or row.get("raw_overlap_residual") in (None, ""):
            row["raw_overlap_residual"] = row.get("raw_overlap_residual_rmse")


def _selection_rule_for_seq(seq: str, rows: list[dict[str, Any]]) -> str:
    if seq == "01":
        return "fixed_target_chunks"
    seq_rows = [row for row in rows if str(row.get("seq", "")).zfill(2) == seq]
    if any(str(row.get("metric_source", "")).startswith("v74tf_h35_prefix_trajectory_recomputed") for row in seq_rows):
        return "v74tf_prefix_available_chunks_until_semantic_risk_proxy_enriched"
    return "cache_manifest_order_until_geometry_risk_proxy_available"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default="01,09,00,02")
    parser.add_argument("--preprocess-root", type=Path, default=PREPROCESS_ROOT)
    parser.add_argument("--v73-report-root", type=Path, default=V73_REPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase1_multiseq_scale_drift_ledger")
    parser.add_argument("--max-non01-chunks", type=int, default=12)
    args = parser.parse_args()

    seqs = parse_seqs(args.seqs)
    rows: list[dict[str, Any]] = []
    for seq in seqs:
        if seq == "01":
            rows.extend(_load_v73_01_rows(args.v73_report_root))
        else:
            prefix_run = discover_v74tf_prefix_run(seq)
            if prefix_run is not None:
                rows.extend(_prefix_trajectory_rows(args.preprocess_root, seq, prefix_run, max_chunks=int(args.max_non01_chunks)))
            else:
                rows.extend(_cache_only_rows(args.preprocess_root, seq, max_chunks=int(args.max_non01_chunks)))
    _add_lowobs(rows)
    add_j_scale(rows)
    target_selection_rows = [
        {
            "seq": seq,
            "target_chunks": ",".join(str(row.get("chunk_id")) for row in rows if row.get("seq") == seq and row.get("is_target_chunk")),
            "target_row_count": sum(1 for row in rows if row.get("seq") == seq and row.get("is_target_chunk")),
            "selection_rule": _selection_rule_for_seq(seq, rows),
        }
        for seq in seqs
    ]
    coverage = summarize_metric_coverage(rows)
    summary = {
        "schema": "acl2_v74tf_phase1_multiseq_scale_drift_ledger_v1",
        "created_at": utc_now(),
        "seqs": seqs,
        "rows": len(rows),
        "metric_coverage_by_seq": coverage,
        "target_selection": target_selection_rows,
        "kitti01_target_rows": sum(1 for row in rows if row.get("seq") == "01" and row.get("is_target_chunk")),
        "kitti09_target_rows": sum(1 for row in rows if row.get("seq") == "09" and row.get("is_target_chunk")),
        "kitti09_complete_metric_rows": coverage.get("09", {}).get("complete_required_metric_rows", 0),
        "phase1_gate_pass": bool(
            sum(1 for row in rows if row.get("seq") == "01" and row.get("is_target_chunk")) >= 11
            and coverage.get("01", {}).get("complete_required_metric_rows", 0) >= 11
            and sum(1 for row in rows if row.get("seq") == "09" and row.get("is_target_chunk")) >= 8
            and coverage.get("09", {}).get("complete_required_metric_rows", 0) >= 8
        ),
        "gate_rule": ">=11 KITTI01 rows and >=8 KITTI09 rows with required scale/future/head-tail/boundary metric coverage.",
        "blocked_reason": "" if False else "See per-seq metric coverage; non-01 rows may be prefix-only or cache-only.",
    }
    if summary["phase1_gate_pass"]:
        summary["blocked_reason"] = ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scale_drift_ledger.csv", rows)
    for seq in seqs:
        if seq != "01":
            write_csv(
                args.out_dir / f"seq{seq}_semantic_geometry_features_by_chunk.csv",
                [row for row in rows if str(row.get("seq", "")).zfill(2) == seq],
            )
    write_json(args.out_dir / "scale_drift_ledger.json", {"summary": summary, "rows": rows})
    write_csv(args.out_dir / "target_chunk_selection_by_seq.csv", target_selection_rows)
    write_json(args.out_dir / "scale_drift_ledger_summary.json", summary)
    print(
        {
            "out_dir": str(args.out_dir),
            "rows": len(rows),
            "phase1_gate_pass": summary["phase1_gate_pass"],
            "metric_coverage_by_seq": coverage,
        }
    )


if __name__ == "__main__":
    main()
