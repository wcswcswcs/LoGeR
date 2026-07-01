#!/usr/bin/env python3
"""Batch-mine v80 selected-write low-support insights from Phase2 chunks.

This diagnostic expands the current selected-write low-support evidence set
without changing runtime behavior.  It builds absolute-error semantic support
maps and selected-write support maps for explicitly listed Phase2 chunks, then
summarizes positives and counterexamples in one table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v80_error_semantic_overlap_support import build as build_error_support
from tools.build_v80_selected_write_support_map import build as build_selected_support


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
PHASE2_DIR = REPORT_ROOT / "phase10_phase2_multiseq_ttt_abs_error_overlap_20260622_2223"
DEFAULT_FRAME_ROWS = PHASE2_DIR / "phase2_abs_error_overlap_frame_rows.csv"
DEFAULT_CHUNK_ROWS = PHASE2_DIR / "phase2_abs_error_overlap_chunk_rows.csv"
DEFAULT_TTT_SUMMARY = PHASE2_DIR / "phase2_abs_error_overlap_summary.json"
DEFAULT_OUT_DIR = REPORT_ROOT / f"phase10_selected_write_extra_insights_{datetime.now().strftime('%Y%m%d_%H%M')}"

DEFAULT_TARGETS = (
    "bad_candidate:02:62",
    "bad_candidate:02:63",
    "bad_candidate:02:64",
    "bad_candidate:02:65",
    "bad_candidate:02:67",
    "bad_candidate:02:68",
    "bad_candidate:02:70",
    "good_counterexample:02:26",
    "good_counterexample:02:27",
    "good_counterexample:02:41",
    "good_counterexample:02:43",
    "good_counterexample:02:44",
    "good_counterexample:02:46",
    "good_counterexample:05:08",
    "good_counterexample:05:22",
    "good_counterexample:05:23",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-rows", type=Path, default=DEFAULT_FRAME_ROWS)
    parser.add_argument("--chunk-rows", type=Path, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--ttt-attribution-summary", type=Path, default=DEFAULT_TTT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--stage-c-root", type=Path, default=Path("results/kitti_preprocess"))
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument("--bad-delta-scale", type=float, default=0.35)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--patch-grid", nargs=2, type=int, default=(19, 66))
    parser.add_argument("--runtime-grid", nargs=2, type=int, default=(19, 66))
    parser.add_argument("--layer-idx", type=int, default=18)
    parser.add_argument("--seed", type=int, default=80623)
    parser.add_argument("--control-seed", type=int, default=90623)
    parser.add_argument("--skip-existing", type=int, default=0)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True)
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_targets(raw_targets: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in raw_targets:
        parts = str(raw).split(":")
        if len(parts) != 3:
            raise ValueError(f"target must be group:seq:chunk, got {raw!r}")
        group, seq, chunk = parts
        out.append({"group": group, "seq": seq.zfill(2), "chunk": int(chunk)})
    return out


def _write_geometry_csv(path: Path, frame_rows: list[dict[str, str]], seq: str, chunk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame_rows:
        if str(row.get("seq", "")).zfill(2) != seq or _int(row.get("chunk")) != int(chunk):
            continue
        frame = _int(row.get("global_frame"))
        if frame is None:
            continue
        baseline_error = _float(row.get("baseline_abs_error_m"))
        delta = _float(row.get("write_probe_delta_vs_native_m"))
        candidate_error = baseline_error + delta
        rows.append(
            {
                "frame": frame,
                "primary_chunk_id": int(chunk),
                "local_frame": _int(row.get("local_frame")),
                "baseline_error_m": baseline_error,
                "candidate_error_m": candidate_error,
                "control_error_m": baseline_error,
                "delta_error_vs_baseline_m": delta,
                "delta_error_vs_control_m": delta,
            }
        )
    _write_csv(path, rows)
    return rows


def _aggregate(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "min": min(clean),
        "mean": sum(clean) / len(clean),
        "max": max(clean),
    }


def main() -> None:
    args = _parse_args()
    targets = _parse_targets(list(args.targets))
    frame_rows = _read_csv(args.frame_rows)
    chunk_rows = _read_csv(args.chunk_rows)
    chunk_by_key = {
        (str(row.get("seq", "")).zfill(2), int(float(row.get("chunk", "nan")))): row
        for row in chunk_rows
        if row.get("chunk")
    }

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target in targets:
        seq = str(target["seq"]).zfill(2)
        chunk = int(target["chunk"])
        group = str(target["group"])
        chunk_row = chunk_by_key.get((seq, chunk), {})
        if not chunk_row:
            errors.append({"seq": seq, "chunk": chunk, "group": group, "error": "missing_phase2_chunk_row"})
            continue

        chunk_dir = args.out_dir / f"{group}_seq{seq}_chunk{chunk:03d}"
        geom_csv = chunk_dir / "per_frame_error_delta.csv"
        generated_frames = _write_geometry_csv(geom_csv, frame_rows, seq, chunk)
        if not generated_frames:
            errors.append({"seq": seq, "chunk": chunk, "group": group, "error": "missing_phase2_frame_rows"})
            continue

        support_dir = chunk_dir / "abs_error_semantic_support"
        selected_dir = chunk_dir / "selected_write_support"
        support_summary_path = support_dir / f"chunk_{chunk:03d}_support_map_summary.json"
        selected_summary_path = selected_dir / f"chunk_{chunk:03d}_selected_write_support_map_summary.json"

        if int(args.skip_existing) and support_summary_path.is_file() and selected_summary_path.is_file():
            support_summary = json.loads(support_summary_path.read_text(encoding="utf-8"))
            selected_summary = json.loads(selected_summary_path.read_text(encoding="utf-8"))
        else:
            support_summary = build_error_support(
                argparse.Namespace(
                    stage_c_cache_dir=args.stage_c_root / seq / "stage_c_cache_semantic_chunks",
                    chunk=chunk,
                    overlap=int(args.overlap),
                    patch_grid=tuple(int(x) for x in args.patch_grid),
                    geometry_error_csv=geom_csv,
                    ttt_attribution_summary=args.ttt_attribution_summary,
                    out_dir=support_dir,
                    kind="source_gate",
                    layer_idx=int(args.layer_idx),
                    bad_delta_scale=float(args.bad_delta_scale),
                    bad_delta_key="baseline_error_m",
                    low_conf_threshold=0.45,
                    risk_penalty_min=0.15,
                    low_conf_penalty=0.25,
                    stable_bonus=0.05,
                    seed=int(args.seed),
                )
            )
            selected_summary = build_selected_support(
                argparse.Namespace(
                    visual_root=Path("."),
                    case="phase2_abs_error_probe",
                    post_delta_pt=Path(chunk_row["post_delta_path"]),
                    source_support_map=Path(support_summary["support_path"]),
                    stage_c_masklet=Path(chunk_row["stage_c_masklet"]),
                    out_dir=selected_dir,
                    chunk=chunk,
                    global_frame=int(_int(chunk_row.get("start_frame")) or generated_frames[0]["frame"]),
                    overlap=int(args.overlap),
                    runtime_grid=tuple(int(x) for x in args.runtime_grid),
                    kind="source_gate",
                    layer_idx=int(args.layer_idx),
                    support_threshold=float(args.support_threshold),
                    d_tok_quantile=0.75,
                    low_conf_threshold=0.55,
                    seed=int(args.seed),
                    control_seed=int(args.control_seed) + chunk,
                    exclude_selected_from_control=1,
                    control_pool="low_support",
                )
            )

        selected_low = _float(selected_summary.get("selected_low_support_given_selected_runtime"))
        support_q10 = _float(support_summary.get("score_q10"))
        baseline_mean = _float(chunk_row.get("baseline_abs_error_mean_m"))
        selected_risk = _float(chunk_row.get("selected_risk_given_selected"))
        is_positive = selected_low >= float(args.support_threshold)
        rows.append(
            {
                "group": group,
                "seq": seq,
                "chunk": chunk,
                "case_types_phase2": chunk_row.get("case_types"),
                "baseline_abs_error_mean_m_phase2": baseline_mean,
                "selected_risk_given_selected_phase2": selected_risk,
                "weighted_selected_risk_error_mean_phase2": _float(chunk_row.get("weighted_selected_risk_error_mean")),
                "support_score_mean": _float(support_summary.get("score_mean")),
                "support_score_q10": support_q10,
                "support_score_q50": _float(support_summary.get("score_q50")),
                "selected_runtime_mass": int(_float(selected_summary.get("selected_runtime_mass"))),
                "runtime_low_support_mass": int(_float(selected_summary.get("runtime_low_support_mass"))),
                "selected_low_support_mass": int(_float(selected_summary.get("selected_low_support_mass"))),
                "selected_low_support_given_selected_runtime": selected_low,
                "diagnostic_positive_flag": is_positive,
                "support_summary": support_summary_path,
                "selected_summary": selected_summary_path,
            }
        )

    positive_rows = [row for row in rows if row.get("diagnostic_positive_flag")]
    negative_rows = [row for row in rows if not row.get("diagnostic_positive_flag")]
    summary = {
        "schema": "acl2_v80_selected_write_support_batch_insights_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "target_count": len(targets),
        "row_count": len(rows),
        "error_count": len(errors),
        "positive_count": len(positive_rows),
        "positive_seq_chunks": [{"seq": row["seq"], "chunk": row["chunk"], "group": row["group"]} for row in positive_rows],
        "counterexample_count": len(negative_rows),
        "selected_low_support_by_group": {
            group: _aggregate(
                [
                    _float(row.get("selected_low_support_given_selected_runtime"))
                    for row in rows
                    if row.get("group") == group
                ]
            )
            for group in sorted({str(row.get("group")) for row in rows})
        },
        "support_q10_by_group": {
            group: _aggregate([_float(row.get("support_score_q10")) for row in rows if row.get("group") == group])
            for group in sorted({str(row.get("group")) for row in rows})
        },
        "top_positive_rows": sorted(
            positive_rows,
            key=lambda row: _float(row.get("selected_low_support_given_selected_runtime")),
            reverse=True,
        )[:8],
        "top_counterexample_rows": sorted(
            negative_rows,
            key=lambda row: _float(row.get("selected_risk_given_selected_phase2")),
            reverse=True,
        )[:8],
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "selected_write_extra_insight_rows.csv", rows)
    _write_json(args.out_dir / "selected_write_extra_insight_summary.json", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
