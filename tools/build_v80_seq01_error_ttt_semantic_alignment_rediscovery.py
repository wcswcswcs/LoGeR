#!/usr/bin/env python3
"""Build a v80 seq01 error/TTT/semantic alignment rediscovery bundle.

This is an offline, diagnostic-only aggregator.  It connects existing seq01
artifacts:

- geometry-error/semantic support maps across canary chunks,
- the selected-write low-support TTT bridge for chunk08/frame232,
- the boundary-scale direction scan,
- the runtime boundary-scale oracle gate summary,
- available seq01 TTT/SWA visual panels.

It does not run the model and does not claim a method gate pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_SUPPORT_DIR = REPORT_ROOT / "phase9_seq01_ref055_v80_error_semantic_support_maps"
DEFAULT_SELECTED_WRITE_DIR = REPORT_ROOT / "phase9_seq01_ref055_v80_selected_write_support_maps"
DEFAULT_SELECTED_WRITE_EXT_DIR = REPORT_ROOT / "phase9_seq01_ref055_v80_selected_write_support_maps_canary_ext"
DEFAULT_DIRECTION = REPORT_ROOT / "phase9_seq01_boundary_scale_direction_canary5_native" / (
    "canary5_boundary_scale_direction_summary.json"
)
DEFAULT_ORACLE = REPORT_ROOT / "phase9_seq01_boundary_scale_oracle_globalfuture_canary5" / (
    "mgf_oracle_globalfuture_canary5_gate_summary.json"
)
DEFAULT_SWA_TTT = REPORT_ROOT / "phase9_seq01_chunk08_manual_ttt_visual_probe_frame232_lw42_swa_ttt_alignment"
DEFAULT_MANUAL_TTT = REPORT_ROOT / "phase9_seq01_chunk08_manual_ttt_visual_probe_frame232_lw42_chunks008_009"
DEFAULT_EXTRA_MANUAL_TTT = [
    REPORT_ROOT / "phase9_seq01_chunk06_manual_ttt_visual_probe_frame174",
    REPORT_ROOT / "phase9_seq01_chunk07_manual_ttt_visual_probe_frame203",
    REPORT_ROOT / "phase9_seq01_chunk10_manual_ttt_visual_probe_frame290",
    REPORT_ROOT / "phase9_seq01_chunk12_manual_ttt_visual_probe_frame348",
]
DEFAULT_OUT_DIR = REPORT_ROOT / "phase9_seq01_error_ttt_semantic_alignment_rediscovery"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, default=DEFAULT_SUPPORT_DIR)
    parser.add_argument(
        "--selected-write-dir",
        type=Path,
        action="append",
        default=None,
        help="Selected-write support-map directory. May be passed multiple times.",
    )
    parser.add_argument("--direction-summary", type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--swa-ttt-dir", type=Path, default=DEFAULT_SWA_TTT)
    parser.add_argument("--manual-ttt-dir", type=Path, default=DEFAULT_MANUAL_TTT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


def _copy_files(src_paths: list[Path], dst_dir: Path) -> list[str]:
    copied: list[str] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for src in src_paths:
        if not src.exists() or not src.is_file():
            continue
        dst = dst_dir / src.name
        if dst.name in seen:
            stem = src.stem
            suffix = src.suffix
            parent_tag = src.parent.name[:30]
            dst = dst_dir / f"{stem}_{parent_tag}{suffix}"
        shutil.copy2(src, dst)
        seen.add(dst.name)
        copied.append(str(dst))
    return copied


def _support_rows(support_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(support_dir.glob("chunk_*_support_map_summary.json")):
        payload = _read_json(path)
        if not payload:
            continue
        chunk = int(payload.get("chunk"))
        score_mean = _safe_float(payload.get("score_mean"))
        out[chunk] = {
            "support_summary": str(path),
            "chunk": chunk,
            "target_start_frame": payload.get("start_frame"),
            "support_score_mean": score_mean,
            "support_low_proxy_1_minus_mean": None if score_mean is None else 1.0 - score_mean,
            "support_score_q10": payload.get("score_q10"),
            "support_score_q50": payload.get("score_q50"),
            "support_score_q90": payload.get("score_q90"),
            "support_tokens_per_frame": payload.get("tokens_per_frame"),
            "support_risk_label_names": payload.get("risk_label_names", []),
            "support_stable_label_names": payload.get("stable_label_names", []),
        }
    return out


def _direction_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("chunks", []):
        chunk = int(row.get("chunk"))
        out[chunk] = {
            "direction_summary": str(path),
            "direction_signature": row.get("direction_signature"),
            "all_key_metrics_same_direction": row.get("all_key_metrics_same_direction"),
            "global_future_best_scale": row.get("global_future_from_boundary_rmse_m_best_scale"),
            "global_future_best_direction": row.get("global_future_from_boundary_rmse_m_best_direction"),
            "global_future_improvement_ratio": row.get("global_future_from_boundary_rmse_m_improvement_ratio"),
            "tail3_future_best_scale": row.get("tail3_to_future_from_boundary_sim3_rmse_m_best_scale"),
            "tail3_future_best_direction": row.get("tail3_to_future_from_boundary_sim3_rmse_m_best_direction"),
            "tail3_future_improvement_ratio": row.get("tail3_to_future_from_boundary_sim3_rmse_m_improvement_ratio"),
        }
    return out


def _oracle_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("chunk_decisions", []):
        chunk = int(row.get("chunk"))
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        head = metrics.get("head10_to_tail10_pose_sim3_rmse_m", {})
        overlap = metrics.get("overlap3_to_future_pose_sim3_rmse_m", {})
        out[chunk] = {
            "oracle_summary": str(path),
            "oracle_head_tail_improvement_vs_baseline_ratio": row.get("head_tail_improvement_vs_baseline_ratio"),
            "oracle_overlap_improvement_vs_baseline_ratio": row.get("overlap_improvement_vs_baseline_ratio"),
            "oracle_head_tail_phaseE_chunk_pass": row.get("head_tail_phaseE_chunk_pass"),
            "oracle_overlap_phaseE_chunk_pass": row.get("overlap_phaseE_chunk_pass"),
            "oracle_head_candidate_rmse": head.get("candidate"),
            "oracle_head_baseline_rmse": head.get("baseline"),
            "oracle_head_best_control_rmse": head.get("best_control"),
            "oracle_overlap_candidate_rmse": overlap.get("candidate"),
            "oracle_overlap_baseline_rmse": overlap.get("baseline"),
            "oracle_overlap_best_control_rmse": overlap.get("best_control"),
        }
    return out


def _selected_write_row(selected_dirs: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for selected_dir in selected_dirs:
        for path in sorted(selected_dir.glob("chunk_*_selected_write_support_map_summary.json")):
            payload = _read_json(path)
            if not payload:
                continue
            chunk = int(payload.get("chunk"))
            selected_given_selected = _safe_float(payload.get("selected_low_support_given_selected_runtime"))
            global_low = _safe_float(payload.get("runtime_low_support_ratio"))
            out[chunk] = {
                "selected_write_summary": str(path),
                "has_ttt_selected_write_evidence": True,
                "selected_write_global_frame": payload.get("global_frame"),
                "selected_runtime_mass": payload.get("selected_runtime_mass"),
                "selected_low_support_mass": payload.get("selected_low_support_mass"),
                "selected_low_support_given_selected_runtime": selected_given_selected,
                "runtime_low_support_ratio": global_low,
                "selected_low_support_enrichment_vs_global": _ratio(selected_given_selected, global_low),
                "selected_low_support_given_low_support": payload.get("selected_low_support_given_low_support"),
                "selected_visual_ratio": payload.get("selected_visual_ratio"),
                "source_post_delta_pt": payload.get("source_post_delta_pt"),
                "source_support_map": payload.get("source_support_map"),
            }
    return out


def _interpret(row: dict[str, Any]) -> str:
    chunk = row.get("chunk")
    selected_enrichment = _safe_float(row.get("selected_low_support_enrichment_vs_global"))
    head_pass = bool(row.get("oracle_head_tail_phaseE_chunk_pass"))
    overlap_pass = bool(row.get("oracle_overlap_phaseE_chunk_pass"))
    low_proxy = _safe_float(row.get("support_low_proxy_1_minus_mean"))
    if chunk == 8 and selected_enrichment is not None and selected_enrichment > 1.2:
        return "local_semantic_explains_ttt_low_support_write_but_not_runtime_scale_gate"
    if head_pass and not overlap_pass:
        return "boundary_scale_head_tail_only_overlap_conflict"
    if low_proxy is not None and low_proxy > 0.3:
        return "semantic_low_support_boundary_but_no_current_control_separation"
    return "no_sufficient_semantic_ttt_alignment_evidence"


def _build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    support = _support_rows(args.support_dir)
    directions = _direction_rows(args.direction_summary)
    oracle = _oracle_rows(args.oracle_summary)
    selected_dirs = _selected_write_dirs(args)
    selected = _selected_write_row(selected_dirs)
    chunks = sorted(set(support) | set(directions) | set(oracle) | set(selected))
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        row: dict[str, Any] = {"chunk": chunk}
        row.update(support.get(chunk, {}))
        row.update(directions.get(chunk, {}))
        row.update(oracle.get(chunk, {}))
        row.update(selected.get(chunk, {"has_ttt_selected_write_evidence": False}))
        row["interpretation"] = _interpret(row)
        rows.append(row)
    return rows


def _mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _build_summary(rows: list[dict[str, Any]], args: argparse.Namespace, copied: dict[str, list[str]]) -> dict[str, Any]:
    support_sorted = sorted(
        [r for r in rows if _safe_float(r.get("support_score_mean")) is not None],
        key=lambda r: float(r["support_score_mean"]),
    )
    selected_chunks = [int(r["chunk"]) for r in rows if r.get("has_ttt_selected_write_evidence")]
    selected_zero_low_support_chunks = [
        int(r["chunk"])
        for r in rows
        if r.get("has_ttt_selected_write_evidence") and _safe_float(r.get("selected_low_support_mass")) == 0.0
    ]
    selected_positive_low_support_chunks = [
        int(r["chunk"])
        for r in rows
        if r.get("has_ttt_selected_write_evidence") and (_safe_float(r.get("selected_low_support_mass")) or 0.0) > 0.0
    ]
    head_pass_chunks = [int(r["chunk"]) for r in rows if bool(r.get("oracle_head_tail_phaseE_chunk_pass"))]
    overlap_pass_chunks = [int(r["chunk"]) for r in rows if bool(r.get("oracle_overlap_phaseE_chunk_pass"))]
    selected_enrichments = [
        _safe_float(r.get("selected_low_support_enrichment_vs_global"))
        for r in rows
        if r.get("has_ttt_selected_write_evidence")
    ]
    image_counts = {key: len(value) for key, value in copied.items()}
    required_panel_sets_present = bool(
        image_counts.get("long_ttt_branch_visual_panels", 0)
        and image_counts.get("mid_swa_qkv_visual_panels", 0)
        and image_counts.get("merge_boundary_visual_panels", 0)
        and image_counts.get("short_qk_pair_visual_panels", 0)
    )
    return {
        "schema": "acl2_v80_seq01_error_ttt_semantic_alignment_rediscovery_v1",
        "status": "rediscovery_partial",
        "v80_goal_achieved": False,
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "support_dir": str(args.support_dir),
        "selected_write_dirs": [str(path) for path in _selected_write_dirs(args)],
        "direction_summary": str(args.direction_summary),
        "oracle_summary": str(args.oracle_summary),
        "row_count": len(rows),
        "canary_chunks": [int(r["chunk"]) for r in rows],
        "ttt_selected_write_evidence_chunks": selected_chunks,
        "selected_write_positive_low_support_chunks": selected_positive_low_support_chunks,
        "selected_write_zero_low_support_chunks": selected_zero_low_support_chunks,
        "lowest_support_chunk": int(support_sorted[0]["chunk"]) if support_sorted else None,
        "lowest_support_score_mean": support_sorted[0].get("support_score_mean") if support_sorted else None,
        "selected_write_low_support_enrichment_mean": _mean(selected_enrichments),
        "oracle_head_tail_pass_chunks": head_pass_chunks,
        "oracle_overlap_pass_chunks": overlap_pass_chunks,
        "all_chunks_have_ttt_selected_write_evidence": len(selected_chunks) == len(rows) and bool(rows),
        "required_phase9_panel_sets_present": required_panel_sets_present,
        "visual_audit_gate_pass": False,
        "visual_audit_reason": (
            "Partial seq01 rediscovery bundle: TTT and SWA/TTT visual panels are present for chunk08, "
            "but short-QK and merge-boundary panel groups are not complete in this bundle."
        ),
        "copied_visual_files": copied,
        "decision": (
            "Semantic/TTT evidence can localize the chunk08/frame232 low-support selected-write region, "
            "but it does not predict the correct boundary-scale action across canary5 and cannot be "
            "promoted without stronger non-GT, control-separated carrier evidence."
        ),
        "next_action": (
            "Do not run more fixed scale sweeps. If continuing, either search for a non-GT gauge "
            "direction signal that explains chunk10/chunk12 without overlap harm, or test an "
            "OUT3/MEMIX no-persistent rule for chunk08 only with same-mass random and geometry-only controls."
        ),
    }


def _write_question_files(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    question_rows = []
    for row in rows:
        question_rows.append(
            {
                "case_id": f"seq01_chunk{int(row['chunk']):03d}",
                "frame": row.get("target_start_frame"),
                "chunk": row.get("chunk"),
                "has_ttt_selected_write_evidence": row.get("has_ttt_selected_write_evidence"),
                "support_score_mean": row.get("support_score_mean"),
                "selected_low_support_enrichment_vs_global": row.get("selected_low_support_enrichment_vs_global"),
                "oracle_head_tail_improvement_vs_baseline_ratio": row.get(
                    "oracle_head_tail_improvement_vs_baseline_ratio"
                ),
                "oracle_overlap_improvement_vs_baseline_ratio": row.get(
                    "oracle_overlap_improvement_vs_baseline_ratio"
                ),
                "visual_question": (
                    "Does the TTT selected-write region coincide with low semantic/geometry support, and can "
                    "that non-GT evidence predict the boundary-scale direction without losing to controls?"
                ),
            }
        )
    _write_csv(out_dir / "failed_case_to_visual_question.csv", question_rows)
    _write_csv(out_dir / "visual_review.csv", rows)


def _write_markdown(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    chunk_lines = []
    for row in rows:
        fmt = {
            "chunk": row.get("chunk"),
            "support_score_mean": row.get("support_score_mean"),
            "has_ttt_selected_write_evidence": row.get("has_ttt_selected_write_evidence"),
            "selected_low_support_enrichment_vs_global": row.get("selected_low_support_enrichment_vs_global"),
            "global_future_best_scale": row.get("global_future_best_scale"),
            "oracle_head_tail_improvement_vs_baseline_ratio": row.get(
                "oracle_head_tail_improvement_vs_baseline_ratio"
            ),
            "oracle_overlap_improvement_vs_baseline_ratio": row.get(
                "oracle_overlap_improvement_vs_baseline_ratio"
            ),
            "interpretation": row.get("interpretation"),
        }
        chunk_lines.append(
            "- chunk {chunk}: support_mean={support_score_mean}, selected_write={has_ttt_selected_write_evidence}, "
            "enrichment={selected_low_support_enrichment_vs_global}, global_scale={global_future_best_scale}, "
            "head_imp={oracle_head_tail_improvement_vs_baseline_ratio}, overlap_imp={oracle_overlap_improvement_vs_baseline_ratio}, "
            "interpretation={interpretation}".format(**fmt)
        )
    insight = """# Seq01 Error/TTT/Semantic Alignment Rediscovery

Status: partial rediscovery artifact, not a method pass.

Key findings:

- Lowest support canary chunk: `{lowest_support_chunk}` with score_mean `{lowest_support_score_mean}`.
- TTT selected-write evidence exists for chunks: `{ttt_selected_write_evidence_chunks}`.
- Selected-write positive low-support chunks: `{selected_write_positive_low_support_chunks}`.
- Selected-write zero low-support chunks: `{selected_write_zero_low_support_chunks}`.
- Mean selected-write low-support enrichment over available selected-write chunks: `{selected_write_low_support_enrichment_mean}`.
- Boundary-scale runtime oracle head-tail pass chunks: `{oracle_head_tail_pass_chunks}`.
- Boundary-scale runtime oracle overlap pass chunks: `{oracle_overlap_pass_chunks}`.
- Full Phase9 visual audit gate pass: `{visual_audit_gate_pass}`.

Per-chunk alignment:

{chunk_lines}

Interpretation:

The chunk08/frame232 TTT selected-write region is strongly enriched in the low-support semantic/geometry map. That supports the user's hypothesis that semantic evidence can explain a bad TTT write location. However, the same chunk fails the boundary-scale runtime oracle against controls. The head-tail runtime wins are chunk10/chunk12, and their selected-write probes show zero low-support overlap under this support-map rule. This means semantic/TTT low-support localization is not the current reliable policy for gauge/scale correction.

Decision:

`{decision}`

Next action:

`{next_action}`
""".format(chunk_lines="\n".join(chunk_lines), **summary)
    (out_dir / "visual_insight.md").write_text(insight, encoding="utf-8")

    hypothesis = """# New Semantic Memory Hypothesis Bank

Status: seq01 partial rediscovery. No method success claimed.

## HYP-V80-SEQ01-ERRTTT-001_SELECTED_WRITE_LOW_SUPPORT_NO_PERSISTENT

- memory body: long-term TTT write memory
- case type: seq01 chunk08/frame232 bad boundary
- visual evidence file: `long_ttt_branch_visual_panels/` and `mid_swa_qkv_visual_panels/`
- semantic role: selected TTT write tokens are enriched in low semantic/geometry support
- proposed action point: OUT3_TTT_OUTLIER_NO_PERSISTENT or MEMIX low-observation no-write
- expected metric: reduce future/overlap drift only if the selected bad write is causal
- controls: same-write-mass random, geometry-only TTL, label-confidence shuffle
- stop rule: if overlap or head-tail does not beat controls on canary chunks, do not promote

## HYP-V80-SEQ01-ERRTTT-002_LOW_SUPPORT_IS_NOT_SCALE_DIRECTION

- memory body: merge/gauge boundary memory
- case type: seq01 canary5 boundary-scale diagnostic
- visual evidence file: `canary_error_ttt_semantic_alignment_rows.csv`
- semantic role: low-support semantic/geometry maps localize risk, but do not choose up/down scale direction
- proposed action point: separate risk localization from gauge action; require a non-GT direction signal before runtime
- expected metric: direction predictor should agree with both head-tail and overlap controls, not only global_future oracle
- controls: opposite scale control, geometry-only, same-mass random semantic mask
- stop rule: if direction remains metric-conflicted, stop scale sweeps

## HYP-V80-SEQ01-ERRTTT-003_HEAD_WIN_CHUNKS_ARE_NOT_LOW_SUPPORT_TTT_WRITES

- memory body: TTT/SWA/merge handoff
- case type: seq01 chunk10 and chunk12 head-tail-only MG3 oracle wins
- visual evidence file: `canary_error_ttt_semantic_alignment_rows.csv`
- semantic role: selected-write low-support enrichment is absent on the head-tail-only runtime wins
- proposed action point: do not use selected-write low-support as the scale/gauge carrier; look for another non-GT direction signal
- expected metric: any new direction signal should explain chunk10/chunk12 without harming overlap and without losing to opposite controls
- controls: same-mass random selected write, geometry-only, opposite scale control
- stop rule: if no non-GT direction signal separates head-tail wins from failures, MG3 remains geometry-only diagnostic
"""
    (out_dir / "new_semantic_memory_hypothesis_bank.md").write_text(hypothesis, encoding="utf-8")


def _copy_visuals(args: argparse.Namespace, out_dir: Path) -> dict[str, list[str]]:
    for name in [
        "long_ttt_branch_visual_panels",
        "mid_swa_qkv_visual_panels",
        "merge_boundary_visual_panels",
        "short_qk_pair_visual_panels",
    ]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    ttt_images: list[Path] = []
    for root in [args.manual_ttt_dir, *DEFAULT_EXTRA_MANUAL_TTT]:
        combined = sorted((root / "combined").glob("**/*.png"))
        visual = sorted((root / "targets").glob("**/LW1_TTT_SEMANTIC_BASE/visual/*.png"))
        ttt_images.extend((combined or visual)[:8])
    swa_images = sorted((args.swa_ttt_dir / "visual_panels").glob("*.png"))[:8]
    copied = {
        "long_ttt_branch_visual_panels": _copy_files(ttt_images, out_dir / "long_ttt_branch_visual_panels"),
        "mid_swa_qkv_visual_panels": _copy_files(swa_images, out_dir / "mid_swa_qkv_visual_panels"),
        "merge_boundary_visual_panels": [],
        "short_qk_pair_visual_panels": [],
    }
    for missing in ["merge_boundary_visual_panels", "short_qk_pair_visual_panels"]:
        (out_dir / missing / "README.md").write_text(
            "Not complete in this seq01 partial rediscovery bundle; required before a full Phase9 visual audit pass.\n",
            encoding="utf-8",
        )
    return copied


def _selected_write_dirs(args: argparse.Namespace) -> list[Path]:
    if args.selected_write_dir:
        return [Path(path) for path in args.selected_write_dir]
    dirs = [DEFAULT_SELECTED_WRITE_DIR]
    if DEFAULT_SELECTED_WRITE_EXT_DIR.exists():
        dirs.append(DEFAULT_SELECTED_WRITE_EXT_DIR)
    return dirs


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(args)
    copied = _copy_visuals(args, args.out_dir)
    summary = _build_summary(rows, args, copied)
    _write_csv(args.out_dir / "canary_error_ttt_semantic_alignment_rows.csv", rows)
    _write_question_files(args.out_dir, rows)
    _write_markdown(args.out_dir, rows, summary)
    _write_json(args.out_dir / "rediscovery_summary.json", summary)
    _write_json(
        args.out_dir / "visual_integrity_audit.json",
        {
            "schema": "acl2_v80_seq01_error_ttt_semantic_alignment_visual_audit_v1",
            "status": "partial",
            "gate_pass": False,
            "method_gate_claimed": False,
            "reason": summary["visual_audit_reason"],
            "required_phase9_panel_sets_present": summary["required_phase9_panel_sets_present"],
            "copied_visual_files": copied,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "summary": str(args.out_dir / "rediscovery_summary.json"),
                "rows": str(args.out_dir / "canary_error_ttt_semantic_alignment_rows.csv"),
                "v80_goal_achieved": False,
                "status": "rediscovery_partial",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
