#!/usr/bin/env python3
"""Join geometry error maps, TTT write evidence, and semantic support for v80.

This is a read-only audit over already materialized artifacts. It answers a
specific question: does the geometry error map plus TTT visualization/support
evidence explain the bad TTT write, and is that explanation strong enough to
promote a runtime rule?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_BRIDGE_ROOT = REPORT_ROOT / "phase9_seq01_ref055_geometry_delta_bridge"
DEFAULT_REDISCOVERY_ROOT = REPORT_ROOT / "phase10_seq01_error_ttt_semantic_alignment_rediscovery_20260622_2030"
DEFAULT_POSTDELTA_ROOT = REPORT_ROOT / "phase10_seq01_ttt_postdelta_region_carrier_audit_20260622_2220"
DEFAULT_DECISIONS = (
    REPORT_ROOT
    / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
    / "thingstuff_radio_qscale_ref055_canary5_decisions.csv"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_geometry_error_ttt_semantic_explanation_20260622_2245"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--rediscovery-root", type=Path, default=DEFAULT_REDISCOVERY_ROOT)
    parser.add_argument("--postdelta-root", type=Path, default=DEFAULT_POSTDELTA_ROOT)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="6,7,8,9,10,12")
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _index_by_chunk(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        chunk = _safe_int(row.get("chunk"))
        if chunk is not None:
            out[chunk] = row
    return out


def _summarize_per_frame(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {
            "per_frame_csv": str(path),
            "per_frame_exists": path.exists(),
            "per_frame_count": 0,
        }

    delta_base = [_safe_float(row.get("delta_error_vs_baseline_m")) for row in rows]
    delta_ctrl = [_safe_float(row.get("delta_error_vs_control_m")) for row in rows]
    delta_base_vals = [value for value in delta_base if value is not None]
    delta_ctrl_vals = [value for value in delta_ctrl if value is not None]
    max_row = None
    max_delta = None
    for row, value in zip(rows, delta_base):
        if value is None:
            continue
        if max_delta is None or value > max_delta:
            max_delta = value
            max_row = row

    return {
        "per_frame_csv": str(path),
        "per_frame_exists": True,
        "per_frame_count": len(rows),
        "delta_error_vs_baseline_m_mean": sum(delta_base_vals) / len(delta_base_vals) if delta_base_vals else None,
        "delta_error_vs_baseline_m_max": max(delta_base_vals) if delta_base_vals else None,
        "delta_error_vs_control_m_mean": sum(delta_ctrl_vals) / len(delta_ctrl_vals) if delta_ctrl_vals else None,
        "delta_error_vs_control_m_max": max(delta_ctrl_vals) if delta_ctrl_vals else None,
        "positive_delta_vs_baseline_frac": (
            sum(1 for value in delta_base_vals if value > 0.0) / len(delta_base_vals) if delta_base_vals else None
        ),
        "max_delta_frame": _safe_int(max_row.get("frame")) if max_row else None,
        "max_delta_primary_chunk": _safe_int(max_row.get("primary_chunk_id")) if max_row else None,
        "max_delta_local_frame": _safe_int(max_row.get("local_frame")) if max_row else None,
    }


def _bridge_row(bridge_root: Path, chunk: int) -> dict[str, Any]:
    chunk_dir = bridge_root / f"chunk{chunk:02d}"
    summary_path = chunk_dir / "geometry_ttt_semantic_bridge_summary.json"
    selected_path = chunk_dir / "selected_geometry_ttt_semantic_rows.csv"
    per_frame_path = chunk_dir / "per_frame_error_delta.csv"
    summary = _read_json(summary_path)
    args = summary.get("args") or {}
    rows = _read_csv(selected_path)
    return {
        "geometry_bridge_summary": str(summary_path),
        "geometry_bridge_summary_exists": summary_path.exists(),
        "bridge_skip_frame_semantic": bool(args.get("skip_frame_semantic")) if args else None,
        "selected_geometry_ttt_semantic_rows_csv": str(selected_path),
        "selected_geometry_ttt_semantic_rows_count": len(rows),
        "trajectory_error_map_xz_png": (summary.get("plot_paths") or {}).get("trajectory_error_map_xz_png"),
        "error_over_frame_png": (summary.get("plot_paths") or {}).get("error_over_frame_png"),
        **_summarize_per_frame(per_frame_path),
    }


def _decision_fields(row: dict[str, str] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "head_tail_phaseE_chunk_pass": _bool_text(row.get("head_tail_phaseE_chunk_pass")),
        "overlap_phaseE_chunk_pass": _bool_text(row.get("overlap_phaseE_chunk_pass")),
        "head_tail_improvement_vs_baseline_ratio": _safe_float(row.get("head_tail_improvement_vs_baseline_ratio")),
        "overlap_improvement_vs_baseline_ratio": _safe_float(row.get("overlap_improvement_vs_baseline_ratio")),
        "head_tail_beats_controls": _bool_text(row.get("head_tail_beats_controls")),
        "overlap_beats_controls": _bool_text(row.get("overlap_beats_controls")),
    }


def _classify_outcome(row: dict[str, Any]) -> str:
    head = bool(row.get("head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "helpful_overlap_safe"
    if head and not overlap:
        return "head_tail_only_overlap_harm"
    return "not_helpful"


def _joined_rows(args: argparse.Namespace, chunks: list[int]) -> list[dict[str, Any]]:
    rediscovery = _index_by_chunk(_read_csv(args.rediscovery_root / "canary_error_ttt_semantic_alignment_rows.csv"))
    postdelta = _index_by_chunk(_read_csv(args.postdelta_root / "ttt_postdelta_region_feature_rows.csv"))
    decisions = _index_by_chunk(_read_csv(args.decisions_csv))

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        red = rediscovery.get(chunk, {})
        post = postdelta.get(chunk, {})
        decision = _decision_fields(decisions.get(chunk))
        row: dict[str, Any] = {
            "chunk": chunk,
            **_bridge_row(args.geometry_bridge_root, chunk),
            "support_score_mean": _safe_float(red.get("support_score_mean")),
            "support_low_proxy_1_minus_mean": _safe_float(red.get("support_low_proxy_1_minus_mean")),
            "has_ttt_selected_write_evidence": _bool_text(red.get("has_ttt_selected_write_evidence")),
            "selected_write_global_frame": _safe_int(red.get("selected_write_global_frame")),
            "selected_runtime_mass": _safe_float(red.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(red.get("selected_low_support_mass")),
            "selected_low_support_given_selected_runtime": _safe_float(
                red.get("selected_low_support_given_selected_runtime")
            ),
            "selected_low_support_enrichment_vs_global": _safe_float(red.get("selected_low_support_enrichment_vs_global")),
            "rediscovery_interpretation": red.get("interpretation"),
            "action_top_low_support_frac": _safe_float(post.get("action_top_low_support_frac")),
            "action_top_high_Dq_frac": _safe_float(post.get("action_top_high_Dq_frac")),
            "action_top_low_support_enrichment": _safe_float(post.get("action_top_low_support_enrichment")),
            "corr_action_delta_low_support": _safe_float(post.get("corr_action_delta_low_support")),
            "postdelta_qscale_outcome_class": post.get("qscale_outcome_class"),
            **decision,
        }
        ratio = row.get("selected_low_support_given_selected_runtime")
        enrichment = row.get("selected_low_support_enrichment_vs_global")
        low_mass = row.get("selected_low_support_mass")
        row["selected_write_low_support_explains"] = bool(
            ratio is not None and ratio >= 0.5 and low_mass is not None and low_mass > 0.0
        )
        row["selected_write_low_support_enriched"] = bool(
            row["selected_write_low_support_explains"]
            and (enrichment is None or enrichment >= 1.2)
        )
        row["qscale_outcome_class"] = _classify_outcome(row)
        out.append(row)
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _audit_rule(name: str, description: str, rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    selected_chunks = [row["chunk"] for row in selected]
    helpful_safe = [
        row["chunk"]
        for row in selected
        if row.get("head_tail_phaseE_chunk_pass") and row.get("overlap_phaseE_chunk_pass")
    ]
    overlap_harm = [
        row["chunk"]
        for row in selected
        if row.get("head_tail_phaseE_chunk_pass") and not row.get("overlap_phaseE_chunk_pass")
    ]
    head_imps = [
        row["head_tail_improvement_vs_baseline_ratio"]
        for row in selected
        if row.get("head_tail_improvement_vs_baseline_ratio") is not None
    ]
    overlap_imps = [
        row["overlap_improvement_vs_baseline_ratio"]
        for row in selected
        if row.get("overlap_improvement_vs_baseline_ratio") is not None
    ]
    return {
        "rule": name,
        "description": description,
        "selected_chunks": selected_chunks,
        "selected_count": len(selected_chunks),
        "helpful_safe_chunks": helpful_safe,
        "false_positive_overlap_harm_chunks": overlap_harm,
        "head_tail_median_improvement_vs_baseline_ratio": _median(head_imps),
        "overlap_median_improvement_vs_baseline_ratio": _median(overlap_imps),
        "diagnostic_selects_chunk08_only": selected_chunks == [8],
        "canary_rule_gate_pass": False,
        "method_gate_claimed": False,
        "note": "Diagnostic-only: canary rule coverage is too narrow for method promotion.",
    }


def _rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _audit_rule(
            "selected_write_low_support_ratio_ge_0p50",
            "Select chunks where >=50% of selected TTT runtime write mass is low-support.",
            rows,
            lambda row: (row.get("selected_low_support_given_selected_runtime") or 0.0) >= 0.5,
        ),
        _audit_rule(
            "support_mean_le_0p60",
            "Select chunks whose semantic/geometry support map mean is <=0.60.",
            rows,
            lambda row: (row.get("support_score_mean") is not None and row["support_score_mean"] <= 0.60),
        ),
        _audit_rule(
            "selected_low_support_mass_gt0",
            "Select chunks with any selected-write low-support mass.",
            rows,
            lambda row: (row.get("selected_low_support_mass") or 0.0) > 0.0,
        ),
        _audit_rule(
            "postdelta_action_top_low_support_frac_ge_0p25",
            "Select chunks where top action post-delta regions are at least 25% low-support.",
            rows,
            lambda row: (row.get("action_top_low_support_frac") or 0.0) >= 0.25,
        ),
        _audit_rule(
            "postdelta_top_high_Dq_frac_ge_0p75",
            "Select chunks where top action post-delta regions are at least 75% high-Dq.",
            rows,
            lambda row: (row.get("action_top_high_Dq_frac") or 0.0) >= 0.75,
        ),
    ]


def _summary(args: argparse.Namespace, rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    visual_audit = _read_json(args.rediscovery_root / "visual_integrity_audit.json")
    rediscovery_summary = _read_json(args.rediscovery_root / "rediscovery_summary.json")
    local_explained_chunks = [row["chunk"] for row in rows if row.get("selected_write_low_support_explains")]
    helpful_safe_chunks = [row["chunk"] for row in rows if row.get("qscale_outcome_class") == "helpful_overlap_safe"]
    overlap_harm_chunks = [row["chunk"] for row in rows if row.get("qscale_outcome_class") == "head_tail_only_overlap_harm"]
    empty_semantic_bridge_chunks = [
        row["chunk"]
        for row in rows
        if row.get("bridge_skip_frame_semantic") and row.get("selected_geometry_ttt_semantic_rows_count") == 0
    ]

    selected_low_rule = next(rule for rule in rules if rule["rule"] == "selected_write_low_support_ratio_ge_0p50")
    high_dq_rule = next(rule for rule in rules if rule["rule"] == "postdelta_top_high_Dq_frac_ge_0p75")

    core_blocker = (
        "Selected-write low-support evidence cleanly explains chunk08, but coverage is chunk08-only. "
        "Broader post-delta high-Dq evidence selects non-helpful and overlap-harm chunks, especially chunk10/chunk12."
    )
    if not local_explained_chunks:
        core_blocker = (
            "The joined artifacts do not provide a selected-write low-support explanation for the canary chunks."
        )

    return {
        "schema": "acl2_v80_geometry_error_ttt_semantic_explanation_audit_v1",
        "chunks": [row["chunk"] for row in rows],
        "row_count": len(rows),
        "diagnostic_only": True,
        "geometry_bridge_root": args.geometry_bridge_root,
        "rediscovery_root": args.rediscovery_root,
        "postdelta_root": args.postdelta_root,
        "decisions_csv": args.decisions_csv,
        "visual_audit_gate_pass": bool(visual_audit.get("gate_pass")),
        "visual_audit_reason": visual_audit.get("reason"),
        "rediscovery_status": rediscovery_summary.get("status"),
        "rediscovery_decision": rediscovery_summary.get("decision"),
        "semantic_bridge_rows_empty_due_skip_frame_semantic_chunks": empty_semantic_bridge_chunks,
        "local_semantic_explains_ttt_low_support_write": bool(local_explained_chunks),
        "local_explained_chunks": local_explained_chunks,
        "helpful_overlap_safe_chunks": helpful_safe_chunks,
        "head_tail_only_overlap_harm_chunks": overlap_harm_chunks,
        "selected_write_low_support_rule_selected_chunks": selected_low_rule["selected_chunks"],
        "postdelta_high_dq_rule_selected_chunks": high_dq_rule["selected_chunks"],
        "postdelta_high_dq_false_positive_overlap_harm_chunks": high_dq_rule["false_positive_overlap_harm_chunks"],
        "core_blocker": core_blocker,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "next_action": (
            "Do not launch a broad runtime from this evidence. The only clean signal is a chunk08-local "
            "OUT3/MEMIX no-persistent hypothesis; it still needs same-mass random and geometry-only controls "
            "and more held-out coverage before any promotion."
        ),
    }


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 Geometry Error / TTT / Semantic Explanation Audit",
        "",
        f"- diagnostic_only: `{summary['diagnostic_only']}`",
        f"- local_semantic_explains_ttt_low_support_write: `{summary['local_semantic_explains_ttt_low_support_write']}`",
        f"- local_explained_chunks: `{summary['local_explained_chunks']}`",
        f"- helpful_overlap_safe_chunks: `{summary['helpful_overlap_safe_chunks']}`",
        f"- head_tail_only_overlap_harm_chunks: `{summary['head_tail_only_overlap_harm_chunks']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        "",
        "## Core Blocker",
        "",
        summary["core_blocker"],
        "",
        "## Rule Audit",
        "",
        "| rule | selected_chunks | helpful_safe | false_positive_overlap_harm | method_gate_claimed |",
        "|---|---|---|---|---|",
    ]
    for rule in rules:
        lines.append(
            f"| {rule['rule']} | {rule['selected_chunks']} | {rule['helpful_safe_chunks']} | "
            f"{rule['false_positive_overlap_harm_chunks']} | {rule['method_gate_claimed']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Chunk Rows",
            "",
            "| chunk | selected_write_low_support_explains | support_mean | selected_low_support_ratio | qscale_outcome | max_delta_frame | max_delta_vs_baseline_m |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    str(row["selected_write_low_support_explains"]),
                    str(row.get("support_score_mean")),
                    str(row.get("selected_low_support_given_selected_runtime")),
                    str(row.get("qscale_outcome_class")),
                    str(row.get("max_delta_frame")),
                    str(row.get("delta_error_vs_baseline_m_max")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    rows = _joined_rows(args, chunks)
    rules = _rule_rows(rows)
    summary = _summary(args, rows, rules)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "geometry_error_ttt_semantic_explanation_rows.csv", rows)
    _write_csv(args.out_dir / "geometry_error_ttt_semantic_rule_audit.csv", rules)
    _write_json(args.out_dir / "geometry_error_ttt_semantic_explanation_summary.json", summary)
    _write_report(args.out_dir / "geometry_error_ttt_semantic_explanation_report.md", summary, rows, rules)
    print(json.dumps(_jsonable({"out_dir": str(args.out_dir), **summary}), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
