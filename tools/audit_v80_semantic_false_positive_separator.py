#!/usr/bin/env python3
"""Audit non-GT separators for v80 semantic/TTT false positives.

This is an offline diagnostic over existing seq01 canary artifacts. It checks
whether support/write fields can preserve the chunk08 local TTT explanation
while rejecting the chunk10/chunk12 overlap-harm false positives.
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
DEFAULT_GEOM_TTT_ROOT = REPORT_ROOT / "phase10_seq01_geometry_error_ttt_semantic_explanation_20260622_2245"
DEFAULT_POSTDELTA_ROOT = REPORT_ROOT / "phase10_seq01_ttt_postdelta_region_carrier_audit_20260622_2220"
DEFAULT_SUPPORT_ROOT = REPORT_ROOT / "phase9_seq01_ref055_v80_error_semantic_support_maps"
DEFAULT_SELECTED_WRITE_ROOT = REPORT_ROOT / "phase9_seq01_ref055_v80_selected_write_support_maps"
DEFAULT_SELECTED_WRITE_EXT_ROOT = REPORT_ROOT / "phase9_seq01_ref055_v80_selected_write_support_maps_canary_ext"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_semantic_false_positive_separator_20260622_2153"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-ttt-root", type=Path, default=DEFAULT_GEOM_TTT_ROOT)
    parser.add_argument("--postdelta-root", type=Path, default=DEFAULT_POSTDELTA_ROOT)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT_ROOT)
    parser.add_argument("--selected-write-root", type=Path, default=DEFAULT_SELECTED_WRITE_ROOT)
    parser.add_argument("--selected-write-ext-root", type=Path, default=DEFAULT_SELECTED_WRITE_EXT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="6,7,8,10,12")
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _index_rows_by_chunk(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is not None:
            out[chunk] = row
    return out


def _selected_summary_path(args: argparse.Namespace, chunk: int) -> Path:
    primary = args.selected_write_root / f"chunk_{chunk:03d}_selected_write_support_map_summary.json"
    if primary.exists():
        return primary
    return args.selected_write_ext_root / f"chunk_{chunk:03d}_selected_write_support_map_summary.json"


def _outcome(row: dict[str, Any]) -> str:
    head = bool(row.get("head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "helpful_overlap_safe"
    if head and not overlap:
        return "head_tail_only_overlap_harm"
    return "not_helpful"


def _support_frame_stats(summary: dict[str, Any]) -> dict[str, Any]:
    frame_rows = summary.get("frame_rows") or []
    if not frame_rows:
        return {}
    support_means = [_safe_float(row.get("support_mean")) for row in frame_rows]
    risk_ratios = [_safe_float(row.get("risk_patch_ratio")) for row in frame_rows]
    stable_ratios = [_safe_float(row.get("stable_patch_ratio")) for row in frame_rows]
    support_vals = [value for value in support_means if value is not None]
    risk_vals = [value for value in risk_ratios if value is not None]
    stable_vals = [value for value in stable_ratios if value is not None]
    return {
        "frame_support_mean_min": min(support_vals) if support_vals else None,
        "frame_support_mean_max": max(support_vals) if support_vals else None,
        "frame_risk_patch_ratio_mean": sum(risk_vals) / len(risk_vals) if risk_vals else None,
        "frame_stable_patch_ratio_mean": sum(stable_vals) / len(stable_vals) if stable_vals else None,
    }


def _build_rows(args: argparse.Namespace, chunks: list[int]) -> list[dict[str, Any]]:
    geom_rows = _index_rows_by_chunk(args.geometry_ttt_root / "geometry_error_ttt_semantic_explanation_rows.csv")
    post_rows = _index_rows_by_chunk(args.postdelta_root / "ttt_postdelta_region_feature_rows.csv")
    rows: list[dict[str, Any]] = []

    for chunk in chunks:
        geom = geom_rows.get(chunk, {})
        post = post_rows.get(chunk, {})
        support_summary_path = args.support_root / f"chunk_{chunk:03d}_support_map_summary.json"
        selected_summary_path = _selected_summary_path(args, chunk)
        support_summary = _read_json(support_summary_path)
        selected_summary = _read_json(selected_summary_path)

        row: dict[str, Any] = {
            "chunk": chunk,
            "support_summary_path": support_summary_path,
            "support_summary_exists": support_summary_path.exists(),
            "selected_write_summary_path": selected_summary_path,
            "selected_write_summary_exists": selected_summary_path.exists(),
            "qscale_outcome_class": geom.get("qscale_outcome_class") or post.get("qscale_outcome_class"),
            "head_tail_phaseE_chunk_pass": _bool_text(geom.get("head_tail_phaseE_chunk_pass") or post.get("head_tail_phaseE_chunk_pass")),
            "overlap_phaseE_chunk_pass": _bool_text(geom.get("overlap_phaseE_chunk_pass") or post.get("overlap_phaseE_chunk_pass")),
            "head_tail_improvement_vs_baseline_ratio": _safe_float(
                geom.get("head_tail_improvement_vs_baseline_ratio") or post.get("head_tail_improvement_vs_baseline_ratio")
            ),
            "overlap_improvement_vs_baseline_ratio": _safe_float(
                geom.get("overlap_improvement_vs_baseline_ratio") or post.get("overlap_improvement_vs_baseline_ratio")
            ),
            "delta_error_vs_baseline_m_mean": _safe_float(geom.get("delta_error_vs_baseline_m_mean")),
            "delta_error_vs_baseline_m_max": _safe_float(geom.get("delta_error_vs_baseline_m_max")),
            "positive_delta_vs_baseline_frac": _safe_float(geom.get("positive_delta_vs_baseline_frac")),
            "support_score_mean": _safe_float(geom.get("support_score_mean") or post.get("support_score_mean")),
            "support_low_proxy_1_minus_mean": _safe_float(
                geom.get("support_low_proxy_1_minus_mean") or post.get("support_low_proxy_1_minus_mean")
            ),
            "global_low_support_frac": _safe_float(post.get("global_low_support_frac")),
            "global_high_Dq_frac": _safe_float(post.get("global_high_Dq_frac")),
            "action_top_high_Dq_frac": _safe_float(post.get("action_top_high_Dq_frac")),
            "action_top_low_support_frac": _safe_float(post.get("action_top_low_support_frac")),
            "action_top_high_Ds_frac": _safe_float(post.get("action_top_high_Ds_frac")),
            "action_top_low_support_enrichment": _safe_float(geom.get("action_top_low_support_enrichment")),
            "selected_runtime_mass": _safe_float(geom.get("selected_runtime_mass") or selected_summary.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(
                geom.get("selected_low_support_mass") or selected_summary.get("selected_low_support_mass")
            ),
            "selected_low_support_given_selected_runtime": _safe_float(
                geom.get("selected_low_support_given_selected_runtime")
                or selected_summary.get("selected_low_support_given_selected_runtime")
            ),
            "selected_low_support_enrichment_vs_global": _safe_float(geom.get("selected_low_support_enrichment_vs_global")),
            "selected_visual_ratio": _safe_float(selected_summary.get("selected_visual_ratio")),
            "runtime_low_support_ratio": _safe_float(selected_summary.get("runtime_low_support_ratio")),
            "support_summary_score_q10": _safe_float(support_summary.get("score_q10")),
            "support_summary_score_q50": _safe_float(support_summary.get("score_q50")),
            "support_summary_score_q90": _safe_float(support_summary.get("score_q90")),
            **_support_frame_stats(support_summary),
        }
        row["outcome_class"] = _outcome(row)
        row["is_helpful_overlap_safe"] = row["outcome_class"] == "helpful_overlap_safe"
        row["is_overlap_harm_false_positive"] = row["outcome_class"] == "head_tail_only_overlap_harm"
        row["selected_write_low_support_flag"] = (
            (row.get("selected_runtime_mass") or 0.0) > 0.0
            and (row.get("selected_low_support_mass") or 0.0) > 0.0
            and (row.get("selected_low_support_given_selected_runtime") or 0.0) >= 0.5
        )
        row["support_low_mean_flag"] = (row.get("support_score_mean") or 1.0) < 0.6
        row["high_dq_top_region_flag"] = (row.get("action_top_high_Dq_frac") or 0.0) >= 0.9
        row["high_dq_high_support_false_positive_risk"] = (
            row["high_dq_top_region_flag"]
            and (row.get("support_score_mean") or 0.0) >= 0.85
            and not row["selected_write_low_support_flag"]
        )
        rows.append(row)
    return rows


def _evaluate_rule(rows: list[dict[str, Any]], name: str, selected_key: str, note: str) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get(selected_key))]
    helpful_selected = [row for row in selected if row.get("is_helpful_overlap_safe")]
    false_positive_selected = [row for row in selected if row.get("is_overlap_harm_false_positive")]
    helpful_total = [row for row in rows if row.get("is_helpful_overlap_safe")]
    false_positive_total = [row for row in rows if row.get("is_overlap_harm_false_positive")]
    return {
        "rule": name,
        "selected_key": selected_key,
        "selected_chunks": [row["chunk"] for row in selected],
        "helpful_selected_chunks": [row["chunk"] for row in helpful_selected],
        "false_positive_selected_chunks": [row["chunk"] for row in false_positive_selected],
        "helpful_total_chunks": [row["chunk"] for row in helpful_total],
        "false_positive_total_chunks": [row["chunk"] for row in false_positive_total],
        "helpful_recall": len(helpful_selected) / len(helpful_total) if helpful_total else None,
        "false_positive_rejection": (
            1.0 - len(false_positive_selected) / len(false_positive_total) if false_positive_total else None
        ),
        "selected_count": len(selected),
        "diagnostic_pass": bool(helpful_selected) and not false_positive_selected,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "note": note,
    }


def _write_report(path: Path, summary: dict[str, Any], rules: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 Semantic False-Positive Separator Audit",
        "",
        "This is a diagnostic-only audit over existing artifacts.",
        "",
        "## Decision",
        "",
        f"- diagnostic_separator_found: `{summary['diagnostic_separator_found']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        "",
        "## Core Finding",
        "",
        summary["core_finding"],
        "",
        "## Rule Audit",
        "",
    ]
    for rule in rules:
        lines.extend(
            [
                f"### {rule['rule']}",
                "",
                f"- selected_chunks: `{rule['selected_chunks']}`",
                f"- helpful_selected_chunks: `{rule['helpful_selected_chunks']}`",
                f"- false_positive_selected_chunks: `{rule['false_positive_selected_chunks']}`",
                f"- diagnostic_pass: `{rule['diagnostic_pass']}`",
                f"- note: {rule['note']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    rows = _build_rows(args, chunks)
    rules = [
        _evaluate_rule(
            rows,
            "selected_write_low_support_separator",
            "selected_write_low_support_flag",
            "Preserves the chunk08 local selected-write explanation and rejects chunk10/chunk12 in this canary set.",
        ),
        _evaluate_rule(
            rows,
            "support_low_mean_separator",
            "support_low_mean_flag",
            "A coarser support-mean proxy for the same chunk08-local signal.",
        ),
        _evaluate_rule(
            rows,
            "high_dq_top_region_rule",
            "high_dq_top_region_flag",
            "Shows why broad post-delta high-Dq rules over-select non-helpful and overlap-harm chunks.",
        ),
        _evaluate_rule(
            rows,
            "high_dq_high_support_false_positive_risk",
            "high_dq_high_support_false_positive_risk",
            "Identifies high-Dq/high-support chunks as semantic false-positive risk rather than low-support TTT errors.",
        ),
    ]

    selected_rule = next(rule for rule in rules if rule["rule"] == "selected_write_low_support_separator")
    high_dq_rule = next(rule for rule in rules if rule["rule"] == "high_dq_top_region_rule")
    diagnostic_separator_found = bool(selected_rule["diagnostic_pass"])
    summary = {
        "schema": "acl2_v80_semantic_false_positive_separator_audit_v1",
        "chunks": chunks,
        "row_count": len(rows),
        "diagnostic_only": True,
        "diagnostic_separator_found": diagnostic_separator_found,
        "best_separator_rule": selected_rule["rule"] if diagnostic_separator_found else None,
        "best_separator_selected_chunks": selected_rule["selected_chunks"] if diagnostic_separator_found else [],
        "overlap_harm_false_positive_chunks": selected_rule["false_positive_total_chunks"],
        "high_dq_rule_selected_chunks": high_dq_rule["selected_chunks"],
        "high_dq_false_positive_selected_chunks": high_dq_rule["false_positive_selected_chunks"],
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "core_finding": (
            "The non-GT low-support selected-write signal separates chunk08 from chunk10/chunk12 in the current "
            "seq01 canary evidence, while broad high-Dq post-delta selection still hits overlap-harm false positives. "
            "This is a useful failure localization, not a promoted runtime method, because prior selected-write "
            "no-persistent/veto smokes did not pass the control-separated geometry gate."
        ),
        "next_action": (
            "Do not repeat broad high-Dq or selected-write veto sweeps. If continuing algorithmically, the next "
            "distinct candidate must combine the low-support selected-write separator with a new actuator or a "
            "held-out multi-case carrier test; this audit alone does not allow runtime promotion."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "semantic_false_positive_separator_rows.csv", rows)
    _write_csv(args.out_dir / "semantic_false_positive_separator_rule_audit.csv", rules)
    _write_json(args.out_dir / "semantic_false_positive_separator_summary.json", summary)
    _write_report(args.out_dir / "semantic_false_positive_separator_report.md", summary, rules)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
