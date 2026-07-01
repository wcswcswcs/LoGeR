#!/usr/bin/env python3
"""Audit TTSA-style temporal-spatial carriers for v80 seq01 canary chunks.

This is an offline diagnostic for the plan's TTSA3R direction:
temporal state evolution plus spatial observation quality jointly decides
whether a memory update/action is safe. It reads existing hmc_state traces and
landed canary outcomes. It does not run a new trajectory experiment and does
not infer success for untested rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_CANARY_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
DEFAULT_DECISIONS = DEFAULT_CANARY_ROOT / "thingstuff_radio_qscale_ref055_canary5_decisions.csv"
DEFAULT_REDISCOVERY_ROWS = (
    REPORT_ROOT
    / "phase10_seq01_error_ttt_semantic_alignment_rediscovery_20260622_2030"
    / "canary_error_ttt_semantic_alignment_rows.csv"
)
DEFAULT_RADIO_ROWS = (
    REPORT_ROOT
    / "phase10_seq01_radio_guard_extra_chunk_coverage_20260622_2140"
    / "radio_guard_extra_chunk_rows.csv"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_ttsa_temporal_spatial_carrier_audit_20260622_2205"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", type=Path, default=DEFAULT_CANARY_ROOT)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--rediscovery-rows", type=Path, default=DEFAULT_REDISCOVERY_ROWS)
    parser.add_argument("--radio-rows", type=Path, default=DEFAULT_RADIO_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="6,7,8,10,12")
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _decision_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "head_tail_phaseE_chunk_pass": _bool_text(row.get("head_tail_phaseE_chunk_pass")),
            "overlap_phaseE_chunk_pass": _bool_text(row.get("overlap_phaseE_chunk_pass")),
            "head_tail_beats_controls": _bool_text(row.get("head_tail_beats_controls")),
            "overlap_beats_controls": _bool_text(row.get("overlap_beats_controls")),
            "head_tail_improvement_vs_baseline_ratio": _safe_float(row.get("head_tail_improvement_vs_baseline_ratio")),
            "overlap_improvement_vs_baseline_ratio": _safe_float(row.get("overlap_improvement_vs_baseline_ratio")),
        }
    return out


def _rediscovery_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "semantic_support_score_mean": _safe_float(row.get("support_score_mean")),
            "semantic_low_proxy_1_minus_mean": _safe_float(row.get("support_low_proxy_1_minus_mean")),
            "selected_runtime_mass": _safe_float(row.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(row.get("selected_low_support_mass")),
            "selected_low_support_given_selected_runtime": _safe_float(
                row.get("selected_low_support_given_selected_runtime")
            ),
            "selected_low_support_enrichment_vs_global": _safe_float(
                row.get("selected_low_support_enrichment_vs_global")
            ),
            "semantic_ttt_interpretation": row.get("interpretation"),
        }
    return out


def _radio_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "radio_lowtrust_score_mean": _safe_float(row.get("radio_lowtrust_score_mean")),
            "radio_boundary_score_mean": _safe_float(row.get("object_boundary_score_mean")),
            "radio_sky_context_score_mean": _safe_float(row.get("radio_sky_context_score_mean")),
            "radio_temporal_stability_mean": _safe_float(row.get("temporal_stability_mean")),
            "radio_dynamic_score_mean": _safe_float(row.get("radio_dynamic_score_mean")),
        }
    return out


def _chunk_dir(root: Path, chunk: int) -> Path:
    return root / f"chunk{chunk:02d}" / "thingstuff_radio_qscale"


def _pick_state_rows(path: Path, chunk: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    current = None
    previous = None
    for row in rows:
        action_idx = _safe_int(row.get("prior_semantic_action_chunk_idx"))
        if action_idx == chunk:
            current = row
        elif action_idx == chunk - 1:
            previous = row
    if current is None and rows:
        current = rows[-1]
    if previous is None and len(rows) >= 2:
        previous = rows[-2]
    return current, previous, rows


def _state_features(root: Path, chunk: int) -> dict[str, Any]:
    run_dir = _chunk_dir(root, chunk)
    current, previous, rows = _pick_state_rows(run_dir / "hmc_state_hash.jsonl", chunk)
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "hmc_state_hash": str(run_dir / "hmc_state_hash.jsonl"),
        "hmc_state_row_count": len(rows),
        "hmc_state_current_found": current is not None,
        "hmc_state_previous_found": previous is not None,
    }
    if current is None:
        return out

    current_keys = {
        "current_action_idx": "prior_semantic_action_chunk_idx",
        "current_start_frame": "start_frame",
        "current_end_frame": "end_frame",
        "memory_ttt_mean_rel_diff": "memory_ttt_mean_rel_diff",
        "memory_ttt_max_rel_diff": "memory_ttt_max_rel_diff",
        "memory_ttt_w0_mean_rel_diff": "memory_ttt_w0_mean_rel_diff",
        "memory_ttt_w1_mean_rel_diff": "memory_ttt_w1_mean_rel_diff",
        "memory_ttt_w2_mean_rel_diff": "memory_ttt_w2_mean_rel_diff",
        "spatial_D_patch_mean": "prior_mean_D_patch",
        "spatial_D_patch_q90": "prior_q90_D_patch",
        "spatial_D_tok_q90": "prior_q90_D_tok",
        "spatial_dynamic_mass_D_gt_050": "prior_dynamic_mass_D_gt_050",
        "spatial_fragmentation": "prior_fragmentation",
        "spatial_anchor_collision": "prior_anchor_collision",
        "cue_quality_pass": "prior_cue_quality_pass",
        "cue_quality_mass_pass": "prior_cue_quality_mass_pass",
        "cue_quality_frag_pass": "prior_cue_quality_frag_pass",
        "cue_quality_anchor_pass": "prior_cue_quality_anchor_pass",
    }
    for out_key, src_key in current_keys.items():
        out[out_key] = current.get(src_key)

    if previous is not None:
        prev_mean = _safe_float(previous.get("memory_ttt_mean_rel_diff"))
        cur_mean = _safe_float(current.get("memory_ttt_mean_rel_diff"))
        out["previous_action_idx"] = previous.get("prior_semantic_action_chunk_idx")
        out["previous_memory_ttt_mean_rel_diff"] = prev_mean
        out["temporal_memory_rel_drop_from_previous"] = (
            prev_mean - cur_mean if prev_mean is not None and cur_mean is not None else None
        )
        out["temporal_memory_rel_ratio_current_over_previous"] = (
            cur_mean / prev_mean if prev_mean not in (None, 0.0) and cur_mean is not None else None
        )

    mem_mean = _safe_float(out.get("memory_ttt_mean_rel_diff"))
    spatial_d = _safe_float(out.get("spatial_D_patch_mean"))
    frag = _safe_float(out.get("spatial_fragmentation"))
    anchor = _safe_float(out.get("spatial_anchor_collision"))
    out["ttsa_mem_over_spatial_D"] = mem_mean / spatial_d if mem_mean is not None and spatial_d not in (None, 0.0) else None
    out["ttsa_mem_over_frag_plus_anchor"] = (
        mem_mean / (frag + anchor) if mem_mean is not None and frag is not None and anchor is not None and frag + anchor > 0 else None
    )
    out["ttsa_spatial_quality_proxy"] = (
        (spatial_d or 0.0) - (frag or 0.0) - (anchor or 0.0)
        if spatial_d is not None and frag is not None and anchor is not None
        else None
    )
    return out


def _classify(row: dict[str, Any]) -> str:
    head = bool(row.get("head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "qscale_helpful_overlap_safe"
    if head and not overlap:
        return "qscale_head_tail_only_overlap_harm"
    return "qscale_not_helpful"


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _safe_float(row.get(key))
    return default if value is None else value


def _median(values: list[float]) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _rules() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    return [
        (
            "ttsa_memory_rel_diff_le_0p0068",
            "Select qscale when current TTT temporal rel-diff is very low.",
            lambda r: _f(r, "memory_ttt_mean_rel_diff", 1.0) <= 0.0068,
        ),
        (
            "ttsa_memory_rel_drop_ge_0p0010",
            "Select qscale when current memory update is much lower than previous chunk.",
            lambda r: _f(r, "temporal_memory_rel_drop_from_previous", -1.0) >= 0.0010,
        ),
        (
            "ttsa_high_spatialD_low_fragmentation",
            "Select qscale when spatial dynamic proxy is high and fragmentation is low.",
            lambda r: _f(r, "spatial_D_patch_mean") >= 0.70 and _f(r, "spatial_fragmentation", 1.0) <= 0.0035,
        ),
        (
            "ttsa_high_spatialD_low_semantic_support",
            "Select qscale when spatial dynamic proxy is high and semantic support is low.",
            lambda r: _f(r, "spatial_D_patch_mean") >= 0.70 and _f(r, "semantic_support_score_mean", 1.0) <= 0.60,
        ),
        (
            "ttsa_low_memdiff_low_semantic_support",
            "Select qscale when temporal rel-diff is low and semantic support is low.",
            lambda r: _f(r, "memory_ttt_mean_rel_diff", 1.0) <= 0.0068
            and _f(r, "semantic_support_score_mean", 1.0) <= 0.60,
        ),
        (
            "ttsa_spatial_quality_proxy_ge_0p65",
            "Select qscale when D-fragmentation-anchor spatial quality proxy is high.",
            lambda r: _f(r, "ttsa_spatial_quality_proxy") >= 0.65,
        ),
        (
            "ttsa_mem_over_frag_anchor_le_0p15",
            "Select qscale when temporal update per frag+anchor risk is low.",
            lambda r: _f(r, "ttsa_mem_over_frag_plus_anchor", 1.0) <= 0.15,
        ),
        (
            "ttsa_radio_lowtrust_temporal_stable",
            "Select qscale when RADIO lowtrust is low and temporal rel-diff is low.",
            lambda r: _f(r, "radio_lowtrust_score_mean", 1.0) <= 0.31
            and _f(r, "memory_ttt_mean_rel_diff", 1.0) <= 0.0068,
        ),
        (
            "ttsa_selected_low_support_temporal_stable",
            "Select qscale when selected write hits low support and temporal rel-diff is low.",
            lambda r: _f(r, "selected_low_support_mass") > 0
            and _f(r, "memory_ttt_mean_rel_diff", 1.0) <= 0.0068,
        ),
    ]


def _rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule_name, description, predicate in _rules():
        selected = [row for row in rows if predicate(row)]
        selected_chunks = [int(row["chunk"]) for row in selected]
        head_tail_pass_chunks = [
            int(row["chunk"]) for row in selected if bool(row.get("head_tail_phaseE_chunk_pass"))
        ]
        overlap_pass_chunks = [int(row["chunk"]) for row in selected if bool(row.get("overlap_phaseE_chunk_pass"))]
        overlap_harm_chunks = [
            int(row["chunk"])
            for row in selected
            if (row.get("overlap_improvement_vs_baseline_ratio") is not None)
            and float(row["overlap_improvement_vs_baseline_ratio"]) < 0.0
        ]
        helpful_safe_chunks = [
            int(row["chunk"]) for row in selected if row.get("qscale_outcome_class") == "qscale_helpful_overlap_safe"
        ]
        false_positive_chunks = [
            int(row["chunk"])
            for row in selected
            if row.get("qscale_outcome_class") == "qscale_head_tail_only_overlap_harm"
        ]
        head_values: list[float] = []
        overlap_values: list[float] = []
        for row in rows:
            if int(row["chunk"]) in selected_chunks:
                head_values.append(float(row.get("head_tail_improvement_vs_baseline_ratio") or 0.0))
                overlap_values.append(float(row.get("overlap_improvement_vs_baseline_ratio") or 0.0))
            else:
                head_values.append(0.0)
                overlap_values.append(0.0)
        head_median = _median(head_values)
        overlap_median = _median(overlap_values)
        canary_rule_gate_pass = bool(
            (len(head_tail_pass_chunks) >= 4 and head_median >= 0.05)
            or (len(overlap_pass_chunks) >= 4 and overlap_median >= 0.05)
        )
        out.append(
            {
                "rule": rule_name,
                "description": description,
                "selected_chunks": selected_chunks,
                "selected_count": len(selected_chunks),
                "helpful_safe_chunks": helpful_safe_chunks,
                "false_positive_overlap_harm_chunks": false_positive_chunks,
                "head_tail_pass_chunks": head_tail_pass_chunks,
                "overlap_pass_chunks": overlap_pass_chunks,
                "overlap_harm_chunks": overlap_harm_chunks,
                "head_tail_median_improvement_with_native_fallback": head_median,
                "overlap_median_improvement_with_native_fallback": overlap_median,
                "diagnostic_separates_chunk08_from_false_positive": helpful_safe_chunks == [8] and not false_positive_chunks,
                "canary_rule_gate_pass": canary_rule_gate_pass,
                "method_gate_claimed": False,
            }
        )
    return out


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 TTSA Temporal-Spatial Carrier Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- core_blocker: {summary['core_blocker']}",
        "",
        "## Chunk Features",
        "",
        "| chunk | class | mem_rel | mem_drop | D_mean | frag | anchor | support | radio_lowtrust | selected_low_support |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    str(row["qscale_outcome_class"]),
                    f"{_f(row, 'memory_ttt_mean_rel_diff'):.6f}",
                    f"{_f(row, 'temporal_memory_rel_drop_from_previous'):.6f}",
                    f"{_f(row, 'spatial_D_patch_mean'):.6f}",
                    f"{_f(row, 'spatial_fragmentation'):.6f}",
                    f"{_f(row, 'spatial_anchor_collision'):.6f}",
                    f"{_f(row, 'semantic_support_score_mean'):.6f}",
                    f"{_f(row, 'radio_lowtrust_score_mean'):.6f}",
                    f"{_f(row, 'selected_low_support_mass'):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rule Audit", ""])
    lines.append("| rule | selected_chunks | separates_chunk08 | gate_pass | overlap_harm |")
    lines.append("|---|---|---:|---:|---|")
    for row in rule_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rule"]),
                    json.dumps(row["selected_chunks"], ensure_ascii=False),
                    str(row["diagnostic_separates_chunk08_from_false_positive"]).lower(),
                    str(row["canary_rule_gate_pass"]).lower(),
                    json.dumps(row["overlap_harm_chunks"], ensure_ascii=False),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    decisions = _decision_by_chunk(args.decisions_csv)
    rediscovery = _rediscovery_by_chunk(args.rediscovery_rows)
    radio = _radio_by_chunk(args.radio_rows)

    rows: list[dict[str, Any]] = []
    missing_chunks: list[int] = []
    for chunk in chunks:
        if chunk not in decisions:
            missing_chunks.append(chunk)
            continue
        row: dict[str, Any] = {"chunk": chunk}
        row.update(decisions[chunk])
        row.update(rediscovery.get(chunk, {}))
        row.update(radio.get(chunk, {}))
        row.update(_state_features(args.canary_root, chunk))
        row["qscale_outcome_class"] = _classify(row)
        rows.append(row)

    rules = _rule_rows(rows)
    gate_pass_rules = [row["rule"] for row in rules if bool(row["canary_rule_gate_pass"])]
    separator_rules = [row["rule"] for row in rules if bool(row["diagnostic_separates_chunk08_from_false_positive"])]
    harm_selecting_rules = [row["rule"] for row in rules if row["false_positive_overlap_harm_chunks"]]

    if gate_pass_rules:
        status = "unexpected_ttsa_gate_pass_requires_runtime_review"
    elif separator_rules:
        status = "ttsa_chunk08_local_diagnostic_only"
    else:
        status = "no_ttsa_temporal_spatial_separability"

    summary = {
        "schema": "acl2_v80_ttsa_temporal_spatial_carrier_audit_v1",
        "status": status,
        "diagnostic_only": True,
        "chunks": chunks,
        "missing_chunks": missing_chunks,
        "row_count": len(rows),
        "rule_count": len(rules),
        "diagnostic_separator_rules": separator_rules,
        "harm_selecting_rules": harm_selecting_rules,
        "deployable_gate_pass_rules": gate_pass_rules,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "core_blocker": (
            "Existing temporal state evolution and spatial quality fields can produce chunk08-local diagnostics, "
            "but rules that broaden coverage either select chunk10/chunk12 overlap-harm false positives or leave "
            "native-fallback medians below the PhaseE gate."
        ),
        "next_action": (
            "Do not launch a TTSA runtime from these scalar trace fields alone. A new carrier would need stronger "
            "per-token/per-region temporal-spatial evidence or held-out coverage, not another threshold over "
            "memory_ttt_rel_diff, D_patch, support, or RADIO lowtrust."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "ttsa_temporal_spatial_carrier_summary.json", summary)
    _write_csv(args.out_dir / "ttsa_temporal_spatial_feature_rows.csv", rows)
    _write_csv(args.out_dir / "ttsa_temporal_spatial_rule_audit.csv", rules)
    _write_report(args.out_dir / "ttsa_temporal_spatial_carrier_report.md", summary, rows, rules)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
