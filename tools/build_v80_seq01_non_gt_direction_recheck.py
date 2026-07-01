#!/usr/bin/env python3
"""Aggregate v80 seq01 non-GT direction and overlap-safety evidence.

This is diagnostic-only. It connects the Phase9 selected-write rediscovery,
boundary-scale oracle, non-GT thingstuff/RADIO qscale smokes, and the
overlap-guard minismoke. It does not claim a method gate pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_ALIGNMENT = REPORT_ROOT / "phase9_seq01_error_ttt_semantic_alignment_rediscovery" / (
    "canary_error_ttt_semantic_alignment_rows.csv"
)
DEFAULT_DIRECTION = REPORT_ROOT / "phase9_seq01_boundary_scale_direction_canary5_native" / (
    "canary5_boundary_scale_direction_summary.json"
)
DEFAULT_ORACLE = REPORT_ROOT / "phase9_seq01_boundary_scale_oracle_globalfuture_canary5" / (
    "mgf_oracle_globalfuture_canary5_gate_summary.json"
)
DEFAULT_QSCALE = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012" / (
    "thingstuff_radio_qscale_ref055_canary5_gate_summary.json"
)
DEFAULT_PURE_RADIO = REPORT_ROOT / "phase9_seq01_pure_radio_qscale_v80support_from_ref055_delta_canary5" / (
    "pure_radio_qscale_v80support_ref055delta_canary5_gate_summary.json"
)
DEFAULT_GUARD = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_ref055_overlap_guard_chunks010_012" / (
    "thingstuff_radio_qscale_ref055_overlap_guard_chunks10_12_gate_summary.json"
)
DEFAULT_GUARD_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_ref055_overlap_guard_chunks010_012"
DEFAULT_TIGHT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_ref055_overlap_guard_tightstate_chunks010_012" / (
    "thingstuff_radio_qscale_ref055_overlap_guard_tightstate_chunks10_12_gate_summary.json"
)
DEFAULT_TIGHT_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_ref055_overlap_guard_tightstate_chunks010_012"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase9_seq01_non_gt_direction_recheck"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-csv", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--direction-summary", type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--qscale-summary", type=Path, default=DEFAULT_QSCALE)
    parser.add_argument("--pure-radio-summary", type=Path, default=DEFAULT_PURE_RADIO)
    parser.add_argument("--guard-summary", type=Path, default=DEFAULT_GUARD)
    parser.add_argument("--guard-root", type=Path, default=DEFAULT_GUARD_ROOT)
    parser.add_argument("--tight-summary", type=Path, default=DEFAULT_TIGHT)
    parser.add_argument("--tight-root", type=Path, default=DEFAULT_TIGHT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _alignment_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = int(row["chunk"])
        out[chunk] = {
            "alignment_csv": str(path),
            "selected_runtime_mass": _safe_float(row.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(row.get("selected_low_support_mass")),
            "selected_low_support_enrichment_vs_global": _safe_float(
                row.get("selected_low_support_enrichment_vs_global")
            ),
            "selected_write_interpretation": row.get("interpretation"),
        }
    return out


def _direction_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_json(path).get("chunks", []):
        chunk = int(row["chunk"])
        out[chunk] = {
            "direction_signature": row.get("direction_signature"),
            "global_future_best_direction": row.get("global_future_from_boundary_rmse_m_best_direction"),
            "global_future_best_scale": row.get("global_future_from_boundary_rmse_m_best_scale"),
            "tail3_future_best_direction": row.get("tail3_to_future_from_boundary_sim3_rmse_m_best_direction"),
            "tail3_future_best_scale": row.get("tail3_to_future_from_boundary_sim3_rmse_m_best_scale"),
            "all_key_metrics_same_direction": row.get("all_key_metrics_same_direction"),
        }
    return out


def _decision_by_chunk(path: Path, prefix: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("chunk_decisions", []):
        chunk = int(row["chunk"])
        out[chunk] = {
            f"{prefix}_head_tail_beats_controls": row.get("head_tail_beats_controls"),
            f"{prefix}_head_tail_improvement_vs_baseline_ratio": _safe_float(
                row.get("head_tail_improvement_vs_baseline_ratio")
            ),
            f"{prefix}_head_tail_phaseE_chunk_pass": row.get("head_tail_phaseE_chunk_pass"),
            f"{prefix}_overlap_beats_controls": row.get("overlap_beats_controls"),
            f"{prefix}_overlap_improvement_vs_baseline_ratio": _safe_float(
                row.get("overlap_improvement_vs_baseline_ratio")
            ),
            f"{prefix}_overlap_phaseE_chunk_pass": row.get("overlap_phaseE_chunk_pass"),
        }
    summary = {
        f"{prefix}_summary_path": str(path),
        f"{prefix}_phaseE_gate_pass": payload.get("phaseE_gate_pass"),
        f"{prefix}_head_tail_pass_count": payload.get("head_tail_pass_count"),
        f"{prefix}_overlap_pass_count": payload.get("overlap_pass_count"),
        f"{prefix}_head_tail_median_improvement_vs_baseline_ratio": payload.get(
            "head_tail_median_improvement_vs_baseline_ratio"
        ),
        f"{prefix}_overlap_median_improvement_vs_baseline_ratio": payload.get(
            "overlap_median_improvement_vs_baseline_ratio"
        ),
    }
    return out, summary


def _trace_by_chunk(root: Path, prefix: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("chunk*/thingstuff_radio_qscale/merge_state_trace.jsonl")):
        chunk_text = path.parent.parent.name.replace("chunk", "")
        if not chunk_text.isdigit():
            continue
        chunk = int(chunk_text)
        selected: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if int(row.get("local_chunk_idx", -1)) == 1:
                    selected = row
        if selected is None:
            continue
        out[chunk] = {
            f"{prefix}_trace": str(path),
            f"{prefix}_trace_fit_reason": selected.get("semantic_merge_fit_reason"),
            f"{prefix}_trace_rejected": selected.get("semantic_merge_native_overlap_guard_rejected"),
            f"{prefix}_trace_reject_reason": selected.get("semantic_merge_reject_reason"),
            f"{prefix}_trace_native_overlap_residual": selected.get("semantic_merge_native_overlap_residual"),
            f"{prefix}_trace_final_overlap_residual": selected.get("semantic_merge_final_overlap_residual"),
            f"{prefix}_trace_scale": selected.get("semantic_merge_scale"),
            f"{prefix}_trace_qscale_factor": selected.get("semantic_merge_qscale_factor"),
            f"{prefix}_trace_qscale_observability": selected.get("semantic_merge_qscale_observability"),
            f"{prefix}_trace_scale_state_action": selected.get("online_scale_state_action"),
            f"{prefix}_trace_scale_state_input_scale": selected.get("online_scale_state_input_scale"),
            f"{prefix}_trace_scale_state_output_scale": selected.get("online_scale_state_output_scale"),
        }
    return out


def _interpret(row: dict[str, Any]) -> str:
    q_head = bool(row.get("qscale_head_tail_phaseE_chunk_pass"))
    q_overlap = bool(row.get("qscale_overlap_phaseE_chunk_pass"))
    guard_rejected = _safe_bool(row.get("guard_trace_rejected"))
    guard_head = bool(row.get("guard_head_tail_phaseE_chunk_pass"))
    guard_overlap_imp = _safe_float(row.get("guard_overlap_improvement_vs_baseline_ratio"))
    selected_low_mass = _safe_float(row.get("selected_low_support_mass")) or 0.0
    if selected_low_mass > 0:
        return "selected_write_low_support_explains_bad_write_not_direction"
    if q_head and not q_overlap and guard_rejected is True:
        return "overlap_guard_removes_head_tail_signal"
    if bool(row.get("tight_head_tail_phaseE_chunk_pass")) and (
        (_safe_float(row.get("tight_overlap_improvement_vs_baseline_ratio")) or 0.0) < 0.0
    ):
        return "tight_scale_state_reduces_but_does_not_fix_overlap_harm"
    if guard_head and guard_overlap_imp is not None and guard_overlap_imp < 0.0:
        return "local_overlap_guard_allows_but_downstream_overlap_worsens"
    if q_head and not q_overlap:
        return "partial_non_gt_head_tail_signal_overlap_unsafe"
    return "no_deployable_non_gt_direction_signal"


def _pass_chunks(rows: list[dict[str, Any]], key: str) -> list[int]:
    return [int(row["chunk"]) for row in rows if bool(row.get(key))]


def main() -> None:
    args = _parse_args()
    alignment = _alignment_by_chunk(args.alignment_csv)
    direction = _direction_by_chunk(args.direction_summary)
    oracle, oracle_summary = _decision_by_chunk(args.oracle_summary, "oracle")
    qscale, qscale_summary = _decision_by_chunk(args.qscale_summary, "qscale")
    pure_radio, pure_radio_summary = _decision_by_chunk(args.pure_radio_summary, "pure_radio")
    guard, guard_summary = _decision_by_chunk(args.guard_summary, "guard")
    tight, tight_summary = _decision_by_chunk(args.tight_summary, "tight")
    guard_trace = _trace_by_chunk(args.guard_root, "guard")
    tight_trace = _trace_by_chunk(args.tight_root, "tight")

    chunks = sorted(
        set(alignment)
        | set(direction)
        | set(oracle)
        | set(qscale)
        | set(pure_radio)
        | set(guard)
        | set(tight)
    )
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        row: dict[str, Any] = {"chunk": int(chunk)}
        row.update(alignment.get(chunk, {}))
        row.update(direction.get(chunk, {}))
        row.update(oracle.get(chunk, {}))
        row.update(qscale.get(chunk, {}))
        row.update(pure_radio.get(chunk, {}))
        row.update(guard.get(chunk, {}))
        row.update(tight.get(chunk, {}))
        row.update(guard_trace.get(chunk, {}))
        row.update(tight_trace.get(chunk, {}))
        row["interpretation"] = _interpret(row)
        rows.append(row)

    summary: dict[str, Any] = {
        "schema": "acl2_v80_seq01_non_gt_direction_recheck_v1",
        "status": "partial_non_gt_direction_signal_guard_no_gate",
        "v80_goal_achieved": False,
        "diagnostic_only": True,
        "chunks": chunks,
        **oracle_summary,
        **qscale_summary,
        **pure_radio_summary,
        **guard_summary,
        **tight_summary,
        "selected_write_low_support_chunks": [
            int(row["chunk"])
            for row in rows
            if (_safe_float(row.get("selected_low_support_mass")) or 0.0) > 0.0
        ],
        "qscale_head_tail_pass_chunks": _pass_chunks(rows, "qscale_head_tail_phaseE_chunk_pass"),
        "qscale_overlap_pass_chunks": _pass_chunks(rows, "qscale_overlap_phaseE_chunk_pass"),
        "guard_head_tail_pass_chunks": _pass_chunks(rows, "guard_head_tail_phaseE_chunk_pass"),
        "guard_overlap_pass_chunks": _pass_chunks(rows, "guard_overlap_phaseE_chunk_pass"),
        "tight_head_tail_pass_chunks": _pass_chunks(rows, "tight_head_tail_phaseE_chunk_pass"),
        "tight_overlap_pass_chunks": _pass_chunks(rows, "tight_overlap_phaseE_chunk_pass"),
        "guard_rejected_chunks": [
            int(row["chunk"]) for row in rows if _safe_bool(row.get("guard_trace_rejected")) is True
        ],
        "guard_retained_chunks": [
            int(row["chunk"]) for row in rows if _safe_bool(row.get("guard_trace_rejected")) is False
        ],
        "tight_rejected_chunks": [
            int(row["chunk"]) for row in rows if _safe_bool(row.get("tight_trace_rejected")) is True
        ],
        "tight_retained_chunks": [
            int(row["chunk"]) for row in rows if _safe_bool(row.get("tight_trace_rejected")) is False
        ],
        "tight_scale_state_clamped_chunks": [
            int(row["chunk"]) for row in rows if str(row.get("tight_trace_scale_state_action") or "") == "clamp"
        ],
        "good_news": (
            "thingstuff+RADIO qscale has a non-GT head-tail signal on chunks 10/12 in the ref055 canary, "
            "but it is not overlap-safe."
        ),
        "blockers": [
            "selected-write low-support explains chunk08 bad-write localization, not chunk10/12 gauge direction",
            "thingstuff+RADIO qscale beats controls for head-tail on chunks 8/10/12 but overlap passes only chunk8",
            "online native-overlap residual guard rejects chunk10 and removes the head-tail signal",
            "the same guard keeps chunk12 because local residual improves, yet downstream overlap still worsens",
            "adding overlap_tight scale-state clamps chunk12 scale from 0.955 to 0.98 and reduces overlap harm, but PhaseE still fails",
        ],
        "next_action": (
            "Do not promote. Stop selected-write/no-persistent and simple qscale sweeps. If continuing, design "
            "a non-GT future-overlap proxy or multi-objective merge/gauge controller; local overlap residual and "
            "simple scale-continuity clamps are insufficient."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "non_gt_direction_recheck_rows.csv", rows)
    _write_json(args.out_dir / "non_gt_direction_recheck_summary.json", summary)
    report = [
        "# v80 seq01 non-GT direction recheck",
        "",
        f"status: {summary['status']}",
        f"v80_goal_achieved: {summary['v80_goal_achieved']}",
        "",
        "## Finding",
        "",
        summary["good_news"],
        "",
        "## Blockers",
        "",
        *[f"- {item}" for item in summary["blockers"]],
        "",
        "## Next action",
        "",
        summary["next_action"],
        "",
    ]
    (args.out_dir / "non_gt_direction_recheck_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")


if __name__ == "__main__":
    main()
