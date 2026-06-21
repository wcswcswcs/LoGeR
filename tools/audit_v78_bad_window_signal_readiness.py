#!/usr/bin/env python3
"""Audit evidence readiness for v78 bad-window signal amplification.

This is diagnostic-only. It joins the existing bad-window rankings with the
currently available SWA action-conditioned, boundary-local, and dual-gate
artifacts. The output is a checklist for what can be replayed now and what
still needs real action/QKV/dual-gate evidence before any runtime selector
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final"
)
DEFAULT_BAD_WINDOW_DIR = REPORT_ROOT / "bad_window_selection/v1_existing_trajectories"
DEFAULT_ACTION_ROWS = (
    REPORT_ROOT
    / "phase9_swa_cache_value_carryover/action_conditioned_signal_audit_v2_crossseq/"
    / "swa_action_conditioned_signal_rows.csv"
)
DEFAULT_DUAL_GATE_ROWS = (
    REPORT_ROOT
    / "phase9_swa_cache_value_carryover/dual_gate_action_signal_v1/"
    / "dual_gate_action_signal_rows.csv"
)
DEFAULT_BOUNDARY_ROWS = (
    REPORT_ROOT
    / "bad_good_case_contrast/v2_unique_scenes_top5/boundary_local_score_audit/"
    / "boundary_local_score_rows.csv"
)
DEFAULT_OUT_DIR = (
    REPORT_ROOT
    / "phase9_swa_cache_value_carryover/bad_window_signal_readiness_v1"
)


FAMILIES: dict[str, dict[str, str]] = {
    "single_chunk": {
        "file": "bad_single_chunk_table.csv",
        "id": "chunk_id",
        "metric": "local_sim3_rmse_m",
        "start": "chunk_start_frame",
        "end": "chunk_end_frame",
    },
    "adjacent_pair": {
        "file": "bad_adjacent_chunk_pair_table.csv",
        "id": "chunk_pair",
        "metric": "tail3_to_future_from_boundary_sim3_rmse_m",
        "start": "pair_start_frame",
        "boundary": "boundary_frame",
        "end": "pair_end_frame",
    },
    "five_chunk": {
        "file": "bad_5chunk_window_table.csv",
        "id": "window_chunks",
        "metric": "window5_joint_sim3_rmse_m",
        "start": "window_start_frame",
        "end": "window_end_frame",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-window-dir", type=Path, default=DEFAULT_BAD_WINDOW_DIR)
    parser.add_argument("--action-rows", type=Path, default=DEFAULT_ACTION_ROWS)
    parser.add_argument("--dual-gate-rows", type=Path, default=DEFAULT_DUAL_GATE_ROWS)
    parser.add_argument("--boundary-local-rows", type=Path, default=DEFAULT_BOUNDARY_ROWS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=True)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _case_key(row: dict[str, str], family: str) -> str:
    spec = FAMILIES[family]
    return f"{str(row.get('sequence', '')).zfill(2)}:{row.get(spec['id'], '')}"


def _action_window_key(row: dict[str, str], family: str) -> str:
    seq = str(row.get("sequence", "")).zfill(2)
    spec = FAMILIES[family]
    if family == "adjacent_pair":
        return (
            f"{seq}:{row.get(spec['start'], '')}:"
            f"{row.get(spec['boundary'], '')}:{row.get(spec['end'], '')}"
        )
    if family == "single_chunk":
        context_start = row.get("frame_start", "")
        return (
            f"{seq}:{context_start}:"
            f"{row.get(spec['start'], '')}:{row.get(spec['end'], '')}"
        )
    return ""


def _index_boundary(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(str(row.get("window_key", "")), []).append(row)
    return out


def _index_action(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(str(row.get("window_key", "")), []).append(row)
    return out


def _index_dual_gate(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("sequence", "")).zfill(2), str(row.get("chunk", "")))
        out.setdefault(key, []).append(row)
    return out


def _unique(values: list[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _boundary_score(rows: list[dict[str, str]]) -> float | None:
    values = [_finite(row.get("boundary_local_score")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(values[0])


def _row_for_case(
    *,
    family: str,
    rank: int,
    row: dict[str, str],
    action_by_window: dict[str, list[dict[str, str]]],
    boundary_by_key: dict[str, list[dict[str, str]]],
    dual_by_seq_chunk: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    spec = FAMILIES[family]
    seq = str(row.get("sequence", "")).zfill(2)
    case_key = _case_key(row, family)
    action_key = _action_window_key(row, family)
    action_rows = action_by_window.get(action_key, []) if action_key else []
    case_boundary = boundary_by_key.get(case_key, [])
    action_boundary = boundary_by_key.get(action_key, []) if action_key else []
    chunk_id = ""
    if family == "single_chunk":
        chunk_id = str(row.get("chunk_id", ""))
    elif family == "adjacent_pair":
        chunk_id = str(row.get("end_chunk_id", ""))
    dual_rows = dual_by_seq_chunk.get((seq, chunk_id), []) if chunk_id else []

    missing: list[str] = []
    if not case_boundary:
        missing.append("boundary_local_case_score")
    if family in {"single_chunk", "adjacent_pair"} and not action_rows:
        missing.append("action_conditioned_signal")
    if family in {"single_chunk", "adjacent_pair"} and not action_boundary:
        missing.append("boundary_local_extra_window_score")
    if family in {"single_chunk", "adjacent_pair"} and not dual_rows:
        missing.append("dual_gate_qkv_head_route_signal")
    if family == "five_chunk":
        missing.append("ttt_long_window_action_or_regime_replay")

    if not missing:
        status = "ready_for_selector_replay"
        next_step = "use_existing_action_conditioned_and_dual_gate_rows"
    elif action_rows:
        status = "partial_action_signal_available"
        next_step = "complete_missing_boundary_or_dual_gate_artifacts_before_runtime_gate"
    elif family == "five_chunk":
        status = "needs_ttt_long_window_regime_diagnostic"
        next_step = "build_five_chunk_regime_shift_visual_and_numeric_replay"
    else:
        status = "needs_new_action_conditioned_smoke"
        next_step = "run_fixed_actions_plus_same_mass_controls_then_recompute_dual_gate"

    return {
        "family": family,
        "bad_rank": rank,
        "run": row.get("run"),
        "sequence": seq,
        "case_id": row.get(spec["id"], ""),
        "metric": spec["metric"],
        "metric_value": _finite(row.get(spec["metric"])),
        "window_start": row.get(spec["start"], ""),
        "boundary_frame": row.get(spec.get("boundary", ""), ""),
        "window_end": row.get(spec["end"], ""),
        "case_window_key": case_key,
        "action_window_key": action_key,
        "has_boundary_case_score": bool(case_boundary),
        "boundary_case_score": _boundary_score(case_boundary),
        "has_boundary_extra_window_score": bool(action_boundary),
        "boundary_extra_window_score": _boundary_score(action_boundary),
        "has_action_conditioned_signal": bool(action_rows),
        "available_actions": ",".join(_unique([r.get("action") for r in action_rows])),
        "available_action_labels": ",".join(
            _unique([r.get("action_label") for r in action_rows])
        ),
        "has_dual_gate_signal": bool(dual_rows),
        "dual_gate_actions": ",".join(_unique([r.get("action") for r in dual_rows])),
        "dual_gate_offline_signal_count": sum(
            1 for r in dual_rows if str(r.get("dual_gate_offline_signal")) == "True"
        ),
        "phase9_gate_pass_count": sum(
            1 for r in dual_rows if str(r.get("phase9_gate_pass")) == "True"
        ),
        "readiness_status": status,
        "missing_evidence": ",".join(missing),
        "recommended_next_step": next_step,
    }


def main() -> None:
    args = parse_args()
    action_rows = _read_csv(args.action_rows)
    boundary_rows = _read_csv(args.boundary_local_rows)
    dual_rows = _read_csv(args.dual_gate_rows)
    action_by_window = _index_action(action_rows)
    boundary_by_key = _index_boundary(boundary_rows)
    dual_by_seq_chunk = _index_dual_gate(dual_rows)

    rows_out: list[dict[str, Any]] = []
    for family, spec in FAMILIES.items():
        rows = _read_csv(args.bad_window_dir / spec["file"])
        rows = rows[: max(0, int(args.top_k))]
        for idx, row in enumerate(rows, start=1):
            rows_out.append(
                _row_for_case(
                    family=family,
                    rank=idx,
                    row=row,
                    action_by_window=action_by_window,
                    boundary_by_key=boundary_by_key,
                    dual_by_seq_chunk=dual_by_seq_chunk,
                )
            )

    status_counts: dict[str, int] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for row in rows_out:
        status = str(row["readiness_status"])
        family = str(row["family"])
        status_counts[status] = status_counts.get(status, 0) + 1
        family_counts.setdefault(family, {})
        family_counts[family][status] = family_counts[family].get(status, 0) + 1

    summary = {
        "schema": "acl2_v78_bad_window_signal_readiness_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "bad_window_dir": str(args.bad_window_dir),
        "action_rows": str(args.action_rows),
        "boundary_local_rows": str(args.boundary_local_rows),
        "dual_gate_rows": str(args.dual_gate_rows),
        "top_k_per_family": int(args.top_k),
        "num_rows": len(rows_out),
        "status_counts": status_counts,
        "family_status_counts": family_counts,
        "ready_windows": [
            row
            for row in rows_out
            if row["readiness_status"] == "ready_for_selector_replay"
        ],
        "next_required_evidence": [
            "Run action-conditioned fixed actions plus same-mass controls for bad adjacent windows without action evidence.",
            "Materialize or summarize direct Q/K/V and per-head route signal before any runtime selector claim.",
            "For five-chunk TTT, build long-window appearance/geometry regime-shift diagnostics rather than SWA route replay.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "bad_window_signal_readiness_rows.csv", rows_out)
    _write_json(args.out_dir / "bad_window_signal_readiness_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
