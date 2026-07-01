#!/usr/bin/env python3
"""Audit whether v101 anchor ids can be joined to Stage-C seed ids.

The Stage-C seed traces prove component-like provenance is available in sampled
SWA top-k payloads.  This audit checks whether those seed ids can be directly
joined back to existing v101 ``anchor_id`` rows.  It is diagnostic-only and
does not authorize action.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
TRACK_U = ROOT / "trackU_true_current_support"
TRACE_ROOTS = [
    ROOT / "stage_c_seed_bridge_target_traces",
    ROOT / "stage_c_seed_bridge_clean_eval_traces_q128",
    ROOT / "stage_c_seed_anchor_probe_ttt_write_smoke",
    ROOT / "stage_c_seed_anchor_probe_ttt_write_diag_smoke",
    ROOT / "stage_c_seed_anchor_probe_ttt_write_diag_clean_eval_q128",
    ROOT / "stage_c_seed_anchor_probe_ttt_write_diag_target_q16",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def tensor_nonnegative_values(value: Any) -> set[int]:
    if not torch.is_tensor(value):
        return set()
    flat = value.detach().cpu().long().reshape(-1)
    return {int(item) for item in flat.tolist() if int(item) >= 0}


def tensor_nonnegative_count(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int((value.detach().cpu().long().reshape(-1) >= 0).sum().item())


def main() -> None:
    support_rows = read_rows(TRACK_U / "anchor_current_support_rows.csv")
    support_anchor_ids = {str(row.get("anchor_id", "")) for row in support_rows if row.get("anchor_id", "")}
    trace_rows: list[dict[str, Any]] = []
    trace_root_stats: dict[str, dict[str, Any]] = {}
    all_seed_ids: set[int] = set()
    all_lifecycle_anchor_ids: set[int] = set()
    all_lifecycle_stage_c_seed_modes: set[int] = set()
    all_lifecycle_anchor_seed_pairs: set[tuple[int, int]] = set()
    load_error_count = 0

    for trace_root in TRACE_ROOTS:
        root_key = str(trace_root)
        trace_root_stats[root_key] = {
            "trace_root": root_key,
            "payload_count": 0,
            "payload_with_stage_c_seed_count": 0,
            "payload_with_ttt_anchor_id_count": 0,
            "lifecycle_row_total": 0,
            "lifecycle_rows_with_stage_c_seed_mode": 0,
            "diagnostic_probe_ttt_write": "probe_ttt_write" in trace_root.name,
        }
        for path in sorted(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")):
            case_id = path.parents[2].name
            trace_root_stats[root_key]["payload_count"] += 1
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                load_error_count += 1
                trace_rows.append(
                    {
                        "trace_root": str(trace_root),
                        "case_id": case_id,
                        "trace_payload_path": str(path),
                        "load_error": f"{type(exc).__name__}:{exc}",
                    }
                )
                continue
            current_seed = payload.get("sampled_query_stage_c_seed_global_track_idx")
            cache_seed = payload.get("current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx")
            ttt_anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
            lifecycle_rows = payload.get("ttt_prev_stable_anchor_lifecycle_rows") or []
            seed_ids = tensor_nonnegative_values(current_seed) | tensor_nonnegative_values(cache_seed)
            all_seed_ids.update(seed_ids)
            ttt_anchor_values = tensor_nonnegative_values(ttt_anchor_ids)
            lifecycle_anchor_ids: set[int] = set()
            lifecycle_stage_c_seed_modes: set[int] = set()
            lifecycle_anchor_seed_pairs: set[tuple[int, int]] = set()
            if isinstance(lifecycle_rows, list):
                for lifecycle_row in lifecycle_rows:
                    if not isinstance(lifecycle_row, dict):
                        continue
                    try:
                        anchor_id = int(lifecycle_row.get("anchor_id"))
                    except (TypeError, ValueError):
                        continue
                    seed_mode_raw = lifecycle_row.get("source_stage_c_seed_global_track_idx_mode")
                    try:
                        seed_mode = int(seed_mode_raw)
                    except (TypeError, ValueError):
                        seed_mode = -1
                    lifecycle_anchor_ids.add(anchor_id)
                    if seed_mode >= 0:
                        lifecycle_stage_c_seed_modes.add(seed_mode)
                        lifecycle_anchor_seed_pairs.add((anchor_id, seed_mode))
            all_lifecycle_anchor_ids.update(lifecycle_anchor_ids)
            all_lifecycle_stage_c_seed_modes.update(lifecycle_stage_c_seed_modes)
            all_lifecycle_anchor_seed_pairs.update(lifecycle_anchor_seed_pairs)
            overlap_with_support_anchor_ids = len({str(item) for item in seed_ids} & support_anchor_ids)
            if seed_ids:
                trace_root_stats[root_key]["payload_with_stage_c_seed_count"] += 1
            if ttt_anchor_values:
                trace_root_stats[root_key]["payload_with_ttt_anchor_id_count"] += 1
            trace_root_stats[root_key]["lifecycle_row_total"] += len(lifecycle_rows)
            trace_root_stats[root_key]["lifecycle_rows_with_stage_c_seed_mode"] += len(lifecycle_anchor_seed_pairs)
            trace_rows.append(
                {
                    "trace_root": str(trace_root),
                    "trace_root_name": trace_root.name,
                    "case_id": case_id,
                    "trace_payload_path": str(path),
                    "current_seed_nonnegative_count": tensor_nonnegative_count(current_seed),
                    "cache_seed_nonnegative_count": tensor_nonnegative_count(cache_seed),
                    "unique_stage_c_seed_id_count": len(seed_ids),
                    "ttt_prev_stable_anchor_identity_available": bool(
                        payload.get("ttt_prev_stable_anchor_identity_available")
                    ),
                    "ttt_prev_stable_anchor_lifecycle_row_count": int(
                        payload.get("ttt_prev_stable_anchor_lifecycle_row_count") or len(lifecycle_rows)
                    ),
                    "topk_ttt_prev_stable_anchor_id_nonnegative_count": tensor_nonnegative_count(ttt_anchor_ids),
                    "unique_ttt_prev_stable_anchor_id_count": len(ttt_anchor_values),
                    "unique_lifecycle_anchor_id_count": len(lifecycle_anchor_ids),
                    "unique_lifecycle_stage_c_seed_mode_count": len(lifecycle_stage_c_seed_modes),
                    "lifecycle_anchor_seed_pair_count": len(lifecycle_anchor_seed_pairs),
                    "direct_stage_c_seed_id_overlap_support_anchor_id_count": overlap_with_support_anchor_ids,
                    "join_feasible_from_this_payload": bool(lifecycle_anchor_seed_pairs),
                    "runtime_action_allowed": False,
                }
            )

    direct_overlap = {str(item) for item in all_seed_ids} & support_anchor_ids
    payload_with_stage_c_seed = sum(1 for row in trace_rows if int(row.get("unique_stage_c_seed_id_count") or 0) > 0)
    payload_with_ttt_anchor_ids = sum(
        1 for row in trace_rows if int(row.get("unique_ttt_prev_stable_anchor_id_count") or 0) > 0
    )
    payload_with_lifecycle_seed_mode = sum(
        1 for row in trace_rows if int(row.get("lifecycle_anchor_seed_pair_count") or 0) > 0
    )
    lifecycle_row_total = sum(int(row.get("ttt_prev_stable_anchor_lifecycle_row_count") or 0) for row in trace_rows)
    diagnostic_anchor_seed_join_feasible = bool(all_lifecycle_anchor_seed_pairs)
    summary = {
        "schema": "acl2_v101_anchor_seed_join_feasibility_v1",
        "status": (
            "complete_diagnostic_join_observed_no_action"
            if diagnostic_anchor_seed_join_feasible
            else "complete_diagnostic_blocked"
        ),
        "diagnostic_only": True,
        "method_goal_achieved": False,
        "runtime_action_allowed": False,
        "anchor_seed_join_feasible": diagnostic_anchor_seed_join_feasible,
        "diagnostic_anchor_seed_join_feasible": diagnostic_anchor_seed_join_feasible,
        "strict_anchor_seed_join_ready_for_action": False,
        "support_anchor_row_count": len(support_rows),
        "support_anchor_unique_id_count": len(support_anchor_ids),
        "trace_root_count": len(TRACE_ROOTS),
        "trace_payload_file_count": len(trace_rows),
        "load_error_count": load_error_count,
        "payload_with_stage_c_seed_count": payload_with_stage_c_seed,
        "payload_with_ttt_anchor_id_count": payload_with_ttt_anchor_ids,
        "payload_with_lifecycle_stage_c_seed_mode_count": payload_with_lifecycle_seed_mode,
        "ttt_prev_stable_anchor_lifecycle_row_total": lifecycle_row_total,
        "unique_stage_c_seed_id_count": len(all_seed_ids),
        "unique_lifecycle_anchor_id_count": len(all_lifecycle_anchor_ids),
        "unique_lifecycle_stage_c_seed_mode_count": len(all_lifecycle_stage_c_seed_modes),
        "lifecycle_anchor_seed_pair_count": len(all_lifecycle_anchor_seed_pairs),
        "direct_stage_c_seed_id_overlap_support_anchor_id_count": len(direct_overlap),
        "blocker": (
            "probe_ttt_write traces with token contribution diagnostics now materialize diagnostic "
            "anchor_id -> stage_c_seed_global_track_idx_mode lifecycle pairs, but this is still no-action "
            "evidence and cannot authorize runtime while Track T/Q2/V/M4 and strict action gates remain failed."
            if diagnostic_anchor_seed_join_feasible
            else (
                "Stage-C seed ids are present in sampled q16/q128 traces, but no TTT stable anchor ids or lifecycle rows "
                "are present, and direct numeric overlap with Track U anchor_id is zero. A true join requires upstream "
                "anchor_id -> stage_c_seed_global_track_idx provenance."
            )
        ),
        "trace_root_stats": list(trace_root_stats.values()),
        "claim": "This audit blocks anchor-level promotion; it does not change runtime authorization.",
    }
    write_rows(FINAL / "anchor_seed_join_feasibility_rows.csv", trace_rows)
    write_json(FINAL / "anchor_seed_join_feasibility_summary.json", summary)
    write_text(
        FINAL / "anchor_seed_join_feasibility_report.md",
        "# Anchor Seed Join Feasibility\n\n"
        f"- Trace payload files: {summary['trace_payload_file_count']}\n"
        f"- Payloads with Stage-C seed ids: {summary['payload_with_stage_c_seed_count']}\n"
        f"- Payloads with TTT stable anchor ids: {summary['payload_with_ttt_anchor_id_count']}\n"
        f"- TTT lifecycle rows total: {summary['ttt_prev_stable_anchor_lifecycle_row_total']}\n"
        f"- Payloads with lifecycle Stage-C seed mode: {summary['payload_with_lifecycle_stage_c_seed_mode_count']}\n"
        f"- Unique lifecycle anchor ids: {summary['unique_lifecycle_anchor_id_count']}\n"
        f"- Unique lifecycle Stage-C seed modes: {summary['unique_lifecycle_stage_c_seed_mode_count']}\n"
        f"- Lifecycle anchor/seed pairs: {summary['lifecycle_anchor_seed_pair_count']}\n"
        f"- Direct seed/support-anchor id overlap: {summary['direct_stage_c_seed_id_overlap_support_anchor_id_count']}\n"
        f"- Join feasible: {summary['anchor_seed_join_feasible']}\n\n"
        f"Blocker: {summary['blocker']}\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
