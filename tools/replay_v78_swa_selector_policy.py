#!/usr/bin/env python3
"""Replay diagnostic SWA selector policies from v78 action-conditioned rows.

This tool does not rerun the model.  It consumes the already-landed
action-conditioned SWA audit table and asks a narrower question:

Given the same candidate actions on each window, would a simple scene-score
selector recover more weak-positive signal than fixed all-head/head-subset
rules?

The result is diagnostic-only.  It is a policy replay over the current measured
windows, not a validated runtime selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ACTION_ROWS = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/action_conditioned_signal_audit_v2_crossseq/"
    "swa_action_conditioned_signal_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/selector_policy_replay_v1"
)


STATIC_POLICIES = {
    "static_all_heads_topq80": {
        "type": "static",
        "action": "P9_34",
        "description": "Always use the all-head topq80 action.",
    },
    "static_head6_topq80": {
        "type": "static",
        "action": "P9_36",
        "description": "Always use the head6 topq80 action.",
    },
    "static_heads0_6_8_topq80": {
        "type": "static",
        "action": "P9_38",
        "description": "Always use the heads0,6,8 topq80 action.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-rows", type=Path, default=DEFAULT_ACTION_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scene-score-threshold",
        type=float,
        default=0.0,
        help="Adaptive policy switches from all-head to head6 at this scene score.",
    )
    parser.add_argument(
        "--min-source-quality-lift",
        type=float,
        default=0.0,
        help="Adaptive policy abstains if selected quality lift is below this floor.",
    )
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _group_by_window(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["window_key"], []).append(row)
    return dict(sorted(grouped.items()))


def _window_sort_key(rows: list[dict[str, str]]) -> tuple[str, int]:
    first = rows[0]
    return first.get("sequence", ""), _int(first.get("chunk"))


def _pick_static(
    rows_by_action: dict[str, dict[str, str]],
    action: str,
) -> tuple[dict[str, str] | None, str]:
    row = rows_by_action.get(action)
    if row is None:
        return None, f"abstain_missing_action_{action}"
    return row, f"static_select_{action}"


def _pick_adaptive(
    rows_by_action: dict[str, dict[str, str]],
    window_rows: list[dict[str, str]],
    scene_score_threshold: float,
    min_source_quality_lift: float,
) -> tuple[dict[str, str] | None, str]:
    first = window_rows[0]
    scene_score = _finite(first.get("scene_geometry_score"))
    quality_lift = _finite(first.get("selected_quality_lift_vs_all_mean"))
    if scene_score is None:
        return None, "abstain_missing_scene_geometry_score"
    if quality_lift is None:
        return None, "abstain_missing_selected_source_quality"
    if quality_lift < min_source_quality_lift:
        return None, (
            f"abstain_source_quality_lift_{quality_lift:.9g}_below_"
            f"{min_source_quality_lift:.9g}"
        )

    action = "P9_36" if scene_score >= scene_score_threshold else "P9_34"
    row = rows_by_action.get(action)
    if row is None:
        return None, f"abstain_missing_adaptive_action_{action}"
    side = "high" if scene_score >= scene_score_threshold else "low"
    return row, (
        f"scene_score_{scene_score:.9g}_{side}_vs_threshold_"
        f"{scene_score_threshold:.9g}_select_{action}"
    )


def _policy_definitions(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    policies = {
        "adaptive_scene_score_sign_v1": {
            "type": "adaptive",
            "description": (
                "If selected-source quality lift is non-negative, choose all-head "
                "P9_34 for scene score below threshold and head6 P9_36 otherwise."
            ),
            "scene_score_threshold": args.scene_score_threshold,
            "min_source_quality_lift": args.min_source_quality_lift,
        }
    }
    policies.update(STATIC_POLICIES)
    return policies


def _trace_row(
    policy_name: str,
    policy: dict[str, Any],
    selected: dict[str, str] | None,
    window_rows: list[dict[str, str]],
    reason: str,
) -> dict[str, Any]:
    first = window_rows[0]
    base = {
        "policy": policy_name,
        "policy_type": policy["type"],
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "sequence": first.get("sequence"),
        "chunk": first.get("chunk"),
        "window_key": first.get("window_key"),
        "scene_geometry_score": first.get("scene_geometry_score"),
        "selected_quality_lift_vs_all_mean": first.get(
            "selected_quality_lift_vs_all_mean"
        ),
        "decision_reason": reason,
        "choice_available": selected is not None,
    }
    if selected is None:
        base.update(
            {
                "selected_suite": None,
                "selected_action": None,
                "selected_action_label": None,
                "selected_head_indices": None,
                "selected_head_count": None,
                "official_metrics_beats_control_count": 0,
                "official_metrics_count": 0,
                "boundary_beats_control_count": 0,
                "boundary_metrics_count": 0,
                "weak_positive_action_label": False,
            }
        )
        return base

    base.update(
        {
            "selected_suite": selected.get("suite"),
            "selected_action": selected.get("action"),
            "selected_action_label": selected.get("action_label"),
            "selected_head_indices": selected.get("head_indices"),
            "selected_head_count": selected.get("head_count"),
            "official_metrics_beats_control_count": selected.get(
                "official_metrics_beats_control_count"
            ),
            "official_metrics_count": selected.get("official_metrics_count"),
            "official_metrics_beats_control": selected.get(
                "official_metrics_beats_control"
            ),
            "boundary_beats_control_count": selected.get(
                "boundary_beats_control_count"
            ),
            "boundary_metrics_count": selected.get("boundary_metrics_count"),
            "boundary_beats_control_metrics": selected.get(
                "boundary_beats_control_metrics"
            ),
            "attention_selected_lift_candidate_minus_control": selected.get(
                "attention_selected_lift_candidate_minus_control"
            ),
            "attention_source_lift_candidate_minus_control": selected.get(
                "attention_source_lift_candidate_minus_control"
            ),
            "weak_positive_action_label": str(
                selected.get("action_label", "")
            ).startswith("weak_positive"),
        }
    )
    return base


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["policy"])
        summary = summaries.setdefault(
            name,
            {
                "policy_type": row["policy_type"],
                "windows": 0,
                "choice_available_count": 0,
                "weak_positive_action_label_count": 0,
                "official_metrics_beats_control_count_sum": 0,
                "official_metrics_count_sum": 0,
                "boundary_beats_control_count_sum": 0,
                "boundary_metrics_count_sum": 0,
                "selected_actions": [],
                "selected_suites": [],
                "decision_reasons": [],
            },
        )
        summary["windows"] += 1
        if row["choice_available"]:
            summary["choice_available_count"] += 1
        if row["weak_positive_action_label"]:
            summary["weak_positive_action_label_count"] += 1
        summary["official_metrics_beats_control_count_sum"] += _int(
            row.get("official_metrics_beats_control_count")
        )
        summary["official_metrics_count_sum"] += _int(
            row.get("official_metrics_count")
        )
        summary["boundary_beats_control_count_sum"] += _int(
            row.get("boundary_beats_control_count")
        )
        summary["boundary_metrics_count_sum"] += _int(row.get("boundary_metrics_count"))
        summary["selected_actions"].append(row.get("selected_action"))
        summary["selected_suites"].append(row.get("selected_suite"))
        summary["decision_reasons"].append(row.get("decision_reason"))

    for summary in summaries.values():
        official_total = summary["official_metrics_count_sum"]
        boundary_total = summary["boundary_metrics_count_sum"]
        windows = summary["windows"]
        summary["official_metrics_beats_control_rate"] = (
            summary["official_metrics_beats_control_count_sum"] / official_total
            if official_total
            else None
        )
        summary["boundary_beats_control_rate"] = (
            summary["boundary_beats_control_count_sum"] / boundary_total
            if boundary_total
            else None
        )
        summary["weak_positive_action_label_rate"] = (
            summary["weak_positive_action_label_count"] / windows if windows else None
        )
    return summaries


def _best_policy(
    summaries: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    if not summaries:
        return None
    name, summary = max(
        summaries.items(),
        key=lambda item: (
            -1.0 if item[1].get(key) is None else float(item[1][key]),
            item[1].get("weak_positive_action_label_rate") or 0.0,
            item[0],
        ),
    )
    return {"policy": name, key: summary.get(key)}


def main() -> None:
    args = parse_args()
    action_rows = _read_csv(args.action_rows)
    grouped = _group_by_window(action_rows)
    policies = _policy_definitions(args)

    trace_rows: list[dict[str, Any]] = []
    for _, window_rows in sorted(grouped.items(), key=lambda item: _window_sort_key(item[1])):
        rows_by_action = {row["action"]: row for row in window_rows}
        for policy_name, policy in policies.items():
            if policy["type"] == "static":
                selected, reason = _pick_static(rows_by_action, policy["action"])
            elif policy["type"] == "adaptive":
                selected, reason = _pick_adaptive(
                    rows_by_action,
                    window_rows,
                    args.scene_score_threshold,
                    args.min_source_quality_lift,
                )
            else:
                raise ValueError(f"unknown policy type: {policy['type']}")
            trace_rows.append(
                _trace_row(policy_name, policy, selected, window_rows, reason)
            )

    summaries = _summarize(trace_rows)
    window_count = len(grouped)
    out_rows = args.out_dir / "swa_selector_policy_replay_rows.csv"
    out_summary = args.out_dir / "swa_selector_policy_replay_summary.json"
    _write_csv(out_rows, trace_rows)
    _write_json(
        out_summary,
        {
            "schema": "acl2_v78_swa_selector_policy_replay_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "input_action_rows": str(args.action_rows),
            "output_rows": str(out_rows),
            "num_input_action_rows": len(action_rows),
            "num_windows": window_count,
            "num_trace_rows": len(trace_rows),
            "policies": policies,
            "policy_summaries": summaries,
            "best_policy_by_official_metrics_rate": _best_policy(
                summaries, "official_metrics_beats_control_rate"
            ),
            "best_policy_by_boundary_metrics_rate": _best_policy(
                summaries, "boundary_beats_control_rate"
            ),
            "interpretation": {
                "adaptive_scene_score_sign_v1": (
                    f"On the current {window_count}-window replay, low scene score selects "
                    "P9_34 all-head and high scene score selects P9_36 head6."
                ),
                "not_runtime_validated": (
                    "This replay uses already-measured outcomes and is not a "
                    "held-out runtime selector validation."
                ),
            },
            "limitations": [
                f"Only {window_count} action-conditioned windows are available in this replay.",
                "The scene-score threshold is a diagnostic sign split, not a trained threshold.",
                "Selected-source quality is reconstructed from the SWA overlap feature dump.",
                "The feature dump is not direct Q/K/V tensor alignment.",
                "Random same-mass mask materialization is not available in this table.",
            ],
            "next_required_evidence": [
                "Materialize selected masks and same-mass random masks per decision.",
                "Add direct cache/current or Q/K/V alignment features if available.",
                "Replay the selector on held-out bad single, adjacent, overlap, and five-chunk windows.",
                "Only then consider a runtime selector smoke gate.",
            ],
        },
    )
    print(json.dumps({"rows": str(out_rows), "summary": str(out_summary)}, indent=2))


if __name__ == "__main__":
    main()
