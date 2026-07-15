#!/usr/bin/env python3
"""Summarize v119 LB-TA trajectory-admission runs."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_NAME = os.environ.get("ACL2_V119_LBTA_RUN_NAME", "stage2_lbta_trajectory_admission").strip()
RUN_ROOT = RESULT_ROOT / RUN_NAME
WORKSPACE = RUN_ROOT / "workspace"
SUMMARY = RUN_ROOT / "summary"
CONFIG_ROWS = RUN_ROOT / "config_rows.csv"
MANIFEST = SUMMARY / "lbta_trajectory_admission_manifest.json"
SURFACE_FILTER = {
    part.strip()
    for part in os.environ.get("ACL2_V119_LBTA_SURFACES", "").split(",")
    if part.strip()
}

CONTROL_VARIANTS = [
    "ta1_internal_low_utility_drop",
    "ta2_semantic_low_support_drop",
    "ta5_reverse_high_support_drop",
    "ta6_temporal_random",
    "ta7_same_internal_bucket_shuffle",
]
CANDIDATE = "ta3_internal_semantic_low_combined_drop"
DEFAULT = "ta0_default"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_selected(raw: str) -> set[int]:
    return {int(part) for part in str(raw or "").split(";") if part.strip()}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_path(row: dict[str, str]) -> Path:
    return WORKSPACE / row["dataset"] / row["seq"] / row["method"] / "eval/traj.json"


def complete_path(row: dict[str, str]) -> Path:
    return WORKSPACE / row["dataset"] / row["seq"] / row["method"] / ".complete.json"


def parse_mask(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            out.append(float(text))
        except ValueError:
            return tuple()
    return tuple(out)


def action_trace_stats(path: Path, selected: set[int], action_mode: str) -> dict[str, Any]:
    rows = 0
    counts: Counter[str] = Counter()
    selected_seen: set[int] = set()
    good: set[int] = set()
    selected_base_keyframes: set[int] = set()
    selected_final_non_keyframes: set[int] = set()
    selected_skip_append: set[int] = set()
    selected_context_only: set[int] = set()
    selected_token_mask_ok: set[int] = set()
    selected_anchor_only: set[int] = set()
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_trace_rows": 0,
            "action_trace_schema_counts": {},
            "selected_action_seen_count": 0,
            "selected_good_action_count": 0,
            "selected_action_fidelity": 0.0 if selected else "",
        }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows += 1
            counts[str(item.get("schema", ""))] += 1
            frame = item.get("sample_position")
            if frame is None:
                continue
            frame_i = int(frame)
            if frame_i not in selected:
                continue
            selected_seen.add(frame_i)
            if item.get("base_is_keyframe") is True:
                selected_base_keyframes.add(frame_i)
            if item.get("final_is_keyframe") is False:
                selected_final_non_keyframes.add(frame_i)
            if item.get("skip_append") is True:
                selected_skip_append.add(frame_i)
            if item.get("forced_context_only") is True and item.get("context_only_append") is True:
                selected_context_only.add(frame_i)
            if item.get("forced_anchor_only") is True and item.get("context_only_special_mode") == "scale_only":
                selected_anchor_only.add(frame_i)
            if (
                item.get("context_only_special_mode") == "token_mask"
                and parse_mask(str(item.get("token_type_mask", ""))) == (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
            ):
                selected_token_mask_ok.add(frame_i)
            if action_mode == "force_non_keyframe":
                if (
                    item.get("forced_non_keyframe") is True
                    and item.get("base_is_keyframe") is True
                    and item.get("final_is_keyframe") is False
                    and item.get("skip_append") is True
                ):
                    good.add(frame_i)
            elif action_mode == "context_only_special":
                if (
                    item.get("forced_context_only") is True
                    and item.get("context_only_append") is True
                    and item.get("context_only_special_mode") == "all_special"
                    and item.get("base_is_keyframe") is True
                    and item.get("final_is_keyframe") is False
                    and item.get("skip_append") is False
                ):
                    good.add(frame_i)
            elif action_mode == "anchor_special_only":
                if (
                    item.get("forced_anchor_only") is True
                    and item.get("context_only_append") is True
                    and item.get("context_only_special_mode") == "scale_only"
                    and item.get("base_is_keyframe") is True
                    and item.get("final_is_keyframe") is False
                    and item.get("skip_append") is False
                ):
                    good.add(frame_i)
            elif action_mode == "trajectory_context_token_mask":
                if (
                    item.get("forced_context_only") is True
                    and item.get("context_only_append") is True
                    and item.get("context_only_special_mode") == "token_mask"
                    and parse_mask(str(item.get("token_type_mask", ""))) == (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
                    and item.get("base_is_keyframe") is True
                    and item.get("final_is_keyframe") is False
                    and item.get("skip_append") is False
                ):
                    good.add(frame_i)
    if not selected:
        fidelity: float | str = 1.0 if rows > 0 else 0.0
    else:
        fidelity = len(good) / len(selected)
    return {
        "action_trace_exists": True,
        "action_trace_rows": rows,
        "action_trace_schema_counts": dict(sorted(counts.items())),
        "selected_action_seen_count": len(selected_seen),
        "selected_base_keyframe_count": len(selected_base_keyframes),
        "selected_final_non_keyframe_count": len(selected_final_non_keyframes),
        "selected_skip_append_count": len(selected_skip_append),
        "selected_context_only_count": len(selected_context_only),
        "selected_anchor_only_count": len(selected_anchor_only),
        "selected_token_mask_ok_count": len(selected_token_mask_ok),
        "selected_good_action_count": len(good),
        "selected_action_fidelity": fidelity,
    }


def summarize_rows(config_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in config_rows:
        metric = read_json(metric_path(cfg))
        ate = fnum(metric.get("ate"))
        selected = parse_selected(cfg.get("selected_frame_ids", ""))
        action_file = ROOT / cfg["action_file"]
        stats = action_trace_stats(action_file, selected, cfg.get("surface_action_mode", ""))
        rows.append(
            {
                "schema": "acl2_v119tf_lbta_trajectory_admission_result_row_v1",
                "seq": cfg["seq"],
                "surface": cfg["surface"],
                "surface_family": cfg.get("surface_family", ""),
                "surface_action_mode": cfg.get("surface_action_mode", ""),
                "variant": cfg["variant"],
                "variant_key": cfg["variant_key"],
                "policy": cfg["policy"],
                "role": cfg["role"],
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "selected_count": cfg.get("selected_count", ""),
                "selected_frame_ids": cfg.get("selected_frame_ids", ""),
                "eval_exists": metric_path(cfg).exists(),
                "complete_marker_exists": complete_path(cfg).exists(),
                "ate": metric.get("ate", ""),
                "rpe_rot": metric.get("rpe_rot", ""),
                "rpe_trans": metric.get("rpe_trans", ""),
                "eval_json": rel(metric_path(cfg)) if metric_path(cfg).exists() else str(metric_path(cfg)),
                "complete_json": rel(complete_path(cfg)) if complete_path(cfg).exists() else str(complete_path(cfg)),
                "action_file": cfg["action_file"],
                "ate_float": ate if ate is not None else "",
                **stats,
            }
        )
    return rows


def compare_surface(rows: list[dict[str, Any]], surface: str) -> dict[str, Any]:
    surface_rows = [row for row in rows if row["surface"] == surface]
    by_variant = {
        variant: {row["seq"]: row for row in surface_rows if row["variant"] == variant}
        for variant in sorted({row["variant"] for row in surface_rows})
    }
    baseline = by_variant.get(DEFAULT, {})
    candidate = by_variant.get(CANDIDATE, {})
    rel_by_variant: dict[str, Any] = {}
    for variant, seq_rows in by_variant.items():
        rels = []
        per_seq = {}
        for seq, row in sorted(seq_rows.items()):
            ate = fnum(row.get("ate"))
            base_ate = fnum(baseline.get(seq, {}).get("ate"))
            rel = (base_ate - ate) / base_ate if ate is not None and base_ate not in (None, 0.0) else None
            if rel is not None:
                rels.append(rel)
            per_seq[seq] = {
                "ate": ate if ate is not None else "",
                "baseline_ate": base_ate if base_ate is not None else "",
                "ate_rel_improvement_vs_ta0": rel if rel is not None else "",
            }
        max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
        rel_by_variant[variant] = {
            "median_ate_rel_improvement_vs_ta0": median(rels) if rels else "",
            "max_harm_vs_ta0": max_harm,
            "both_sequences_improve_vs_ta0": bool(rels) and all(value > 0 for value in rels),
            "pilot_gate": bool(rels) and median(rels) >= 0.03 and max_harm <= 0.01 and all(value > 0 for value in rels),
            "per_seq": per_seq,
        }

    control_comparisons: dict[str, Any] = {}
    for control in CONTROL_VARIANTS:
        ctrl_rows = by_variant.get(control, {})
        seqs = sorted(set(candidate) & set(ctrl_rows))
        deltas = []
        per_seq = {}
        for seq in seqs:
            cand_ate = fnum(candidate[seq].get("ate"))
            ctrl_ate = fnum(ctrl_rows[seq].get("ate"))
            if cand_ate is None or ctrl_ate is None:
                continue
            delta = cand_ate - ctrl_ate
            deltas.append(delta)
            per_seq[seq] = {
                "candidate_ate": cand_ate,
                "control_ate": ctrl_ate,
                "candidate_minus_control_ate": delta,
            }
        control_comparisons[control] = {
            "seqs": seqs,
            "all_candidate_better": bool(deltas) and all(delta < 0 for delta in deltas),
            "median_candidate_minus_control_ate": median(deltas) if deltas else "",
            "per_seq": per_seq,
        }

    complete = bool(surface_rows) and all(row["eval_exists"] and row["complete_marker_exists"] for row in surface_rows)
    action_fidelity = bool(surface_rows) and all(fnum(row.get("selected_action_fidelity")) == 1.0 for row in surface_rows)
    candidate_better_all_controls = all(
        control_comparisons.get(control, {}).get("all_candidate_better") is True for control in CONTROL_VARIANTS
    )
    baseline_gate = bool(rel_by_variant.get(CANDIDATE, {}).get("pilot_gate", False))
    candidate_pass = complete and action_fidelity and candidate_better_all_controls and baseline_gate
    best_by_median = sorted(
        [
            (variant, stats.get("median_ate_rel_improvement_vs_ta0"))
            for variant, stats in rel_by_variant.items()
            if variant != DEFAULT and isinstance(stats.get("median_ate_rel_improvement_vs_ta0"), float)
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "surface": surface,
        "row_count": len(surface_rows),
        "complete": complete,
        "action_fidelity": action_fidelity,
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "candidate_pass": candidate_pass,
        "candidate_variant": CANDIDATE,
        "rel_by_variant": rel_by_variant,
        "control_comparisons": control_comparisons,
        "best_variant_by_median_rel": (
            {"variant": best_by_median[0][0], "median_ate_rel_improvement_vs_ta0": best_by_median[0][1]}
            if best_by_median
            else {}
        ),
    }


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v119-TF LB-TA Trajectory Admission Summary",
        "",
        f"decision: `{summary['decision']}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{summary['complete']}`",
        f"action_fidelity: `{summary['action_fidelity']}`",
        "",
        "## Surface Gates",
        "",
        "| surface | complete | action fidelity | TA3 baseline gate | TA3 better all controls | best median rel vs TA0 | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for surface, item in sorted(summary["surface_comparisons"].items()):
        best = item.get("best_variant_by_median_rel", {})
        lines.append(
            f"| `{surface}` | `{item['complete']}` | `{item['action_fidelity']}` | "
            f"`{item['baseline_gate']}` | `{item['candidate_better_all_controls']}` | "
            f"{best.get('median_ate_rel_improvement_vs_ta0', '')} | "
            f"`{'PASS' if item['candidate_pass'] else 'NO_GO'}` |"
        )
    lines += [
        "",
        "## Candidate Rows",
        "",
        "| surface | seq | variant | ATE | selected | action fidelity |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["variant"] != CANDIDATE:
            continue
        lines.append(
            f"| `{row['surface']}` | `{row['seq']}` | `{row['variant']}` | "
            f"{row['ate']} | {row['selected_count']} | {row['selected_action_fidelity']} |"
        )
    lines += [
        "",
        "## Evidence Boundary",
        "",
        summary["truthfulness_boundary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = read_json(MANIFEST)
    config_rows = read_csv(CONFIG_ROWS)
    if not manifest or not config_rows:
        raise FileNotFoundError(f"missing manifest/config rows under {RUN_ROOT}")
    if SURFACE_FILTER:
        config_rows = [row for row in config_rows if row.get("surface") in SURFACE_FILTER]
        if not config_rows:
            raise RuntimeError(f"surface filter matched no config rows: {sorted(SURFACE_FILTER)}")
    rows = summarize_rows(config_rows)
    surface_suffix = "_".join(sorted(SURFACE_FILTER))
    suffix = f"_{surface_suffix}" if surface_suffix else ""
    rows_csv = SUMMARY / f"lbta_trajectory_admission_rows{suffix}.csv"
    write_csv(rows_csv, rows)
    surface_comparisons = {
        surface: compare_surface(rows, surface)
        for surface in sorted({row["surface"] for row in rows})
    }
    complete = bool(rows) and all(row["eval_exists"] and row["complete_marker_exists"] for row in rows)
    action_fidelity = bool(rows) and all(fnum(row.get("selected_action_fidelity")) == 1.0 for row in rows)
    any_candidate_pass = any(item["candidate_pass"] for item in surface_comparisons.values())
    if not complete:
        decision = "LBTA_TRAJECTORY_ADMISSION_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "LBTA_TRAJECTORY_ADMISSION_ACTION_FIDELITY_NO_GO"
    elif any_candidate_pass:
        decision = "LBTA_TRAJECTORY_ADMISSION_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "LBTA_TRAJECTORY_ADMISSION_CONTROL_OR_BASELINE_NO_GO"
    summary = {
        "schema": "acl2_v119tf_lbta_trajectory_admission_summary_v1",
        "decision": decision,
        "global_goal_achieved": False,
        "run_root": rel(RUN_ROOT),
        "manifest": rel(MANIFEST),
        "rows_csv": rel(rows_csv),
        "report": rel(SUMMARY / f"LBTA_TRAJECTORY_ADMISSION_REPORT{suffix}.md"),
        "surface_filter": sorted(SURFACE_FILTER),
        "complete": complete,
        "action_fidelity": action_fidelity,
        "row_count": len(rows),
        "surface_comparisons": surface_comparisons,
        "truthfulness_boundary": (
            "Summary reads only generated method eval/traj.json and raw action traces. "
            "TA1 internal scores were built from current-code no-action predictions/confidence; "
            "TA2/TA3 semantic scores were built from causal SEM-V3 prefix rows. "
            "No fabricated metric, GT runtime cue, external depth, SLAM, training, or output trajectory post-processing is used."
        ),
    }
    summary_path = SUMMARY / f"lbta_trajectory_admission_summary{suffix}.json"
    write_json(summary_path, summary)
    (SUMMARY / f"LBTA_TRAJECTORY_ADMISSION_REPORT{suffix}.md").write_text(build_report(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
