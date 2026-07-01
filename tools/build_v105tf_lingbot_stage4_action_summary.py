#!/usr/bin/env python3
"""Summarize ACL2 v105-TF LingBot Stage 4 action-pilot results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE2 = RESULT_ROOT / "stage2_gca_trace"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
STAGE4 = RESULT_ROOT / "stage4_lingbot_action_pilot_or_blocked"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"
SEMGEOM_ACTION = "semantic_geometry_write_filter"
SEMGEOM_CONTEXT_ACTION = "semantic_geometry_context_only_demote"
SEMANTIC_SAFETY_CONTEXT_ACTION = "semantic_safety_strong_context_only_demote"
SEMANTIC_SAFETY_ANCHOR_ACTION = "semantic_safety_anchor_only_demote"
SEMANTIC_HEADLOCAL_CONTEXT_ACTION = "semantic_headlocal_relaxed_context_only_demote"
SEMANTIC_HEADLOCAL_SAFETY_CONTEXT_ACTION = "semantic_headlocal_safety_context_only_demote"
SEMANTIC_ACTIONS = [
    SEMGEOM_ACTION,
    SEMGEOM_CONTEXT_ACTION,
    SEMANTIC_SAFETY_CONTEXT_ACTION,
    SEMANTIC_SAFETY_ANCHOR_ACTION,
    SEMANTIC_HEADLOCAL_CONTEXT_ACTION,
    SEMANTIC_HEADLOCAL_SAFETY_CONTEXT_ACTION,
]
GEOMETRY_ACTION = "geometry_only_local_window_write_filter"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_indices(raw: str) -> list[int]:
    return [int(x) for x in str(raw or "").split(";") if x != ""]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frames.append(int(vals[0]))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    return np.asarray(frames, dtype=np.int64), np.stack(mats, axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var_src = float(np.mean(np.sum(x * x, axis=1)))
    if var_src <= 1e-12:
        return 1.0, np.eye(3), mu_dst - mu_src
    cov = (y.T @ x) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def sim3_residuals(gt_path: Path, pred_path: Path) -> dict[int, float]:
    gt_frames, gt = load_traj(gt_path)
    pred_frames, pred = load_traj(pred_path)
    if not np.array_equal(gt_frames, pred_frames):
        raise ValueError(f"frame mismatch: {gt_path} vs {pred_path}")
    gt_pos = gt[:, :3, 3]
    pred_pos = pred[:, :3, 3]
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    aligned = scale * (pred_pos @ rot.T) + trans
    residual = np.linalg.norm(aligned - gt_pos, axis=1)
    return {int(frame): float(err) for frame, err in zip(gt_frames, residual)}


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def max_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=np.float64)))


def label_positions() -> dict[str, dict[str, list[int]]]:
    labels: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"bad": [], "good": [], "all": []})
    for row in read_csv(STAGE3 / "frame_semantic_geometry_rows.csv"):
        seq = str(row["seq"])
        pos = int(float(row["sample_position"]))
        labels[seq]["all"].append(pos)
        if parse_bool(row.get("bad_label")):
            labels[seq]["bad"].append(pos)
        if parse_bool(row.get("good_label")):
            labels[seq]["good"].append(pos)
    return labels


def rel_pairs(baseline: dict[int, float], action: dict[int, float], positions: list[int]) -> tuple[list[float], list[float], list[float]]:
    improvements: list[float] = []
    harms: list[float] = []
    deltas: list[float] = []
    for pos in positions:
        if pos not in baseline or pos not in action:
            continue
        base = float(baseline[pos])
        act = float(action[pos])
        denom = max(abs(base), 1e-9)
        improvements.append((base - act) / denom)
        harms.append((act - base) / denom)
        deltas.append(act - base)
    return improvements, harms, deltas


def build() -> dict[str, Any]:
    labels = label_positions()
    action_rows = read_csv(STAGE4 / "action_config_rows.csv")
    run_results = read_csv(STAGE4 / "run_results.csv")
    run_failures = [row for row in run_results if int(row.get("returncode", 1)) != 0]
    parity = json.loads((STAGE2 / "trace_summary.json").read_text(encoding="utf-8"))
    no_action_parity = bool(parity.get("stage2_trace_parity_pass"))

    metric_rows: list[dict[str, Any]] = []
    per_action_pairs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for cfg in action_rows:
        seq = str(cfg["seq"])
        dataset = cfg["dataset"]
        method = cfg["method"]
        action_label = cfg["action_label"]
        forced_indices = parse_indices(cfg.get("force_non_keyframe_indices", ""))
        is_context_only_action = (
            action_label.endswith("_context_only_demote")
            or action_label.endswith("_anchor_only_demote")
        )
        gt_traj = STAGE2 / f"workspace/{dataset}/{seq}/gt/traj.txt"
        baseline_traj = STAGE2 / f"workspace/{dataset}/{seq}/{BASELINE_METHOD}/traj.txt"
        action_traj = STAGE4 / f"workspace/{dataset}/{seq}/{method}/traj.txt"

        baseline_res = sim3_residuals(gt_traj, baseline_traj)
        action_res = sim3_residuals(gt_traj, action_traj)
        bad_improve, _, _ = rel_pairs(baseline_res, action_res, labels[seq]["bad"])
        _, good_harm, _ = rel_pairs(baseline_res, action_res, labels[seq]["good"])
        _, all_harm, _ = rel_pairs(baseline_res, action_res, labels[seq]["all"])

        action_jsonl = load_jsonl(Path(cfg["action_file"]))
        trace_jsonl = load_jsonl(Path(cfg["trace_file"]))
        kv_rows = [row for row in trace_jsonl if row.get("row_type") == "kv_cache_provenance"]
        global_idx_count = len({int(row.get("global_idx", -1)) for row in kv_rows}) or 1
        forced_action_rows = [row for row in action_jsonl if row.get("forced_non_keyframe")]
        forced_context_rows = [row for row in action_jsonl if row.get("forced_context_only")]
        skip_action_rows = [row for row in action_jsonl if row.get("skip_append")]
        context_action_rows = [row for row in action_jsonl if row.get("context_only_append")]
        skip_kv_rows = [row for row in kv_rows if row.get("skip_append")]
        context_kv_rows = [row for row in kv_rows if row.get("context_only_append")]
        trace_error_rows = [row for row in trace_jsonl if row.get("row_type") == "trace_error"]
        expected_skip_kv = len(forced_indices) * global_idx_count
        expected_context_kv = len(forced_indices) * global_idx_count
        if is_context_only_action:
            action_fidelity_pass = (
                len(forced_context_rows) == len(forced_indices)
                and len(context_action_rows) == len(forced_indices)
                and len(skip_action_rows) == 0
                and sorted(int(row["sample_position"]) for row in forced_context_rows) == forced_indices
            )
        else:
            action_fidelity_pass = (
                len(forced_action_rows) == len(forced_indices)
                and len(skip_action_rows) == len(forced_indices)
                and sorted(int(row["sample_position"]) for row in forced_action_rows) == forced_indices
            )
        trace_fidelity_pass = (
            len(trace_error_rows) == 0
            and (
                len(context_kv_rows) == expected_context_kv
                if is_context_only_action
                else len(skip_kv_rows) == expected_skip_kv
            )
            and len(kv_rows) > 0
            and action_fidelity_pass
        )

        row = {
            "schema": "acl2_v105tf_lingbot_stage4_action_metric_v1",
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_label": action_label,
            "forced_count": len(forced_indices),
            "forced_indices": ";".join(str(x) for x in forced_indices),
            "bad_rows": len(labels[seq]["bad"]),
            "good_rows": len(labels[seq]["good"]),
            "bad_l3_median_improvement": median(bad_improve),
            "good_median_harm": median(good_harm),
            "good_max_harm": max_or_nan(good_harm),
            "all_median_harm": median(all_harm),
            "rolling_worse_fraction_gt_0p05": float(np.mean([x > 0.05 for x in all_harm])) if all_harm else float("nan"),
            "action_rows": len(action_jsonl),
            "forced_action_rows": len(forced_action_rows),
            "forced_context_rows": len(forced_context_rows),
            "skip_action_rows": len(skip_action_rows),
            "context_action_rows": len(context_action_rows),
            "trace_rows": len(trace_jsonl),
            "kv_cache_provenance_rows": len(kv_rows),
            "skip_kv_rows": len(skip_kv_rows),
            "expected_skip_kv_rows": expected_skip_kv,
            "context_only_kv_rows": len(context_kv_rows),
            "expected_context_only_kv_rows": expected_context_kv,
            "trace_error_rows": len(trace_error_rows),
            "action_fidelity_pass": action_fidelity_pass,
            "trace_fidelity_pass": trace_fidelity_pass,
            "no_action_parity_baseline_valid": no_action_parity,
            "traj_exists": action_traj.exists(),
        }
        metric_rows.append(row)
        for key in ["bad_l3_median_improvement", "good_median_harm", "good_max_harm", "rolling_worse_fraction_gt_0p05"]:
            per_action_pairs[action_label][key].append(float(row[key]))
        per_action_pairs[action_label]["trace_fidelity_pass"].append(float(trace_fidelity_pass))

        # Aggregate over frame-level pairs, not over sequence medians.
        per_action_pairs[action_label]["bad_improvements"].extend(bad_improve)
        per_action_pairs[action_label]["good_harms"].extend(good_harm)
        per_action_pairs[action_label]["all_harms"].extend(all_harm)

    aggregate_rows: list[dict[str, Any]] = []
    for action_label, vals in sorted(per_action_pairs.items()):
        bad_improvements = vals["bad_improvements"]
        good_harms = vals["good_harms"]
        all_harms = vals["all_harms"]
        aggregate_rows.append(
            {
                "schema": "acl2_v105tf_lingbot_stage4_action_aggregate_v1",
                "action_label": action_label,
                "bad_l3_median_improvement": median(bad_improvements),
                "good_median_harm": median(good_harms),
                "good_max_harm": max_or_nan(good_harms),
                "rolling_worse_fraction_gt_0p05": float(np.mean([x > 0.05 for x in all_harms])) if all_harms else float("nan"),
                "bad_pair_count": len(bad_improvements),
                "good_pair_count": len(good_harms),
                "trace_fidelity_pass": all(bool(x) for x in vals["trace_fidelity_pass"]),
            }
        )

    aggregate_by_action = {row["action_label"]: row for row in aggregate_rows}
    semantic_candidates = [aggregate_by_action.get(label, {}) for label in SEMANTIC_ACTIONS]
    semantic_candidates = [row for row in semantic_candidates if row]
    sem = max(
        semantic_candidates,
        key=lambda row: (
            float(row["bad_l3_median_improvement"]),
            -float(row["good_median_harm"]),
            -float(row["good_max_harm"]),
        ),
        default={},
    )
    geom = aggregate_by_action.get(GEOMETRY_ACTION, {})
    sem_beats_geometry_or_safety = False
    if sem and geom:
        sem_beats_geometry_or_safety = (
            float(sem["bad_l3_median_improvement"]) > float(geom["bad_l3_median_improvement"])
            or float(sem["good_median_harm"]) < float(geom["good_median_harm"])
            or float(sem["good_max_harm"]) < float(geom["good_max_harm"])
        )
    passing_semantic_actions = [
        row for row in semantic_candidates
        if (
            no_action_parity
            and not run_failures
            and bool(row["trace_fidelity_pass"])
            and float(row["bad_l3_median_improvement"]) >= 0.05
            and float(row["good_median_harm"]) <= 0.02
            and float(row["good_max_harm"]) <= 0.05
            and sem_beats_geometry_or_safety
        )
    ]
    stage4_action_pass = bool(passing_semantic_actions)

    write_csv(STAGE4 / "action_metric_rows.csv", metric_rows)
    write_csv(STAGE4 / "action_aggregate_metrics.csv", aggregate_rows)

    summary = {
        "schema": "acl2_v105tf_lingbot_stage4_action_summary_v1",
        "stage4_action_pass": stage4_action_pass,
        "no_action_parity_baseline_valid": no_action_parity,
        "run_result_rows": len(run_results),
        "run_failures": len(run_failures),
        "semantic_geometry_beats_geometry_or_safety": sem_beats_geometry_or_safety,
        "semantic_geometry_metrics": sem,
        "semantic_geometry_write_filter_metrics": aggregate_by_action.get(SEMGEOM_ACTION, {}),
        "semantic_geometry_context_only_metrics": aggregate_by_action.get(SEMGEOM_CONTEXT_ACTION, {}),
        "semantic_safety_strong_context_only_metrics": aggregate_by_action.get(SEMANTIC_SAFETY_CONTEXT_ACTION, {}),
        "semantic_safety_anchor_only_metrics": aggregate_by_action.get(SEMANTIC_SAFETY_ANCHOR_ACTION, {}),
        "semantic_headlocal_relaxed_context_only_metrics": aggregate_by_action.get(SEMANTIC_HEADLOCAL_CONTEXT_ACTION, {}),
        "semantic_headlocal_safety_context_only_metrics": aggregate_by_action.get(SEMANTIC_HEADLOCAL_SAFETY_CONTEXT_ACTION, {}),
        "passing_semantic_actions": passing_semantic_actions,
        "geometry_only_metrics": geom,
        "aggregate_metrics": aggregate_rows,
    }
    (STAGE4 / "stage4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if sem and bool(sem["trace_fidelity_pass"]) and float(sem["bad_l3_median_improvement"]) < 0.05:
        (STAGE4 / "TRACE_PASS_NO_L3_EFFECT.md").write_text(
            "# TRACE_PASS_NO_L3_EFFECT\n\n"
            "Stage4 semantic+geometry actions changed the intended KV path, but the best semantic action did not meet "
            "the required bad L3 median improvement >= 5% on the trace32 pilot.\n\n"
            f"- best_semantic_action: `{sem.get('action_label')}`\n"
            f"- bad_l3_median_improvement: `{sem['bad_l3_median_improvement']}`\n"
            f"- good_median_harm: `{sem['good_median_harm']}`\n"
            f"- good_max_harm: `{sem['good_max_harm']}`\n",
            encoding="utf-8",
        )
    if not stage4_action_pass:
        (STAGE4 / "ACTION_POLICY_FAILURE.md").write_text(
            "# Stage4 Action Policy Failure\n\n"
            "No Stage4 action pass is claimed. See `stage4_summary.json`, `action_metric_rows.csv`, "
            "and `action_aggregate_metrics.csv` for measured results.\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
