#!/usr/bin/env python3
"""Summarize ACL2 v106 Stage5 query/head-local minimization results."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
STAGE2_V105 = V105 / "stage2_gca_trace"
STAGE3_V105 = V105 / "stage3_lingbot_oracle"
STAGE2 = V106 / "stage2_moge_metric_verifier"
STAGE3 = V106 / "stage3_memory_role_disambiguation"
STAGE5 = V106 / "stage5_query_head_local_minimization"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"


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


def parse_json_map(raw: str) -> dict[int, list[int]]:
    if not raw:
        return {}
    data = json.loads(raw)
    return {int(frame): [int(head) for head in heads] for frame, heads in data.items()}


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
    vals = [x for x in values if math.isfinite(float(x))]
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def max_or_nan(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(float(x))]
    return float(np.max(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def label_positions() -> dict[str, dict[str, list[int]]]:
    labels: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"bad": [], "good": [], "all": []})
    for row in read_csv(STAGE3_V105 / "frame_semantic_geometry_rows.csv"):
        seq = str(row["seq"])
        pos = int(float(row["sample_position"]))
        labels[seq]["all"].append(pos)
        if parse_bool(row.get("bad_label")):
            labels[seq]["bad"].append(pos)
        if parse_bool(row.get("good_label")):
            labels[seq]["good"].append(pos)
    return labels


def rel_pairs(baseline: dict[int, float], action: dict[int, float], positions: list[int]) -> tuple[list[float], list[float]]:
    improvements: list[float] = []
    harms: list[float] = []
    for pos in positions:
        if pos not in baseline or pos not in action:
            continue
        base = float(baseline[pos])
        act = float(action[pos])
        denom = max(abs(base), 1e-9)
        improvements.append((base - act) / denom)
        harms.append((act - base) / denom)
    return improvements, harms


def aggregate_row(action_name: str, vals: dict[str, list[float]]) -> dict[str, Any]:
    all_harms = vals["all_harms"]
    return {
        "schema": "acl2_v106tf_stage5_headlocal_aggregate_row_v1",
        "action_name": action_name,
        "bad_L3_median_improvement": median(vals["bad_improvements"]),
        "good_median_harm": median(vals["good_harms"]),
        "good_max_harm": max_or_nan(vals["good_harms"]),
        "rolling_worse_fraction_delta": float(np.mean([x > 0.05 for x in all_harms])) if all_harms else float("nan"),
        "bad_pair_count": len(vals["bad_improvements"]),
        "good_pair_count": len(vals["good_harms"]),
        "trace_fidelity_pass": all(bool(x) for x in vals["trace_fidelity_pass"]),
    }


def build() -> dict[str, Any]:
    labels = label_positions()
    config_rows = read_csv(STAGE5 / "action_config_rows.csv")
    scope_rows = read_csv(STAGE5 / "head_scope_rows.csv")
    run_results = read_csv(STAGE5 / "run_results.csv") if (STAGE5 / "run_results.csv").exists() else []
    run_failures = [row for row in run_results if int(row.get("returncode", 1)) != 0]
    stage2 = json.loads((STAGE2 / "stage2_summary.json").read_text(encoding="utf-8"))
    stage3 = json.loads((STAGE3 / "stage3_summary.json").read_text(encoding="utf-8"))

    metric_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cfg in config_rows:
        seq = str(cfg["seq"])
        dataset = cfg["dataset"]
        method = cfg["method"]
        action_name = cfg["action_name"]
        head_map = parse_json_map(cfg.get("head_action_map_json", ""))
        head_pair_count = sum(len(heads) for heads in head_map.values())
        gt_traj = STAGE2_V105 / f"workspace/{dataset}/{seq}/gt/traj.txt"
        baseline_traj = STAGE2_V105 / f"workspace/{dataset}/{seq}/{BASELINE_METHOD}/traj.txt"
        action_traj = STAGE5 / f"workspace/{dataset}/{seq}/{method}/traj.txt"
        traj_exists = action_traj.exists()
        if traj_exists:
            baseline_res = sim3_residuals(gt_traj, baseline_traj)
            action_res = sim3_residuals(gt_traj, action_traj)
            bad_improve, _ = rel_pairs(baseline_res, action_res, labels[seq]["bad"])
            _, good_harm = rel_pairs(baseline_res, action_res, labels[seq]["good"])
            _, all_harm = rel_pairs(baseline_res, action_res, labels[seq]["all"])
        else:
            bad_improve, good_harm, all_harm = [], [], []

        action_jsonl = load_jsonl(Path(cfg["action_file"]))
        trace_jsonl = load_jsonl(Path(cfg["trace_file"]))
        headlocal_rows = [row for row in action_jsonl if row.get("headlocal_action_enabled")]
        expected_frames = set(head_map)
        observed_frames = {int(row["sample_position"]) for row in headlocal_rows}
        observed_pairs = sum(
            len(str(row.get("headlocal_action_heads", "")).split(","))
            for row in headlocal_rows
            if row.get("headlocal_action_heads")
        )
        trace_error_rows = [row for row in trace_jsonl if row.get("row_type") == "trace_error"]
        kv_rows = [row for row in trace_jsonl if row.get("row_type") == "kv_cache_provenance"]
        action_fidelity_pass = (observed_frames == expected_frames and observed_pairs == head_pair_count)
        if action_name == "no_action":
            action_fidelity_pass = not headlocal_rows and head_pair_count == 0
        trace_fidelity_pass = traj_exists and not trace_error_rows and bool(kv_rows) and action_fidelity_pass

        row = {
            "schema": "acl2_v106tf_stage5_headlocal_metric_row_v1",
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": action_name,
            "action_family": cfg["action_family"],
            "control_for": cfg.get("control_for", ""),
            "promotion_eligible": cfg.get("promotion_eligible", ""),
            "stage5_action_mode": cfg["stage5_action_mode"],
            "head_action_pair_count": head_pair_count,
            "head_action_frames": ";".join(str(frame) for frame in sorted(head_map)),
            "bad_rows": len(labels[seq]["bad"]),
            "good_rows": len(labels[seq]["good"]),
            "bad_L3_median_improvement": median(bad_improve),
            "good_median_harm": median(good_harm),
            "good_max_harm": max_or_nan(good_harm),
            "rolling_worse_fraction_delta": float(np.mean([x > 0.05 for x in all_harm])) if all_harm else float("nan"),
            "action_rows": len(action_jsonl),
            "headlocal_action_rows": len(headlocal_rows),
            "observed_head_action_pairs": observed_pairs,
            "trace_rows": len(trace_jsonl),
            "kv_cache_provenance_rows": len(kv_rows),
            "trace_error_rows": len(trace_error_rows),
            "trace_fidelity_pass": trace_fidelity_pass,
            "traj_exists": traj_exists,
            "metric_available": traj_exists,
        }
        metric_rows.append(row)
        aggregate[action_name]["bad_improvements"].extend(bad_improve)
        aggregate[action_name]["good_harms"].extend(good_harm)
        aggregate[action_name]["all_harms"].extend(all_harm)
        aggregate[action_name]["trace_fidelity_pass"].append(float(trace_fidelity_pass))

    aggregate_rows = [aggregate_row(action_name, vals) for action_name, vals in sorted(aggregate.items())]
    aggregate_by_action = {row["action_name"]: row for row in aggregate_rows}
    no_action = aggregate_by_action.get("no_action", {})
    control_by_candidate: dict[str, str] = {}
    for cfg in config_rows:
        if cfg["action_family"] == "head_random_control" and cfg.get("control_for"):
            control_by_candidate[cfg["control_for"]] = cfg["action_name"]

    scope_eval_rows: list[dict[str, Any]] = []
    for scope in scope_rows:
        name = scope["scope_name"]
        available = parse_bool(scope.get("available", "false"))
        promotion_eligible = parse_bool(scope.get("promotion_eligible", "false"))
        candidate = aggregate_by_action.get(name, {})
        control_name = control_by_candidate.get(name, "")
        control = aggregate_by_action.get(control_name, {})
        margin = (
            float(candidate.get("bad_L3_median_improvement", float("nan")))
            - float(control.get("bad_L3_median_improvement", float("nan")))
            if candidate and control else float("nan")
        )
        has_profile = bool(scope.get("memory_role_profile") and scope.get("semantic_role_profile"))
        gate_pass = bool(
            available
            and promotion_eligible
            and candidate
            and bool(candidate.get("trace_fidelity_pass"))
            and float(candidate.get("bad_L3_median_improvement", -1.0)) >= 0.05
            and float(candidate.get("good_median_harm", 1.0)) <= 0.02
            and float(candidate.get("good_max_harm", 1.0)) <= 0.05
            and float(candidate.get("rolling_worse_fraction_delta", 1.0)) <= float(no_action.get("rolling_worse_fraction_delta", 0.0))
            and margin > 0.05
            and has_profile
        )
        scope_eval_rows.append(
            {
                "schema": "acl2_v106tf_stage5_scope_eval_row_v1",
                "scope_name": name,
                "stage5_step": scope.get("stage5_step", ""),
                "available": available,
                "promotion_eligible": promotion_eligible,
                "stage5_action_mode": scope.get("stage5_action_mode", ""),
                "head_action_pair_count": scope.get("head_action_pair_count", ""),
                "bad_L3_median_improvement": candidate.get("bad_L3_median_improvement", ""),
                "good_median_harm": candidate.get("good_median_harm", ""),
                "good_max_harm": candidate.get("good_max_harm", ""),
                "rolling_worse_fraction_delta": candidate.get("rolling_worse_fraction_delta", ""),
                "trace_fidelity_pass": candidate.get("trace_fidelity_pass", ""),
                "head_random_control": control_name,
                "head_random_bad_L3_median_improvement": control.get("bad_L3_median_improvement", ""),
                "head_random_margin": margin if math.isfinite(margin) else "",
                "gate_pass": gate_pass,
                "label_profile": scope.get("label_profile", ""),
                "memory_role_profile": scope.get("memory_role_profile", ""),
                "context_path_profile": scope.get("context_path_profile", ""),
                "semantic_role_profile": scope.get("semantic_role_profile", ""),
                "blocker": "" if gate_pass else (
                    scope.get("skip_reason", "") if not available
                    else "failed_stage5_l3_good_harm_rolling_or_random_margin_gate"
                ),
            }
        )

    control_rows = [
        {
            **row,
            "control_for": next((cfg.get("control_for", "") for cfg in config_rows if cfg["action_name"] == row["action_name"]), ""),
        }
        for row in aggregate_rows
        if row["action_name"].startswith("random_same_count__")
    ]

    passing = [row for row in scope_eval_rows if row["gate_pass"]]
    best_candidates = sorted(
        [row for row in scope_eval_rows if row["available"] and row["promotion_eligible"]],
        key=lambda row: (
            bool(row["gate_pass"]),
            float(row["bad_L3_median_improvement"] or -999),
            -float(row["good_max_harm"] or 999),
        ),
        reverse=True,
    )
    stage5_pass = bool(
        stage2.get("stage2_moge_pass")
        and stage3.get("stage3_disambiguation_pass")
        and passing
        and not run_failures
    )

    write_csv(STAGE5 / "headlocal_action_metrics.csv", metric_rows)
    write_csv(STAGE5 / "headlocal_action_aggregate_metrics.csv", aggregate_rows)
    write_csv(STAGE5 / "head_random_control_rows.csv", control_rows)
    write_csv(STAGE5 / "scope_eval_rows.csv", scope_eval_rows)

    summary = {
        "schema": "acl2_v106tf_stage5_headlocal_minimization_summary_v1",
        "stage3_disambiguation_pass": stage3.get("stage3_disambiguation_pass", False),
        "moge_available": stage2.get("moge_available", False),
        "moge_proxy_or_missing": stage2.get("moge_proxy_or_missing", True),
        "stage5_runtime_run": bool(metric_rows and not run_failures),
        "stage5_minimization_pass": stage5_pass,
        "stage5_status": "HEADLOCAL_MINIMIZATION_PASS" if stage5_pass else "HEADLOCAL_MINIMIZATION_NO_GO",
        "blocker": "" if stage5_pass else "no_stage5_scope_passed_l3_good_harm_rolling_and_random_margin_gate",
        "no_fabricated_metrics": True,
        "run_result_rows": len(run_results),
        "run_failures": len(run_failures),
        "scope_eval_rows": scope_eval_rows,
        "best_candidate": best_candidates[0] if best_candidates else {},
        "passing_scopes": passing,
        "outputs": {
            "head_scope_rows": (STAGE5 / "head_scope_rows.csv").relative_to(ROOT).as_posix(),
            "headlocal_action_metrics": (STAGE5 / "headlocal_action_metrics.csv").relative_to(ROOT).as_posix(),
            "headlocal_action_aggregate_metrics": (STAGE5 / "headlocal_action_aggregate_metrics.csv").relative_to(ROOT).as_posix(),
            "head_random_control_rows": (STAGE5 / "head_random_control_rows.csv").relative_to(ROOT).as_posix(),
            "scope_eval_rows": (STAGE5 / "scope_eval_rows.csv").relative_to(ROOT).as_posix(),
            "stage5_summary": (STAGE5 / "stage5_summary.json").relative_to(ROOT).as_posix(),
        },
    }
    (STAGE5 / "stage5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not stage5_pass:
        lines = [
            "# HEADLOCAL_DISAMBIGUATION_FAIL",
            "",
            "Stage5 did not produce a promotion-eligible scope that passed all gates.",
            "",
            "Required gates:",
            "",
            "- bad L3 median improvement >= 0.05",
            "- good median harm <= 0.02",
            "- good max harm <= 0.05",
            "- rolling worse fraction delta <= no_action",
            "- matched head-random same-count margin > 0.05",
            "- trace fidelity pass",
            "",
            "Scope outcomes:",
            "",
        ]
        for row in scope_eval_rows:
            lines.append(
                f"- {row['scope_name']}: pass={row['gate_pass']} bad={row['bad_L3_median_improvement']} "
                f"good_median={row['good_median_harm']} good_max={row['good_max_harm']} "
                f"rolling={row['rolling_worse_fraction_delta']} margin={row['head_random_margin']} "
                f"blocker={row['blocker']}"
            )
        lines.append("")
        lines.append("No Stage7 full/rolling validation is allowed from this Stage5 result.")
        (STAGE5 / "HEADLOCAL_DISAMBIGUATION_FAIL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
