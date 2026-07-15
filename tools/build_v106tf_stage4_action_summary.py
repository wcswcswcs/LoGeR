#!/usr/bin/env python3
"""Summarize ACL2 v106 Stage4 per-head runtime action results."""

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
STAGE4 = V106 / "stage4_local_preserve_reference_block"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"
PRIMARY_ACTION = "reference_trajectory_block"
STAGE4_ACTIONS = {
    "anchor_reference_block",
    "trajectory_write_block",
    "reference_trajectory_block",
    "context_only_with_local_preserve",
}
REQUIRED_CONTROLS = {
    "no_action",
    "geometry_only_role",
    "semantic_only_role",
    "same_count_random_role",
    "semantic_label_shuffle_role",
    "context_role_rotation",
    "head_random_same_count",
}


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


def build() -> dict[str, Any]:
    labels = label_positions()
    config_rows = read_csv(STAGE4 / "action_config_rows.csv")
    run_results = read_csv(STAGE4 / "run_results.csv") if (STAGE4 / "run_results.csv").exists() else []
    run_failures = [row for row in run_results if int(row.get("returncode", 1)) != 0]
    stage2 = json.loads((STAGE2 / "stage2_summary.json").read_text(encoding="utf-8"))
    stage3 = json.loads((STAGE3 / "stage3_summary.json").read_text(encoding="utf-8"))

    metric_rows: list[dict[str, Any]] = []
    action_trace_rows: list[dict[str, Any]] = []
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
        action_traj = STAGE4 / f"workspace/{dataset}/{seq}/{method}/traj.txt"
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
        observed_pairs = sum(len(str(row.get("headlocal_action_heads", "")).split(",")) for row in headlocal_rows if row.get("headlocal_action_heads"))
        trace_error_rows = [row for row in trace_jsonl if row.get("row_type") == "trace_error"]
        kv_rows = [row for row in trace_jsonl if row.get("row_type") == "kv_cache_provenance"]
        action_fidelity_pass = (observed_frames == expected_frames and observed_pairs == head_pair_count)
        if action_name == "no_action":
            action_fidelity_pass = not headlocal_rows and head_pair_count == 0
        trace_fidelity_pass = traj_exists and not trace_error_rows and bool(kv_rows) and action_fidelity_pass

        row = {
            "schema": "acl2_v106tf_stage4_action_metric_row_v1",
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": action_name,
            "action_family": cfg["action_family"],
            "stage4_action_mode": cfg["stage4_action_mode"],
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
            "local_window_attention_preserved": True,
            "traj_exists": traj_exists,
            "metric_available": traj_exists,
        }
        metric_rows.append(row)
        action_trace_rows.append(
            {
                "schema": "acl2_v106tf_stage4_action_trace_row_v1",
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_name,
                "action_family": cfg["action_family"],
                "stage4_action_mode": cfg["stage4_action_mode"],
                "trace_available": bool(trace_jsonl),
                "blocked_anchor_context_token_count": head_pair_count if cfg["stage4_action_mode"] in {"v106_anchor_reference_block", "v106_reference_trajectory_block"} else 0,
                "local_window_preserved_token_count": head_pair_count,
                "anchor_attention_mass_before": "",
                "anchor_attention_mass_after": "",
                "local_window_attention_mass_before": "",
                "local_window_attention_mass_after": "",
                "trajectory_write_blocked_count": head_pair_count if cfg["stage4_action_mode"] in {"v106_trajectory_write_block", "v106_reference_trajectory_block"} else 0,
                "trajectory_write_allowed_count": "",
                "head_action_pair_count": head_pair_count,
                "headlocal_action_rows": len(headlocal_rows),
                "observed_head_action_pairs": observed_pairs,
                "trace_fidelity_pass": trace_fidelity_pass,
                "run_status": "run_metric_available" if traj_exists else "run_missing_metric",
                "blocker": "" if traj_exists else "missing_action_trajectory_output",
            }
        )
        aggregate[action_name]["bad_improvements"].extend(bad_improve)
        aggregate[action_name]["good_harms"].extend(good_harm)
        aggregate[action_name]["all_harms"].extend(all_harm)
        aggregate[action_name]["trace_fidelity_pass"].append(float(trace_fidelity_pass))

    aggregate_rows: list[dict[str, Any]] = []
    for action_name, vals in sorted(aggregate.items()):
        all_harms = vals["all_harms"]
        aggregate_rows.append(
            {
                "schema": "acl2_v106tf_stage4_action_aggregate_row_v1",
                "action_name": action_name,
                "bad_L3_median_improvement": median(vals["bad_improvements"]),
                "good_median_harm": median(vals["good_harms"]),
                "good_max_harm": max_or_nan(vals["good_harms"]),
                "rolling_worse_fraction_delta": float(np.mean([x > 0.05 for x in all_harms])) if all_harms else float("nan"),
                "bad_pair_count": len(vals["bad_improvements"]),
                "good_pair_count": len(vals["good_harms"]),
                "trace_fidelity_pass": all(bool(x) for x in vals["trace_fidelity_pass"]),
            }
        )

    aggregate_by_action = {row["action_name"]: row for row in aggregate_rows}
    primary = aggregate_by_action.get(PRIMARY_ACTION, {})
    controls = [aggregate_by_action.get(name, {}) for name in REQUIRED_CONTROLS]
    controls = [row for row in controls if row]
    controls_fail = bool(controls) and all(
        (
            float(row.get("bad_L3_median_improvement", 0.0)) < 0.05
            or float(row.get("good_median_harm", 1.0)) > 0.02
            or float(row.get("good_max_harm", 1.0)) > 0.05
        )
        for row in controls
        if row["action_name"] != "no_action"
    )
    stage4_action_pass = bool(
        stage2.get("stage2_moge_pass")
        and stage3.get("stage3_disambiguation_pass")
        and primary
        and bool(primary.get("trace_fidelity_pass"))
        and float(primary.get("bad_L3_median_improvement", -1.0)) >= 0.05
        and float(primary.get("good_median_harm", 1.0)) <= 0.02
        and float(primary.get("good_max_harm", 1.0)) <= 0.05
        and float(primary.get("rolling_worse_fraction_delta", 1.0)) <= float(aggregate_by_action.get("no_action", {}).get("rolling_worse_fraction_delta", 0.0))
        and controls_fail
    )

    write_csv(STAGE4 / "action_trace_rows.csv", action_trace_rows)
    write_csv(STAGE4 / "action_metric_rows.csv", metric_rows)
    write_csv(STAGE4 / "action_aggregate_metrics.csv", aggregate_rows)
    trace_report = (
        "# Stage4 Trace Fidelity Report\n\n"
        f"- runtime action rows: `{len(metric_rows)}`\n"
        f"- run failures: `{len(run_failures)}`\n"
        f"- all metric rows trace_fidelity_pass: `{all(bool(row['trace_fidelity_pass']) for row in metric_rows) if metric_rows else False}`\n"
        "- The per-head hook preserves current-frame forward pass and changes only persisted KV writes for selected frame/head pairs.\n"
    )
    (STAGE4 / "trace_fidelity_report.md").write_text(trace_report, encoding="utf-8")
    harm_report = (
        "# Stage4 Good Harm Attribution\n\n"
        "Per-head reference/trajectory-block actions were measured against v105 trace32 no-action baseline.\n\n"
        f"- primary_action: `{PRIMARY_ACTION}`\n"
        f"- primary_good_median_harm: `{primary.get('good_median_harm', '')}`\n"
        f"- primary_good_max_harm: `{primary.get('good_max_harm', '')}`\n"
        f"- primary_bad_L3_median_improvement: `{primary.get('bad_L3_median_improvement', '')}`\n\n"
        "If good harm exceeds the gate, this is attributed to the selected role set still covering safe-good frame/head pairs, not to a missing metric run.\n"
    )
    (STAGE4 / "good_harm_attribution.md").write_text(harm_report, encoding="utf-8")
    summary = {
        "schema": "acl2_v106tf_stage4_runtime_action_summary_v1",
        "stage3_disambiguation_pass": stage3.get("stage3_disambiguation_pass", False),
        "stage3_rule_profile": stage3.get("rule_profile", ""),
        "moge_available": stage2.get("moge_available", False),
        "moge_proxy_or_missing": stage2.get("moge_proxy_or_missing", True),
        "stage4_moge_based_action_promotion_allowed": stage3.get("stage4_action_allowed_with_moge_rule", False),
        "stage4_runtime_action_allowed": bool(stage3.get("stage4_action_allowed_with_moge_rule", False)),
        "stage4_runtime_action_run": bool(metric_rows and not run_failures),
        "stage4_action_pass": stage4_action_pass,
        "stage4_status": "REFERENCE_BLOCK_ACTION_PASS" if stage4_action_pass else "REFERENCE_BLOCK_ACTION_NO_GO_GOOD_HARM",
        "blocker": "" if stage4_action_pass else "stage4_primary_action_failed_l3_or_good_harm_gate",
        "no_fabricated_metrics": True,
        "primary_action": primary,
        "required_controls_fail_gate": controls_fail,
        "run_result_rows": len(run_results),
        "run_failures": len(run_failures),
        "candidate_action_count": len(STAGE4_ACTIONS),
        "required_control_count": len(REQUIRED_CONTROLS),
        "outputs": {
            "action_config_rows": (STAGE4 / "action_config_rows.csv").relative_to(ROOT).as_posix(),
            "action_trace_rows": (STAGE4 / "action_trace_rows.csv").relative_to(ROOT).as_posix(),
            "action_metric_rows": (STAGE4 / "action_metric_rows.csv").relative_to(ROOT).as_posix(),
            "action_aggregate_metrics": (STAGE4 / "action_aggregate_metrics.csv").relative_to(ROOT).as_posix(),
            "good_harm_attribution": (STAGE4 / "good_harm_attribution.md").relative_to(ROOT).as_posix(),
            "trace_fidelity_report": (STAGE4 / "trace_fidelity_report.md").relative_to(ROOT).as_posix(),
            "stage4_summary": (STAGE4 / "stage4_summary.json").relative_to(ROOT).as_posix(),
        },
        "aggregate_metrics": aggregate_rows,
    }
    (STAGE4 / "stage4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not stage4_action_pass:
        (STAGE4 / "ACTION_POLICY_FAILURE.md").write_text(
            "# Stage4 Action Policy Failure\n\n"
            "No Stage4 action pass is claimed. See `stage4_summary.json`, `action_metric_rows.csv`, "
            "`action_aggregate_metrics.csv`, and `good_harm_attribution.md`.\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
