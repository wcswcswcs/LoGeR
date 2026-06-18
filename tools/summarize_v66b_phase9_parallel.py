#!/usr/bin/env python3
"""Summarize v66B phase9 parallel READ/TTT rollouts.

The script is intentionally narrow and audit-oriented: it reads only completed
rollout artifacts, keeps missing values as null/blank, and writes CSV/JSON
tables used by the execution and recap logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_DIR = (
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "phase9_parallel_continuation/rollouts"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "report_final/phase9_parallel_continuation"
)
DEFAULT_EXTRA_RUN_DIRS = [
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "phase5_read_smoke/rollouts/V66B_P5_READ_BASE_DENSE_IGNORE_96F_AFTER_PRIORFIX"
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [v for v in (_safe_float(value) for value in values) if v is not None]
    return sum(nums) / len(nums) if nums else None


def _sum(values: Iterable[Any]) -> Optional[float]:
    nums = [v for v in (_safe_float(value) for value in values) if v is not None]
    return sum(nums) if nums else None


def _max(values: Iterable[Any]) -> Optional[float]:
    nums = [v for v in (_safe_float(value) for value in values) if v is not None]
    return max(nums) if nums else None


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_average(path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not path.is_file():
        return None, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 3 and parts[0].lower().rstrip(":") == "average":
            return _safe_float(parts[1]), _safe_float(parts[2])
    return None, None


def _category(run_name: str) -> str:
    if "_READ_" in run_name:
        return "read"
    if "_TTT_" in run_name:
        return "ttt"
    return "other"


def _frame_scope(run_name: str) -> str:
    if "_704_" in run_name:
        return "704F"
    if run_name.endswith("_96F") or "_96F_" in run_name:
        return "96F"
    return "unknown"


def _control_summary(row: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    control = row.get("control_trace")
    if not isinstance(control, Mapping):
        return {}
    hooks = control.get("hook_effect_summary")
    if not isinstance(hooks, Mapping):
        return {}
    value = hooks.get(path)
    return value if isinstance(value, Mapping) else {}


def _count_rows(rows: Sequence[Mapping[str, Any]], key: str, truthy: bool = True) -> int:
    if truthy:
        return sum(1 for row in rows if bool(row.get(key)))
    return sum(1 for row in rows if row.get(key) is not None)


def _count_value(rows: Sequence[Mapping[str, Any]], key: str, value: Any) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def _first_nonempty_dict(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        value = row.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def _first_present(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _compare_txt(path: Path, base_path: Optional[Path]) -> Tuple[Optional[int], Optional[float]]:
    if not path.is_file() or base_path is None or not base_path.is_file():
        return None, None
    left = path.read_text(encoding="utf-8", errors="replace").splitlines()
    right = base_path.read_text(encoding="utf-8", errors="replace").splitlines()
    diff_lines = 0
    max_abs = 0.0
    for a, b in zip(left, right):
        if a != b:
            diff_lines += 1
        a_parts = a.split()
        b_parts = b.split()
        for av, bv in zip(a_parts, b_parts):
            af = _safe_float(av)
            bf = _safe_float(bv)
            if af is not None and bf is not None:
                max_abs = max(max_abs, abs(af - bf))
    diff_lines += abs(len(left) - len(right))
    return diff_lines, max_abs


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = _clean(row.get(key))
                if value is None:
                    out[key] = ""
                elif isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    run = run_dir.name
    category = _category(run)
    frame_scope = _frame_scope(run)
    hmc_rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    ate_rmse, ate_sse = _parse_average(run_dir / "results_sim3" / "results_ate.txt")
    rpe_trans, rpe_rot = _parse_average(run_dir / "results_sim3" / "results_rpe.txt")
    wall = _read_json(run_dir / "wall_time_summary.json")
    correctness = _read_json(run_dir / "hmc_correctness_summary.json")

    frame_hooks = [_control_summary(row, "frame_attention") for row in hmc_rows]
    swa_hooks = [_control_summary(row, "swa_read") for row in hmc_rows]
    ttt_hooks = [_control_summary(row, "ttt_apply") for row in hmc_rows]
    semantic_sources = sorted({str(row.get("prior_semantic_source")) for row in hmc_rows if row.get("prior_semantic_source")})

    return {
        "run": run,
        "artifact_dir": str(run_dir),
        "category": category,
        "frame_scope": frame_scope,
        "ate_rmse": ate_rmse,
        "ate_sse": ate_sse,
        "rpe_trans": rpe_trans,
        "rpe_rot": rpe_rot,
        "wall_seconds": wall.get("wall_seconds"),
        "gpu": wall.get("gpu"),
        "traj_sha256": _sha256(run_dir / "01.txt"),
        "hmc_rows": len(hmc_rows),
        "num_chunks": correctness.get("num_chunks"),
        "dense_available_rows": _count_rows(hmc_rows, "prior_dense_semantic_available"),
        "semantic_sources": ",".join(semantic_sources),
        "semantic_role_consumed_rows": _count_rows(hmc_rows, "prior_semantic_role_consumed_any"),
        "condition_conflict_available_rows": _count_rows(hmc_rows, "prior_condition_signal_conflict_available"),
        "condition_conflict_chunk_broadcast_rows": _count_value(hmc_rows, "prior_condition_signal_conflict_level", "chunk_broadcast"),
        "condition_conflict_token_exact_rows": _count_rows(hmc_rows, "prior_condition_signal_conflict_token_exact"),
        "condition_conflict_value_avg": _mean(row.get("prior_condition_signal_conflict_value") for row in hmc_rows),
        "condition_conflict_token_mean_avg": _mean(row.get("prior_condition_signal_conflict_token_mean") for row in hmc_rows),
        "condition_conflict_token_p90_avg": _mean(row.get("prior_condition_signal_conflict_token_p90") for row in hmc_rows),
        "condition_scale_risk_available_rows": _count_rows(hmc_rows, "prior_condition_signal_scale_risk_available"),
        "condition_scale_risk_chunk_broadcast_rows": _count_value(hmc_rows, "prior_condition_signal_scale_risk_level", "chunk_broadcast"),
        "condition_scale_risk_token_exact_rows": _count_rows(hmc_rows, "prior_condition_signal_scale_risk_token_exact"),
        "condition_scale_risk_value_avg": _mean(row.get("prior_condition_signal_scale_risk_value") for row in hmc_rows),
        "condition_scale_risk_token_mean_avg": _mean(row.get("prior_condition_signal_scale_risk_token_mean") for row in hmc_rows),
        "condition_scale_risk_token_p90_avg": _mean(row.get("prior_condition_signal_scale_risk_token_p90") for row in hmc_rows),
        "ttt_write_present_rows": _count_rows(hmc_rows, "prior_ttt_write_present"),
        "frame_source_skip_applied_total": _sum(hook.get("num_context_source_skip_applied") for hook in frame_hooks),
        "frame_semantic_anchor_boost_applied_total": _sum(hook.get("num_semantic_anchor_boost_applied") for hook in frame_hooks),
        "frame_empty_source_events_total": _sum(hook.get("num_context_empty_source_events") for hook in frame_hooks),
        "frame_max_source_control_tokens": _max(hook.get("max_context_source_control_tokens") for hook in frame_hooks),
        "frame_max_source_boost_tokens": _max(hook.get("max_context_source_boost_tokens") for hook in frame_hooks),
        "frame_mean_keep_ratio": _mean(hook.get("mean_context_source_keep_ratio") for hook in frame_hooks),
        "frame_mean_abs_bias_mean": _mean(hook.get("mean_abs_bias") for hook in frame_hooks),
        "attention_mass_before_mean": _mean(hook.get("mean_attention_mass_removed_before") for hook in frame_hooks),
        "attention_mass_after_mean": _mean(hook.get("mean_attention_mass_removed_after") for hook in frame_hooks),
        "swa_overlap_bias_applied_total": _sum(hook.get("num_swa_overlap_bias_applied") for hook in swa_hooks),
        "swa_overlap_source_gate_applied_total": _sum(hook.get("num_swa_overlap_source_gate_applied") for hook in swa_hooks),
        "swa_overlap_source_gate_mean_avg": _mean(hook.get("mean_swa_overlap_source_gate") for hook in swa_hooks),
        "swa_overlap_source_gate_delta_avg": _mean(hook.get("mean_swa_overlap_source_gate_delta") for hook in swa_hooks),
        "swa_overlap_source_gate_max_delta": _max(hook.get("max_swa_overlap_source_gate_delta") for hook in swa_hooks),
        "swa_overlap_source_gate_score_mean_avg": _mean(hook.get("mean_swa_overlap_source_score") for hook in swa_hooks),
        "swa_overlap_source_gate_score_q90_avg": _mean(hook.get("mean_swa_overlap_source_score_q90") for hook in swa_hooks),
        "swa_overlap_source_replace_applied_total": _sum(hook.get("num_swa_overlap_source_replace_applied") for hook in swa_hooks),
        "swa_overlap_source_replace_alpha_avg": _mean(hook.get("mean_swa_overlap_source_replace_alpha") for hook in swa_hooks),
        "swa_overlap_source_replace_alpha_p90_avg": _mean(hook.get("mean_swa_overlap_source_replace_alpha_p90") for hook in swa_hooks),
        "swa_overlap_source_replace_score_mean_avg": _mean(hook.get("mean_swa_overlap_source_replace_score") for hook in swa_hooks),
        "prior_ttt_write_mean_avg": _mean(row.get("prior_ttt_write_mean") for row in hmc_rows),
        "prior_ttt_write_mean_min": _mean(row.get("prior_ttt_write_mean") for row in hmc_rows if row.get("prior_ttt_write_present")),
        "semantic_anchor_token_count_avg": _mean(row.get("prior_semantic_anchor_token_count") for row in hmc_rows),
        "semantic_anchor_static_ratio_avg": _mean(row.get("prior_semantic_anchor_static_semantic_ratio") for row in hmc_rows),
        "semantic_anchor_dynamic_ratio_avg": _mean(row.get("prior_semantic_anchor_dynamic_semantic_ratio") for row in hmc_rows),
        "anchor_write_mass_delta_avg": _mean(row.get("prior_anchor_write_mass_delta") for row in hmc_rows),
        "anchor_floor_applied_rows": _count_rows(hmc_rows, "prior_semantic_anchor_write_floor_applied"),
        "post_zp_action_delta_norm_mean": _mean(row.get("probe_ttt_write_action_delta_norm_mean") for row in hmc_rows),
        "post_zp_committed_delta_norm_mean": _mean(row.get("probe_ttt_write_post_delta_norm_mean") for row in hmc_rows),
        "post_zp_native_delta_norm_mean": _mean(row.get("probe_ttt_write_native_delta_norm_mean") for row in hmc_rows),
        "ttt_native_cosine_mean": _mean(row.get("probe_ttt_write_native_cosine_mean") for row in hmc_rows),
        "ttt_action_native_cosine_mean": _mean(row.get("probe_ttt_write_action_native_cosine_mean") for row in hmc_rows),
        "ttt_apply_context_skip_total": _sum(hook.get("num_context_source_skip_applied") for hook in ttt_hooks),
        "role_counts_first": _first_nonempty_dict(hmc_rows, "prior_semantic_role_counts"),
        "semantic_role_control_mode": _first_present(hmc_rows, "prior_semantic_role_control_mode"),
        "semantic_role_control_seed": _first_present(hmc_rows, "prior_semantic_role_control_seed"),
        "semantic_role_control_applied_rows": _count_rows(hmc_rows, "prior_semantic_role_control_applied"),
        "semantic_role_control_changed_fraction_avg": _mean(row.get("prior_semantic_role_control_changed_fraction") for row in hmc_rows),
        "R_ttt_role_counts_before_control_first": _first_nonempty_dict(hmc_rows, "prior_R_ttt_role_counts_before_control"),
        "R_ttt_role_counts_after_control_first": _first_nonempty_dict(hmc_rows, "prior_R_ttt_role_counts_after_control"),
        "semantic_swa_role_control_mode": _first_present(hmc_rows, "prior_semantic_swa_role_control_mode"),
        "semantic_swa_role_control_seed": _first_present(hmc_rows, "prior_semantic_swa_role_control_seed"),
        "semantic_swa_role_control_applied_rows": _count_rows(hmc_rows, "prior_semantic_swa_role_control_applied"),
        "semantic_swa_role_control_changed_fraction_avg": _mean(row.get("prior_semantic_swa_role_control_changed_fraction") for row in hmc_rows),
        "R_swa_role_counts_before_control_first": _first_nonempty_dict(hmc_rows, "prior_R_swa_role_counts_before_control"),
        "R_swa_role_counts_after_control_first": _first_nonempty_dict(hmc_rows, "prior_R_swa_role_counts_after_control"),
        "semantic_role_swa_negative_scale": _first_present(hmc_rows, "prior_semantic_role_swa_negative_scale"),
        "semantic_role_swa_protect_scale": _first_present(hmc_rows, "prior_semantic_role_swa_protect_scale"),
        "semantic_role_swa_negative_count_avg": _mean(row.get("prior_semantic_role_swa_negative_count") for row in hmc_rows),
        "semantic_role_swa_protect_count_avg": _mean(row.get("prior_semantic_role_swa_protect_count") for row in hmc_rows),
        "semantic_role_swa_protect_adjusted_rows": _count_rows(hmc_rows, "prior_semantic_role_swa_protect_adjusted"),
        "semantic_role_swa_score_before_mean_avg": _mean(row.get("prior_semantic_role_swa_score_before_mean") for row in hmc_rows),
        "semantic_role_swa_score_after_mean_avg": _mean(row.get("prior_semantic_role_swa_score_after_mean") for row in hmc_rows),
        "semantic_role_swa_score_protect_before_mean_avg": _mean(row.get("prior_semantic_role_swa_score_protect_before_mean") for row in hmc_rows),
        "semantic_role_swa_score_protect_after_mean_avg": _mean(row.get("prior_semantic_role_swa_score_protect_after_mean") for row in hmc_rows),
        "semantic_role_swa_score_negative_after_mean_avg": _mean(row.get("prior_semantic_role_swa_score_negative_after_mean") for row in hmc_rows),
    }


def _base_for(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    category = row.get("category")
    frame_scope = row.get("frame_scope")
    if category == "read":
        if frame_scope == "704F":
            needles = ["V66B_P9_704_READ_BASE_DENSE_IGNORE"]
        else:
            needles = [
                "V66B_P5_READ_BASE_DENSE_IGNORE_96F_AFTER_PRIORFIX",
                "V66B_P9_READ_BASE_DENSE_IGNORE",
            ]
    elif category == "ttt":
        needles = [f"V66B_P9_{'704_' if frame_scope == '704F' else ''}TTT_BASE_DENSE_IGNORE"]
    else:
        return None
    candidates = [
        base for base in rows
        if base.get("category") == category
        and base.get("frame_scope") == frame_scope
        and any(str(base.get("run", "")).startswith(needle) for needle in needles)
    ]
    return candidates[0] if candidates else None


def summarize(rollout_dir: Path, out_dir: Path, extra_run_dirs: Sequence[Path]) -> Dict[str, Any]:
    run_dirs = sorted(path for path in rollout_dir.iterdir() if path.is_dir() and path.name.startswith("V66B_P"))
    run_dirs.extend(path for path in extra_run_dirs if path.is_dir())
    rows = [_summarize_run(path) for path in run_dirs]

    by_run = {row["run"]: row for row in rows}
    for row in rows:
        base = _base_for(row, rows)
        if base is None:
            row["ate_delta_vs_category_base"] = None
            row["traj_diff_lines_vs_category_base"] = None
            row["traj_max_abs_diff_vs_category_base"] = None
            continue
        ate = _safe_float(row.get("ate_rmse"))
        base_ate = _safe_float(base.get("ate_rmse"))
        row["ate_delta_vs_category_base"] = (ate - base_ate) if ate is not None and base_ate is not None else None
        diff_lines, max_abs = _compare_txt(
            Path(str(row["artifact_dir"])) / "01.txt",
            Path(str(base["artifact_dir"])) / "01.txt",
        )
        row["traj_diff_lines_vs_category_base"] = diff_lines
        row["traj_max_abs_diff_vs_category_base"] = max_abs

    rows.sort(key=lambda row: (str(row.get("frame_scope")), str(row.get("category")), str(row.get("run"))))
    _write_csv(out_dir / "phase9_parallel_metrics.csv", rows)
    _write_json(out_dir / "phase9_parallel_metrics.json", rows)

    def sorted_subset(category: str, frame_scope: str) -> List[Dict[str, Any]]:
        subset = [row for row in rows if row.get("category") == category and row.get("frame_scope") == frame_scope]
        return sorted(subset, key=lambda row: (_safe_float(row.get("ate_rmse")) is None, _safe_float(row.get("ate_rmse")) or 0.0))

    summary = {
        "rollout_dir": str(rollout_dir),
        "extra_run_dirs": [str(path) for path in extra_run_dirs],
        "out_dir": str(out_dir),
        "run_count": len(rows),
        "read_96f_sorted_by_ate": sorted_subset("read", "96F"),
        "ttt_96f_sorted_by_ate": sorted_subset("ttt", "96F"),
        "read_704f_sorted_by_ate": sorted_subset("read", "704F"),
        "ttt_704f_sorted_by_ate": sorted_subset("ttt", "704F"),
        "missing_ate_runs": [row["run"] for row in rows if row.get("ate_rmse") is None],
    }
    _write_json(out_dir / "phase9_parallel_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--extra-run-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional completed run directory to include, usually an external baseline.",
    )
    args = parser.parse_args()
    extra_run_dirs = args.extra_run_dir if args.extra_run_dir is not None else DEFAULT_EXTRA_RUN_DIRS
    summary = summarize(args.rollout_dir, args.out_dir, extra_run_dirs)
    print(json.dumps(_clean({
        "run_count": summary["run_count"],
        "missing_ate_runs": summary["missing_ate_runs"],
        "out_dir": summary["out_dir"],
    }), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
