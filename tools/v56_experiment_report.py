#!/usr/bin/env python3
"""Generate ACL2 v56 H35 semantic/new-TTT-action reports from landed artifacts.

The script reads rollout outputs only.  Missing metrics remain empty or are
represented by explicit no-data figures; no planned value is converted into an
observed value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v47_adaptive_ttt_writer_report import _walk  # noqa: E402
from tools.v53_experiment_report import (  # noqa: E402
    C9_P0_ATE,
    _fmt,
    _iter_run_dirs,
    _mean,
    _plot_no_data,
    _read_jsonl,
    _safe_float,
    _summarize_runs,
    _write_csv,
    _write_json,
)
from tools.v53_full_sequence_drift_autopsy import (  # noqa: E402
    _load_kitti_gt,
    _load_run_poses,
    _segment_error,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_H35_704 = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_refine_screen/rollouts/V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F"
)
DEFAULT_H35_FULL = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)
DOC_EXEC = REPO_ROOT / "docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_Fast_执行日志.md"
DOC_REVIEW = REPO_ROOT / "docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_Fast_实验结果复盘.md"
PLAN_DOC = "docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_FastPlan.md"
H35_FULL_ATE = 35.74089695811434
H35_SUCCESS_A = 33.7409
H35_STRONG = 34.7409
H35_MIN_PROGRESS = 35.2409

SEGMENTS: Sequence[Tuple[str, int, int]] = (
    ("seg0_000_384", 0, 384),
    ("seg1_384_700", 384, 700),
    ("seg2_700_end", 700, 20000),
    ("window_200_300", 200, 300),
    ("window_400_600", 400, 600),
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_yaml_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def _is_full(row: Mapping[str, Any]) -> bool:
    return bool(row.get("full_kitti01")) or int(row.get("frames") or 0) >= 1000


def _is_704(row: Mapping[str, Any]) -> bool:
    return 650 <= int(row.get("frames") or 0) <= 750 or "704" in str(row.get("row") or row.get("run_name") or "").upper()


def _is_smoke(row: Mapping[str, Any]) -> bool:
    return 1 <= int(row.get("frames") or 0) <= 140 or "96F" in str(row.get("row") or row.get("run_name") or "").upper()


def _candidate(row: Mapping[str, Any]) -> str:
    cand = str(row.get("candidate") or "").strip()
    if cand:
        return cand
    run = str(row.get("run_name") or "")
    match = re.search(r"(A[1-4]|B[1-4]|H35|COMBO)", run)
    return match.group(1) if match else ""


def _track(row: Mapping[str, Any]) -> str:
    track = str(row.get("track") or "").strip()
    if track:
        return track
    cand = _candidate(row)
    if cand.startswith("A"):
        return "A"
    if cand.startswith("B"):
        return "B"
    if cand == "H35":
        return "phase0"
    if cand == "COMBO":
        return "combo"
    return ""


def _manual_percentage_pass(row: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    role = str(row.get("role_mode_config") or audit.get("role_mode") or "").lower()
    role_ok = any(token in role for token in ("adaptive", "binary", "risk_veto", "state_energy", "sc_gamma", "stable_anchor"))
    return bool(
        role_ok
        and _safe_float(audit.get("manual_positive_frac"), 999.0) == 0.0
        and _safe_float(audit.get("manual_negative_frac"), 999.0) == 0.0
        and _safe_float(audit.get("manual_neutral_lambda"), 999.0) == 0.0
        and bool(audit.get("no_manual_tri_replay_percentages", False))
    )


def _extra_debug(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: Dict[str, Any] = {}
    stage_hits = []
    sem_label_counts = []
    sem_fallbacks = []
    sem_applied = 0
    context_applied = 0
    context_empty = []
    source_tokens = []
    mass_removed_before = []
    mass_removed_after = []
    mass_retained_before = []
    mass_retained_after = []
    stable_mass = []
    risk_mass = []
    no_long_mass = []
    role_thresholds = []
    native_gate_scale = []
    native_gate_cos = []
    dual_long_override = 0
    dual_tensors = []
    commit_norms = []
    for row in rows:
        for node in _walk(row):
            if "stage_c_cache_hit" in node:
                stage_hits.append(1.0 if bool(node.get("stage_c_cache_hit")) else 0.0)
            if node.get("v31_semantic_label_count") is not None:
                sem_label_counts.append(_safe_float(node.get("v31_semantic_label_count")))
            if node.get("v31_semantic_label_fallback_ratio") is not None:
                sem_fallbacks.append(_safe_float(node.get("v31_semantic_label_fallback_ratio")))
            if node.get("v31_semantic_recondition_applied") is True:
                sem_applied += 1
            if node.get("num_context_source_skip_applied") is not None:
                context_applied += int(node.get("num_context_source_skip_applied") or 0)
            if node.get("num_context_empty_source_events") is not None:
                context_empty.append(_safe_float(node.get("num_context_empty_source_events")))
            if node.get("max_context_source_skip_tokens") is not None:
                source_tokens.append(_safe_float(node.get("max_context_source_skip_tokens")))
            for key, target in (
                ("mean_attention_mass_removed_before", mass_removed_before),
                ("mean_attention_mass_removed_after", mass_removed_after),
                ("mean_attention_mass_retained_before", mass_retained_before),
                ("mean_attention_mass_retained_after", mass_retained_after),
            ):
                if node.get(key) is not None:
                    target.append(_safe_float(node.get(key)))
            if node.get("ttt_tri_replay_stable_anchor_token_mass") is not None:
                stable_mass.append(_safe_float(node.get("ttt_tri_replay_stable_anchor_token_mass")))
            if node.get("ttt_tri_replay_risk_token_mass") is not None:
                risk_mass.append(_safe_float(node.get("ttt_tri_replay_risk_token_mass")))
            if node.get("ttt_tri_replay_no_long_write_token_mass") is not None:
                no_long_mass.append(_safe_float(node.get("ttt_tri_replay_no_long_write_token_mass")))
            for key in ("ttt_tri_replay_binary_anchor_threshold", "ttt_tri_replay_risk_veto_threshold"):
                if node.get(key) is not None:
                    role_thresholds.append(_safe_float(node.get(key)))
            if node.get("ttt_write_native_delta_gate_scale_mean") is not None:
                native_gate_scale.append(_safe_float(node.get("ttt_write_native_delta_gate_scale_mean")))
            if node.get("ttt_write_native_delta_gate_cos_mean") is not None:
                native_gate_cos.append(_safe_float(node.get("ttt_write_native_delta_gate_cos_mean")))
            if node.get("ttt_dual_lifetime_long_old_override") is True:
                dual_long_override += 1
            if node.get("ttt_dual_lifetime_long_old_override_tensors") is not None:
                dual_tensors.append(_safe_float(node.get("ttt_dual_lifetime_long_old_override_tensors")))
            if node.get("ttt_write_commit_filter_num_tensors") is not None:
                commit_norms.append(_safe_float(node.get("ttt_write_commit_filter_num_tensors")))
    log_path = run_dir / "01.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")

        def add_numbers(key: str, target: List[float]) -> None:
            if target:
                return
            pattern = re.compile(rf"['\"]{re.escape(key)}['\"]:\s*([-+0-9.eE]+)")
            target.extend(_safe_float(match.group(1)) for match in pattern.finditer(text))

        def count_true(key: str) -> int:
            pattern = re.compile(rf"['\"]{re.escape(key)}['\"]:\s*(True|False|true|false)")
            return sum(1 for match in pattern.finditer(text) if match.group(1).lower() == "true")

        add_numbers("v31_semantic_label_count", sem_label_counts)
        add_numbers("v31_semantic_label_fallback_ratio", sem_fallbacks)
        add_numbers("max_context_source_skip_tokens", source_tokens)
        add_numbers("mean_attention_mass_removed_before", mass_removed_before)
        add_numbers("mean_attention_mass_removed_after", mass_removed_after)
        add_numbers("mean_attention_mass_retained_before", mass_retained_before)
        add_numbers("mean_attention_mass_retained_after", mass_retained_after)
        add_numbers("ttt_tri_replay_stable_anchor_token_mass", stable_mass)
        add_numbers("ttt_tri_replay_risk_token_mass", risk_mass)
        add_numbers("ttt_tri_replay_no_long_write_token_mass", no_long_mass)
        add_numbers("ttt_tri_replay_binary_anchor_threshold", role_thresholds)
        add_numbers("ttt_tri_replay_risk_veto_threshold", role_thresholds)
        add_numbers("ttt_write_native_delta_gate_scale_mean", native_gate_scale)
        add_numbers("ttt_write_native_delta_gate_cos_mean", native_gate_cos)
        add_numbers("ttt_dual_lifetime_long_old_override_tensors", dual_tensors)
        dual_long_override = max(dual_long_override, count_true("ttt_dual_lifetime_long_old_override"))
    out["stage_c_cache_hit_rate"] = _mean(stage_hits)
    out["stage_c_cache_hit_count"] = int(sum(stage_hits)) if stage_hits else 0
    out["stage_c_cache_seen"] = len(stage_hits)
    out["semantic_label_count_mean"] = _mean(sem_label_counts)
    out["semantic_label_fallback_ratio_mean"] = _mean(sem_fallbacks)
    out["semantic_recondition_applied_count"] = sem_applied
    out["context_source_skip_applied_count"] = context_applied
    out["context_empty_source_events"] = _mean(context_empty)
    out["affected_source_token_count_max"] = max([v for v in source_tokens if math.isfinite(v)], default=None)
    out["source_influence_mass_removed_before_mean"] = _mean(mass_removed_before)
    out["source_influence_mass_removed_after_mean"] = _mean(mass_removed_after)
    out["source_influence_mass_retained_before_mean"] = _mean(mass_retained_before)
    out["source_influence_mass_retained_after_mean"] = _mean(mass_retained_after)
    out["stable_anchor_token_mass_mean"] = _mean(stable_mass)
    out["risk_token_mass_mean"] = _mean(risk_mass)
    out["no_long_write_token_mass_mean"] = _mean(no_long_mass)
    out["role_threshold_mean"] = _mean(role_thresholds)
    out["native_delta_gate_scale_mean"] = _mean(native_gate_scale)
    out["native_delta_gate_cos_mean"] = _mean(native_gate_cos)
    out["dual_lifetime_long_override_count"] = dual_long_override
    out["dual_lifetime_override_tensors_mean"] = _mean(dual_tensors)
    out["commit_delta_tensor_count_mean"] = _mean(commit_norms)
    return out


def _stage_c_cache_superset_path(
    cache_root: Path,
    *,
    chunk_idx: int,
    start: int,
    end: int,
) -> Optional[Path]:
    prefix = f"chunk_{int(chunk_idx):03d}_{int(start):06d}_"
    best: Optional[Tuple[int, Path]] = None
    for chunk_dir in cache_root.glob(prefix + "*"):
        if not chunk_dir.is_dir():
            continue
        parts = chunk_dir.name.split("_")
        if len(parts) != 4:
            continue
        try:
            cached_end = int(parts[3])
        except ValueError:
            continue
        cache_path = chunk_dir / "masklet.pt"
        if cached_end >= int(end) and cache_path.is_file():
            if best is None or cached_end < best[0]:
                best = (cached_end, cache_path)
    return best[1] if best is not None else None


def _stage_c_cache_artifact_debug(run_dir: Path, config: Path) -> Dict[str, Any]:
    mode = _first_yaml_value(config, "stage_c_cache_mode").lower()
    cache_dir = _first_yaml_value(config, "stage_c_cache_dir")
    if mode in {"", "off", "none"} or not cache_dir:
        return {}
    log_path = run_dir / "01.log"
    chunks: List[Tuple[int, int, int]] = []
    if log_path.is_file():
        pattern = re.compile(r"# V2 Chunk\s+(\d+)/\d+\s+frames\s+\[(\d+),\s*(\d+)\)")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                chunks.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    cache_root = Path(cache_dir)
    hit_count = 0
    exact_count = 0
    superset_count = 0
    missing: List[str] = []
    for chunk_idx, start, end in chunks:
        exact = cache_root / f"chunk_{chunk_idx:03d}_{start:06d}_{end:06d}" / "masklet.pt"
        if exact.is_file():
            hit_count += 1
            exact_count += 1
            continue
        superset = _stage_c_cache_superset_path(cache_root, chunk_idx=chunk_idx, start=start, end=end)
        if superset is not None:
            hit_count += 1
            superset_count += 1
            continue
        missing.append(str(exact))
    return {
        "stage_c_cache_hit_rate": (hit_count / len(chunks) if chunks else None),
        "stage_c_cache_hit_count": hit_count,
        "stage_c_cache_seen": len(chunks),
        "stage_c_cache_exact_hit_count": exact_count,
        "stage_c_cache_superset_hit_count": superset_count,
        "stage_c_cache_missing_count": len(missing),
        "stage_c_cache_missing_examples": ";".join(missing[:3]),
    }


def _augment_rows(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        config = run_dir / "effective_config.yaml"
        row["candidate"] = _first_yaml_value(config, "candidate") or _candidate(row)
        row["track"] = _first_yaml_value(config, "track") or _track(row)
        row["semantic_desc"] = _first_yaml_value(config, "semantic_desc")
        row["ttt_action_desc"] = _first_yaml_value(config, "ttt_action_desc")
        row["stage_c_cache_mode"] = _first_yaml_value(config, "stage_c_cache_mode")
        row["semantic_role_policy"] = _first_yaml_value(config, "semantic_role_policy")
        row["enable_context_source_skip"] = _first_yaml_value(config, "enable_context_source_skip")
        row["native_delta_gate_mode"] = _first_yaml_value(config, "ttt_write_native_delta_gate_mode")
        row["transient_mode"] = _first_yaml_value(config, "ttt_write_gradient_reversal_transient_mode")
        manual_audit = _read_json(run_dir / "adaptive_ttt_audit.json")
        row["manual_percentage_audit_pass"] = _manual_percentage_pass(row, manual_audit)
        row.update(_extra_debug(run_dir))
        row.update(_stage_c_cache_artifact_debug(run_dir, config))
        chunk_mean = _safe_float(row.get("chunk_total_seconds_mean"), 999.0)
        wall_min = _safe_float(row.get("wall_time_min"), 999.0)
        hmc_rows = int(row.get("hmc_rows") or 0)
        frames = int(row.get("frames") or 0)
        projected = _safe_float(row.get("projected_full_wall_time_min"), 999.0)
        status_ok = row.get("status") == "done"
        full_run = _is_full(row)
        row["v56_runtime_gate_basis"] = "wall_time_min<=28, chunk_total_seconds_mean<=42, hmc_rows/frame completeness; probe_ttt_write_seconds_mean may be unavailable"
        row["full_runtime_gate_pass"] = bool(
            status_ok
            and full_run
            and wall_min <= 28.0
            and chunk_mean <= 42.0
            and hmc_rows >= 38
            and frames == 1101
        )
        row["smoke_runtime_gate_pass"] = bool(status_ok and chunk_mean <= 42.0)
        row["projected_runtime_gate_pass"] = bool(status_ok and chunk_mean <= 42.0 and projected <= 28.0)
        pose = _load_run_poses(run_dir, gt_poses, gt_pos)
        if pose.get("pose_status") != "done":
            row["segment_pose_status"] = pose.get("pose_status")
            continue
        frames = pose["frames"]
        aligned_pos = pose["aligned_pos"]
        for key, start, end in SEGMENTS:
            seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
            for metric_key, value in seg.items():
                row[f"{key}_{metric_key}"] = value


def _screen_decision(row: Mapping[str, Any], h35_704: Mapping[str, Any], track: str) -> str:
    if row.get("status") != "done":
        return "failed_or_incomplete"
    if row.get("no_chunk_policy_pass") is not True or row.get("manual_percentage_audit_pass") is not True:
        return "audit_fail_stop"
    if _safe_float(row.get("projected_full_wall_time_min"), 0.0) > 28.0:
        return "efficiency_repair_required"
    ate = _safe_float(row.get("ATE"), 999.0)
    h35 = _safe_float(h35_704.get("ATE"), 999.0)
    if ate <= h35 - 0.5:
        return "promote_full"
    rolling_ok = _safe_float(row.get("rolling100_p90"), 999.0) <= _safe_float(h35_704.get("rolling100_p90"), -999.0)
    seg_ok = _safe_float(row.get("seg1_384_700_rmse"), 999.0) <= _safe_float(h35_704.get("seg1_384_700_rmse"), -999.0)
    if ate <= h35 + 0.25 and (rolling_ok or seg_ok):
        return "borderline_full_allowed"
    if track == "A" and ate > h35 + 0.25:
        return "stop_semantic_704_regression"
    if track == "B" and ate > h35 + 0.25:
        return "stop_ttt_704_regression"
    return "stop_no_screen_signal"


def _write_registry(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, [dict(row) for row in rows])


def _plot_bar(path: Path, rows: Sequence[Mapping[str, Any]], key: str, title: str, ylabel: str) -> None:
    pts = [(str(r.get("candidate") or r.get("run_name")), _safe_float(r.get(key))) for r in rows]
    pts = [(x, y) for x, y in pts if math.isfinite(y)]
    if not pts:
        _plot_no_data(path, title, f"metric {key} unavailable")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(pts) * 1.1), 4))
    ax.bar([p[0] for p in pts], [p[1] for p in pts], color="#4C78A8")
    ax.axhline(H35_FULL_ATE, color="#F58518", linestyle="--", linewidth=1.0, label="H35 full ATE")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_segments(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    keys = ("seg0_000_384_rmse", "seg1_384_700_rmse", "seg2_700_end_rmse")
    pts = [(str(r.get("candidate") or r.get("run_name")), [_safe_float(r.get(k)) for k in keys]) for r in rows]
    pts = [(name, vals) for name, vals in pts if all(math.isfinite(v) for v in vals)]
    if not pts:
        _plot_no_data(path, title, "segment RMSE unavailable")
        return
    x = np.arange(len(keys))
    width = 0.8 / max(len(pts), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, (name, vals) in enumerate(pts):
        ax.bar(x + idx * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(pts) - 1) / 2)
    ax.set_xticklabels(["000-384", "384-700", "700-end"])
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _timeline_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for idx, row in enumerate(_read_jsonl(run_dir / "hmc_state_hash.jsonl")):
        flat: Dict[str, Any] = {"chunk_idx": int(row.get("chunk_idx", idx))}
        for node in _walk(row):
            for key in (
                "ttt_tri_replay_pos_mass",
                "ttt_tri_replay_neu_mass",
                "ttt_tri_replay_neg_mass",
                "ttt_tri_replay_stable_anchor_token_mass",
                "ttt_tri_replay_no_long_write_token_mass",
                "ttt_tri_replay_risk_token_mass",
                "ttt_tri_replay_binary_anchor_threshold",
                "ttt_tri_replay_risk_veto_threshold",
                "ttt_write_native_delta_gate_scale_mean",
                "ttt_write_native_delta_gate_cos_mean",
            ):
                if key in node and key not in flat:
                    flat[key] = node.get(key)
        rows.append(flat)
    return rows


def _plot_timeline(path: Path, rows: Sequence[Mapping[str, Any]], keys: Sequence[str], title: str) -> None:
    good = []
    for key in keys:
        pts = [(_safe_float(r.get("chunk_idx")), _safe_float(r.get(key))) for r in rows]
        pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
        if pts:
            good.append((key, pts))
    if not good:
        _plot_no_data(path, title, "timeline metric unavailable")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, pts in good:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.1, label=key)
    ax.set_xlabel("chunk_idx")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _copy_artifacts_for_best(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_a = min([r for r in rows if _track(r) == "A" and _is_full(r) and r.get("status") == "done"], key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    best_b = min([r for r in rows if _track(r) == "B" and _is_full(r) and r.get("status") == "done"], key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    best_any = best_a or best_b
    _plot_bar(out_dir / "rolling100_error_timeline.png", [r for r in (best_a, best_b) if r], "rolling100_p90", "rolling100 diagnostic", "rolling100 p90")
    _plot_segments(out_dir / "segment_error_bar.png", [r for r in (best_a, best_b) if r], "segment RMSE by best full candidates")
    for name, note in (
        ("semantic_label_overlay.png", "no per-frame semantic overlay artifact is emitted by the current runner"),
        ("D_g_base_vs_D_sem_vs_D_final.png", "only summary semantic residual fields are logged; dense per-token map is unavailable"),
        ("source_influence_mass_map.png", "attention mass summary exists but dense spatial map is unavailable"),
        ("affected_source_mask_overlay.png", "context source skip mask overlay is not emitted by the current runner"),
        ("static_anchor_rescue_overlay.png", "static anchor spatial overlay is not emitted by the current runner"),
    ):
        _plot_no_data(out_dir / name, name, note)
    if best_any:
        timeline = _timeline_rows(Path(str(best_any.get("run_dir"))))
    else:
        timeline = []
    _plot_timeline(
        out_dir / "role_mass_timeline.png",
        timeline,
        ("ttt_tri_replay_pos_mass", "ttt_tri_replay_neu_mass", "ttt_tri_replay_neg_mass", "ttt_tri_replay_stable_anchor_token_mass", "ttt_tri_replay_no_long_write_token_mass"),
        "role mass timeline",
    )
    _plot_timeline(
        out_dir / "threshold_timeline.png",
        timeline,
        ("ttt_tri_replay_binary_anchor_threshold", "ttt_tri_replay_risk_veto_threshold"),
        "threshold timeline",
    )
    for name in (
        "post_zp_delta_norm_by_chunk.png",
        "branch_layer_delta_heatmap.png",
        "candidate_native_cosine_timeline.png",
        "long_short_update_energy_timeline.png",
        "segment_error_timeline.png",
    ):
        _plot_no_data(out_dir / name, name, "requested low-level figure is unavailable unless detailed tensor traces are enabled")


def _summarize_group(rows: Sequence[Mapping[str, Any]], track: str, full: bool) -> List[Mapping[str, Any]]:
    return [
        r
        for r in rows
        if _track(r) == track and ((_is_full(r) if full else _is_704(r)))
    ]


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    if not rows:
        return ["无 landed run。"]
    lines = ["| " + " | ".join(title for title, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        vals = []
        for _title, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(_fmt(value))
            elif key in {"ATE", "Rot", "FinalErr", "wall_time_min", "chunk_total_seconds_mean"}:
                vals.append(_fmt(value))
            else:
                vals.append(str(value if value is not None else "NA"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _write_failure_reports(out_dir: Path, a704: Sequence[Mapping[str, Any]], b704: Sequence[Mapping[str, Any]]) -> None:
    if a704 and not any(str(r.get("screen_decision")) in {"promote_full", "borderline_full_allowed"} for r in a704):
        lines = [
            "# semantic_failure_report",
            "",
            "Track A 704F 全部未达到 full gate。以下只使用 landed debug 字段，不补造缺失信息。",
            "",
            "| candidate | ATE | delta_vs_H35_704 | semantic_label_count_mean | fallback_ratio | affected_source_tokens | mass_removed_before | mass_removed_after | decision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in a704:
            lines.append(
                f"| {row.get('candidate')} | {_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_H35_704'))} | "
                f"{_fmt(row.get('semantic_label_count_mean'))} | {_fmt(row.get('semantic_label_fallback_ratio_mean'))} | "
                f"{_fmt(row.get('affected_source_token_count_max'))} | {_fmt(row.get('source_influence_mass_removed_before_mean'))} | "
                f"{_fmt(row.get('source_influence_mass_removed_after_mean'))} | {row.get('screen_decision')} |"
            )
        lines.extend(["", "判断: 若 source mass/label 字段为 NA，则失败分析边界是 runner 没有写出该证据，而不是证据为 0。"])
        _write_text(out_dir / "semantic_failure_report.md", lines)
    if b704 and not any(str(r.get("screen_decision")) in {"promote_full", "borderline_full_allowed"} for r in b704):
        lines = [
            "# ttt_action_failure_report",
            "",
            "Track B 704F 全部未达到 full gate。以下只使用 landed debug 字段。",
            "",
            "| candidate | ATE | delta_vs_H35_704 | stable_mass | risk_mass | no_long_mass | threshold | native_gate_scale | dual_override | decision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in b704:
            lines.append(
                f"| {row.get('candidate')} | {_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_H35_704'))} | "
                f"{_fmt(row.get('stable_anchor_token_mass_mean'))} | {_fmt(row.get('risk_token_mass_mean'))} | "
                f"{_fmt(row.get('no_long_write_token_mass_mean'))} | {_fmt(row.get('role_threshold_mean'))} | "
                f"{_fmt(row.get('native_delta_gate_scale_mean'))} | {_fmt(row.get('dual_lifetime_long_override_count'))} | {row.get('screen_decision')} |"
            )
        lines.extend(["", "判断: 不再扫 fixed threshold；下一步应优先看 mass 是否退化和 delta direction 是否偏离。"])
        _write_text(out_dir / "ttt_action_failure_report.md", lines)


def _write_final_docs(
    result_root: Path,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    h35_ref_full: Mapping[str, Any],
    h35_ref_704: Mapping[str, Any],
) -> None:
    h35_repeat = min([r for r in rows if _candidate(r) == "H35" and _is_full(r)], key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    a704 = _summarize_group(rows, "A", full=False)
    b704 = _summarize_group(rows, "B", full=False)
    afull = _summarize_group(rows, "A", full=True)
    bfull = _summarize_group(rows, "B", full=True)
    best_a = min(afull, key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    best_b = min(bfull, key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    best_any = min([r for r in (best_a, best_b) if r], key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)

    drift = None
    if h35_repeat:
        drift = _safe_float(h35_repeat.get("ATE")) - _safe_float(h35_ref_full.get("ATE"))

    exec_lines = [
        "# ACL2 v56 H35 SemanticBoost NewTTTAction Fast 执行日志",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"结果复盘: `{DOC_REVIEW.relative_to(REPO_ROOT)}`",
        f"工作目录: `{REPO_ROOT}`",
        f"结果根目录: `{result_root.relative_to(REPO_ROOT)}`",
        "",
        "## 执行边界",
        "",
        "- 所有指标只来自落盘 artifact、`01.log`、`hmc_state_hash.jsonl`、`wall_time_summary.json` 或 evaluation 输出。",
        "- 不使用 absolute chunk-id policy，不使用手工 tri replay percentage。",
        "- Track A 使用 Stage C cache 时强制 `stage_c_cache_mode=read` 和 `stage_c_cache_require_hit=1`。",
        "- 单条 full run runtime gate 为 wall time <= 28min；`probe_ttt_write_seconds_mean` 缺失时保持 unavailable。",
        "",
        "## 代码与工具修改",
        "",
        "| 文件 | 修改内容 |",
        "|---|---|",
        "| `loger/pipeline/ttt_write_controller.py` | 新增 v56 binary stable-anchor / risk-veto role modes，使用当前 chunk Otsu/median fallback 阈值，不使用 top percentage；记录 stable/risk/no-long-write mass。 |",
        "| `tools/run_v56_h35_semantic_ttt_action_candidate.sh` | 新增 v56 统一 runner，覆盖 H35 repeat、Track A A1-A4、Track B B1-B4 和可选 combo。 |",
        "| `tools/v56_experiment_report.py` | 新增 artifact-only 报告工具，生成 registry、failure report、diagnostic figures、执行日志和复盘。 |",
        "",
        "验证命令:",
        "",
        "```bash",
        f"{sys.executable} -m py_compile loger/pipeline/ttt_write_controller.py tools/v56_experiment_report.py",
        "bash -n tools/run_v56_h35_semantic_ttt_action_candidate.sh",
        "```",
        "",
        "## 运行命令清单",
        "",
        "| row | run | GPU | frames | status | wall min | ATE | 输出目录 |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        exec_lines.append(
            f"| `{row.get('row')}` | `{row.get('run_name')}` | {row.get('gpu') or _first_yaml_value(Path(str(row.get('run_dir'))) / 'effective_config.yaml', 'gpu')} | "
            f"{row.get('frames')} | {row.get('status')} | {_fmt(row.get('wall_time_min'))} | {_fmt(row.get('ATE'))} | `{Path(str(row.get('run_dir'))).relative_to(REPO_ROOT)}` |"
        )
    exec_lines.extend([
        "",
        "复现单条 run 的模板:",
        "",
        "```bash",
        "tools/run_v56_h35_semantic_ttt_action_candidate.sh <GPU> <ROW> <RUN_NAME>",
        "```",
        "",
        "每个 run 目录都写有 `effective_config.yaml`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json` 和 `reproduce_command.sh`。",
        "",
        "## 报告生成",
        "",
        "```bash",
        f"{sys.executable} tools/v56_experiment_report.py",
        "```",
    ])
    _write_text(DOC_EXEC, exec_lines)

    review_lines = [
        "# ACL2 v56 H35 SemanticBoost NewTTTAction Fast 实验结果复盘",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"执行日志: `{DOC_EXEC.relative_to(REPO_ROOT)}`",
        f"结果根目录: `{result_root.relative_to(REPO_ROOT)}`",
        "",
        "结论先行: 本文件只基于已落盘结果生成；未运行或未写出的指标保持 NA/unavailable。",
        "",
        "## Phase 0 H35 baseline",
        "",
        f"- H35 landed reference full ATE: `{_fmt(h35_ref_full.get('ATE'))}`。",
        f"- H35 landed reference 704F ATE: `{_fmt(h35_ref_704.get('ATE'))}`。",
    ]
    if h35_repeat:
        review_lines.extend([
            f"- H35 repeat full ATE: `{_fmt(h35_repeat.get('ATE'))}`，drift vs landed H35: `{_fmt(drift)}`。",
            f"- H35 repeat Rot/FinalErr: `{_fmt(h35_repeat.get('Rot'))}` / `{_fmt(h35_repeat.get('FinalErr'))}`。",
            f"- H35 repeat runtime: wall `{_fmt(h35_repeat.get('wall_time_min'))}min`, chunk mean `{_fmt(h35_repeat.get('chunk_total_seconds_mean'))}s`, probe TTT mean `{_fmt(h35_repeat.get('probe_ttt_write_seconds_mean'))}`。",
        ])
    else:
        review_lines.append("- H35 repeat full 未完成或未发现。")
    review_lines.extend([
        "",
        "## Track A 704F screen",
        "",
        *_md_table(a704, (("candidate", "candidate"), ("ATE", "ATE"), ("delta_vs_H35_704", "delta_vs_H35_704"), ("stage_c_hit", "stage_c_cache_hit_rate"), ("sem_labels", "semantic_label_count_mean"), ("source_tokens", "affected_source_token_count_max"), ("decision", "screen_decision"))),
        "",
        "## Track A full",
        "",
        *_md_table(afull, (("candidate", "candidate"), ("ATE", "ATE"), ("delta_vs_H35", "delta_vs_H35_full"), ("Rot", "Rot"), ("FinalErr", "FinalErr"), ("wall_min", "wall_time_min"), ("progress", "minimum_progress_pass"))),
        "",
        "## Track B 704F screen",
        "",
        *_md_table(b704, (("candidate", "candidate"), ("ATE", "ATE"), ("delta_vs_H35_704", "delta_vs_H35_704"), ("stable_mass", "stable_anchor_token_mass_mean"), ("risk_mass", "risk_token_mass_mean"), ("no_long", "no_long_write_token_mass_mean"), ("decision", "screen_decision"))),
        "",
        "## Track B full",
        "",
        *_md_table(bfull, (("candidate", "candidate"), ("ATE", "ATE"), ("delta_vs_H35", "delta_vs_H35_full"), ("Rot", "Rot"), ("FinalErr", "FinalErr"), ("wall_min", "wall_time_min"), ("progress", "minimum_progress_pass"))),
        "",
        "## 关键分析",
        "",
    ])
    if best_a:
        review_lines.append(
            f"- Track A best full 是 `{best_a.get('candidate')}`，ATE `{_fmt(best_a.get('ATE'))}`，"
            f"相对 H35 full `{_fmt(best_a.get('delta_vs_H35_full'))}`。"
        )
    else:
        review_lines.append("- Track A 没有 landed full；若 704F 未过 gate，按计划不继续 full。")
    if best_b:
        review_lines.append(
            f"- Track B best full 是 `{best_b.get('candidate')}`，ATE `{_fmt(best_b.get('ATE'))}`，"
            f"相对 H35 full `{_fmt(best_b.get('delta_vs_H35_full'))}`。"
        )
    else:
        review_lines.append("- Track B 没有 landed full；若 704F 未过 gate，按计划不继续 full。")
    if best_any:
        review_lines.append(
            f"- 全部 v56 best full 是 `{best_any.get('candidate')}`，ATE `{_fmt(best_any.get('ATE'))}`。"
        )
    review_lines.extend([
        "- `probe_ttt_write_seconds_mean` 如果为 NA，原因是该 run 未写出 `timing_summary.json` 中的 probe 字段；本报告没有替代或补造。",
        "- requested dense overlays 若显示 no-data，是因为当前 runner 未落盘对应空间图，不代表对应量为 0。",
        "",
        "## 判定",
        "",
    ])
    if best_a and _safe_float(best_a.get("ATE"), 999.0) <= H35_SUCCESS_A:
        review_lines.append("- Track A success: semantic 在 H35 上达到 >=2m 改善。")
    elif best_a and _safe_float(best_a.get("ATE"), 999.0) <= H35_STRONG:
        review_lines.append("- Track A strong signal: semantic 在 H35 上达到 >=1m 改善。")
    elif best_a and _safe_float(best_a.get("ATE"), 999.0) <= H35_MIN_PROGRESS:
        review_lines.append("- Track A minimum progress: semantic 在 H35 上达到 >=0.5m 改善。")
    else:
        review_lines.append("- Track A 未证明 semantic full ATE 相对 H35 改善 >=0.5m。")
    if best_b and _safe_float(best_b.get("ATE"), 999.0) <= H35_STRONG:
        review_lines.append("- Track B success: new TTT action 相对 H35 达到 >=1m 改善。")
    elif best_b and _safe_float(best_b.get("ATE"), 999.0) <= H35_MIN_PROGRESS:
        review_lines.append("- Track B minimum progress: new TTT action 相对 H35 达到 >=0.5m 改善。")
    else:
        review_lines.append("- Track B 未证明 new TTT action full ATE 相对 H35 改善 >=0.5m。")
    review_lines.extend([
        "",
        "## 产物",
        "",
        f"- final report: `{(out_dir / 'v56_final_report.md').relative_to(REPO_ROOT)}`",
        f"- registries: `{out_dir.relative_to(REPO_ROOT)}`",
        f"- figures: `{(out_dir / 'figures').relative_to(REPO_ROOT)}`",
    ])
    _write_text(DOC_REVIEW, review_lines)
    _write_text(out_dir / "v56_final_report.md", review_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--gt", default=DEFAULT_GT, type=Path)
    parser.add_argument("--h35-full", default=DEFAULT_H35_FULL, type=Path)
    parser.add_argument("--h35-704", default=DEFAULT_H35_704, type=Path)
    args = parser.parse_args()

    result_root = args.result_root
    out_dir = args.out_dir or result_root / "report_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _iter_run_dirs([result_root])
    rows = _summarize_runs(run_dirs, args.gt)
    _augment_rows(rows, args.gt)

    h35_rows = _summarize_runs([args.h35_full, args.h35_704], args.gt)
    _augment_rows(h35_rows, args.gt)
    h35_full = next((r for r in h35_rows if _is_full(r)), {})
    h35_704 = next((r for r in h35_rows if _is_704(r)), {})

    for row in rows:
        row["delta_vs_H35_full"] = _safe_float(row.get("ATE")) - _safe_float(h35_full.get("ATE")) if row.get("ATE") is not None else None
        row["delta_vs_H35_704"] = _safe_float(row.get("ATE")) - _safe_float(h35_704.get("ATE")) if row.get("ATE") is not None else None
        if _is_704(row) and _track(row) in {"A", "B"}:
            row["screen_decision"] = _screen_decision(row, h35_704, _track(row))
        row["minimum_progress_pass"] = bool(_is_full(row) and _safe_float(row.get("ATE"), 999.0) <= H35_MIN_PROGRESS)
        row["strong_signal_pass"] = bool(_is_full(row) and _safe_float(row.get("ATE"), 999.0) <= H35_STRONG)
        row["success_pass"] = bool(_is_full(row) and _safe_float(row.get("ATE"), 999.0) <= H35_SUCCESS_A)
        row["runtime_wall_pass"] = bool(not _is_full(row) or _safe_float(row.get("wall_time_min"), 999.0) <= 28.0)

    _write_registry(out_dir / "v56_all_registry.csv", rows)
    _write_registry(out_dir / "v56_h35_reference_registry.csv", h35_rows)
    _write_registry(out_dir / "v56_phase0_h35_repeat_registry.csv", [r for r in rows if _candidate(r) == "H35"])
    _write_registry(out_dir / "v56_track_a_smoke_registry.csv", [r for r in rows if _track(r) == "A" and _is_smoke(r)])
    _write_registry(out_dir / "v56_track_a_704f_registry.csv", [r for r in rows if _track(r) == "A" and _is_704(r)])
    _write_registry(out_dir / "v56_track_a_full_registry.csv", [r for r in rows if _track(r) == "A" and _is_full(r)])
    _write_registry(out_dir / "v56_track_b_smoke_registry.csv", [r for r in rows if _track(r) == "B" and _is_smoke(r)])
    _write_registry(out_dir / "v56_track_b_704f_registry.csv", [r for r in rows if _track(r) == "B" and _is_704(r)])
    _write_registry(out_dir / "v56_track_b_full_registry.csv", [r for r in rows if _track(r) == "B" and _is_full(r)])

    figs = out_dir / "figures"
    _plot_bar(figs / "full_ate_bar.png", [r for r in rows if _is_full(r)], "ATE", "full ATE", "ATE")
    _plot_bar(figs / "screen_704_ate_bar.png", [r for r in rows if _is_704(r)], "ATE", "704F screen ATE", "ATE")
    _copy_artifacts_for_best(rows, figs)
    _write_failure_reports(out_dir, [r for r in rows if _track(r) == "A" and _is_704(r)], [r for r in rows if _track(r) == "B" and _is_704(r)])

    summary = {
        "result_root": str(result_root),
        "run_count": len(rows),
        "h35_reference_full_ate": h35_full.get("ATE"),
        "h35_reference_704_ate": h35_704.get("ATE"),
        "best_track_a_full_ate": min((_safe_float(r.get("ATE"), 999.0) for r in rows if _track(r) == "A" and _is_full(r)), default=None),
        "best_track_b_full_ate": min((_safe_float(r.get("ATE"), 999.0) for r in rows if _track(r) == "B" and _is_full(r)), default=None),
    }
    _write_json(out_dir / "v56_summary.json", summary)
    _write_final_docs(result_root, out_dir, rows, h35_full, h35_704)


if __name__ == "__main__":
    main()
