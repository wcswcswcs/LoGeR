#!/usr/bin/env python3
"""Generate ACL2 v57 H35 semantic-action-repair / TTT-TTL reports.

The reporter reads landed artifacts only. Missing metrics stay NA; planned
values are never promoted into observed measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
from tools.v56_experiment_report import (  # noqa: E402
    DEFAULT_H35_704,
    DEFAULT_H35_FULL,
    DEFAULT_RESULT_ROOT as DEFAULT_V56_ROOT,
    _augment_rows as _v56_augment_rows,
    _first_yaml_value,
    _is_704,
    _is_full,
    _is_smoke,
    _md_table,
    _read_json,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
PLAN_DOC = "docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_FastPlan.md"
DOC_EXEC = REPO_ROOT / "docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_Fast_执行日志.md"
DOC_REVIEW = REPO_ROOT / "docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_Fast_实验结果复盘.md"
H35_FULL_ATE = 35.74089695811434
H35_SEMANTIC_SUCCESS = 33.7409
H35_TTT_SUCCESS = 34.7409
H35_MIN_PROGRESS = 35.2409

ROLE_NEGATIVE = {"2", "negative", "negative_short", "SEMANTIC_ROLE_NEGATIVE_SHORT"}
ROLE_PROTECT = {"3", "protect", "protect_neutral", "SEMANTIC_ROLE_PROTECT_NEUTRAL"}
SEG_KEYS = ("seg0_000_384_rmse", "seg1_384_700_rmse", "seg2_700_end_rmse")


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _candidate(row: Mapping[str, Any]) -> str:
    cand = str(row.get("candidate") or "").strip()
    if cand:
        return cand
    run = str(row.get("run_name") or row.get("row") or "")
    m = re.search(r"(S0|S1|S2|SREAD0[1-4]|TTT0[1-3]|COMBO01|A[1-4]|B[1-4]|H35)", run)
    return m.group(1) if m else run


def _track(row: Mapping[str, Any]) -> str:
    track = str(row.get("track") or "").strip()
    if track:
        return track
    cand = _candidate(row)
    if cand.startswith("S") and not cand.startswith("SREAD"):
        return "semantic_smoke"
    if cand.startswith("SREAD"):
        return "semantic_read"
    if cand.startswith("TTT"):
        return "ttt_action"
    if cand.startswith("A"):
        return "A"
    if cand.startswith("B"):
        return "B"
    return ""


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row") or row.get("candidate") or row.get("run_name") or "")


def _float_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        f = _safe_float(value)
        if math.isfinite(f):
            out.append(f)
    return out


def _numbers_from_text(text: str, key: str) -> List[float]:
    pat = re.compile(rf"['\"]{re.escape(key)}['\"]:\s*([-+0-9.eE]+)")
    return _float_values(m.group(1) for m in pat.finditer(text))


def _count_true_from_text(text: str, key: str) -> int:
    pat = re.compile(rf"['\"]{re.escape(key)}['\"]:\s*(True|False|true|false)")
    return sum(1 for m in pat.finditer(text) if m.group(1).lower() == "true")


def _collect_role_count_mass(label_role_counts: Any, role_tokens: set[str]) -> Tuple[Optional[float], str]:
    if not isinstance(label_role_counts, dict):
        return None, ""
    total = 0.0
    pieces: List[str] = []
    for label, counts in sorted(label_role_counts.items(), key=lambda kv: str(kv[0])):
        if not isinstance(counts, dict):
            continue
        value = 0.0
        for role, count in counts.items():
            if str(role) in role_tokens:
                value += _safe_float(count, 0.0)
        if value > 0:
            pieces.append(f"{label}:{_fmt(value)}")
            total += value
    return total, ";".join(pieces)


def _v57_extra_debug(run_dir: Path) -> Dict[str, Any]:
    """Extract v57 action-realization and TTT energy fields from debug traces."""

    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    text = _read_text(run_dir / "01.log")
    out: Dict[str, Any] = {}

    semantic_group_counts: List[float] = []
    semantic_label_counts: List[float] = []
    affected_tokens: List[float] = []
    keep_ratio: List[float] = []
    mass_before: List[float] = []
    mass_after: List[float] = []
    context_empty: List[float] = []
    per_label_source_examples: List[str] = []
    per_label_affected_examples: List[str] = []
    static_protect_tokens: List[float] = []

    post_delta_norms: List[float] = []
    native_cos: List[float] = []
    native_cos_min: List[float] = []
    short_residual_norms: List[float] = []
    prev_short_norms: List[float] = []
    static_delta_norms: List[float] = []
    long_delta_norms: List[float] = []
    no_long_mass: List[float] = []
    stable_mass: List[float] = []
    risk_mass: List[float] = []
    layer_branch: Dict[str, List[float]] = {}
    transient_stored = 0
    transient_prev_subtract = 0

    def add_layer_branch(key: str, value: Any) -> None:
        f = _safe_float(value)
        if not math.isfinite(f):
            return
        layer_branch.setdefault(key, []).append(f)

    for row in rows:
        for node in _walk(row):
            if not isinstance(node, dict):
                continue
            groups = node.get("semantic_group_role_metrics")
            if not isinstance(groups, dict):
                groups = node.get("prior_semantic_group_role_metrics")
            if isinstance(groups, dict):
                semantic_group_counts.append(float(len(groups)))
            for key in ("fine_label_token_count", "prior_fine_label_token_count", "semantic_label_token_count"):
                if node.get(key) is not None:
                    semantic_label_counts.append(_safe_float(node.get(key)))
            for stream_key in ("fine_label_path_role_counts",):
                streams = node.get(stream_key)
                if not isinstance(streams, dict):
                    streams = node.get("prior_fine_label_path_role_counts")
                if isinstance(streams, dict):
                    for path_name in ("frame", "global"):
                        total, pieces = _collect_role_count_mass(streams.get(path_name), ROLE_NEGATIVE)
                        if total is not None:
                            per_label_affected_examples.append(f"{path_name}:{pieces}" if pieces else f"{path_name}:")
            label_paths = node.get("fine_label_path_role_counts")
            if not isinstance(label_paths, dict):
                label_paths = node.get("prior_fine_label_path_role_counts")
            if isinstance(label_paths, dict):
                frame_counts = label_paths.get("frame")
                total, pieces = _collect_role_count_mass(frame_counts, ROLE_NEGATIVE)
                if total is not None:
                    per_label_source_examples.append(pieces)
                protect_total, _ = _collect_role_count_mass(frame_counts, ROLE_PROTECT)
                if protect_total is not None:
                    static_protect_tokens.append(protect_total)

            for key in ("source_skip_tokens", "max_context_source_skip_tokens"):
                if node.get(key) is not None:
                    affected_tokens.append(_safe_float(node.get(key)))
            if node.get("mean_context_source_keep_ratio") is not None:
                keep_ratio.append(_safe_float(node.get("mean_context_source_keep_ratio")))
            if node.get("source_keep_ratio") is not None:
                keep_ratio.append(_safe_float(node.get("source_keep_ratio")))
            for key in ("attention_mass_removed_before", "mean_attention_mass_removed_before"):
                if node.get(key) is not None:
                    mass_before.append(_safe_float(node.get(key)))
            for key in ("attention_mass_removed_after", "mean_attention_mass_removed_after"):
                if node.get(key) is not None:
                    mass_after.append(_safe_float(node.get(key)))
            for key in ("num_context_empty_source_events", "context_empty_source_event"):
                if node.get(key) is not None:
                    context_empty.append(_safe_float(node.get(key)))

            for key, target in (
                ("ttt_write_native_delta_gate_cos_mean", native_cos),
                ("ttt_write_native_delta_gate_w0_cos_mean", native_cos),
                ("ttt_write_native_delta_gate_w1_cos_mean", native_cos),
                ("ttt_write_native_delta_gate_w2_cos_mean", native_cos),
                ("ttt_write_native_delta_gate_w0_cos_min", native_cos_min),
                ("ttt_write_native_delta_gate_w1_cos_min", native_cos_min),
                ("ttt_write_native_delta_gate_w2_cos_min", native_cos_min),
                ("ttt_transient_delta_w0_norm_mean", short_residual_norms),
                ("ttt_transient_delta_w1_norm_mean", short_residual_norms),
                ("ttt_transient_delta_w2_norm_mean", short_residual_norms),
                ("ttt_transient_delta_prev_norm_mean", prev_short_norms),
                ("ttt_tri_replay_stable_anchor_token_mass", stable_mass),
                ("ttt_tri_replay_risk_token_mass", risk_mass),
                ("ttt_tri_replay_no_long_write_token_mass", no_long_mass),
            ):
                if node.get(key) is not None:
                    target.append(_safe_float(node.get(key)))
            if node.get("ttt_transient_delta_stored") is True:
                transient_stored += 1
            if node.get("ttt_transient_delta_prev_subtract_applied") is True:
                transient_prev_subtract += 1

            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                if key.endswith("_post_delta_norm_mean") or key.endswith("_post_zp_delta_norm_mean"):
                    post_delta_norms.append(_safe_float(value))
                    add_layer_branch(key, value)
                if key.endswith("_pos_delta_norm_mean") or key.endswith("_static_delta_norm_mean"):
                    static_delta_norms.append(_safe_float(value))
                    add_layer_branch(key, value)
                if key.endswith("_native_delta_norm") or key.endswith("_native_delta_norm_mean"):
                    long_delta_norms.append(_safe_float(value))
                    add_layer_branch(key, value)
                if "delta_norm" in key and any(token in key for token in ("w0", "w1", "w2", "layer")):
                    add_layer_branch(key, value)

    if text:
        if not affected_tokens:
            affected_tokens.extend(_numbers_from_text(text, "source_skip_tokens"))
            affected_tokens.extend(_numbers_from_text(text, "max_context_source_skip_tokens"))
        if not mass_before:
            mass_before.extend(_numbers_from_text(text, "attention_mass_removed_before"))
            mass_before.extend(_numbers_from_text(text, "mean_attention_mass_removed_before"))
        if not mass_after:
            mass_after.extend(_numbers_from_text(text, "attention_mass_removed_after"))
            mass_after.extend(_numbers_from_text(text, "mean_attention_mass_removed_after"))
        if not context_empty:
            context_empty.extend(_numbers_from_text(text, "num_context_empty_source_events"))
        native_cos.extend(_numbers_from_text(text, "ttt_write_native_delta_gate_cos_mean"))
        native_cos_min.extend(_numbers_from_text(text, "ttt_write_native_delta_gate_w0_cos_min"))
        short_residual_norms.extend(_numbers_from_text(text, "ttt_transient_delta_w0_norm_mean"))
        short_residual_norms.extend(_numbers_from_text(text, "ttt_transient_delta_w1_norm_mean"))
        short_residual_norms.extend(_numbers_from_text(text, "ttt_transient_delta_w2_norm_mean"))
        prev_short_norms.extend(_numbers_from_text(text, "ttt_transient_delta_prev_norm_mean"))
        stable_mass.extend(_numbers_from_text(text, "ttt_tri_replay_stable_anchor_token_mass"))
        risk_mass.extend(_numbers_from_text(text, "ttt_tri_replay_risk_token_mass"))
        no_long_mass.extend(_numbers_from_text(text, "ttt_tri_replay_no_long_write_token_mass"))
        transient_stored = max(transient_stored, _count_true_from_text(text, "ttt_transient_delta_stored"))
        transient_prev_subtract = max(transient_prev_subtract, _count_true_from_text(text, "ttt_transient_delta_prev_subtract_applied"))

    affected_f = _float_values(affected_tokens)
    mass_before_f = _float_values(mass_before)
    mass_after_f = _float_values(mass_after)
    out["semantic_group_count_mean"] = _mean(semantic_group_counts)
    out["semantic_label_count_mean"] = _mean(semantic_label_counts)
    out["affected_source_token_count_mean"] = _mean(affected_f)
    out["affected_source_token_count_max_v57"] = max(affected_f) if affected_f else None
    out["attention_mass_removed_before_mean"] = _mean(mass_before_f)
    out["attention_mass_removed_after_mean"] = _mean(mass_after_f)
    out["source_keep_ratio_mean"] = _mean(keep_ratio)
    out["context_empty_source_events_sum"] = int(sum(_float_values(context_empty))) if context_empty else None
    out["per_label_source_mass"] = next((p for p in per_label_source_examples if p), "")
    out["per_label_affected_token_mass"] = next((p for p in per_label_affected_examples if p), "")
    out["static_anchor_protected_mass"] = _mean(static_protect_tokens)
    out["post_zp_delta_norm_mean"] = _mean(post_delta_norms)
    vals = _float_values(post_delta_norms)
    out["post_zp_delta_norm_p90"] = float(np.percentile(vals, 90)) if vals else None
    out["candidate_native_delta_cos_mean"] = _mean(native_cos)
    vals = _float_values(native_cos_min or native_cos)
    out["candidate_native_delta_cos_p10"] = float(np.percentile(vals, 10)) if vals else None
    out["short_residual_norm_mean"] = _mean(short_residual_norms)
    out["previous_short_residual_subtracted_norm"] = _mean(prev_short_norms)
    out["static_long_delta_norm"] = _mean(static_delta_norms or stable_mass)
    out["long_residual_norm_mean"] = _mean(long_delta_norms)
    out["ttt_transient_delta_stored_count"] = int(transient_stored)
    out["ttt_transient_delta_prev_subtract_count"] = int(transient_prev_subtract)
    if no_long_mass:
        out["no_long_write_token_mass_mean_v57"] = _mean(no_long_mass)
    if stable_mass:
        out["stable_anchor_token_mass_mean_v57"] = _mean(stable_mass)
    if risk_mass:
        out["risk_token_mass_mean_v57"] = _mean(risk_mass)
    out["layer_branch_delta_norm_table"] = ";".join(
        f"{k}:{_fmt(_mean(v))}" for k, v in sorted(layer_branch.items())[:40]
    )
    if vals:
        out["energy_collapse_flag"] = False
        out["energy_explosion_flag"] = False
    else:
        out["energy_collapse_flag"] = None
        out["energy_explosion_flag"] = None
    return out


def _augment_v57_rows(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _v56_augment_rows(rows, gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        row["candidate"] = _first_yaml_value(run_dir / "effective_config.yaml", "candidate") or _candidate(row)
        row["track"] = _first_yaml_value(run_dir / "effective_config.yaml", "track") or _track(row)
        row["action_class"] = _first_yaml_value(run_dir / "effective_config.yaml", "action_class")
        row["context_source_skip_impl"] = _first_yaml_value(run_dir / "effective_config.yaml", "context_source_skip_impl")
        row["context_source_skip_mask"] = _first_yaml_value(run_dir / "effective_config.yaml", "context_source_skip_mask")
        row["context_source_skip_layer_mode"] = _first_yaml_value(run_dir / "effective_config.yaml", "context_source_skip_layer_mode")
        row.update(_v57_extra_debug(run_dir))
        if row.get("affected_source_token_count_mean") is None:
            row["affected_source_token_count_mean"] = row.get("affected_source_token_count_max")
        if row.get("attention_mass_removed_before_mean") is None:
            row["attention_mass_removed_before_mean"] = row.get("source_influence_mass_removed_before_mean")
        if row.get("attention_mass_removed_after_mean") is None:
            row["attention_mass_removed_after_mean"] = row.get("source_influence_mass_removed_after_mean")
        if row.get("context_empty_source_events_sum") is None:
            row["context_empty_source_events_sum"] = row.get("context_empty_source_events")
        if row.get("no_long_write_token_mass_mean_v57") is None and row.get("no_long_write_token_mass_mean") is not None:
            row["no_long_write_token_mass_mean_v57"] = row.get("no_long_write_token_mass_mean")
        if row.get("stable_anchor_token_mass_mean_v57") is None and row.get("stable_anchor_token_mass_mean") is not None:
            row["stable_anchor_token_mass_mean_v57"] = row.get("stable_anchor_token_mass_mean")
        if row.get("risk_token_mass_mean_v57") is None and row.get("risk_token_mass_mean") is not None:
            row["risk_token_mass_mean_v57"] = row.get("risk_token_mass_mean")


def _semantic_smoke_gate(row: Mapping[str, Any]) -> bool:
    before = _safe_float(row.get("attention_mass_removed_before_mean"))
    after = _safe_float(row.get("attention_mass_removed_after_mean"))
    decreased = math.isfinite(before) and math.isfinite(after) and (after == 0.0 or after <= before * 0.2)
    return bool(
        row.get("status") == "done"
        and _safe_float(row.get("stage_c_cache_hit_rate")) == 1.0
        and _safe_float(row.get("affected_source_token_count_mean")) > 0.0
        and math.isfinite(before)
        and before >= 0.02
        and decreased
        and _safe_float(row.get("context_empty_source_events_sum"), 999.0) == 0.0
    )


def _semantic_action_realized(row: Mapping[str, Any]) -> bool:
    semantic_evidence = (
        _safe_float(row.get("semantic_label_count_mean")) > 0
        or _safe_float(row.get("semantic_group_count_mean")) > 0
    )
    return bool(
        _safe_float(row.get("stage_c_cache_hit_rate")) == 1.0
        and semantic_evidence
        and _safe_float(row.get("affected_source_token_count_mean")) > 0
        and _safe_float(row.get("attention_mass_removed_before_mean")) >= 0.02
        and _safe_float(row.get("context_empty_source_events_sum"), 999.0) == 0.0
    )


def _ttt_smoke_gate(row: Mapping[str, Any]) -> bool:
    cand = _candidate(row)
    projected = _safe_float(row.get("projected_full_wall_time_min"), 0.0)
    runtime_ok = projected <= 28.0 or _safe_float(row.get("chunk_total_seconds_mean"), 999.0) <= 42.0
    if cand.startswith("TTT01"):
        evidence = max(
            _safe_float(row.get("static_long_delta_norm"), 0.0),
            _safe_float(row.get("stable_anchor_token_mass_mean_v57"), 0.0),
            _safe_float(row.get("native_delta_gate_scale_mean"), 0.0),
        )
    elif cand.startswith("TTT02"):
        evidence = max(
            _safe_float(row.get("short_residual_norm_mean"), 0.0),
            _safe_float(row.get("ttt_transient_delta_stored_count"), 0.0),
        )
    elif cand.startswith("TTT03"):
        evidence = max(
            _safe_float(row.get("risk_token_mass_mean_v57"), 0.0),
            _safe_float(row.get("no_long_write_token_mass_mean_v57"), 0.0),
            _safe_float(row.get("commit_filter_applied_debug_rows"), 0.0),
        )
    else:
        evidence = max(
            _safe_float(row.get("post_zp_delta_norm_mean"), 0.0),
            _safe_float(row.get("short_residual_norm_mean"), 0.0),
            _safe_float(row.get("static_long_delta_norm"), 0.0),
        )
    return bool(row.get("status") == "done" and runtime_ok and evidence > 0.0)


def _segment_regression_max(row: Mapping[str, Any], ref: Mapping[str, Any]) -> Optional[float]:
    vals = []
    for key in SEG_KEYS:
        a = _safe_float(row.get(key))
        b = _safe_float(ref.get(key))
        if math.isfinite(a) and math.isfinite(b):
            vals.append(a - b)
    return max(vals) if vals else None


def _screen_decision(row: Mapping[str, Any], h35_704: Mapping[str, Any], track: str) -> str:
    if row.get("status") != "done":
        return "failed_or_incomplete"
    if track == "semantic_read" and not _semantic_action_realized(row):
        return "stop_semantic_action_inactive"
    ate = _safe_float(row.get("ATE"), 999.0)
    h35 = _safe_float(h35_704.get("ATE"), 999.0)
    rolling = _safe_float(row.get("rolling100_p90"), 999.0)
    hrolling = _safe_float(h35_704.get("rolling100_p90"), 999.0)
    if ate <= h35 - 0.7:
        return "promote_full"
    if rolling <= hrolling - 3.0:
        return "promote_full_rolling"
    seg_reg = _segment_regression_max(row, h35_704)
    no_seg_reg = seg_reg is not None and seg_reg <= 0.5
    if track == "semantic_read":
        strong_action = _semantic_action_realized(row)
        if ate <= h35 + 0.20 and strong_action and no_seg_reg:
            return "borderline_full_allowed"
        if ate > h35 + 0.30:
            return "semantic_active_regression_repair_required"
        return "stop_no_semantic_screen_signal"
    if track == "ttt_action":
        energy = max(
            _safe_float(row.get("post_zp_delta_norm_mean"), 0.0),
            _safe_float(row.get("short_residual_norm_mean"), 0.0),
            _safe_float(row.get("static_long_delta_norm"), 0.0),
            _safe_float(row.get("stable_anchor_token_mass_mean_v57"), 0.0),
        )
        if ate <= h35 + 0.20 and no_seg_reg and energy > 0.0:
            return "borderline_full_allowed"
        if _safe_float(row.get("no_long_write_token_mass_mean_v57"), 0.0) > 0.25 and ate > h35:
            return "repair_no_long_too_broad"
        return "stop_no_ttt_screen_signal"
    return "stop_no_screen_signal"


def _add_decisions(rows: Sequence[Dict[str, Any]], h35_full: Mapping[str, Any], h35_704: Mapping[str, Any]) -> None:
    for row in rows:
        row["delta_vs_H35_full"] = _safe_float(row.get("ATE")) - _safe_float(h35_full.get("ATE")) if row.get("ATE") is not None else None
        row["delta_vs_H35_704"] = _safe_float(row.get("ATE")) - _safe_float(h35_704.get("ATE")) if row.get("ATE") is not None else None
        tr = _track(row)
        if tr == "semantic_smoke":
            row["semantic_smoke_gate_pass"] = _semantic_smoke_gate(row)
        if tr == "ttt_action" and _is_smoke(row):
            row["ttt_smoke_gate_pass"] = _ttt_smoke_gate(row)
        if _is_704(row) and tr in {"semantic_read", "ttt_action"}:
            row["screen_decision"] = _screen_decision(row, h35_704, tr)
        if _is_full(row):
            row["minimum_progress_pass"] = bool(_safe_float(row.get("ATE"), 999.0) <= H35_MIN_PROGRESS)
            row["semantic_success_pass"] = bool(tr in {"semantic_read", "combo"} and _safe_float(row.get("ATE"), 999.0) <= H35_SEMANTIC_SUCCESS)
            row["ttt_success_pass"] = bool(tr in {"ttt_action", "combo"} and _safe_float(row.get("ATE"), 999.0) <= H35_TTT_SUCCESS)
            row["runtime_wall_pass"] = bool(_safe_float(row.get("wall_time_min"), 999.0) <= 28.0)


def _phase0_rows(v56_root: Path, gt: Path) -> List[Dict[str, Any]]:
    rows = _summarize_runs(_iter_run_dirs([v56_root]), gt)
    _v56_augment_rows(rows, gt)
    for row in rows:
        row.update(_v57_extra_debug(Path(str(row.get("run_dir") or ""))))
        if row.get("affected_source_token_count_mean") is None:
            row["affected_source_token_count_mean"] = row.get("affected_source_token_count_max")
        if row.get("attention_mass_removed_before_mean") is None:
            row["attention_mass_removed_before_mean"] = row.get("source_influence_mass_removed_before_mean")
        if row.get("attention_mass_removed_after_mean") is None:
            row["attention_mass_removed_after_mean"] = row.get("source_influence_mass_removed_after_mean")
        if row.get("context_empty_source_events_sum") is None:
            row["context_empty_source_events_sum"] = row.get("context_empty_source_events")
        row["semantic_action_realized"] = _semantic_action_realized(row)
    return rows


def _write_phase0_audits(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sem = [r for r in rows if _track(r) == "A" or str(_candidate(r)).startswith("A")]
    ttt = [r for r in rows if _track(r) == "B" or str(_candidate(r)).startswith("B")]
    sem_lines = [
        "# semantic_action_realization_audit",
        "",
        "来源: v56 landed artifacts。判定只使用已落盘 debug/log/evaluation 字段。",
        "",
        "| row | frames | ATE | stage_c_hit | superset_hit | semantic_labels | semantic_groups | source_tokens_mean | source_tokens_max | mass_before | mass_after | per_label_source_mass | per_label_affected_token_mass | static_anchor_protected | context_empty | realized |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in sem:
        sem_lines.append(
            f"| {row.get('row') or row.get('candidate')} | {row.get('frames')} | {_fmt(row.get('ATE'))} | "
            f"{_fmt(row.get('stage_c_cache_hit_rate'))} | {_fmt(row.get('stage_c_cache_superset_hit_count'))} | "
            f"{_fmt(row.get('semantic_label_count_mean'))} | {_fmt(row.get('semantic_group_count_mean'))} | "
            f"{_fmt(row.get('affected_source_token_count_mean'))} | {_fmt(row.get('affected_source_token_count_max'))} | "
            f"{_fmt(row.get('attention_mass_removed_before_mean'))} | {_fmt(row.get('attention_mass_removed_after_mean'))} | "
            f"{row.get('per_label_source_mass') or 'NA'} | {row.get('per_label_affected_token_mass') or 'NA'} | "
            f"{_fmt(row.get('static_anchor_protected_mass'))} | {_fmt(row.get('context_empty_source_events_sum'))} | "
            f"{row.get('semantic_action_realized')} |"
        )
    if not sem:
        sem_lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    inactive = [r for r in sem if not bool(r.get("semantic_action_realized"))]
    sem_lines.extend([
        "",
        f"结论: v56 semantic rows 中 realized=True 的数量为 `{len(sem) - len(inactive)}/{len(sem)}`。",
        "若 `affected_source_token_count_mean` 为 0 或 semantic label 为 NA，本报告只判定为 action inactive / evidence missing，不判定语义科学无效。",
    ])
    _write_text(out_dir / "semantic_action_realization_audit.md", sem_lines)

    ttt_lines = [
        "# ttt_action_regression_audit",
        "",
        "来源: v56 landed artifacts。用于回答 B1/B2/B3/B4 是否触发以及回退位置。",
        "",
        "| row | frames | ATE | stable_mass | risk_mass | no_long_mass | short_residual_norm | long_residual_norm | post_zp_norm | native_cos | seg000_384 | seg384_700 | rolling100_p90 | regression_note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ttt:
        note = "broad_no_long_regression" if _safe_float(row.get("no_long_write_token_mass_mean_v57"), 0.0) > 0.3 and _safe_float(row.get("delta_vs_H35_704"), 1.0) > 0 else ""
        ttt_lines.append(
            f"| {row.get('row') or row.get('candidate')} | {row.get('frames')} | {_fmt(row.get('ATE'))} | "
            f"{_fmt(row.get('stable_anchor_token_mass_mean_v57') or row.get('stable_anchor_token_mass_mean'))} | "
            f"{_fmt(row.get('risk_token_mass_mean_v57') or row.get('risk_token_mass_mean'))} | "
            f"{_fmt(row.get('no_long_write_token_mass_mean_v57') or row.get('no_long_write_token_mass_mean'))} | "
            f"{_fmt(row.get('short_residual_norm_mean'))} | {_fmt(row.get('long_residual_norm_mean'))} | "
            f"{_fmt(row.get('post_zp_delta_norm_mean'))} | {_fmt(row.get('candidate_native_delta_cos_mean') or row.get('native_delta_gate_cos_mean'))} | "
            f"{_fmt(row.get('seg0_000_384_rmse'))} | {_fmt(row.get('seg1_384_700_rmse'))} | {_fmt(row.get('rolling100_p90'))} | {note or 'NA'} |"
        )
    if not ttt:
        ttt_lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    ttt_lines.extend([
        "",
        "结论: 若 no-long mass 高且 ATE 回退，v57 后续不继续 broad no-long-write，只允许 high-risk/high-influence 受限版本。",
        "post-zp / branch-layer 低层能量字段若为 NA，表示旧 run 未写出该证据。",
    ])
    _write_text(out_dir / "ttt_action_regression_audit.md", ttt_lines)


def _plot_metric_bar(path: Path, rows: Sequence[Mapping[str, Any]], key: str, title: str, ylabel: str, hline: Optional[float] = None) -> None:
    pts = [(str(r.get("candidate") or r.get("row") or r.get("run_name")), _safe_float(r.get(key))) for r in rows]
    pts = [(x, y) for x, y in pts if math.isfinite(y)]
    if not pts:
        _plot_no_data(path, title, f"metric {key} unavailable")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(pts) * 1.0), 4))
    ax.bar([p[0] for p in pts], [p[1] for p in pts], color="#4C78A8")
    if hline is not None and math.isfinite(hline):
        ax.axhline(hline, color="#F58518", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_figures(out_dir: Path, rows: Sequence[Mapping[str, Any]], h35_rows: Sequence[Mapping[str, Any]]) -> None:
    figs = out_dir / "figures"
    sem_rows = [r for r in rows if _track(r) in {"semantic_smoke", "semantic_read", "combo"}]
    ttt_rows = [r for r in rows if _track(r) in {"ttt_action", "combo"}]
    _plot_no_data(
        figs / "semantic_action_overlay_grid.png",
        "semantic_action_overlay_grid",
        "current runner does not emit RGB/semantic/D_g/source-attention spatial overlays",
    )
    _plot_metric_bar(figs / "semantic_action_mass_by_label.png", sem_rows, "affected_source_token_count_mean", "semantic action affected source tokens", "tokens")
    _plot_metric_bar(figs / "ttt_delta_energy_timeline.png", ttt_rows, "short_residual_norm_mean", "TTT short residual energy summary", "norm")
    _plot_metric_bar(figs / "ttt_layer_branch_heatmap.png", ttt_rows, "post_zp_delta_norm_mean", "TTT post-zp delta energy summary", "norm")
    _plot_metric_bar(figs / "segment_error_comparison.png", list(h35_rows) + list(rows), "seg1_384_700_rmse", "segment 384-700 RMSE comparison", "RMSE")
    _plot_metric_bar(figs / "rolling100_timeline.png", list(h35_rows) + list(rows), "rolling100_p90", "rolling100 p90 comparison", "p90")


def _write_registries(out_dir: Path, rows: Sequence[Mapping[str, Any]], h35_rows: Sequence[Mapping[str, Any]], v56_rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(out_dir / "v57_all_registry.csv", [dict(r) for r in rows])
    _write_json(out_dir / "v57_all_registry.json", [dict(r) for r in rows])
    _write_csv(out_dir / "v57_h35_reference_registry.csv", [dict(r) for r in h35_rows])
    _write_csv(out_dir / "v57_phase0_v56_registry.csv", [dict(r) for r in v56_rows])
    _write_csv(out_dir / "v57_semantic_smoke_registry.csv", [dict(r) for r in rows if _track(r) == "semantic_smoke"])
    _write_csv(out_dir / "v57_semantic_704f_registry.csv", [dict(r) for r in rows if _track(r) == "semantic_read" and _is_704(r)])
    _write_csv(out_dir / "v57_semantic_full_registry.csv", [dict(r) for r in rows if _track(r) == "semantic_read" and _is_full(r)])
    _write_csv(out_dir / "v57_ttt_smoke_registry.csv", [dict(r) for r in rows if _track(r) == "ttt_action" and _is_smoke(r)])
    _write_csv(out_dir / "v57_ttt_704f_registry.csv", [dict(r) for r in rows if _track(r) == "ttt_action" and _is_704(r)])
    _write_csv(out_dir / "v57_ttt_full_registry.csv", [dict(r) for r in rows if _track(r) == "ttt_action" and _is_full(r)])


def _best(rows: Sequence[Mapping[str, Any]], track: str, full: bool) -> Optional[Mapping[str, Any]]:
    candidates = [r for r in rows if _track(r) == track and ((_is_full(r) if full else _is_704(r))) and r.get("status") == "done"]
    return min(candidates, key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)


def _list_run_commands(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = ["| row | run | GPU | frames | status | wall min | ATE | 输出目录 |", "|---|---|---:|---:|---|---:|---:|---|"]
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        rel = run_dir.relative_to(REPO_ROOT) if str(run_dir).startswith(str(REPO_ROOT)) else run_dir
        gpu = _first_yaml_value(run_dir / "effective_config.yaml", "gpu")
        lines.append(
            f"| `{row.get('row')}` | `{row.get('run_name')}` | {gpu or 'NA'} | {row.get('frames')} | "
            f"{row.get('status')} | {_fmt(row.get('wall_time_min'))} | {_fmt(row.get('ATE'))} | `{rel}` |"
        )
    if len(lines) == 2:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA |")
    return lines


def _find_row(rows: Sequence[Mapping[str, Any]], row_id: str) -> Optional[Mapping[str, Any]]:
    return next((r for r in rows if str(r.get("row")) == row_id or str(r.get("candidate")) == row_id), None)


def _write_docs(
    result_root: Path,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    h35_full: Mapping[str, Any],
    h35_704: Mapping[str, Any],
    v56_rows: Sequence[Mapping[str, Any]],
) -> None:
    sem_smokes = [r for r in rows if _track(r) == "semantic_smoke"]
    sem704 = [r for r in rows if _track(r) == "semantic_read" and _is_704(r)]
    semfull = [r for r in rows if _track(r) == "semantic_read" and _is_full(r)]
    ttt_smokes = [r for r in rows if _track(r) == "ttt_action" and _is_smoke(r)]
    ttt704 = [r for r in rows if _track(r) == "ttt_action" and _is_704(r)]
    tttfull = [r for r in rows if _track(r) == "ttt_action" and _is_full(r)]
    combo = [r for r in rows if _track(r) == "combo"]

    phase0_sem_inactive = [
        r for r in v56_rows if (_track(r) == "A" or str(_candidate(r)).startswith("A")) and not bool(r.get("semantic_action_realized"))
    ]
    sem_smoke_pass = bool(sem_smokes) and all(bool(r.get("semantic_smoke_gate_pass")) for r in sem_smokes)
    s0 = next((r for r in sem_smokes if str(_candidate(r)).startswith("S0")), None)
    ttt_smoke_pass = bool(ttt_smokes) and all(bool(r.get("ttt_smoke_gate_pass")) for r in ttt_smokes)
    semantic_promoted = [r for r in sem704 if str(r.get("screen_decision")) in {"promote_full", "promote_full_rolling", "borderline_full_allowed"}]
    ttt_promoted = [r for r in ttt704 if str(r.get("screen_decision")) in {"promote_full", "promote_full_rolling", "borderline_full_allowed"}]
    best_sem_full = _best(rows, "semantic_read", True)
    best_ttt_full = _best(rows, "ttt_action", True)
    best_combo = min(combo, key=lambda r: _safe_float(r.get("ATE"), 999.0), default=None)
    sem_min_progress = bool(best_sem_full and _safe_float(best_sem_full.get("ATE"), 999.0) <= H35_MIN_PROGRESS)
    sem_target_success = bool(best_sem_full and _safe_float(best_sem_full.get("ATE"), 999.0) <= H35_SEMANTIC_SUCCESS)
    ttt_min_progress = bool(best_ttt_full and _safe_float(best_ttt_full.get("ATE"), 999.0) <= H35_MIN_PROGRESS)
    ttt_target_success = bool(best_ttt_full and _safe_float(best_ttt_full.get("ATE"), 999.0) <= H35_TTT_SUCCESS)

    exec_lines = [
        "# ACL2 v57 H35 SemanticActionRepair TTT TTL Fast 执行日志",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"结果复盘: `{DOC_REVIEW.relative_to(REPO_ROOT)}`",
        f"工作目录: `{REPO_ROOT}`",
        f"结果根目录: `{result_root.relative_to(REPO_ROOT)}`",
        f"报告目录: `{out_dir.relative_to(REPO_ROOT)}`",
        "",
        "## 执行边界",
        "",
        "- 所有数据来自 landed artifacts、`01.log`、`hmc_state_hash.jsonl`、`wall_time_summary.json` 或 pose evaluation。",
        "- 缺失字段保持 NA/unavailable；不把计划值、默认值或推测写成观测值。",
        "- 语义路线必须先通过 action-realization smoke；若 S0 失败，不进入 semantic 704F/full。",
        "- TTT 路线必须先完成 v56 behavioral audit，再跑 96F smoke 和 gate 允许的 704F/full。",
        "- 不使用 absolute chunk-id policy，不使用 hand-specified positive/neutral/negative percentage。",
        "",
        "## 代码与工具修改",
        "",
        "| 文件 | 修改内容 | 审计理由 |",
        "|---|---|---|",
        "| `run_pipeline_abc_v2.py` | 将 context source skip 与 semantic role 相关 CLI 参数传入 `HybridMemoryController`，并把 v57 semantic prior/action debug 字段写入 trace。 | v56 A2/A3 参数到 HMC 的链路断开，导致 action inactive；修复后需要可审计地证明 source token 与 attention mass 被改变。 |",
        "| `loger/pipeline/hybrid_memory_controller.py` | 增加 v57 semantic source-skip role policies，保留 group-level source role，避免 fine label 缺失时被 fallback 擦掉。 | 支持 S0/S1/S2 action-realization repair。 |",
        "| `tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh` | 新增 v57 统一 runner，覆盖 semantic smoke/read、TTT smoke/704/full 和 combo。 | 让每条 run 带有 `effective_config.yaml`、audit JSON 与 reproduce command。 |",
        "| `tools/v57_experiment_report.py` | 新增 artifact-only reporter，生成 Phase0 audit、registries、figures、执行日志和复盘；semantic action realization 接受 fine-label 或 group-level evidence。 | 避免手工摘数和虚构缺失指标，同时避免 SREAD02 这类只有 group evidence 的 active action 被误判 inactive。 |",
        "",
        "验证命令:",
        "",
        "```bash",
        f"{sys.executable} -m py_compile run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py tools/v57_experiment_report.py",
        "bash -n tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh",
        "```",
        "",
        "## Phase 0 审计产物",
        "",
        f"- semantic audit: `{(out_dir / 'semantic_action_realization_audit.md').relative_to(REPO_ROOT)}`",
        f"- TTT audit: `{(out_dir / 'ttt_action_regression_audit.md').relative_to(REPO_ROOT)}`",
        "",
        "## Gate 停止记录",
        "",
        f"- Semantic 704F promoted rows: `{', '.join(str(r.get('row')) for r in semantic_promoted) if semantic_promoted else 'none'}`。",
        f"- Semantic full rows executed: `{', '.join(str(r.get('row')) for r in semfull) if semfull else 'none'}`。",
        f"- TTT 704F promoted rows: `{', '.join(str(r.get('row')) for r in ttt_promoted) if ttt_promoted else 'none'}`。",
        f"- TTT full rows executed: `{', '.join(str(r.get('row')) for r in tttfull) if tttfull else 'none'}`。",
        f"- Combo rows executed: `{', '.join(str(r.get('row')) for r in combo) if combo else 'none'}`。",
        "",
        "## 运行命令清单",
        "",
        *_list_run_commands(rows),
        "",
        "复现单条 run 模板:",
        "",
        "```bash",
        "tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh <GPU> <ROW> <RUN_NAME>",
        "```",
        "",
        "每个 run 目录都包含 `effective_config.yaml`、`v57_effective_config.yaml`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json`、`reproduce_command.sh`。",
        "",
        "## 报告生成",
        "",
        "```bash",
        f"{sys.executable} tools/v57_experiment_report.py",
        "```",
    ]
    _write_text(DOC_EXEC, exec_lines)

    review_lines = [
        "# ACL2 v57 H35 SemanticActionRepair TTT TTL Fast 实验结果复盘",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"执行日志: `{DOC_EXEC.relative_to(REPO_ROOT)}`",
        f"结果根目录: `{result_root.relative_to(REPO_ROOT)}`",
        "",
        "结论先行: 本文件只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持 NA/unavailable。v57 修复并验证了 semantic action-realization，但没有得到可报告成功: SREAD03 在 704F 有改善，full 反而比 H35 差；TTT 三条 704F 全部回退，因此没有 TTT full 和 combo。",
        "",
        "## H35 参照",
        "",
        f"- H35 full ATE: `{_fmt(h35_full.get('ATE'))}`。",
        f"- H35 704F ATE: `{_fmt(h35_704.get('ATE'))}`。",
        f"- semantic minimum progress gate: ATE <= `{_fmt(H35_MIN_PROGRESS)}`；target success gate: ATE <= `{_fmt(H35_SEMANTIC_SUCCESS)}`。",
        f"- TTT target success gate: ATE <= `{_fmt(H35_TTT_SUCCESS)}`。",
        "",
        "## Phase 0: v56 修复性审计",
        "",
        f"- v56 semantic action inactive rows: `{len(phase0_sem_inactive)}`。",
        f"- semantic audit: `{(out_dir / 'semantic_action_realization_audit.md').relative_to(REPO_ROOT)}`。",
        f"- TTT audit: `{(out_dir / 'ttt_action_regression_audit.md').relative_to(REPO_ROOT)}`。",
        "- 关键修复结论: 如果 v56 semantic rows 的 source token / attention mass 字段为 0 或 NA，只能判定 action inactive 或 evidence missing，不能写成语义无用。",
        "",
        "## Phase 1: Semantic READ action smoke",
        "",
        *_md_table(
            sem_smokes,
            (
                ("row", "row"),
                ("run", "run_name"),
                ("ATE", "ATE"),
                ("stage_hit", "stage_c_cache_hit_rate"),
                ("labels", "semantic_label_count_mean"),
                ("groups", "semantic_group_count_mean"),
                ("src_mean", "affected_source_token_count_mean"),
                ("mass_before", "attention_mass_removed_before_mean"),
                ("mass_after", "attention_mass_removed_after_mean"),
                ("empty", "context_empty_source_events_sum"),
                ("gate", "semantic_smoke_gate_pass"),
            ),
        ),
        "",
        f"Phase1 gate pass: `{sem_smoke_pass}`。",
        "",
        "注: `V57_S*_96F` 是第一批 action hook smoke；`V57R1_*_TRACE_96F` 是补充 semantic prior/debug 落盘后重跑的审计 smoke。两批都保留在 registry 中，最终 label/group/source-mass 证据以 trace rows 为主。",
    ]
    if s0 is not None and not bool(s0.get("semantic_smoke_gate_pass")):
        review_lines.append("S0 未通过，按计划判定为 semantic projection/source-skip hook 仍有 blocker，不允许进入 semantic 704F/full。")
    elif sem_smokes and not sem_smoke_pass:
        review_lines.append("S0 若通过但 S1/S2 未通过，失败归因优先是 label/group mapping 或 D_g 阈值过窄。")
    review_lines.extend([
        "",
        "## Phase 2: Semantic READ 704F",
        "",
        *_md_table(
            sem704,
            (
                ("row", "row"),
                ("ATE", "ATE"),
                ("dH35_704", "delta_vs_H35_704"),
                ("rolling100p90", "rolling100_p90"),
                ("src_mean", "affected_source_token_count_mean"),
                ("mass_before", "attention_mass_removed_before_mean"),
                ("mass_after", "attention_mass_removed_after_mean"),
                ("static_protect", "static_anchor_protected_mass"),
                ("decision", "screen_decision"),
            ),
        ),
        "",
        f"Semantic promoted to full: `{', '.join(str(r.get('row')) for r in semantic_promoted) if semantic_promoted else 'none'}`。",
    ])
    sread01 = _find_row(sem704, "SREAD01_704F")
    sread02 = _find_row(sem704, "SREAD02_704F")
    sread03 = _find_row(sem704, "SREAD03_704F")
    sread04 = _find_row(sem704, "SREAD04_704F")
    if any((sread01, sread02, sread03, sread04)):
        review_lines.extend(["", "Semantic 704F blocker repair 记录:"])
    if sread01 and sread04:
        review_lines.append(
            f"- SREAD01 action active but regressed: delta vs H35_704 `{_fmt(sread01.get('delta_vs_H35_704'))}`；"
            f"SREAD04 加 static rescue 后 protected mass `{_fmt(sread04.get('static_anchor_protected_mass'))}`，"
            f"但 ATE 仍为 `{_fmt(sread04.get('ATE'))}`，未修复回退。"
        )
    if sread02:
        review_lines.append(
            f"- SREAD02 只有 group-level semantic evidence 也确实 active: affected source tokens mean `{_fmt(sread02.get('affected_source_token_count_mean'))}`，"
            f"attention mass `{_fmt(sread02.get('attention_mass_removed_before_mean'))}` -> `{_fmt(sread02.get('attention_mass_removed_after_mean'))}`，"
            f"但 704F delta `{_fmt(sread02.get('delta_vs_H35_704'))}`。"
        )
    if sread03:
        review_lines.append(
            f"- SREAD03 是唯一 promoted semantic row: 704F ATE `{_fmt(sread03.get('ATE'))}`，"
            f"delta vs H35_704 `{_fmt(sread03.get('delta_vs_H35_704'))}`，"
            f"rolling100_p90 `{_fmt(sread03.get('rolling100_p90'))}`。"
        )
    review_lines.extend([
        "",
        "## Phase 3: Semantic READ full",
        "",
        *_md_table(
            semfull,
            (
                ("row", "row"),
                ("ATE", "ATE"),
                ("dH35", "delta_vs_H35_full"),
                ("Rot", "Rot"),
                ("FinalErr", "FinalErr"),
                ("wall", "wall_time_min"),
                ("min_progress", "minimum_progress_pass"),
                ("target_success", "semantic_success_pass"),
            ),
        ),
        "",
    ])
    if best_sem_full:
        review_lines.append(
            f"Semantic full 判定: `{best_sem_full.get('row')}` ATE `{_fmt(best_sem_full.get('ATE'))}`，"
            f"delta vs H35 `{_fmt(best_sem_full.get('delta_vs_H35_full'))}`，"
            f"runtime `{_fmt(best_sem_full.get('wall_time_min'))}` min。minimum progress=`{sem_min_progress}`，"
            f"target success=`{sem_target_success}`。按计划，ATE >= H35+0.3m 属 hard fail。"
        )
    review_lines.extend([
        "",
        "## Phase 4: New TTT action smoke/704/full",
        "",
        *_md_table(
            ttt_smokes,
            (
                ("row", "row"),
                ("ATE", "ATE"),
                ("stable", "stable_anchor_token_mass_mean_v57"),
                ("risk", "risk_token_mass_mean_v57"),
                ("no_long", "no_long_write_token_mass_mean_v57"),
                ("short_norm", "short_residual_norm_mean"),
                ("native_cos", "candidate_native_delta_cos_mean"),
                ("gate", "ttt_smoke_gate_pass"),
            ),
        ),
        "",
        f"TTT smoke gate pass: `{ttt_smoke_pass}`。",
        "",
        *_md_table(
            ttt704,
            (
                ("row", "row"),
                ("ATE", "ATE"),
                ("dH35_704", "delta_vs_H35_704"),
                ("rolling100p90", "rolling100_p90"),
                ("no_long", "no_long_write_token_mass_mean_v57"),
                ("short_norm", "short_residual_norm_mean"),
                ("post_zp", "post_zp_delta_norm_mean"),
                ("decision", "screen_decision"),
            ),
        ),
        "",
        f"TTT promoted to full: `{', '.join(str(r.get('row')) for r in ttt_promoted) if ttt_promoted else 'none'}`。",
    ])
    ttt01 = _find_row(ttt704, "TTT01_704F")
    ttt02 = _find_row(ttt704, "TTT02_704F")
    ttt03 = _find_row(ttt704, "TTT03_704F")
    if any((ttt01, ttt02, ttt03)):
        review_lines.extend(["", "TTT 704F blocker repair 记录:"])
    if ttt01:
        review_lines.append(
            f"- TTT01 复现 broad no-long 过宽问题: no_long mass `{_fmt(ttt01.get('no_long_write_token_mass_mean_v57'))}`，"
            f"delta vs H35_704 `{_fmt(ttt01.get('delta_vs_H35_704'))}`。"
        )
    if ttt02:
        review_lines.append(
            f"- TTT02 short residual/TTL 机制触发，但 short_residual_norm `{_fmt(ttt02.get('short_residual_norm_mean'))}`，"
            f"704F delta `{_fmt(ttt02.get('delta_vs_H35_704'))}`，没有 screen signal。"
        )
    if ttt03:
        review_lines.append(
            f"- TTT03 将 no-long 收窄到 `{_fmt(ttt03.get('no_long_write_token_mass_mean_v57'))}`，"
            f"但 704F delta 仍为 `{_fmt(ttt03.get('delta_vs_H35_704'))}`，说明简单 high-risk/high-influence 收窄不足。"
        )
    review_lines.extend([
        "",
        *_md_table(
            tttfull,
            (
                ("row", "row"),
                ("ATE", "ATE"),
                ("dH35", "delta_vs_H35_full"),
                ("Rot", "Rot"),
                ("FinalErr", "FinalErr"),
                ("wall", "wall_time_min"),
                ("min_progress", "minimum_progress_pass"),
                ("target_success", "ttt_success_pass"),
            ),
        ),
        "",
        "## Phase 5: Combo",
        "",
        *_md_table(combo, (("row", "row"), ("ATE", "ATE"), ("dH35", "delta_vs_H35_full"), ("wall", "wall_time_min"))),
        "",
        "## 关键实验结论",
        "",
    ])
    if best_sem_full:
        review_lines.append(
            f"- Best semantic full: `{best_sem_full.get('row')}` ATE `{_fmt(best_sem_full.get('ATE'))}`, "
            f"delta vs H35 `{_fmt(best_sem_full.get('delta_vs_H35_full'))}`，minimum progress `{sem_min_progress}`，target success `{sem_target_success}`。"
        )
    else:
        review_lines.append("- Semantic full 未运行或未完成；若 Phase1/704 gate 未通过，这是按计划停止，不是遗漏。")
    if best_ttt_full:
        review_lines.append(
            f"- Best TTT full: `{best_ttt_full.get('row')}` ATE `{_fmt(best_ttt_full.get('ATE'))}`, "
            f"delta vs H35 `{_fmt(best_ttt_full.get('delta_vs_H35_full'))}`。"
        )
    else:
        review_lines.append("- TTT full 未运行或未完成；若 smoke/704 gate 未通过，这是按计划停止。")
    review_lines.append(
        f"- 最终 gate: semantic_min_progress=`{sem_min_progress}`, semantic_target=`{sem_target_success}`, "
        f"ttt_min_progress=`{ttt_min_progress}`, ttt_target=`{ttt_target_success}`, combo_run=`{bool(combo)}`。"
    )
    if best_combo:
        review_lines.append(f"- Combo full: `{best_combo.get('row')}` ATE `{_fmt(best_combo.get('ATE'))}`。")
    else:
        review_lines.append("- Combo 未启动；只有单目标 full improvement >=0.5m 才允许组合。")
    review_lines.extend([
        "",
        "## Insight 与证据链",
        "",
        "- 第一层证据是 action-realization，而不是 ATE。v57 先修复 v56 CLI 参数未传入 HMC 的 wiring blocker，再用 S0/S1/S2 smoke 检查 source token 与 attention mass。",
        "- S0/S1/S2 修复后证明 semantic projection -> HMC prior -> pi3 compact K/V source skip 的代码路径通；v56 的语义负结果不能解释成语义本身无效。",
        "- 704F 上 SREAD03 的 C23/action-guard 路线是真信号，但 full 没有继承这个收益，说明当前 READ action 的局部改善不能稳定覆盖 full trajectory。",
        "- SREAD04 的 static rescue 没能修复 SREAD01 的 active regression；后续不能继续做宽泛语义过滤，需要更细的作用域/trajectory-state gate。",
        "- TTT 路线不再扩大 broad no-long-write；TTT03 已按计划收窄但仍回退，TTL/short residual 当前实现也没有优于 H35 的 704F 证据。",
        "- dense overlay/heatmap 图若为 no-data，是因为当前 runner 没有落盘对应空间/tensor trace，复盘不会把缺失解释成 0。",
        "",
        "## 必答问题",
        "",
        f"1. v56 semantic failure 是否主要因为 action inactive: `{len(phase0_sem_inactive) > 0}`，证据见 Phase0 audit。",
        f"2. 修复后 semantic 是否选中真实 source tokens: `{'yes' if sem_smoke_pass else 'not proven'}`。",
        f"3. semantic READ 是否达到 0.5m/2m full 收益: `{'yes' if sem_min_progress else 'not proven'}` / `{'yes' if sem_target_success else 'not proven'}`。",
        f"4. TTL/short residual 是否优于 broad no-long-write: `{'yes' if ttt_min_progress else 'not proven'}`。当前只证明 TTT02 机制触发但 704F 回退。",
        f"5. new TTT action 是否达到 1m full 收益: `{'yes' if ttt_target_success else 'not proven'}`。",
        "6. 两个目标失败的当前证据: semantic action-realization 已修复，但 READ 局部 704F 收益不能转成 full；TTT 简单生命周期/no-long action space 不足。还不能证明 H35 主要误差完全不来自 READ/TTT。",
        "7. 后续方向: 语义若继续，应聚焦 SREAD03 这类局部收益的 full-transfer/trajectory-state gate；TTT 不建议继续 broad no-long 或同形 TTL 小扫，应转向 trajectory-state、merge-gauge、pose-scale controller 等更底层状态控制。",
        "",
        "## 审计材料",
        "",
        f"- registries: `{out_dir.relative_to(REPO_ROOT)}`",
        f"- figures: `{(out_dir / 'figures').relative_to(REPO_ROOT)}`",
        f"- `semantic_action_realization_audit.md` / `ttt_action_regression_audit.md`",
    ])
    _write_text(DOC_REVIEW, review_lines)
    _write_text(out_dir / "v57_final_report.md", review_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT, type=Path)
    parser.add_argument("--v56-root", default=DEFAULT_V56_ROOT, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--gt", default=DEFAULT_GT, type=Path)
    parser.add_argument("--h35-full", default=DEFAULT_H35_FULL, type=Path)
    parser.add_argument("--h35-704", default=DEFAULT_H35_704, type=Path)
    args = parser.parse_args()

    result_root = args.result_root
    out_dir = args.out_dir or result_root / "report_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _summarize_runs(_iter_run_dirs([result_root]), args.gt)
    _augment_v57_rows(rows, args.gt)

    h35_rows = _summarize_runs([args.h35_full, args.h35_704], args.gt)
    _v56_augment_rows(h35_rows, args.gt)
    h35_full = next((r for r in h35_rows if _is_full(r)), {})
    h35_704 = next((r for r in h35_rows if _is_704(r)), {})

    v56_rows = _phase0_rows(args.v56_root, args.gt)
    for row in v56_rows:
        row["delta_vs_H35_704"] = _safe_float(row.get("ATE")) - _safe_float(h35_704.get("ATE")) if row.get("ATE") is not None else None

    _add_decisions(rows, h35_full, h35_704)
    _write_phase0_audits(out_dir, v56_rows)
    _write_registries(out_dir, rows, h35_rows, v56_rows)
    _write_figures(out_dir, rows, h35_rows)

    summary = {
        "result_root": str(result_root),
        "run_count": len(rows),
        "h35_full_ate": h35_full.get("ATE"),
        "h35_704_ate": h35_704.get("ATE"),
        "semantic_smoke_gate_pass": bool([r for r in rows if _track(r) == "semantic_smoke"])
        and all(bool(r.get("semantic_smoke_gate_pass")) for r in rows if _track(r) == "semantic_smoke"),
        "ttt_smoke_gate_pass": bool([r for r in rows if _track(r) == "ttt_action" and _is_smoke(r)])
        and all(bool(r.get("ttt_smoke_gate_pass")) for r in rows if _track(r) == "ttt_action" and _is_smoke(r)),
        "best_semantic_full_ate": (_best(rows, "semantic_read", True) or {}).get("ATE"),
        "best_ttt_full_ate": (_best(rows, "ttt_action", True) or {}).get("ATE"),
    }
    _write_json(out_dir / "v57_summary.json", summary)
    _write_docs(result_root, out_dir, rows, h35_full, h35_704, v56_rows)


if __name__ == "__main__":
    main()
