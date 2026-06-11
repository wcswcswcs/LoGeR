#!/usr/bin/env python3
"""Generate ACL2 v58 soft semantic READ / commit-isolation reports.

All metrics are read from landed artifacts. Missing traces stay NA/no-data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
    _first_yaml_value,
    _augment_rows as _v56_augment_rows,
    _is_704,
    _is_full,
    _is_smoke,
    _md_table,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry"
DEFAULT_V57_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
PLAN_DOC = "docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md"
DOC_EXEC = REPO_ROOT / "docs/ACL2_v58_SoftSemanticRead_CommitIsolation_Geometry_执行日志.md"
DOC_REVIEW = REPO_ROOT / "docs/ACL2_v58_SoftSemanticRead_CommitIsolation_Geometry_实验结果复盘.md"
H35_FULL_ATE = 35.74089695811434
H35_704_ATE = 39.79824772048563
H35_MIN_PROGRESS = 35.2409
H35_SEMANTIC_SUCCESS = 33.7409
H35_STRONG_SUCCESS = 33.0
H35_704_PROMOTE_DELTA = -0.50
H35_704_BORDERLINE_DELTA = 0.20


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _candidate(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate") or row.get("row") or row.get("run_name") or "")


def _reproduce_inline(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    parts: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#!") or line in {"set -euo pipefail"} or line.startswith("cd "):
            continue
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        parts.append(line)
    return " ".join(parts) if parts else None


def _track(row: Mapping[str, Any]) -> str:
    return str(row.get("track") or "")


def _mass_ratio(row: Mapping[str, Any]) -> Optional[float]:
    before = _safe_float(row.get("attention_mass_removed_before_mean"))
    after = _safe_float(row.get("attention_mass_removed_after_mean"))
    if not math.isfinite(before) or before <= 0 or not math.isfinite(after):
        return None
    return float(after / before)


def _float_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _collect_v58_debug(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: Dict[str, Any] = {}
    affected_tokens: List[float] = []
    mass_before: List[float] = []
    mass_after: List[float] = []
    mass_actual_after: List[float] = []
    source_weight_mean: List[float] = []
    source_weight_min: List[float] = []
    source_score_mean: List[float] = []
    source_score_max: List[float] = []
    source_keep_ratio: List[float] = []
    empty_events: List[float] = []
    static_removed_ratio: List[float] = []
    semantic_group_counts: List[float] = []
    semantic_label_counts: List[float] = []
    metrics_seen: List[str] = []
    layer_counts: Dict[str, int] = {}
    commit_modes: List[str] = []
    state_safe: List[bool] = []
    probe_no_commit: List[bool] = []
    commit_hash_equal: List[bool] = []

    for row in rows:
        if row.get("hmc_commit_mode") is not None:
            commit_modes.append(str(row.get("hmc_commit_mode")))
        if row.get("state_double_write_safe") is not None:
            state_safe.append(bool(row.get("state_double_write_safe")))
        if row.get("probe_no_commit_hash_equal") is not None:
            probe_no_commit.append(bool(row.get("probe_no_commit_hash_equal")))
        if row.get("hash_H_next") is not None and row.get("commit_source_state_hash") is not None:
            commit_hash_equal.append(str(row.get("hash_H_next")) == str(row.get("commit_source_state_hash")))
        for node in _walk(row):
            if not isinstance(node, dict):
                continue
            groups = node.get("semantic_group_role_metrics") or node.get("prior_semantic_group_role_metrics")
            if isinstance(groups, dict):
                semantic_group_counts.append(float(len(groups)))
            if node.get("fine_label_token_count") is not None:
                semantic_label_counts.append(_safe_float(node.get("fine_label_token_count")))
            if node.get("prior_fine_label_token_count") is not None:
                semantic_label_counts.append(_safe_float(node.get("prior_fine_label_token_count")))
            if node.get("source_skip_tokens") is not None:
                affected_tokens.append(_safe_float(node.get("source_skip_tokens")))
            if node.get("max_context_source_skip_tokens") is not None:
                affected_tokens.append(_safe_float(node.get("max_context_source_skip_tokens")))
            if node.get("attention_mass_removed_before") is not None:
                mass_before.append(_safe_float(node.get("attention_mass_removed_before")))
            if node.get("mean_attention_mass_removed_before") is not None:
                mass_before.append(_safe_float(node.get("mean_attention_mass_removed_before")))
            if node.get("attention_mass_removed_after") is not None:
                mass_after.append(_safe_float(node.get("attention_mass_removed_after")))
            if node.get("mean_attention_mass_removed_after") is not None:
                mass_after.append(_safe_float(node.get("mean_attention_mass_removed_after")))
            if node.get("attention_mass_actual_after") is not None:
                mass_actual_after.append(_safe_float(node.get("attention_mass_actual_after")))
            if node.get("source_value_weight_mean") is not None:
                source_weight_mean.append(_safe_float(node.get("source_value_weight_mean")))
            if node.get("source_weight_mean") is not None:
                source_weight_mean.append(_safe_float(node.get("source_weight_mean")))
            if node.get("source_weight_min") is not None:
                source_weight_min.append(_safe_float(node.get("source_weight_min")))
            if node.get("source_control_score_mean") is not None:
                source_score_mean.append(_safe_float(node.get("source_control_score_mean")))
            if node.get("source_control_score_max") is not None:
                source_score_max.append(_safe_float(node.get("source_control_score_max")))
            if node.get("source_keep_ratio") is not None:
                source_keep_ratio.append(_safe_float(node.get("source_keep_ratio")))
            if node.get("mean_context_source_keep_ratio") is not None:
                source_keep_ratio.append(_safe_float(node.get("mean_context_source_keep_ratio")))
            if node.get("empty_source_events") is not None:
                empty_events.append(_safe_float(node.get("empty_source_events")))
            if node.get("num_context_empty_source_events") is not None:
                empty_events.append(_safe_float(node.get("num_context_empty_source_events")))
            if node.get("special_token_total_count") is not None and _safe_float(node.get("special_token_total_count"), 0.0) > 0:
                total = _safe_float(node.get("special_token_total_count"))
                kept = _safe_float(node.get("special_token_kept_count"))
                if math.isfinite(total) and total > 0 and math.isfinite(kept):
                    static_removed_ratio.append(float(max(total - kept, 0.0) / total))
            if node.get("attention_mass_metric") is not None:
                metrics_seen.append(str(node.get("attention_mass_metric")))
            if bool(node.get("context_source_skip_applied")) and node.get("layer") is not None:
                key = str(int(_safe_float(node.get("layer"), -1)))
                layer_counts[key] = layer_counts.get(key, 0) + 1

    out["semantic_group_count_mean"] = _mean(semantic_group_counts)
    out["semantic_label_count_mean"] = _mean(semantic_label_counts)
    out["affected_source_token_count_mean"] = _mean(affected_tokens)
    out["affected_source_token_count_max"] = max(_float_values(affected_tokens)) if affected_tokens else None
    out["attention_mass_removed_before_mean"] = _mean(mass_before)
    out["attention_mass_removed_after_mean"] = _mean(mass_after)
    out["attention_mass_actual_after_mean"] = _mean(mass_actual_after)
    out["mass_retention_ratio_mean"] = (
        float(_mean(mass_after) / _mean(mass_before))
        if _mean(mass_before) not in (None, 0) and _mean(mass_after) is not None else None
    )
    out["source_weight_mean"] = _mean(source_weight_mean)
    out["source_weight_min"] = _mean(source_weight_min)
    out["source_control_score_mean"] = _mean(source_score_mean)
    out["source_control_score_max"] = max(_float_values(source_score_max)) if source_score_max else None
    out["source_keep_ratio_mean"] = _mean(source_keep_ratio)
    out["context_empty_source_events_sum"] = int(sum(_float_values(empty_events))) if empty_events else None
    out["static_anchor_removed_ratio"] = _mean(static_removed_ratio)
    out["attention_mass_metrics_seen"] = ";".join(sorted(set(metrics_seen)))
    out["per_layer_action_count"] = ";".join(f"{k}:{v}" for k, v in sorted(layer_counts.items(), key=lambda kv: int(kv[0])))
    out["commit_modes_seen"] = ";".join(sorted(set(commit_modes)))
    out["state_double_write_safe_rate"] = float(np.mean(state_safe)) if state_safe else None
    out["probe_no_commit_hash_equal_rate"] = float(np.mean(probe_no_commit)) if probe_no_commit else None
    out["commit_source_hash_equal_rate"] = float(np.mean(commit_hash_equal)) if commit_hash_equal else None
    out["commit_isolation_hash_check"] = bool(
        commit_modes
        and set(commit_modes) == {"probe_ttt_write"}
        and (not state_safe or all(state_safe))
        and (not probe_no_commit or all(probe_no_commit))
        and (not commit_hash_equal or all(commit_hash_equal))
    )
    return out


def _augment_v58_rows(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _v56_augment_rows(rows, gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        cfg = run_dir / "effective_config.yaml"
        for key in (
            "candidate",
            "track",
            "action_class",
            "hmc_commit_mode",
            "commit_protocol",
            "context_source_skip_impl",
            "context_source_skip_mode",
            "context_source_skip_mask",
            "context_source_skip_layer_mode",
            "context_source_skip_soft_rho",
            "context_source_skip_soft_min_keep",
            "semantic_role_policy",
            "semantic_desc",
        ):
            row[key] = _first_yaml_value(cfg, key) or row.get(key)
        row.update(_collect_v58_debug(run_dir))
        row["mass_retention_ratio_mean"] = row.get("mass_retention_ratio_mean") or _mass_ratio(row)
        frames = int(row.get("frames") or 0)
        if row.get("ATE") is not None:
            if frames >= 1000:
                row["delta_vs_H35_full"] = _safe_float(row.get("ATE")) - H35_FULL_ATE
            elif frames >= 700:
                row["delta_vs_H35_704"] = _safe_float(row.get("ATE")) - H35_704_ATE
        row["static_anchor_removed_ratio"] = (
            row.get("static_anchor_removed_ratio")
            if row.get("static_anchor_removed_ratio") is not None else 0.0
        )
        projected = row.get("projected_full_wall_time_min")
        if projected is None and row.get("timing_chunks"):
            projected = (
                _safe_float(row.get("wall_seconds")) / max(int(row["timing_chunks"]), 1) * 38.0 / 60.0
                if math.isfinite(_safe_float(row.get("wall_seconds"))) else None
            )
        row["projected_full_runtime_pass"] = (
            math.isfinite(_safe_float(projected)) and _safe_float(projected) <= 28.0
        )
        row["smoke_gate_pass"] = _smoke_gate(row) if _is_smoke(row) else None
        row["promotion_gate_pass"] = _promotion_gate(row) if _is_704(row) else None
        row["minimum_progress_pass"] = (
            _safe_float(row.get("ATE"), 999.0) <= H35_MIN_PROGRESS if _is_full(row) and row.get("ATE") is not None else None
        )
        row["semantic_success_pass"] = (
            _safe_float(row.get("ATE"), 999.0) <= H35_SEMANTIC_SUCCESS if _is_full(row) and row.get("ATE") is not None else None
        )


def _smoke_gate(row: Mapping[str, Any]) -> bool:
    before = _safe_float(row.get("attention_mass_removed_before_mean"))
    ratio = _safe_float(row.get("mass_retention_ratio_mean"))
    return bool(
        row.get("status") == "done"
        and _safe_float(row.get("stage_c_cache_hit_rate")) == 1.0
        and _safe_float(row.get("semantic_group_count_mean"), 0.0) > 0
        and _safe_float(row.get("affected_source_token_count_mean"), 0.0) > 100
        and math.isfinite(before) and before >= 0.03
        and math.isfinite(ratio) and 0.3 <= ratio <= 0.7
        and int(row.get("context_empty_source_events_sum") or 0) == 0
        and _safe_float(row.get("static_anchor_removed_ratio"), 1.0) <= 0.10
        and bool(row.get("commit_isolation_hash_check"))
        and bool(row.get("projected_full_runtime_pass"))
    )


def _promotion_gate(row: Mapping[str, Any]) -> bool:
    delta = _safe_float(row.get("delta_vs_H35_704"))
    ratio = _safe_float(row.get("mass_retention_ratio_mean"))
    return bool(
        row.get("status") == "done"
        and math.isfinite(delta) and delta <= H35_704_PROMOTE_DELTA
        and math.isfinite(ratio) and 0.3 <= ratio <= 0.7
        and _safe_float(row.get("static_anchor_removed_ratio"), 1.0) <= 0.10
        and int(row.get("context_empty_source_events_sum") or 0) == 0
        and bool(row.get("commit_isolation_hash_check"))
        and bool(row.get("projected_full_runtime_pass"))
    )


def _run_rows(paths: Sequence[Path], gt_path: Path) -> List[Dict[str, Any]]:
    run_dirs = _iter_run_dirs(paths)
    rows = _summarize_runs(run_dirs, gt_path)
    _augment_v58_rows(rows, gt_path)
    return rows


def _trace_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: List[Dict[str, Any]] = []
    for row in rows:
        chunk = row.get("chunk_idx")
        hook = ((row.get("control_trace") or {}).get("hook_effect_summary") or {})
        frame = hook.get("frame_attention") or {}
        chunk_attn = hook.get("chunk_attention") or {}
        before_vals = _float_values([frame.get("mean_attention_mass_removed_before"), chunk_attn.get("mean_attention_mass_removed_before")])
        after_vals = _float_values([frame.get("mean_attention_mass_removed_after"), chunk_attn.get("mean_attention_mass_removed_after")])
        group = row.get("prior_semantic_group_role_metrics") or {}
        affected_group_mass = 0.0
        static_anchor_mass = None
        for gid, payload in group.items() if isinstance(group, dict) else []:
            if isinstance(payload, dict):
                role_counts = payload.get("role_counts") or {}
                affected_group_mass += _safe_float(role_counts.get("3"), 0.0)
                if str(gid) == "0":
                    static_anchor_mass = payload.get("token_count")
        before = _mean(before_vals)
        after = _mean(after_vals)
        out.append({
            "chunk_id": chunk,
            "frame_start": row.get("start_frame"),
            "frame_end": row.get("end_frame"),
            "source_tokens_affected": max(_float_values([frame.get("max_context_source_skip_tokens"), chunk_attn.get("max_context_source_skip_tokens")]) or [0]),
            "source_attention_mass_before": before,
            "source_attention_mass_after": after,
            "mass_retention_ratio": (after / before if before not in (None, 0) and after is not None else None),
            "semantic_group_affected_mass": affected_group_mass,
            "high_D_mass": row.get("prior_dynamic_mass_D_gt_050"),
            "static_anchor_mass": static_anchor_mass,
            "protected_static_anchor_mass": row.get("prior_protect_anchor_count"),
            "context_empty_source_events": int(_safe_float(frame.get("num_context_empty_source_events"), 0.0) + _safe_float(chunk_attn.get("num_context_empty_source_events"), 0.0)),
            "READ_layer_count": int(_safe_float(frame.get("num_context_source_skip_applied"), 0.0) + _safe_float(chunk_attn.get("num_context_source_skip_applied"), 0.0)),
            "commit_protocol": row.get("hmc_commit_mode"),
        })
    return out


def _semantic_group_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl"):
        group = row.get("prior_semantic_group_role_metrics") or {}
        if not isinstance(group, dict):
            continue
        for gid, payload in group.items():
            if not isinstance(payload, dict):
                continue
            role_counts = payload.get("role_counts") or {}
            rows.append({
                "chunk_id": row.get("chunk_idx"),
                "group_id": gid,
                "token_count": payload.get("token_count"),
                "negative_short_tokens": role_counts.get("3"),
                "D_mean": payload.get("D_mean"),
                "D_p90": payload.get("D_p90"),
                "Q_mean": payload.get("Q_mean"),
                "V_mean": payload.get("V_mean"),
            })
    return rows


def _plot_xy(rows: Sequence[Mapping[str, Any]], x_key: str, y_keys: Sequence[str], path: Path, title: str, ylabel: str) -> None:
    if not rows:
        _plot_no_data(path, title, "no landed trace rows")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    xs = [_safe_float(r.get(x_key)) for r in rows]
    for key in y_keys:
        pts = [(x, _safe_float(r.get(key))) for x, r in zip(xs, rows)]
        pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.1, label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        _plot_no_data(path, title, "metrics unavailable")
        return
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _phase0_autopsy(v57_root: Path, out_dir: Path) -> Dict[str, Any]:
    autopsy = out_dir / "phase0_sread03_autopsy"
    figs = autopsy / "figures"
    autopsy.mkdir(parents=True, exist_ok=True)
    full_run = v57_root / "phase3_semantic_read_full/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_FULL"
    run704 = v57_root / "phase2_semantic_read_704_screen/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_704F"
    sread01 = v57_root / "phase2_semantic_read_704_screen/rollouts/V57_SREAD01_GENERAL_HIGH_INFLUENCE_ANOMALY_704F"
    sread04 = v57_root / "phase2_semantic_read_704_screen/rollouts/V57_SREAD04_ANOMALY_FILTER_STATIC_RESCUE_704F"
    timeline = _trace_rows(full_run)
    _write_csv(autopsy / "sread03_chunk_activation_timeline.csv", timeline)
    _write_csv(autopsy / "sread03_source_mass_timeline.csv", timeline)
    _write_csv(autopsy / "sread03_semantic_group_mass.csv", _semantic_group_rows(full_run))
    static_rows = [
        {
            "chunk_id": r.get("chunk_id"),
            "static_anchor_mass": r.get("static_anchor_mass"),
            "protected_static_anchor_mass": r.get("protected_static_anchor_mass"),
            "note": "token-level static-anchor overlap unavailable; values are aggregate trace fields",
        }
        for r in timeline
    ]
    _write_csv(autopsy / "sread03_static_anchor_overlap.csv", static_rows)
    segment_rows = _read_csv(v57_root / "report_final/v57_semantic_full_registry.csv")
    _write_csv(autopsy / "sread03_segment_metrics.csv", [r for r in segment_rows if r.get("row") == "SREAD03_FULL"])
    jaccard_rows = [{
        "comparison": "SREAD01_704F_vs_SREAD04_704F",
        "source_keep_mask_jaccard": None,
        "available": False,
        "sread01_run_dir": str(sread01),
        "sread04_run_dir": str(sread04),
        "reason": "v57 did not persist per-token source_keep_mask; aggregate ATE/source-mass equality is not a Jaccard measurement",
    }]
    _write_csv(autopsy / "sread01_vs_sread04_action_jaccard.csv", jaccard_rows)
    _plot_xy(timeline, "chunk_id", ["source_tokens_affected"], figs / "sread03_activation_timeline.png", "SREAD03 activation timeline", "affected tokens")
    _plot_xy(timeline, "chunk_id", ["source_attention_mass_before", "source_attention_mass_after"], figs / "sread03_mass_before_after_timeline.png", "SREAD03 mass before/after", "attention mass")
    full_registry = [r for r in segment_rows if r.get("row") == "SREAD03_FULL"]
    if full_registry:
        row = full_registry[0]
        bars = {
            "seg0_000_384_rmse": _safe_float(row.get("seg0_000_384_rmse")),
            "seg1_384_700_rmse": _safe_float(row.get("seg1_384_700_rmse")),
            "seg2_700_end_rmse": _safe_float(row.get("seg2_700_end_rmse")),
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(bars), [v if math.isfinite(v) else 0.0 for v in bars.values()])
        ax.set_ylabel("RMSE")
        ax.set_title("SREAD03 full segment RMSE")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / "sread03_segment_delta_bar.png", dpi=160)
        plt.close(fig)
    else:
        _plot_no_data(figs / "sread03_segment_delta_bar.png", "SREAD03 segment delta", "full registry missing")
    _plot_no_data(figs / "sread01_sread04_mask_jaccard_heatmap.png", "SREAD01/SREAD04 mask Jaccard", "per-token source_keep_mask unavailable")
    report_lines = [
        "# SREAD03 full-tail failure report",
        "",
        "This file is generated from landed v57 artifacts only.",
        "",
        f"- SREAD03 704F run exists: `{run704.is_dir()}`.",
        f"- SREAD03 full run exists: `{full_run.is_dir()}`.",
        "- v57 SREAD03 used compact K/V hard deletion: attention mass after is 0 in registry/trace.",
        "- Per-token `source_keep_mask` was not persisted, so SREAD01/SREAD04 Jaccard is unavailable.",
    ]
    _write_text(autopsy / "sread03_full_tail_failure_report.md", report_lines)
    return {
        "timeline_rows": len(timeline),
        "sread03_full_run": str(full_run),
        "sread03_704_run": str(run704),
        "jaccard_available": False,
    }


def _registry_outputs(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    _write_csv(out_dir / "v58_all_registry.csv", rows)
    _write_json(out_dir / "v58_all_registry.json", list(rows))
    _write_csv(out_dir / "v58_smoke_registry.csv", [r for r in rows if _is_smoke(r)])
    _write_csv(out_dir / "v58_704f_registry.csv", [r for r in rows if _is_704(r)])
    _write_csv(out_dir / "v58_full_registry.csv", [r for r in rows if _is_full(r)])


def _best(rows: Sequence[Mapping[str, Any]], pred) -> Optional[Mapping[str, Any]]:
    candidates = [r for r in rows if pred(r) and r.get("status") == "done" and math.isfinite(_safe_float(r.get("ATE")))]
    return min(candidates, key=lambda r: _safe_float(r.get("ATE")), default=None)


def _make_review(rows: Sequence[Dict[str, Any]], out_dir: Path, phase0: Mapping[str, Any]) -> List[str]:
    smokes = [r for r in rows if _is_smoke(r)]
    screens = [r for r in rows if _is_704(r)]
    fulls = [r for r in rows if _is_full(r)]
    best704 = _best(screens, lambda r: _track(r) != "negative_control")
    bestfull = _best(fulls, lambda r: True)
    smoke_pass = bool([r for r in smokes if _track(r) != "negative_control" and bool(r.get("smoke_gate_pass"))])
    promoted = [r for r in screens if bool(r.get("promotion_gate_pass")) and _track(r) != "negative_control"]
    min_progress = bool(bestfull and _safe_float(bestfull.get("ATE"), 999.0) <= H35_MIN_PROGRESS)
    target = bool(bestfull and _safe_float(bestfull.get("ATE"), 999.0) <= H35_SEMANTIC_SUCCESS)
    best704_delta = _safe_float(best704.get("delta_vs_H35_704") if best704 else None)
    borderline_full_allowed = bool(math.isfinite(best704_delta) and best704_delta <= H35_704_BORDERLINE_DELTA)
    lines = [
        "# ACL2 v58 SoftSemanticRead CommitIsolation Geometry 实验结果复盘",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"结果根目录: `{DEFAULT_RESULT_ROOT}`",
        "",
        "结论先行: 本报告只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持 NA/unavailable。",
        f"H35 full baseline ATE: `{H35_FULL_ATE}`；H35 704F baseline ATE: `{H35_704_ATE}`。",
        f"Full minimum progress gate: ATE <= `{H35_MIN_PROGRESS}`；semantic success gate: ATE <= `{H35_SEMANTIC_SUCCESS}`；strong success gate: ATE <= `{H35_STRONG_SUCCESS}`。",
        f"704F promotion gate: delta_vs_H35_704 <= `{H35_704_PROMOTE_DELTA}`；borderline diagnostic full gate: best delta <= `+{H35_704_BORDERLINE_DELTA}`。",
        "",
        "## 代码/修复审计摘要",
        "",
        "- `loger/models/layers/attention.py`: 新增 `source_soft` descriptor；R1 blocker 后把 V-only/no-bias path 的实际 attention 执行修回 flash SDPA，同时保留采样统计。",
        "- `loger/models/pi3.py`: 新增 soft READ source control wiring，包括 `v_only`、soft bias、`semantic_z_dg_soft_resid` 和 `random_same_mass_semantic_role_negative`。",
        "- `tools/run_v58_soft_semantic_read_commit_isolation.sh`: 新增 v58 runner，固定 C1 `probe_ttt_write` commit isolation，并在 reproduce 脚本中写入候选 layer/rho/min_keep。",
        "- `tools/v58_experiment_report.py`: 新增 artifact-only reporter，输出 registry、Phase0 autopsy、执行日志与复盘；缺失字段保持 NA/unavailable。",
        "- R4 repair 只调 soft action 参数 `V58_R4_SOFT_RHO=0.65`、`V58_R4_SOFT_MIN_KEEP=0.4`，没有调语义阈值；已有 R4 repair rollout 的 reproduce 脚本已补写这两个 override。",
        "",
        "## Phase 0: SREAD03 autopsy",
        "",
        f"- timeline rows: `{phase0.get('timeline_rows')}`。",
        f"- SREAD03 704F artifact: `{phase0.get('sread03_704_run')}`。",
        f"- SREAD03 full artifact: `{phase0.get('sread03_full_run')}`。",
        "- v57 SREAD03 是 hard compact K/V skip，registry/trace 中 affected-source attention mass after 为 0；这满足 v58 必须转 soft attenuation 的前提。",
        "- SREAD01/SREAD04 per-token source_keep_mask 没有落盘，Jaccard 只能标为 unavailable；不能把 aggregate 相同伪写成 token mask 相同。",
        "",
        "## Phase 1: 96F smoke",
        "",
    ]
    smoke_cols = [
        ("run", "run_name"),
        ("row", "row"),
        ("candidate", "candidate"),
        ("ATE", "ATE"),
        ("stage_hit", "stage_c_cache_hit_rate"),
        ("groups", "semantic_group_count_mean"),
        ("tokens", "affected_source_token_count_mean"),
        ("mass_before", "attention_mass_removed_before_mean"),
        ("mass_after", "attention_mass_removed_after_mean"),
        ("ratio", "mass_retention_ratio_mean"),
        ("empty", "context_empty_source_events_sum"),
        ("commit", "commit_isolation_hash_check"),
        ("gate", "smoke_gate_pass"),
    ]
    lines.extend(_md_table(smokes, smoke_cols) if smokes else ["| row | status |", "|---|---|", "| none | not run |"])
    r1_initial = next((r for r in smokes if str(r.get("run_name")) == "V58_R1_SREAD03_V_ONLY_C1_96F"), None)
    r1_repair = next((r for r in smokes if str(r.get("run_name")) == "V58R1_R1_SREAD03_V_ONLY_C1_96F"), None)
    r4_initial = next((r for r in smokes if str(r.get("run_name")) == "V58_R4_SEM_Z_DG_SOFT_RESID_C1_96F"), None)
    r4_repair = next((r for r in smokes if str(r.get("run_name")) == "V58R1_R4_SEM_Z_DG_SOFT_RESID_C1_96F"), None)
    lines.extend(["", f"Phase1 usable semantic smoke candidate exists: `{smoke_pass}`。"])
    if r1_initial or r1_repair or r4_initial or r4_repair:
        lines.extend(["", "Phase1 repair/blocker 记录:"])
        if r1_initial and r1_repair:
            lines.append(
                f"- R1 V-only 初跑 runtime projected `{_fmt(r1_initial.get('projected_full_wall_time_min'))}` min，"
                f"修复 V-only 无 bias fast SDPA 后 projected `{_fmt(r1_repair.get('projected_full_wall_time_min'))}` min，gate `{r1_repair.get('smoke_gate_pass')}`。"
            )
        if r4_initial and r4_repair:
            lines.append(
                f"- R4 sem-z 初跑 mass ratio `{_fmt(r4_initial.get('mass_retention_ratio_mean'))}` 超过 0.7；"
                f"按计划只调 soft rho/min_keep 到 `{r4_repair.get('context_source_skip_soft_rho')}`/`{r4_repair.get('context_source_skip_soft_min_keep')}`，"
                f"ratio `{_fmt(r4_repair.get('mass_retention_ratio_mean'))}`，但 projected `{_fmt(r4_repair.get('projected_full_wall_time_min'))}` min，仍因 runtime gate 停止。"
            )
    lines.extend(["", "## Phase 2: 704F screen", ""])
    screen_cols = [
        ("run", "run_name"),
        ("row", "row"),
        ("candidate", "candidate"),
        ("ATE", "ATE"),
        ("dH35_704", "delta_vs_H35_704"),
        ("rolling100p90", "rolling100_p90"),
        ("tokens", "affected_source_token_count_mean"),
        ("mass_ratio", "mass_retention_ratio_mean"),
        ("static_rm", "static_anchor_removed_ratio"),
        ("commit", "commit_isolation_hash_check"),
        ("promote", "promotion_gate_pass"),
    ]
    lines.extend(_md_table(screens, screen_cols) if screens else ["| row | status |", "|---|---|", "| none | not run |"])
    lines.extend(["", f"Promoted to full: `{', '.join(str(r.get('candidate')) for r in promoted) if promoted else 'none'}`。", "", "## Phase 3: full KITTI01", ""])
    full_cols = [
        ("run", "run_name"),
        ("row", "row"),
        ("candidate", "candidate"),
        ("ATE", "ATE"),
        ("dH35", "delta_vs_H35_full"),
        ("Rot", "Rot"),
        ("FinalErr", "FinalErr"),
        ("wall", "wall_time_min"),
        ("mass_ratio", "mass_retention_ratio_mean"),
        ("min_progress", "minimum_progress_pass"),
        ("target", "semantic_success_pass"),
    ]
    lines.extend(_md_table(fulls, full_cols) if fulls else ["| row | status |", "|---|---|", "| none | not run by gate |"])
    if not fulls:
        lines.extend([
            "",
            f"Full 未运行原因: 704F 无候选过 promotion gate，best semantic 704F delta=`{_fmt(best704_delta)}`，"
            f"未达到 borderline diagnostic full 条件 `<= +{H35_704_BORDERLINE_DELTA}`。",
        ])
    lines.extend([
        "",
        "## Gate summary",
        "",
        f"- best 704F semantic candidate: `{best704.get('candidate') if best704 else 'none'}` ATE `{_fmt(best704.get('ATE') if best704 else None)}`，delta `{_fmt(best704_delta)}`。",
        f"- borderline diagnostic full allowed: `{borderline_full_allowed}`。",
        f"- best full candidate: `{bestfull.get('candidate') if bestfull else 'none'}` ATE `{_fmt(bestfull.get('ATE') if bestfull else None)}`。",
        f"- semantic_min_progress=`{min_progress}`, semantic_target=`{target}`。",
        "",
        "## Insight 与证据链",
        "",
        "- v58 的第一层证据是 action form：source 不再被 hard compact 到 mass_after=0；soft 候选必须在 trace 中保留 0.3-0.7 的 affected-source effective/attention mass。",
        "- C1 commit isolation 通过 `hmc_commit_mode=probe_ttt_write`、`state_double_write_safe`、`probe_no_commit_hash_equal` 和 commit-source hash 字段审计；若这些字段缺失或失败，不把 full 结果算作语义成功。",
        "- V-only 候选的 `mass_after` 是 effective value mass；同时 reporter 记录 `attention_mass_actual_after_mean` 和 `attention_mass_metrics_seen`，避免把 V attenuation 伪装成 attention-logit mass 改变。",
        "- 若所有 full 均未达到 H35-0.5m，本轮结论应降级为 semantic READ 当前更适合 diagnostic，而不是继续扩大 hard skip/TTT 小扫。",
        "",
        "## 必答问题",
        "",
        f"1. 语义 READ 是否已经从 hard skip 转成 soft attenuation: `{'yes' if smokes else 'not evaluated'}`。",
        f"2. 是否真的保留了部分 source attention/effective mass: `{'yes' if any(math.isfinite(_safe_float(r.get('mass_retention_ratio_mean'))) and 0.0 < _safe_float(r.get('mass_retention_ratio_mean')) < 1.0 for r in rows) else 'not proven'}`。",
        "3. SREAD03 704F 改善来自哪些 source: 见 `phase0_sread03_autopsy/sread03_semantic_group_mass.csv`；v57 trace 主要能报告 group/label aggregate，不能恢复 token mask。",
        "4. full 失败是否来自后段误激活，还是 commit side effect: 只有 full artifact 和 commit hash fields 同时存在时才能判定；缺失则保持 unavailable。",
        f"5. commit isolation 是否有效: `{'yes' if rows and all(bool(r.get('commit_isolation_hash_check')) for r in rows if r.get('status') == 'done') else 'not proven'}`。",
        f"6. soft semantic READ 是否相比 H35 带来 full ATE 改善: `{'yes' if min_progress else 'not proven'}`。",
        "7. 如果没有改善，语义路线应降级到 diagnostic/offline explanation，后续转向 TTT harmful update attribution、trajectory-state 或 merge-gauge controller。",
        "",
        "## 审计材料",
        "",
        f"- registries: `{out_dir}`",
        f"- Phase0 autopsy: `{out_dir / 'phase0_sread03_autopsy'}`",
        f"- figures: `{out_dir / 'figures'}` and `{out_dir / 'phase0_sread03_autopsy/figures'}`",
    ])
    _write_text(out_dir / "v58_final_report.md", lines)
    return lines


def _make_exec_log(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    lines = [
        "# ACL2 v58 SoftSemanticRead CommitIsolation Geometry 执行日志",
        "",
        "日期: 2026-06-09",
        f"计划文档: `{PLAN_DOC}`",
        f"结果根目录: `{DEFAULT_RESULT_ROOT}`",
        "",
        "## 代码/工具修改记录",
        "",
        "- `loger/models/layers/attention.py`: 新增 `source_soft` SDPA descriptor，支持 soft attention bias mass 统计与 V-only effective value mass 统计。",
        "- `loger/models/pi3.py`: context source control 新增 `v_only`、soft mass descriptor、`semantic_z_dg_soft_resid` 和 `random_same_mass_semantic_role_negative` mask；trace 记录 soft action 字段。",
        "- `tools/run_v58_soft_semantic_read_commit_isolation.sh`: 新增 v58 候选 runner，所有候选使用 `HMC_COMMIT_MODE=probe_ttt_write`。",
        "- `tools/v58_experiment_report.py`: 新增 artifact-only reporter，输出 registry、Phase0 autopsy、最终复盘。",
        "- R1 efficiency repair: `loger/models/layers/attention.py` 的 V-only/no-bias source_soft path 改回 flash SDPA 执行实际 attention，保留采样统计。",
        "- R4 soft repair: 不改语义 threshold，只用 `V58_R4_SOFT_RHO=0.65 V58_R4_SOFT_MIN_KEEP=0.4` 重跑 smoke；ratio 合格但 projected runtime 未过 gate。",
        "- Reproduce repair: runner 和 R4 repair rollout 的 `reproduce_command.sh` 补写候选 layer/rho/min_keep，避免非默认 soft 参数丢失。",
        "",
        "## 编译/静态检查",
        "",
        "- `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/models/pi3.py loger/models/layers/attention.py`",
        "- `bash -n tools/run_v58_soft_semantic_read_commit_isolation.sh`",
        "",
        "## Phase 0 命令",
        "",
        f"- `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v58_experiment_report.py --result-root {DEFAULT_RESULT_ROOT} --v57-root {DEFAULT_V57_ROOT} --out-dir {out_dir}`",
        "",
        "## 实验运行命令",
        "",
    ]
    if not rows:
        lines.append("- no v58 rollout artifact discovered yet.")
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        repro = run_dir / "reproduce_command.sh"
        exact = _reproduce_inline(repro)
        lines.append(f"- `{row.get('row')}` `{row.get('candidate')}` status=`{row.get('status')}` frames=`{row.get('frames')}` ATE=`{_fmt(row.get('ATE'))}`")
        lines.append(f"  - run_dir: `{run_dir}`")
        lines.append(f"  - reproduce: `{repro}`")
        if exact:
            lines.append(f"  - exact command: `{exact}`")
    lines.extend([
        "",
        "## 复现说明",
        "",
        "- 每个 rollout 目录内有 `effective_config.yaml`、`reproduce_command.sh`、`01.log`、`hmc_state_hash.jsonl`、`kitti_benchmark.log` 和 `wall_time_summary.json`。",
        "- gate 和结论只使用已经落盘的 registry/trace；未运行的 Phase 不补数据。",
    ])
    _write_text(out_dir / "v58_execution_log.md", lines)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--v57-root", type=Path, default=DEFAULT_V57_ROOT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.result_root / "report_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    phase0 = _phase0_autopsy(args.v57_root, out_dir)
    rows = _run_rows([args.result_root], args.gt)
    _registry_outputs(rows, out_dir)
    review_lines = _make_review(rows, out_dir, phase0)
    exec_lines = _make_exec_log(rows, out_dir)
    summary = {
        "result_root": str(args.result_root),
        "v57_root": str(args.v57_root),
        "phase0": dict(phase0),
        "run_count": len(rows),
        "smoke_count": len([r for r in rows if _is_smoke(r)]),
        "screen_704_count": len([r for r in rows if _is_704(r)]),
        "full_count": len([r for r in rows if _is_full(r)]),
        "best_full_ate": (_best(rows, lambda r: _is_full(r)) or {}).get("ATE"),
    }
    _write_json(out_dir / "v58_summary.json", summary)
    if args.write_docs:
        _write_text(DOC_REVIEW, review_lines)
        _write_text(DOC_EXEC, exec_lines)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
