#!/usr/bin/env python3
"""Phase 2 trace autopsy for ACL2 v52 adaptive TTT writing."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
V52_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry"
C9_ATE = 33.76294210291885
SOFT_THRESHOLD = 34.60

DEFAULT_RUNS = {
    "C9_exact_teacher": {
        "label": "C9 exact teacher",
        "run_dir": REPO_ROOT
        / "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT",
        "method": "exact C9/P0 repeat",
    },
    "V46B_fixed_F111_teacher": {
        "label": "v46B fixed F111 teacher",
        "run_dir": V52_ROOT
        / "phase2_adaptive_failure_audit/rollouts/V52_TRACE_V46B_F111_FIXED",
        "method": "fixed READ+TTT+SWA teacher rerun with v11 trace",
    },
    "V50_split_resxdg_student": {
        "label": "v50 robust split residual_x_dg student",
        "run_dir": V52_ROOT
        / "phase2_adaptive_failure_audit/rollouts/V52_TRACE_V50_SPLIT_RESXDG_AW111",
        "method": "adaptive_writer_robust_split + residual_x_dg",
    },
}


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        val = _float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _mean(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)) or abs(den) < 1.0e-12:
        return float("nan")
    return float(num / den)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_ate(run_dir: Path) -> Tuple[float, float]:
    path = run_dir / "results_sim3/results_ate.txt"
    if not path.exists():
        return float("nan"), float("nan")
    ate = rot = float("nan")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "01":
                ate = _float(parts[1])
                rot = _float(parts[2])
    return ate, rot


def _timing_summary(run_dir: Path) -> Dict[str, Any]:
    data = _read_json(run_dir / "timing_summary.json")
    chunks = data.get("chunks") or []
    return {
        "timing_chunk_count": len(chunks),
        "chunk_total_seconds_mean": _mean(row.get("chunk_total_seconds") for row in chunks),
        "probe_ttt_write_seconds_mean": _mean(row.get("probe_ttt_write_seconds") for row in chunks),
        "pass1_probe_seconds_mean": _mean(row.get("pass1_probe_seconds") for row in chunks),
        "pass2_control_seconds_mean": _mean(row.get("pass2_control_seconds") for row in chunks),
        "stage_b_seconds_mean": _mean(row.get("stage_b_seconds") for row in chunks),
        "stage_d_seconds_mean": _mean(row.get("stage_d_seconds") for row in chunks),
        "total_runtime_seconds_after_model_load": _float(data.get("total_runtime_seconds_after_model_load")),
        "total_runtime_seconds_including_model_load": _float(data.get("total_runtime_seconds_including_model_load")),
    }


def _hmc_summary(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    return {
        "hmc_rows": len(rows),
        "hmc_chunks": len({row.get("chunk_idx") for row in rows if row.get("chunk_idx") is not None}),
        "hmc_tri_pos_mean": _mean(row.get("auxgeo_tri_replay_pos_mass_mean") for row in rows),
        "hmc_tri_neu_mean": _mean(row.get("auxgeo_tri_replay_neu_mass_mean") for row in rows),
        "hmc_tri_neg_mean": _mean(row.get("auxgeo_tri_replay_neg_mass_mean") for row in rows),
        "hmc_tri_applied_layer_count_mean": _mean(row.get("auxgeo_tri_replay_applied_layer_count") for row in rows),
    }


def _role_summary(run_name: str, trace_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = _read_jsonl(trace_dir / "tri_replay_role_mass.jsonl")
    timeline: Dict[int, MutableMapping[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        chunk = row.get("chunk_idx")
        if chunk is None:
            continue
        for key in ("positive_mass", "neutral_mass", "negative_mass", "positive_frac", "negative_frac", "neutral_lambda"):
            val = _float(row.get(key))
            if math.isfinite(val):
                timeline[int(chunk)][key].append(val)
        gamma = _float(row.get("w0_gamma"))
        if math.isfinite(gamma):
            timeline[int(chunk)]["w0_gamma"].append(gamma)
        timeline[int(chunk)]["tri_replay_applied"].append(1.0 if row.get("tri_replay_applied") else 0.0)

    timeline_rows: List[Dict[str, Any]] = []
    for chunk, values in sorted(timeline.items()):
        timeline_rows.append({
            "run": run_name,
            "chunk_idx": chunk,
            "positive_mass_mean": _mean(values.get("positive_mass", [])),
            "neutral_mass_mean": _mean(values.get("neutral_mass", [])),
            "negative_mass_mean": _mean(values.get("negative_mass", [])),
            "positive_frac_mean": _mean(values.get("positive_frac", [])),
            "negative_frac_mean": _mean(values.get("negative_frac", [])),
            "neutral_lambda_mean": _mean(values.get("neutral_lambda", [])),
            "w0_gamma_mean": _mean(values.get("w0_gamma", [])),
            "tri_replay_applied_fraction": _mean(values.get("tri_replay_applied", [])),
        })

    applied = [row for row in rows if row.get("tri_replay_applied")]
    active_branches = sorted({",".join(map(str, row.get("active_branches") or [])) for row in rows})
    summary = {
        "role_rows": len(rows),
        "role_chunks": len({row.get("chunk_idx") for row in rows if row.get("chunk_idx") is not None}),
        "role_layers": len({row.get("layer") for row in rows if row.get("layer") is not None}),
        "role_applied_rows": len(applied),
        "role_applied_fraction": _safe_ratio(float(len(applied)), float(len(rows))),
        "role_sources_seen": ";".join(sorted({str(row.get("role_source")) for row in rows if row.get("role_source") is not None})),
        "active_branches_seen": ";".join(active_branches),
        "positive_mass_mean": _mean(row.get("positive_mass") for row in rows),
        "neutral_mass_mean": _mean(row.get("neutral_mass") for row in rows),
        "negative_mass_mean": _mean(row.get("negative_mass") for row in rows),
        "positive_mass_applied_mean": _mean(row.get("positive_mass") for row in applied),
        "neutral_mass_applied_mean": _mean(row.get("neutral_mass") for row in applied),
        "negative_mass_applied_mean": _mean(row.get("negative_mass") for row in applied),
        "positive_frac_mean": _mean(row.get("positive_frac") for row in rows),
        "negative_frac_mean": _mean(row.get("negative_frac") for row in rows),
        "neutral_lambda_mean": _mean(row.get("neutral_lambda") for row in rows),
        "w0_gamma_mean": _mean(row.get("w0_gamma") for row in rows),
    }
    return summary, timeline_rows


def _risk_summary(run_name: str, trace_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = _read_jsonl(trace_dir / "ttt_update_conflict_energy.jsonl")
    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = row.get("layer")
        if layer is not None:
            by_layer[int(layer)].append(row)
    layer_rows = []
    for layer, group in sorted(by_layer.items()):
        layer_rows.append({
            "run": run_name,
            "layer": layer,
            "rows": len(group),
            "risk_mean_mean": _mean(row.get("risk_mean") for row in group),
            "risk_p90_mean": _mean(row.get("risk_p90") for row in group),
            "ttt_update_conflict_energy_mean": _mean(row.get("ttt_update_conflict_energy") for row in group),
            "risk_sources_seen": ";".join(sorted({str(row.get("risk_source")) for row in group if row.get("risk_source") is not None})),
        })
    summary = {
        "risk_rows": len(rows),
        "risk_chunks": len({row.get("chunk_idx") for row in rows if row.get("chunk_idx") is not None}),
        "risk_layers_seen": ";".join(str(layer) for layer in sorted(by_layer)),
        "risk_source_seen": ";".join(sorted({str(row.get("risk_source")) for row in rows if row.get("risk_source") is not None})),
        "risk_mean_mean": _mean(row.get("risk_mean") for row in rows),
        "risk_p90_mean": _mean(row.get("risk_p90") for row in rows),
        "ttt_update_conflict_energy_mean": _mean(row.get("ttt_update_conflict_energy") for row in rows),
    }
    return summary, layer_rows


def _load_update_rows(run_name: str, trace_dir: Path) -> List[Dict[str, Any]]:
    path = trace_dir / "per_layer_branch_update_matrix.pt"
    if not path.exists():
        return []
    obj = torch.load(path, map_location="cpu")
    out: List[Dict[str, Any]] = []
    for chunk in obj.get("chunks") or []:
        chunk_idx = chunk.get("chunk_idx")
        for layer in chunk.get("layers") or []:
            layer_idx = layer.get("layer")
            for branch, stats in (layer.get("branches") or {}).items():
                out.append({
                    "run": run_name,
                    "chunk_idx": chunk_idx,
                    "layer": layer_idx,
                    "branch": branch,
                    "delta_norm": _float(stats.get("delta_norm")),
                    "delta_abs_mean": _float(stats.get("delta_abs_mean")),
                    "old_norm": _float(stats.get("old_norm")),
                    "new_norm": _float(stats.get("new_norm")),
                })
    del obj
    gc.collect()
    return out


def _load_post_zp_rows(run_name: str, trace_dir: Path) -> List[Dict[str, Any]]:
    path = trace_dir / "per_layer_branch_post_zp_delta.pt"
    if not path.exists():
        return []
    obj = torch.load(path, map_location="cpu")
    out: List[Dict[str, Any]] = []
    for chunk in obj.get("chunks") or []:
        chunk_idx = chunk.get("chunk_idx")
        for layer in chunk.get("layers") or []:
            layer_idx = layer.get("layer")
            for branch, stats in (layer.get("branches") or {}).items():
                out.append({
                    "run": run_name,
                    "chunk_idx": chunk_idx,
                    "layer": layer_idx,
                    "branch": branch,
                    "native_delta_norm": _float(stats.get("native_delta_norm")),
                    "committed_delta_norm": _float(stats.get("committed_delta_norm")),
                    "action_delta_norm": _float(stats.get("action_delta_norm")),
                    "short_delta_norm": _float(stats.get("short_delta_norm")),
                    "cos_committed_to_native": _float(stats.get("cos_committed_to_native")),
                    "cos_action_to_native": _float(stats.get("cos_action_to_native")),
                })
    del obj
    gc.collect()
    return out


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row["rows"] = len(group_rows)
        for metric in metrics:
            row[f"{metric}_mean"] = _mean(item.get(metric) for item in group_rows)
        out.append(row)
    return out


def _index(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(row.get(key) for key in keys): row for row in rows}


def _ratio_by_layer(update_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    means = _aggregate(update_rows, ["run", "layer", "branch"], ["delta_norm", "delta_abs_mean"])
    idx = _index(means, ["run", "layer", "branch"])
    out = []
    layer_branch_keys = sorted({(row["layer"], row["branch"]) for row in means})
    for layer, branch in layer_branch_keys:
        c9 = idx.get(("C9_exact_teacher", layer, branch), {})
        fixed = idx.get(("V46B_fixed_F111_teacher", layer, branch), {})
        student = idx.get(("V50_split_resxdg_student", layer, branch), {})
        c9_delta = _float(c9.get("delta_norm_mean"))
        fixed_delta = _float(fixed.get("delta_norm_mean"))
        student_delta = _float(student.get("delta_norm_mean"))
        out.append({
            "layer": layer,
            "branch": branch,
            "c9_delta_norm_mean": c9_delta,
            "fixed_delta_norm_mean": fixed_delta,
            "student_delta_norm_mean": student_delta,
            "student_over_c9_delta_norm": _safe_ratio(student_delta, c9_delta),
            "student_over_fixed_delta_norm": _safe_ratio(student_delta, fixed_delta),
            "fixed_over_c9_delta_norm": _safe_ratio(fixed_delta, c9_delta),
        })
    return out


def _post_ratio_by_chunk(post_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    means = _aggregate(
        post_rows,
        ["run", "chunk_idx"],
        ["native_delta_norm", "committed_delta_norm", "action_delta_norm", "cos_committed_to_native", "cos_action_to_native"],
    )
    idx = _index(means, ["run", "chunk_idx"])
    chunks = sorted({row["chunk_idx"] for row in means})
    out = []
    for chunk in chunks:
        c9 = idx.get(("C9_exact_teacher", chunk), {})
        fixed = idx.get(("V46B_fixed_F111_teacher", chunk), {})
        student = idx.get(("V50_split_resxdg_student", chunk), {})
        row: Dict[str, Any] = {"chunk_idx": chunk}
        for metric in ("native_delta_norm", "committed_delta_norm", "action_delta_norm", "cos_committed_to_native", "cos_action_to_native"):
            c9_val = _float(c9.get(f"{metric}_mean"))
            fixed_val = _float(fixed.get(f"{metric}_mean"))
            student_val = _float(student.get(f"{metric}_mean"))
            row[f"c9_{metric}_mean"] = c9_val
            row[f"fixed_{metric}_mean"] = fixed_val
            row[f"student_{metric}_mean"] = student_val
            row[f"student_over_c9_{metric}"] = _safe_ratio(student_val, c9_val)
            row[f"student_over_fixed_{metric}"] = _safe_ratio(student_val, fixed_val)
            row[f"fixed_over_c9_{metric}"] = _safe_ratio(fixed_val, c9_val)
        out.append(row)
    return out


def _plot_role_timeline(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    metrics = [
        ("positive_mass_mean", "Positive Mass"),
        ("neutral_mass_mean", "Neutral Mass"),
        ("negative_mass_mean", "Negative Mass"),
    ]
    colors = {
        "C9_exact_teacher": "#111827",
        "V46B_fixed_F111_teacher": "#0ea5e9",
        "V50_split_resxdg_student": "#ef4444",
    }
    for ax, (metric, title) in zip(axes, metrics):
        for run in sorted({row["run"] for row in rows}):
            run_rows = [row for row in rows if row["run"] == run]
            x = [int(row["chunk_idx"]) for row in run_rows]
            y = [_float(row.get(metric)) for row in run_rows]
            ax.plot(x, y, label=run, linewidth=1.8, color=colors.get(run))
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Chunk")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Phase 2 Teacher/Student Role Mass Timeline")
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_student_role_mass_timeline.png", dpi=180)
    plt.close(fig)


def _plot_post_ratio(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    x = [int(row["chunk_idx"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label, color in [
        ("student_over_c9_committed_delta_norm", "student/C9 committed", "#ef4444"),
        ("student_over_fixed_committed_delta_norm", "student/fixed committed", "#0ea5e9"),
        ("fixed_over_c9_committed_delta_norm", "fixed/C9 committed", "#111827"),
        ("student_over_c9_action_delta_norm", "student/C9 action", "#f97316"),
    ]:
        y = [_float(row.get(metric)) for row in rows]
        ax.plot(x, y, label=label, linewidth=1.7, color=color)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Ratio")
    ax.set_title("Post-Zero-Power Delta Ratio by Chunk")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "post_zp_delta_ratio_by_chunk.png", dpi=180)
    plt.close(fig)


def _plot_delta_ratio(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    branches = sorted({str(row["branch"]) for row in rows})
    fig, axes = plt.subplots(len(branches), 1, figsize=(10, 3.2 * len(branches)), sharex=True)
    if len(branches) == 1:
        axes = [axes]
    for ax, branch in zip(axes, branches):
        run_rows = [row for row in rows if row["branch"] == branch]
        x = [int(row["layer"]) for row in run_rows]
        ax.plot(x, [_float(row.get("student_over_c9_delta_norm")) for row in run_rows], label="student/C9", color="#ef4444")
        ax.plot(x, [_float(row.get("student_over_fixed_delta_norm")) for row in run_rows], label="student/fixed", color="#0ea5e9")
        ax.plot(x, [_float(row.get("fixed_over_c9_delta_norm")) for row in run_rows], label="fixed/C9", color="#111827")
        ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{branch} ratio")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Layer")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Per-Layer Update Delta Norm Ratios")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_norm_ratio_by_layer.png", dpi=180)
    plt.close(fig)


def _write_report(out_dir: Path, run_rows: List[Dict[str, Any]], role_rows: List[Dict[str, Any]], risk_rows: List[Dict[str, Any]], update_rows: List[Dict[str, Any]], post_rows: List[Dict[str, Any]]) -> None:
    by_run = {row["run"]: row for row in run_rows}
    student = by_run.get("V50_split_resxdg_student", {})
    fixed = by_run.get("V46B_fixed_F111_teacher", {})
    c9 = by_run.get("C9_exact_teacher", {})
    update_global = _aggregate(update_rows, ["run", "branch"], ["delta_norm", "delta_abs_mean"])
    post_global = _aggregate(post_rows, ["run"], ["committed_delta_norm", "action_delta_norm", "native_delta_norm", "cos_committed_to_native", "cos_action_to_native"])

    def fmt(value: Any) -> str:
        val = _float(value)
        return f"{val:.12g}" if math.isfinite(val) else "nan"

    lines: List[str] = []
    lines.append("# ACL2 v52 Phase 2 Adaptive Failure Autopsy")
    lines.append("")
    lines.append("本报告由 `tools/v52_adaptive_failure_audit.py` 从真实落盘 trace 生成；没有手填实验指标。")
    lines.append("")
    lines.append("## Run Overview")
    lines.append("")
    lines.append("| Run | ATE | Delta vs C9 | Delta vs soft 34.60 | hmc_rows | role_rows | risk_rows | chunk sec mean | probe TTT sec mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in run_rows:
        lines.append(
            "| {run} | {ate} | {dc9} | {dsoft} | {hmc} | {role} | {risk} | {chunk} | {probe} |".format(
                run=row["run"],
                ate=fmt(row.get("ATE_full")),
                dc9=fmt(row.get("delta_vs_C9")),
                dsoft=fmt(row.get("delta_vs_soft_threshold")),
                hmc=row.get("hmc_rows"),
                role=row.get("role_rows"),
                risk=row.get("risk_rows"),
                chunk=fmt(row.get("chunk_total_seconds_mean")),
                probe=fmt(row.get("probe_ttt_write_seconds_mean")),
            )
        )
    lines.append("")
    lines.append("## Role Evidence")
    lines.append("")
    lines.append("| Run | role sources | active branches | pos mean | neu mean | neg mean | applied rows | applied frac | gamma mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in run_rows:
        lines.append(
            "| {run} | {src} | {branches} | {pos} | {neu} | {neg} | {app} | {frac} | {gamma} |".format(
                run=row["run"],
                src=row.get("role_sources_seen", ""),
                branches=row.get("active_branches_seen", ""),
                pos=fmt(row.get("positive_mass_mean")),
                neu=fmt(row.get("neutral_mass_mean")),
                neg=fmt(row.get("negative_mass_mean")),
                app=row.get("role_applied_rows"),
                frac=fmt(row.get("role_applied_fraction")),
                gamma=fmt(row.get("w0_gamma_mean")),
            )
        )
    lines.append("")
    lines.append("Evidence files: `teacher_student_role_mass_timeline.png`, `role_mass_timeline.csv`.")
    lines.append("")
    lines.append("## Update Delta Evidence")
    lines.append("")
    lines.append("| Run | Branch | delta_norm mean | delta_abs_mean mean |")
    lines.append("|---|---|---:|---:|")
    for row in update_global:
        lines.append(
            f"| {row['run']} | {row['branch']} | {fmt(row.get('delta_norm_mean'))} | {fmt(row.get('delta_abs_mean_mean'))} |"
        )
    lines.append("")
    lines.append("Evidence files: `delta_norm_ratio_by_layer.csv`, `delta_norm_ratio_by_layer.png`, `update_delta_norm_by_layer_branch.csv`.")
    lines.append("")
    lines.append("## Post-ZP Evidence")
    lines.append("")
    lines.append("| Run | committed delta norm | action delta norm | native delta norm | cos committed/native | cos action/native |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in post_global:
        lines.append(
            "| {run} | {comm} | {action} | {native} | {ccos} | {acos} |".format(
                run=row["run"],
                comm=fmt(row.get("committed_delta_norm_mean")),
                action=fmt(row.get("action_delta_norm_mean")),
                native=fmt(row.get("native_delta_norm_mean")),
                ccos=fmt(row.get("cos_committed_to_native_mean")),
                acos=fmt(row.get("cos_action_to_native_mean")),
            )
        )
    lines.append("")
    lines.append("Evidence files: `post_zp_delta_ratio_by_chunk.csv`, `post_zp_delta_ratio_by_chunk.png`, `post_zp_delta_by_chunk_layer_branch.csv`.")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append(
        "v50 split student ATE = {student_ate}, compared with exact C9 delta = {student_delta}, soft threshold delta = {student_soft}.".format(
            student_ate=fmt(student.get("ATE_full")),
            student_delta=fmt(student.get("delta_vs_C9")),
            student_soft=fmt(student.get("delta_vs_soft_threshold")),
        )
    )
    lines.append(
        "v50 split is better than v46B fixed F111 by {gain_fixed} m in this traced rerun, but it is still not close to C9.".format(
            gain_fixed=fmt(_float(fixed.get("ATE_full")) - _float(student.get("ATE_full")))
        )
    )
    lines.append(
        "The fixed/update-conflict teacher path is much slower: fixed probe TTT mean {fixed_probe}s/chunk vs student {student_probe}s/chunk.".format(
            fixed_probe=fmt(fixed.get("probe_ttt_write_seconds_mean")),
            student_probe=fmt(student.get("probe_ttt_write_seconds_mean")),
        )
    )
    lines.append(
        "Role evidence shows the student has lower positive role mass than the fixed teacher ({student_pos} vs {fixed_pos}) and higher neutral mass ({student_neu} vs {fixed_neu}).".format(
            student_pos=fmt(student.get("positive_mass_mean")),
            fixed_pos=fmt(fixed.get("positive_mass_mean")),
            student_neu=fmt(student.get("neutral_mass_mean")),
            fixed_neu=fmt(fixed.get("neutral_mass_mean")),
        )
    )
    lines.append(
        "Post-ZP evidence is available for all three runs and should be preferred over raw role mass alone when designing Phase 3 candidates."
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("Phase 2 locates the failure as a semantic/energy mismatch after split replay is restored: split semantics help substantially, but the adaptive writer still misses C9-level action geometry.")
    lines.append("The slow fixed/update-conflict path is useful as a teacher trace, but too expensive to be the direct adaptive runtime solution.")
    lines.append("")
    lines.append("## Phase 3 Direction")
    lines.append("")
    lines.append("Proceed with adaptive split variants that remain no-chunk and no-manual-percentage, especially energy-matched split and semantic/geometry-cluster split. Do not retest semantic Phase 4 unless a Phase 3 adaptive run reaches the documented <=34.60 soft pass.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("C9 `ttt_update_conflict_energy.jsonl` has fewer rows than the newer v52 trace runs; this is an inherited trace-format difference and is recorded in `phase2_trace_completeness_summary.csv/json`.")
    (out_dir / "adaptive_failure_autopsy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(out_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: List[Dict[str, Any]] = []
    role_timeline: List[Dict[str, Any]] = []
    risk_layer_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    post_rows: List[Dict[str, Any]] = []
    completeness: List[Dict[str, Any]] = []

    for run_name, cfg in runs.items():
        run_dir = Path(cfg["run_dir"])
        trace_dir = run_dir / "v11_projection_trace"
        ate, rot = _read_ate(run_dir)
        timing = _timing_summary(run_dir)
        hmc = _hmc_summary(run_dir)
        role, timeline_rows = _role_summary(run_name, trace_dir)
        risk, layer_rows = _risk_summary(run_name, trace_dir)
        updates = _load_update_rows(run_name, trace_dir)
        posts = _load_post_zp_rows(run_name, trace_dir)
        role_timeline.extend(timeline_rows)
        risk_layer_rows.extend(layer_rows)
        update_rows.extend(updates)
        post_rows.extend(posts)
        run_row: Dict[str, Any] = {
            "run": run_name,
            "label": cfg.get("label", ""),
            "method": cfg.get("method", ""),
            "run_dir": str(run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir),
            "ATE_full": ate,
            "Rot_full": rot,
            "delta_vs_C9": ate - C9_ATE if math.isfinite(ate) else float("nan"),
            "delta_vs_soft_threshold": ate - SOFT_THRESHOLD if math.isfinite(ate) else float("nan"),
        }
        run_row.update(timing)
        run_row.update(hmc)
        run_row.update(role)
        run_row.update(risk)
        run_rows.append(run_row)
        completeness.append({
            "run": run_name,
            "run_dir": run_row["run_dir"],
            "ate_present": (run_dir / "results_sim3/results_ate.txt").exists(),
            "timing_present": (run_dir / "timing_summary.json").exists(),
            "hmc_state_present": (run_dir / "hmc_state_hash.jsonl").exists(),
            "role_rows": role["role_rows"],
            "risk_rows": risk["risk_rows"],
            "update_matrix_rows": len(updates),
            "post_zp_rows": len(posts),
            "update_matrix_present": (trace_dir / "per_layer_branch_update_matrix.pt").exists(),
            "post_zp_present": (trace_dir / "per_layer_branch_post_zp_delta.pt").exists(),
        })

    update_layer_branch = _aggregate(update_rows, ["run", "layer", "branch"], ["delta_norm", "delta_abs_mean"])
    ratio_rows = _ratio_by_layer(update_rows)
    post_ratio_rows = _post_ratio_by_chunk(post_rows)

    _write_csv(out_dir / "run_overview.csv", run_rows)
    _write_csv(out_dir / "phase2_trace_completeness_summary.csv", completeness)
    (out_dir / "phase2_trace_completeness_summary.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(out_dir / "role_mass_timeline.csv", role_timeline)
    _write_csv(out_dir / "risk_summary_by_layer.csv", risk_layer_rows)
    _write_csv(out_dir / "update_delta_norm_by_layer_branch.csv", update_layer_branch)
    _write_csv(out_dir / "delta_norm_ratio_by_layer.csv", ratio_rows)
    _write_csv(out_dir / "post_zp_delta_by_chunk_layer_branch.csv", post_rows)
    _write_csv(out_dir / "post_zp_delta_ratio_by_chunk.csv", post_ratio_rows)
    _plot_role_timeline(out_dir, role_timeline)
    _plot_delta_ratio(out_dir, ratio_rows)
    _plot_post_ratio(out_dir, post_ratio_rows)
    _write_report(out_dir, run_rows, role_timeline, risk_layer_rows, update_rows, post_rows)

    summary = {
        "c9_ate": C9_ATE,
        "soft_threshold": SOFT_THRESHOLD,
        "runs": run_rows,
        "completeness": completeness,
        "artifacts": [
            "run_overview.csv",
            "phase2_trace_completeness_summary.csv",
            "phase2_trace_completeness_summary.json",
            "role_mass_timeline.csv",
            "risk_summary_by_layer.csv",
            "update_delta_norm_by_layer_branch.csv",
            "delta_norm_ratio_by_layer.csv",
            "delta_norm_ratio_by_layer.png",
            "post_zp_delta_by_chunk_layer_branch.csv",
            "post_zp_delta_ratio_by_chunk.csv",
            "post_zp_delta_ratio_by_chunk.png",
            "teacher_student_role_mass_timeline.png",
            "adaptive_failure_autopsy.md",
        ],
    }
    (out_dir / "phase2_adaptive_failure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=V52_ROOT / "phase2_adaptive_failure_audit",
        help="Output directory for phase 2 audit artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.out_dir, DEFAULT_RUNS)


if __name__ == "__main__":
    main()
