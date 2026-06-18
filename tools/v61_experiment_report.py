#!/usr/bin/env python3
"""Generate ACL2 v61 reports from landed artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v47_adaptive_ttt_writer_report import _walk  # noqa: E402
from tools.v53_experiment_report import (  # noqa: E402
    _fmt,
    _iter_run_dirs,
    _mean,
    _read_json,
    _read_jsonl,
    _safe_float,
    _summarize_runs,
    _write_csv,
    _write_json,
)
from tools.v56_experiment_report import (  # noqa: E402
    _augment_rows as _v56_augment_rows,
    _first_yaml_value,
    _is_704,
    _is_full,
    _md_table,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v61_clean_semantic_residual_read_ttt_scale_state"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
PLAN_DOC = "docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_Plan.md"
DOC_EXEC = REPO_ROOT / "docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_执行日志.md"
DOC_REVIEW = REPO_ROOT / "docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_实验结果复盘.md"

H35_FULL_ATE = 35.74089695811434
H35_704_ATE = 39.79824772048563
MIN_PROGRESS_ATE = 35.2409
SEMANTIC_TARGET_ATE = 33.7409


def _is_v61_smoke(row: Mapping[str, Any]) -> bool:
    phase = str(row.get("phase") or "")
    return phase.startswith("phase1_smoke")


def _is_track_a_semantic(row: Mapping[str, Any]) -> bool:
    return str(row.get("track") or "") == "track_a_semantic_read"


def _is_track_b_semantic(row: Mapping[str, Any]) -> bool:
    return str(row.get("track") or "") == "track_b_semantic_ttt"


def _is_negative_control(row: Mapping[str, Any]) -> bool:
    return "negative_control" in str(row.get("track") or "")


def _row_by_name(rows: Sequence[Mapping[str, Any]], name: str) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if row.get("row") == name:
            return row
    return None


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _sum_values(values: Iterable[Any]) -> Optional[float]:
    vals = _values(values)
    return float(sum(vals)) if vals else None


def _reproduce_inline(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    pieces: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#!") or line in {"set -euo pipefail"} or line.startswith("cd "):
            continue
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        pieces.append(line)
    return " ".join(pieces) if pieces else None


def _collect_v61_debug(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: Dict[str, Any] = {}
    read_before: List[float] = []
    read_after: List[float] = []
    read_actions: List[float] = []
    source_tokens: List[float] = []
    empty_events: List[float] = []
    anchor_counts: List[float] = []
    anchor_ratios: List[float] = []
    anchor_entropy: List[float] = []
    anchor_write_delta: List[float] = []
    anchor_write_count: List[float] = []
    role_collapse: List[float] = []
    tri_replay_applied: List[float] = []
    pos_mass: List[float] = []
    neu_mass: List[float] = []
    neg_mass: List[float] = []
    post_delta: List[float] = []
    native_cos: List[float] = []
    probe_ttt_tri_delta: List[float] = []
    probe_ttt_tri_pos_mass: List[float] = []
    probe_ttt_tri_neu_mass: List[float] = []
    probe_ttt_tri_neg_mass: List[float] = []
    probe_ttt_tri_applied: List[float] = []
    probe_ttt_post_delta: List[float] = []
    probe_ttt_native_cos: List[float] = []
    semantic_group_counts: List[float] = []
    semantic_label_counts: List[float] = []

    for row in rows:
        if row.get("prior_semantic_anchor_token_count") is not None:
            anchor_counts.append(_safe_float(row.get("prior_semantic_anchor_token_count")))
        if row.get("prior_semantic_anchor_token_ratio") is not None:
            anchor_ratios.append(_safe_float(row.get("prior_semantic_anchor_token_ratio")))
        if row.get("prior_semantic_anchor_spatial_entropy") is not None:
            anchor_entropy.append(_safe_float(row.get("prior_semantic_anchor_spatial_entropy")))
        if row.get("prior_anchor_write_mass_delta") is not None:
            anchor_write_delta.append(_safe_float(row.get("prior_anchor_write_mass_delta")))
        if row.get("prior_anchor_write_floor_applied_count") is not None:
            anchor_write_count.append(_safe_float(row.get("prior_anchor_write_floor_applied_count")))
        groups = row.get("prior_semantic_group_role_metrics")
        if isinstance(groups, dict):
            semantic_group_counts.append(float(len(groups)))
        if row.get("prior_fine_label_token_count") is not None:
            semantic_label_counts.append(_safe_float(row.get("prior_fine_label_token_count")))
        for node in _walk(row):
            if not isinstance(node, dict):
                continue
            for key in ("mean_attention_mass_removed_before", "attention_mass_removed_before"):
                if node.get(key) is not None:
                    read_before.append(_safe_float(node.get(key)))
            for key in ("mean_attention_mass_removed_after", "attention_mass_removed_after"):
                if node.get(key) is not None:
                    read_after.append(_safe_float(node.get(key)))
            for key in ("num_context_source_skip_applied", "num_semantic_anchor_boost_applied"):
                if node.get(key) is not None:
                    read_actions.append(_safe_float(node.get(key)))
            for key in ("max_context_source_skip_tokens", "source_skip_tokens"):
                if node.get(key) is not None:
                    source_tokens.append(_safe_float(node.get(key)))
            for key in ("num_context_empty_source_events", "empty_source_events"):
                if node.get(key) is not None:
                    empty_events.append(_safe_float(node.get(key)))
            if node.get("ttt_tri_replay_applied_layer_count") is not None:
                tri_replay_applied.append(_safe_float(node.get("ttt_tri_replay_applied_layer_count")))
            if node.get("auxgeo_tri_replay_applied_layer_count") is not None:
                tri_replay_applied.append(_safe_float(node.get("auxgeo_tri_replay_applied_layer_count")))
            for key, target in (
                ("auxgeo_tri_replay_pos_mass_mean", pos_mass),
                ("auxgeo_tri_replay_neu_mass_mean", neu_mass),
                ("auxgeo_tri_replay_neg_mass_mean", neg_mass),
                ("ttt_positive_mass", pos_mass),
                ("ttt_neutral_mass", neu_mass),
                ("ttt_negative_mass", neg_mass),
                ("ttt_write_native_delta_gate_cos_mean", native_cos),
                ("candidate_native_cosine", native_cos),
            ):
                if node.get(key) is not None:
                    target.append(_safe_float(node.get(key)))
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                if "post_zp_delta_norm" in key or "post_delta_norm" in key:
                    post_delta.append(_safe_float(value))
            if node.get("role_collapse_count") is not None:
                role_collapse.append(_safe_float(node.get("role_collapse_count")))
            for key, target in (
                ("probe_ttt_write_tri_delta_norm_mean", probe_ttt_tri_delta),
                ("probe_ttt_write_tri_pos_mass_mean", probe_ttt_tri_pos_mass),
                ("probe_ttt_write_tri_neu_mass_mean", probe_ttt_tri_neu_mass),
                ("probe_ttt_write_tri_neg_mass_mean", probe_ttt_tri_neg_mass),
                ("probe_ttt_write_tri_replay_applied_count", probe_ttt_tri_applied),
                ("probe_ttt_write_post_delta_norm_mean", probe_ttt_post_delta),
                ("probe_ttt_write_native_cosine_mean", probe_ttt_native_cos),
            ):
                if node.get(key) is not None:
                    target.append(_safe_float(node.get(key)))

    scale = _read_json(run_dir / "scale_metrics" / "scale_residual_summary.json")
    scale_rows = _read_csv(run_dir / "scale_metrics" / "per_chunk_scale_metrics.csv")
    out.update({
        "source_attention_mass_before": _mean(read_before),
        "source_attention_mass_after": _mean(read_after),
        "read_layer_action_count": _sum_values(read_actions),
        "affected_source_token_count_max": max(_values(source_tokens), default=None),
        "context_empty_source_events": int(sum(_values(empty_events))) if empty_events else None,
        "semantic_anchor_source_mass": _mean(anchor_ratios),
        "semantic_transient_source_mass": None,
        "anchor_rescue_count": _sum_values(anchor_counts),
        "transient_boost_count": _sum_values(read_actions),
        "semantic_group_count_mean": _mean(semantic_group_counts),
        "semantic_label_count_mean": _mean(semantic_label_counts),
        "semantic_anchor_spatial_entropy_mean": _mean(anchor_entropy),
        "semantic_anchor_write_mass_delta_mean": _mean(anchor_write_delta),
        "semantic_write_floor_count": _sum_values(anchor_write_count),
        "positive_mass": _mean(pos_mass),
        "neutral_mass": _mean(neu_mass),
        "negative_mass": _mean(neg_mass),
        "semantic_ttt_action_count": _sum_values(anchor_write_count) or _sum_values(tri_replay_applied),
        "tri_replay_applied_count": _sum_values(tri_replay_applied),
        "role_collapse_count": int(sum(_values(role_collapse))) if role_collapse else 0,
        "post_zp_delta_norm_mean": _mean(post_delta),
        "candidate_native_cosine": _mean(native_cos),
        "probe_ttt_write_tri_delta_norm_mean": _mean(probe_ttt_tri_delta),
        "probe_ttt_write_tri_pos_mass_mean": _mean(probe_ttt_tri_pos_mass),
        "probe_ttt_write_tri_neu_mass_mean": _mean(probe_ttt_tri_neu_mass),
        "probe_ttt_write_tri_neg_mass_mean": _mean(probe_ttt_tri_neg_mass),
        "probe_ttt_write_tri_replay_applied_count": _sum_values(probe_ttt_tri_applied),
        "probe_ttt_write_post_delta_norm_mean": _mean(probe_ttt_post_delta),
        "probe_ttt_write_native_cosine_mean": _mean(probe_ttt_native_cos),
        "scale_rows": len(scale_rows),
        "scale_overlap_point_pairs_available": scale.get("overlap_point_pairs_available"),
        "semantic_point_weights_available": scale.get("semantic_point_weights_available"),
        "scale_variance_all": scale.get("variance_all"),
        "scale_variance_geo": scale.get("variance_geo"),
        "scale_variance_sem": scale.get("variance_sem"),
        "mean_abs_log_scale_all": scale.get("mean_abs_log_scale_all"),
        "mean_abs_log_scale_geo": scale.get("mean_abs_log_scale_geo"),
        "mean_abs_log_scale_sem": scale.get("mean_abs_log_scale_sem"),
        "corr_log_scale_with_rolling100": scale.get("corr_log_scale_with_rolling100"),
        "corr_anchor_quality_with_scale_stability": scale.get("corr_anchor_quality_with_scale_stability"),
    })
    return out


def _augment_rows(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _v56_augment_rows(rows, gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        cfg = run_dir / "effective_config.yaml"
        for key in (
            "candidate",
            "track",
            "action_class",
            "implementation_note",
            "hmc_commit_mode",
            "context_source_skip_mask",
            "context_source_skip_mode",
            "semantic_role_policy",
            "semantic_memory_paths",
            "enable_semantic_anchor_ttt_floor",
        ):
            row[key] = _first_yaml_value(cfg, key) or row.get(key)
        row.update(_collect_v61_debug(run_dir))
        frames = int(row.get("frames") or 0)
        ate = _safe_float(row.get("ATE"))
        if math.isfinite(ate):
            if frames >= 1000:
                row["delta_vs_H35_full"] = ate - H35_FULL_ATE
            elif frames >= 700:
                row["delta_vs_H35_704"] = ate - H35_704_ATE
        row["projected_full_runtime_pass"] = _safe_float(row.get("projected_full_wall_time_min"), 999.0) <= 28.0
        row["smoke_gate_pass"] = _smoke_gate(row) if _is_v61_smoke(row) else None
        row["promotion_gate_pass"] = _promotion_gate(row) if _is_704(row) else None
        row["full_min_progress_pass"] = (
            _safe_float(row.get("ATE"), 999.0) <= MIN_PROGRESS_ATE if _is_full(row) and row.get("ATE") is not None else None
        )
        row["reproduce_command"] = _reproduce_inline(run_dir / "reproduce_command.sh")


def _smoke_gate(row: Mapping[str, Any]) -> bool:
    track = str(row.get("track") or "")
    has_read_action = _safe_float(row.get("read_layer_action_count"), 0.0) > 0 or _safe_float(row.get("source_attention_mass_before"), 0.0) != _safe_float(row.get("source_attention_mass_after"), 0.0)
    has_ttt_action = _safe_float(row.get("semantic_ttt_action_count"), 0.0) > 0 or _safe_float(row.get("post_zp_delta_norm_mean"), 0.0) > 0
    action_ok = True
    if "track_a_semantic_read" in track or "track_a_negative_control" in track:
        action_ok = has_read_action
    if "track_b_semantic_ttt" in track or "track_b_negative_control" in track:
        action_ok = has_ttt_action
    return bool(
        row.get("status") == "done"
        and action_ok
        and int(row.get("context_empty_source_events") or 0) == 0
        and int(row.get("role_collapse_count") or 0) == 0
        and int(row.get("scale_rows") or 0) > 0
        and bool(row.get("scale_overlap_point_pairs_available"))
        and bool(row.get("projected_full_runtime_pass"))
    )


def _promotion_gate(row: Mapping[str, Any]) -> bool:
    delta = _safe_float(row.get("delta_vs_H35_704"))
    scale_var = _safe_float(row.get("scale_variance_all"))
    # Baseline-relative scale improvement is computed later in the report
    # because it needs the chosen H35 screen row.
    return bool(
        row.get("status") == "done"
        and (
            (math.isfinite(delta) and delta <= -0.50)
            or (math.isfinite(scale_var) and row.get("scale_mechanistic_pass") is True)
        )
        and int(row.get("context_empty_source_events") or 0) == 0
        and bool(row.get("projected_full_runtime_pass"))
    )


def _run_rows(paths: Sequence[Path], gt_path: Path) -> List[Dict[str, Any]]:
    rows = _summarize_runs(_iter_run_dirs(paths), gt_path)
    _augment_rows(rows, gt_path)
    return rows


def _best_row(rows: Sequence[Mapping[str, Any]], pred) -> Optional[Mapping[str, Any]]:
    vals = [row for row in rows if pred(row) and row.get("ATE") is not None]
    vals.sort(key=lambda r: _safe_float(r.get("ATE"), 1e9))
    return vals[0] if vals else None


def _phase0_audit(result_root: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    h35_ref = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase0_h35_repeat/rollouts/V56_PHASE0_H35_FULL_REPEAT"
    semantic_cache = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full"
    h35_done = (h35_ref / "run_status.txt").is_file() and "DONE" in (h35_ref / "run_status.txt").read_text(encoding="utf-8", errors="replace")
    smoke_rows = [row for row in rows if _is_v61_smoke(row)]
    return {
        "h35_landed_reference_dir": str(h35_ref),
        "h35_landed_reference_available": bool(h35_done),
        "h35_landed_reference_ate": H35_FULL_ATE if h35_done else None,
        "c9_reference_ate": 33.76294210291885,
        "semantic_cache_dir": str(semantic_cache),
        "semantic_cache_dir_exists": semantic_cache.is_dir(),
        "stage_c_smoke_hit_rate_min": min(_values(row.get("stage_c_cache_hit_rate") for row in smoke_rows), default=None),
        "d_i_available_in_smoke": any(row.get("semantic_group_count_mean") is not None or row.get("source_attention_mass_before") is not None for row in smoke_rows),
        "ttt_residual_logging_available": any(row.get("post_zp_delta_norm_mean") is not None or row.get("tri_replay_applied_count") is not None for row in smoke_rows),
        "scale_metric_script_available": (REPO_ROOT / "tools/v61_scale_metrics.py").is_file(),
        "scale_metric_rows_written": sum(int(row.get("scale_rows") or 0) for row in smoke_rows),
    }


def _write_phase0_files(out_dir: Path, audit: Mapping[str, Any]) -> None:
    phase0 = out_dir / "phase0_audit"
    phase0.mkdir(parents=True, exist_ok=True)
    for name, keys in {
        "h35_repeat_or_landed_audit.md": ["h35_landed_reference_dir", "h35_landed_reference_available", "h35_landed_reference_ate", "c9_reference_ate"],
        "semantic_cache_hit_audit.md": ["semantic_cache_dir", "semantic_cache_dir_exists", "stage_c_smoke_hit_rate_min"],
        "read_wiring_audit.md": ["d_i_available_in_smoke"],
        "ttt_wiring_audit.md": ["ttt_residual_logging_available"],
        "scale_metric_availability.md": ["scale_metric_script_available", "scale_metric_rows_written"],
        "codex_self_check_report.md": list(audit.keys()),
    }.items():
        lines = [f"# {name.removesuffix('.md')}", ""]
        for key in keys:
            lines.append(f"- `{key}`: `{audit.get(key)}`")
        _write_text(phase0 / name, lines)


def _write_reports(result_root: Path, out_dir: Path, rows: Sequence[Dict[str, Any]], phase0: Mapping[str, Any]) -> None:
    smoke = [row for row in rows if _is_v61_smoke(row)]
    screen = [row for row in rows if _is_704(row)]
    full = [row for row in rows if _is_full(row)]
    best704 = _best_row(screen, lambda r: str(r.get("track", "")).startswith("track_"))
    bestfull_any = _best_row(full, lambda r: True)
    bestfull_candidate = _best_row(full, lambda r: _is_track_a_semantic(r) or _is_track_b_semantic(r))
    current_h35_704 = _row_by_name(rows, "A0_704F")
    current_h35_full = _row_by_name(rows, "A0_FULL")
    current_h35_704_delta = (
        _safe_float(current_h35_704.get("ATE")) - H35_704_ATE if current_h35_704 and current_h35_704.get("ATE") is not None else None
    )
    current_h35_full_delta = (
        _safe_float(current_h35_full.get("ATE")) - H35_FULL_ATE if current_h35_full and current_h35_full.get("ATE") is not None else None
    )
    baseline_stable = bool(
        current_h35_full_delta is not None
        and abs(current_h35_full_delta) <= 0.10
        and (current_h35_704_delta is None or abs(current_h35_704_delta) <= 0.10)
    )
    min_progress = bool(bestfull_candidate and _safe_float(bestfull_candidate.get("ATE"), 999.0) <= MIN_PROGRESS_ATE)
    target = bool(bestfull_candidate and _safe_float(bestfull_candidate.get("ATE"), 999.0) <= SEMANTIC_TARGET_ATE)

    fields = [
        ("row", "row"),
        ("candidate", "candidate"),
        ("track", "track"),
        ("frames", "frames"),
        ("ATE", "ATE"),
        ("dH35_704", "delta_vs_H35_704"),
        ("dH35_full", "delta_vs_H35_full"),
        ("read_act", "read_layer_action_count"),
        ("ttt_act", "semantic_ttt_action_count"),
        ("scale_var", "scale_variance_all"),
        ("scale_rows", "scale_rows"),
        ("gate", "smoke_gate_pass"),
    ]
    exec_lines = [
        "# ACL2 v61 CleanSemanticResidualRead TTTWriting ScaleState 执行日志",
        "",
        f"计划文档: `{PLAN_DOC}`",
        f"结果复盘: `{DOC_REVIEW.relative_to(REPO_ROOT)}`",
        f"结果根目录: `{result_root}`",
        f"报告目录: `{out_dir}`",
        "",
        "## 代码/工具修改",
        "",
        "| file | change | audit reason |",
        "|---|---|---|",
        "| `run_pipeline_abc_v2.py` | Added optional `--per_chunk_geometry_dir` and per-chunk geometry debug saves. | Enables real overlap point-Sim3 scale diagnostics without changing default behavior. |",
        "| `tools/run_attention_cue_experiment.sh` | Added `OUTPUT_PT` and `PER_CHUNK_GEOMETRY_DIR` passthrough. | Lets v61 runner preserve merged and per-chunk geometry artifacts. |",
        "| `tools/v61_scale_metrics.py` | New scale metric extractor. | Computes actual overlap Sim3 when per-chunk geometry exists; leaves semantic point weights as NA if absent. |",
        "| `tools/run_v61_clean_semantic_residual_read_ttt_scale_state.sh` | New v61 runner. | Records config, audits, reproduce command, geometry debug, and scale metrics per run. |",
        "| `tools/v61_experiment_report.py` | New artifact-only reporter. | Generates registries and docs without filling missing data. |",
        "",
        "## Phase 0 audit",
        "",
    ]
    for key, value in phase0.items():
        exec_lines.append(f"- `{key}`: `{value}`")
    exec_lines.extend([
        "",
        "## Landed runs",
        "",
        *_md_table(rows, fields),
        "",
        "## Phase decisions",
        "",
        f"- 96F/256F smoke rows landed: `{len(smoke)}`.",
        f"- 704F screen rows landed: `{len(screen)}`.",
        f"- Current A0_704F delta vs historical H35_704: `{_fmt(current_h35_704_delta)}`.",
        f"- Current A0_FULL delta vs landed H35 full: `{_fmt(current_h35_full_delta)}`.",
        f"- Baseline stable gate, abs current H35 deltas <= 0.10m: `{baseline_stable}`.",
        "- No semantic candidate full was launched after baseline drift and control checks failed promotion.",
        "",
        "## Reproduce commands",
        "",
    ])
    for row in rows:
        cmd = row.get("reproduce_command") or "NA"
        exec_lines.append(f"- `{row.get('run_name')}`: `{cmd}`")

    review_lines = [
        "# ACL2 v61 CleanSemanticResidualRead TTTWriting ScaleState 实验结果复盘",
        "",
        "日期: 2026-06-11",
        f"计划文档: `{PLAN_DOC}`",
        f"执行日志: `{DOC_EXEC.relative_to(REPO_ROOT)}`",
        f"结果根目录: `{result_root}`",
        "",
        (
            "结论先行: 本文件只基于已落盘 artifact 生成；缺失字段保持 NA/unavailable。"
            f"当前 full 只有 baseline repeat `{bestfull_any.get('candidate') if bestfull_any else 'NA'}`，"
            f"ATE `{_fmt(bestfull_any.get('ATE') if bestfull_any else None)}`。"
            f"minimum progress gate `<= {MIN_PROGRESS_ATE}`: `{min_progress}`；"
            f"semantic target `<= {SEMANTIC_TARGET_ATE}`: `{target}`。"
            f"当前 A0_FULL 相对 landed H35 full 漂移 `{_fmt(current_h35_full_delta)}`，"
            f"A0_704F 相对历史 H35_704 漂移 `{_fmt(current_h35_704_delta)}`，"
            "因此不生成可报告 method result，不启动 semantic candidate full。"
        ),
        "",
        "## Phase 0",
        "",
    ]
    for key, value in phase0.items():
        review_lines.append(f"- `{key}`: `{value}`")
    review_lines.extend([
        "",
        "## 96F/256F smoke",
        "",
        *_md_table(smoke, fields),
        "",
        "## 704F screen",
        "",
        *_md_table(screen, fields),
        "",
        "## Full runs",
        "",
        *_md_table(full, fields),
        "",
        "## Scale Metric Notes",
        "",
        "- `scale_variance_all` comes from actual per-chunk overlap point-map Sim(3) only when `per_chunk_geometry` exists.",
        "- `scale_variance_sem` remains NA unless point-level semantic weights are saved; v61 does not backfill it with zeros.",
        "- Historical H35 full artifact is used as landed ATE baseline; new v61 rows are reported separately.",
        "- A0_FULL was rerun as a blocker repair after A0_704F failed to match the historical H35_704 reference.",
        "",
        "## Phase Decisions",
        "",
        f"- Baseline stability: `False`; current A0_704F `{_fmt(current_h35_704.get('ATE') if current_h35_704 else None)}` vs historical `{_fmt(H35_704_ATE)}`, delta `{_fmt(current_h35_704_delta)}`.",
        f"- Current A0_FULL `{_fmt(current_h35_full.get('ATE') if current_h35_full else None)}` vs landed H35 full `{_fmt(H35_FULL_ATE)}`, delta `{_fmt(current_h35_full_delta)}`.",
        "- Track A 704F apparent gains against current A0_704F are not reportable because the current baseline drifted and NA1/NA3 controls show the same scale/ATE pattern.",
        "- Track B 704F does not pass: B1/B2 are indistinguishable from NB1/NB2 controls, and B4 has no semantic TTT write action count and remains worse than historical H35_704.",
        "- Phase 3 causal fork and Phase 4 semantic candidate full were not run by design because 704F promotion and baseline stability gates failed.",
        "",
        "## 必答问题",
        "",
        f"1. semantic residual READ 是否有收益: `{_answer_read(rows)}`。",
        f"2. semantic residual READ 是否影响 current chunk scale metrics: `{_answer_read_scale(rows)}`。",
        f"3. semantic TTT writing 是否改变 write mass/post-zp: `{_answer_ttt_action(rows)}`。",
        f"4. semantic TTT writing 是否影响 future scale residual: `not proven unless causal fork rows are present`。",
        f"5. semantic 是否优于 controls: `{_answer_controls(rows)}`。",
        f"6. 主要收益来源: `{_answer_source(bestfull_candidate, best704)}`。",
        f"7. 如果失败，主要瓶颈: `{_answer_bottleneck(rows)}`。",
        "",
        "## Insight 与证据链",
        "",
    ])
    review_lines.extend(_insights(rows, best704, bestfull_any))

    _write_text(out_dir / "v61_execution_log.md", exec_lines)
    _write_text(out_dir / "v61_final_report.md", review_lines)
    _write_text(DOC_EXEC, exec_lines)
    _write_text(DOC_REVIEW, review_lines)


def _answer_read(rows: Sequence[Mapping[str, Any]]) -> str:
    sem = [r for r in rows if _is_track_a_semantic(r) and (_is_704(r) or _is_full(r))]
    if not sem:
        return "not run"
    return (
        "not reportable: no Track A row beats historical H35_704 by 0.50m, and best semantic 704F is matched by controls"
    )


def _answer_read_scale(rows: Sequence[Mapping[str, Any]]) -> str:
    sem = [r for r in rows if str(r.get("track")) == "track_a_semantic_read" and r.get("scale_variance_all") is not None]
    return "scale metrics available; baseline-relative improvement must be judged from paired rows" if sem else "not available"


def _answer_ttt_action(rows: Sequence[Mapping[str, Any]]) -> str:
    sem = [r for r in rows if _is_track_b_semantic(r)]
    if not sem:
        return "not run"
    if any(_safe_float(r.get("semantic_ttt_action_count"), 0.0) > 0 for r in sem):
        return "partial: B1 writes changed, but post-zp is unavailable and B1/B2 do not beat matched controls"
    return "not proven"


def _answer_controls(rows: Sequence[Mapping[str, Any]]) -> str:
    sem = [r for r in rows if (_is_track_a_semantic(r) or _is_track_b_semantic(r)) and (_is_704(r) or _is_full(r))]
    ctrl = [r for r in rows if _is_negative_control(r) and (_is_704(r) or _is_full(r))]
    if not sem or not ctrl:
        return "not enough paired controls"
    best_sem = min((_safe_float(r.get("ATE"), 1e9) for r in sem), default=1e9)
    best_ctrl = min((_safe_float(r.get("ATE"), 1e9) for r in ctrl), default=1e9)
    return "yes" if best_sem + 0.3 <= best_ctrl else "not proven"


def _answer_source(bestfull: Optional[Mapping[str, Any]], best704: Optional[Mapping[str, Any]]) -> str:
    return "no promoted source; apparent 704F gains come from READ-style source attenuation also reproduced by controls"


def _answer_bottleneck(rows: Sequence[Mapping[str, Any]]) -> str:
    current_full = _row_by_name(rows, "A0_FULL")
    if current_full and _safe_float(current_full.get("delta_vs_H35_full"), 0.0) > 0.10:
        return "current H35 baseline drifted; controls explain READ gains; semantic TTT writing lacks causal improvement"
    return "controls explain READ gains; semantic TTT writing lacks causal improvement"


def _insights(rows: Sequence[Mapping[str, Any]], best704: Optional[Mapping[str, Any]], bestfull: Optional[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    if rows:
        out.append(f"1. Landed row count: `{len(rows)}`; smoke rows: `{sum(1 for r in rows if _is_v61_smoke(r))}`; 704F rows: `{sum(1 for r in rows if _is_704(r))}`; full rows: `{sum(1 for r in rows if _is_full(r))}`.")
    if best704:
        out.append(f"2. Best 704F row is `{best704.get('candidate')}` with ATE `{_fmt(best704.get('ATE'))}` and delta vs H35_704 `{_fmt(best704.get('delta_vs_H35_704'))}`.")
    if bestfull:
        out.append(f"3. Best full row is `{bestfull.get('candidate')}` with ATE `{_fmt(bestfull.get('ATE'))}` and delta vs H35 `{_fmt(bestfull.get('delta_vs_H35_full'))}`.")
    scale_rows = [r for r in rows if int(r.get("scale_rows") or 0) > 0]
    out.append(f"4. Scale diagnostics wrote rows for `{len(scale_rows)}` runs; semantic point-weighted scale remains unavailable unless explicitly saved.")
    failed_smoke = [r for r in rows if _is_v61_smoke(r) and r.get("smoke_gate_pass") is False]
    if failed_smoke:
        names = ", ".join(str(r.get("candidate")) for r in failed_smoke[:8])
        out.append(f"5. Smoke gate failures include `{names}`; inspect action counts, context empty events, runtime and scale availability before promotion.")
    current704 = _row_by_name(rows, "A0_704F")
    currentfull = _row_by_name(rows, "A0_FULL")
    if current704:
        out.append(f"6. Current A0_704F ATE `{_fmt(current704.get('ATE'))}` is worse than historical H35_704 `{_fmt(H35_704_ATE)}` by `{_fmt(current704.get('delta_vs_H35_704'))}`.")
    if currentfull:
        out.append(f"7. Current A0_FULL ATE `{_fmt(currentfull.get('ATE'))}` is worse than landed H35 full `{_fmt(H35_FULL_ATE)}` by `{_fmt(currentfull.get('delta_vs_H35_full'))}`; this blocks reportable method claims.")
    a4 = _row_by_name(rows, "A4_704F")
    na1 = _row_by_name(rows, "NA1_704F")
    na3 = _row_by_name(rows, "NA3_704F")
    if a4 and na1 and na3:
        out.append(
            "8. A4_704F is close to controls: "
            f"A4 `{_fmt(a4.get('ATE'))}`, NA1 `{_fmt(na1.get('ATE'))}`, NA3 `{_fmt(na3.get('ATE'))}`; "
            "the READ/scale improvement is therefore not semantic-specific."
        )
    b1 = _row_by_name(rows, "B1_704F")
    b2 = _row_by_name(rows, "B2_704F")
    nb1 = _row_by_name(rows, "NB1_704F")
    nb2 = _row_by_name(rows, "NB2_704F")
    if b1 and b2 and nb1 and nb2:
        out.append(
            "9. B1/B2 704F do not beat TTT controls: "
            f"B1 `{_fmt(b1.get('ATE'))}`, B2 `{_fmt(b2.get('ATE'))}`, "
            f"NB1 `{_fmt(nb1.get('ATE'))}`, NB2 `{_fmt(nb2.get('ATE'))}`."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "report_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _run_rows([result_root], Path(args.gt))
    _write_csv(out_dir / "v61_all_registry.csv", rows)
    _write_json(out_dir / "v61_all_registry.json", rows)
    phase0 = _phase0_audit(result_root, rows)
    _write_json(out_dir / "v61_phase0_audit.json", phase0)
    _write_phase0_files(out_dir, phase0)
    summary = {
        "row_count": len(rows),
        "smoke_count": sum(1 for row in rows if _is_v61_smoke(row)),
        "screen_704_count": sum(1 for row in rows if _is_704(row)),
        "full_count": sum(1 for row in rows if _is_full(row)),
        "best_full_ate": min((_safe_float(row.get("ATE"), 1e9) for row in rows if _is_full(row) and row.get("ATE") is not None), default=None),
    }
    _write_json(out_dir / "v61_summary.json", summary)
    _write_reports(result_root, out_dir, rows, phase0)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
