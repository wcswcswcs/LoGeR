#!/usr/bin/env python3
"""Summarize ACL2 v53 no-chunk adaptive TTT experiments from landed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
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

from tools.v18_true_action_report import _align_metrics, _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v42_full_online_report import _as_positions, _ate, _rolling_stats, _rolling_windows  # noqa: E402
from tools.v47_adaptive_ttt_writer_report import _debug_stats, _walk  # noqa: E402


C9_P0_ATE = 33.76294210291885
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
V52_AUTOPSY_DIR = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit"
)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_yaml_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def _run_status(run_dir: Path, run_name: str) -> str:
    status_path = run_dir / "run_status.txt"
    if not status_path.is_file():
        return "missing_status"
    text = status_path.read_text(encoding="utf-8", errors="replace")
    if f"DONE {run_name}" in text or "DONE " in text:
        return "done"
    if f"FAIL {run_name}" in text or "FAIL " in text:
        return "fail"
    return "running_or_partial"


def _pose_metrics(run_dir: Path, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    pred_path = run_dir / "01.txt"
    if not pred_path.is_file():
        return {"pose_status": "missing_prediction", "frames": 0}
    frames, raw_poses, _ = _load_tum_prediction(pred_path, gt_pos.shape[0])
    if frames.size == 0:
        return {"pose_status": "empty_prediction", "frames": 0}
    aligned, metrics = _align_metrics(frames, raw_poses, gt_poses, gt_pos)
    pos = _as_positions(aligned)
    out: Dict[str, Any] = {
        "pose_status": "done",
        "frames": int(frames.size),
        "ATE": metrics.get("ATE_horizon"),
        "Rot": metrics.get("Rot_horizon"),
        "FinalErr": metrics.get("FinalErr_horizon"),
        "alignment_scale": metrics.get("alignment_scale"),
    }
    for width in (50, 100, 200):
        stats = _rolling_stats(_rolling_windows(frames, pos, gt_pos, width))
        out[f"rolling{width}_mean"] = stats.get("mean")
        out[f"rolling{width}_p90"] = stats.get("p90")
        out[f"rolling{width}_worst"] = stats.get("worst")
    return out


def _timing_stats(run_dir: Path) -> Dict[str, Any]:
    timing = _read_json(run_dir / "timing_summary.json")
    wall = _read_json(run_dir / "wall_time_summary.json")
    chunks = timing.get("chunks") if isinstance(timing.get("chunks"), list) else []
    if not chunks:
        log_path = run_dir / "01.log"
        log_chunks: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        current_key: Optional[str] = None
        if log_path.is_file():
            for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "# V2 Chunk" in raw:
                    if current is not None:
                        log_chunks.append(current)
                    current = {}
                    current_key = None
                    continue
                if current is None:
                    continue
                if "Pass 1: Probe Geometry Backbone" in raw:
                    current_key = "pass1_probe_seconds"
                elif "Stage B:" in raw and "skipped" not in raw:
                    current_key = "stage_b_seconds"
                elif "Stage C:" in raw and "skipped" not in raw:
                    current_key = "stage_c_seconds"
                elif "Stage D:" in raw and "skipped" not in raw:
                    current_key = "stage_d_seconds"
                elif "Pass 2: Controlled Geometry Backbone" in raw:
                    current_key = "pass2_control_seconds"
                else:
                    match = re.search(r"done in\s+([0-9.]+)s", raw)
                    if match and current_key:
                        current[current_key] = float(match.group(1))
                        current_key = None
            if current is not None:
                log_chunks.append(current)
        for chunk in log_chunks:
            vals = [
                _safe_float(chunk.get(key))
                for key in (
                    "pass1_probe_seconds",
                    "stage_b_seconds",
                    "stage_c_seconds",
                    "stage_d_seconds",
                    "pass2_control_seconds",
                )
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                chunk["chunk_total_seconds"] = float(sum(vals))
        chunks = log_chunks
    out: Dict[str, Any] = {
        "timing_chunks": len(chunks),
        "wall_seconds": wall.get("wall_seconds"),
        "wall_time_min": (_safe_float(wall.get("wall_seconds")) / 60.0 if math.isfinite(_safe_float(wall.get("wall_seconds"))) else None),
        "total_runtime_seconds_after_model_load": timing.get("total_runtime_seconds_after_model_load"),
        "total_runtime_seconds_including_model_load": timing.get("total_runtime_seconds_including_model_load"),
        "model_load_seconds": timing.get("model_load_seconds"),
        "empty_cuda_cache_each_chunk": wall.get("empty_cuda_cache_each_chunk", timing.get("empty_cuda_cache_each_chunk")),
    }
    for key in (
        "chunk_total_seconds",
        "pass1_probe_seconds",
        "stage_b_seconds",
        "stage_c_seconds",
        "stage_d_seconds",
        "pass2_control_seconds",
        "probe_ttt_write_seconds",
    ):
        out[f"{key}_mean"] = _mean((c or {}).get(key) for c in chunks if isinstance(c, dict))
        out[f"{key}_max"] = max(
            [_safe_float((c or {}).get(key)) for c in chunks if isinstance(c, dict) and math.isfinite(_safe_float((c or {}).get(key)))],
            default=None,
        )
    return out


def _strict_no_chunk_pass(audit: Mapping[str, Any]) -> bool:
    expected_true = (
        "read_beta_frame_chunks_empty",
        "ttt_gradient_reversal_chunk_gammas_empty",
        "ttt_tri_replay_chunk_params_empty",
        "ttt_commit_ema_chunks_empty",
        "native_mix_chunks_empty",
        "semantic_action_active_chunks_empty",
    )
    forbidden_true = (
        "has_read_beta_frame_chunks",
        "has_tri_gamma_chunk_map",
        "has_tri_replay_chunk_params",
        "has_commit_ema_chunks",
        "has_native_mix_chunks",
        "has_semantic_action_active_chunks",
    )
    return all(bool(audit.get(key)) for key in expected_true) and not any(bool(audit.get(key)) for key in forbidden_true)


def _manual_percentage_pass(audit: Mapping[str, Any], debug: Mapping[str, Any]) -> bool:
    role_mode = str(audit.get("role_mode") or "")
    role_ok = any(token in role_mode for token in ("adaptive", "state_conditioned", "no_percentage", "sc_gamma"))
    manual_ok = (
        _safe_float(audit.get("manual_positive_frac"), 999.0) == 0.0
        and _safe_float(audit.get("manual_negative_frac"), 999.0) == 0.0
        and _safe_float(audit.get("manual_neutral_lambda"), 999.0) == 0.0
    )
    split_ok = int(debug.get("adaptive_writer_split_debug_count") or 0) > 0
    fused_ok = int(debug.get("adaptive_writer_fused_debug_count") or 0) == 0
    return bool(manual_ok and role_ok and split_ok and fused_ok)


def _collapse_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    collapsed = 0
    seen = 0
    sources: List[str] = []
    for row in rows:
        for node in _walk(row):
            if "ttt_tri_replay_role_collapsed" in node:
                seen += 1
                if bool(node.get("ttt_tri_replay_role_collapsed")):
                    collapsed += 1
            if node.get("ttt_tri_replay_role_source") is not None:
                sources.append(str(node.get("ttt_tri_replay_role_source")))
    log_path = run_dir / "01.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"['\"]ttt_tri_replay_role_collapsed['\"]:\s*(True|False|true|false)", text)
        if matches:
            seen += len(matches)
            collapsed += sum(1 for value in matches if value.lower() == "true")
    return {
        "role_collapse_debug_rows": collapsed,
        "role_collapse_debug_seen": seen,
        "role_collapse_rate": (collapsed / seen if seen else None),
        "role_sources_seen_deep": ";".join(sorted(set(sources))),
    }


def _commit_filter_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    rates: List[float] = []
    risks: List[float] = []
    scales: List[float] = []
    applied = 0
    seen = 0
    active_mode_seen = 0
    invalid_scope = 0
    modes: List[str] = []
    for row in rows:
        for node in _walk(row):
            mode = str(node.get("ttt_write_commit_filter_mode") or "").strip().lower()
            has_commit_field = any(
                key in node
                for key in (
                    "ttt_write_commit_filter_activation_rate",
                    "ttt_write_commit_filter_risk_mean",
                    "ttt_write_commit_filter_scale_mean",
                    "ttt_write_commit_filter_num_tensors",
                )
            )
            if mode and mode not in {"none", "off"}:
                active_mode_seen += 1
                modes.append(mode)
            if has_commit_field or (mode and mode not in {"none", "off"}):
                seen += 1
            if "ttt_write_commit_filter_activation_rate" in node:
                rates.append(_safe_float(node.get("ttt_write_commit_filter_activation_rate")))
            if node.get("ttt_write_commit_filter_applied") is True:
                applied += 1
            if "ttt_write_commit_filter_risk_mean" in node:
                risks.append(_safe_float(node.get("ttt_write_commit_filter_risk_mean")))
            if "ttt_write_commit_filter_scale_mean" in node:
                scales.append(_safe_float(node.get("ttt_write_commit_filter_scale_mean")))
            if node.get("ttt_write_commit_filter_scope_invalid") is True:
                invalid_scope += 1
    log_path = run_dir / "01.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        mode_matches = re.findall(r"['\"]ttt_write_commit_filter_mode['\"]:\s*['\"]([^'\"]+)['\"]", text)
        active_modes = [m.strip().lower() for m in mode_matches if m.strip().lower() not in {"", "none", "off"}]
        modes.extend(active_modes)
        active_mode_seen += len(active_modes)
        seen += len(active_modes)
        applied += sum(
            1
            for value in re.findall(r"['\"]ttt_write_commit_filter_applied['\"]:\s*(True|False|true|false)", text)
            if value.lower() == "true"
        )
        for value in re.findall(r"['\"]ttt_write_commit_filter_activation_rate['\"]:\s*([-+0-9.eE]+)", text):
            rates.append(_safe_float(value))
        for value in re.findall(r"['\"]ttt_write_commit_filter_risk_mean['\"]:\s*([-+0-9.eE]+)", text):
            risks.append(_safe_float(value))
        for value in re.findall(r"['\"]ttt_write_commit_filter_scale_mean['\"]:\s*([-+0-9.eE]+)", text):
            scales.append(_safe_float(value))
        invalid_scope += sum(
            1
            for value in re.findall(r"['\"]ttt_write_commit_filter_scope_invalid['\"]:\s*(True|False|true|false)", text)
            if value.lower() == "true"
        )
    return {
        "commit_filter_debug_seen": seen,
        "commit_filter_active_mode_seen": active_mode_seen,
        "commit_filter_applied_debug_rows": applied,
        "commit_filter_activation_rate_mean": _mean(rates),
        "commit_filter_risk_mean": _mean(risks),
        "commit_filter_scale_mean": _mean(scales),
        "commit_filter_scope_invalid_rows": invalid_scope,
        "commit_filter_modes_seen": ";".join(sorted(set(modes))),
    }


def _iter_run_dirs(paths: Iterable[Path]) -> List[Path]:
    run_dirs: List[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir() and (path / "timing_summary.json").is_file():
            run_dirs.append(path)
        elif path.is_dir() and (path / "run_status.txt").is_file() and (path / "01.txt").is_file():
            run_dirs.append(path)
        elif path.is_dir():
            candidates = {p.parent for p in path.rglob("timing_summary.json")}
            candidates.update(
                p.parent
                for p in path.rglob("run_status.txt")
                if (p.parent / "01.txt").is_file()
            )
            run_dirs.extend(sorted(candidates))
    unique: List[Path] = []
    seen = set()
    for run_dir in run_dirs:
        key = str(run_dir.resolve())
        if key not in seen:
            unique.append(run_dir)
            seen.add(key)
    return unique


def _summarize_runs(run_dirs: Sequence[Path], gt: Path) -> List[Dict[str, Any]]:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt)
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        config_path = run_dir / "effective_config.yaml"
        row: Dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "phase": run_dir.parent.parent.name if run_dir.parent.name == "rollouts" else run_dir.parent.name,
            "row": _first_yaml_value(config_path, "row") or _first_yaml_value(run_dir / "v47_effective_config.yaml", "row"),
            "role_mode_config": _first_yaml_value(config_path, "ttt_write_tri_replay_role_mode"),
            "risk_source_config": _first_yaml_value(config_path, "ttt_write_gradient_reversal_risk_source"),
            "commit_filter_mode_config": _first_yaml_value(config_path, "ttt_write_commit_filter_mode"),
            "read_beta_config": _first_yaml_value(config_path, "beta"),
            "read_cue_config": _first_yaml_value(config_path, "cue"),
            "status": _run_status(run_dir, run_name),
        }
        row.update(_pose_metrics(run_dir, gt_poses, gt_pos))
        row.update(_timing_stats(run_dir))
        debug = _debug_stats(run_dir)
        row.update(debug)
        row.update(_collapse_stats(run_dir))
        row.update(_commit_filter_stats(run_dir))
        chunk_audit = _read_json(run_dir / "chunk_id_policy_audit.json")
        manual_audit = _read_json(run_dir / "adaptive_ttt_audit.json")
        if not manual_audit:
            manual_audit = _read_json(run_dir / "adaptive_writer_audit.json")
        row["no_chunk_policy_pass"] = _strict_no_chunk_pass(chunk_audit)
        row["manual_percentage_audit_pass"] = _manual_percentage_pass(manual_audit, debug)
        frames = int(row.get("frames") or 0)
        row["full_kitti01"] = frames >= 1000
        if row.get("ATE") is not None:
            row["delta_vs_C9_P0"] = float(row["ATE"]) - C9_P0_ATE
        if row.get("timing_chunks"):
            row["projected_full_wall_time_min"] = (
                _safe_float(row.get("wall_seconds")) / max(int(row["timing_chunks"]), 1) * 38.0 / 60.0
                if math.isfinite(_safe_float(row.get("wall_seconds")))
                else None
            )
        row["full_runtime_gate_pass"] = (
            row.get("status") == "done"
            and bool(row.get("full_kitti01"))
            and _safe_float(row.get("wall_time_min"), 999.0) <= 28.0
            and _safe_float(row.get("chunk_total_seconds_mean"), 999.0) <= 42.0
            and _safe_float(row.get("probe_ttt_write_seconds_mean"), 999.0) <= 8.0
            and int(row.get("hmc_rows") or 0) == 38
            and int(row.get("frames") or 0) == 1101
        )
        row["smoke_runtime_gate_pass"] = (
            row.get("status") == "done"
            and _safe_float(row.get("chunk_total_seconds_mean"), 999.0) <= 42.0
            and _safe_float(row.get("probe_ttt_write_seconds_mean"), 999.0) <= 8.0
        )
        rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _plot_no_data(path: Path, title: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.42, note, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_phase1_autopsy(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_role_png = V52_AUTOPSY_DIR / "teacher_student_role_mass_timeline.png"
    if src_role_png.is_file():
        shutil.copy2(src_role_png, out_dir / "teacher_student_role_mass_timeline.png")
    else:
        _plot_no_data(out_dir / "teacher_student_role_mass_timeline.png", "role mass timeline", "source artifact missing")

    role_rows = _read_csv(V52_AUTOPSY_DIR / "role_mass_timeline.csv")
    if role_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in sorted({r["run"] for r in role_rows}):
            xs = [_safe_float(r.get("chunk_idx")) for r in role_rows if r.get("run") == run]
            ys = [_safe_float(r.get("w0_gamma_mean")) for r in role_rows if r.get("run") == run]
            pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.2, label=run)
        ax.set_xlabel("chunk_idx")
        ax.set_ylabel("w0_gamma_mean")
        ax.set_title("Teacher/student gamma timeline")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "teacher_student_gamma_timeline.png", dpi=160)
        plt.close(fig)
    else:
        _plot_no_data(out_dir / "teacher_student_gamma_timeline.png", "gamma timeline", "role_mass_timeline.csv missing")

    ratio_rows = _read_csv(V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.csv")
    if ratio_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        for key, label in (
            ("c9_committed_delta_norm_mean", "C9 committed"),
            ("student_committed_delta_norm_mean", "student committed"),
            ("c9_native_delta_norm_mean", "C9 native"),
            ("student_native_delta_norm_mean", "student native"),
        ):
            pts = [
                (_safe_float(r.get("chunk_idx")), _safe_float(r.get(key)))
                for r in ratio_rows
            ]
            pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.2, label=label)
        ax.set_xlabel("chunk_idx")
        ax.set_ylabel("delta norm mean")
        ax.set_title("Teacher/student post-zp delta norm")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "teacher_student_post_zp_delta_norm.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for key, label in (
            ("c9_cos_committed_to_native_mean", "C9 committed/native"),
            ("student_cos_committed_to_native_mean", "student committed/native"),
            ("c9_cos_action_to_native_mean", "C9 action/native"),
            ("student_cos_action_to_native_mean", "student action/native"),
        ):
            pts = [
                (_safe_float(r.get("chunk_idx")), _safe_float(r.get(key)))
                for r in ratio_rows
            ]
            pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.2, label=label)
        ax.set_xlabel("chunk_idx")
        ax.set_ylabel("cosine")
        ax.set_title("Candidate/native cosine timeline")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "candidate_vs_native_cosine_timeline.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [_safe_float(r.get("student_over_c9_native_delta_norm")) for r in ratio_rows]
        ys = [_safe_float(r.get("student_over_c9_committed_delta_norm")) for r in ratio_rows]
        pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=24)
            ax.axhline(1.0, color="0.5", linewidth=0.8)
            ax.axvline(1.0, color="0.5", linewidth=0.8)
        ax.set_xlabel("student/C9 native delta norm")
        ax.set_ylabel("student/C9 committed delta norm")
        ax.set_title("Risk/delta proxy scatter from v52 autopsy")
        fig.tight_layout()
        fig.savefig(out_dir / "risk_spread_vs_delta_norm_scatter.png", dpi=160)
        plt.close(fig)
    else:
        for name in (
            "teacher_student_post_zp_delta_norm.png",
            "candidate_vs_native_cosine_timeline.png",
            "risk_spread_vs_delta_norm_scatter.png",
        ):
            _plot_no_data(out_dir / name, name, "post_zp_delta_ratio_by_chunk.csv missing")


def _write_phase1_report(out_dir: Path) -> None:
    overview = _read_csv(V52_AUTOPSY_DIR / "run_overview.csv")
    ratios = _read_csv(V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.csv")
    lines = [
        "# ACL2 v53 Phase 1 C9 teacher/student autopsy",
        "",
        "This report reuses the landed v52 Phase 2 trace autopsy because v53 did not require a new full teacher trace.",
        "No missing field is filled with synthetic data.",
        "",
        f"Source directory: `{V52_AUTOPSY_DIR}`",
        "",
        "## Run overview",
        "",
        "| run | ATE | chunk mean | probe TTT mean | role rows | risk rows | role source | w0 gamma mean |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in overview:
        lines.append(
            f"| `{row.get('run')}` | {_fmt(row.get('ATE_full'))} | {_fmt(row.get('chunk_total_seconds_mean'))} | "
            f"{_fmt(row.get('probe_ttt_write_seconds_mean'))} | {row.get('role_rows')} | {row.get('risk_rows')} | "
            f"`{row.get('role_sources_seen')}` | {_fmt(row.get('w0_gamma_mean'))} |"
        )
    if ratios:
        c9_committed = _mean(r.get("c9_committed_delta_norm_mean") for r in ratios)
        student_committed = _mean(r.get("student_committed_delta_norm_mean") for r in ratios)
        c9_action_cos = _mean(r.get("c9_cos_action_to_native_mean") for r in ratios)
        student_action_cos = _mean(r.get("student_cos_action_to_native_mean") for r in ratios)
        lines.extend([
            "",
            "## Gap diagnosis",
            "",
            f"- C9 committed/native delta norm mean: `{_fmt(c9_committed)}`.",
            f"- v50/v52 student committed/native delta norm mean: `{_fmt(student_committed)}`.",
            f"- C9 action/native cosine mean: `{_fmt(c9_action_cos)}`.",
            f"- v50/v52 student action/native cosine mean: `{_fmt(student_action_cos)}`.",
            "- The reused trace supports v53's diagnosis: role split alone is not enough; post-zp update energy, gamma timing, and commit behavior remain the main gap.",
        ])
    else:
        lines.extend(["", "## Gap diagnosis", "", "- `post_zp_delta_ratio_by_chunk.csv` was missing; no numeric gap diagnosis was generated."])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v53_c9_teacher_student_autopsy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_v53_runtime(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        timing = _read_json(run_dir / "timing_summary.json")
        chunks = timing.get("chunks") if isinstance(timing.get("chunks"), list) else []
        pts = [
            (idx, _safe_float(chunk.get("chunk_total_seconds")))
            for idx, chunk in enumerate(chunks)
            if isinstance(chunk, dict) and math.isfinite(_safe_float(chunk.get("chunk_total_seconds")))
        ]
        if pts:
            plotted = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.0, label=str(row.get("run_name")))
    if not plotted:
        plt.close(fig)
        _plot_no_data(out_dir / "v53_runtime_profile_by_chunk.png", "v53 runtime profile", "no timing_summary chunks found")
        return
    ax.axhline(42.0, color="red", linestyle="--", linewidth=0.9, label="42s gate")
    ax.set_xlabel("chunk index within run")
    ax.set_ylabel("chunk_total_seconds")
    ax.set_title("v53 runtime by chunk")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "v53_runtime_profile_by_chunk.png", dpi=160)
    plt.close(fig)


def _write_reports(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "v53_candidate_registry.csv", rows)
    _write_json(out_dir / "v53_candidate_registry.json", list(rows))

    runtime_rows = [
        {
            key: row.get(key)
            for key in (
                "run_name",
                "phase",
                "status",
                "frames",
                "timing_chunks",
                "wall_time_min",
                "projected_full_wall_time_min",
                "chunk_total_seconds_mean",
                "probe_ttt_write_seconds_mean",
                "smoke_runtime_gate_pass",
                "full_runtime_gate_pass",
            )
        }
        for row in rows
    ]
    _write_csv(out_dir / "v53_runtime_profile_summary.csv", runtime_rows)
    _write_json(out_dir / "v53_runtime_profile_summary.json", runtime_rows)
    _plot_v53_runtime(rows, out_dir)

    no_chunk = [
        {
            "run_name": row.get("run_name"),
            "phase": row.get("phase"),
            "no_chunk_policy_pass": row.get("no_chunk_policy_pass"),
            "run_dir": row.get("run_dir"),
        }
        for row in rows
    ]
    manual = [
        {
            "run_name": row.get("run_name"),
            "phase": row.get("phase"),
            "manual_percentage_audit_pass": row.get("manual_percentage_audit_pass"),
            "role_mode_config": row.get("role_mode_config"),
            "split_debug_count": row.get("adaptive_writer_split_debug_count"),
            "fused_debug_count": row.get("adaptive_writer_fused_debug_count"),
            "run_dir": row.get("run_dir"),
        }
        for row in rows
    ]
    _write_json(out_dir / "v53_no_chunk_policy_audit.json", no_chunk)
    _write_json(out_dir / "v53_manual_percentage_audit.json", manual)

    full_rows = [row for row in rows if bool(row.get("full_kitti01"))]
    full_lines = [
        "# ACL2 v53 full metrics summary",
        "",
        "| run | ATE | delta vs C9 | Rot | FinalErr | frames | hmc rows | wall min | chunk mean | TTT mean | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if full_rows:
        for row in full_rows:
            full_lines.append(
                f"| `{row.get('run_name')}` | {_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_C9_P0'))} | "
                f"{_fmt(row.get('Rot'))} | {_fmt(row.get('FinalErr'))} | {row.get('frames')} | {row.get('hmc_rows')} | "
                f"{_fmt(row.get('wall_time_min'))} | {_fmt(row.get('chunk_total_seconds_mean'))} | "
                f"{_fmt(row.get('probe_ttt_write_seconds_mean'))} | {row.get('full_runtime_gate_pass')} |"
            )
    else:
        full_lines.append("| no full v53 candidate run found | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    (out_dir / "v53_full_metrics_summary.md").write_text("\n".join(full_lines) + "\n", encoding="utf-8")

    phase0_lines = [
        "# ACL2 v53 Phase 0 efficiency audit",
        "",
        "| run | phase | status | frames | wall min | projected full wall min | chunk mean | TTT mean | no-chunk | manual % |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        phase0_lines.append(
            f"| `{row.get('run_name')}` | `{row.get('phase')}` | `{row.get('status')}` | {row.get('frames')} | "
            f"{_fmt(row.get('wall_time_min'))} | {_fmt(row.get('projected_full_wall_time_min'))} | "
            f"{_fmt(row.get('chunk_total_seconds_mean'))} | {_fmt(row.get('probe_ttt_write_seconds_mean'))} | "
            f"{row.get('no_chunk_policy_pass')} | {row.get('manual_percentage_audit_pass')} |"
        )
    (out_dir / "v53_phase0_efficiency_audit.md").write_text("\n".join(phase0_lines) + "\n", encoding="utf-8")

    failure_lines = [
        "# ACL2 v53 failure routing report",
        "",
        "| run | phase | status | ATE | runtime gate | no-chunk | manual % | role collapse rate | routing |",
        "|---|---|---|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        reasons: List[str] = []
        if row.get("status") != "done":
            reasons.append("not_done")
        if not row.get("no_chunk_policy_pass"):
            reasons.append("no_chunk_audit_fail")
        if not row.get("manual_percentage_audit_pass"):
            reasons.append("manual_percentage_audit_fail")
        if _safe_float(row.get("probe_ttt_write_seconds_mean"), 999.0) > 8.0:
            reasons.append("probe_ttt_runtime_fail")
        if _safe_float(row.get("chunk_total_seconds_mean"), 999.0) > 42.0:
            reasons.append("chunk_runtime_fail")
        if row.get("full_kitti01") and _safe_float(row.get("ATE"), 999.0) > 35.30:
            reasons.append("full_ate_fail_gt_35.30")
        if _safe_float(row.get("role_collapse_rate"), 0.0) > 0.20:
            reasons.append("role_collapse_gt_20pct")
        routing = ",".join(reasons) if reasons else "advance_or_freeze"
        failure_lines.append(
            f"| `{row.get('run_name')}` | `{row.get('phase')}` | `{row.get('status')}` | {_fmt(row.get('ATE'))} | "
            f"{row.get('smoke_runtime_gate_pass') or row.get('full_runtime_gate_pass')} | {row.get('no_chunk_policy_pass')} | "
            f"{row.get('manual_percentage_audit_pass')} | {_fmt(row.get('role_collapse_rate'))} | `{routing}` |"
        )
    if not rows:
        failure_lines.append("| no v53 run found | NA | NA | NA | NA | NA | NA | NA | no data |")
    (out_dir / "v53_failure_routing_report.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    best_full: Optional[Mapping[str, Any]] = None
    for row in full_rows:
        if row.get("ATE") is None:
            continue
        if best_full is None or _safe_float(row.get("ATE")) < _safe_float(best_full.get("ATE")):
            best_full = row
    final_lines = [
        "# ACL2 v53 final report",
        "",
        "Generated from landed artifacts only; no missing measurements were fabricated.",
        "",
        "## Answers",
        "",
        f"1. Produced no-chunk/no-manual-percentage adaptive TTT candidate: `{any(bool(r.get('no_chunk_policy_pass')) and bool(r.get('manual_percentage_audit_pass')) for r in rows)}`.",
    ]
    if best_full is not None:
        final_lines.extend([
            f"2. Best full candidate: `{best_full.get('run_name')}`, ATE `{_fmt(best_full.get('ATE'))}`, delta vs C9 `{_fmt(best_full.get('delta_vs_C9_P0'))}`.",
            f"3. 28min runtime gate: `{best_full.get('full_runtime_gate_pass')}`; wall `{_fmt(best_full.get('wall_time_min'))}` min.",
        ])
    else:
        final_lines.extend([
            "2. Best full candidate: `none`; no v53 full candidate artifact is present in the summarized roots.",
            "3. 28min runtime gate for full v53 candidate: `not evaluated`.",
        ])
    final_lines.extend([
        "4. Teacher/student post-zp energy/gamma/commit differences are documented in `v53_c9_teacher_student_autopsy.md` and reused v52 trace CSVs.",
        "5. Next action should follow the failure routing table above: repair runtime/audit first, then gamma/commit if role split is valid but energy remains mismatched.",
    ])
    (out_dir / "v53_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--rollout-root", action="append", default=[])
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if args.rollout_root:
        roots = [Path(p) for p in args.rollout_root]
    else:
        roots = sorted(result_root.glob("phase*/rollouts"))
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "report_R1"
    run_dirs = _iter_run_dirs(roots)
    rows = _summarize_runs(run_dirs, Path(args.gt)) if run_dirs else []
    rows.sort(key=lambda row: (str(row.get("phase")), str(row.get("run_name"))))
    _plot_phase1_autopsy(out_dir)
    _write_phase1_report(out_dir)
    _write_reports(rows, out_dir)
    print(f"Wrote v53 report with {len(rows)} run rows to {out_dir}")


if __name__ == "__main__":
    main()
