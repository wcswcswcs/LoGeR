#!/usr/bin/env python3
"""Audit reset-window drift state against TTT/HMC debug signals.

This is intentionally offline: it joins trajectory diagnostics with per-chunk
HMC debug JSONL so we can decide which no-GT state is worth turning into a
controller signal before running more full KITTI jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


Number = float | int


DEFAULT_WINDOWS = (
    "pre:0-4",
    "body:5-9",
    "exit:10-12",
    "mid:13-15",
    "c16:16-16",
    "handoff:17-18",
    "post:19-24",
    "tail:25-37",
)

RUN_LEVEL_LABELS = (
    "ate_rmse",
    "final_error",
    "yaw_rmse",
    "sim3_scale",
    "seg_200_300",
    "seg_200_400",
    "seg_400_600",
)

CHUNK_LABELS = (
    "chunk_rmse_m",
    "chunk_mean_m",
    "chunk_delta_error_m",
    "chunk_abs_delta_error_m",
    "chunk_drift_vec_norm_m",
    "chunk_drift_x_m",
    "chunk_drift_y_m",
    "chunk_drift_z_m",
    "chunk_yaw_delta_deg",
    "future3_chunk_rmse_mean",
)


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def _rmse(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1e-18 or vy <= 1e-18:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_name_path(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    name, raw_path = spec.split("=", 1)
    return name, Path(raw_path)


def _parse_windows(specs: Sequence[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for spec in specs:
        if not spec:
            continue
        name, span = spec.split(":", 1)
        start_s, end_s = span.split("-", 1)
        out.append((name, int(start_s), int(end_s)))
    return out


def _segment_key(start: int, end: int) -> str:
    return f"seg_{start}_{end}"


def _flatten_numeric(prefix: str, value: Any, out: Dict[str, float]) -> None:
    if _is_number(value):
        out[prefix] = float(value)
        return
    if isinstance(value, bool):
        out[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            if not key or "hash" in str(key).lower():
                continue
            new_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(new_key, sub, out)


def _load_summary(diag_dir: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    path = diag_dir / "summary.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    run_metrics: Dict[str, Dict[str, float]] = {}
    run_paths: Dict[str, str] = {}
    for run in data.get("runs", []):
        name = str(run.get("name"))
        run_paths[name] = str(run.get("path", ""))
        metrics = {
            "ate_rmse": _to_float(run.get("aligned_ate_rmse_m")),
            "final_error": _to_float(run.get("final_error_m")),
            "yaw_rmse": _to_float(run.get("yaw_rmse_deg")),
            "sim3_scale": _to_float(run.get("sim3_scale")),
            "rmse_x": _to_float(run.get("axis_rmse_x_m")),
            "rmse_y": _to_float(run.get("axis_rmse_y_m")),
            "rmse_z": _to_float(run.get("axis_rmse_z_m")),
        }
        for item in run.get("segment_summary", {}).get("fixed_segments", []):
            start = _to_int(item.get("start"))
            end = _to_int(item.get("end"))
            metrics[_segment_key(start, end)] = _to_float(item.get("ate_rmse_m"))
        run_metrics[name] = metrics
    return run_metrics, run_paths


def _load_segments(diag_dir: Path, run_metrics: Dict[str, Dict[str, float]]) -> None:
    for row in _load_csv(diag_dir / "segment_errors.csv"):
        run = row.get("run", "")
        start = _to_int(row.get("start"))
        end = _to_int(row.get("end"))
        seg_len = _to_int(row.get("segment_len"))
        if seg_len != end - start:
            continue
        run_metrics.setdefault(run, {})[_segment_key(start, end)] = _to_float(row.get("ate_rmse_m"))


def _load_frame_rows(diag_dir: Path) -> Dict[Tuple[str, int], List[Dict[str, float]]]:
    by_run_chunk: Dict[Tuple[str, int], List[Dict[str, float]]] = defaultdict(list)
    for row in _load_csv(diag_dir / "per_frame_errors.csv"):
        run = str(row.get("run", ""))
        chunk_idx = _to_int(row.get("chunk_idx"))
        if not run or chunk_idx < 0:
            continue
        by_run_chunk[(run, chunk_idx)].append(
            {
                "frame": _to_float(row.get("frame")),
                "aligned_error_m": _to_float(row.get("aligned_error_m")),
                "aligned_error_x_m": _to_float(row.get("aligned_error_x_m")),
                "aligned_error_y_m": _to_float(row.get("aligned_error_y_m")),
                "aligned_error_z_m": _to_float(row.get("aligned_error_z_m")),
                "yaw_error_deg": _to_float(row.get("yaw_error_deg")),
            }
        )
    for rows in by_run_chunk.values():
        rows.sort(key=lambda item: item["frame"])
    return by_run_chunk


def _load_chunk_rows(diag_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in _load_csv(diag_dir / "chunk_errors.csv"):
        run = str(row.get("run", ""))
        chunk_idx = _to_int(row.get("chunk_idx"))
        if not run or chunk_idx < 0:
            continue
        out[(run, chunk_idx)] = {
            "run": run,
            "chunk_idx": chunk_idx,
            "start_frame": _to_int(row.get("start")),
            "end_frame": _to_int(row.get("end")),
            "num_valid": _to_int(row.get("num_valid")),
            "chunk_rmse_m": _to_float(row.get("rmse_m")),
            "chunk_mean_m": _to_float(row.get("mean_m")),
            "chunk_max_m": _to_float(row.get("max_m")),
            "chunk_worst_frame": _to_int(row.get("worst_frame")),
            "chunk_start_error_m": _to_float(row.get("start_error_m")),
            "chunk_end_error_m": _to_float(row.get("end_error_m")),
            "chunk_axis_rmse_x_m": _to_float(row.get("axis_rmse_x_m")),
            "chunk_axis_rmse_y_m": _to_float(row.get("axis_rmse_y_m")),
            "chunk_axis_rmse_z_m": _to_float(row.get("axis_rmse_z_m")),
        }
    return out


def _load_hmc_features(run_dirs: Mapping[str, Path]) -> Dict[Tuple[str, int], Dict[str, float]]:
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for run, run_dir in run_dirs.items():
        path = run_dir / "hmc_state_hash.jsonl"
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            chunk_idx = _to_int(row.get("chunk_idx"))
            if chunk_idx < 0:
                continue
            features: Dict[str, float] = {}
            for key, value in row.items():
                if key in {
                    "control_trace",
                    "memory_side_effect",
                    "probe_hmc_hook_trace_counts",
                    "hook_trace_counts",
                    "read_path_controls_requested",
                }:
                    continue
                if _is_number(value) or isinstance(value, bool):
                    features[f"hmc.{key}"] = float(value)
            mem = row.get("memory_side_effect", {})
            if isinstance(mem, dict):
                _flatten_numeric("hmc.memory_side_effect", mem, features)
            hooks = row.get("control_trace", {}).get("hook_effect_summary", {})
            if isinstance(hooks, dict):
                _flatten_numeric("hmc.hooks", hooks, features)
            out[(run, chunk_idx)] = features
    return out


def _build_chunk_attribution(
    diag_dir: Path,
    run_dirs: Mapping[str, Path],
    run_metrics: Dict[str, Dict[str, float]],
    *,
    reset_every: int,
) -> List[Dict[str, Any]]:
    frames = _load_frame_rows(diag_dir)
    chunks = _load_chunk_rows(diag_dir)
    hmc = _load_hmc_features(run_dirs)
    by_run_chunk_rmse: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (run, chunk_idx), row in chunks.items():
        by_run_chunk_rmse[run][chunk_idx] = float(row.get("chunk_rmse_m", float("nan")))

    rows: List[Dict[str, Any]] = []
    for key, base in sorted(chunks.items()):
        run, chunk_idx = key
        out = dict(base)
        out["reset_group"] = chunk_idx // reset_every if reset_every > 0 else 0
        out["reset_phase"] = chunk_idx % reset_every if reset_every > 0 else chunk_idx

        frame_rows = frames.get(key, [])
        if frame_rows:
            first = frame_rows[0]
            last = frame_rows[-1]
            dx = last["aligned_error_x_m"] - first["aligned_error_x_m"]
            dy = last["aligned_error_y_m"] - first["aligned_error_y_m"]
            dz = last["aligned_error_z_m"] - first["aligned_error_z_m"]
            out["chunk_drift_x_m"] = dx
            out["chunk_drift_y_m"] = dy
            out["chunk_drift_z_m"] = dz
            out["chunk_drift_vec_norm_m"] = math.sqrt(dx * dx + dy * dy + dz * dz)
            out["chunk_yaw_delta_deg"] = last["yaw_error_deg"] - first["yaw_error_deg"]
            out["chunk_aligned_error_start_x_m"] = first["aligned_error_x_m"]
            out["chunk_aligned_error_start_y_m"] = first["aligned_error_y_m"]
            out["chunk_aligned_error_start_z_m"] = first["aligned_error_z_m"]
            out["chunk_aligned_error_end_x_m"] = last["aligned_error_x_m"]
            out["chunk_aligned_error_end_y_m"] = last["aligned_error_y_m"]
            out["chunk_aligned_error_end_z_m"] = last["aligned_error_z_m"]
            out["chunk_frame_error_slope_m_per_frame"] = (
                last["aligned_error_m"] - first["aligned_error_m"]
            ) / max(last["frame"] - first["frame"], 1.0)
        out["chunk_delta_error_m"] = _to_float(out.get("chunk_end_error_m")) - _to_float(out.get("chunk_start_error_m"))
        out["chunk_abs_delta_error_m"] = abs(out["chunk_delta_error_m"])
        future = [
            by_run_chunk_rmse[run].get(i, float("nan"))
            for i in range(chunk_idx + 1, chunk_idx + 4)
        ]
        out["future3_chunk_rmse_mean"] = _mean(future)

        for label in RUN_LEVEL_LABELS:
            out[label] = run_metrics.get(run, {}).get(label, float("nan"))
        for name, value in hmc.get(key, {}).items():
            out[name] = value
        rows.append(out)
    return rows


def _feature_columns(rows: List[Mapping[str, Any]]) -> List[str]:
    protected = {
        "run",
        "chunk_idx",
        "start_frame",
        "end_frame",
        "num_valid",
        "reset_group",
        "reset_phase",
    }
    labels = set(CHUNK_LABELS) | set(RUN_LEVEL_LABELS)
    cols: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key in protected or key in labels:
                continue
            if _is_number(value):
                cols.append(key)
    return sorted(set(cols))


def _correlations(rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    features = _feature_columns(rows)
    targets = list(CHUNK_LABELS) + list(RUN_LEVEL_LABELS)
    out: List[Dict[str, Any]] = []
    for target in targets:
        ys = [_to_float(row.get(target)) for row in rows]
        for feature in features:
            xs = [_to_float(row.get(feature)) for row in rows]
            n = sum(1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y))
            corr = _pearson(xs, ys)
            if n >= 5 and math.isfinite(corr):
                out.append(
                    {
                        "target": target,
                        "feature": feature,
                        "pearson_r": corr,
                        "abs_r": abs(corr),
                        "n": n,
                    }
                )
    out.sort(key=lambda row: (row["target"], -row["abs_r"], row["feature"]))
    return out


def _window_summary(rows: List[Mapping[str, Any]], windows: Sequence[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    by_run: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[str(row.get("run"))].append(row)

    selected_feature_names = [
        "hmc.memory_ttt_mean_rel_diff",
        "hmc.memory_ttt_w0_mean_rel_diff",
        "hmc.pass1_pass2_pose_t_mean",
        "hmc.pass1_pass2_pose_r_deg_mean",
        "hmc.pass1_pass2_world_points_l1_mean",
        "hmc.prior_mean_D_patch",
        "hmc.prior_dynamic_mass_D_gt_050",
        "hmc.prior_hmc_write_score_mean",
        "hmc.hooks.swa_read.mean_swa_overlap_source_replace_alpha",
        "hmc.hooks.swa_read.mean_swa_overlap_source_replace_score",
    ]
    out: List[Dict[str, Any]] = []
    for run, run_rows in sorted(by_run.items()):
        for name, start, end in windows:
            vals = [row for row in run_rows if start <= _to_int(row.get("chunk_idx")) <= end]
            if not vals:
                continue
            dx = sum(_to_float(row.get("chunk_drift_x_m"), 0.0) for row in vals)
            dy = sum(_to_float(row.get("chunk_drift_y_m"), 0.0) for row in vals)
            dz = sum(_to_float(row.get("chunk_drift_z_m"), 0.0) for row in vals)
            row_out: Dict[str, Any] = {
                "run": run,
                "window": name,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_count": len(vals),
                "mean_chunk_rmse_m": _mean(_to_float(row.get("chunk_rmse_m")) for row in vals),
                "rmse_chunk_rmse_m": _rmse(_to_float(row.get("chunk_rmse_m")) for row in vals),
                "mean_delta_error_m": _mean(_to_float(row.get("chunk_delta_error_m")) for row in vals),
                "sum_drift_x_m": dx,
                "sum_drift_y_m": dy,
                "sum_drift_z_m": dz,
                "sum_drift_vec_norm_m": math.sqrt(dx * dx + dy * dy + dz * dz),
                "mean_drift_vec_norm_m": _mean(_to_float(row.get("chunk_drift_vec_norm_m")) for row in vals),
                "mean_yaw_delta_deg": _mean(_to_float(row.get("chunk_yaw_delta_deg")) for row in vals),
            }
            for feature in selected_feature_names:
                row_out[feature] = _mean(_to_float(row.get(feature)) for row in vals)
            for label in RUN_LEVEL_LABELS:
                row_out[label] = _to_float(vals[0].get(label))
            out.append(row_out)
    return out


def _window_correlations(rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_window: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[str(row.get("window"))].append(row)
    protected = {"run", "window", "chunk_start", "chunk_end", "chunk_count"}
    labels = set(RUN_LEVEL_LABELS)
    out: List[Dict[str, Any]] = []
    for window, vals in sorted(by_window.items()):
        feature_names = sorted(
            {
                key
                for row in vals
                for key, value in row.items()
                if key not in protected and key not in labels and _is_number(value)
            }
        )
        for target in RUN_LEVEL_LABELS:
            ys = [_to_float(row.get(target)) for row in vals]
            for feature in feature_names:
                xs = [_to_float(row.get(feature)) for row in vals]
                n = sum(1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y))
                corr = _pearson(xs, ys)
                if n >= 4 and math.isfinite(corr):
                    out.append(
                        {
                            "window": window,
                            "target": target,
                            "feature": feature,
                            "pearson_r": corr,
                            "abs_r": abs(corr),
                            "n": n,
                        }
                    )
    out.sort(key=lambda row: (row["window"], row["target"], -row["abs_r"], row["feature"]))
    return out


def _reference_deltas(rows: List[Mapping[str, Any]], reference_run: str) -> List[Dict[str, Any]]:
    by_key = {(str(row.get("run")), _to_int(row.get("chunk_idx"))): row for row in rows}
    ref_by_chunk = {
        chunk_idx: row
        for (run, chunk_idx), row in by_key.items()
        if run == reference_run
    }
    out: List[Dict[str, Any]] = []
    metrics = [
        "chunk_rmse_m",
        "chunk_delta_error_m",
        "chunk_drift_vec_norm_m",
        "chunk_drift_x_m",
        "chunk_drift_y_m",
        "chunk_drift_z_m",
        "chunk_yaw_delta_deg",
        "hmc.memory_ttt_w0_mean_rel_diff",
        "hmc.pass1_pass2_pose_t_mean",
        "hmc.prior_mean_D_patch",
        "hmc.hooks.swa_read.mean_swa_overlap_source_replace_alpha",
    ]
    for row in rows:
        run = str(row.get("run"))
        chunk_idx = _to_int(row.get("chunk_idx"))
        ref = ref_by_chunk.get(chunk_idx)
        if run == reference_run or ref is None:
            continue
        delta_row: Dict[str, Any] = {"run": run, "reference_run": reference_run, "chunk_idx": chunk_idx}
        for metric in metrics:
            delta_row[f"delta_{metric}"] = _to_float(row.get(metric)) - _to_float(ref.get(metric))
        out.append(delta_row)
    return out


def _write_markdown(
    path: Path,
    rows: List[Mapping[str, Any]],
    windows: List[Mapping[str, Any]],
    corr_rows: List[Mapping[str, Any]],
    window_corr_rows: List[Mapping[str, Any]],
    reference_run: str,
) -> None:
    run_best: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        run_best.setdefault(str(row.get("run")), row)
    ranked = sorted(run_best.values(), key=lambda row: _to_float(row.get("ate_rmse")))
    body = [w for w in windows if str(w.get("window")) == "body"]
    exit_rows = [w for w in windows if str(w.get("window")) == "exit"]
    handoff = [w for w in windows if str(w.get("window")) in {"c16", "handoff"}]

    def fmt(value: Any, digits: int = 4) -> str:
        v = _to_float(value)
        return f"{v:.{digits}f}" if math.isfinite(v) else "nan"

    lines: List[str] = []
    lines.append("# H6 Window Drift-State Audit")
    lines.append("")
    lines.append(f"reference_run = `{reference_run}`")
    lines.append("")
    lines.append("## Run ranking")
    lines.append("")
    lines.append("| Run | ATE | FinalErr | Yaw | [200,300) | [400,600) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in ranked[:12]:
        lines.append(
            f"| `{row.get('run')}` | {fmt(row.get('ate_rmse'))} | {fmt(row.get('final_error'))} | "
            f"{fmt(row.get('yaw_rmse'))} | {fmt(row.get('seg_200_300'))} | {fmt(row.get('seg_400_600'))} |"
        )
    lines.append("")
    lines.append("## Window drift")
    lines.append("")
    lines.append("| Run | Window | mean chunk RMSE | sum drift norm | drift x/y/z | mean pose t | mean w0 rel |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(body + exit_rows + handoff, key=lambda r: (str(r.get("run")), str(r.get("window")))):
        xyz = f"{fmt(row.get('sum_drift_x_m'), 2)}/{fmt(row.get('sum_drift_y_m'), 2)}/{fmt(row.get('sum_drift_z_m'), 2)}"
        lines.append(
            f"| `{row.get('run')}` | `{row.get('window')}` | {fmt(row.get('mean_chunk_rmse_m'))} | "
            f"{fmt(row.get('sum_drift_vec_norm_m'))} | {xyz} | "
            f"{fmt(row.get('hmc.pass1_pass2_pose_t_mean'), 5)} | "
            f"{fmt(row.get('hmc.memory_ttt_w0_mean_rel_diff'), 5)} |"
        )
    lines.append("")
    lines.append("## Strongest Correlations")
    lines.append("")
    for target in ["chunk_drift_vec_norm_m", "chunk_delta_error_m", "seg_200_300", "seg_400_600", "ate_rmse"]:
        subset = [row for row in corr_rows if row.get("target") == target][:8]
        if not subset:
            continue
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Feature | r | n |")
        lines.append("|---|---:|---:|")
        for row in subset:
            lines.append(f"| `{row.get('feature')}` | {fmt(row.get('pearson_r'), 3)} | {row.get('n')} |")
        lines.append("")
    lines.append("## Window-Level Correlations")
    lines.append("")
    for window in ["body", "exit", "c16", "handoff", "post"]:
        for target in ["ate_rmse", "seg_200_300", "seg_400_600", "final_error", "yaw_rmse"]:
            subset = [row for row in window_corr_rows if row.get("window") == window and row.get("target") == target][:5]
            if not subset:
                continue
            lines.append(f"### {window} -> {target}")
            lines.append("")
            lines.append("| Feature | r | n |")
            lines.append("|---|---:|---:|")
            for row in subset:
                lines.append(f"| `{row.get('feature')}` | {fmt(row.get('pearson_r'), 3)} | {row.get('n')} |")
            lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append("- `chunk_attribution.csv`: per-run/per-chunk trajectory labels plus HMC/SWA/TTT no-GT state.")
    lines.append("- `window_summary.csv`: reset-window role summaries for body/exit/c16/handoff windows.")
    lines.append("- `feature_correlations.csv`: Pearson correlations for candidate drift-state proxies.")
    lines.append("- `delta_vs_reference.csv`: per-chunk deltas against the reference run.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", required=True, help="trajectory diagnostics directory")
    parser.add_argument("--run-dir", action="append", default=[], help="RUN=path/to/run_dir containing hmc_state_hash.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-run", default="")
    parser.add_argument("--reset-every", type=int, default=5)
    parser.add_argument("--window", action="append", default=list(DEFAULT_WINDOWS), help="NAME:START-END chunk window")
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_metrics, summary_paths = _load_summary(diag_dir)
    _load_segments(diag_dir, run_metrics)
    run_dirs: Dict[str, Path] = {}
    for spec in args.run_dir:
        name, path = _parse_name_path(spec)
        run_dirs[name] = path
    for name, pred_path in summary_paths.items():
        if name not in run_dirs and pred_path:
            run_dirs[name] = Path(pred_path).parent

    rows = _build_chunk_attribution(diag_dir, run_dirs, run_metrics, reset_every=args.reset_every)
    if not rows:
        raise SystemExit(f"No chunk attribution rows could be built from {diag_dir}")

    reference_run = args.reference_run or min(
        {str(row["run"]) for row in rows},
        key=lambda run: run_metrics.get(run, {}).get("ate_rmse", float("inf")),
    )
    windows = _window_summary(rows, _parse_windows(args.window))
    corr_rows = _correlations(rows)
    window_corr_rows = _window_correlations(windows)
    deltas = _reference_deltas(rows, reference_run)

    fields = sorted({key for row in rows for key in row.keys()})
    _write_csv(out_dir / "chunk_attribution.csv", rows, fields)
    _write_csv(out_dir / "window_summary.csv", windows)
    _write_csv(out_dir / "feature_correlations.csv", corr_rows, ["target", "feature", "pearson_r", "abs_r", "n"])
    _write_csv(
        out_dir / "window_feature_correlations.csv",
        window_corr_rows,
        ["window", "target", "feature", "pearson_r", "abs_r", "n"],
    )
    _write_csv(out_dir / "delta_vs_reference.csv", deltas)
    _write_markdown(out_dir / "h6_window_drift_state_audit.md", rows, windows, corr_rows, window_corr_rows, reference_run)
    print(f"Wrote H6 drift-state audit to {out_dir}")


if __name__ == "__main__":
    main()
