#!/usr/bin/env python3
"""Generate ACL2 v46B clean FRAME_ATTN/TTT/SWA factorial reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import (  # noqa: E402
    _align_metrics,
    _load_kitti_gt,
    _load_tum_prediction,
    _segment_ate,
)
from tools.v42_full_online_report import (  # noqa: E402
    _as_positions,
    _ate,
    _rolling_stats,
    _rolling_windows,
)


DEFAULT_RESULT_ROOT = (
    "/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/"
    "acl2_v46b_component_attribution_frame_ttt_swa"
)
DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"

ROWS: List[Tuple[str, int, int, int]] = [
    ("F000_NONE", 0, 0, 0),
    ("F100_ONLY_FRAME_ATTN", 1, 0, 0),
    ("F010_ONLY_TTT", 0, 1, 0),
    ("F001_ONLY_SWA", 0, 0, 1),
    ("F110_FRAME_ATTN_TTT", 1, 1, 0),
    ("F101_FRAME_ATTN_SWA", 1, 0, 1),
    ("F011_TTT_SWA", 0, 1, 1),
    ("F111_ALL_THREE", 1, 1, 1),
]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_status_done(path: Path, run_name: str) -> bool:
    status = path / "run_status.txt"
    return status.exists() and f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vals.append(number)
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _p90(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vals.append(number)
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), 0.90))


def _run_name(row: str) -> str:
    return f"V46B_{row}"


def _metric_path(run_dir: Path) -> Path:
    return run_dir / "01.txt"


def _parse_rpe_metrics(run_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    path = run_dir / "results_sim3" / "results_rpe.txt"
    if not path.exists():
        return None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] == "Average:":
            continue
        try:
            return float(parts[1]), float(parts[2])
        except ValueError:
            continue
    return None, None


def _compute_pose_metrics(run_dir: Path, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    path = _metric_path(run_dir)
    if not path.exists():
        return {"status": "missing_prediction", "frames": 0}
    frames, raw_poses, _ = _load_tum_prediction(path, gt_pos.shape[0])
    if frames.size == 0:
        return {"status": "empty_prediction", "frames": 0}
    aligned, metrics = _align_metrics(frames, raw_poses, gt_poses, gt_pos)
    out: Dict[str, Any] = {
        "status": "done",
        "frames": int(frames.size),
        "ATE_full": metrics.get("ATE_horizon"),
        "Rot_full": metrics.get("Rot_horizon"),
        "FinalErr_full": metrics.get("FinalErr_horizon"),
        "alignment_scale": metrics.get("alignment_scale"),
        "segment_200_300_ATE": _segment_ate(frames, aligned, gt_pos, 200, 300),
        "segment_400_600_ATE": _segment_ate(frames, aligned, gt_pos, 400, 600),
    }
    pos = _as_positions(aligned)
    for width in (50, 100, 200):
        windows = _rolling_windows(frames, pos, gt_pos, width)
        stats = _rolling_stats(windows)
        out[f"rolling{width}_count"] = stats.get("count")
        out[f"rolling{width}_mean"] = stats.get("mean")
        out[f"rolling{width}_p90"] = stats.get("p90")
        out[f"rolling{width}_worst"] = stats.get("worst")
    rpe_t, rpe_r = _parse_rpe_metrics(run_dir)
    out["RPE_t_full"] = rpe_t
    out["RPE_r_full"] = rpe_r
    return out


def _hook_summaries(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    trace = row.get("control_trace")
    if not isinstance(trace, dict):
        return []
    hooks = trace.get("hook_effect_summary")
    if not isinstance(hooks, dict):
        return []
    return [item for item in hooks.values() if isinstance(item, dict)]


def _implemented_paths(row: Dict[str, Any]) -> List[str]:
    trace = row.get("control_trace")
    if not isinstance(trace, dict):
        return []
    paths = trace.get("implemented_paths")
    if isinstance(paths, list):
        return [str(item) for item in paths]
    return []


def _action_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    tri_rows = [
        row for row in rows
        if int(row.get("auxgeo_tri_replay_applied_layer_count") or 0) > 0
    ]
    conflict_values: List[float] = []
    for row in rows:
        summary = row.get("v11_projection_trace_summary")
        if isinstance(summary, dict) and summary.get("ttt_update_conflict_energy") is not None:
            conflict_values.append(float(summary["ttt_update_conflict_energy"]))
    swa_applied = 0
    swa_keep_vals: List[float] = []
    for row in rows:
        for hook in _hook_summaries(row):
            swa_applied += int(hook.get("num_swa_overlap_source_replace_applied") or 0)
            if hook.get("mean_context_source_keep_ratio") is not None:
                try:
                    swa_keep_vals.append(float(hook["mean_context_source_keep_ratio"]))
                except (TypeError, ValueError):
                    pass
    frame_active = any(
        "frame_attention" in _implemented_paths(row)
        or bool((row.get("read_path_controls_requested") or {}).get("frame"))
        for row in rows
        if isinstance(row.get("read_path_controls_requested"), dict)
    )
    return {
        "hmc_rows": len(rows),
        "frame_attn_read_control_active": bool(frame_active),
        "read_beta_effective_mean": _mean(row.get("prior_beta_frame_effective") for row in rows),
        "read_beta_policy": next((str(row.get("prior_beta_policy")) for row in rows if row.get("prior_beta_policy")), ""),
        "D_g_mean": _mean(
            row.get("prior_mean_D_tok")
            if row.get("prior_mean_D_tok") is not None
            else ((row.get("v11_projection_trace_summary") or {}).get("projection_role_mass") or {}).get("D_tok_mean")
            for row in rows
        ),
        "D_g_p90": _mean(
            row.get("prior_q90_D_tok")
            if row.get("prior_q90_D_tok") is not None
            else ((row.get("v11_projection_trace_summary") or {}).get("projection_role_mass") or {}).get("D_tok_p90")
            for row in rows
        ),
        "ttt_tri_replay_applied_count": len(tri_rows),
        "ttt_tri_replay_applied_layer_count_sum": sum(int(row.get("auxgeo_tri_replay_applied_layer_count") or 0) for row in rows),
        "ttt_positive_mass_mean": _mean(row.get("auxgeo_tri_replay_pos_mass_mean") for row in tri_rows),
        "ttt_neutral_mass_mean": _mean(row.get("auxgeo_tri_replay_neu_mass_mean") for row in tri_rows),
        "ttt_negative_mass_mean": _mean(row.get("auxgeo_tri_replay_neg_mass_mean") for row in tri_rows),
        "ttt_update_conflict_energy_mean": _mean(conflict_values),
        "ttt_update_conflict_energy_p90": _p90(conflict_values),
        "swa_overlap_replace_applied_count": int(swa_applied),
        "swa_source_keep_ratio": _mean(swa_keep_vals),
        "swa_boundary_10f": None,
        "swa_boundary_20f": None,
        "swa_boundary_note": "not emitted in hmc_state_hash; requires separate boundary diagnostic if needed",
    }


def _read_boundary_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_dir = str(row.get("run_dir") or "")
            if not run_dir:
                continue
            lookup[run_dir] = row
    return lookup


def _row_valid(row: Dict[str, Any]) -> Tuple[bool, str]:
    reasons: List[str] = []
    if row.get("status") != "done":
        reasons.append(str(row.get("status") or "not_done"))
    if not row.get("no_chunk_policy_pass"):
        reasons.append("chunk_id_policy_not_clean")
    if bool(row["FRAME_ATTN"]) != bool(row.get("frame_attn_read_control_active")):
        reasons.append("frame_attn_debug_mismatch")
    if bool(row["TTT"]):
        if int(row.get("ttt_tri_replay_applied_count") or 0) <= 0:
            reasons.append("ttt_not_applied")
    elif int(row.get("ttt_tri_replay_applied_count") or 0) != 0:
        reasons.append("ttt_applied_when_off")
    if bool(row["SWA"]):
        if int(row.get("swa_overlap_replace_applied_count") or 0) <= 0:
            reasons.append("swa_not_applied")
    elif int(row.get("swa_overlap_replace_applied_count") or 0) != 0:
        reasons.append("swa_applied_when_off")
    return not reasons, ";".join(reasons)


def _load_factorial_rows(
    root: Path,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
    boundary_lookup: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row_name, frame, ttt, swa in ROWS:
        run_name = _run_name(row_name)
        run_dir = root / run_name
        metrics = _compute_pose_metrics(run_dir, gt_poses, gt_pos)
        action = _action_stats(run_dir)
        chunk_audit = _read_json(run_dir / "chunk_id_policy_audit.json")
        no_chunk = not any(bool(chunk_audit.get(key)) for key in (
            "has_read_beta_frame_chunks",
            "has_tri_gamma_chunk_map",
            "has_tri_replay_chunk_params",
            "has_commit_ema_chunks",
        ))
        row: Dict[str, Any] = {
            "row": row_name,
            "run_name": run_name,
            "FRAME_ATTN": frame,
            "TTT": ttt,
            "SWA": swa,
            "run_dir": str(run_dir),
            "run_status_done": _read_status_done(run_dir, run_name),
            "no_chunk_policy_pass": bool(no_chunk),
            "absolute_chunk_id_policy_audit": json.dumps(chunk_audit, sort_keys=True),
        }
        row.update(metrics)
        row.update(action)
        boundary = boundary_lookup.get(str(run_dir), {})
        if boundary:
            row["swa_boundary_10f"] = boundary.get("mean_boundary_10f_ATE")
            row["swa_boundary_20f"] = boundary.get("mean_boundary_20f_ATE")
            row["swa_boundary_note"] = "from v27_swa_boundary_diagnostics; reference=F000_NONE"
        valid, reason = _row_valid(row)
        row["row_valid"] = bool(valid)
        row["invalid_reason"] = reason
        out.append(row)
    return out


def _gain(rows_by_name: Mapping[str, Dict[str, Any]], row: str) -> Optional[float]:
    base = rows_by_name.get("F000_NONE", {})
    cur = rows_by_name.get(row, {})
    try:
        return float(base["ATE_full"]) - float(cur["ATE_full"])
    except (KeyError, TypeError, ValueError):
        return None


def _main_effects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {row["row"]: row for row in rows}
    effects: List[Dict[str, Any]] = []
    for row_name, factor in (
        ("F100_ONLY_FRAME_ATTN", "FRAME_ATTN_only"),
        ("F010_ONLY_TTT", "TTT_only"),
        ("F001_ONLY_SWA", "SWA_only"),
        ("F110_FRAME_ATTN_TTT", "FRAME_ATTN_TTT"),
        ("F101_FRAME_ATTN_SWA", "FRAME_ATTN_SWA"),
        ("F011_TTT_SWA", "TTT_SWA"),
        ("F111_ALL_THREE", "ALL_THREE"),
    ):
        effects.append({
            "factor_or_combo": factor,
            "row": row_name,
            "ATE_full": by_name.get(row_name, {}).get("ATE_full"),
            "gain_vs_F000": _gain(by_name, row_name),
            "row_valid": by_name.get(row_name, {}).get("row_valid"),
            "invalid_reason": by_name.get(row_name, {}).get("invalid_reason"),
        })
    return effects


def _interactions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {row["row"]: row for row in rows}
    g100 = _gain(by_name, "F100_ONLY_FRAME_ATTN")
    g010 = _gain(by_name, "F010_ONLY_TTT")
    g001 = _gain(by_name, "F001_ONLY_SWA")
    g110 = _gain(by_name, "F110_FRAME_ATTN_TTT")
    g101 = _gain(by_name, "F101_FRAME_ATTN_SWA")
    g011 = _gain(by_name, "F011_TTT_SWA")
    g111 = _gain(by_name, "F111_ALL_THREE")

    def classify_read_ttt() -> str:
        if None in (g100, g010, g110):
            return "missing"
        assert g100 is not None and g010 is not None and g110 is not None
        if g110 > max(g100, g010) + 0.20:
            return "synergy"
        if abs(g110 - g010) < 0.10:
            return "read_near_zero_under_ttt"
        if g110 < g010 - 0.20:
            return "conflict"
        return "mixed_or_small"

    def classify_read_swa() -> str:
        if None in (g100, g001, g101):
            return "missing"
        assert g100 is not None and g001 is not None and g101 is not None
        if g101 > max(g100, g001) + 0.20:
            return "synergy"
        if abs(g101 - g100) < 0.10:
            return "swa_near_zero_under_read"
        if g101 < g100 - 0.20:
            return "conflict"
        return "mixed_or_small"

    boundary_rows: List[Dict[str, Any]] = []
    for row in rows:
        try:
            boundary_10f = float(row.get("swa_boundary_10f"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(boundary_10f):
            boundary_rows.append(row)
    if boundary_rows:
        best_boundary = min(boundary_rows, key=lambda item: float(item["swa_boundary_10f"]))
        boundary_note = (
            "from v27_swa_boundary_diagnostics; "
            f"best_10f={best_boundary.get('row')}:{float(best_boundary['swa_boundary_10f']):.6f}"
        )
    else:
        boundary_note = "boundary_10f/20f unavailable unless separate diagnostic is generated"

    return [
        {
            "interaction": "READ_x_TTT",
            "gain_FRAME_ATTN": g100,
            "gain_TTT": g010,
            "gain_combo": g110,
            "synergy_margin": None if None in (g100, g010, g110) else g110 - max(g100, g010),  # type: ignore[operator]
            "classification": classify_read_ttt(),
        },
        {
            "interaction": "READ_x_SWA",
            "gain_FRAME_ATTN": g100,
            "gain_SWA": g001,
            "gain_combo": g101,
            "synergy_margin": None if None in (g100, g001, g101) else g101 - max(g100, g001),  # type: ignore[operator]
            "classification": classify_read_swa(),
            "boundary_note": boundary_note,
        },
        {
            "interaction": "TTT_x_SWA",
            "gain_TTT": g010,
            "gain_SWA": g001,
            "gain_combo": g011,
            "synergy_margin": None if None in (g010, g001, g011) else g011 - max(g010, g001),  # type: ignore[operator]
        },
        {
            "interaction": "THREE_WAY",
            "gain_all_three": g111,
            "gain_best_pair": None if None in (g110, g101, g011) else max(g110, g101, g011),  # type: ignore[arg-type]
            "three_way_margin": None if None in (g110, g101, g011, g111) else g111 - max(g110, g101, g011),  # type: ignore[operator]
        },
    ]


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.6f}"


def _write_summary(path: Path, rows: List[Dict[str, Any]], effects: List[Dict[str, Any]], interactions: List[Dict[str, Any]]) -> None:
    by_name = {row["row"]: row for row in rows}
    lines: List[str] = []
    lines.append("# ACL2 v46B Phase2 FRAME_ATTN / TTT / SWA factorial summary")
    lines.append("")
    lines.append("This report is generated only from landed artifacts. Missing values are not filled.")
    lines.append("")
    lines.append("## Registry")
    lines.append("")
    lines.append("| Row | F | T | S | valid | ATE | APPLIED frame/ttt/swa | invalid reason |")
    lines.append("|---|---:|---:|---:|---|---:|---|---|")
    for row in rows:
        applied = (
            f"{int(bool(row.get('frame_attn_read_control_active')))}/"
            f"{int(row.get('ttt_tri_replay_applied_count') or 0)}/"
            f"{int(row.get('swa_overlap_replace_applied_count') or 0)}"
        )
        lines.append(
            f"| `{row['row']}` | {row['FRAME_ATTN']} | {row['TTT']} | {row['SWA']} | "
            f"{row.get('row_valid')} | {_format_float(row.get('ATE_full'))} | {applied} | {row.get('invalid_reason') or ''} |"
        )
    lines.append("")
    lines.append("## Main Effects")
    lines.append("")
    lines.append("| Factor/combo | row | ATE | Gain vs F000 |")
    lines.append("|---|---|---:|---:|")
    for item in effects:
        lines.append(
            f"| {item['factor_or_combo']} | `{item['row']}` | "
            f"{_format_float(item.get('ATE_full'))} | {_format_float(item.get('gain_vs_F000'))} |"
        )
    lines.append("")
    lines.append("## Interactions")
    lines.append("")
    lines.append("| Interaction | classification | synergy margin |")
    lines.append("|---|---|---:|")
    for item in interactions:
        lines.append(
            f"| {item['interaction']} | {item.get('classification', '')} | {_format_float(item.get('synergy_margin'))} |"
        )
    lines.append("")
    lines.append("## Required Natural-Language Answers")
    lines.append("")
    for label, row_name in (
        ("FRAME_ATTN alone", "F100_ONLY_FRAME_ATTN"),
        ("TTT alone", "F010_ONLY_TTT"),
        ("SWA alone", "F001_ONLY_SWA"),
    ):
        gain = _gain(by_name, row_name)
        lines.append(f"- {label}: gain vs F000 = {_format_float(gain)} m; row valid = {by_name.get(row_name, {}).get('row_valid')}.")
    for inter in interactions:
        if inter["interaction"] in {"READ_x_TTT", "READ_x_SWA"}:
            lines.append(
                f"- {inter['interaction']}: classification = {inter.get('classification')}; "
                f"synergy margin = {_format_float(inter.get('synergy_margin'))} m."
            )
    best = min(
        (row for row in rows if row.get("ATE_full") is not None and row.get("row_valid")),
        key=lambda row: float(row["ATE_full"]),
        default=None,
    )
    if best:
        lines.append(f"- Best valid row: `{best['row']}` with ATE {_format_float(best.get('ATE_full'))}.")
    else:
        lines.append("- Best valid row: none; at least one required debug/gate is missing.")
    lines.append("")
    lines.append("## Audit Notes")
    lines.append("")
    lines.append("- RPE_t/RPE_r are parsed from `results_sim3/results_rpe.txt` when the KITTI benchmark artifact exists.")
    has_boundary = any(row.get("swa_boundary_10f") is not None for row in rows)
    if has_boundary:
        lines.append("- SWA boundary_10f/boundary_20f are imported from `v27_swa_boundary_diagnostics`; missing rows are not filled.")
    else:
        lines.append("- SWA boundary_10f/boundary_20f are left as missing unless a separate boundary diagnostic is generated; the report does not fabricate them.")
    lines.append("- A row is invalid if enabled components do not appear in action debug, disabled components appear, or chunk-id audit is not clean.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (out_dir / "plot_generation_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return
    by_name = {row["row"]: row for row in rows}
    base = by_name.get("F000_NONE", {}).get("ATE_full")
    gains = []
    labels = []
    for row_name, _, _, _ in ROWS:
        labels.append(row_name.replace("_", "\n"))
        gain = _gain(by_name, row_name)
        gains.append(float("nan") if gain is None else gain)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(np.asarray([gains], dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([0])
    ax.set_yticklabels(["Gain vs F000"])
    for idx, gain in enumerate(gains):
        ax.text(idx, 0, _format_float(gain), ha="center", va="center", fontsize=7)
    ax.set_title("v46B component interaction heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "component_interaction_heatmap.png", dpi=180)
    plt.close(fig)

    seg200 = []
    seg400 = []
    for row_name, _, _, _ in ROWS:
        row = by_name.get(row_name, {})
        try:
            seg200.append(float(row.get("segment_200_300_ATE")) - float(by_name["F000_NONE"].get("segment_200_300_ATE")))
        except (TypeError, ValueError, KeyError):
            seg200.append(float("nan"))
        try:
            seg400.append(float(row.get("segment_400_600_ATE")) - float(by_name["F000_NONE"].get("segment_400_600_ATE")))
        except (TypeError, ValueError, KeyError):
            seg400.append(float("nan"))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x, seg200, label="[200,300) delta")
    ax.bar(x, seg400, bottom=np.nan_to_num(seg200, nan=0.0), label="[400,600) delta")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ATE delta vs F000")
    ax.set_title("v46B segment delta stacked bar")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "segment_delta_stacked_bar.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--rollout-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gt", default=DEFAULT_GT)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rollout_root = Path(args.rollout_root) if args.rollout_root else result_root / "phase2_factorial" / "rollouts"
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "phase2_factorial" / "report_R1"
    out_dir.mkdir(parents=True, exist_ok=True)

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    boundary_lookup = _read_boundary_lookup(out_dir / "swa_boundary" / "swa_boundary_summary.csv")
    rows = _load_factorial_rows(rollout_root, gt_poses, gt_pos, boundary_lookup)
    effects = _main_effects(rows)
    interactions = _interactions(rows)

    _write_csv(out_dir / "phase2_factorial_registry.csv", rows)
    _write_csv(out_dir / "phase2_component_main_effects.csv", effects)
    _write_csv(out_dir / "phase2_component_interactions.csv", interactions)
    _write_json(out_dir / "phase2_factorial_registry.json", rows)
    _write_summary(out_dir / "phase2_frame_ttt_swa_summary.md", rows, effects, interactions)
    _write_plots(out_dir, rows)
    print(f"Wrote v46B report to {out_dir}")


if __name__ == "__main__":
    main()
