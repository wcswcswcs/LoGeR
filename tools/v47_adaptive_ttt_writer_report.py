#!/usr/bin/env python3
"""Report ACL2 v47 adaptive TTT writer runs from landed artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import _align_metrics, _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v42_full_online_report import _as_positions, _ate, _rolling_stats, _rolling_windows  # noqa: E402


DEFAULT_RESULT_ROOT = (
    "/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/"
    "acl2_v47_adaptive_ttt_writer_nochunk"
)
DEFAULT_V46B_ROOT = (
    "/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/"
    "acl2_v46b_component_attribution_frame_ttt_swa"
)
DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"
C9_P0_ATE = 33.76294210291885

ROW_TO_FIXED_REFERENCE = {
    "AW010_ADAPTIVE_TTT_ONLY": "F010_ONLY_TTT",
    "AW110_FRAME_ADAPTIVE_TTT": "F110_FRAME_ATTN_TTT",
    "AW111_FRAME_ADAPTIVE_TTT_SWA": "F111_ALL_THREE",
}


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


def _first_yaml_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def _done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    return status.exists() and f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


def _pose_metrics(run_dir: Path, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    path = run_dir / "01.txt"
    if not path.exists():
        return {"status": "missing_prediction", "frames": 0}
    frames, raw_poses, _ = _load_tum_prediction(path, gt_pos.shape[0])
    if frames.size == 0:
        return {"status": "empty_prediction", "frames": 0}
    aligned, metrics = _align_metrics(frames, raw_poses, gt_poses, gt_pos)
    pos = _as_positions(aligned)
    out: Dict[str, Any] = {
        "status": "done",
        "frames": int(frames.size),
        "ATE_full": metrics.get("ATE_horizon"),
        "Rot_full": metrics.get("Rot_horizon"),
        "FinalErr_full": metrics.get("FinalErr_horizon"),
        "alignment_scale": metrics.get("alignment_scale"),
    }
    for width in (50, 100, 200):
        stats = _rolling_stats(_rolling_windows(frames, pos, gt_pos, width))
        out[f"rolling{width}_mean"] = stats.get("mean")
        out[f"rolling{width}_p90"] = stats.get("p90")
        out[f"rolling{width}_worst"] = stats.get("worst")
    return out


def _walk(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _debug_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    role_modes: List[str] = []
    adaptive_gammas: List[float] = []
    adaptive_lambdas: List[float] = []
    adaptive_fused_count = 0
    adaptive_split_count = 0
    role_source_values: List[str] = []
    for row in rows:
        for node in _walk(row):
            if node.get("ttt_tri_replay_role_mode") is not None:
                role_modes.append(str(node.get("ttt_tri_replay_role_mode")))
            if node.get("ttt_tri_replay_adaptive_gamma") is not None:
                adaptive_gammas.append(float(node["ttt_tri_replay_adaptive_gamma"]))
            if node.get("ttt_tri_replay_adaptive_neutral_lambda") is not None:
                adaptive_lambdas.append(float(node["ttt_tri_replay_adaptive_neutral_lambda"]))
            if node.get("ttt_tri_replay_adaptive_writer_fused") is True:
                adaptive_fused_count += 1
            if node.get("ttt_tri_replay_adaptive_writer_split") is True:
                adaptive_split_count += 1
            if node.get("ttt_tri_replay_role_source") is not None:
                role_source_values.append(str(node.get("ttt_tri_replay_role_source")))
    log_text = ""
    log_path = run_dir / "01.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if log_text:
        role_modes.extend(re.findall(r"['\"]ttt_tri_replay_role_mode['\"]:\s*['\"]([^'\"]+)['\"]", log_text))
        role_source_values.extend(re.findall(r"['\"]ttt_tri_replay_role_source['\"]:\s*['\"]([^'\"]+)['\"]", log_text))
        adaptive_fused_count += len(re.findall(r"['\"]ttt_tri_replay_adaptive_writer_fused['\"]:\s*True", log_text))
        adaptive_split_count += len(re.findall(r"['\"]ttt_tri_replay_adaptive_writer_split['\"]:\s*True", log_text))
        adaptive_gammas.extend(
            float(x)
            for x in re.findall(r"['\"]ttt_tri_replay_adaptive_gamma['\"]:\s*([0-9.eE+-]+)", log_text)
        )
        adaptive_lambdas.extend(
            float(x)
            for x in re.findall(r"['\"]ttt_tri_replay_adaptive_neutral_lambda['\"]:\s*([0-9.eE+-]+)", log_text)
        )
    tri_rows = [
        row for row in rows
        if int(row.get("auxgeo_tri_replay_applied_layer_count") or 0) > 0
    ]
    return {
        "hmc_rows": len(rows),
        "ttt_tri_replay_applied_count": len(tri_rows),
        "ttt_tri_replay_applied_layer_count_sum": sum(int(row.get("auxgeo_tri_replay_applied_layer_count") or 0) for row in rows),
        "ttt_positive_mass_mean": _mean(row.get("auxgeo_tri_replay_pos_mass_mean") for row in tri_rows),
        "ttt_neutral_mass_mean": _mean(row.get("auxgeo_tri_replay_neu_mass_mean") for row in tri_rows),
        "ttt_negative_mass_mean": _mean(row.get("auxgeo_tri_replay_neg_mass_mean") for row in tri_rows),
        "adaptive_writer_fused_debug_count": adaptive_fused_count,
        "adaptive_writer_split_debug_count": adaptive_split_count,
        "adaptive_gamma_mean": _mean(adaptive_gammas),
        "adaptive_neutral_lambda_mean": _mean(adaptive_lambdas),
        "role_modes_seen": ",".join(sorted(set(role_modes))),
        "role_sources_seen": ",".join(sorted(set(role_source_values))),
    }


def _timing_stats(run_dir: Path) -> Dict[str, Any]:
    timing = _read_json(run_dir / "timing_summary.json")
    chunks = timing.get("chunks", []) if isinstance(timing, dict) else []
    if not isinstance(chunks, list):
        chunks = []
    return {
        "timing_chunks": len(chunks),
        "chunk_total_seconds_mean": _mean((c or {}).get("chunk_total_seconds") for c in chunks if isinstance(c, dict)),
        "probe_ttt_write_seconds_mean": _mean((c or {}).get("probe_ttt_write_seconds") for c in chunks if isinstance(c, dict)),
        "total_runtime_seconds_including_model_load": timing.get("total_runtime_seconds_including_model_load"),
        "wall_seconds": _read_json(run_dir / "wall_time_summary.json").get("wall_seconds"),
    }


def _read_fixed_references(v46b_root: Path) -> Dict[str, Dict[str, Any]]:
    path = v46b_root / "phase2_factorial" / "report_R1" / "phase2_factorial_registry.csv"
    if not path.exists():
        return {}
    refs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            refs[str(row.get("row") or "")] = dict(row)
    return refs


def _load_rows(rollout_root: Path, gt_poses: np.ndarray, gt_pos: np.ndarray, fixed_refs: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in rollout_root.iterdir() if p.is_dir()):
        if not (run_dir / "v47_effective_config.yaml").exists():
            continue
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        config_path = run_dir / "v47_effective_config.yaml"
        row_name = _first_yaml_value(config_path, "row") or run_name.replace("V47_", "", 1)
        row: Dict[str, Any] = {
            "row": row_name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "run_status_done": _done(run_dir, run_name),
        }
        row.update(_pose_metrics(run_dir, gt_poses, gt_pos))
        row.update(_debug_stats(run_dir))
        row.update(_timing_stats(run_dir))
        chunk_audit = _read_json(run_dir / "chunk_id_policy_audit.json")
        adapt_audit = _read_json(run_dir / "adaptive_writer_audit.json")
        row["no_chunk_policy_pass"] = not any(bool(chunk_audit.get(key)) for key in (
            "has_read_beta_frame_chunks",
            "has_tri_gamma_chunk_map",
            "has_tri_replay_chunk_params",
            "has_commit_ema_chunks",
        ))
        row["adaptive_writer_audit_pass"] = bool(adapt_audit.get("adaptive_ttt_writer")) and bool(adapt_audit.get("no_manual_tri_replay_percentages"))
        row["role_mode_config"] = adapt_audit.get("role_mode")
        full_kitti01 = int(row.get("frames") or 0) >= 1000
        row["full_kitti01"] = bool(full_kitti01)
        try:
            ate = float(row["ATE_full"])
            row["delta_vs_C9_P0"] = ate - C9_P0_ATE if full_kitti01 else None
        except (KeyError, TypeError, ValueError):
            pass
        fixed_key = ROW_TO_FIXED_REFERENCE.get(row_name, "")
        fixed = fixed_refs.get(fixed_key, {})
        row["fixed_reference_row"] = fixed_key
        if full_kitti01 and fixed.get("ATE_full") not in (None, "") and row.get("ATE_full") is not None:
            row["delta_vs_fixed_percentage_TTT"] = float(row["ATE_full"]) - float(fixed["ATE_full"])
        if fixed.get("chunk_total_seconds_mean") not in (None, "") and row.get("chunk_total_seconds_mean") is not None:
            row["chunk_total_seconds_delta_vs_fixed"] = float(row["chunk_total_seconds_mean"]) - float(fixed["chunk_total_seconds_mean"])
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.6f}"


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# ACL2 v47 adaptive TTT writer report")
    lines.append("")
    lines.append("Generated from landed artifacts only. Missing runs are not filled.")
    lines.append("")
    lines.append("| Run | Row | status | frames | full | ATE | delta vs C9/P0 | fixed ref | delta vs fixed TTT | chunk sec mean | probe_ttt sec mean | adaptive gamma mean | valid audit |")
    lines.append("|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| `{row.get('run_name')}` | `{row.get('row')}` | `{row.get('status')}` | {row.get('frames')} | {row.get('full_kitti01')} | "
            f"{_fmt(row.get('ATE_full'))} | {_fmt(row.get('delta_vs_C9_P0'))} | "
            f"`{row.get('fixed_reference_row') or ''}` | {_fmt(row.get('delta_vs_fixed_percentage_TTT'))} | "
            f"{_fmt(row.get('chunk_total_seconds_mean'))} | {_fmt(row.get('probe_ttt_write_seconds_mean'))} | "
            f"{_fmt(row.get('adaptive_gamma_mean'))} | {row.get('no_chunk_policy_pass')}/{row.get('adaptive_writer_audit_pass')} |"
        )
    lines.append("")
    lines.append("Gate note: C9/P0 reference ATE is 33.76294210291885; acceptable tolerance requested by user is roughly +0.3m.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--rollout-root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--v46b-result-root", default=DEFAULT_V46B_ROOT)
    parser.add_argument("--gt", default=DEFAULT_GT)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rollout_root = Path(args.rollout_root) if args.rollout_root else result_root / "phase1_adaptive_writer" / "rollouts"
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "phase1_adaptive_writer" / "report_R1"
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    fixed_refs = _read_fixed_references(Path(args.v46b_result_root))
    rows = _load_rows(rollout_root, gt_poses, gt_pos, fixed_refs)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "v47_adaptive_ttt_writer_registry.csv", rows)
    _write_json(out_dir / "v47_adaptive_ttt_writer_registry.json", rows)
    _write_md(out_dir / "v47_adaptive_ttt_writer_report.md", rows)
    print(f"Wrote v47 adaptive TTT writer report to {out_dir}")


if __name__ == "__main__":
    main()
