#!/usr/bin/env python3
"""Build v40 health/no-op atlas from landed rollout artifacts only.

The script audits Phase 0 no-op drift and writes aggregate health timelines.
Missing tensor-level evidence is recorded explicitly instead of reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import (  # noqa: E402
    _align_metrics,
    _load_kitti_gt,
    _load_tum_prediction,
    _raw_diff,
)


DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"


def _parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"invalid_json_line": line[:160]})
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"invalid_json": str(path)}


def _status_done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return False
    return f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="ignore")


def _run_name(prefix: str, parent: str, candidate: str, chunk: int, horizon: int) -> str:
    return f"{prefix}_{parent}_{candidate}_chunk{int(chunk)}_h{int(horizon)}_globalgate_H9parent_SWKS3"


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _max_nested_hook(rows: Sequence[Dict[str, Any]], key: str) -> float:
    best = 0.0
    for row in rows:
        hook = row.get("hook_effect_summary")
        if not isinstance(hook, dict):
            continue
        for payload in hook.values():
            if not isinstance(payload, dict):
                continue
            val = _float(payload.get(key), 0.0)
            if val > best:
                best = val
    return best


def _sum_nested_hook(rows: Sequence[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        hook = row.get("hook_effect_summary")
        if not isinstance(hook, dict):
            continue
        for payload in hook.values():
            if isinstance(payload, dict):
                total += _float(payload.get(key), 0.0)
    return total


def _count_attention_mass_rows(rows: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        hook = row.get("hook_effect_summary")
        if not isinstance(hook, dict):
            continue
        if any(bool(v.get("attention_mass_available")) for v in hook.values() if isinstance(v, dict)):
            count += 1
    return count


def _run_health_row(
    run_dir: Path,
    parent: str,
    candidate: str,
    chunk: int,
    horizon: int,
    phase: str,
) -> Dict[str, Any]:
    hook_rows = _read_jsonl(run_dir / "hook_effect_summary.jsonl")
    cue_rows = _read_jsonl(run_dir / "cue_quality_per_chunk.jsonl")
    path_rows = _read_jsonl(run_dir / "semantic_memory_path_summary.jsonl")
    role_rows = _read_jsonl(run_dir / "semantic_role_summary.jsonl")
    hmc_hash_rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    ttt_rows = _read_jsonl(run_dir / "hmc_control_summary.jsonl")
    cue_summary = _read_json(run_dir / "cue_quality_summary.json")
    return {
        "phase": phase,
        "parent": parent,
        "chunk": int(chunk),
        "horizon": int(horizon),
        "candidate": candidate,
        "run_dir": str(run_dir),
        "cue_quality_rows": len(cue_rows),
        "hook_effect_rows": len(hook_rows),
        "source_influence_rows": len(path_rows),
        "semantic_role_rows": len(role_rows),
        "hmc_state_hash_rows": len(hmc_hash_rows),
        "ttt_health_rows": len(ttt_rows),
        "swa_health_rows": sum(1 for r in hook_rows if "swa_read" in r.get("hook_effect_summary", {})),
        "attention_mass_rows": _count_attention_mass_rows(hook_rows),
        "context_empty_source_events": _sum_nested_hook(hook_rows, "num_context_empty_source_events"),
        "max_context_source_skip_tokens": _max_nested_hook(hook_rows, "max_context_source_skip_tokens"),
        "max_source_gate_delta": _max_nested_hook(hook_rows, "max_abs_gate_delta"),
        "max_swa_overlap_source_gate_delta": _max_nested_hook(hook_rows, "max_swa_overlap_source_gate_delta"),
        "max_removed_attention_mass_before": _max_nested_hook(hook_rows, "mean_attention_mass_removed_before"),
        "cue_quality_pass_fraction": cue_summary.get("cue_quality_pass_fraction"),
        "mean_prior_dynamic_mass_D_gt_050": cue_summary.get("mean_prior_dynamic_mass_D_gt_050"),
        "max_prior_dynamic_mass_D_gt_050": cue_summary.get("max_prior_dynamic_mass_D_gt_050"),
        "mean_prior_anchor_collision": cue_summary.get("mean_prior_anchor_collision"),
        "max_prior_anchor_collision": cue_summary.get("max_prior_anchor_collision"),
        "mean_prior_fragmentation": cue_summary.get("mean_prior_fragmentation"),
        "max_prior_fragmentation": cue_summary.get("max_prior_fragmentation"),
    }


def _plot_heatmap(path: Path, rows: Sequence[Dict[str, Any]], value_key: str, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not rows:
        return False
    labels = [f"{r['parent']}/c{r['chunk']}/{r['candidate']}" for r in rows]
    values = np.asarray([_float(r.get(value_key), 0.0) for r in rows], dtype=np.float64)[None, :]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.28), 2.8))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_yticks([0])
    ax.set_yticklabels([value_key])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase0-rollout-root", type=Path, required=True)
    parser.add_argument("--phase0-prefix", required=True)
    parser.add_argument("--phase0-candidates", required=True)
    parser.add_argument("--phase1-rollout-root", type=Path)
    parser.add_argument("--phase1-prefix")
    parser.add_argument("--phase1-candidates", default="")
    parser.add_argument("--parents", default="H9,C9")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--phase1-horizon", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--noop-reference", default="P0_00_C9_REFERENCE")
    parser.add_argument("--noop-ate-tol", type=float, default=1e-9)
    parser.add_argument("--noop-raw-tol", type=float, default=1e-9)
    args = parser.parse_args()

    parents = _parse_csv_strs(args.parents)
    chunks = _parse_csv_ints(args.chunks)
    phase0_candidates = _parse_csv_strs(args.phase0_candidates)
    phase1_candidates = _parse_csv_strs(args.phase1_candidates)
    _, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))

    noop_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for parent in parents:
        for chunk in chunks:
            ref_name = _run_name(args.phase0_prefix, parent, args.noop_reference, chunk, args.horizon)
            ref_dir = args.phase0_rollout_root / ref_name
            if not _status_done(ref_dir, ref_name) or not (ref_dir / "01.txt").exists():
                missing.append({"phase": "phase0", "parent": parent, "chunk": chunk, "candidate": args.noop_reference, "reason": "missing_reference"})
                continue
            ref_frames, ref_poses, _ = _load_tum_prediction(ref_dir / "01.txt", gt_pos.shape[0])
            ref_lookup = {int(f): p for f, p in zip(ref_frames.astype(np.int64), ref_poses)}
            _, ref_metrics = _align_metrics(ref_frames.astype(np.int64), ref_poses, gt_poses, gt_pos)
            for candidate in phase0_candidates:
                run_name = _run_name(args.phase0_prefix, parent, candidate, chunk, args.horizon)
                run_dir = args.phase0_rollout_root / run_name
                if not _status_done(run_dir, run_name) or not (run_dir / "01.txt").exists():
                    missing.append({"phase": "phase0", "parent": parent, "chunk": chunk, "candidate": candidate, "run_name": run_name, "reason": "missing_or_not_done"})
                    continue
                frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
                frames = frames.astype(np.int64)
                _, metrics = _align_metrics(frames, poses, gt_poses, gt_pos)
                raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames, poses, ref_lookup)
                row = {
                    "phase": "phase0",
                    "parent": parent,
                    "chunk": int(chunk),
                    "horizon": int(args.horizon),
                    "candidate": candidate,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "ATE": float(metrics["ATE_horizon"]),
                    "reference_ATE": float(ref_metrics["ATE_horizon"]),
                    "ATE_delta_vs_noop_reference": float(metrics["ATE_horizon"] - ref_metrics["ATE_horizon"]),
                    "Rot": float(metrics.get("Rot_horizon", float("nan"))),
                    "FinalErr": float(metrics.get("FinalErr_horizon", float("nan"))),
                    "raw_pose_max_abs_diff_vs_reference": raw_max_abs,
                    "raw_translation_max_diff_vs_reference": raw_max_trans,
                    "timestamp_equal": bool(timestamp_equal),
                }
                row.update(_run_health_row(run_dir, parent, candidate, chunk, args.horizon, "phase0"))
                noop_rows.append(row)
                health_rows.append(row)

    if args.phase1_rollout_root and args.phase1_prefix and phase1_candidates:
        for parent in parents:
            for chunk in chunks:
                for candidate in phase1_candidates:
                    run_name = _run_name(args.phase1_prefix, parent, candidate, chunk, args.phase1_horizon)
                    run_dir = args.phase1_rollout_root / run_name
                    if not _status_done(run_dir, run_name) or not (run_dir / "01.txt").exists():
                        missing.append({"phase": "phase1", "parent": parent, "chunk": chunk, "candidate": candidate, "run_name": run_name, "reason": "missing_or_not_done"})
                        continue
                    health_rows.append(_run_health_row(run_dir, parent, candidate, chunk, args.phase1_horizon, "phase1"))

    out_dir = args.out_dir
    atlas_dir = args.root / "health_atlas"
    _write_csv(out_dir / "phase0_noop_health_rows.csv", noop_rows)
    _write_csv(out_dir / "missing_rows.csv", missing)
    _write_csv(atlas_dir / "chunk_health_timeline.csv", health_rows)
    _write_csv(atlas_dir / "read_health_by_chunk.csv", [
        {k: r.get(k) for k in (
            "phase", "parent", "chunk", "candidate", "cue_quality_rows", "cue_quality_pass_fraction",
            "mean_prior_dynamic_mass_D_gt_050", "max_prior_dynamic_mass_D_gt_050",
            "mean_prior_anchor_collision", "max_prior_anchor_collision",
            "mean_prior_fragmentation", "max_prior_fragmentation",
            "source_influence_rows", "max_context_source_skip_tokens",
            "max_removed_attention_mass_before", "context_empty_source_events",
        )}
        for r in health_rows
    ])
    _write_csv(atlas_dir / "swa_health_by_boundary.csv", [
        {k: r.get(k) for k in (
            "phase", "parent", "chunk", "candidate", "swa_health_rows",
            "max_swa_overlap_source_gate_delta", "context_empty_source_events",
        )}
        for r in health_rows
    ])
    _write_csv(atlas_dir / "ttt_health_by_chunk.csv", [
        {k: r.get(k) for k in (
            "phase", "parent", "chunk", "candidate", "ttt_health_rows",
            "semantic_role_rows", "source_influence_rows",
        )}
        for r in health_rows
    ])
    _write_csv(atlas_dir / "geometry_health_by_chunk.csv", [
        {k: r.get(k) for k in (
            "phase", "parent", "chunk", "candidate", "hmc_state_hash_rows",
            "raw_pose_max_abs_diff_vs_reference", "raw_translation_max_diff_vs_reference",
        )}
        for r in health_rows
    ])
    _write_csv(atlas_dir / "appearance_health_by_semantic.csv", [{
        "status": "explainability_missing",
        "reason": "v40 runtime rollouts do not land per-label appearance tensors; use v39/v40 offline appearance atlas separately if required",
    }])
    _write_csv(atlas_dir / "memory_path_influence_by_semantic.csv", [{
        "status": "aggregate_only",
        "semantic_memory_path_summary_rows": sum(int(r.get("source_influence_rows") or 0) for r in health_rows),
        "per_label_path_influence": "explainability_missing_when_not_landed",
    }])

    phase0_expected = len(parents) * len(chunks) * len(phase0_candidates)
    no_op_candidates = [r for r in noop_rows if r["candidate"] != args.noop_reference]
    max_ate_delta = max([abs(_float(r.get("ATE_delta_vs_noop_reference"), 0.0)) for r in no_op_candidates] or [0.0])
    max_raw_diff = max([_float(r.get("raw_pose_max_abs_diff_vs_reference"), 0.0) for r in no_op_candidates] or [0.0])
    required_streams_nonempty = all(
        int(r.get("cue_quality_rows") or 0) > 0
        and int(r.get("hook_effect_rows") or 0) > 0
        and int(r.get("hmc_state_hash_rows") or 0) > 0
        for r in noop_rows
    )
    context_empty_total = sum(_float(r.get("context_empty_source_events"), 0.0) for r in noop_rows)
    phase0_gate_pass = (
        len(noop_rows) == phase0_expected
        and not [m for m in missing if m.get("phase") == "phase0"]
        and max_ate_delta <= args.noop_ate_tol
        and max_raw_diff <= args.noop_raw_tol
        and required_streams_nonempty
        and context_empty_total == 0.0
    )
    summary = {
        "phase0_expected_rows": phase0_expected,
        "phase0_rows_done": len(noop_rows),
        "phase0_missing_rows": len([m for m in missing if m.get("phase") == "phase0"]),
        "phase0_gate_pass": bool(phase0_gate_pass),
        "max_abs_ATE_delta_vs_noop_reference": max_ate_delta,
        "max_raw_pose_abs_diff_vs_noop_reference": max_raw_diff,
        "required_health_streams_nonempty": bool(required_streams_nonempty),
        "context_empty_source_events_total": context_empty_total,
        "health_rows_total": len(health_rows),
        "cue_quality_rows_total": sum(int(r.get("cue_quality_rows") or 0) for r in health_rows),
        "source_influence_rows_total": sum(int(r.get("source_influence_rows") or 0) for r in health_rows),
        "swa_health_rows_total": sum(int(r.get("swa_health_rows") or 0) for r in health_rows),
        "ttt_health_rows_total": sum(int(r.get("ttt_health_rows") or 0) for r in health_rows),
        "attention_mass_rows_total": sum(int(r.get("attention_mass_rows") or 0) for r in health_rows),
        "appearance_evidence_level": "explainability_missing_runtime_rollout_spatial_tensors",
    }
    _write_json(out_dir / "v40_phase0_health_summary.json", summary)
    _write_json(atlas_dir / "health_flag_summary.json", summary)
    _plot_heatmap(atlas_dir / "chunk_health_timeline_heatmap.png", health_rows, "cue_quality_pass_fraction", "v40 cue quality pass fraction")
    _plot_heatmap(atlas_dir / "read_swa_ttt_health_timeline.png", health_rows, "source_influence_rows", "v40 aggregate path influence rows")
    _write_csv(out_dir / "spatial_tensor_boundary.csv", [
        {"artifact": "D_g_heatmap", "status": "explainability_missing_when_not_landed"},
        {"artifact": "source_attention_mass_heatmap", "status": "explainability_missing_when_not_landed"},
        {"artifact": "SWA_overlap_nonoverlap_source_mass_map", "status": "aggregate_only_when_hook_summary_landed"},
        {"artifact": "TTT_update_contribution_map", "status": "explainability_missing_when_not_landed"},
    ])
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
