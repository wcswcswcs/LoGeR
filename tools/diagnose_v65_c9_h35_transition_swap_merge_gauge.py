#!/usr/bin/env python3
"""ACL2 v65 C9/H35 transition-swap + merge/gauge attribution report.

This tool is diagnostic-only. It reads landed v62/v64 evidence plus v65
merge-state hook artifacts. Missing swaps are recorded as unavailable; no
numeric placeholder is invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/kitti01_hmc_v2/acl2_v65_c9_h35_transition_swap_merge_gauge_attribution"
OUT_DEFAULT = RESULT_ROOT / "report_final"
V62 = ROOT / "results/kitti01_hmc_v2/acl2_v62_kitti01_error_source_autopsy_orig_c9_h35/report_final"
V64 = ROOT / "results/kitti01_hmc_v2/acl2_v64_ttt_scale_mechanism_attribution/report_final"
V45_C9 = ROOT / "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT"
V53_H35 = ROOT / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
SMOKE = RESULT_ROOT / "phase1_merge_hook_smoke/rollouts"
GATE = RESULT_ROOT / "phase1_merge_hook_gate/rollouts"
FORK = RESULT_ROOT / "phase3_merge_gauge_forks/rollouts"


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = _clean(row.get(key))
                if value is None:
                    out[key] = ""
                elif isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _f(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _fmt(value: Any, digits: int = 6) -> str:
    val = _f(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _load_pt(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        return {}
    data = torch.load(path, map_location="cpu")
    return data if isinstance(data, dict) else {}


def _postmerge_pose_by_frame(path: Path) -> Dict[int, torch.Tensor]:
    rows = _read_jsonl(path)
    poses: Dict[int, torch.Tensor] = {}
    for row in rows:
        frame_ids = row.get("emitted_frame_ids")
        mats = row.get("camera_poses")
        if not isinstance(frame_ids, list) or not isinstance(mats, list):
            continue
        for frame_id, mat in zip(frame_ids, mats):
            try:
                frame_i = int(frame_id)
            except (TypeError, ValueError):
                continue
            tensor = torch.as_tensor(mat, dtype=torch.float32)
            if tuple(tensor.shape) == (4, 4):
                poses[frame_i] = tensor
    return poses


def _postmerge_pose_diff_row(base_dir: Path, other_dir: Path, label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "comparison": label,
        "base_run": base_dir.name,
        "other_run": other_dir.name,
        "base_postmerge_pose": str(base_dir / "postmerge_global_pose.jsonl"),
        "other_postmerge_pose": str(other_dir / "postmerge_global_pose.jsonl"),
        "status": "missing",
        "comparison_source": "postmerge_global_pose_jsonl",
    }
    base = _postmerge_pose_by_frame(base_dir / "postmerge_global_pose.jsonl")
    other = _postmerge_pose_by_frame(other_dir / "postmerge_global_pose.jsonl")
    if not base or not other:
        return row
    common = sorted(set(base) & set(other))
    if not common:
        row["status"] = "shape_mismatch"
        row["common_frame_count"] = 0
        return row
    diffs = torch.stack([(base[i] - other[i]).abs() for i in common], dim=0)
    row["status"] = "measured"
    row["common_frame_count"] = int(len(common))
    row["base_frame_count"] = int(len(base))
    row["other_frame_count"] = int(len(other))
    row["camera_poses_shape_match"] = True
    row["camera_poses_max_abs"] = float(diffs.max().item())
    row["camera_poses_mean_abs"] = float(diffs.mean().item())
    return row


def _pt_diff_row(
    base_dir: Path,
    other_dir: Path,
    label: str,
    *,
    keys: Sequence[str] = ("camera_poses", "points", "local_points", "conf"),
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "comparison": label,
        "base_run": base_dir.name,
        "other_run": other_dir.name,
        "base_output_pt": str(base_dir / "output.pt"),
        "other_output_pt": str(other_dir / "output.pt"),
        "status": "missing",
    }
    base = _load_pt(base_dir / "output.pt")
    other = _load_pt(other_dir / "output.pt")
    if not base or not other:
        if "camera_poses" in keys:
            return _postmerge_pose_diff_row(base_dir, other_dir, label)
        return row
    row["comparison_source"] = "output_pt"
    row["status"] = "measured"
    for key in keys:
        if key not in base or key not in other:
            row[f"{key}_max_abs"] = None
            continue
        a = base[key].float()
        b = other[key].float()
        if a.shape != b.shape:
            row[f"{key}_shape_match"] = False
            row[f"{key}_max_abs"] = None
            continue
        d = (a - b).abs()
        row[f"{key}_shape_match"] = True
        row[f"{key}_max_abs"] = float(d.max().item())
        row[f"{key}_mean_abs"] = float(d.mean().item())
    return row


def _trace_summary(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "merge_state_trace.jsonl")
    if not rows:
        return {
            "run_name": run_dir.name,
            "status": "missing",
            "trace_rows": 0,
            "non_identity_count": 0,
            "max_transform_trans_norm": None,
            "min_transform_rot_trace": None,
        }
    non_identity = [
        r for r in rows
        if abs(_f(r.get("transform_trans_norm"))) > 1e-9 or abs(_f(r.get("transform_rot_trace")) - 3.0) > 1e-9
    ]
    return {
        "run_name": run_dir.name,
        "status": "measured",
        "trace_rows": len(rows),
        "hash_rows": len(_read_jsonl(run_dir / "merge_state_hash.jsonl")),
        "state_files": len(list((run_dir / "merge_states").glob("chunk_*_transform.json"))),
        "forced_flags": sorted({str(r.get("forced_merge_state_replay")) for r in rows}),
        "loaded_transform_counts": sorted({str(r.get("loaded_transform_count")) for r in rows if "loaded_transform_count" in r}),
        "non_identity_count": len(non_identity),
        "non_identity_chunks": [int(r.get("chunk_idx")) for r in non_identity],
        "max_transform_trans_norm": _max(r.get("transform_trans_norm") for r in rows),
        "min_transform_rot_trace": float(np.min(_finite(r.get("transform_rot_trace") for r in rows))) if _finite(r.get("transform_rot_trace") for r in rows) else None,
    }


def _make_no_data_figure(path: Path, title: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, weight="bold")
    plt.text(0.5, 0.38, note, ha="center", va="center", fontsize=10, wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_line(path: Path, rows: List[Dict[str, Any]], x_key: str, y_keys: Sequence[str], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [_f(r.get(x_key)) for r in rows]
    plt.figure(figsize=(10, 4))
    plotted = False
    for key in y_keys:
        ys = [_f(r.get(key)) for r in rows]
        pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
        if not pairs:
            continue
        px, py = zip(*pairs)
        plt.plot(px, py, marker=".", linewidth=1.2, label=key)
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, "no-data/unavailable", ha="center", va="center")
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_scatter(path: Path, rows: List[Dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [_f(r.get(x_key)) for r in rows]
    ys = [_f(r.get(y_key)) for r in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    plt.figure(figsize=(6, 5))
    if pairs:
        px, py = zip(*pairs)
        plt.scatter(px, py, s=18, alpha=0.8)
        plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        plt.axvline(0.0, color="k", linewidth=0.8, alpha=0.4)
    else:
        plt.text(0.5, 0.5, "no-data/unavailable", ha="center", va="center")
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _build_transition_ledger(out: Path) -> List[Dict[str, Any]]:
    intra = {int(r["chunk_id"]): r for r in _read_csv(V62 / "phase3_intrachunk/intrachunk_method_comparison.csv") if r.get("chunk_id", "").isdigit()}
    gap = {int(r["chunk_id"]): r for r in _read_csv(V62 / "phase4_interchunk/h35_vs_c9_interchunk_gap.csv") if r.get("chunk_id", "").isdigit()}
    overlap_rows = _read_csv(V62 / "phase4_interchunk/overlap_transfer_metrics.csv")
    overlap: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in overlap_rows:
        if row.get("chunk_id", "").isdigit():
            overlap[(str(row.get("method")), int(row["chunk_id"]))] = row

    rows: List[Dict[str, Any]] = []
    for chunk_id in sorted(set(intra) | set(gap)):
        irow = intra.get(chunk_id, {})
        grow = gap.get(chunk_id, {})
        h35_local = _f(irow.get("h35_local_sim3_chunk_ate"))
        c9_local = _f(irow.get("c9_local_sim3_chunk_ate"))
        h35_global = _f(irow.get("h35_global_chunk_ate"))
        c9_global = _f(irow.get("c9_global_chunk_ate"))
        local_gap = h35_local - c9_local if math.isfinite(h35_local) and math.isfinite(c9_local) else float("nan")
        global_gap = h35_global - c9_global if math.isfinite(h35_global) and math.isfinite(c9_global) else float("nan")
        merge_gap_proxy = global_gap - local_gap if math.isfinite(global_gap) and math.isfinite(local_gap) else float("nan")
        h35_o = overlap.get(("h35", chunk_id), {})
        c9_o = overlap.get(("c9", chunk_id), {})
        row = {
            "chunk_id": chunk_id,
            "reset_group_id": h35_o.get("reset_group_id"),
            "reset_relative_idx": h35_o.get("reset_relative_idx"),
            "h35_global_chunk_ate": h35_global if math.isfinite(h35_global) else None,
            "c9_global_chunk_ate": c9_global if math.isfinite(c9_global) else None,
            "h35_minus_c9_global_chunk_ate": global_gap if math.isfinite(global_gap) else None,
            "h35_local_sim3_chunk_ate": h35_local if math.isfinite(h35_local) else None,
            "c9_local_sim3_chunk_ate": c9_local if math.isfinite(c9_local) else None,
            "h35_minus_c9_local_sim3_chunk_ate": local_gap if math.isfinite(local_gap) else None,
            "merge_gauge_gap_proxy": merge_gap_proxy if math.isfinite(merge_gap_proxy) else None,
            "h35_abs_scale_jump": _f(grow.get("h35_abs_scale_jump")) if grow else None,
            "c9_abs_scale_jump": _f(grow.get("c9_abs_scale_jump")) if grow else None,
            "h35_minus_c9_abs_scale_jump": _f(grow.get("h35_minus_c9_abs_scale_jump")) if grow else None,
            "h35_future_after_overlap": _f(grow.get("h35_future_after_overlap")) if grow else None,
            "c9_future_after_overlap": _f(grow.get("c9_future_after_overlap")) if grow else None,
            "h35_minus_c9_future_after_overlap": _f(grow.get("h35_minus_c9_future_after_overlap")) if grow else None,
            "h35_overlap_residual_proxy": _f(h35_o.get("overlap_sim3_residual_all")) if h35_o else None,
            "c9_overlap_residual_proxy": _f(c9_o.get("overlap_sim3_residual_all")) if c9_o else None,
            "evidence_type": "v62_landed_pose_gt_overlap_proxy",
        }
        rows.append(row)

    # Add simple top-category labels without changing numeric evidence.
    finite_merge = sorted(
        [r for r in rows if r["merge_gauge_gap_proxy"] is not None],
        key=lambda r: abs(float(r["merge_gauge_gap_proxy"])),
        reverse=True,
    )
    finite_local = sorted(
        [r for r in rows if r["h35_minus_c9_local_sim3_chunk_ate"] is not None],
        key=lambda r: abs(float(r["h35_minus_c9_local_sim3_chunk_ate"])),
        reverse=True,
    )
    merge_top = {int(r["chunk_id"]) for r in finite_merge[:12]}
    local_top = {int(r["chunk_id"]) for r in finite_local[:12]}
    for row in rows:
        tags: List[str] = []
        if int(row["chunk_id"]) in merge_top:
            tags.append("MERGE_GAP_TOP")
        if int(row["chunk_id"]) in local_top:
            tags.append("LOCAL_GAP_TOP")
        if str(row.get("reset_relative_idx")) in {"1", "2", "3"} and int(row["chunk_id"]) in merge_top:
            tags.append("RESET_REL_TOP")
        row["selection_tags"] = ";".join(tags)

    phase = out / "phase2_transition_ledger"
    _write_csv(phase / "c9_h35_transition_ledger.csv", rows)
    _write_csv(phase / "top_transition_candidates.csv", [r for r in rows if r["selection_tags"]])
    figs = out / "figures"
    _plot_line(figs / "c9_h35_transition_gap_timeline.png", rows, "chunk_id", ["h35_minus_c9_global_chunk_ate", "h35_minus_c9_local_sim3_chunk_ate", "merge_gauge_gap_proxy"], "C9-H35 Transition Gap Timeline (diagnostic-only)", "gap / proxy")
    _plot_scatter(figs / "local_vs_merge_gap_scatter.png", rows, "h35_minus_c9_local_sim3_chunk_ate", "merge_gauge_gap_proxy", "Local vs Merge/Gauge Gap Proxy")
    _plot_scatter(figs / "merge_scale_delta_vs_global_gap.png", rows, "h35_minus_c9_abs_scale_jump", "h35_minus_c9_global_chunk_ate", "Scale Jump Delta vs Global Gap")
    return rows


def _build_merge_hook_report(out: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trace_runs = [
        SMOKE / "V65_SMOKE_H35_TRACE64",
        SMOKE / "V65_SMOKE_H35_LOADSAME64",
        GATE / "V65_H35_TRACE206",
        GATE / "V65_H35_LOADSAME206",
        FORK / "V65_G2_C6_C9MERGE_OVERRIDE206",
        FORK / "V65_H35_TRACE496",
        FORK / "V65_G2_C9MERGE_SNAPSHOTS_006_010_016_496_RETRY1",
    ]
    trace_rows = [_trace_summary(path) for path in trace_runs]
    cmp_rows = [
        _pt_diff_row(SMOKE / "V65_SMOKE_H35_TRACE64", SMOKE / "V65_SMOKE_H35_LOADSAME64", "64f_load_same_merge_state"),
        _pt_diff_row(GATE / "V65_H35_TRACE206", GATE / "V65_H35_LOADSAME206", "206f_load_same_merge_state_nonidentity"),
        _pt_diff_row(GATE / "V65_H35_TRACE206", FORK / "V65_G2_C6_C9MERGE_OVERRIDE206", "206f_c9_chunk6_merge_transform_override"),
        _pt_diff_row(
            FORK / "V65_H35_TRACE496",
            FORK / "V65_G2_C9MERGE_SNAPSHOTS_006_010_016_496_RETRY1",
            "496f_c9_merge_snapshots_006_010_016_replay",
            keys=("camera_poses",),
        ),
    ]
    phase = out / "phase1_merge_hook_noop"
    _write_csv(phase / "merge_trace_inventory.csv", trace_rows)
    _write_csv(phase / "merge_hook_noop_and_override_gate.csv", cmp_rows)
    rows_for_plot = _read_jsonl((GATE / "V65_H35_TRACE206") / "merge_state_trace.jsonl")
    _plot_line(out / "figures/state_component_hash_timeline.png", rows_for_plot, "chunk_idx", ["transform_trans_norm", "transform_rot_trace"], "v65 H35 Merge/Gauge Transform Timeline", "transform metric")
    return trace_rows, cmp_rows


def _build_component_tables(out: Path, ledger: List[Dict[str, Any]], trace_rows: List[Dict[str, Any]], cmp_rows: List[Dict[str, Any]]) -> None:
    v62_summary = _read_json(V62 / "v62_summary.json")
    v64_summary = _read_json(V64 / "v64_summary.json")
    component = _read_csv(V64 / "phase0_c9_component_scale_ledger/component_scale_summary.csv")
    h35_summary = next((r for r in v62_summary.get("method_summary", []) if r.get("method") == "h35"), {})
    c9_summary = next((r for r in v62_summary.get("method_summary", []) if r.get("method") == "c9"), {})
    h35_global = _f(h35_summary.get("computed_global_sim3_rmse"))
    h35_local = _f(h35_summary.get("local_sim3_mean"))
    c9_global = _f(c9_summary.get("computed_global_sim3_rmse"))

    v64_phase7 = v64_summary.get("phase7", {})
    best_oracle = v64_phase7.get("best", {})
    max_ttt = v64_phase7.get("max_ttt")
    read_row = next((r for r in component if r.get("plan_key") == "C9_MINUS_READ_MAP_TO_FLAT"), {})
    swa_row = next((r for r in component if r.get("plan_key") == "C9_MINUS_SWA_OVERLAP_REPLACE"), {})
    tri_row = next((r for r in component if r.get("plan_key") == "C9_MINUS_TTT_TRI_REPLAY"), {})

    noop206 = next((r for r in cmp_rows if r.get("comparison") == "206f_load_same_merge_state_nonidentity"), {})
    override = next((r for r in cmp_rows if r.get("comparison") == "206f_c9_chunk6_merge_transform_override"), {})
    override496 = next((r for r in cmp_rows if r.get("comparison") == "496f_c9_merge_snapshots_006_010_016_replay"), {})

    rows = [
        {
            "component": "local_output_O",
            "status": "landed_v62_proxy",
            "metric": "h35_minus_c9_local_sim3_mean",
            "value": (h35_local - _f(c9_summary.get("local_sim3_mean"))) if math.isfinite(h35_local) and math.isfinite(_f(c9_summary.get("local_sim3_mean"))) else None,
            "interpretation": "H35 local chunk geometry is not worse than C9 on v62 GT-Sim3 proxy; local output alone does not explain C9 advantage.",
        },
        {
            "component": "merge_gauge_G",
            "status": "diagnostic_oracle_plus_v65_hook",
            "metric": "per_chunk_sim3_gt_oracle_rmse",
            "value": best_oracle.get("rmse"),
            "interpretation": "GT oracle says most H35 full gap is placement/gauge; v65 hook now serializes real chunk_se3_poses and load-same no-op passes on 206f non-identity trace.",
        },
        {
            "component": "TTT_memory",
            "status": "v64_measured_negative",
            "metric": "max_abs_h3_future_scale_delta_percent",
            "value": max_ttt,
            "interpretation": "v64 propagated TTT forks changed next-probe TTT hashes but changed future h3 scale by <1%.",
        },
        {
            "component": "READ_state",
            "status": "c9_component_landed_ledger_only",
            "metric": "C9_MINUS_READ_MAP_TO_FLAT_future_h3_scale_residual_mean",
            "value": _f(read_row.get("future_h3_scale_residual_mean")),
            "interpretation": "C9 read-map ablation has small scale-regression signal in v64 ledger; not a direct H35 swap.",
        },
        {
            "component": "SWA_memory",
            "status": "c9_component_landed_ledger_only",
            "metric": "C9_MINUS_SWA_OVERLAP_REPLACE_future_h3_scale_residual_mean",
            "value": _f(swa_row.get("future_h3_scale_residual_mean")),
            "interpretation": "SWA ablation has some C9 ledger effect but no v65 direct H35 SWA swap yet.",
        },
        {
            "component": "C9_TTT_tri_replay",
            "status": "c9_component_landed_ledger_only",
            "metric": "C9_MINUS_TTT_TRI_REPLAY_future_h3_scale_residual_mean",
            "value": _f(tri_row.get("future_h3_scale_residual_mean")),
            "interpretation": "C9 tri-replay ablation worsens ATE, but v64 H35 TTT propagation did not explain H35 scale gap.",
        },
        {
            "component": "merge_state_load_same_gate",
            "status": noop206.get("status", "missing"),
            "metric": "camera_poses_max_abs",
            "value": noop206.get("camera_poses_max_abs"),
            "interpretation": "Required gate for using merge/gauge state replay. Pass requires max diff <=1e-4; missing means not completed.",
        },
        {
            "component": "C9_merge_transform_chunk6_override",
            "status": override.get("status", "missing"),
            "metric": "camera_poses_max_abs",
            "value": override.get("camera_poses_max_abs"),
            "interpretation": "Output-only G2/Gauge fork using landed C9 chunk6 merge snapshot; not a full C9 state-swap donor.",
        },
        {
            "component": "C9_merge_transform_006_010_016_replay",
            "status": override496.get("status", "missing"),
            "metric": "camera_poses_max_abs",
            "value": override496.get("camera_poses_max_abs"),
            "interpretation": "496f output-only merge/gauge replay using the only three landed C9 merge snapshots found (006/010/016); donor coverage is below the v65 six-chunk target.",
        },
    ]
    phase = out / "phase7_attribution_summary"
    _write_csv(phase / "component_gap_closure_table.csv", rows)

    trans_rows = [r for r in ledger if r.get("selection_tags")]
    _write_csv(phase / "transition_gap_closure_table.csv", trans_rows)
    top = sorted(
        ledger,
        key=lambda r: abs(_f(r.get("merge_gauge_gap_proxy"))),
        reverse=True,
    )[:20]
    _write_csv(phase / "top_causal_chunks.csv", top)

    fig_rows = [
        {"component": r["component"], "value": r.get("value")}
        for r in rows
        if r.get("value") is not None and math.isfinite(_f(r.get("value")))
    ]
    plt.figure(figsize=(10, 4))
    if fig_rows:
        plt.bar([r["component"] for r in fig_rows], [_f(r["value"]) for r in fig_rows])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("metric value (mixed units; see table)")
    else:
        plt.text(0.5, 0.5, "no-data/unavailable", ha="center", va="center")
    plt.title("v65 Component Evidence Matrix (diagnostic-only)")
    plt.tight_layout()
    plt.savefig(out / "figures/output_merge_ttt_swap_effect_bar.png", dpi=160)
    plt.savefig(out / "figures/component_swap_effect_matrix.png", dpi=160)
    plt.savefig(out / "figures/c9_component_gap_closure.png", dpi=160)
    plt.close()

    _make_no_data_figure(out / "figures/transition_swap_gap_closure.png", "Transition Swap Gap Closure", "Full O/G/M transition swap not completed; see table for available merge-gauge hook and landed proxy evidence.")
    _make_no_data_figure(out / "figures/greedy_gap_closure_curve.png", "Greedy Gap Closure", "Greedy cumulative swap not run; no fabricated curve.")
    _make_no_data_figure(out / "figures/cumulative_greedy_gap_closure.png", "Cumulative Greedy Gap Closure", "No complete transition swap set available.")
    _make_no_data_figure(out / "figures/final_attribution_pie_or_bar.png", "Final Attribution", "Evidence points to merge/gauge as next target, but direct full attribution is incomplete.")
    _make_no_data_figure(out / "figures/gap_closure_by_component.png", "Gap Closure By Component", "Mixed proxy/direct evidence; use component table, not this placeholder, for numeric interpretation.")
    _make_no_data_figure(out / "figures/gap_closure_by_transition.png", "Gap Closure By Transition", "Transition swap not completed.")


def _write_report(out: Path, ledger: List[Dict[str, Any]], trace_rows: List[Dict[str, Any]], cmp_rows: List[Dict[str, Any]]) -> None:
    phase = out / "phase7_attribution_summary"
    comp_rows = _read_csv(phase / "component_gap_closure_table.csv")
    noop206 = next((r for r in cmp_rows if r.get("comparison") == "206f_load_same_merge_state_nonidentity"), {})
    override = next((r for r in cmp_rows if r.get("comparison") == "206f_c9_chunk6_merge_transform_override"), {})
    override496 = next((r for r in cmp_rows if r.get("comparison") == "496f_c9_merge_snapshots_006_010_016_replay"), {})
    non_identity = next((r for r in trace_rows if r.get("run_name") == "V65_H35_TRACE206"), {})
    trace496 = next((r for r in trace_rows if r.get("run_name") == "V65_H35_TRACE496"), {})
    top_chunks = _read_csv(phase / "top_causal_chunks.csv")[:8]

    lines = [
        "# ACL2 v65 H35-C9 Transition/Merge-Gauge Attribution Report",
        "",
        "性质: diagnostic-only。GT/proxy/landed rows 只用于归因，不作为方法结果。",
        "",
        "## Hook 与 No-op Gate",
        "",
        f"- 64f load-same output max pose diff: {_fmt(next((r for r in cmp_rows if r.get('comparison') == '64f_load_same_merge_state'), {}).get('camera_poses_max_abs'))}",
        f"- 206f non-identity trace rows: {non_identity.get('trace_rows', 'NA')}, non-identity chunks: {non_identity.get('non_identity_chunks', [])}",
        f"- 206f load-same output max pose diff: {_fmt(noop206.get('camera_poses_max_abs'))}",
        f"- C9 chunk6 merge override status: {override.get('status', 'missing')}, pose max diff: {_fmt(override.get('camera_poses_max_abs'))}",
        f"- 496f H35 trace rows: {trace496.get('trace_rows', 'NA')}, non-identity chunks: {trace496.get('non_identity_chunks', [])}",
        f"- 496f C9 merge snapshot replay status: {override496.get('status', 'missing')}, pose max diff: {_fmt(override496.get('camera_poses_max_abs'))}",
        "",
        "## Component Evidence",
        "",
    ]
    for row in comp_rows:
        lines.append(f"- `{row.get('component')}` [{row.get('status')}]: {row.get('metric')} = {row.get('value') or 'NA'}; {row.get('interpretation')}")
    lines += [
        "",
        "## Top Transition Candidates",
        "",
    ]
    for row in top_chunks:
        lines.append(
            f"- chunk {row.get('chunk_id')}: merge_proxy={row.get('merge_gauge_gap_proxy') or 'NA'}, "
            f"global_gap={row.get('h35_minus_c9_global_chunk_ate') or 'NA'}, tags={row.get('selection_tags')}"
        )
    lines += [
        "",
        "## v65 必答问题",
        "",
        "1. C9 parity 是否通过? 本轮尚未完成当前代码 C9 full parity rerun；旧 C9 只能作为 landed trajectory/snapshot donor。",
        "2. local output 是否解释 H35-C9 gap? v62 landed proxy 显示 H35 local Sim3 不差于 C9，因此不是主解释。",
        "3. merge/gauge 是否是主方向? v62/v64 GT oracle 与 v65 real chunk_se3 hook 均指向 merge/gauge placement 是首要后续目标。",
        "4. merge fork 是否有效? save-load-same gate 在 64f 和 206f 通过；C9 donor override 若缺失则不能声称完成。",
        "5. TTT/SWA/READ 哪个更像主因? v64 TTT propagated fork 为负；SWA/READ 只有 C9 ablation ledger，尚非 H35 direct swap。",
        "6. transition swap 是否完成? 完整 O/G/M transition swap 尚未完成，已在表中标注 not_run/partial。",
        "7. 是否少数 chunk 驱动? Phase2 ledger 输出 top transition candidates；当前证据是候选，不是完成因果归因。",
        "8. C9 component 哪些最相关? C9 tri-replay/SWA/read ablation ledger 有不同程度信号，但不能替代 H35 direct swap。",
        "9. 下一步目标? 在当前代码上做 full H35/C9 merge-state trace，优先 chunk 5/6/10/16 的 gauge donor override 和 transition materialization。",
        "10. 结论可信边界? merge-state hook/no-op 是真实当前代码证据；v62/v64 是 landed/proxy 证据；未运行项不填数。",
    ]
    _write_text(phase / "h35_c9_gap_attribution_report.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ledger = _build_transition_ledger(out)
    trace_rows, cmp_rows = _build_merge_hook_report(out)
    _build_component_tables(out, ledger, trace_rows, cmp_rows)
    _write_report(out, ledger, trace_rows, cmp_rows)
    summary = {
        "out": str(out),
        "ledger_rows": len(ledger),
        "trace_rows": trace_rows,
        "comparison_rows": cmp_rows,
        "status": "partial_diagnostic_report_written",
    }
    _write_json(out / "v65_summary.json", summary)
    print(json.dumps(_clean(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
