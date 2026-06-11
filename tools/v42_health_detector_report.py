#!/usr/bin/env python3
"""Build v42 full-sequence health detector artifacts from landed no-op run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import _align_metrics, _load_kitti_gt, _load_tum_prediction  # noqa: E402


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_clean(row), ensure_ascii=False, sort_keys=True) + "\n")


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
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _robust_z(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(1.4826 * mad, 1e-6)
    return [float((v - med) / scale) for v in values]


def _rolling_ate_by_start(run_dir: Path, width: int, gt_path: Path) -> Dict[int, float]:
    pred = run_dir / "01.txt"
    if not pred.exists():
        return {}
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    frames, poses, _ = _load_tum_prediction(pred, gt_pos.shape[0])
    aligned, _metrics = _align_metrics(frames.astype(np.int64), poses, gt_poses, gt_pos)
    pos = aligned[:, :3, 3]
    out: Dict[int, float] = {}
    for start in range(int(frames.min()), int(frames.max()) - int(width) + 2):
        end = start + int(width)
        mask = (frames >= start) & (frames < end)
        if int(mask.sum()) < 3:
            continue
        gt_subset = gt_pos[frames[mask].astype(np.int64)]
        diff = pos[mask] - gt_subset
        out[int(start)] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return out


def _make_plots(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    chunks = [int(r["chunk"]) for r in rows]
    if not chunks:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for key, label in [
        ("H_read", "H_read"),
        ("H_swa", "H_swa"),
        ("H_ttt", "H_ttt"),
        ("H_app", "H_app"),
    ]:
        plt.plot(chunks, [float(r[key]) for r in rows], marker="o", label=label)
    plt.plot(chunks, [float(r["selected"]) for r in rows], marker="s", label="selected")
    plt.xlabel("chunk")
    plt.ylabel("robust score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "chunk_health_timeline.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    bottom = np.zeros(len(chunks), dtype=np.float64)
    for key, label in [
        ("z_highd_source_mass", "highD"),
        ("z_semantic_anomaly_mass", "semantic-z"),
        ("z_source_influence_proxy", "source influence"),
        ("z_app_proxy", "appearance proxy"),
    ]:
        vals = np.asarray([max(float(r[key]), 0.0) for r in rows], dtype=np.float64)
        plt.bar(chunks, vals, bottom=bottom, label=label)
        bottom += vals
    plt.xlabel("chunk")
    plt.ylabel("positive robust components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "health_component_stackplot.png", dpi=160)
    plt.close()


def _write_md(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v42 Bad Chunk Report",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "phase1_gate_pass",
        "selected_bad_chunks",
        "selected_bad_chunk_ratio",
        "selection_uses_ATE",
        "selection_uses_fixed_chunk_or_segment",
        "hmc_rows",
        "health_chunk_count",
        "detector_rule",
        "top_rolling100_bad_chunk_diagnostic",
    ]:
        lines.append(f"- `{key}`: `{_clean(summary.get(key))}`")
    lines.extend([
        "",
        "## Chunks",
        "",
        "| Chunk | Selected | H_read | H_swa | H_ttt | H_app | Evidence count | Rolling100 diagnostic |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {int(row['chunk'])} | {int(row['selected'])} | {float(row['H_read']):.4f} | {float(row['H_swa']):.4f} | {float(row['H_ttt']):.4f} | {float(row['H_app']):.4f} | {int(row['read_evidence_count'])} | {_fmt(row.get('rolling100_ate_diagnostic'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    args = parser.parse_args()

    hmc_rows = _read_jsonl(args.reference_run / "hmc_state_hash.jsonl")
    by_chunk: Dict[int, Dict[str, Any]] = {}
    for row in hmc_rows:
        if "chunk_idx" not in row:
            continue
        by_chunk[int(row["chunk_idx"])] = row

    chunks = sorted(by_chunk)
    highd = [_f(by_chunk[c], "prior_dynamic_mass_D_gt_050") + 0.25 * _f(by_chunk[c], "prior_q90_D_patch") for c in chunks]
    semz = [_f(by_chunk[c], "prior_v32_semantic_z_high_mass") + 0.25 * _f(by_chunk[c], "prior_v32_semantic_d_q90") for c in chunks]
    source = [_f(by_chunk[c], "prior_hmc_write_selected_mass") + 0.25 * _f(by_chunk[c], "prior_mean_D_tok") for c in chunks]
    app = [_f(by_chunk[c], "prior_fragmentation") + 0.25 * (1.0 - _f(by_chunk[c], "prior_cue_quality_pass", 1.0)) for c in chunks]
    static = [_f(by_chunk[c], "prior_protect_patch_mass") + 0.001 * _f(by_chunk[c], "prior_protect_anchor_count") for c in chunks]
    swa = [_f(by_chunk[c], "pass1_pass2_pose_t_mean") + _f(by_chunk[c], "pass1_pass2_pose_r_deg_mean") for c in chunks]
    ttt = [_f(by_chunk[c], "memory_ttt_mean_rel_diff") + _f(by_chunk[c], "prior_ttt_write_mean") for c in chunks]

    z_highd = _robust_z(highd)
    z_semz = _robust_z(semz)
    z_source = _robust_z(source)
    z_app = _robust_z(app)
    z_static = _robust_z(static)
    z_swa = _robust_z(swa)
    z_ttt = _robust_z(ttt)

    rows: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        evidence_count = sum(1 for value in (z_highd[idx], z_semz[idx], z_source[idx], z_app[idx]) if value > 0.5)
        h_read = z_highd[idx] + z_semz[idx] + z_source[idx] + z_app[idx] - max(z_static[idx], 0.0)
        rows.append({
            "chunk": int(chunk),
            "highd_source_mass_proxy": highd[idx],
            "semantic_anomaly_mass_proxy": semz[idx],
            "source_influence_proxy": source[idx],
            "appearance_proxy": app[idx],
            "static_anchor_proxy": static[idx],
            "z_highd_source_mass": z_highd[idx],
            "z_semantic_anomaly_mass": z_semz[idx],
            "z_source_influence_proxy": z_source[idx],
            "z_app_proxy": z_app[idx],
            "z_static_anchor_proxy": z_static[idx],
            "H_read": float(h_read),
            "H_swa": float(z_swa[idx]),
            "H_ttt": float(z_ttt[idx]),
            "H_app": float(z_app[idx]),
            "read_evidence_count": int(evidence_count),
            "selected": 0,
        })

    max_selected = max(1, int(math.floor(0.20 * max(len(rows), 1))))
    eligible = [r for r in rows if int(r["read_evidence_count"]) >= 2 and float(r["H_read"]) > 0.0]
    if not eligible and rows:
        eligible = [max(rows, key=lambda r: float(r["H_read"]))]
    selected = sorted(eligible, key=lambda r: float(r["H_read"]), reverse=True)[:max_selected]
    selected_chunks = sorted(int(r["chunk"]) for r in selected)
    for row in rows:
        row["selected"] = int(int(row["chunk"]) in selected_chunks)

    rolling = _rolling_ate_by_start(args.reference_run, 100, Path(args.gt))
    if rolling:
        for row in rows:
            chunk = int(row["chunk"])
            # Global chunk start frames in this pipeline advance by 29 after the
            # first 32-frame chunk. This is diagnostic only, not detector input.
            start = 0 if chunk == 0 else 32 + (chunk - 1) * 29
            vals = [v for s, v in rolling.items() if start <= int(s) < start + 32]
            row["rolling100_ate_diagnostic"] = float(max(vals)) if vals else None
        top_start = max(rolling, key=lambda k: rolling[k])
        top_chunk = 0 if top_start < 32 else int((top_start - 32) // 29 + 1)
    else:
        top_chunk = None

    ratio = float(len(selected_chunks) / len(rows)) if rows else 0.0
    summary = {
        "phase1_gate_pass": bool(0.0 < ratio <= 0.20 and rows),
        "selected_bad_chunks": selected_chunks,
        "selected_bad_chunk_ratio": ratio,
        "selection_uses_ATE": False,
        "selection_uses_fixed_chunk_or_segment": False,
        "hmc_rows": len(hmc_rows),
        "health_chunk_count": len(rows),
        "detector_rule": "top_20pct_read_health_with_at_least_two_positive_read_evidence_components",
        "top_rolling100_bad_chunk_diagnostic": top_chunk,
        "rolling100_used_for_selection": False,
        "explainability_boundary": "aggregate_health_only_spatial_maps_not_required_for_phase1_selection",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "chunk_health_table.csv", rows)
    _write_jsonl(args.out_dir / "chunk_health_flags.jsonl", rows)
    _write_csv(args.out_dir / "health_component_by_chunk.csv", rows)
    _write_json(args.out_dir / "selected_bad_chunks.json", {
        "selected_bad_chunks": selected_chunks,
        "selection_uses_ATE": False,
        "selection_uses_fixed_chunk_or_segment": False,
    })
    _write_csv(args.out_dir / "health_vs_rolling_window_diagnostic.csv", rows)
    _write_json(args.out_dir / "v42_health_detector_summary.json", summary)
    _write_md(args.out_dir / "bad_chunk_report.md", summary, rows)
    _make_plots(args.out_dir, rows)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

