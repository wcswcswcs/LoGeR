#!/usr/bin/env python3
"""Summarize cached Stage C masklets for ACL2 semantic-prior audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

import torch


GROUP_NAMES = {
    0: "STRUCTURE_ANCHOR",
    1: "STATIC_THING",
    2: "MOVABLE_THING",
    3: "LOW_VALUE_STUFF",
    4: "UNCERTAIN_REGION",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_manifest(masklet: Dict[str, Any], masklet_path: Path) -> Dict[str, Any]:
    manifest = masklet.get("manifest")
    if isinstance(manifest, dict):
        return manifest
    manifest_path = masklet_path.with_name("manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _focus_overlap(start: int, end: int, focus_start: int, focus_end: int) -> float:
    denom = max(1, end - start)
    inter = max(0, min(end, focus_end) - max(start, focus_start))
    return float(inter) / float(denom)


def _chunk_row(masklet_path: Path, focus_start: int, focus_end: int) -> Dict[str, Any]:
    data = torch.load(masklet_path, map_location="cpu")
    manifest = _parse_manifest(data, masklet_path)
    start = int(manifest.get("start_frame", -1))
    end = int(manifest.get("end_frame", start + int(data.get("num_frames", 0))))
    chunk_idx = int(manifest.get("chunk_idx", -1))
    J = int(data.get("num_masklets", 0))
    T = int(data.get("num_frames", 0))
    H = int(data.get("frame_height", 0))
    W = int(data.get("frame_width", 0))
    area = float(max(1, H * W))
    labels = list(data.get("L_sem", []))
    groups = data.get("G_sem", torch.empty(0, dtype=torch.long)).long()
    visible = data.get("V_mask", torch.empty((0, T), dtype=torch.bool)).bool()
    quality = data.get("Q_mask", torch.empty((0, T), dtype=torch.float32)).float()
    masks = data.get("M_mask", torch.empty((0, T, H, W), dtype=torch.float32))

    row: Dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "start_frame": start,
        "end_frame": end,
        "focus_overlap": _focus_overlap(start, end, focus_start, focus_end),
        "num_masklets": J,
        "num_frames": T,
        "height": H,
        "width": W,
        "masklet_labels": ";".join(labels),
    }

    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        row[f"label_count::{label}"] = int(count)

    group_counts = Counter(int(g.item()) for g in groups)
    for gid, name in GROUP_NAMES.items():
        row[f"masklet_count_{name}"] = int(group_counts.get(gid, 0))

    if J <= 0 or T <= 0 or H <= 0 or W <= 0:
        row.update({
            "coverage_mean": 0.0,
            "coverage_max": 0.0,
            "coverage_min": 0.0,
            "coverage_p10": 0.0,
            "coverage_p90": 0.0,
            "visible_masklet_frame_frac": 0.0,
            "trust_visible_mean": 0.0,
            "trust_all_mean": 0.0,
        })
        for name in GROUP_NAMES.values():
            row[f"coverage_{name}"] = 0.0
            row[f"trust_visible_{name}"] = 0.0
        return row

    mask_bool = masks > 0.5
    union = mask_bool.any(dim=0)
    cov_frame = union.float().flatten(1).mean(dim=1)
    cov_sorted = torch.sort(cov_frame).values
    p10_idx = min(int(0.10 * max(0, T - 1)), T - 1)
    p90_idx = min(int(0.90 * max(0, T - 1)), T - 1)
    row.update({
        "coverage_mean": float(cov_frame.mean().item()),
        "coverage_max": float(cov_frame.max().item()),
        "coverage_min": float(cov_frame.min().item()),
        "coverage_p10": float(cov_sorted[p10_idx].item()),
        "coverage_p90": float(cov_sorted[p90_idx].item()),
        "visible_masklet_frame_frac": float(visible.float().mean().item()) if visible.numel() else 0.0,
        "trust_visible_mean": float(quality[visible].mean().item()) if visible.any() else 0.0,
        "trust_all_mean": float((visible.float() * quality).mean().item()) if visible.numel() else 0.0,
    })

    for gid, name in GROUP_NAMES.items():
        idx = groups == gid
        if bool(idx.any().item()):
            group_union = mask_bool[idx].any(dim=0)
            row[f"coverage_{name}"] = float(group_union.float().sum().item() / (max(1, T) * area))
            group_visible = visible[idx]
            group_quality = quality[idx]
            row[f"trust_visible_{name}"] = (
                float(group_quality[group_visible].mean().item()) if group_visible.any() else 0.0
            )
        else:
            row[f"coverage_{name}"] = 0.0
            row[f"trust_visible_{name}"] = 0.0
    return row


def _read_prior_debug(path: Optional[Path]) -> Dict[int, Dict[str, float]]:
    if not path or not path.exists():
        return {}
    numeric_by_chunk: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            chunk = int(rec.get("chunk_idx", -1))
            if chunk < 0:
                continue
            for key in (
                "mean_A_tok",
                "min_A_tok",
                "max_A_tok",
                "std_A_tok",
                "prior_debug_mean_a_pix",
                "prior_debug_mean_r_mask",
                "prior_debug_mean_elig",
                "prior_debug_B_chunk_geo",
                "mean_prior_flat",
                "min_prior_flat",
                "max_prior_flat",
            ):
                if key in rec:
                    numeric_by_chunk[chunk][key].append(_safe_float(rec[key]))
    out: Dict[int, Dict[str, float]] = {}
    for chunk, cols in numeric_by_chunk.items():
        out[chunk] = {f"prior_avg_{k}": mean(v) for k, v in cols.items() if v}
    return out


def _read_kitti_metrics(run_dir: Optional[Path]) -> Dict[str, float]:
    if not run_dir:
        return {}
    out: Dict[str, float] = {}
    ate_path = run_dir / "results_sim3" / "results_ate.txt"
    rpe_path = run_dir / "results_sim3" / "results_rpe.txt"
    if ate_path.exists():
        for line in ate_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Average:"):
                parts = line.split()
                if len(parts) >= 3:
                    out["ate_rmse"] = _safe_float(parts[1])
                    out["rot_rmse"] = _safe_float(parts[2])
    if rpe_path.exists():
        for line in rpe_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Average:"):
                parts = line.split()
                if len(parts) >= 3:
                    out["rpe_t"] = _safe_float(parts[1])
                    out["rpe_r"] = _safe_float(parts[2])
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _weighted_mean(rows: Iterable[Dict[str, Any]], key: str, weight_key: Optional[str] = None) -> float:
    vals = []
    weights = []
    for row in rows:
        vals.append(_safe_float(row.get(key, 0.0)))
        weights.append(_safe_float(row.get(weight_key, 1.0)) if weight_key else 1.0)
    denom = sum(weights)
    if not vals or denom <= 0:
        return 0.0
    return sum(v * w for v, w in zip(vals, weights)) / denom


def _write_summary(path: Path, rows: List[Dict[str, Any]], metrics: Dict[str, float], focus_start: int, focus_end: int) -> None:
    covered_rows = [r for r in rows if _safe_float(r.get("coverage_mean")) > 0.0]
    focus_rows = [r for r in rows if _safe_float(r.get("focus_overlap")) > 0.0]
    mean_cov = _weighted_mean(rows, "coverage_mean")
    focus_cov = _weighted_mean(focus_rows, "coverage_mean", "focus_overlap") if focus_rows else 0.0
    chunks_with_masks = len(covered_rows)
    med_cov = median([_safe_float(r.get("coverage_mean")) for r in rows]) if rows else 0.0
    max_cov = max([_safe_float(r.get("coverage_max")) for r in rows], default=0.0)
    mean_masklets = _weighted_mean(rows, "num_masklets")
    group_lines = []
    for gid, name in GROUP_NAMES.items():
        group_lines.append(
            f"| `{name}` | "
            f"{sum(int(r.get(f'masklet_count_{name}', 0)) for r in rows)} | "
            f"{_fmt(_weighted_mean(rows, f'coverage_{name}'))} | "
            f"{_fmt(_weighted_mean(focus_rows, f'coverage_{name}', 'focus_overlap') if focus_rows else 0.0)} |"
        )

    gate = "FAIL"
    gate_reason = "mask coverage too sparse for semantic-write promotion"
    if mean_cov >= 0.70 and focus_cov >= 0.70:
        gate = "PASS"
        gate_reason = "coverage is high enough for semantic-write promotion"
    elif mean_cov >= 0.30 or focus_cov >= 0.30:
        gate = "WEAK"
        gate_reason = "partial coverage; semantic-write only as diagnostic"

    lines = [
        "# ACL2 v6 Phase 2 Semantic Cache Audit",
        "",
        f"Focus segment: `[{focus_start},{focus_end})`",
        "",
        "## KITTI Metrics",
        "",
    ]
    if metrics:
        lines.append("| ATE RMSE | Rot RMSE | RPE t | RPE r |")
        lines.append("|---:|---:|---:|---:|")
        lines.append(
            f"| {_fmt(metrics.get('ate_rmse', 0.0), 4)} | {_fmt(metrics.get('rot_rmse', 0.0), 4)} | "
            f"{_fmt(metrics.get('rpe_t', 0.0), 4)} | {_fmt(metrics.get('rpe_r', 0.0), 4)} |"
        )
    else:
        lines.append("No KITTI metrics found for this run directory.")
    lines.extend([
        "",
        "## Coverage",
        "",
        "| Chunks | Chunks With Masklets | Mean Masklets / Chunk | Mean Coverage | Median Coverage | Focus Coverage | Max Frame Coverage | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| {len(rows)} | {chunks_with_masks} | {_fmt(mean_masklets, 2)} | {_fmt(mean_cov)} | {_fmt(med_cov)} | {_fmt(focus_cov)} | {_fmt(max_cov)} | `{gate}` |",
        "",
        f"Gate reason: {gate_reason}.",
        "",
        "## Semantic Group Mass",
        "",
        "| Group | Masklet Count | Mean Coverage | Focus Coverage |",
        "|---|---:|---:|---:|",
        *group_lines,
        "",
        "## Focus Chunks",
        "",
        "| Chunk | Frame Range | Focus Overlap | Masklets | Coverage | Structure | Low Stuff | Movable | Labels |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in sorted(focus_rows, key=lambda x: int(x.get("chunk_idx", -1))):
        lines.append(
            f"| {int(r.get('chunk_idx', -1))} | `[{int(r.get('start_frame', -1))},{int(r.get('end_frame', -1))})` | "
            f"{_fmt(_safe_float(r.get('focus_overlap')))} | {int(r.get('num_masklets', 0))} | "
            f"{_fmt(_safe_float(r.get('coverage_mean')))} | "
            f"{_fmt(_safe_float(r.get('coverage_STRUCTURE_ANCHOR')))} | "
            f"{_fmt(_safe_float(r.get('coverage_LOW_VALUE_STUFF')))} | "
            f"{_fmt(_safe_float(r.get('coverage_MOVABLE_THING')))} | "
            f"`{str(r.get('masklet_labels', ''))[:80]}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--focus-start", type=int, default=200)
    parser.add_argument("--focus-end", type=int, default=300)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None
    out_dir = Path(args.out_dir)
    masklet_paths = sorted(cache_dir.glob("chunk_*/masklet.pt"))
    if not masklet_paths:
        raise SystemExit(f"No masklet cache files found under {cache_dir}")

    rows = [_chunk_row(p, args.focus_start, args.focus_end) for p in masklet_paths]
    prior = _read_prior_debug((run_dir / "prior_debug.jsonl") if run_dir else None)
    for row in rows:
        row.update(prior.get(int(row.get("chunk_idx", -1)), {}))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_chunk_semantic.csv", rows)
    focus_rows = [r for r in rows if _safe_float(r.get("focus_overlap")) > 0.0]
    _write_csv(out_dir / "key_chunks_200_300_semantic.csv", focus_rows)

    label_rows = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("label_count::"):
                label_rows.append({
                    "chunk_idx": row["chunk_idx"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "label": key.split("::", 1)[1],
                    "count": int(value),
                })
    _write_csv(out_dir / "label_counts_by_chunk.csv", label_rows)

    metrics = _read_kitti_metrics(run_dir)
    _write_summary(out_dir / "semantic_summary.md", rows, metrics, args.focus_start, args.focus_end)
    print(f"Wrote semantic audit to {out_dir}")


if __name__ == "__main__":
    main()
