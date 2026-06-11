#!/usr/bin/env python3
"""Passive semantic attribution aggregation for ACL2 v24."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


GROUP_NAMES = {
    "0": "STRUCTURE_ANCHOR",
    "1": "STATIC_THING",
    "2": "MOVABLE_THING",
    "3": "LOW_VALUE_STUFF",
    "4": "UNCERTAIN_REGION",
}

DEFAULT_STAGE_C_AUDIT_DIR = (
    "results/kitti01_hmc_v2/"
    "acl2_v6_stage_c_cache_mask2former_cityscapes_full/semantic_audit"
)


def _jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _candidate_from_run(run_name: str) -> str:
    if "PASSIVE_DEBUG_ONLY" in run_name:
        return "PASSIVE_DEBUG_ONLY"
    if "K1_H9" in run_name:
        return "K1_H9"
    return run_name


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


def _load_stage_c_fine_label_rows(audit_dir: Path, chunks: Set[int]) -> List[Dict[str, object]]:
    """Load predicted Mask2Former fine-label coverage from the existing Stage C audit.

    v24 runtime memory-path summaries are intentionally coarse-group keyed today.
    This audit-side evidence is used only to separate "no fine labels exist" from
    "fine labels exist but runtime policy collapsed them to coarse groups".
    """
    path = audit_dir / "label_counts_by_chunk.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chunk = _safe_int(row.get("chunk_idx"))
            if chunks and chunk not in chunks:
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            rows.append(
                {
                    "chunk_idx": chunk,
                    "start_frame": row.get("start_frame"),
                    "end_frame": row.get("end_frame"),
                    "fine_label": label,
                    "count": row.get("count", 0),
                    "source": str(path),
                    "is_gt_semantic": False,
                }
            )
    return rows


def _plot_heatmap(out_png: Path, rows: List[Dict[str, object]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return False
    groups = sorted({str(row["group_id"]) for row in rows})
    paths = ["frame", "global", "swa", "ttt", "lifecycle"]
    if not groups:
        return False
    matrix = np.zeros((len(groups), len(paths)), dtype=float)
    for row in rows:
        gi = groups.index(str(row["group_id"]))
        for pi, path in enumerate(paths):
            if bool(row.get(f"{path}_consumed")):
                matrix[gi, pi] += _safe_float(row.get("token_count"))
    fig, ax = plt.subplots(figsize=(8, max(3, len(groups) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(paths)), paths, rotation=30, ha="right")
    ax.set_yticks(range(len(groups)), [GROUP_NAMES.get(g, g) for g in groups])
    ax.set_title("v24 semantic group memory path heatmap")
    fig.colorbar(im, ax=ax, label="token count mass")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stage-c-audit-dir", default=DEFAULT_STAGE_C_AUDIT_DIR)
    args = parser.parse_args()

    results = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results / "phase1_passive_semantic_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = results / "rollouts"

    coverage_rows: List[Dict[str, object]] = []
    heatmap_rows: List[Dict[str, object]] = []
    context_rows: List[Dict[str, object]] = []
    runs_seen = set()
    groups_seen = set()
    consumed_paths = defaultdict(int)
    chunks_seen: Set[int] = set()

    for mem_path in sorted(rollouts.glob("*/semantic_memory_path_summary.jsonl")) if rollouts.exists() else []:
        if ".INVALID" in str(mem_path):
            continue
        run = mem_path.parent.name
        for row in _jsonl(mem_path):
            metrics = row.get("semantic_group_role_metrics") or {}
            if not isinstance(metrics, dict):
                continue
            runs_seen.add(run)
            chunk_idx = _safe_int(row.get("chunk_idx"))
            if chunk_idx >= 0:
                chunks_seen.add(chunk_idx)
            path_flags = {
                "frame": bool(row.get("frame_semantic_source_consumed")),
                "global": bool(row.get("chunk_global_semantic_source_consumed")),
                "swa": bool(row.get("swa_semantic_source_consumed")),
                "ttt": bool(row.get("ttt_semantic_role_consumed")),
                "lifecycle": bool(row.get("lifecycle_semantic_role_consumed")),
            }
            for name, flag in path_flags.items():
                if flag:
                    consumed_paths[name] += 1
            for group_id, group_metrics in metrics.items():
                if not isinstance(group_metrics, dict):
                    continue
                groups_seen.add(str(group_id))
                base = {
                    "run_name": run,
                    "candidate_id": _candidate_from_run(run),
                    "chunk_idx": row.get("chunk_idx"),
                    "semantic_role_policy": row.get("semantic_role_policy"),
                    "semantic_memory_paths": row.get("semantic_memory_paths"),
                    "group_id": group_id,
                    "group_name": GROUP_NAMES.get(str(group_id), str(group_id)),
                    "token_count": group_metrics.get("token_count", 0),
                    "D_mean": group_metrics.get("D_mean", 0.0),
                    "D_p90": group_metrics.get("D_p90", 0.0),
                    "Q_mean": group_metrics.get("Q_mean", 0.0),
                    "V_mean": group_metrics.get("V_mean", 0.0),
                    "role_counts_json": json.dumps(group_metrics.get("role_counts") or {}, ensure_ascii=False, sort_keys=True),
                }
                coverage_rows.append(base)
                heatmap_rows.append({**base, **{f"{k}_consumed": v for k, v in path_flags.items()}})

    for ctx_path in sorted(rollouts.glob("*/context_skip_summary.jsonl")) if rollouts.exists() else []:
        if ".INVALID" in str(ctx_path):
            continue
        run = ctx_path.parent.name
        for row in _jsonl(ctx_path):
            chunk_idx = _safe_int(row.get("chunk_idx"))
            if chunk_idx >= 0:
                chunks_seen.add(chunk_idx)
            context_rows.append(
                {
                    "run_name": run,
                    "candidate_id": _candidate_from_run(run),
                    "chunk_idx": row.get("chunk_idx"),
                    "path": row.get("path"),
                    "context_source_skip_requested": row.get("context_source_skip_requested"),
                    "context_source_skip_impl": row.get("context_source_skip_impl"),
                    "context_source_skip_scope": row.get("context_source_skip_scope"),
                    "context_source_skip_mask": row.get("context_source_skip_mask"),
                    "mean_context_source_keep_ratio": row.get("mean_context_source_keep_ratio"),
                    "num_context_source_skip_applied": row.get("num_context_source_skip_applied"),
                    "num_context_empty_source_events": row.get("num_context_empty_source_events"),
                    "max_context_source_skip_tokens": row.get("max_context_source_skip_tokens"),
                }
            )

    _write_csv(out_dir / "semantic_group_coverage_by_run.csv", coverage_rows)
    _write_csv(out_dir / "semantic_group_memory_path_heatmap.csv", heatmap_rows)
    _write_csv(out_dir / "context_skip_by_run.csv", context_rows)

    fine_label_rows = _load_stage_c_fine_label_rows(Path(args.stage_c_audit_dir).resolve(), chunks_seen)
    _write_csv(out_dir / "fine_label_coverage_by_chunk.csv", fine_label_rows)

    plotted = _plot_heatmap(out_dir / "semantic_group_memory_path_heatmap.png", heatmap_rows)
    major_group_count = len(groups_seen)
    fine_labels_seen = sorted({str(row["fine_label"]) for row in fine_label_rows})
    fine_label_count = len(fine_labels_seen)
    source_or_write_nonempty = bool(coverage_rows)
    role_difference = any(
        str(row.get("role_counts_json") or "{}") not in {"{}", "null"}
        for row in coverage_rows
    )
    coarse_group_diversity_gate_pass = major_group_count >= 4
    fine_label_diversity_gate_pass = fine_label_count >= 4
    semantic_diversity_gate_pass = bool(coarse_group_diversity_gate_pass or fine_label_diversity_gate_pass)
    summary = {
        "num_runs_with_semantic_memory_metrics": len(runs_seen),
        "major_semantic_group_count": major_group_count,
        "groups_seen": sorted(groups_seen),
        "coarse_group_diversity_gate_pass": coarse_group_diversity_gate_pass,
        "fine_label_count": fine_label_count,
        "fine_labels_seen": fine_labels_seen,
        "fine_label_source": str((Path(args.stage_c_audit_dir).resolve() / "label_counts_by_chunk.csv")),
        "fine_label_is_gt_semantic": False,
        "fine_label_diversity_gate_pass": fine_label_diversity_gate_pass,
        "semantic_diversity_gate_pass": semantic_diversity_gate_pass,
        "runtime_fine_role_policy_available": False,
        "runtime_fine_role_policy_note": (
            "Stage C predicted fine labels are present in the cache audit, but the current runtime "
            "semantic memory path metrics are coarse-group keyed. Fine-named v24 candidates are "
            "therefore coarse-fallback diagnostics, not fine-runtime-policy proof."
        ),
        "consumed_path_counts": dict(consumed_paths),
        "num_context_rows": len(context_rows),
        "context_skip_empty_source_event_runs": sorted(
            {
                str(row["run_name"])
                for row in context_rows
                if _safe_float(row.get("num_context_empty_source_events")) > 0.0
            }
        ),
        "phase1_gate_pass": bool(
            semantic_diversity_gate_pass
            and source_or_write_nonempty
            and role_difference
            and len(consumed_paths) >= 2
        ),
        "aggregate_heatmap_png_written": plotted,
        "map_level_visualization_note": (
            "Per-pixel RGB/semantic/D_g/TTT maps are not present in the current landed debug artifacts; "
            "this script outputs aggregate CSV/PNG heatmaps only and does not fabricate maps."
        ),
    }
    (out_dir / "passive_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ACL2 v24 Passive Semantic Attribution",
        "",
        f"num_runs_with_semantic_memory_metrics: `{summary['num_runs_with_semantic_memory_metrics']}`",
        f"major_semantic_group_count: `{summary['major_semantic_group_count']}`",
        f"coarse_group_diversity_gate_pass: `{str(coarse_group_diversity_gate_pass).lower()}`",
        f"fine_label_count: `{summary['fine_label_count']}`",
        f"fine_label_diversity_gate_pass: `{str(fine_label_diversity_gate_pass).lower()}`",
        f"phase1_gate_pass: `{str(summary['phase1_gate_pass']).lower()}`",
        f"aggregate_heatmap_png_written: `{str(plotted).lower()}`",
        "",
        "Fine-label coverage comes from the existing Stage C predicted-masklet audit, not from GT semantic labels.",
        "",
        "Per-pixel visualization artifacts were not generated because the current smoke logs do not contain map-level tensors.",
    ]
    (out_dir / "passive_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if summary["phase1_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
