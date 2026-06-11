#!/usr/bin/env python3
"""Passive fine-label semantic attribution for ACL2 v26/v27."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROLE_NAMES = {
    "0": "FALLBACK",
    "1": "POSITIVE_LONG",
    "2": "NEUTRAL_KEEP",
    "3": "NEGATIVE_SHORT",
    "4": "PROTECT_NEUTRAL",
}

LABEL_NAMES = {
    -1: "unassigned",
    0: "unknown",
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    20: "sky",
    21: "vegetation",
    22: "grass",
}

LABEL_TO_COARSE = {
    1: "STRUCTURE_ANCHOR",
    2: "STRUCTURE_ANCHOR",
    3: "STRUCTURE_ANCHOR",
    4: "STRUCTURE_ANCHOR",
    5: "STRUCTURE_ANCHOR",
    20: "LOW_VALUE_STUFF",
    21: "LOW_VALUE_STUFF",
    22: "LOW_VALUE_STUFF",
}


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                continue
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _label_name(label_id: Any) -> str:
    idx = _safe_int(label_id, 0)
    return LABEL_NAMES.get(idx, f"label_{idx}")


def _dominant_role(role_counts: Dict[str, Any]) -> Tuple[str, int]:
    best_role = "missing"
    best_count = 0
    for role_id, count in role_counts.items():
        n = _safe_int(count, 0)
        if n > best_count:
            best_role = str(role_id)
            best_count = n
    return best_role, best_count


def _candidate_from_run(run_name: str) -> str:
    parts = run_name.split("_chunk", 1)[0]
    for prefix in (
        "V28_P1_PASSIVE_R1_",
        "V28_P1_PASSIVE_R2_",
        "V28_P0_SMOKE_R1_",
        "V28_P0_SMOKE_R2_",
        "V28_P0_SMOKE_R3_",
        "V27_P1_PASSIVE_R1_",
        "V27_P0_SMOKE_R1_",
        "V27_P0_SMOKE_R2_",
        "V26_P1_PASSIVE_R1_",
        "V26_P0_SMOKE_R1_",
    ):
        if parts.startswith(prefix):
            return parts[len(prefix):]
    return parts


def _collect_runtime_rows(results_root: Path, run_prefix: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    condition_acc: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    condition_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for path in sorted((results_root / "rollouts").glob("*/semantic_memory_path_summary.jsonl")):
        if ".INVALID" in str(path):
            continue
        run_name = path.parent.name
        if run_prefix and not run_name.startswith(run_prefix):
            continue
        candidate_id = _candidate_from_run(run_name)
        for row in _jsonl(path):
            chunk_idx = row.get("chunk_idx")
            cond = row.get("fine_label_condition_metrics") or {}
            if isinstance(cond, dict):
                for label_id, metrics in cond.items():
                    if not isinstance(metrics, dict):
                        continue
                    token_count = _safe_float(metrics.get("token_count"))
                    if token_count <= 0:
                        continue
                    key = str(label_id)
                    acc = condition_acc[key]
                    acc["token_count"] += token_count
                    acc["D_mean_weighted"] += _safe_float(metrics.get("D_mean")) * token_count
                    acc["D_p90_weighted"] += _safe_float(metrics.get("D_p90")) * token_count
                    acc["Q_mean_weighted"] += _safe_float(metrics.get("Q_mean")) * token_count
                    acc["Q_p10_weighted"] += _safe_float(metrics.get("Q_p10")) * token_count
                    acc["conflict_mean_weighted"] += _safe_float(metrics.get("conflict_mean")) * token_count
                    acc["conflict_p90_weighted"] += _safe_float(metrics.get("conflict_p90")) * token_count
                    acc["scale_risk_mean_weighted"] += _safe_float(metrics.get("scale_risk_mean")) * token_count
                    acc["scale_risk_p90_weighted"] += _safe_float(metrics.get("scale_risk_p90")) * token_count
                    acc["conflict_available"] = max(
                        acc["conflict_available"],
                        1.0 if bool(row.get("condition_signal_conflict_available", False)) else 0.0,
                    )
                    acc["scale_risk_available"] = max(
                        acc["scale_risk_available"],
                        1.0 if bool(row.get("condition_signal_scale_risk_available", False)) else 0.0,
                    )
                    condition_rows.append(
                        {
                            "run_name": run_name,
                            "candidate_id": candidate_id,
                            "chunk_idx": chunk_idx,
                            "fine_label_id": label_id,
                            "fine_label": _label_name(label_id),
                            "coarse_group": LABEL_TO_COARSE.get(_safe_int(label_id, -999), "UNKNOWN"),
                            "token_count": token_count,
                            "D_mean": metrics.get("D_mean"),
                            "D_p90": metrics.get("D_p90"),
                            "Q_mean": metrics.get("Q_mean"),
                            "Q_p10": metrics.get("Q_p10"),
                            "conflict_mean": metrics.get("conflict_mean"),
                            "conflict_p90": metrics.get("conflict_p90"),
                            "scale_risk_mean": metrics.get("scale_risk_mean"),
                            "scale_risk_p90": metrics.get("scale_risk_p90"),
                            "conflict_available": bool(row.get("condition_signal_conflict_available", False)),
                            "conflict_level": row.get("condition_signal_conflict_level"),
                            "conflict_source": row.get("condition_signal_conflict_source"),
                            "scale_risk_available": bool(row.get("condition_signal_scale_risk_available", False)),
                            "scale_risk_level": row.get("condition_signal_scale_risk_level"),
                            "scale_risk_source": row.get("condition_signal_scale_risk_source"),
                            "source": "semantic_memory_path_summary.jsonl",
                        }
                    )

            path_roles = row.get("fine_label_path_role_counts") or {}
            if not isinstance(path_roles, dict):
                continue
            for path_name, by_label in path_roles.items():
                if not isinstance(by_label, dict):
                    continue
                for label_id, role_counts in by_label.items():
                    if not isinstance(role_counts, dict):
                        continue
                    total = sum(_safe_int(v, 0) for v in role_counts.values())
                    dom_role, dom_count = _dominant_role(role_counts)
                    memory_rows.append(
                        {
                            "run_name": run_name,
                            "candidate_id": candidate_id,
                            "chunk_idx": chunk_idx,
                            "path": path_name,
                            "fine_label_id": label_id,
                            "fine_label": _label_name(label_id),
                            "coarse_group": LABEL_TO_COARSE.get(_safe_int(label_id, -999), "UNKNOWN"),
                            "total_role_tokens": total,
                            "dominant_role_id": dom_role,
                            "dominant_role": ROLE_NAMES.get(dom_role, dom_role),
                            "dominant_role_fraction": float(dom_count / total) if total > 0 else 0.0,
                            "role_counts_json": json.dumps(role_counts, sort_keys=True),
                        }
                    )
                    action_rows.append(_action_row(run_name, candidate_id, chunk_idx, path_name, label_id, role_counts))

    aggregate_rows: List[Dict[str, Any]] = []
    for label_id, acc in sorted(condition_acc.items(), key=lambda item: _safe_int(item[0], 9999)):
        total = acc["token_count"]
        aggregate_rows.append(
            {
                "fine_label_id": label_id,
                "fine_label": _label_name(label_id),
                "coarse_group": LABEL_TO_COARSE.get(_safe_int(label_id, -999), "UNKNOWN"),
                "token_count": total,
                "D_mean_weighted": acc["D_mean_weighted"] / total if total else 0.0,
                "D_p90_weighted": acc["D_p90_weighted"] / total if total else 0.0,
                "Q_mean_weighted": acc["Q_mean_weighted"] / total if total else 0.0,
                "Q_p10_weighted": acc["Q_p10_weighted"] / total if total else 0.0,
                "conflict_mean_weighted": acc["conflict_mean_weighted"] / total if total else 0.0,
                "conflict_p90_weighted": acc["conflict_p90_weighted"] / total if total else 0.0,
                "scale_risk_mean_weighted": acc["scale_risk_mean_weighted"] / total if total else 0.0,
                "scale_risk_p90_weighted": acc["scale_risk_p90_weighted"] / total if total else 0.0,
                "conflict_available": bool(acc["conflict_available"]),
                "scale_risk_available": bool(acc["scale_risk_available"]),
                "note": "D/Q are token signals; conflict/scale-risk may be provenance-tagged broadcasts.",
            }
        )
    return aggregate_rows, memory_rows, action_rows


def _action_row(
    run_name: str,
    candidate_id: str,
    chunk_idx: Any,
    path_name: str,
    label_id: Any,
    role_counts: Dict[str, Any],
) -> Dict[str, Any]:
    total = sum(_safe_int(v, 0) for v in role_counts.values())
    positive = _safe_int(role_counts.get("1"), 0)
    neutral = _safe_int(role_counts.get("2"), 0)
    negative = _safe_int(role_counts.get("3"), 0)
    protect = _safe_int(role_counts.get("4"), 0)
    fallback = _safe_int(role_counts.get("0"), 0)
    if path_name in {"frame", "global"}:
        keep_like = positive + protect
        weak_like = neutral
        skip_like = negative
    elif path_name == "swa":
        keep_like = protect + neutral
        weak_like = positive
        skip_like = negative
    else:
        keep_like = positive
        weak_like = neutral + protect
        skip_like = negative
    return {
        "run_name": run_name,
        "candidate_id": candidate_id,
        "chunk_idx": chunk_idx,
        "path": path_name,
        "fine_label_id": label_id,
        "fine_label": _label_name(label_id),
        "coarse_group": LABEL_TO_COARSE.get(_safe_int(label_id, -999), "UNKNOWN"),
        "total_tokens": total,
        "keep_or_positive_tokens": keep_like,
        "weak_or_neutral_tokens": weak_like,
        "skip_or_negative_tokens": skip_like,
        "fallback_tokens": fallback,
        "keep_or_positive_fraction": keep_like / total if total else 0.0,
        "skip_or_negative_fraction": skip_like / total if total else 0.0,
        "role_counts_json": json.dumps(role_counts, sort_keys=True),
    }


def _collect_chunk_labels(results_root: Path, stage_c_audit: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((results_root / "rollouts").glob("*/semantic_group_summary.jsonl")):
        if ".INVALID" in str(path):
            continue
        for row in _jsonl(path):
            counts = row.get("fine_label_name_counts") or {}
            if isinstance(counts, dict):
                for label, count in counts.items():
                    rows.append(
                        {
                            "chunk_idx": row.get("chunk_idx"),
                            "start_frame": row.get("start_frame"),
                            "end_frame": row.get("end_frame"),
                            "fine_label": label,
                            "count": count,
                            "source": "runtime_semantic_group_summary",
                        }
                    )
    stage_path = stage_c_audit / "label_counts_by_chunk.csv"
    if stage_path.exists():
        with stage_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "chunk_idx": row.get("chunk_idx"),
                        "start_frame": row.get("start_frame"),
                        "end_frame": row.get("end_frame"),
                        "fine_label": row.get("label"),
                        "count": row.get("count"),
                        "source": str(stage_path),
                    }
                )
    return rows


def _collect_candidate_metrics(prior_roots: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in prior_roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.glob("phase2_*_report_R1/candidate_vs_H9_delta_by_horizon.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        {
                            "source_report": str(csv_path),
                            "candidate_id": row.get("candidate_id"),
                            "chunk_id": row.get("chunk_id") or row.get("chunk"),
                            "horizon": row.get("horizon"),
                            "ATE_delta_vs_H9": row.get("ATE_delta_vs_H9"),
                            "window_200_300_delta_vs_H9": row.get("window_200_300_delta_vs_H9") or row.get("[200,300)_delta_vs_H9"),
                            "window_400_600_delta_vs_H9": row.get("window_400_600_delta_vs_H9") or row.get("[400,600)_delta_vs_H9"),
                            "attribution_scope": "candidate_level_only_not_per_label",
                        }
                    )
    return rows


def _make_plots(out_dir: Path, condition_rows: List[Dict[str, Any]], action_rows: List[Dict[str, Any]]) -> Dict[str, bool]:
    status = {
        "per_label_Dg_distribution": False,
        "per_label_ttt_role_mass": False,
        "per_label_path_role_mass": False,
    }
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return status

    if condition_rows:
        labels = [str(row["fine_label"]) for row in condition_rows]
        d_mean = [_safe_float(row.get("D_mean_weighted")) for row in condition_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
        ax.bar(labels, d_mean)
        ax.set_ylabel("weighted D_g mean")
        ax.set_title("v26 per-label D_g mean")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "per_label_Dg_distribution.png", dpi=160)
        plt.close(fig)
        status["per_label_Dg_distribution"] = True

    if action_rows:
        ttt = [row for row in action_rows if row.get("path") == "ttt"]
        labels = sorted({str(row["fine_label"]) for row in ttt})
        if labels:
            pos = []
            neu = []
            neg = []
            for label in labels:
                subset = [row for row in ttt if row.get("fine_label") == label]
                pos.append(sum(_safe_float(row.get("keep_or_positive_tokens")) for row in subset))
                neu.append(sum(_safe_float(row.get("weak_or_neutral_tokens")) for row in subset))
                neg.append(sum(_safe_float(row.get("skip_or_negative_tokens")) for row in subset))
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
            xs = range(len(labels))
            ax.bar(xs, pos, label="positive")
            ax.bar(xs, neu, bottom=pos, label="neutral")
            ax.bar(xs, neg, bottom=[p + n for p, n in zip(pos, neu)], label="negative")
            ax.set_xticks(list(xs), labels, rotation=30)
            ax.set_title("v26 per-label TTT role mass")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "per_label_TTT_role_mass.png", dpi=160)
            plt.close(fig)
            status["per_label_ttt_role_mass"] = True
            status["per_label_path_role_mass"] = True
    return status


def _write_report(out_dir: Path, summary: Dict[str, Any]) -> None:
    phase_label = str(summary.get("phase", "phase1_passive_fine_label_attribution"))
    if "v27" in phase_label:
        title = "# ACL2 v27 Phase 1 Passive Semantic-Risk Attribution"
        boundary = (
            "Conflict and scale-risk are available with recorded provenance. "
            "Current v27 evidence uses provenance-tagged broadcast conditions unless a row states a finer level."
        )
    else:
        title = "# ACL2 v26 Phase 1 Passive Fine-Label Attribution"
        boundary = (
            "Conflict and scale-risk may be unavailable in v26; if unavailable, Phase 2 must be interpreted as fine-label + D/Q conditioned only."
        )
    lines = [
        title,
        "",
        "This report uses landed runtime summaries only; it does not treat predicted video-masklet semantic as GT.",
        "",
        "## Gate Summary",
        "",
        f"phase1_gate_pass = `{str(summary['phase1_gate_pass']).lower()}`",
        f"fine_label_count = `{summary['fine_label_count']}`",
        f"coarse_internal_label_diversity = `{str(summary['coarse_internal_label_diversity']).lower()}`",
        f"path_role_diversity_within_coarse = `{str(summary['path_role_diversity_within_coarse']).lower()}`",
        f"D_mean_range_within_coarse_max = `{summary['D_mean_range_within_coarse_max']}`",
        f"conflict_signal_available = `{str(summary['conflict_signal_available']).lower()}`",
        f"scale_risk_signal_available = `{str(summary['scale_risk_signal_available']).lower()}`",
        "",
        "## Boundary",
        "",
        boundary,
    ]
    (out_dir / "phase1_label_condition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v26_videomasklet_semanticrole_router_allmemory_parallel")
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--stage-c-audit-dir", default="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/semantic_audit")
    parser.add_argument("--prior-result-root", action="append", default=[
        "results/kitti01_hmc_v2/acl2_v25b_videomasklet_semanticprior_allmemory_parallel",
    ])
    parser.add_argument("--run-prefix", default="", help="Optional rollout-name prefix filter, e.g. V28_P1_PASSIVE_R1.")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = results_root / "phase1_passive_attribution"
    visual_dir = results_root / "phase1_visual_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    condition_rows, memory_rows, action_rows = _collect_runtime_rows(results_root, args.run_prefix)
    chunk_rows = _collect_chunk_labels(results_root, Path(args.stage_c_audit_dir).resolve())
    metric_rows = _collect_candidate_metrics([Path(p).resolve() for p in args.prior_result_root])

    _write_csv(out_dir / "per_label_dg_conflict_scale.csv", condition_rows)
    _write_csv(out_dir / "per_label_memory_mass.csv", memory_rows)
    _write_csv(out_dir / "per_label_path_action.csv", action_rows)
    _write_csv(out_dir / "per_chunk_label_distribution.csv", chunk_rows)
    _write_csv(out_dir / "per_label_segment_error_corr.csv", metric_rows)

    d_ranges: Dict[str, List[float]] = defaultdict(list)
    for row in condition_rows:
        d_ranges[str(row.get("coarse_group"))].append(_safe_float(row.get("D_mean_weighted")))
    coarse_ranges = {
        group: (max(vals) - min(vals) if vals else 0.0)
        for group, vals in d_ranges.items()
    }
    label_by_coarse = defaultdict(set)
    for row in condition_rows:
        label_by_coarse[str(row.get("coarse_group"))].add(str(row.get("fine_label")))
    coarse_internal_label_diversity = any(len(labels) >= 2 for labels in label_by_coarse.values())

    dominant_by_coarse_path: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in memory_rows:
        dominant_by_coarse_path[(str(row.get("coarse_group")), str(row.get("path")))].add(str(row.get("dominant_role_id")))
    path_role_diversity_within_coarse = any(len(roles) >= 2 for roles in dominant_by_coarse_path.values())
    d_range_max = max(coarse_ranges.values()) if coarse_ranges else 0.0
    phase1_gate_pass = bool(
        coarse_internal_label_diversity
        and (path_role_diversity_within_coarse or d_range_max >= 0.03)
        and bool(condition_rows)
        and bool(memory_rows)
    )

    corr_rows = [
        {
            "criterion": "coarse_internal_label_diversity",
            "value": coarse_internal_label_diversity,
            "detail": json.dumps({k: sorted(v) for k, v in label_by_coarse.items()}, ensure_ascii=False, sort_keys=True),
        },
        {
            "criterion": "path_role_diversity_within_coarse",
            "value": path_role_diversity_within_coarse,
            "detail": json.dumps({f"{k[0]}::{k[1]}": sorted(v) for k, v in dominant_by_coarse_path.items()}, ensure_ascii=False, sort_keys=True),
        },
        {
            "criterion": "D_mean_range_within_coarse",
            "value": d_range_max,
            "detail": json.dumps(coarse_ranges, ensure_ascii=False, sort_keys=True),
        },
    ]
    _write_csv(out_dir / "label_condition_correlation_summary.csv", corr_rows)
    plot_status = _make_plots(visual_dir, condition_rows, action_rows)
    (visual_dir / "visual_dashboard_limitations.md").write_text(
        "\n".join(
            [
                "# v26 Phase 1 Visual Dashboard Boundary",
                "",
                "Generated plots are aggregate runtime attribution plots from landed JSONL summaries.",
                "RGB/fine-label spatial overlays, conflict overlays, and scale-risk overlays are not generated here because conflict/scale-risk are not token-aligned runtime signals in the current implementation.",
                "This limitation is recorded so it cannot be mistaken for completed visual evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    conflict_signal_available = any(bool(row.get("conflict_available")) for row in condition_rows)
    scale_risk_signal_available = any(bool(row.get("scale_risk_available")) for row in condition_rows)
    summary = {
        "phase": args.phase_name or "v26_phase1_passive_fine_label_attribution",
        "phase1_gate_pass": phase1_gate_pass,
        "fine_label_count": len({str(row.get("fine_label")) for row in condition_rows}),
        "condition_rows": len(condition_rows),
        "memory_rows": len(memory_rows),
        "action_rows": len(action_rows),
        "coarse_internal_label_diversity": coarse_internal_label_diversity,
        "path_role_diversity_within_coarse": path_role_diversity_within_coarse,
        "D_mean_range_within_coarse": coarse_ranges,
        "D_mean_range_within_coarse_max": d_range_max,
        "conflict_signal_available": conflict_signal_available,
        "scale_risk_signal_available": scale_risk_signal_available,
        "candidate_metric_rows": len(metric_rows),
        "plots": plot_status,
        "counts_as_deployable_online_success": False,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
    }
    (out_dir / "passive_attribution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_dir, summary)
    return 0 if phase1_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
