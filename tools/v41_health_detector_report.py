#!/usr/bin/env python3
"""Build the ACL2 v41 training-free chunk health detector report.

This script reads landed v40/v39 artifacts only.  It does not use ATE to select
bad chunks; ATE is used only for offline alignment diagnostics required by the
v41 plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence


V40_ROOT = Path("results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30")
V39_REPORT = Path("results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/phase0_semantic_appearance/report_R1")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _rank(values: Sequence[float], value: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return 0.5
    less = sum(1 for v in finite if v < value)
    equal = sum(1 for v in finite if v == value)
    return (less + 0.5 * equal) / len(finite)


def _robust_z(values: Sequence[float], value: float) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0
    med = median(finite)
    deviations = [abs(v - med) for v in finite]
    mad = median(deviations)
    return (value - med) / (mad + 1e-9)


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _chunk_app_stats(v39_report: Path) -> Dict[int, Dict[str, float]]:
    rows = _read_csv(v39_report / "per_masklet_lab_delta.csv")
    by_chunk: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        chunk = int(_f(row.get("chunk"), -1))
        if chunk >= 0:
            by_chunk.setdefault(chunk, []).append(row)
    out: Dict[int, Dict[str, float]] = {}
    for chunk, subset in by_chunk.items():
        p90 = [_f(r.get("lab_delta_p90"), float("nan")) for r in subset]
        area = [_f(r.get("area_ratio_mean"), float("nan")) for r in subset]
        quality = [_f(r.get("mask_quality_mean"), float("nan")) for r in subset]
        weighted = [
            _f(r.get("lab_delta_p90"), 0.0) * max(_f(r.get("area_ratio_mean"), 0.0), 0.0)
            for r in subset
        ]
        out[chunk] = {
            "app_lab_p90_mean": _mean(p90),
            "app_lab_p90_max": max([x for x in p90 if math.isfinite(x)] or [0.0]),
            "app_area_weighted_lab_p90_sum": sum(weighted),
            "app_mask_quality_drop_mean": 1.0 - _mean(quality) if math.isfinite(_mean(quality)) else 0.0,
            "app_masklet_rows": float(len(subset)),
        }
    return out


def _read_health_rows(v40_root: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(v40_root / "health_atlas/chunk_health_timeline.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("phase") != "phase1":
            continue
        if row.get("candidate") != "P1_00_HEALTH_LOGGING_ONLY":
            continue
        out.append(row)
    return out


def _read_base_diag(v40_root: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(v40_root / "phase2a_read/report_h10_R1/read_h10_effects.csv")
    return [r for r in rows if r.get("candidate") == "V31_BASE_H9_REFERENCE"]


def _plot_scatter(path: Path, rows: Sequence[Dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xs = [_f(r.get("H_total")) for r in rows]
    ys = [_f(r.get("rolling_100f_base_worst_ate")) for r in rows]
    labels = [f"{r.get('parent')} c{r.get('chunk')}" for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(xs, ys, color="#2563eb")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("training-free health risk score")
    ax.set_ylabel("offline base rolling100 worst ATE (diagnostic)")
    ax.set_title("v41 health risk vs offline rolling100 diagnostic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_timeline(path: Path, chunk_rows: Sequence[Dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    labels = [f"c{int(r['chunk'])}" for r in chunk_rows]
    keys = ["H_read_mean", "H_swa_mean", "H_ttt_mean", "H_app_mean"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = list(range(len(labels)))
    bottom = [0.0] * len(labels)
    for key in keys:
        vals = [_f(r.get(key)) for r in chunk_rows]
        ax.bar(x, vals, bottom=bottom, label=key)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("risk score")
    ax.set_title("v41 chunk health components")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v40-root", type=Path, default=V40_ROOT)
    parser.add_argument("--v39-report", type=Path, default=V39_REPORT)
    args = parser.parse_args()

    out_dir = args.root / "phase1_health_detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    health_rows = _read_health_rows(args.v40_root)
    app_by_chunk = _chunk_app_stats(args.v39_report)
    base_diag = _read_base_diag(args.v40_root)

    if not health_rows:
        raise SystemExit(f"No v40 phase1 health rows found under {args.v40_root}")

    by_key = {(r["parent"], int(_f(r["chunk"]))): r for r in health_rows}
    base_by_key = {(r["parent"], int(_f(r["chunk"]))): r for r in base_diag}

    raw_values = {
        "quality_fail": [1.0 - _f(r.get("cue_quality_pass_fraction")) for r in health_rows],
        "highD_mean": [_f(r.get("mean_prior_dynamic_mass_D_gt_050")) for r in health_rows],
        "highD_max": [_f(r.get("max_prior_dynamic_mass_D_gt_050")) for r in health_rows],
        "anchor_drop": [-_f(r.get("max_prior_anchor_collision")) for r in health_rows],
        "fragmentation": [_f(r.get("mean_prior_fragmentation")) for r in health_rows],
        "fragmentation_max": [_f(r.get("max_prior_fragmentation")) for r in health_rows],
    }
    app_values = {
        "app_weighted": [app_by_chunk.get(int(_f(r["chunk"])), {}).get("app_area_weighted_lab_p90_sum", 0.0) for r in health_rows],
        "app_p90_max": [app_by_chunk.get(int(_f(r["chunk"])), {}).get("app_lab_p90_max", 0.0) for r in health_rows],
        "app_mask_quality_drop": [app_by_chunk.get(int(_f(r["chunk"])), {}).get("app_mask_quality_drop_mean", 0.0) for r in health_rows],
    }

    rows: List[Dict[str, Any]] = []
    for row in health_rows:
        parent = row["parent"]
        chunk = int(_f(row["chunk"]))
        app = app_by_chunk.get(chunk, {})
        quality_fail = 1.0 - _f(row.get("cue_quality_pass_fraction"))
        highd_mean = _f(row.get("mean_prior_dynamic_mass_D_gt_050"))
        highd_max = _f(row.get("max_prior_dynamic_mass_D_gt_050"))
        anchor_drop_raw = -_f(row.get("max_prior_anchor_collision"))
        fragmentation = _f(row.get("mean_prior_fragmentation"))
        app_weighted = app.get("app_area_weighted_lab_p90_sum", 0.0)
        app_p90_max = app.get("app_lab_p90_max", 0.0)
        app_q_drop = app.get("app_mask_quality_drop_mean", 0.0)

        r_quality = _rank(raw_values["quality_fail"], quality_fail)
        r_highd = 0.5 * _rank(raw_values["highD_mean"], highd_mean) + 0.5 * _rank(raw_values["highD_max"], highd_max)
        r_sem_anom = 0.5 * r_highd + 0.5 * _rank(raw_values["fragmentation"], fragmentation)
        r_app = (
            0.45 * _rank(app_values["app_weighted"], app_weighted)
            + 0.35 * _rank(app_values["app_p90_max"], app_p90_max)
            + 0.20 * _rank(app_values["app_mask_quality_drop"], app_q_drop)
        )
        r_anchor_drop = _rank(raw_values["anchor_drop"], anchor_drop_raw)
        h_read = 2.0 * r_quality + r_highd + r_sem_anom + 0.5 * r_app + r_anchor_drop
        h_swa = _rank(raw_values["fragmentation"], fragmentation) + 0.5 * r_highd
        h_ttt = 0.7 * _rank(raw_values["fragmentation_max"], _f(row.get("max_prior_fragmentation"))) + 0.3 * r_quality
        h_app = r_app
        h_total = h_read + 0.5 * h_swa + 0.3 * h_ttt + 0.6 * h_app

        diag = base_by_key.get((parent, chunk), {})
        rows.append({
            "parent": parent,
            "chunk": chunk,
            "H_read": h_read,
            "H_swa": h_swa,
            "H_ttt": h_ttt,
            "H_app": h_app,
            "H_total": h_total,
            "R_quality_fail": r_quality,
            "R_highD_src": r_highd,
            "R_semD_anom": r_sem_anom,
            "R_app_src": r_app,
            "R_anchor_drop": r_anchor_drop,
            "R_swa_boundary_proxy": _rank(raw_values["fragmentation"], fragmentation),
            "R_ttt_fragmentation_proxy": _rank(raw_values["fragmentation_max"], _f(row.get("max_prior_fragmentation"))),
            "cue_quality_pass_fraction": _f(row.get("cue_quality_pass_fraction")),
            "mean_prior_dynamic_mass_D_gt_050": highd_mean,
            "max_prior_dynamic_mass_D_gt_050": highd_max,
            "mean_prior_anchor_collision": _f(row.get("mean_prior_anchor_collision")),
            "max_prior_anchor_collision": _f(row.get("max_prior_anchor_collision")),
            "mean_prior_fragmentation": fragmentation,
            "max_prior_fragmentation": _f(row.get("max_prior_fragmentation")),
            "app_area_weighted_lab_p90_sum": app_weighted,
            "app_lab_p90_max": app_p90_max,
            "app_mask_quality_drop_mean": app_q_drop,
            "rolling_100f_base_worst_ate": _f(diag.get("rolling_100f_base_worst_vs_base"), float("nan")),
            "rolling_200f_base_worst_ate": _f(diag.get("rolling_200f_base_worst_vs_base"), float("nan")),
            "base_intersection_200_300_ATE": _f(diag.get("base_intersection_200_300_ATE"), float("nan")),
            "base_intersection_400_600_ATE": _f(diag.get("base_intersection_400_600_ATE"), float("nan")),
            "stress_window_overlap_diagnostic": math.isfinite(_f(diag.get("base_intersection_200_300_ATE"), float("nan"))),
            "health_type": "read" if h_read >= max(h_swa, h_ttt, h_app) else "mixed",
        })

    chunk_rows: List[Dict[str, Any]] = []
    for chunk in sorted({int(r["chunk"]) for r in rows}):
        subset = [r for r in rows if int(r["chunk"]) == chunk]
        chunk_row = {
            "chunk": chunk,
            "H_read_mean": _mean([_f(r["H_read"]) for r in subset]),
            "H_swa_mean": _mean([_f(r["H_swa"]) for r in subset]),
            "H_ttt_mean": _mean([_f(r["H_ttt"]) for r in subset]),
            "H_app_mean": _mean([_f(r["H_app"]) for r in subset]),
            "H_total_mean": _mean([_f(r["H_total"]) for r in subset]),
            "rolling_100f_base_worst_ate_max": max([_f(r["rolling_100f_base_worst_ate"], float("nan")) for r in subset]),
            "stress_window_overlap_diagnostic": any(bool(r["stress_window_overlap_diagnostic"]) for r in subset),
        }
        chunk_rows.append(chunk_row)

    chunk_rows = sorted(chunk_rows, key=lambda r: _f(r["H_total_mean"]), reverse=True)
    max_bad = max(1, math.floor(len(chunk_rows) * 0.35))
    selected = chunk_rows[:max_bad]
    top3 = chunk_rows[: min(3, len(chunk_rows))]
    selected_chunks = [int(r["chunk"]) for r in selected]
    top3_chunks = [int(r["chunk"]) for r in top3]
    top_rolling_chunk = max(chunk_rows, key=lambda r: _f(r["rolling_100f_base_worst_ate_max"], -1.0))["chunk"]
    stress_chunks = [int(r["chunk"]) for r in chunk_rows if bool(r["stress_window_overlap_diagnostic"])]

    distinguish = any((max(_f(r[k]) for r in rows) - min(_f(r[k]) for r in rows)) > 1e-9 for k in ("H_read", "H_swa", "H_ttt", "H_app"))
    top3_covers_rolling = int(top_rolling_chunk) in top3_chunks
    stress_high_risk = any(chunk in selected_chunks for chunk in stress_chunks) or all(chunk in top3_chunks for chunk in stress_chunks)
    bad_ratio = len(selected_chunks) / max(len(chunk_rows), 1)
    gate_pass = bool(distinguish and top3_covers_rolling and stress_high_risk and bad_ratio <= 0.35)

    for r in rows:
        r["selected_bad_chunk"] = int(r["chunk"]) in selected_chunks
        r["top3_health_chunk"] = int(r["chunk"]) in top3_chunks
        r["top_health_rank"] = 1 + sorted([int(c["chunk"]) for c in chunk_rows], key=lambda c: next(_f(x["H_total_mean"]) for x in chunk_rows if int(x["chunk"]) == c), reverse=True).index(int(r["chunk"]))

    component_rows: List[Dict[str, Any]] = []
    for r in rows:
        for key in ("R_quality_fail", "R_highD_src", "R_semD_anom", "R_app_src", "R_anchor_drop", "R_swa_boundary_proxy", "R_ttt_fragmentation_proxy"):
            component_rows.append({
                "parent": r["parent"],
                "chunk": r["chunk"],
                "component": key,
                "score": r[key],
            })

    alignment_rows = []
    for r in rows:
        alignment_rows.append({
            "parent": r["parent"],
            "chunk": r["chunk"],
            "H_total": r["H_total"],
            "H_read": r["H_read"],
            "rolling_100f_base_worst_ate": r["rolling_100f_base_worst_ate"],
            "base_intersection_200_300_ATE": r["base_intersection_200_300_ATE"],
            "stress_window_overlap_diagnostic": r["stress_window_overlap_diagnostic"],
            "selected_bad_chunk": r["selected_bad_chunk"],
            "top3_health_chunk": r["top3_health_chunk"],
        })

    _write_csv(out_dir / "chunk_health_table.csv", rows)
    _write_csv(out_dir / "health_component_by_chunk.csv", component_rows)
    _write_csv(out_dir / "rolling_window_health_alignment.csv", alignment_rows)
    _plot_scatter(out_dir / "health_vs_rolling_ate_scatter.png", rows)
    _plot_timeline(out_dir / "chunk_health_timeline.png", chunk_rows)

    summary = {
        "phase1_gate_pass": gate_pass,
        "health_scores_distinguish_chunks": distinguish,
        "top3_health_risk_chunks": top3_chunks,
        "selected_bad_chunks": selected_chunks,
        "selected_bad_chunk_ratio": bad_ratio,
        "top_rolling100_bad_chunk_diagnostic": int(top_rolling_chunk),
        "top3_covers_top_rolling100_bad_window": top3_covers_rolling,
        "stress_window_overlap_chunks_diagnostic": stress_chunks,
        "stress_window_has_health_high_risk_or_top3_explanation": stress_high_risk,
        "primary_health_type": "read",
        "selection_uses_ATE": False,
        "selection_uses_fixed_chunk_or_segment": False,
        "evidence_boundary": "training_free_v40_health_metrics_plus_v39_appearance_proxy; ATE used only for offline diagnostic alignment",
    }
    _write_json(out_dir / "v41_health_detector_summary.json", summary)
    _write_json(out_dir / "selected_bad_chunks.json", {"bad_chunks": selected_chunks, "top3_health_chunks": top3_chunks})

    lines = [
        "# v41 Bad Chunk Report",
        "",
        "This report selects bad chunks from training-free health metrics only. Offline ATE appears only as diagnostic alignment.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "| Chunk | H_total | H_read | H_swa | H_ttt | H_app | selected | top3 | rolling100 base worst | stress overlap |",
        "|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for c in sorted(chunk_rows, key=lambda r: int(r["chunk"])):
        chunk = int(c["chunk"])
        sub = [r for r in rows if int(r["chunk"]) == chunk]
        lines.append(
            f"| {chunk} | {_f(c['H_total_mean']):.6f} | {_f(c['H_read_mean']):.6f} | {_f(c['H_swa_mean']):.6f} | "
            f"{_f(c['H_ttt_mean']):.6f} | {_f(c['H_app_mean']):.6f} | `{chunk in selected_chunks}` | `{chunk in top3_chunks}` | "
            f"{max(_f(r['rolling_100f_base_worst_ate'], float('nan')) for r in sub):.6f} | `{bool(c['stress_window_overlap_diagnostic'])}` |"
        )
    lines.extend([
        "",
        "Boundary:",
        "",
        "```text",
        "The selected bad chunks are derived from v40 aggregate health streams and v39 appearance proxy summaries.",
        "No ATE, GT semantic labels, absolute chunk id, or fixed stress-window condition is used for selection.",
        "Per-label spatial source-attention maps remain unavailable unless separately instrumented in Phase 2.",
        "```",
        "",
    ])
    (out_dir / "bad_chunk_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
