#!/usr/bin/env python3
"""Diagnose whether v69 anchor-bank scores predict KITTI01 failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-summary", required=True, type=Path)
    parser.add_argument("--v62-report", required=True, type=Path)
    parser.add_argument("--v67-observability", type=Path)
    parser.add_argument("--v68-report", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repair-mode", choices=("base", "road_boundary", "near_static", "raw_overlap_residual"), default="base")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    fields = list(fieldnames) if fieldnames is not None else sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _float(value: Any) -> float:
    if value in (None, "", "None", "nan", "NaN"):
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _by_chunk(rows: Iterable[Mapping[str, Any]], method: Optional[str] = None) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if method is not None and str(row.get("method")) != method:
            continue
        if row.get("chunk_id") in (None, ""):
            continue
        out[int(row["chunk_id"])] = row
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 1e-12:
        return None
    return float((rx @ ry) / denom)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    mask = np.isfinite(s)
    s = s[mask]
    y = y[mask]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = _rankdata(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _robust_norm(values: Sequence[float]) -> Dict[int, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {}
    lo = float(np.quantile(arr, 0.05))
    hi = float(np.quantile(arr, 0.95))
    if abs(hi - lo) < 1e-9:
        lo = float(arr.min())
        hi = float(arr.max())
    out = {}
    for idx, value in enumerate(values):
        if not math.isfinite(value):
            out[idx] = 0.0
        else:
            out[idx] = max(0.0, min(1.0, (float(value) - lo) / (hi - lo + 1e-9)))
    return out


def _top30_labels(values: Sequence[float]) -> List[int]:
    arr = np.asarray(values, dtype=float)
    labels = [0 for _ in values]
    finite_idx = [i for i, v in enumerate(arr) if math.isfinite(float(v))]
    if not finite_idx:
        return labels
    k = max(1, int(math.ceil(0.30 * len(finite_idx))))
    ranked = sorted(finite_idx, key=lambda i: float(arr[i]), reverse=True)[:k]
    for idx in ranked:
        labels[idx] = 1
    return labels


def _load_s5_labels(v68_report: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if v68_report is None:
        return {}
    candidates = [
        v68_report / "phaseE_merge_multichunk" / "combo_s5_taildrop_nativeguard" / "phaseE_multichunk_decisions.csv",
        v68_report / "phaseE_merge_multichunk" / "s5_scale_only_alpha06_log07_globalcachefix" / "phaseE_multichunk_decisions.csv",
    ]
    out: Dict[int, Dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        for row in _read_csv(path):
            chunk = int(row["chunk"])
            positive = _bool(row.get("head_tail_phaseE_chunk_pass")) or _bool(row.get("overlap_phaseE_chunk_pass"))
            out[chunk] = {
                "chunk_id": chunk,
                "s5_positive": bool(positive),
                "source": str(path),
                "head_tail_improvement_vs_baseline_ratio": _float(row.get("head_tail_improvement_vs_baseline_ratio")),
                "overlap_improvement_vs_baseline_ratio": _float(row.get("overlap_improvement_vs_baseline_ratio")),
                "head_tail_beats_controls": _bool(row.get("head_tail_beats_controls")),
                "overlap_beats_controls": _bool(row.get("overlap_beats_controls")),
            }
    return out


def _build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    anchor_rows = _read_csv(args.anchor_summary)
    intrachunk = _by_chunk(_read_csv(args.v62_report / "phase5_intrachunk_scale" / "h35_intrachunk_scale_metrics.csv"), method="h35")
    overlap = _by_chunk(_read_csv(args.v62_report / "phase4_interchunk" / "overlap_transfer_metrics.csv"), method="h35")
    taxonomy = _by_chunk(_read_csv(args.v62_report / "phase7_taxonomy" / "chunk_error_taxonomy.csv"), method="h35")
    v67 = _by_chunk(_read_csv(args.v67_observability) if args.v67_observability else [])
    s5 = _load_s5_labels(args.v68_report)

    valid_scale_counts = [_float(r.get("valid_scale_anchor_count")) for r in anchor_rows]
    count_norm = _robust_norm(valid_scale_counts)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(anchor_rows):
        chunk = int(row["chunk_id"])
        coverage = max(0.0, min(1.0, _float(row.get("anchor_grid_coverage"))))
        entropy = max(0.0, min(1.0, _float(row.get("anchor_spatial_entropy"))))
        gram = _float(row.get("gram_motion_mean"))
        stability = 0.5 if not math.isfinite(gram) else max(0.0, min(1.0, 1.0 - gram))
        cond = _float(row.get("condition_score_median"))
        if not math.isfinite(cond):
            cond = max(0.0, min(1.0, _float(row.get("road_boundary_anchor_mass")) + _float(row.get("vertical_anchor_mass"))))
        temporal = np.nanmean(
            [
                _float(row.get("anchor_visible_head_frac")),
                _float(row.get("anchor_visible_tail_frac")),
                _float(row.get("anchor_visible_overlap_frac")),
            ]
        )
        if not math.isfinite(float(temporal)):
            temporal = 0.0
        Q = max(
            0.0,
            min(
                1.0,
                0.30 * count_norm.get(idx, 0.0)
                + 0.20 * max(0.0, min(1.0, coverage * entropy))
                + 0.20 * stability
                + 0.15 * max(0.0, min(1.0, cond))
                + 0.15 * max(0.0, min(1.0, float(temporal))),
            ),
        )
        lowtrust = max(0.0, min(1.0, 1.0 - _float(row.get("semantic_trust_mean"))))
        R = max(
            0.0,
            min(
                1.0,
                0.25 * _float(row.get("sky_risk_mass"))
                + 0.25 * _float(row.get("dynamic_risk_mass"))
                + 0.20 * _float(row.get("vegetation_risk_mass"))
                + 0.20 * _float(row.get("ground_anchor_mass"))
                + 0.10 * lowtrust,
            ),
        )
        inrow = intrachunk.get(chunk, {})
        overrow = overlap.get(chunk, {})
        taxrow = taxonomy.get(chunk, {})
        v67row = v67.get(chunk, {})
        future = _float(overrow.get("nonoverlap_future_error_after_overlap_sim3"))
        if not math.isfinite(future):
            future = _float(taxrow.get("future_after_overlap_sim3"))
        h35_gap = _float(v67row.get("H35_minus_C9_gap"))
        out = {
            "chunk_id": chunk,
            "frame_start": int(_float(row.get("frame_start"),)),
            "frame_end": int(_float(row.get("frame_end"),)),
            "Q_anchor": Q,
            "R_anchor": R,
            "O_anchor": Q - R,
            "valid_anchor_count": _float(row.get("valid_anchor_count")),
            "valid_scale_anchor_count": _float(row.get("valid_scale_anchor_count")),
            "anchor_grid_coverage": coverage,
            "anchor_spatial_entropy": entropy,
            "anchor_condition_score": cond,
            "road_boundary_anchor_mass": _float(row.get("road_boundary_anchor_mass")),
            "vertical_anchor_mass": _float(row.get("vertical_anchor_mass")),
            "anchor_gram_motion_mean": gram,
            "anchor_score_top10_mean": _float(row.get("anchor_score_top10_mean")),
            "sky_risk_mass": _float(row.get("sky_risk_mass")),
            "dynamic_risk_mass": _float(row.get("dynamic_risk_mass")),
            "vegetation_risk_mass": _float(row.get("vegetation_risk_mass")),
            "ground_anchor_mass": _float(row.get("ground_anchor_mass")),
            "semantic_trust_mean": _float(row.get("semantic_trust_mean")),
            "phaseA_quality_pass": _bool(row.get("anchor_bank_quality_pass")),
            "phaseA_reject_reason": row.get("reject_reason"),
            "head_to_tail_transfer_ratio": _float(inrow.get("head_to_tail_transfer_ratio")),
            "intra_scale_variance": _float(inrow.get("intra_scale_variance")),
            "future_after_overlap_error": future,
            "raw_overlap_residual": _float(overrow.get("overlap_sim3_residual_all")),
            "boundary_jump": _float(overrow.get("boundary_pose_jump")),
            "rolling100_error": _float(overrow.get("rolling100_error")),
            "global_chunk_ate": _float(overrow.get("global_chunk_ate")),
            "local_sim3_chunk_ate": _float(overrow.get("local_sim3_chunk_ate")),
            "H35_minus_C9_gap": h35_gap,
            "near_static_ratio": _float(v67row.get("near_static_ratio")),
            "v62_primary_error_type": taxrow.get("primary_error_type"),
            "repair_mode": str(args.repair_mode),
        }
        if chunk in s5:
            out.update(s5[chunk])
        else:
            out["s5_positive"] = None
            out["s5_label_source"] = None
        rows.append(out)
    if args.repair_mode != "base":
        raw_norm = _robust_norm([_float(r.get("raw_overlap_residual")) for r in rows])
        near_norm = _robust_norm([_float(r.get("near_static_ratio")) for r in rows])
        for idx, row in enumerate(rows):
            Q = _float(row.get("Q_anchor"))
            R = _float(row.get("R_anchor"))
            if args.repair_mode == "road_boundary":
                rb = max(0.0, min(1.0, _float(row.get("road_boundary_anchor_mass"))))
                Q = max(0.0, min(1.0, Q + 0.15 * rb))
                R = max(0.0, min(1.0, R - 0.05 * rb))
            elif args.repair_mode == "near_static":
                near = near_norm.get(idx, 0.0)
                cond_score = max(0.0, min(1.0, _float(row.get("anchor_condition_score"))))
                Q = max(0.0, min(1.0, 0.85 * Q + 0.10 * near + 0.05 * cond_score))
            elif args.repair_mode == "raw_overlap_residual":
                residual = raw_norm.get(idx, 0.0)
                R = max(0.0, min(1.0, R + 0.20 * residual))
            row["Q_anchor"] = Q
            row["R_anchor"] = R
            row["O_anchor"] = Q - R
    meta = {"s5_label_chunks": len(s5), "s5_positive_count": int(sum(1 for r in s5.values() if r.get("s5_positive")))}
    return rows, meta


def _feature_importance(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    features = [
        "Q_anchor",
        "R_anchor",
        "O_anchor",
        "valid_anchor_count",
        "valid_scale_anchor_count",
        "anchor_grid_coverage",
        "anchor_spatial_entropy",
        "anchor_condition_score",
        "anchor_gram_motion_mean",
        "anchor_score_top10_mean",
        "sky_risk_mass",
        "dynamic_risk_mass",
        "vegetation_risk_mass",
        "ground_anchor_mass",
        "semantic_trust_mean",
    ]
    targets = ["head_to_tail_transfer_ratio", "future_after_overlap_error", "intra_scale_variance", "H35_minus_C9_gap"]
    out = []
    for feat in features:
        row: Dict[str, Any] = {"feature": feat}
        vals = [_float(r.get(feat)) for r in rows]
        max_abs = 0.0
        for target in targets:
            corr = _spearman(vals, [_float(r.get(target)) for r in rows])
            row[f"spearman_{target}"] = corr
            if corr is not None:
                max_abs = max(max_abs, abs(corr))
        row["max_abs_spearman"] = max_abs
        out.append(row)
    return sorted(out, key=lambda r: float(r.get("max_abs_spearman") or 0.0), reverse=True)


def _mismatch_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    O = np.asarray([_float(r.get("O_anchor")) for r in rows], dtype=float)
    future = np.asarray([_float(r.get("future_after_overlap_error")) for r in rows], dtype=float)
    head = np.asarray([_float(r.get("head_to_tail_transfer_ratio")) for r in rows], dtype=float)
    finite_O = O[np.isfinite(O)]
    if finite_O.size == 0:
        return []
    low = float(np.quantile(finite_O, 0.30))
    high = float(np.quantile(finite_O, 0.70))
    future_bad = set(i for i, v in enumerate(_top30_labels(future)) if v == 1)
    head_bad = set(i for i, v in enumerate(_top30_labels(head)) if v == 1)
    out = []
    for i, row in enumerate(rows):
        bad = i in future_bad or i in head_bad
        oi = _float(row.get("O_anchor"))
        if not math.isfinite(oi):
            continue
        if oi >= high and bad:
            kind = "high_O_but_bad_failure"
        elif oi <= low and not bad:
            kind = "low_O_but_not_top30_bad"
        else:
            continue
        out.append({**dict(row), "mismatch_type": kind})
    return out


def _save_figures(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("future_after_overlap_error", "O_anchor_vs_future_after_overlap.png"),
        ("head_to_tail_transfer_ratio", "O_anchor_vs_head_tail.png"),
        ("intra_scale_variance", "O_anchor_vs_scale_cv.png"),
    ]
    xs = [_float(r.get("O_anchor")) for r in rows]
    chunks = [int(r["chunk_id"]) for r in rows]
    for target, fname in pairs:
        ys = [_float(r.get(target)) for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
        ax.scatter(xs, ys, s=22)
        for x, y, c in zip(xs, ys, chunks):
            if math.isfinite(x) and math.isfinite(y) and c in {6, 7, 8, 10, 12, 19, 20, 29, 30, 31, 32}:
                ax.text(x, y, str(c), fontsize=7)
        ax.set_xlabel("O_anchor")
        ax.set_ylabel(target)
        ax.set_title(target)
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, meta = _build_rows(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "anchor_predictiveness_rows.csv", rows)
    feature_rows = _feature_importance(rows)
    _write_csv(args.out_dir / "anchor_feature_importance.csv", feature_rows)
    mismatch = _mismatch_rows(rows)
    _write_csv(args.out_dir / "anchor_mismatch_chunks.csv", mismatch)
    s5_rows = [
        {
            "chunk_id": r["chunk_id"],
            "s5_positive": r.get("s5_positive"),
            "head_tail_improvement_vs_baseline_ratio": r.get("head_tail_improvement_vs_baseline_ratio"),
            "overlap_improvement_vs_baseline_ratio": r.get("overlap_improvement_vs_baseline_ratio"),
            "source": r.get("source"),
        }
        for r in rows
        if r.get("s5_positive") is not None
    ]
    _write_csv(args.out_dir / "s5_positive_label_table.csv", s5_rows)
    _save_figures(args.out_dir, rows)

    O = [_float(r.get("O_anchor")) for r in rows]
    future = [_float(r.get("future_after_overlap_error")) for r in rows]
    head = [_float(r.get("head_to_tail_transfer_ratio")) for r in rows]
    scale = [_float(r.get("intra_scale_variance")) for r in rows]
    h35_gap = [_float(r.get("H35_minus_C9_gap")) for r in rows]
    low_O = [-x if math.isfinite(x) else float("nan") for x in O]
    s5_labels = [1 if r.get("s5_positive") is True else 0 for r in rows if r.get("s5_positive") is not None]
    s5_scores = [_float(r.get("O_anchor")) for r in rows if r.get("s5_positive") is not None]
    summary = {
        "schema": "acl2_v69_anchor_predictiveness_v1",
        "anchor_summary": str(args.anchor_summary),
        "v62_report": str(args.v62_report),
        "v67_observability": str(args.v67_observability) if args.v67_observability else None,
        "v68_report": str(args.v68_report) if args.v68_report else None,
        "repair_mode": str(args.repair_mode),
        "num_chunks": len(rows),
        "s5_label_meta": meta,
        "correlations": {
            "spearman_O_anchor_future_after_overlap_error": _spearman(O, future),
            "spearman_O_anchor_head_to_tail_transfer_ratio": _spearman(O, head),
            "spearman_O_anchor_intra_scale_variance": _spearman(O, scale),
            "spearman_O_anchor_H35_minus_C9_gap": _spearman(O, h35_gap),
        },
        "auc": {
            "auc_low_O_top30_future_bad": _auc(low_O, _top30_labels(future)),
            "auc_low_O_top30_head_tail_bad": _auc(low_O, _top30_labels(head)),
            "auc_O_anchor_s5_positive": _auc(s5_scores, s5_labels) if len(set(s5_labels)) > 1 else None,
        },
        "outputs": {
            "rows": str(args.out_dir / "anchor_predictiveness_rows.csv"),
            "feature_importance": str(args.out_dir / "anchor_feature_importance.csv"),
            "mismatch_chunks": str(args.out_dir / "anchor_mismatch_chunks.csv"),
            "s5_positive_label_table": str(args.out_dir / "s5_positive_label_table.csv"),
            "figures": str(args.out_dir / "figures"),
        },
    }
    cor = summary["correlations"]
    auc = summary["auc"]
    gate = {
        "corr_future_le_neg0p30": cor["spearman_O_anchor_future_after_overlap_error"] is not None
        and cor["spearman_O_anchor_future_after_overlap_error"] <= -0.30,
        "corr_head_tail_le_neg0p30": cor["spearman_O_anchor_head_to_tail_transfer_ratio"] is not None
        and cor["spearman_O_anchor_head_to_tail_transfer_ratio"] <= -0.30,
        "auc_low_O_future_or_head_ge_0p65": max(
            [x for x in (auc["auc_low_O_top30_future_bad"], auc["auc_low_O_top30_head_tail_bad"]) if x is not None] or [float("-inf")]
        )
        >= 0.65,
        "auc_s5_ge_0p70": auc["auc_O_anchor_s5_positive"] is not None and auc["auc_O_anchor_s5_positive"] >= 0.70,
    }
    summary["phaseB_gate"] = gate
    summary["phaseB_gate_pass"] = bool(any(gate.values()))
    _write_json(args.out_dir / "anchor_predictiveness_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
