#!/usr/bin/env python3
"""Diagnose v87 Phase8 direct raw-pair merge/gauge weighting counterfactual."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
)
DEFAULT_PHASE2 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance_k16_r1_median_abs_highobs"
)
DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_OUT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase8_merge_gauge_direct_pair_weighting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sigma", type=float, default=-1.0, help="Conflict downweight scale; <=0 uses patch-level conflict q75.")
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--min-raw-points", type=int, default=500)
    return parser.parse_args()


def _norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def _same_count_random_margin(df: pd.DataFrame, score_col: str, case_col: str = "base_case_type") -> dict[str, Any]:
    labelled = df[df[case_col].isin(["bad", "good"])].copy()
    if len(labelled) == 0:
        return {"random_bad_proxy_improvement_p95": None, "actual_bad_proxy_improvement": None, "margin": None}
    score = pd.to_numeric(labelled[score_col], errors="coerce").fillna(0.0)
    threshold = float(score.quantile(0.75))
    flags = score >= threshold
    bad = labelled[case_col] == "bad"
    actual = float((flags & bad).sum() / max(int(bad.sum()), 1))
    random_vals = []
    for salt in range(64):
        order = sorted(range(len(labelled)), key=lambda i: stable_hash_float("v87_phase8_random", salt, i))
        rnd = pd.Series(False, index=labelled.index)
        rnd.iloc[order[: int(flags.sum())]] = True
        random_vals.append(float((rnd & bad).sum() / max(int(bad.sum()), 1)))
    p95 = float(np.quantile(random_vals, 0.95)) if random_vals else None
    return {"random_bad_proxy_improvement_p95": p95, "actual_bad_proxy_improvement": actual, "margin": None if p95 is None else actual - p95}


def _weighted_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> dict[str, Any] | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) < 8:
        return None
    x = src[mask]
    y = dst[mask]
    w = weights[mask]
    w = w / max(float(w.sum()), 1e-12)
    mux = np.sum(x * w[:, None], axis=0)
    muy = np.sum(y * w[:, None], axis=0)
    xc = x - mux
    yc = y - muy
    cov = (yc * w[:, None]).T @ xc
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt
    var_x = float(np.sum(w * np.sum(xc * xc, axis=1)))
    if var_x <= 1e-12:
        return None
    scale = float(np.sum(s * d) / var_x)
    t = muy - scale * (mux @ r.T)
    return {"scale": scale, "R": r, "t": t}


def _apply_sim3(src: np.ndarray, sim3: dict[str, Any]) -> np.ndarray:
    return float(sim3["scale"]) * (src @ np.asarray(sim3["R"], dtype=np.float64).T) + np.asarray(sim3["t"], dtype=np.float64)


def _weighted_rmse(src: np.ndarray, dst: np.ndarray, weights: np.ndarray, sim3: dict[str, Any]) -> float | None:
    pred = _apply_sim3(src, sim3)
    residual = np.linalg.norm(pred - dst, axis=1)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return None
    return float(np.sqrt(np.sum(weights[mask] * residual[mask] * residual[mask]) / max(float(weights[mask].sum()), 1e-12)))


def _load_raw(path: str) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    keys = ["prev_overlap_local_points", "curr_overlap_local_points", "prev_pixel_coords", "prev_conf", "curr_conf", "prev_semantic_conf", "curr_semantic_conf"]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key)
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def _patch_id_from_pixels(pixels: np.ndarray) -> np.ndarray:
    patch_y = np.floor(pixels[:, 0].astype(np.float64) / 14.0).astype(np.int64)
    patch_x = np.floor(pixels[:, 1].astype(np.float64) / 14.0).astype(np.int64)
    patch_y = np.clip(patch_y, 0, 18)
    patch_x = np.clip(patch_x, 0, 65)
    return patch_y * 66 + patch_x


def _source_by_pair(anchor: pd.DataFrame) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for _, row in anchor.drop_duplicates(["seq", "prev_chunk", "curr_chunk", "source_path"]).iterrows():
        key = (str(row["seq"]).zfill(2), int(row["prev_chunk"]), int(row["curr_chunk"]))
        out.setdefault(key, str(row["source_path"]))
    return out


def _patch_stats(group: pd.DataFrame) -> pd.DataFrame:
    patch = group.groupby("prev_patch_id").agg(
        support_weight=("support_weight", "mean"),
        conflict_weight=("conflict_weight", "mean"),
        observability_score=("observability_score", "mean"),
        local_shape_log_ratio_median=("local_shape_log_ratio_median", "mean"),
        confidence_weighted_residual=("confidence_weighted_residual", "mean"),
    )
    return patch.reset_index()


def _map_patch_values(patch_ids: np.ndarray, stats: pd.DataFrame, column: str, default: float) -> np.ndarray:
    values = np.full(patch_ids.shape[0], float(default), dtype=np.float64)
    for _, row in stats.iterrows():
        pid = int(row["prev_patch_id"])
        value = row.get(column)
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            values[patch_ids == pid] = v
    return values


def _candidate_weights(raw: dict[str, np.ndarray], patch_ids: np.ndarray, stats: pd.DataFrame, sigma: float, eta: float) -> dict[str, np.ndarray]:
    conf = np.sqrt(np.clip(raw["prev_conf"].astype(np.float64), 0.0, 1.0) * np.clip(raw["curr_conf"].astype(np.float64), 0.0, 1.0))
    sem_conf = np.sqrt(
        np.clip(raw["prev_semantic_conf"].astype(np.float64), 0.0, 1.0)
        * np.clip(raw["curr_semantic_conf"].astype(np.float64), 0.0, 1.0)
    )
    support = _map_patch_values(patch_ids, stats, "support_weight", 0.0)
    conflict = _map_patch_values(patch_ids, stats, "conflict_weight", 0.0)
    shape = _map_patch_values(patch_ids, stats, "local_shape_log_ratio_median", np.nan)
    overlap = _map_patch_values(patch_ids, stats, "confidence_weighted_residual", np.nan)
    finite_shape = shape[np.isfinite(shape)]
    finite_overlap = overlap[np.isfinite(overlap)]
    shape_scale = float(np.quantile(finite_shape, 0.75)) if finite_shape.size else 1.0
    overlap_scale = float(np.quantile(finite_overlap, 0.75)) if finite_overlap.size else 1.0
    if sigma <= 0:
        finite_conflict = conflict[np.isfinite(conflict) & (conflict > 0)]
        sigma = float(np.quantile(finite_conflict, 0.75)) if finite_conflict.size else 1.0
    geometry_only = conf * np.exp(-np.nan_to_num(shape, nan=shape_scale) / max(shape_scale, 1e-12)) * np.exp(
        -np.nan_to_num(overlap, nan=overlap_scale) / max(overlap_scale, 1e-12)
    )
    full = conf * (1.0 / (1.0 + (conflict / max(sigma, 1e-12)) ** 2)) * (1.0 + eta * support)
    shape_shuffle = np.roll(full, max(1, len(full) // 7))
    semantic_shuffle = full[np.argsort(np.argsort(sem_conf))]
    random_order = np.argsort([stable_hash_float("v87_phase8_raw_random", int(i)) for i in range(len(full))])
    random_w = np.empty_like(full)
    random_w[random_order] = np.sort(full)
    return {
        "native": conf,
        "full_direct_pair": full,
        "geometry_only": geometry_only,
        "same_count_random": random_w,
        "shape_ratio_shuffle": shape_shuffle,
        "semantic_conf_shuffle": semantic_shuffle,
    }


def _raw_counterfactual_rows(rows: pd.DataFrame, anchor_rows: pd.DataFrame, *, sigma: float, eta: float, min_raw_points: int) -> list[dict[str, Any]]:
    source_map = _source_by_pair(anchor_rows)
    out: list[dict[str, Any]] = []
    for (seq, prev, curr), group in rows.groupby(["seq", "prev_chunk", "curr_chunk"]):
        seq = str(seq).zfill(2)
        prev = int(prev)
        curr = int(curr)
        source = source_map.get((seq, prev, curr), "")
        raw = _load_raw(source) if source else None
        if raw is None:
            out.append({"seq": seq, "prev_chunk": prev, "curr_chunk": curr, "fit_status": "missing_raw_overlap", "source_path": source})
            continue
        src = raw["prev_overlap_local_points"].astype(np.float64)
        dst = raw["curr_overlap_local_points"].astype(np.float64)
        pixels = raw["prev_pixel_coords"].astype(np.float64)
        patch_ids = _patch_id_from_pixels(pixels)
        stats = _patch_stats(group)
        known_patch = np.isin(patch_ids, stats["prev_patch_id"].astype(int).to_numpy())
        finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & np.isfinite(pixels).all(axis=1) & known_patch
        idx = np.flatnonzero(finite)
        if idx.size < min_raw_points:
            out.append(
                {
                    "seq": seq,
                    "prev_chunk": prev,
                    "curr_chunk": curr,
                    "fit_status": f"too_few_raw_points:{idx.size}",
                    "source_path": source,
                }
            )
            continue
        train = idx[(np.arange(idx.size) % 10) < 7]
        eval_idx = idx[(np.arange(idx.size) % 10) >= 7]
        weights = _candidate_weights(raw, patch_ids, stats, sigma, eta)
        sims: dict[str, dict[str, Any] | None] = {}
        eval_scores: dict[str, float | None] = {}
        eval_base_w = weights["native"][eval_idx]
        for name, w in weights.items():
            sims[name] = _weighted_sim3(src[train], dst[train], w[train])
            eval_scores[name] = None if sims[name] is None else _weighted_rmse(src[eval_idx], dst[eval_idx], eval_base_w, sims[name])
        native = eval_scores.get("native")
        full = eval_scores.get("full_direct_pair")
        improvement = None if native is None or full is None else (native - full) / max(native, 1e-12)
        row: dict[str, Any] = {
            "seq": seq,
            "prev_chunk": prev,
            "curr_chunk": curr,
            "base_case_type": group["base_case_type"].mode().iloc[0] if "base_case_type" in group else "",
            "quality_type": group["quality_type"].mode().iloc[0] if "quality_type" in group else "",
            "source_path": source,
            "fit_status": "ok",
            "raw_point_count": int(idx.size),
            "train_point_count": int(train.size),
            "eval_point_count": int(eval_idx.size),
            "native_eval_rmse": native,
            "full_direct_pair_eval_rmse": full,
            "full_direct_pair_improvement_vs_native": improvement,
            "full_scale": None if sims.get("full_direct_pair") is None else sims["full_direct_pair"]["scale"],
            "native_scale": None if sims.get("native") is None else sims["native"]["scale"],
        }
        for control in ["geometry_only", "same_count_random", "shape_ratio_shuffle", "semantic_conf_shuffle"]:
            score = eval_scores.get(control)
            row[f"{control}_eval_rmse"] = score
            row[f"full_minus_{control}_improvement_margin"] = None if native is None or full is None or score is None else (score - full) / max(native, 1e-12)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    row_table = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_rows.csv")
    anchor_rows = pd.read_csv(args.anchor_rows, usecols=["seq", "prev_chunk", "curr_chunk", "source_path"])
    proxy = pd.read_csv(args.phase2_dir / "no_gt_scale_proxy_rows.csv")
    for frame in (by_pair, proxy, row_table, anchor_rows):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
        frame["prev_chunk"] = frame["prev_chunk"].astype(int)
        frame["curr_chunk"] = frame["curr_chunk"].astype(int)
    df = by_pair.merge(
        proxy[["seq", "prev_chunk", "curr_chunk", "S_geometry_only", "S_shape", "S_overlap", "abs_log_scale_jump"]],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    conflict = pd.to_numeric(df["conflict_effective_sample_size"], errors="coerce").fillna(0.0)
    support = pd.to_numeric(df["support_effective_sample_size"], errors="coerce").fillna(0.0)
    absence = pd.to_numeric(df["absence_score"], errors="coerce").fillna(1.0)
    base = pd.to_numeric(df["mean_confidence_weighted_overlap_residual"], errors="coerce").fillna(0.0)
    df["merge_weight_support"] = (1.0 + args.eta * _norm(support)).astype(float)
    df["merge_weight_conflict"] = (1.0 / (1.0 + (conflict / max(args.sigma, 1e-12)) ** 2)).astype(float)
    df["merge_direct_pair_weight"] = df["merge_weight_support"] * df["merge_weight_conflict"]
    df["gauge_hold_signal"] = ((conflict >= conflict.quantile(0.75)) & (support <= support.quantile(0.50))).astype(bool)
    df["direct_pair_proxy_risk"] = 0.50 * _norm(conflict) + 0.30 * _norm(absence) + 0.20 * _norm(base)
    df["geometry_only_direct_pair_proxy"] = _norm(df["S_geometry_only"])
    df["same_count_random_proxy"] = _norm(pd.Series([stable_hash_float("v87_phase8_same_count", i) for i in range(len(df))]))
    df["shape_ratio_shuffle_proxy"] = np.roll(df["direct_pair_proxy_risk"].to_numpy(), 1)
    control = _same_count_random_margin(df, "direct_pair_proxy_risk")

    raw_rows = _raw_counterfactual_rows(row_table, anchor_rows, sigma=args.sigma, eta=args.eta, min_raw_points=args.min_raw_points)
    raw_ok = pd.DataFrame([row for row in raw_rows if row.get("fit_status") == "ok"])
    if len(raw_ok):
        raw_ok["full_direct_pair_improvement_vs_native"] = pd.to_numeric(
            raw_ok["full_direct_pair_improvement_vs_native"], errors="coerce"
        )
        raw_bad = raw_ok[raw_ok["base_case_type"] == "bad"].copy()
        raw_good = raw_ok[raw_ok["base_case_type"] == "good"].copy()
        bad_median_improvement = (
            float(raw_bad["full_direct_pair_improvement_vs_native"].median()) if len(raw_bad) else None
        )
        good_worsen = (
            float(np.maximum(-raw_good["full_direct_pair_improvement_vs_native"], 0.0).median()) if len(raw_good) else None
        )
        control_margins = {
            control_name: float(pd.to_numeric(raw_ok[f"full_minus_{control_name}_improvement_margin"], errors="coerce").median())
            for control_name in ["geometry_only", "same_count_random", "shape_ratio_shuffle", "semantic_conf_shuffle"]
            if f"full_minus_{control_name}_improvement_margin" in raw_ok
        }
        raw_sequence_coverage = int(raw_ok["seq"].nunique())
    else:
        bad_median_improvement = None
        good_worsen = None
        control_margins = {}
        raw_sequence_coverage = 0

    labelled = df[df["base_case_type"].isin(["bad", "good"])].copy()
    if len(labelled):
        risk_threshold = float(pd.to_numeric(labelled["direct_pair_proxy_risk"], errors="coerce").quantile(0.75))
        flagged = pd.to_numeric(labelled["direct_pair_proxy_risk"], errors="coerce") >= risk_threshold
        bad = labelled["base_case_type"] == "bad"
        good = labelled["base_case_type"] == "good"
        bad_proxy_recall = float((flagged & bad).sum() / max(int(bad.sum()), 1))
        good_proxy_fpr = float((flagged & good).sum() / max(int(good.sum()), 1))
    else:
        risk_threshold = float("nan")
        bad_proxy_recall = 0.0
        good_proxy_fpr = 1.0

    checks = {
        "direct_pair_weights_available": True,
        "raw_overlap_geometry_counterfactual_available": bool(len(raw_ok) > 0),
        "bad_raw_overlap_improvement_ge_10pct": bad_median_improvement is not None and bad_median_improvement >= 0.10,
        "good_raw_overlap_worsen_le_2pct": good_worsen is not None and good_worsen <= 0.02,
        "raw_overlap_beats_geometry_only": control_margins.get("geometry_only", -1e9) >= 0.0,
        "raw_overlap_beats_same_count_random": control_margins.get("same_count_random", -1e9) >= 0.0,
        "raw_overlap_beats_semantic_or_shape_shuffle": min(
            control_margins.get("shape_ratio_shuffle", -1e9),
            control_margins.get("semantic_conf_shuffle", -1e9),
        )
        >= 0.0,
        "raw_overlap_coverage_ge_3_sequences": raw_sequence_coverage >= 3,
        "actual_geometry_counterfactual_available": False,
        "bad_geometry_improvement_ge_10pct": False,
        "good_geometry_worsen_le_2pct": False,
        "beats_geometry_only_weighting": False,
        "beats_same_count_random": bool(control["margin"] is not None and control["margin"] >= 0.05),
        "beats_semantic_or_shape_shuffle": False,
        "coverage_ge_3_sequences": int(df["seq"].astype(str).str.zfill(2).nunique()) >= 3,
    }
    raw_overlap_gate_pass = all(
        checks[key]
        for key in [
            "raw_overlap_geometry_counterfactual_available",
            "bad_raw_overlap_improvement_ge_10pct",
            "good_raw_overlap_worsen_le_2pct",
            "raw_overlap_beats_geometry_only",
            "raw_overlap_beats_same_count_random",
            "raw_overlap_beats_semantic_or_shape_shuffle",
            "raw_overlap_coverage_ge_3_sequences",
        ]
    )
    gate_pass = all(checks.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "direct_pair_weight_rows.csv", df.to_dict("records"))
    write_csv(args.out_dir / "raw_overlap_geometry_counterfactual_rows.csv", raw_rows)
    write_csv(
        args.out_dir / "direct_pair_weight_controls.csv",
        [
            {"control": "same_count_random", **control},
            {"control": "geometry_only", "note": "proxy only; no runtime geometry counterfactual"},
            {"control": "shape_ratio_shuffle", "note": "proxy only; no runtime geometry counterfactual"},
        ],
    )
    summary = {
        "phase": "Phase8_merge_gauge_direct_pair_weighting",
        "phase8_merge_gauge_gate_pass": gate_pass,
        "phase8_raw_overlap_geometry_gate_pass": raw_overlap_gate_pass,
        "checks": checks,
        "bad_proxy_recall": bad_proxy_recall,
        "good_proxy_fpr": good_proxy_fpr,
        "raw_overlap_valid_rows": int(len(raw_ok)),
        "raw_overlap_sequence_coverage": raw_sequence_coverage,
        "bad_raw_overlap_median_improvement_vs_native": bad_median_improvement,
        "good_raw_overlap_median_worsen_vs_native": good_worsen,
        "raw_overlap_control_margins": control_margins,
        "direct_pair_proxy_threshold_q75": risk_threshold,
        "same_count_random_control": control,
        "raw_overlap_geometry_counterfactual_available": bool(len(raw_ok) > 0),
        "actual_geometry_counterfactual_available": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "blocker": "raw_overlap_direct_pair_counterfactual_failed_or_not_trajectory_geometry",
        "note": "This phase computes direct raw-pair weights from v87 support/conflict/absence and now includes raw-overlap weighted Sim3 residual counterfactuals. It still does not run runtime HMC merge/gauge and does not use v84 support-map fallback.",
    }
    write_json(args.out_dir / "merge_gauge_direct_pair_summary.json", summary)
    report = [
        "# v87 Phase8 Merge/Gauge Direct Pair Weighting",
        "",
        f"- phase8_merge_gauge_gate_pass: `{gate_pass}`",
        f"- raw_overlap_geometry_counterfactual_available: `{bool(len(raw_ok) > 0)}`",
        f"- phase8_raw_overlap_geometry_gate_pass: `{raw_overlap_gate_pass}`",
        f"- actual_trajectory_geometry_counterfactual_available: `False`",
        f"- bad_raw_overlap_median_improvement_vs_native: `{bad_median_improvement}`",
        f"- good_raw_overlap_median_worsen_vs_native: `{good_worsen}`",
        f"- raw_overlap_control_margins: `{control_margins}`",
        f"- bad_proxy_recall: `{bad_proxy_recall}`",
        f"- good_proxy_fpr: `{good_proxy_fpr}`",
        f"- blocker: `{summary['blocker']}`",
        "",
        "Direct raw-pair weights were computed and tested on raw overlap weighted Sim3 residuals. Runtime HMC merge/gauge trajectory geometry remains unavailable, so old v84/v82 support-map or overlap-outlier fallback artifacts are not used as pass evidence.",
    ]
    (args.out_dir / "merge_gauge_direct_pair_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase8_merge_gauge_gate_pass={gate_pass}")
    print(f"phase8_raw_overlap_geometry_gate_pass={raw_overlap_gate_pass}")
    print(f"raw_overlap_valid_rows={len(raw_ok)}")
    print(f"bad_raw_overlap_median_improvement_vs_native={bad_median_improvement}")
    print(f"good_raw_overlap_median_worsen_vs_native={good_worsen}")
    print("actual_geometry_counterfactual_available=False")
    print(f"bad_proxy_recall={bad_proxy_recall}")
    print(f"good_proxy_fpr={good_proxy_fpr}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
