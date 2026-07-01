#!/usr/bin/env python3
"""Fit ACL2 v86 robust low-degree latent transports and heldout controls."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import effective_sample_size, stable_hash_float, weighted_rank, weighted_residual, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")
DEFAULT_FEATURE_PT = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_features.pt"
)
DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase2_robust_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--control-permutations", type=int, default=64)
    return parser.parse_args()


def _load_features(path: Path, dim: int) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    q = payload["q_features"].detach().cpu().float().numpy()[:, :dim].astype(np.float64)
    k = payload["k_features"].detach().cpu().float().numpy()[:, :dim].astype(np.float64)
    return q, k


def _group_id(row: pd.Series) -> str:
    frame = row.get("curr_frame_id")
    patch = row.get("curr_patch_id")
    try:
        frame_i = int(float(frame))
    except (TypeError, ValueError):
        frame_i = -1
    try:
        patch_i = int(float(patch))
    except (TypeError, ValueError):
        patch_i = -1
    if frame_i >= 0:
        return f"frame{frame_i}_patchbin{patch_i // 8}"
    return f"patchbin{patch_i // 8}"


def _split_indices(pair_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({_group_id(row) for _, row in pair_rows.iterrows()})
    groups = sorted(groups, key=lambda g: stable_hash_float(pair_rows.iloc[0]["seq"], pair_rows.iloc[0]["prev_chunk"], pair_rows.iloc[0]["curr_chunk"], g))
    if len(groups) <= 1:
        idx = np.arange(len(pair_rows))
        return idx[: max(1, len(idx) - 1)], idx[max(1, len(idx) - 1) :]
    split = max(1, min(len(groups) - 1, int(round(0.70 * len(groups)))))
    train_groups = set(groups[:split])
    train_mask = np.asarray([_group_id(row) in train_groups for _, row in pair_rows.iterrows()], dtype=bool)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    if test_idx.size == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    if train_idx.size == 0:
        train_idx = test_idx[:1]
    return train_idx, test_idx


def _ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    xw = X * w[:, None]
    a = Y.T @ xw + lam * np.eye(d)
    b = X.T @ xw + lam * np.eye(d)
    return a @ np.linalg.pinv(b)


def _scalar(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    denom = float(np.sum(w * np.sum(X * X, axis=1)))
    numer = float(np.sum(w * np.sum(Y * X, axis=1)))
    s = numer / max(denom, 1e-12)
    return np.eye(X.shape[1]) * s


def _diagonal(X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    denom = np.sum(w[:, None] * X * X, axis=0) + lam
    numer = np.sum(w[:, None] * Y * X, axis=0) + lam
    return np.diag(numer / np.maximum(denom, 1e-12))


def _procrustes(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    h = Y.T @ (X * w[:, None])
    u, _, vt = np.linalg.svd(h, full_matrices=False)
    c = u @ vt
    if np.linalg.det(c) < 0:
        u[:, -1] *= -1.0
        c = u @ vt
    return c


def _low_rank_residual(base: np.ndarray, rank: int = 2) -> np.ndarray:
    d = base.shape[0]
    delta = base - np.eye(d)
    u, s, vt = np.linalg.svd(delta, full_matrices=False)
    r = min(rank, len(s))
    return np.eye(d) + (u[:, :r] * s[:r]) @ vt[:r]


def _fit_family(family: str, X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    if family == "C0_identity":
        return np.eye(d)
    if family == "C1_scalar_identity":
        return _scalar(X, Y, w)
    if family == "C2_diagonal_near_identity":
        return _diagonal(X, Y, w, lam)
    if family == "C3_weighted_orthogonal_procrustes":
        return _procrustes(X, Y, w)
    if family == "C4_near_identity_ridge":
        return _ridge(X, Y, w, lam)
    if family == "C5_low_rank_residual_rank2":
        return _low_rank_residual(_ridge(X, Y, w, lam), rank=2)
    if family == "C6_full_rank_ridge_upper_bound":
        return _ridge(X, Y, w, 1e-6)
    raise ValueError(f"unknown family {family}")


def _apply(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ C.T


def _gain(identity_resid: float, aligned_resid: float) -> float:
    if not math.isfinite(identity_resid):
        return float("nan")
    return float((identity_resid - aligned_resid) / max(abs(identity_resid), 1e-12))


def _matrix_stats(C: np.ndarray) -> dict[str, float]:
    s = np.linalg.svd(C, compute_uv=False)
    cond = float(s[0] / max(s[-1], 1e-12)) if s.size else float("inf")
    return {
        "condition_number": cond,
        "spectral_norm": float(s[0]) if s.size else float("nan"),
        "fro_norm_C_minus_I": float(np.linalg.norm(C - np.eye(C.shape[0]), ord="fro")),
    }


def _control_gains(
    C: np.ndarray,
    Xh: np.ndarray,
    Yh: np.ndarray,
    wh: np.ndarray,
    seed_parts: list[Any],
    permutations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = Xh.shape[0]
    if n <= 1:
        return rows
    for control in [
        "R1_same_query_random_key",
        "R2_same_key_random_query",
        "R3_same_pair_mass_random",
        "R4_semantic_shuffle_pair",
        "R5_geometry_shuffle_pair",
        "R6_frame_offset_pair",
        "R7_source_only_anchor_control",
    ]:
        for perm_id in range(permutations):
            rng_seed = int(stable_hash_float(*seed_parts, control, perm_id) * (2**32 - 1))
            rng = np.random.default_rng(rng_seed)
            perm = rng.permutation(n)
            if control in {"R1_same_query_random_key", "R4_semantic_shuffle_pair", "R6_frame_offset_pair"}:
                Xc, Yc = Xh, Yh[perm]
            elif control in {"R2_same_key_random_query", "R5_geometry_shuffle_pair"}:
                Xc, Yc = Xh[perm], Yh
            else:
                Xc, Yc = Xh[perm], Yh[np.roll(perm, 1)]
            identity = weighted_residual(Yc, Xc, wh)
            aligned = weighted_residual(Yc, _apply(C, Xc), wh)
            rows.append(
                {
                    "control_type": control,
                    "perm_id": perm_id,
                    "control_identity_residual": identity,
                    "control_aligned_residual": aligned,
                    "control_alignment_gain": _gain(identity, aligned),
                }
            )
    return rows


def _overfit_flags(
    *,
    family: str,
    feature_dim: int,
    neff: float,
    rank_q: int,
    rank_k: int,
    matrix_stats: dict[str, float],
    alignment_gain: float,
    train_resid: float,
    heldout_resid: float,
) -> list[str]:
    flags: list[str] = []
    if matrix_stats["condition_number"] > 100:
        flags.append("condition_number_gt_100")
    if family not in {"C0_identity", "C6_full_rank_ridge_upper_bound"}:
        if matrix_stats["fro_norm_C_minus_I"] / math.sqrt(max(feature_dim, 1)) > 0.50:
            flags.append("fro_norm_action_bound_gt_0p50")
    if alignment_gain < 0:
        flags.append("heldout_gain_lt_0")
    gap = abs(train_resid - heldout_resid) / max(abs(train_resid), 1e-12)
    if gap > 0.20:
        flags.append("train_heldout_gap_gt_0p20")
    if neff < 3 * feature_dim:
        flags.append("effective_sample_lt_3d")
    if rank_q < 0.75 * feature_dim:
        flags.append("weighted_rank_q_lt_0p75d")
    if rank_k < 0.75 * feature_dim:
        flags.append("weighted_rank_k_lt_0p75d")
    return flags


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.phase1_dir / "soft_pair_rows.csv")
    by_pair = pd.read_csv(args.phase1_dir / "soft_pair_by_seq_chunk.csv")
    q_all, k_all = _load_features(args.features, args.feature_dim)
    support_col = f"support_sufficient_for_dim{args.feature_dim}"
    if support_col in by_pair.columns:
        sufficient = by_pair[by_pair[support_col].astype(str).str.lower() == "true"].copy()
    else:
        rank_floor = math.ceil(0.75 * args.feature_dim)
        sufficient = by_pair[
            (pd.to_numeric(by_pair["effective_sample_size"], errors="coerce") >= 3 * args.feature_dim)
            & (pd.to_numeric(by_pair["weighted_rank_q"], errors="coerce") >= rank_floor)
            & (pd.to_numeric(by_pair["weighted_rank_k"], errors="coerce") >= rank_floor)
        ].copy()

    fit_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    overfit_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    families = [
        "C0_identity",
        "C1_scalar_identity",
        "C2_diagonal_near_identity",
        "C3_weighted_orthogonal_procrustes",
        "C4_near_identity_ridge",
        "C5_low_rank_residual_rank2",
        "C6_full_rank_ridge_upper_bound",
    ]
    directions = [
        ("current_to_history", "q_to_k"),
        ("history_to_current", "k_to_q"),
    ]

    for _, pair in sufficient.iterrows():
        mask = (
            (rows["seq"].astype(str).str.zfill(2) == str(pair["seq"]).zfill(2))
            & (rows["prev_chunk"].astype(int) == int(pair["prev_chunk"]))
            & (rows["curr_chunk"].astype(int) == int(pair["curr_chunk"]))
            & (rows["w_fit"].astype(float) > 0.0)
        )
        pair_rows = rows[mask].reset_index(drop=True)
        if len(pair_rows) < 4:
            continue
        local_train, local_test = _split_indices(pair_rows)
        global_indices = pair_rows["anchor_row_index"].astype(int).to_numpy()
        w_all = pair_rows["w_fit"].astype(float).to_numpy()
        for direction, direction_short in directions:
            if direction_short == "q_to_k":
                Xfull, Yfull = q_all[global_indices], k_all[global_indices]
            else:
                Xfull, Yfull = k_all[global_indices], q_all[global_indices]
            Xtr, Ytr, wtr = Xfull[local_train], Yfull[local_train], w_all[local_train]
            Xh, Yh, wh = Xfull[local_test], Yfull[local_test], w_all[local_test]
            neff = effective_sample_size(w_all)
            rank_q = weighted_rank(q_all[global_indices], w_all)
            rank_k = weighted_rank(k_all[global_indices], w_all)
            for family in families:
                C = _fit_family(family, Xtr, Ytr, wtr, args.ridge_lambda)
                key = (
                    f"seq{str(pair['seq']).zfill(2)}_c{int(pair['prev_chunk']):03d}_{int(pair['curr_chunk']):03d}_"
                    f"{direction}_{family}"
                )
                matrices[key] = C.astype(np.float32)
                train_identity = weighted_residual(Ytr, Xtr, wtr)
                train_aligned = weighted_residual(Ytr, _apply(C, Xtr), wtr)
                heldout_identity = weighted_residual(Yh, Xh, wh)
                heldout_aligned = weighted_residual(Yh, _apply(C, Xh), wh)
                align_gain = _gain(heldout_identity, heldout_aligned)
                controls = _control_gains(
                    C,
                    Xh,
                    Yh,
                    wh,
                    [pair["seq"], pair["prev_chunk"], pair["curr_chunk"], direction, family],
                    args.control_permutations,
                )
                for row in controls:
                    row.update(
                        {
                            "seq": str(pair["seq"]).zfill(2),
                            "prev_chunk": int(pair["prev_chunk"]),
                            "curr_chunk": int(pair["curr_chunk"]),
                            "case_label": pair["case_label"],
                            "C_family": family,
                            "direction": direction,
                        }
                    )
                control_rows.extend(controls)
                random_gains = [
                    r["control_alignment_gain"]
                    for r in controls
                    if str(r["control_type"]).startswith(("R1", "R2", "R3")) and math.isfinite(r["control_alignment_gain"])
                ]
                shuffle_gains = [
                    r["control_alignment_gain"]
                    for r in controls
                    if str(r["control_type"]).startswith(("R4", "R5", "R6", "R7"))
                    and math.isfinite(r["control_alignment_gain"])
                ]
                random_p95 = float(np.percentile(random_gains, 95)) if random_gains else float("nan")
                shuffle_p95 = float(np.percentile(shuffle_gains, 95)) if shuffle_gains else float("nan")
                stats = _matrix_stats(C)
                flags = _overfit_flags(
                    family=family,
                    feature_dim=args.feature_dim,
                    neff=neff,
                    rank_q=rank_q,
                    rank_k=rank_k,
                    matrix_stats=stats,
                    alignment_gain=align_gain,
                    train_resid=train_aligned,
                    heldout_resid=heldout_aligned,
                )
                valid = (
                    family not in {"C0_identity", "C6_full_rank_ridge_upper_bound"}
                    and not flags
                    and align_gain >= 0.05
                    and (align_gain - random_p95) >= 0.03
                    and (align_gain - shuffle_p95) >= 0.03
                )
                fit_row = {
                    "seq": str(pair["seq"]).zfill(2),
                    "prev_chunk": int(pair["prev_chunk"]),
                    "curr_chunk": int(pair["curr_chunk"]),
                    "case_label": pair["case_label"],
                    "support_state": pair["support_state_preliminary"],
                    "C_family": family,
                    "direction": direction,
                    "feature_dim": args.feature_dim,
                    "fit_pair_count": int(local_train.size),
                    "heldout_pair_count": int(local_test.size),
                    "effective_sample_size": neff,
                    "weighted_rank_q": rank_q,
                    "weighted_rank_k": rank_k,
                    "condition_number": stats["condition_number"],
                    "spectral_norm": stats["spectral_norm"],
                    "fro_norm_C_minus_I": stats["fro_norm_C_minus_I"],
                    "train_identity_residual": train_identity,
                    "train_residual": train_aligned,
                    "heldout_identity_residual": heldout_identity,
                    "heldout_aligned_residual": heldout_aligned,
                    "alignment_gain": align_gain,
                    "random_p95_gain": random_p95,
                    "shuffle_p95_gain": shuffle_p95,
                    "actual_minus_random_p95": align_gain - random_p95,
                    "actual_minus_shuffle_p95": align_gain - shuffle_p95,
                    "train_heldout_gap": abs(train_aligned - heldout_aligned) / max(abs(train_aligned), 1e-12),
                    "overfit_flag": bool(flags),
                    "overfit_reasons": ";".join(flags),
                    "valid_for_next_phase": valid,
                    "c_matrix_key": key,
                }
                fit_rows.append(fit_row)
                matrix_rows.append({k: fit_row[k] for k in fit_row if k in {
                    "seq",
                    "prev_chunk",
                    "curr_chunk",
                    "case_label",
                    "C_family",
                    "direction",
                    "feature_dim",
                    "condition_number",
                    "spectral_norm",
                    "fro_norm_C_minus_I",
                    "c_matrix_key",
                }})
                if flags:
                    overfit_rows.append(
                        {
                            "seq": fit_row["seq"],
                            "prev_chunk": fit_row["prev_chunk"],
                            "curr_chunk": fit_row["curr_chunk"],
                            "case_label": fit_row["case_label"],
                            "C_family": family,
                            "direction": direction,
                            "overfit_reasons": ";".join(flags),
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "c_fit_rows.csv", fit_rows)
    write_csv(args.out_dir / "heldout_alignment_rows.csv", fit_rows)
    write_csv(args.out_dir / "control_alignment_rows.csv", control_rows)
    write_csv(args.out_dir / "c_matrix_stats.csv", matrix_rows)
    write_csv(args.out_dir / "overfit_flags.csv", overfit_rows)
    np.savez_compressed(args.out_dir / "c_matrices.npz", **matrices)
    valid_rows = [row for row in fit_rows if row["valid_for_next_phase"] and row["direction"] == "current_to_history"]
    summary = {
        "phase": "Phase2_robust_transport_fit",
        "feature_dim": args.feature_dim,
        "support_sufficient_input_pairs": int(len(sufficient)),
        "support_sufficient_filter": support_col if support_col in by_pair.columns else "computed_from_effective_sample_and_weighted_rank",
        "fit_rows": int(len(fit_rows)),
        "valid_current_to_history_rows": int(len(valid_rows)),
        "bad_valid_current_to_history_rows": int(sum(1 for row in valid_rows if row["case_label"] == "bad")),
        "sequence_coverage_valid_current_to_history": int(len({row["seq"] for row in valid_rows})),
        "families": families,
        "directions": [item[0] for item in directions],
        "control_permutations": args.control_permutations,
        "notes": [
            "C6 full-rank ridge is diagnostic upper bound only and is never valid_for_next_phase.",
            "Controls are deterministic heldout feature shuffles from the same artifact universe.",
            "No route mass, geometry action, or TTT is run in Phase2.",
        ],
    }
    write_json(args.out_dir / "c_fit_summary.json", summary)
    print(f"fit_rows={len(fit_rows)}")
    print(f"valid_current_to_history_rows={summary['valid_current_to_history_rows']}")
    print(f"bad_valid_current_to_history_rows={summary['bad_valid_current_to_history_rows']}")
    print(f"sequence_coverage_valid_current_to_history={summary['sequence_coverage_valid_current_to_history']}")


if __name__ == "__main__":
    main()
