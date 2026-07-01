#!/usr/bin/env python3
"""Fit v87 state-conditioned latent transport on SUPPORT rows only."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import effective_sample_size, stable_hash_float, weighted_residual, write_csv, write_json


DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
)
DEFAULT_FEATURE_PT = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_features.pt"
)
DEFAULT_OUT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase3_state_conditioned_latent_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    parser.add_argument("--min-support-ess", type=float, default=12.0)
    parser.add_argument("--min-heldout-rows", type=int, default=3)
    return parser.parse_args()


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["q_features"].detach().cpu().float().numpy(), payload["k_features"].detach().cpu().float().numpy()


def _fit_near_identity_ridge(q: np.ndarray, k: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    dim = q.shape[1]
    ww = np.sqrt(np.maximum(w, 0.0))[:, None]
    qw = q * ww
    kw = k * ww
    lhs = qw.T @ qw + lam * np.eye(dim)
    rhs = qw.T @ kw + lam * np.eye(dim)
    return np.linalg.solve(lhs, rhs)


def _split_mask(ids: list[str]) -> np.ndarray:
    vals = np.asarray([stable_hash_float("v87_phase3_split", item) for item in ids], dtype=np.float64)
    return vals < 0.70


def _residual(q: np.ndarray, k: np.ndarray, w: np.ndarray, c: np.ndarray | None = None) -> float:
    pred = q if c is None else q @ c
    return weighted_residual(pred, k, w)


def _matrix_stats(c: np.ndarray) -> dict[str, Any]:
    try:
        s = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "fro_norm_C_minus_I": None,
            "spectral_norm_C": None,
            "condition_number_C": None,
            "effective_rank_C": None,
        }
    dim = c.shape[0]
    threshold = max(float(s[0]) * 1e-5, 1e-10) if s.size else 1e-10
    return {
        "fro_norm_C_minus_I": float(np.linalg.norm(c - np.eye(dim))),
        "spectral_norm_C": float(s[0]) if s.size else None,
        "condition_number_C": float(s[0] / max(float(s[-1]), 1e-12)) if s.size else None,
        "effective_rank_C": int(np.sum(s > threshold)),
    }


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_rows.csv")
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    q_all, k_all = _load_features(args.features)
    fit_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for _, pair in by_pair.iterrows():
        seq = str(pair["seq"]).zfill(2)
        prev = int(pair["prev_chunk"])
        curr = int(pair["curr_chunk"])
        group = rows[(rows["seq"].astype(str).str.zfill(2) == seq) & (rows["prev_chunk"].astype(int) == prev) & (rows["curr_chunk"].astype(int) == curr)]
        support = group[group["state_label"].astype(str) == "SUPPORT"].copy()
        conflict = group[group["state_label"].astype(str) == "CONFLICT"].copy()
        support_w = pd.to_numeric(support.get("support_weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        support_ess = effective_sample_size(support_w)
        status = "ok"
        reason = ""
        if len(support) == 0:
            status = "not_fit"
            reason = "no_support_state_rows"
        elif support_ess < args.min_support_ess:
            status = "not_fit"
            reason = f"support_ess_below_min:{support_ess:.6g}"
        train_mask = np.zeros(len(support), dtype=bool)
        if status == "ok":
            train_mask = _split_mask([str(v) for v in support["pair_id"].tolist()])
            if int((~train_mask).sum()) < args.min_heldout_rows:
                status = "not_fit"
                reason = f"heldout_rows_below_min:{int((~train_mask).sum())}"
        if status != "ok":
            fit_rows.append(
                {
                    "seq": seq,
                    "prev_chunk": prev,
                    "curr_chunk": curr,
                    "base_case_type": pair.get("base_case_type"),
                    "quality_type": pair.get("quality_type"),
                    "direction": "current_to_history",
                    "C_family": "C4_near_identity_ridge",
                    "fit_status": status,
                    "invalid_reason": reason,
                    "support_row_count": int(len(support)),
                    "support_effective_sample_size": support_ess,
                    "conflict_row_count": int(len(conflict)),
                    "support_identity_residual": "",
                    "support_aligned_residual": "",
                    "support_alignment_gain": "",
                    "support_actual_minus_random_p95": "",
                    "support_actual_minus_shuffle_p95": "",
                    "conflict_identity_residual": "",
                    "conflict_aligned_residual": "",
                    "conflict_mismatch_score": "",
                    "support_conflict_gap": "",
                    "train_heldout_gap": "",
                    "fro_norm_C_minus_I": "",
                    "spectral_norm_C": "",
                    "condition_number_C": "",
                    "effective_rank_C": "",
                    "overfit_flag": "",
                    "abs_log_scale_jump_gt": pair.get("abs_log_scale_jump_gt"),
                }
            )
            continue
        support_idx = support["anchor_row_index"].astype(int).to_numpy()
        q = q_all[support_idx, : args.feature_dim].astype(np.float64)
        k = k_all[support_idx, : args.feature_dim].astype(np.float64)
        w = support_w
        c = _fit_near_identity_ridge(q[train_mask], k[train_mask], w[train_mask], args.ridge_lambda)
        train_identity = _residual(q[train_mask], k[train_mask], w[train_mask], None)
        train_aligned = _residual(q[train_mask], k[train_mask], w[train_mask], c)
        held_q = q[~train_mask]
        held_k = k[~train_mask]
        held_w = w[~train_mask]
        held_identity = _residual(held_q, held_k, held_w, None)
        held_aligned = _residual(held_q, held_k, held_w, c)
        gain = (held_identity - held_aligned) / max(held_identity, 1e-12)
        random_scores = []
        shuffle_scores = []
        for salt in range(32):
            order = np.argsort([stable_hash_float("v87_random", seq, prev, curr, salt, i) for i in range(len(held_k))])
            shuffled_k = held_k[order]
            random_scores.append((held_identity - _residual(held_q, shuffled_k, held_w, c)) / max(held_identity, 1e-12))
            shuffle_scores.append((held_identity - _residual(held_q, np.roll(held_k, 1, axis=0), held_w, c)) / max(held_identity, 1e-12))
        actual_minus_random = float(gain - np.quantile(random_scores, 0.95)) if random_scores else float("nan")
        actual_minus_shuffle = float(gain - np.quantile(shuffle_scores, 0.95)) if shuffle_scores else float("nan")
        conflict_identity = ""
        conflict_aligned = ""
        mismatch = ""
        gap = ""
        if len(conflict):
            conflict_idx = conflict["anchor_row_index"].astype(int).to_numpy()
            conflict_w = pd.to_numeric(conflict["conflict_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            cq = q_all[conflict_idx, : args.feature_dim].astype(np.float64)
            ck = k_all[conflict_idx, : args.feature_dim].astype(np.float64)
            conflict_identity = _residual(cq, ck, conflict_w, None)
            conflict_aligned = _residual(cq, ck, conflict_w, c)
            mismatch = conflict_aligned
            gap = float(conflict_aligned - held_aligned)
        train_gap = abs((train_identity - train_aligned) / max(train_identity, 1e-12) - gain)
        stats = _matrix_stats(c)
        overfit = bool(train_gap > 0.20 or stats.get("condition_number_C") is None or float(stats.get("condition_number_C") or 0) > 1e4)
        key = f"{seq}_{prev}_{curr}_C4_near_identity_ridge"
        matrices[key] = c.astype(np.float32)
        fit_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "base_case_type": pair.get("base_case_type"),
                "quality_type": pair.get("quality_type"),
                "direction": "current_to_history",
                "C_family": "C4_near_identity_ridge",
                "fit_status": "ok",
                "invalid_reason": "",
                "support_row_count": int(len(support)),
                "support_effective_sample_size": support_ess,
                "conflict_row_count": int(len(conflict)),
                "support_identity_residual": held_identity,
                "support_aligned_residual": held_aligned,
                "support_alignment_gain": gain,
                "support_actual_minus_random_p95": actual_minus_random,
                "support_actual_minus_shuffle_p95": actual_minus_shuffle,
                "conflict_identity_residual": conflict_identity,
                "conflict_aligned_residual": conflict_aligned,
                "conflict_mismatch_score": mismatch,
                "support_conflict_gap": gap,
                "train_heldout_gap": train_gap,
                **stats,
                "overfit_flag": overfit,
                "abs_log_scale_jump_gt": pair.get("abs_log_scale_jump_gt"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "state_conditioned_c_fit_rows.csv", fit_rows)
    if matrices:
        np.savez(args.out_dir / "state_conditioned_c_matrices.npz", **matrices)
    invalid_reason_counts = {str(k): int(v) for k, v in pd.Series([row.get("invalid_reason") for row in fit_rows]).value_counts().items()}
    summary = {
        "phase": "Phase3_state_conditioned_latent_transport_fit",
        "pair_rows": len(fit_rows),
        "valid_fit_rows": sum(1 for row in fit_rows if row.get("fit_status") == "ok"),
        "not_fit_rows": sum(1 for row in fit_rows if row.get("fit_status") != "ok"),
        "invalid_reason_counts": invalid_reason_counts,
        "feature_dim": args.feature_dim,
        "ridge_lambda": args.ridge_lambda,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "state_conditioned_fit_summary.json", summary)
    print(f"valid_fit_rows={summary['valid_fit_rows']}")
    print(f"not_fit_rows={summary['not_fit_rows']}")
    print(f"invalid_reason_counts={summary['invalid_reason_counts']}")


if __name__ == "__main__":
    main()
