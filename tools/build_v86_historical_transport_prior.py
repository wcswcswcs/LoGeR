#!/usr/bin/env python3
"""Build ACL2 v86 historical transport prior diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import effective_sample_size, pair_key, weighted_residual, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")
DEFAULT_PHASE2 = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase2_robust_transport_ridge10")
DEFAULT_FEATURE_PT = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_features.pt"
)
DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase3_historical_prior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    return parser.parse_args()


def _load_features(path: Path, dim: int) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload["q_features"].detach().cpu().float().numpy()[:, :dim].astype(np.float64),
        payload["k_features"].detach().cpu().float().numpy()[:, :dim].astype(np.float64),
    )


def _best_valid_c(phase2_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = pd.read_csv(phase2_dir / "c_fit_rows.csv")
    matrices = np.load(phase2_dir / "c_matrices.npz")
    valid = rows[
        (rows["direction"] == "current_to_history")
        & (rows["valid_for_next_phase"].astype(str).str.lower() == "true")
    ].copy()
    best: dict[tuple[str, int, int], dict[str, Any]] = {}
    for _, row in valid.sort_values("alignment_gain", ascending=False).iterrows():
        key = (str(row["seq"]).zfill(2), int(row["prev_chunk"]), int(row["curr_chunk"]))
        if key in best:
            continue
        c_key = str(row["c_matrix_key"])
        best[key] = {
            "C": matrices[c_key].astype(np.float64),
            "C_family": row["C_family"],
            "fit_quality": float(row["alignment_gain"]),
            "c_matrix_key": c_key,
        }
    return best


def main() -> None:
    args = parse_args()
    soft = pd.read_csv(args.phase1_dir / "soft_pair_rows.csv")
    by_pair = pd.read_csv(args.phase1_dir / "soft_pair_by_seq_chunk.csv")
    q_all, k_all = _load_features(args.features, args.feature_dim)
    valid_c = _best_valid_c(args.phase2_dir)

    rows: list[dict[str, Any]] = []
    for seq, seq_pairs in by_pair.sort_values(["seq", "curr_chunk"]).groupby(by_pair["seq"].astype(str).str.zfill(2)):
        prior: np.ndarray | None = None
        prior_age = 0
        prior_valid_count = 0
        for _, pair in seq_pairs.iterrows():
            key = (str(pair["seq"]).zfill(2), int(pair["prev_chunk"]), int(pair["curr_chunk"]))
            pair_rows = soft[
                (soft["seq"].astype(str).str.zfill(2) == key[0])
                & (soft["prev_chunk"].astype(int) == key[1])
                & (soft["curr_chunk"].astype(int) == key[2])
            ]
            idxs = pair_rows["anchor_row_index"].astype(int).to_numpy()
            weights = pair_rows["w_fit"].astype(float).to_numpy()
            support = effective_sample_size(weights)
            if prior is not None and np.sum(weights > 0) >= 2 and np.sum(weights) > 0:
                q = q_all[idxs]
                k = k_all[idxs]
                identity = weighted_residual(k, q, weights)
                prior_resid = weighted_residual(k, q @ prior.T, weights)
                prior_mismatch = (prior_resid - identity) / max(abs(identity), 1e-12)
                prior_available = True
            else:
                identity = None
                prior_resid = None
                prior_mismatch = None
                prior_available = False
            if key in valid_c:
                C_valid = valid_c[key]["C"]
                if prior is None:
                    prior = C_valid.copy()
                else:
                    prior = args.ema_alpha * C_valid + (1.0 - args.ema_alpha) * prior
                prior_valid_count += 1
                prior_age = 0
                update_status = "UPDATE"
                current_C_distance = 0.0
                fit_quality = valid_c[key]["fit_quality"]
            else:
                prior_age += 1 if prior is not None else 0
                update_status = str(pair.get("support_state_preliminary"))
                current_C_distance = None
                fit_quality = None
            rows.append(
                {
                    "seq": key[0],
                    "prev_chunk": key[1],
                    "curr_chunk": key[2],
                    "case_label": pair.get("case_label"),
                    "quality_label": pair.get("quality_label"),
                    "historical_prior_available": prior_available,
                    "prior_age": prior_age if prior is not None else "",
                    "prior_valid_count": prior_valid_count,
                    "current_support_state": pair.get("support_state_preliminary"),
                    "current_weighted_support": support,
                    "current_identity_residual": identity,
                    "current_prior_residual": prior_resid,
                    "prior_mismatch_score": prior_mismatch,
                    "prior_conflict_score": max(0.0, float(prior_mismatch)) if prior_mismatch is not None else None,
                    "anchor_absence_score": pair.get("anchor_absence_score"),
                    "low_observability_score": pair.get("anchor_absence_score"),
                    "current_C_vs_prior_C_distance": current_C_distance,
                    "prior_fit_quality": fit_quality,
                    "state_after_pair": update_status,
                    "valid_C_used_for_update": key in valid_c,
                    "valid_C_family": valid_c.get(key, {}).get("C_family", ""),
                    "valid_C_matrix_key": valid_c.get(key, {}).get("c_matrix_key", ""),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "historical_prior_rows.csv", rows)
    summary = {
        "phase": "Phase3_historical_transport_prior",
        "rows": len(rows),
        "valid_C_update_count": len(valid_c),
        "prior_available_rows": sum(1 for row in rows if row["historical_prior_available"]),
        "bad_prior_available_rows": sum(1 for row in rows if row["case_label"] == "bad" and row["historical_prior_available"]),
        "ema_alpha": args.ema_alpha,
        "source_phase2_dir": str(args.phase2_dir),
        "note": "Prior updates use only earlier valid current-to-history C rows within each sequence.",
    }
    write_json(args.out_dir / "historical_prior_summary.json", summary)
    print(f"valid_C_update_count={summary['valid_C_update_count']}")
    print(f"prior_available_rows={summary['prior_available_rows']}")
    print(f"bad_prior_available_rows={summary['bad_prior_available_rows']}")


if __name__ == "__main__":
    main()
