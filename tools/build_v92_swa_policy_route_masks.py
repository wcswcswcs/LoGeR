#!/usr/bin/env python3
"""Build v92 Phase4 SWA query/pair policy route mask variants."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v91_external_mask_materialization import (  # noqa: E402
    PATCH_TOKEN_COUNT,
    _positions_for_pair,
    _select_tracklets,
)
from tools.v86_soft_latent_utils import write_csv, write_json  # noqa: E402
from tools.v91_semantic_regime_utils import normalize_pair_columns  # noqa: E402
from tools.v92_semantic_policy_carrier_utils import ROOT, RISK_STATES, V91_ROOT, seq_text  # noqa: E402


DEFAULT_OUT = ROOT / "phase4_swa_policy_route_masks"
DEFAULT_POLICY_ROWS = ROOT / "phase1_semantic_policy_row_bank/semantic_policy_rows.csv"
DEFAULT_TRACKLET_ROWS = V91_ROOT / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_POLICY_ROWS)
    parser.add_argument("--tracklet-rows", type=Path, default=DEFAULT_TRACKLET_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _stable_random_score(pair_id: str, salt: float) -> float:
    x = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(pair_id))) + salt
    return float(math.sin(x * 12.9898) * 43758.5453 % 1.0)


def _geometry_state(policy: pd.DataFrame) -> pd.Series:
    risk_count = int(policy["policy_state"].astype(str).isin(RISK_STATES).sum())
    if risk_count <= 0:
        return pd.Series(["HOLD"] * len(policy), index=policy.index)
    score_cols = ["B_proxy", "boundary_mass", "H_mode", "S_context", "observability_score"]
    score = pd.Series(0.0, index=policy.index, dtype=float)
    for col in score_cols:
        if col in policy.columns:
            values = pd.to_numeric(policy[col], errors="coerce").fillna(0.0).astype(float)
            denom = max(float(values.max() - values.min()), 1e-12)
            score = score + (values - float(values.min())) / denom
    chosen = set(score.sort_values(ascending=False).head(risk_count).index.tolist())
    return pd.Series(["RESET_RISK" if idx in chosen else "HOLD" for idx in policy.index], index=policy.index)


def _same_count_random_state(policy: pd.DataFrame) -> pd.Series:
    risk_count = int(policy["policy_state"].astype(str).isin(RISK_STATES).sum())
    chosen = set(
        policy.assign(_rand=[_stable_random_score(pid, 9200.0) for pid in policy["pair_id"].astype(str)])
        .sort_values("_rand", ascending=False)
        .head(risk_count)
        .index.tolist()
    )
    return pd.Series(["RESET_RISK" if idx in chosen else "HOLD" for idx in policy.index], index=policy.index)


def _variant_specs(policy: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"variant": "v92_policy_risk_pair_mask", "state_col": "policy_state", "description": "actual Phase1 semantic policy risk states"},
        {"variant": "v92_semantic_shuffle_pair_mask", "state_col": "semantic_shuffle_state", "description": "Phase1 semantic label shuffle control states"},
        {"variant": "v92_component_shuffle_pair_mask", "state_col": "component_shuffle_state", "description": "Phase1 component shuffle control states"},
        {"variant": "v92_regime_shuffle_pair_mask", "state_col": "regime_shuffle_state", "description": "Phase1 regime shuffle control states"},
        {
            "variant": "v92_same_count_random_pair_mask",
            "state_values": _same_count_random_state(policy),
            "description": "deterministic same-count random risk rows",
        },
        {
            "variant": "v92_geometry_only_risk_pair_mask",
            "state_values": _geometry_state(policy),
            "description": "no-GT geometry/proxy score top-count risk rows",
        },
    ]


def _materialize_variant(
    *,
    variant: str,
    policy: pd.DataFrame,
    tracklets_by_pair: dict[str, pd.DataFrame],
    state_values: pd.Series,
    description: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    position_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    variant_position_count = 0
    variant_source_count = 0
    variant_query_count = 0
    variant_points = 0
    selected_pair_count = 0
    for idx, prow in policy.iterrows():
        control_state = str(state_values.loc[idx])
        if control_state not in RISK_STATES:
            continue
        selected_pair_count += 1
        pid = str(prow["pair_id"])
        pair_tracklets = tracklets_by_pair.get(pid, pd.DataFrame())
        select_row = prow.copy()
        select_row["policy_state"] = control_state
        selected_tracklets = _select_tracklets(pair_tracklets, select_row)
        raw_path = Path(str(prow.get("source_path") or ""))
        positions, pos_summary = _positions_for_pair(raw_path, selected_tracklets)
        variant_position_count += int(len(positions))
        variant_source_count += int(sum(1 for row in positions if row["side"] == "source"))
        variant_query_count += int(sum(1 for row in positions if row["side"] == "query"))
        variant_points += int(pos_summary.get("selected_overlap_points") or 0)
        tracklet_type_counts = (
            selected_tracklets["tracklet_type"].astype(str).value_counts().to_dict()
            if not selected_tracklets.empty and "tracklet_type" in selected_tracklets.columns
            else {}
        )
        pair_rows.append(
            {
                "variant": variant,
                "seq": seq_text(prow["seq"]),
                "prev_chunk": int(prow["prev_chunk"]),
                "curr_chunk": int(prow["curr_chunk"]),
                "pair_id": pid,
                "original_policy_state": str(prow.get("policy_state", "")),
                "route_policy_state": control_state,
                "regime": str(prow.get("regime", "")),
                "description": description,
                "selected_tracklet_type_counts": tracklet_type_counts,
                "source_path": str(raw_path),
                **pos_summary,
            }
        )
        for pos in positions:
            position_rows.append(
                {
                    "variant": variant,
                    "seq": seq_text(prow["seq"]),
                    "prev_chunk": int(prow["prev_chunk"]),
                    "curr_chunk": int(prow["curr_chunk"]),
                    "pair_id": pid,
                    "original_policy_state": str(prow.get("policy_state", "")),
                    "route_policy_state": control_state,
                    "policy_state": control_state,
                    "regime": str(prow.get("regime", "")),
                    "source_path": str(raw_path),
                    **pos,
                }
            )
    variant_row = {
        "variant": variant,
        "description": description,
        "policy_pair_rows": int(selected_pair_count),
        "position_rows": int(variant_position_count),
        "query_unique_patch_positions_sum": int(variant_query_count),
        "source_unique_patch_positions_sum": int(variant_source_count),
        "selected_overlap_points_sum": int(variant_points),
        "source_patch_density_vs_3frame_overlap": float(
            variant_source_count / max(1, 3 * PATCH_TOKEN_COUNT * max(1, selected_pair_count))
        ),
    }
    return position_rows, pair_rows, variant_row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = normalize_pair_columns(pd.read_csv(args.policy_rows))
    policy["seq"] = policy["seq"].map(seq_text)
    tracklets = normalize_pair_columns(pd.read_csv(args.tracklet_rows))
    tracklets_by_pair = {str(pid): group.copy() for pid, group in tracklets.groupby("pair_id")}

    all_positions: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for spec in _variant_specs(policy):
        state_values = spec.get("state_values")
        if state_values is None:
            state_values = policy[str(spec["state_col"])].astype(str)
        positions, pairs, variant_row = _materialize_variant(
            variant=str(spec["variant"]),
            policy=policy,
            tracklets_by_pair=tracklets_by_pair,
            state_values=state_values,
            description=str(spec["description"]),
        )
        all_positions.extend(positions)
        all_pair_rows.extend(pairs)
        variant_rows.append(variant_row)

    write_csv(args.out_dir / "v92_swa_policy_route_mask_positions.csv", all_positions)
    write_csv(args.out_dir / "v92_swa_policy_route_mask_pair_summary.csv", all_pair_rows)
    write_csv(args.out_dir / "v92_swa_policy_route_mask_variant_summary.csv", variant_rows)
    summary = {
        "phase": "Phase4_v92_swa_policy_route_mask_materialization",
        "policy_rows": int(len(policy)),
        "variants": variant_rows,
        "position_rows": int(len(all_positions)),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "selection_rule": "Rows are selected from Phase1 states or controls; selected tracklets are mapped to query/source patch positions from raw overlap pairs.",
        "no_success_claim": "These CSVs are route-dump inputs only; carrier success requires true route/control smoke and audit.",
    }
    write_json(args.out_dir / "materialization_summary.json", summary)
    print(f"policy_rows={summary['policy_rows']}")
    print(f"position_rows={summary['position_rows']}")
    for row in variant_rows:
        print(
            "variant={variant} policy_pair_rows={policy_pair_rows} "
            "query_positions={query_unique_patch_positions_sum} source_positions={source_unique_patch_positions_sum}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
