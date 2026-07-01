#!/usr/bin/env python3
"""Build v93 Phase7 SWA secondary query/pair route mask variants."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v91_external_mask_materialization import (  # noqa: E402
    PATCH_TOKEN_COUNT,
    RISK_STATES,
    _positions_for_pair,
    _select_tracklets,
)
from tools.v86_soft_latent_utils import write_csv, write_json  # noqa: E402
from tools.v91_semantic_regime_utils import normalize_pair_columns  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V91_ROOT, V92_ROOT, seq_text  # noqa: E402


DEFAULT_OUT = ROOT / "phase7_swa_secondary_carrier/route_masks"
DEFAULT_POLICY_ROWS = ROOT / "phase2_object_topology_policy/object_topology_policy_rows.csv"
DEFAULT_V92_ROWS = V92_ROOT / "phase1_semantic_policy_row_bank/semantic_policy_rows.csv"
DEFAULT_TRACKLET_ROWS = V91_ROOT / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv"

V93_TO_ROUTE_STATE = {
    "REJECT_OBJECT_CONFLICT": "REJECT",
    "DELAY_COMMIT": "DELAY",
    "UPDATE_OBJECT_GAUGE": "RESET_RISK",
    "GEOMETRY_RISK": "RESET_RISK",
    "RESET_RISK": "RESET_RISK",
    "DELAY": "DELAY",
    "REJECT": "REJECT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_POLICY_ROWS)
    parser.add_argument("--v92-rows", type=Path, default=DEFAULT_V92_ROWS)
    parser.add_argument("--tracklet-rows", type=Path, default=DEFAULT_TRACKLET_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _stable_score(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _route_state(value: Any) -> str:
    text = str(value or "").strip()
    return V93_TO_ROUTE_STATE.get(text, "HOLD")


def _route_state_series(values: pd.Series) -> pd.Series:
    return values.astype(str).map(_route_state)


def _same_count_random_state(policy: pd.DataFrame, actual_states: pd.Series) -> pd.Series:
    out = pd.Series(["HOLD"] * len(policy), index=policy.index, dtype=object)
    group_cols = ["seq", "quality_type"]
    for _, group in policy.groupby(group_cols, dropna=False):
        group_states = actual_states.loc[group.index]
        risk_count = int(group_states.isin(RISK_STATES).sum())
        if risk_count <= 0:
            continue
        ordered = sorted(group.index.tolist(), key=lambda idx: _stable_score("v93_phase7_random", policy.loc[idx, "pair_id"]))
        for idx in ordered[:risk_count]:
            out.loc[idx] = "RESET_RISK"
    return out


def _load_policy(policy_path: Path, v92_path: Path) -> pd.DataFrame:
    policy = normalize_pair_columns(pd.read_csv(policy_path))
    policy["seq"] = policy["seq"].map(seq_text)
    v92 = normalize_pair_columns(pd.read_csv(v92_path))
    v92["seq"] = v92["seq"].map(seq_text)
    carry_cols = [
        "pair_id",
        "source_path",
        "regime",
        "S_context",
        "P_reject",
        "P_delay",
        "P_reset_risk",
        "tracklet_type_counts",
    ]
    carry_cols = [col for col in carry_cols if col in v92.columns]
    merged = policy.merge(v92[carry_cols], on="pair_id", how="left", suffixes=("", "_v92route"))
    if "source_path" not in merged.columns:
        merged["source_path"] = ""
    if "regime" not in merged.columns and "regime_v92route" in merged.columns:
        merged["regime"] = merged["regime_v92route"]
    return merged


def _variant_specs(policy: pd.DataFrame) -> list[dict[str, Any]]:
    actual_states = _route_state_series(policy["p5_combined_object_policy"])
    return [
        {
            "variant": "v93_object_policy_pair_mask",
            "state_values": actual_states,
            "description": "actual v93 P5 object policy mapped to SWA route states",
        },
        {
            "variant": "v93_object_shuffle_pair_mask",
            "state_values": _route_state_series(policy["p7_object_shuffle_control"]),
            "description": "v93 object shuffle control mapped to SWA route states",
        },
        {
            "variant": "v93_component_shuffle_pair_mask",
            "state_values": _route_state_series(policy["p8_component_shuffle_control"]),
            "description": "v93 component shuffle control mapped to SWA route states",
        },
        {
            "variant": "v93_semantic_shuffle_pair_mask",
            "state_values": _route_state_series(policy["p9_semantic_label_shuffle_control"]),
            "description": "v93 semantic label shuffle control mapped to SWA route states",
        },
        {
            "variant": "v93_regime_shuffle_pair_mask",
            "state_values": _route_state_series(policy["p10_regime_shuffle_control"]),
            "description": "v93 regime shuffle control mapped to SWA route states",
        },
        {
            "variant": "v93_same_count_random_pair_mask",
            "state_values": _same_count_random_state(policy, actual_states),
            "description": "deterministic same-count random rows per seq/quality group",
        },
        {
            "variant": "v93_geometry_only_pair_mask",
            "state_values": _route_state_series(policy["p6_geometry_only_control"]),
            "description": "v93 geometry-only control mapped to SWA route states",
        },
    ]


def _select_row(prow: pd.Series, route_state: str) -> pd.Series:
    row = prow.copy()
    row["policy_state"] = route_state
    row["P_reject"] = 1.0 if route_state == "REJECT" else row.get("P_reject", 0.0)
    row["P_delay"] = 1.0 if route_state == "DELAY" else row.get("P_delay", 0.0)
    row["P_reset_risk"] = 1.0 if route_state == "RESET_RISK" else row.get("P_reset_risk", 0.0)
    return row


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
    position_count = 0
    source_count = 0
    query_count = 0
    selected_points = 0
    selected_pairs = 0
    load_errors: dict[str, int] = {}
    for idx, prow in policy.iterrows():
        route_state = str(state_values.loc[idx])
        if route_state not in RISK_STATES:
            continue
        selected_pairs += 1
        pid = str(prow["pair_id"])
        pair_tracklets = tracklets_by_pair.get(pid, pd.DataFrame())
        selected_tracklets = _select_tracklets(pair_tracklets, _select_row(prow, route_state))
        raw_path = Path(str(prow.get("source_path") or ""))
        positions, pos_summary = _positions_for_pair(raw_path, selected_tracklets)
        if pos_summary.get("load_error"):
            key = str(pos_summary.get("load_error"))
            load_errors[key] = load_errors.get(key, 0) + 1
        position_count += int(len(positions))
        source_count += int(sum(1 for row in positions if row["side"] == "source"))
        query_count += int(sum(1 for row in positions if row["side"] == "query"))
        selected_points += int(pos_summary.get("selected_overlap_points") or 0)
        tracklet_type_counts = (
            selected_tracklets["tracklet_type"].astype(str).value_counts().to_dict()
            if not selected_tracklets.empty and "tracklet_type" in selected_tracklets.columns
            else {}
        )
        pair_row = {
            "variant": variant,
            "seq": seq_text(prow["seq"]),
            "prev_chunk": int(prow["prev_chunk"]),
            "curr_chunk": int(prow["curr_chunk"]),
            "pair_id": pid,
            "original_p5_policy": str(prow.get("p5_combined_object_policy", "")),
            "route_policy_state": route_state,
            "route_state_mapping": "v93 object/gauge states mapped to v91 RESET_RISK/DELAY/REJECT selector states",
            "regime": str(prow.get("regime", "")),
            "quality_type": str(prow.get("quality_type", "")),
            "description": description,
            "selected_tracklet_type_counts": tracklet_type_counts,
            "source_path": str(raw_path),
            **pos_summary,
        }
        pair_rows.append(pair_row)
        for pos in positions:
            position_rows.append(
                {
                    "variant": variant,
                    "seq": seq_text(prow["seq"]),
                    "prev_chunk": int(prow["prev_chunk"]),
                    "curr_chunk": int(prow["curr_chunk"]),
                    "pair_id": pid,
                    "original_p5_policy": str(prow.get("p5_combined_object_policy", "")),
                    "route_policy_state": route_state,
                    "policy_state": route_state,
                    "regime": str(prow.get("regime", "")),
                    "quality_type": str(prow.get("quality_type", "")),
                    "source_path": str(raw_path),
                    **pos,
                }
            )
    variant_row = {
        "variant": variant,
        "description": description,
        "policy_pair_rows": int(selected_pairs),
        "position_rows": int(position_count),
        "query_unique_patch_positions_sum": int(query_count),
        "source_unique_patch_positions_sum": int(source_count),
        "selected_overlap_points_sum": int(selected_points),
        "source_patch_density_vs_3frame_overlap": float(
            source_count / max(1, 3 * PATCH_TOKEN_COUNT * max(1, selected_pairs))
        ),
        "load_errors": load_errors,
    }
    return position_rows, pair_rows, variant_row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = _load_policy(args.policy_rows, args.v92_rows)
    tracklets = normalize_pair_columns(pd.read_csv(args.tracklet_rows))
    tracklets_by_pair = {str(pid): group.copy() for pid, group in tracklets.groupby("pair_id")}

    all_positions: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    for spec in _variant_specs(policy):
        positions, pairs, variant_row = _materialize_variant(
            variant=str(spec["variant"]),
            policy=policy,
            tracklets_by_pair=tracklets_by_pair,
            state_values=spec["state_values"],
            description=str(spec["description"]),
        )
        all_positions.extend(positions)
        all_pairs.extend(pairs)
        variants.append(variant_row)

    write_csv(args.out_dir / "v93_swa_secondary_route_mask_positions.csv", all_positions)
    write_csv(args.out_dir / "v93_swa_secondary_route_mask_pair_summary.csv", all_pairs)
    write_csv(args.out_dir / "v93_swa_secondary_route_mask_variant_summary.csv", variants)
    actual_row = next((row for row in variants if row["variant"] == "v93_object_policy_pair_mask"), {})
    summary = {
        "phase": "Phase7_v93_swa_secondary_route_mask_materialization",
        "entered": True,
        "entry_reason": "Phase5 counterfactual upper bound failed with no measured bad residual movement; plan directs SWA secondary/action-surface rediscovery.",
        "policy_rows": int(len(policy)),
        "variants": variants,
        "position_rows": int(len(all_positions)),
        "materialization_feasible": bool(int(actual_row.get("position_rows") or 0) > 0),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "selection_rule": "Rows are selected from v93 object policy/control states, mapped to RESET_RISK/DELAY/REJECT, then tracklet label pairs are materialized into query/source patch positions.",
        "no_success_claim": "These CSVs are route-dump inputs only; Phase7 success requires true route/control audit.",
    }
    write_json(args.out_dir / "materialization_summary.json", summary)
    print(f"materialization_feasible={summary['materialization_feasible']}")
    print(f"policy_rows={summary['policy_rows']}")
    print(f"position_rows={summary['position_rows']}")
    for row in variants:
        print(
            "variant={variant} policy_pair_rows={policy_pair_rows} "
            "query_positions={query_unique_patch_positions_sum} source_positions={source_unique_patch_positions_sum}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
