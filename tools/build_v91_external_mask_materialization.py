#!/usr/bin/env python3
"""Materialize v91 policy-risk topology tracklets into an external SWA mask CSV.

This is a Phase7 repair artifact.  It converts rows selected by the no-GT
Phase5 semantic memory policy into runtime-consumable source/query patch
positions for the existing external-mask SWA route-dump hook.  The CSV is not a
carrier success claim; it is only an auditable input for a true route dump.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, normalize_pair_columns


DEFAULT_OUT = ROOT / "phase7_carrier_attribution_or_blocked/v91_external_mask_materialization"
PATCH_GRID = (19, 66)
PATCH_TOKEN_COUNT = PATCH_GRID[0] * PATCH_GRID[1]
MODEL_PATCH_START_IDX_NOTE = "Runtime PI3 hook adds the live patch_start_idx offset; CSV stores per-frame patch token indices only."
RISK_STATES = {"RESET_RISK", "DELAY", "REJECT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--policy-rows",
        type=Path,
        default=ROOT / "phase5_memory_update_policy/policy_state_rows.csv",
    )
    parser.add_argument(
        "--tracklet-rows",
        type=Path,
        default=ROOT / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv",
    )
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def patch_indices(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = coords[:, 0].float()
    x = coords[:, 1].float()
    py = torch.clamp((y / (376.0 / PATCH_GRID[0])).floor().long(), 0, PATCH_GRID[0] - 1)
    px = torch.clamp((x / (1408.0 / PATCH_GRID[1])).floor().long(), 0, PATCH_GRID[1] - 1)
    flat = py * PATCH_GRID[1] + px
    return py, px, flat


def _variant_policy_rows(policy: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "v91_policy_risk_source_mask":
        return policy[policy["policy_state"].astype(str).isin(RISK_STATES)].copy()
    if variant == "v91_policy_reset_risk_source_mask":
        return policy[policy["policy_state"].astype(str).eq("RESET_RISK")].copy()
    if variant == "v91_policy_delay_source_mask":
        return policy[policy["policy_state"].astype(str).eq("DELAY")].copy()
    if variant == "v91_policy_reject_source_mask":
        return policy[policy["policy_state"].astype(str).eq("REJECT")].copy()
    raise ValueError(f"unknown variant: {variant}")


def _select_tracklets(pair_tracklets: pd.DataFrame, policy_row: pd.Series) -> pd.DataFrame:
    if pair_tracklets.empty:
        return pair_tracklets
    state = str(policy_row.get("policy_state", ""))
    types = pair_tracklets["tracklet_type"].astype(str)
    if state == "REJECT":
        selected = pair_tracklets[types.isin(["INVALID_CROSS_BOUNDARY", "DYNAMIC_TRANSIENT", "SPLIT_MERGE_UNSTABLE"])]
        if selected.empty:
            selected = pair_tracklets[(pair_tracklets["same_label"].astype(str).str.lower() != "true") | (pd.to_numeric(pair_tracklets["cross_component_boundary_ratio"], errors="coerce").fillna(0.0) >= 0.55)]
        return selected if not selected.empty else pair_tracklets
    if state == "DELAY":
        selected = pair_tracklets[types.eq("MULTIMODE_UNSAFE")]
        if selected.empty:
            entropy = pd.to_numeric(pair_tracklets["mode_entropy"], errors="coerce").fillna(0.0)
            selected = pair_tracklets[entropy >= float(entropy.quantile(0.75))]
        return selected if not selected.empty else pair_tracklets
    if state == "RESET_RISK":
        selected_parts = []
        invalid = pair_tracklets[types.isin(["INVALID_CROSS_BOUNDARY", "DYNAMIC_TRANSIENT", "SPLIT_MERGE_UNSTABLE"])]
        context = pair_tracklets[types.eq("CONTEXT_LOWOBS")]
        if float(policy_row.get("S_invalid", 0.0) or 0.0) > 0.0 or float(policy_row.get("P_reject", 0.0) or 0.0) > 0.0:
            selected_parts.append(invalid)
        if (
            float(policy_row.get("S_context", 0.0) or 0.0) >= 0.25
            or str(policy_row.get("regime", "")).upper() == "REGIME_LOWOBS_CONTEXT"
        ):
            selected_parts.append(context)
        if selected_parts:
            selected = pd.concat(selected_parts, ignore_index=False).drop_duplicates(subset=["tracklet_id"])
        else:
            selected = pd.DataFrame(columns=pair_tracklets.columns)
        if selected.empty:
            selected = pair_tracklets[~types.str.startswith("VALID")]
        return selected if not selected.empty else pair_tracklets
    return pair_tracklets.iloc[0:0]


def _positions_for_pair(raw_path: Path, selected_tracklets: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    if selected_tracklets.empty:
        return out_rows, {"selected_tracklet_rows": 0, "selected_overlap_points": 0, "query_unique_patch_positions": 0, "source_unique_patch_positions": 0}
    try:
        raw = torch_load(raw_path)
    except Exception as exc:  # noqa: BLE001
        return out_rows, {
            "selected_tracklet_rows": int(len(selected_tracklets)),
            "selected_overlap_points": 0,
            "query_unique_patch_positions": 0,
            "source_unique_patch_positions": 0,
            "load_error": type(exc).__name__,
        }

    required = [
        "prev_pixel_coords",
        "curr_pixel_coords",
        "prev_frame_ids",
        "curr_frame_ids",
        "prev_semantic_labels",
        "curr_semantic_labels",
    ]
    if not all(torch.is_tensor(raw.get(key)) for key in required):
        return out_rows, {
            "selected_tracklet_rows": int(len(selected_tracklets)),
            "selected_overlap_points": 0,
            "query_unique_patch_positions": 0,
            "source_unique_patch_positions": 0,
            "load_error": "missing_required_tensor",
        }

    prev_labels = raw["prev_semantic_labels"].long()
    curr_labels = raw["curr_semantic_labels"].long()
    selected_mask = torch.zeros_like(prev_labels, dtype=torch.bool)
    label_pairs: set[tuple[int, int]] = set()
    for _, row in selected_tracklets.iterrows():
        try:
            label_pairs.add((int(row["prev_label"]), int(row["curr_label"])))
        except Exception:  # noqa: BLE001
            continue
    for prev_label, curr_label in sorted(label_pairs):
        selected_mask |= (prev_labels == int(prev_label)) & (curr_labels == int(curr_label))

    idx = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if int(idx.numel()) == 0:
        return out_rows, {
            "selected_tracklet_rows": int(len(selected_tracklets)),
            "selected_label_pairs": int(len(label_pairs)),
            "selected_overlap_points": 0,
            "query_unique_patch_positions": 0,
            "source_unique_patch_positions": 0,
        }

    curr_py, curr_px, curr_flat = patch_indices(raw["curr_pixel_coords"][idx])
    prev_py, prev_px, prev_flat = patch_indices(raw["prev_pixel_coords"][idx])
    curr_start = int(raw.get("curr_start_frame", 0) or 0)
    prev_start = int(raw.get("prev_start_frame", 0) or 0)
    q_local = (raw["curr_frame_ids"][idx].long() - curr_start).clamp_min(0)
    s_local = (raw["prev_frame_ids"][idx].long() - prev_start).clamp_min(0)
    query_positions = sorted({(int(f), int(p), int(y), int(x)) for f, p, y, x in zip(q_local.tolist(), curr_flat.tolist(), curr_py.tolist(), curr_px.tolist())})
    source_positions = sorted({(int(f), int(p), int(y), int(x)) for f, p, y, x in zip(s_local.tolist(), prev_flat.tolist(), prev_py.tolist(), prev_px.tolist())})
    for side, positions in [("query", query_positions), ("source", source_positions)]:
        for local_frame, patch_flat, py, px in positions:
            out_rows.append(
                {
                    "side": side,
                    "local_frame": int(local_frame),
                    "patch_token_index": int(patch_flat),
                    "patch_y": int(py),
                    "patch_x": int(px),
                    "model_token_index_requires_patch_start_offset": True,
                    "model_patch_start_idx_note": MODEL_PATCH_START_IDX_NOTE,
                }
            )
    return out_rows, {
        "selected_tracklet_rows": int(len(selected_tracklets)),
        "selected_label_pairs": int(len(label_pairs)),
        "selected_overlap_points": int(idx.numel()),
        "query_unique_patch_positions": int(len(query_positions)),
        "source_unique_patch_positions": int(len(source_positions)),
        "source_local_frames": sorted({int(f) for f, *_ in source_positions}),
        "query_local_frames": sorted({int(f) for f, *_ in query_positions}),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = normalize_pair_columns(pd.read_csv(args.policy_rows))
    tracklets = normalize_pair_columns(pd.read_csv(args.tracklet_rows))
    tracklets_by_pair = {pid: group.copy() for pid, group in tracklets.groupby("pair_id")}
    variants = [
        "v91_policy_risk_source_mask",
        "v91_policy_reset_risk_source_mask",
        "v91_policy_delay_source_mask",
        "v91_policy_reject_source_mask",
    ]

    all_positions: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        v_policy = _variant_policy_rows(policy, variant)
        v_position_count = 0
        v_source_count = 0
        v_query_count = 0
        v_points = 0
        for _, prow in v_policy.iterrows():
            pid = str(prow["pair_id"])
            pair_tracklets = tracklets_by_pair.get(pid, pd.DataFrame(columns=tracklets.columns))
            selected_tracklets = _select_tracklets(pair_tracklets, prow)
            raw_path = Path(str(prow.get("source_path") or ""))
            positions, pos_summary = _positions_for_pair(raw_path, selected_tracklets)
            v_position_count += int(len(positions))
            v_source_count += int(sum(1 for row in positions if row["side"] == "source"))
            v_query_count += int(sum(1 for row in positions if row["side"] == "query"))
            v_points += int(pos_summary.get("selected_overlap_points") or 0)
            tracklet_type_counts = selected_tracklets["tracklet_type"].astype(str).value_counts().to_dict() if not selected_tracklets.empty else {}
            pair_rows.append(
                {
                    "variant": variant,
                    "seq": str(prow["seq"]).zfill(2),
                    "prev_chunk": int(prow["prev_chunk"]),
                    "curr_chunk": int(prow["curr_chunk"]),
                    "pair_id": pid,
                    "policy_state": str(prow.get("policy_state", "")),
                    "regime": str(prow.get("regime", "")),
                    "selected_tracklet_type_counts": tracklet_type_counts,
                    "source_path": str(raw_path),
                    **pos_summary,
                }
            )
            for pos in positions:
                all_positions.append(
                    {
                        "variant": variant,
                        "seq": str(prow["seq"]).zfill(2),
                        "prev_chunk": int(prow["prev_chunk"]),
                        "curr_chunk": int(prow["curr_chunk"]),
                        "pair_id": pid,
                        "policy_state": str(prow.get("policy_state", "")),
                        "regime": str(prow.get("regime", "")),
                        "source_path": str(raw_path),
                        **pos,
                    }
                )
        variant_rows.append(
            {
                "variant": variant,
                "policy_pair_rows": int(len(v_policy)),
                "position_rows": int(v_position_count),
                "query_unique_patch_positions_sum": int(v_query_count),
                "source_unique_patch_positions_sum": int(v_source_count),
                "selected_overlap_points_sum": int(v_points),
                "source_patch_density_vs_3frame_overlap": float(v_source_count / max(1, 3 * PATCH_TOKEN_COUNT * max(1, int(len(v_policy))))),
            }
        )

    write_csv(args.out_dir / "anchor_route_mask_positions.csv", all_positions)
    write_csv(args.out_dir / "v91_external_mask_pair_summary.csv", pair_rows)
    write_csv(args.out_dir / "v91_external_mask_variant_summary.csv", variant_rows)
    materialization_feasible = bool(
        any(row["variant"] == "v91_policy_risk_source_mask" and int(row["source_unique_patch_positions_sum"]) > 0 for row in variant_rows)
    )
    summary = {
        "phase": "Phase7_v91_external_mask_materialization",
        "materialization_feasible": materialization_feasible,
        "policy_rows": int(len(policy)),
        "risk_policy_rows": int(policy["policy_state"].astype(str).isin(RISK_STATES).sum()),
        "position_rows": int(len(all_positions)),
        "variants": variant_rows,
        "runtime_external_mask_hook_available": True,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "selection_rule": "policy_state in RESET_RISK/DELAY/REJECT from Phase5 no-GT policy; selected tracklet label-pairs are mapped to raw-overlap patch positions.",
        "no_success_claim": "This CSV is a route-dump input only; carrier success still requires true route/control evidence.",
    }
    write_json(args.out_dir / "materialization_summary.json", summary)
    print(f"materialization_feasible={summary['materialization_feasible']}")
    print(f"risk_policy_rows={summary['risk_policy_rows']}")
    print(f"position_rows={summary['position_rows']}")
    for row in variant_rows:
        print(
            "variant={variant} policy_pair_rows={policy_pair_rows} source_positions={source_unique_patch_positions_sum} query_positions={query_unique_patch_positions_sum}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
