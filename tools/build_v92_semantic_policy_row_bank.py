#!/usr/bin/env python3
"""Build v92 Phase1 semantic policy row bank by freezing v91 R9 policy rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, RISK_STATES, V91_PHASE5, V91_PHASE7, V91_ROOT, pair_id, seq_text


DEFAULT_OUT = ROOT / "phase1_semantic_policy_row_bank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v91-root", type=Path, default=V91_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _counts_by_pair(tracklet_path: Path) -> dict[str, dict[str, int]]:
    if not tracklet_path.exists():
        return {}
    df = pd.read_csv(tracklet_path)
    out: dict[str, dict[str, int]] = {}
    for pair, group in df.groupby("pair_id"):
        counts = group["tracklet_type"].astype(str).value_counts().to_dict()
        out[str(pair)] = {str(k): int(v) for k, v in counts.items()}
    return out


def _materialization_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row.get("pair_id", ""))] = {
            "materialized_variant": row.get("variant", ""),
            "materialized_selected_tracklet_rows": row.get("selected_tracklet_rows", ""),
            "materialized_selected_label_pairs": row.get("selected_label_pairs", ""),
            "materialized_selected_overlap_points": row.get("selected_overlap_points", ""),
            "materialized_query_patch_positions": row.get("query_unique_patch_positions", ""),
            "materialized_source_patch_positions": row.get("source_unique_patch_positions", ""),
            "materialized_query_local_frames": row.get("query_local_frames", ""),
            "materialized_source_local_frames": row.get("source_local_frames", ""),
        }
    return out


def _route_pairs(root: Path) -> set[str]:
    out: set[str] = set()
    if not root.exists():
        return out
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("seq") or "_chunk" not in name:
            continue
        seq_part, chunk_part = name.split("_chunk", 1)
        seq = seq_part.replace("seq", "")
        chunk_digits = ""
        for char in chunk_part:
            if char.isdigit():
                chunk_digits += char
            else:
                break
        if not chunk_digits:
            continue
        curr = int(chunk_digits)
        out.add(pair_id(seq, curr - 1, curr))
    return out


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    phase5_dir = args.v91_root / "phase5_memory_update_policy"
    phase7_dir = args.v91_root / "phase7_carrier_attribution_or_blocked"
    policy_path = phase5_dir / "policy_state_rows.csv"
    df = pd.read_csv(policy_path)
    df["seq"] = df["seq"].map(seq_text)
    df["prev_chunk"] = pd.to_numeric(df["prev_chunk"], errors="coerce").fillna(0).astype(int)
    df["curr_chunk"] = pd.to_numeric(df["curr_chunk"], errors="coerce").fillna(0).astype(int)
    df["pair_id"] = [pair_id(s, p, c) for s, p, c in zip(df["seq"], df["prev_chunk"], df["curr_chunk"])]
    h_mode = pd.to_numeric(df.get("H_mode"), errors="coerce").fillna(0.0)
    h_mode_max = max(float(h_mode.max()), 1e-12)
    tracklet_counts = _counts_by_pair(args.v91_root / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_rows.csv")
    mat_by_pair = _materialization_by_pair(phase7_dir / "v91_external_mask_materialization/v91_external_mask_pair_summary.csv")
    route_pairs = _route_pairs(phase7_dir / "route_dump_smoke")
    direct_proxy_path = phase7_dir / "direct_boundary_update_trace_proxy.csv"
    direct_proxy_pairs = set()
    if direct_proxy_path.exists():
        proxy_df = pd.read_csv(direct_proxy_path)
        direct_proxy_pairs = set(proxy_df["pair_id"].astype(str).tolist())
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pid = str(row.get("pair_id", ""))
        mat = mat_by_pair.get(pid, {})
        policy_state = str(row.get("policy_state", ""))
        item = row.to_dict()
        item.update(
            {
                "quality_type": row.get("base_case_type", ""),
                "policy_risk_positive": policy_state in RISK_STATES,
                "S_multimode": float(h_mode.loc[row.name] / h_mode_max),
                "S_boundary": row.get("boundary_mass", ""),
                "tracklet_type_counts": json.dumps(tracklet_counts.get(pid, {}), ensure_ascii=False, sort_keys=True),
                "component_consistency": max(0.0, 1.0 - float(pd.to_numeric(pd.Series([row.get("invalid_tracklet_ratio")]), errors="coerce").fillna(0.0).iloc[0])),
                "materialized_available": bool(pid in mat_by_pair),
                "materialized_query_patch_positions": mat.get("materialized_query_patch_positions", ""),
                "materialized_source_patch_positions": mat.get("materialized_source_patch_positions", ""),
                "materialized_selected_tracklet_rows": mat.get("materialized_selected_tracklet_rows", ""),
                "materialized_selected_overlap_points": mat.get("materialized_selected_overlap_points", ""),
                "route_smoke_available": bool(pid in route_pairs),
                "boundary_trace_available": bool(pid in direct_proxy_pairs),
                "boundary_trace_scope": "policy_proxy_not_runtime_trace" if pid in direct_proxy_pairs else "unavailable",
                "v91_policy_source_path": str(policy_path),
            }
        )
        rows.append(item)
    fields = list(rows[0].keys()) if rows else []
    write_csv(out / "semantic_policy_rows.csv", rows, fields)
    by_pair_fields = [
        "seq",
        "prev_chunk",
        "curr_chunk",
        "pair_id",
        "policy_state",
        "policy_risk_positive",
        "regime",
        "base_case_type",
        "quality_type",
        "S_valid",
        "S_invalid",
        "S_context",
        "S_multimode",
        "S_boundary",
        "semantic_confidence_mean",
        "component_consistency",
        "tracklet_type_counts",
        "materialized_available",
        "materialized_query_patch_positions",
        "materialized_source_patch_positions",
        "route_smoke_available",
        "boundary_trace_available",
        "boundary_trace_scope",
        "bad_good_label_used_for_assignment",
        "scale_label_used_for_assignment",
    ]
    write_csv(out / "semantic_policy_by_pair.csv", rows, by_pair_fields)
    controls_src = phase5_dir / "policy_state_audit_controls.csv"
    if controls_src.exists():
        controls = pd.read_csv(controls_src).to_dict("records")
        write_csv(out / "shuffle_controls.csv", controls)
    summary = {
        "phase": "Phase1_semantic_policy_row_bank_build",
        "row_count": int(len(rows)),
        "sequence_coverage": int(df["seq"].nunique()),
        "state_counts": {str(k): int(v) for k, v in df["policy_state"].astype(str).value_counts().to_dict().items()},
        "route_smoke_pair_count": int(len(route_pairs)),
        "materialized_pair_count": int(len(mat_by_pair)),
        "boundary_proxy_pair_count": int(len(direct_proxy_pairs)),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(out / "semantic_policy_row_bank_summary.json", summary)
    (out / "policy_state_definition.md").write_text(
        "\n".join(
            [
                "# v92 Semantic Policy State Definition",
                "",
                "Rows are frozen from v91 Phase5 R9 `policy_state_rows.csv`.",
                "",
                "- Risk-positive states: `RESET_RISK`, `DELAY`, `REJECT`.",
                "- Hold/context states: `HOLD`, `ABSTAIN`.",
                "- `S_multimode` is computed as `H_mode / max(H_mode)` for row-bank normalization only.",
                "- `S_boundary` is copied from v91 `boundary_mass`.",
                "- `component_consistency` is computed as `1 - invalid_tracklet_ratio` from v91 row fields.",
                "- `boundary_trace_scope=policy_proxy_not_runtime_trace` means v91 direct boundary proxy exists, not a true carrier trace.",
                "- `bad_good_label_used_for_assignment` and `scale_label_used_for_assignment` must stay false for policy-construction audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"row_count={summary['row_count']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"state_counts={summary['state_counts']}")
    print(f"route_smoke_pair_count={summary['route_smoke_pair_count']}")
    print(f"materialized_pair_count={summary['materialized_pair_count']}")
    print(f"boundary_proxy_pair_count={summary['boundary_proxy_pair_count']}")


if __name__ == "__main__":
    main()
