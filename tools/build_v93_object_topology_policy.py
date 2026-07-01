#!/usr/bin/env python3
"""Build fixed v93 Phase2 object-topology policy family rows."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.v86_soft_latent_utils import safe_float, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V92_ROOT, seq_text  # noqa: E402


RISK_STATES = {"RESET_RISK", "DELAY", "REJECT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=ROOT / "phase1_object_identity_row_join")
    parser.add_argument("--v92-root", type=Path, default=V92_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase2_object_topology_policy")
    return parser.parse_args()


def num(value: Any, default: float = 0.0) -> float:
    out = safe_float(value)
    return default if out is None else float(out)


def flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def stable_key(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_shuffle(df: pd.DataFrame, col: str, salt: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    group_cols = ["seq", "quality_type"]
    for _, group in df.groupby(group_cols, dropna=False):
        idx = list(group.index)
        if len(idx) <= 1:
            out.loc[idx] = df.loc[idx, col].to_list()
            continue
        ordered = sorted(idx, key=lambda i: stable_key(salt, i, df.loc[i, "pair_id"]))
        values = [df.loc[i, col] for i in ordered]
        values = values[-1:] + values[:-1]
        for i, value in zip(ordered, values):
            out.loc[i] = value
    return out


def identity_confident(row: pd.Series) -> bool:
    return flag(row.get("has_object_identity")) and num(row.get("object_identity_confidence")) >= 0.35


def match_confident(row: pd.Series) -> bool:
    return flag(row.get("match_backed_component_relation")) and num(row.get("match_support_confidence")) >= 0.50


def exact_cross_object(row: pd.Series) -> bool:
    cross = max(num(row.get("cross_object_ratio")), num(row.get("boundary_global_cross_ratio")))
    new_id = num(row.get("boundary_new_id_ratio"))
    return cross >= 0.34 or new_id >= 0.34


def stable_same_boundary(row: pd.Series) -> bool:
    cross = max(num(row.get("cross_object_ratio")), num(row.get("boundary_global_cross_ratio")))
    return (
        identity_confident(row)
        and num(row.get("object_identity_confidence")) >= 0.75
        and num(row.get("same_object_ratio")) >= 0.80
        and cross <= 0.20
        and num(row.get("S_invalid")) < 0.10
        and num(row.get("S_lowobs")) < 0.35
    )


def p1_object_interior(row: pd.Series) -> str:
    has_object = identity_confident(row)
    has_radio = flag(row.get("radio_available"))
    radio_interior_dominant = (
        has_radio
        and num(row.get("radio_object_interior")) > num(row.get("radio_boundary"))
        and num(row.get("radio_temporal_stability")) >= 0.50
    )
    object_like = has_object or radio_interior_dominant
    cross_low = max(num(row.get("cross_object_ratio")), num(row.get("boundary_global_cross_ratio"))) <= 0.10
    not_boundary_dominant = num(row.get("object_interior_ratio")) > num(row.get("object_boundary_ratio")) or radio_interior_dominant
    if (
        object_like
        and cross_low
        and not_boundary_dominant
        and num(row.get("same_object_ratio")) >= 0.80
        and num(row.get("object_interior_ratio")) >= 0.05
    ):
        return "UPDATE_OBJECT_GAUGE"
    return "ABSTAIN_OBJECT_SOURCE_MISSING"


def p2_cross_object_reject(row: pd.Series) -> str:
    cross = num(row.get("cross_object_ratio"))
    boundary = num(row.get("object_boundary_ratio"))
    if identity_confident(row) and exact_cross_object(row):
        return "REJECT_OBJECT_CONFLICT"
    if match_confident(row) and cross >= 0.95 and boundary >= 0.90:
        return "REJECT_OBJECT_CONFLICT"
    return "ABSTAIN_NO_CONFLICT"


def p3_lowobs_hold(row: pd.Series) -> str:
    if num(row.get("temporal_stability")) < 0.20 or "no_joined" in str(row.get("source_scope")):
        return "HOLD_GAUGE"
    return "ABSTAIN_NOT_LOWOBS"


def p4_multimode_delay(row: pd.Series) -> str:
    multimode = num(row.get("S_multimode")) >= 0.80 or str(row.get("policy_state_v92")) == "DELAY"
    guarded_multimode = multimode and not stable_same_boundary(row) and (
        num(row.get("object_identity_confidence")) < 0.75
        or exact_cross_object(row)
        or num(row.get("S_invalid")) >= 0.10
        or num(row.get("S_lowobs")) >= 0.35
    )
    reset_invalid = (
        str(row.get("policy_state_v92")) == "RESET_RISK"
        and num(row.get("S_invalid")) >= 0.50
        and (identity_confident(row) or match_confident(row))
    )
    if guarded_multimode or reset_invalid:
        return "DELAY_COMMIT"
    return "ABSTAIN_NOT_MULTIMODE"


def p5_combined(row: pd.Series) -> str:
    # Reject invalid cross-object evidence first; delay multimode unsafe rows;
    # update only when explicit object/RADIO source supports it; otherwise hold.
    p2 = p2_cross_object_reject(row)
    if p2 == "REJECT_OBJECT_CONFLICT":
        return p2
    p4 = p4_multimode_delay(row)
    if p4 == "DELAY_COMMIT":
        return p4
    p1 = p1_object_interior(row)
    if p1 == "UPDATE_OBJECT_GAUGE":
        return p1
    p3 = p3_lowobs_hold(row)
    if p3 == "HOLD_GAUGE":
        return p3
    return "HOLD_GAUGE"


def p6_geometry(row: pd.Series) -> str:
    if num(row.get("S_boundary")) >= 0.90 or num(row.get("S_multimode")) >= 0.80 or num(row.get("S_invalid")) >= 0.30:
        return "GEOMETRY_RISK"
    return "GEOMETRY_HOLD"


def repair_require_match_or_radio(row: pd.Series) -> str:
    if p4_multimode_delay(row) == "DELAY_COMMIT":
        return "DELAY_COMMIT"
    strong = identity_confident(row) or match_confident(row) or (
        flag(row.get("radio_available")) and num(row.get("radio_temporal_stability_mean")) >= 0.50
    )
    if identity_confident(row) and exact_cross_object(row):
        return "REJECT_OBJECT_CONFLICT"
    if match_confident(row) and num(row.get("cross_object_ratio")) >= 0.95 and num(row.get("object_boundary_ratio")) >= 0.90:
        return "REJECT_OBJECT_CONFLICT"
    if strong and p1_object_interior(row) == "UPDATE_OBJECT_GAUGE":
        return "UPDATE_OBJECT_GAUGE"
    return "HOLD_GAUGE"


def repair_multimode_delay(row: pd.Series) -> str:
    if num(row.get("S_multimode")) >= 0.75:
        return "DELAY_COMMIT"
    return p5_combined(row)


def repair_radio_guarded(row: pd.Series) -> str:
    if not (flag(row.get("radio_available")) or identity_confident(row)):
        return "HOLD_GAUGE"
    if p1_object_interior(row) == "UPDATE_OBJECT_GAUGE":
        return "UPDATE_OBJECT_GAUGE"
    if identity_confident(row) and exact_cross_object(row):
        return "REJECT_OBJECT_CONFLICT"
    if match_confident(row) and num(row.get("radio_boundary")) >= num(row.get("radio_object_interior")):
        return "REJECT_OBJECT_CONFLICT"
    return "HOLD_GAUGE"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    join = pd.read_csv(args.phase1_dir / "object_identity_row_join.csv")
    v92 = pd.read_csv(args.v92_root / "phase1_semantic_policy_row_bank/semantic_policy_rows.csv")
    join["seq"] = join["seq"].map(seq_text)
    v92["seq"] = v92["seq"].map(seq_text)
    extra_cols = [
        "pair_id",
        "S_multimode",
        "S_boundary",
        "S_invalid",
        "S_lowobs",
        "semantic_shuffle_state",
        "component_shuffle_state",
        "regime_shuffle_state",
    ]
    df = join.merge(v92[extra_cols], on="pair_id", how="left")
    df["p0_v92_policy_baseline"] = df["policy_state_v92"].astype(str)
    df["p1_object_interior_update"] = df.apply(p1_object_interior, axis=1)
    df["p2_cross_object_reject"] = df.apply(p2_cross_object_reject, axis=1)
    df["p3_lowobs_hold"] = df.apply(p3_lowobs_hold, axis=1)
    df["p4_multimode_delay"] = df.apply(p4_multimode_delay, axis=1)
    df["p5_combined_object_policy"] = df.apply(p5_combined, axis=1)
    df["p6_geometry_only_control"] = df.apply(p6_geometry, axis=1)

    # Controls preserve row labels/geometry while disrupting the semantic/object dimensions.
    for col in ["same_object_ratio", "cross_object_ratio", "object_boundary_ratio", "object_interior_ratio", "temporal_stability"]:
        df[f"object_shuffle_{col}"] = stable_shuffle(df, col, "object")
        df[f"component_shuffle_{col}"] = stable_shuffle(df, col, "component")

    object_df = df.copy()
    for col in ["same_object_ratio", "cross_object_ratio", "object_boundary_ratio", "object_interior_ratio", "temporal_stability"]:
        object_df[col] = object_df[f"object_shuffle_{col}"]
    component_df = df.copy()
    for col in ["same_object_ratio", "cross_object_ratio", "object_boundary_ratio", "object_interior_ratio", "temporal_stability"]:
        component_df[col] = component_df[f"component_shuffle_{col}"]
    df["p7_object_shuffle_control"] = object_df.apply(p5_combined, axis=1)
    df["p8_component_shuffle_control"] = component_df.apply(p5_combined, axis=1)
    df["p9_semantic_label_shuffle_control"] = df["semantic_shuffle_state"].astype(str)
    df["p10_regime_shuffle_control"] = df["regime_shuffle_state"].astype(str)
    df["p5_repair_require_match_or_radio"] = df.apply(repair_require_match_or_radio, axis=1)
    df["p5_repair_multimode_delay"] = df.apply(repair_multimode_delay, axis=1)
    df["p5_repair_radio_guarded"] = df.apply(repair_radio_guarded, axis=1)

    selected_cols = [
        "pair_id",
        "seq",
        "prev_chunk",
        "curr_chunk",
        "base_case_type",
        "quality_type",
        "labelled",
        "abs_log_scale_jump_gt",
        "policy_state_v92",
        "S_multimode",
        "S_boundary",
        "S_invalid",
        "S_lowobs",
        "has_object_identity",
        "has_global_object_id",
        "object_identity_source",
        "radio_available",
        "source_scope",
        "object_identity_confidence",
        "object_id_source",
        "object_id_confidence",
        "prev_global_object_id",
        "curr_global_object_id",
        "same_global_object_id",
        "same_object_ratio",
        "cross_object_ratio",
        "object_boundary_ratio",
        "object_interior_ratio",
        "temporal_stability",
        "boundary_audit_rows",
        "boundary_valid_global_rows",
        "boundary_same_global_rows",
        "boundary_cross_global_rows",
        "boundary_kept_global_id_rows",
        "boundary_new_id_rows",
        "boundary_global_same_ratio",
        "boundary_global_cross_ratio",
        "boundary_new_id_ratio",
        "boundary_mean_mask_iou",
        "boundary_mean_box_iou",
        "boundary_mean_containment",
        "boundary_mean_area_similarity",
        "boundary_mean_center_dist_norm",
        "boundary_decisions",
        "boundary_audit_path",
        "feature_match_support_count",
        "match_support_confidence",
        "radio_interior_mean",
        "radio_boundary_mean",
        "radio_temporal_stability_mean",
        "p0_v92_policy_baseline",
        "p1_object_interior_update",
        "p2_cross_object_reject",
        "p3_lowobs_hold",
        "p4_multimode_delay",
        "p5_combined_object_policy",
        "p6_geometry_only_control",
        "p7_object_shuffle_control",
        "p8_component_shuffle_control",
        "p9_semantic_label_shuffle_control",
        "p10_regime_shuffle_control",
        "p5_repair_require_match_or_radio",
        "p5_repair_multimode_delay",
        "p5_repair_radio_guarded",
    ]
    rows = df[selected_cols].to_dict("records")
    write_csv(args.out_dir / "object_topology_policy_rows.csv", rows)
    write_json(
        args.out_dir / "object_topology_policy_build_summary.json",
        {
            "phase": "Phase2_object_topology_policy_build",
            "row_count": int(len(df)),
            "labelled_row_count": int(df["base_case_type"].astype(str).isin(["bad", "good"]).sum()),
            "policy_families": [
                "P0_v92_policy_baseline",
                "P1_object_interior_update",
                "P2_cross_object_reject",
                "P3_lowobs_hold",
                "P4_multimode_delay",
                "P5_combined_object_policy",
                "P6_geometry_only_control",
                "P7_object_shuffle_control",
                "P8_component_shuffle_control",
                "P9_semantic_label_shuffle_control",
                "P10_regime_shuffle_control",
                "P5_repair_require_match_or_radio",
                "P5_repair_multimode_delay",
                "P5_repair_radio_guarded",
            ],
            "thresholds": {
                "object_identity_confidence_min": 0.35,
                "stable_same_boundary_confidence_min": 0.75,
                "stable_same_boundary_cross_ratio_max": 0.20,
                "boundary_global_cross_or_new_id_ratio_min": 0.34,
                "P1_same_object_ratio_min": 0.80,
                "P1_object_interior_ratio_min": 0.05,
                "P1_cross_object_ratio_max": 0.10,
                "P1_requires_not_boundary_dominant": True,
                "P1_radio_temporal_stability_min": 0.50,
                "P2_cross_object_ratio_min": 0.95,
                "P2_object_boundary_ratio_min": 0.90,
                "P2_match_support_confidence_min": 0.50,
                "P3_temporal_stability_min": 0.20,
                "P4_S_multimode_min": 0.80,
                "P4_reset_invalid_min": 0.50,
                "P6_S_boundary_min": 0.90,
                "P6_S_multimode_min": 0.80,
                "P6_S_invalid_min": 0.30,
            },
            "scope_note": "P2/P5 use exact SAM31 boundary global-id rows when available; component/RADIO-only rows remain diagnostic and cannot claim object identity.",
            "runtime_action_allowed": False,
            "counterfactual_allowed": False,
            "ttt_allowed": False,
        },
    )
    print(f"row_count={len(df)}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
