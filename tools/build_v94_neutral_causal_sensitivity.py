#!/usr/bin/env python3
"""Build v94 Phase3 neutral causal sensitivity audit artifacts.

This phase is intentionally conservative: v93 exposes trace-level merge/gauge
upper-bound counterfactuals, but not a rerun trajectory with J_handoff_probe.
The generated summaries therefore separate carrier trace movement from geometry
or runtime sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
PHASE1 = ROOT / "phase1_boundary_failure_atlas"
PHASE2 = ROOT / "phase2_true_carrier_trace"
V93_PHASE5 = (
    Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")
    / "phase5_merge_gauge_counterfactual_upper_bound"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_nonzero(value: Any, eps: float = 1e-12) -> bool:
    number = to_float(value)
    return math.isfinite(number) and abs(number) > eps


def add_probe_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    primary = out["failure_type_primary"].fillna("")
    secondary = out.get("failure_type_secondary", pd.Series([""] * len(out))).fillna("")
    labels = out["case_label_offline_only"].fillna("")
    evidence = out.get("semantic_evidence_type_majority", pd.Series([""] * len(out))).fillna("")
    sem_invalid = pd.to_numeric(out.get("semantic_invalid_mass", 0), errors="coerce").fillna(0.0)

    out["probe_handoff_scale"] = primary.eq("HANDOFF_SCALE")
    out["probe_handoff_gauge"] = primary.eq("HANDOFF_GAUGE")
    out["probe_good_safe"] = labels.eq("good") | primary.eq("SAFE_OR_UNASSIGNED")
    out["probe_low_observability"] = primary.eq("LOW_OBSERVABILITY") | secondary.str.contains(
        "LOW_OBSERVABILITY", na=False
    )
    out["probe_semantic_invalid"] = secondary.str.contains("SEMANTIC_INVALID", na=False) | sem_invalid.gt(0)
    out["probe_multimode"] = (
        primary.eq("MULTIMODE_CONFLICT")
        | secondary.str.contains("MULTIMODE_CONFLICT", na=False)
        | evidence.eq("SEM_MULTIMODE_UNSAFE")
    )

    probe_cols = [
        "probe_handoff_scale",
        "probe_handoff_gauge",
        "probe_good_safe",
        "probe_low_observability",
        "probe_semantic_invalid",
        "probe_multimode",
    ]
    out["in_balanced_probe_universe"] = out[probe_cols].any(axis=1)
    out["probe_tags"] = out.apply(
        lambda row: ";".join(col.replace("probe_", "") for col in probe_cols if bool(row[col])),
        axis=1,
    )
    return out


def balanced_probe_summary(df: pd.DataFrame) -> dict[str, Any]:
    sequence_coverage = int(df.loc[df["in_balanced_probe_universe"], "seq"].nunique())
    counts = {
        "handoff_scale_rows": int(df["probe_handoff_scale"].sum()),
        "handoff_gauge_rows": int(df["probe_handoff_gauge"].sum()),
        "good_safe_rows": int(df["probe_good_safe"].sum()),
        "low_observability_rows": int(df["probe_low_observability"].sum()),
        "semantic_invalid_rows": int(df["probe_semantic_invalid"].sum()),
        "multimode_rows": int(df["probe_multimode"].sum()),
        "probe_universe_rows": int(df["in_balanced_probe_universe"].sum()),
        "sequence_coverage": sequence_coverage,
    }
    checks = {
        "handoff_scale_rows_ge_8": counts["handoff_scale_rows"] >= 8,
        "handoff_gauge_rows_ge_8": counts["handoff_gauge_rows"] >= 8,
        "good_safe_rows_ge_12": counts["good_safe_rows"] >= 12,
        "sequence_coverage_ge_4": sequence_coverage >= 4,
        "include_low_observability_rows": counts["low_observability_rows"] > 0,
        "include_semantic_invalid_rows": counts["semantic_invalid_rows"] > 0,
        "include_multimode_rows": counts["multimode_rows"] > 0,
    }
    return {
        **counts,
        "balanced_probe_gate_pass": all(checks.values()),
        "balanced_probe_checks": checks,
    }


def family_row(
    *,
    intervention: str,
    carrier_body: str,
    family: str = "",
    source: dict[str, Any] | None = None,
    available: bool = True,
    mapping_note: str = "",
    unavailable_reason: str = "",
) -> dict[str, Any]:
    source = source or {}
    actual_traj = truthy(source.get("actual_runtime_trajectory_counterfactual_available"))
    bad_i = to_float(source.get("bad_median_residual_improvement_ratio"))
    good_w = to_float(source.get("good_median_residual_worsen_ratio"))
    good_max_w = to_float(source.get("good_max_residual_worsen_ratio"))
    carrier_delta = finite_nonzero(source.get("bad_mean_boundary_update_norm_reduction")) or finite_nonzero(
        source.get("good_mean_boundary_update_norm_reduction")
    )

    row = {
        "intervention": intervention,
        "carrier_body": carrier_body,
        "available_artifact": bool(available),
        "mapped_v93_family": family,
        "mapping_note": mapping_note,
        "unavailable_reason": unavailable_reason,
        "action_row_count": source.get("action_row_count", ""),
        "labelled_action_row_count": source.get("labelled_action_row_count", ""),
        "sequence_coverage": source.get("sequence_coverage", ""),
        "actual_runtime_trajectory_counterfactual_available": actual_traj,
        "j_handoff_probe_available": False,
        "J_handoff_native_metric": "",
        "J_handoff_probe_metric": "",
        "bad_median_I_J": "",
        "trace_bad_median_residual_improvement": source.get("bad_median_residual_improvement", ""),
        "trace_bad_median_residual_improvement_ratio": source.get("bad_median_residual_improvement_ratio", ""),
        "trace_good_median_residual_worsen": source.get("good_median_residual_worsen", ""),
        "trace_good_median_residual_worsen_ratio": source.get("good_median_residual_worsen_ratio", ""),
        "trace_good_max_residual_worsen": source.get("good_max_residual_worsen", ""),
        "trace_good_max_residual_worsen_ratio": source.get("good_max_residual_worsen_ratio", ""),
        "bad_mean_boundary_update_norm_reduction": source.get("bad_mean_boundary_update_norm_reduction", ""),
        "good_mean_boundary_update_norm_reduction": source.get("good_mean_boundary_update_norm_reduction", ""),
        "carrier_state_delta_observed": carrier_delta,
        "bad_residual_improvement_gate": truthy(source.get("bad_residual_improvement_gate")),
        "good_median_protection_gate": truthy(source.get("good_median_protection_gate")),
        "good_catastrophic_worsen_absent": truthy(source.get("good_catastrophic_worsen_absent")),
        "sequence_coverage_gate": truthy(source.get("sequence_coverage_gate")),
        "beats_best_control_gate": truthy(source.get("beats_best_control_gate")),
        "phase3_sensitive": False,
    }

    # Phase3 sensitivity requires a measured J_handoff probe or a key geometry
    # metric. The available v93 evidence only edits recorded trace fields.
    row["phase3_sensitive_reason"] = (
        "unavailable_artifact"
        if not available
        else "trace_level_only_no_J_handoff_probe"
        if not actual_traj
        else "gate_failed"
    )
    row["trace_bad_I_nonnegative"] = math.isfinite(bad_i) and bad_i >= 0
    row["trace_good_median_worsen_le_0p02"] = math.isfinite(good_w) and good_w <= 0.02
    row["trace_good_max_worsen_le_0p02"] = math.isfinite(good_max_w) and good_max_w <= 0.02
    return row


def build_family_rows(cf_rows: pd.DataFrame) -> list[dict[str, Any]]:
    by_family = {str(row["family"]): row.to_dict() for _, row in cf_rows.iterrows()}
    rows = [
        family_row(
            intervention="READ_NATIVE",
            carrier_body="READ",
            available=False,
            unavailable_reason="native baseline only; no READ intervention artifact with J_handoff_probe",
        ),
        family_row(
            intervention="READ_NO_SEMANTIC_CONTROL",
            carrier_body="READ",
            available=False,
            unavailable_reason="no compliant READ no-semantic probe artifact; read_QK unavailable in Phase2 hidden audit",
        ),
        family_row(
            intervention="READ_HOLD_CONTEXT_ONLY",
            carrier_body="READ",
            available=False,
            unavailable_reason="no compliant READ context-hold intervention artifact",
        ),
        family_row(
            intervention="SWA_NATIVE",
            carrier_body="SWA",
            available=False,
            unavailable_reason="native baseline only; Phase2 has sample-level SWA audit but no full boundary per-head route trace",
        ),
        family_row(
            intervention="SWA_NO_EXTERNAL_MASK",
            carrier_body="SWA",
            available=False,
            unavailable_reason="no compliant full-boundary SWA no-external-mask probe artifact",
        ),
        family_row(
            intervention="SWA_ROUTE_HOLD",
            carrier_body="SWA",
            available=False,
            unavailable_reason="no compliant full-boundary SWA route-hold probe artifact",
        ),
        family_row(
            intervention="SWA_CACHE_HOLD",
            carrier_body="SWA",
            available=False,
            unavailable_reason="no compliant full-boundary SWA cache-hold probe artifact",
        ),
        family_row(
            intervention="MERGE_NATIVE",
            carrier_body="merge_gauge",
            family="CF0_native_trace",
            source=by_family.get("CF0_native_trace", {}),
            mapping_note="native trace baseline from v93 Phase5 upper-bound table",
        ),
        family_row(
            intervention="MERGE_BOUNDARY_HOLD",
            carrier_body="merge_gauge",
            family="CF1_geometry_only_hold",
            source=by_family.get("CF1_geometry_only_hold", {}),
            mapping_note="trace-level hold sets selected merge residual delta and boundary update norm to zero; no trajectory rerun",
        ),
        family_row(
            intervention="MERGE_NO_REFRESH",
            carrier_body="merge_gauge",
            family="CF6_full_object_policy",
            source=by_family.get("CF6_full_object_policy", {}),
            mapping_note="trace-level selected hold/reject/delay family; no trajectory rerun",
        ),
        family_row(
            intervention="MERGE_ROBUST_NATIVE_ONLY",
            carrier_body="merge_gauge",
            available=False,
            unavailable_reason="no v94/v93 artifact reruns native merge with robust-only weighting as a trajectory probe",
        ),
        family_row(
            intervention="TTT_NATIVE",
            carrier_body="TTT",
            available=False,
            unavailable_reason="TTT entry is diagnostic-only and blocked until a merge/SWA carrier passes",
        ),
        family_row(
            intervention="TTT_NO_WRITE_DIAGNOSTIC",
            carrier_body="TTT",
            available=False,
            unavailable_reason="TTT no-write probe is not eligible without carrier pass and has no J_handoff_probe artifact",
        ),
        family_row(
            intervention="OBJECT_SHUFFLE_CONTROL",
            carrier_body="control",
            family="CF7_object_shuffle_control",
            source=by_family.get("CF7_object_shuffle_control", {}),
            mapping_note="v93 control family for beat-control comparison",
        ),
        family_row(
            intervention="COMPONENT_SHUFFLE_CONTROL",
            carrier_body="control",
            family="CF8_component_shuffle_control",
            source=by_family.get("CF8_component_shuffle_control", {}),
            mapping_note="v93 control family for beat-control comparison",
        ),
        family_row(
            intervention="SAME_COUNT_RANDOM_CONTROL",
            carrier_body="control",
            family="CF9_same_count_random_control",
            source=by_family.get("CF9_same_count_random_control", {}),
            mapping_note="v93 same-count random control family",
        ),
    ]
    return rows


def build_effect_rows(phase1: pd.DataFrame, cf6_effect: pd.DataFrame) -> list[dict[str, Any]]:
    phase1_small = phase1[
        [
            "pair_id",
            "failure_type_primary",
            "failure_type_secondary",
            "case_label_offline_only",
            "J_handoff",
            "future_after_overlap",
            "boundary_jump",
            "adjacent_log_scale_jump_offline",
            "trace_path",
        ]
    ].copy()
    merged = cf6_effect.merge(phase1_small, on="pair_id", how="left", suffixes=("_cf6", "_phase1"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        native_boundary = to_float(row.get("native_boundary_update_norm"))
        probe_boundary = to_float(row.get("cf_boundary_update_norm"))
        native_resid = to_float(row.get("native_merge_residual_delta"))
        probe_resid = to_float(row.get("cf_merge_residual_delta"))
        residual_improvement = to_float(row.get("merge_residual_improvement"))
        rows.append(
            {
                "pair_id": row.get("pair_id"),
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "case_label_offline_only": row.get("case_label_offline_only"),
                "base_case_type": row.get("base_case_type"),
                "quality_type": row.get("quality_type"),
                "failure_type_primary": row.get("failure_type_primary"),
                "failure_type_secondary": row.get("failure_type_secondary"),
                "intervention": "MERGE_NO_REFRESH",
                "cf6_action": row.get("cf6_action"),
                "p5_combined_object_policy": row.get("p5_combined_object_policy"),
                "native_merge_residual_delta": row.get("native_merge_residual_delta"),
                "probe_merge_residual_delta": row.get("cf_merge_residual_delta"),
                "trace_residual_improvement": row.get("merge_residual_improvement"),
                "trace_residual_improvement_ratio": residual_improvement / (abs(native_resid) + 1e-12)
                if math.isfinite(residual_improvement) and math.isfinite(native_resid)
                else "",
                "native_boundary_update_norm": row.get("native_boundary_update_norm"),
                "probe_boundary_update_norm": row.get("cf_boundary_update_norm"),
                "carrier_state_delta": native_boundary - probe_boundary
                if math.isfinite(native_boundary) and math.isfinite(probe_boundary)
                else "",
                "residual_effect_sign": row.get("residual_effect_sign"),
                "J_handoff_native": row.get("J_handoff"),
                "J_handoff_probe": "",
                "Delta_J_handoff": "",
                "I_J": "",
                "abs_log_scale_jump_native": row.get("adjacent_log_scale_jump_offline"),
                "abs_log_scale_jump_probe": "",
                "future_error_native": row.get("future_after_overlap"),
                "future_error_probe": "",
                "boundary_jump_native": row.get("boundary_jump"),
                "boundary_jump_probe": "",
                "good_row_worsen": "",
                "bad_row_improvement": "",
                "probe_scope": "trace_level_upper_bound_only_no_trajectory_rerun",
                "trace_path": row.get("trace_path_cf6") or row.get("trace_path"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2)
    parser.add_argument("--v93-phase5-dir", type=Path, default=V93_PHASE5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase3_neutral_causal_sensitivity")
    args = parser.parse_args()

    phase1_summary = read_json(args.phase1_dir / "phase1_gate_summary.json")
    phase2_summary = read_json(args.phase2_dir / "phase2_gate_summary.json")
    cf_summary = read_json(args.v93_phase5_dir / "counterfactual_upper_bound_summary.json")
    phase1_rows = pd.read_csv(args.phase1_dir / "boundary_failure_rows.csv")
    cf_rows = pd.read_csv(args.v93_phase5_dir / "counterfactual_upper_bound_rows.csv")
    cf6_effect = pd.read_csv(args.v93_phase5_dir / "counterfactual_cf6_effect_rows.csv")

    probe_rows = add_probe_flags(phase1_rows)
    probe_summary = balanced_probe_summary(probe_rows)
    family_rows = build_family_rows(cf_rows)
    effect_rows = build_effect_rows(probe_rows, cf6_effect)

    sensitive = [row for row in family_rows if row["phase3_sensitive"]]
    trace_delta_rows = [row for row in family_rows if row.get("carrier_state_delta_observed")]
    unavailable_interventions = [row for row in family_rows if not row["available_artifact"]]
    merge_trace_families = [
        row
        for row in family_rows
        if row["carrier_body"] == "merge_gauge" and row["available_artifact"] and row["intervention"] != "MERGE_NATIVE"
    ]

    checks = {
        "phase1_gate_pass": truthy(phase1_summary.get("phase1_gate_pass")),
        "phase2_gate_pass": truthy(phase2_summary.get("phase2_gate_pass")),
        "balanced_probe_gate_pass": probe_summary["balanced_probe_gate_pass"],
        "any_carrier_trace_delta_observed": bool(trace_delta_rows),
        "any_runtime_trajectory_probe_available": any(
            row.get("actual_runtime_trajectory_counterfactual_available") for row in family_rows
        ),
        "any_j_handoff_probe_available": any(row.get("j_handoff_probe_available") for row in family_rows),
        "any_sensitive_memory_body": bool(sensitive),
        "merge_trace_upper_bound_counterfactual_pass": truthy(cf_summary.get("phase5_counterfactual_gate_pass")),
        "merge_trace_beats_controls": any(
            row.get("beats_best_control_gate") for row in merge_trace_families
        ),
        "merge_trace_good_catastrophic_absent": all(
            row.get("good_catastrophic_worsen_absent") for row in merge_trace_families
        ),
    }

    blockers: list[str] = []
    if not checks["balanced_probe_gate_pass"]:
        blockers.append(
            "balanced_probe_set_insufficient:"
            f"handoff_scale={probe_summary['handoff_scale_rows']}/8,"
            f"handoff_gauge={probe_summary['handoff_gauge_rows']}/8,"
            f"good_safe={probe_summary['good_safe_rows']}/12"
        )
    if not checks["any_runtime_trajectory_probe_available"] or not checks["any_j_handoff_probe_available"]:
        blockers.append("no_runtime_or_J_handoff_probe_counterfactual_available")
    if not checks["merge_trace_upper_bound_counterfactual_pass"]:
        blockers.append(str(cf_summary.get("blocker") or "merge_trace_upper_bound_gate_failed"))
    if not checks["any_sensitive_memory_body"]:
        blockers.append("no_memory_body_shows_phase3_causal_sensitivity")

    summary = {
        "phase": "Phase3_neutral_causal_sensitivity_probe",
        "entered": checks["phase1_gate_pass"] and checks["phase2_gate_pass"],
        "phase3_gate_pass": bool(sensitive) and probe_summary["balanced_probe_gate_pass"],
        "sensitive_memory_bodies": [row["carrier_body"] for row in sensitive],
        "checks": checks,
        "balanced_probe": probe_summary,
        "carrier_trace_delta_family_count": len(trace_delta_rows),
        "unavailable_intervention_count": len(unavailable_interventions),
        "available_merge_trace_interventions": [row["intervention"] for row in merge_trace_families],
        "v93_counterfactual_summary_path": str(args.v93_phase5_dir / "counterfactual_upper_bound_summary.json"),
        "v93_counterfactual_model": cf_summary.get("counterfactual_model"),
        "trace_level_upper_bound_only": bool(cf_summary.get("trace_level_upper_bound_only", True)),
        "actual_runtime_trajectory_counterfactual_available": bool(
            cf_summary.get("actual_runtime_trajectory_counterfactual_available")
        ),
        "blocker": ";".join(blockers),
        "stop_rule_triggered": "Phase3 no memory body shows causal sensitivity",
        "phase4_allowed": False,
        "counterfactual_allowed": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "repair_attempt_result": (
            "Existing v93 trace-level merge/gauge upper-bound evidence was reused and audited. "
            "It changes carrier trace fields but lacks runtime/J_handoff probe geometry and fails "
            "good/bad/control gates; READ/SWA/TTT intervention artifacts are unavailable or blocked."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe_out = probe_rows.loc[probe_rows["in_balanced_probe_universe"]].copy()
    probe_out.to_csv(args.out_dir / "balanced_probe_set_rows.csv", index=False)
    write_csv(args.out_dir / "intervention_family_summary.csv", family_rows)
    write_csv(args.out_dir / "intervention_effect_rows.csv", effect_rows)
    write_json(args.out_dir / "phase3_gate_summary.json", summary)

    print(f"phase3_gate_pass={summary['phase3_gate_pass']}")
    print(f"balanced_probe_gate_pass={probe_summary['balanced_probe_gate_pass']}")
    print(f"handoff_scale_rows={probe_summary['handoff_scale_rows']}")
    print(f"handoff_gauge_rows={probe_summary['handoff_gauge_rows']}")
    print(f"good_safe_rows={probe_summary['good_safe_rows']}")
    print(f"carrier_trace_delta_family_count={len(trace_delta_rows)}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
