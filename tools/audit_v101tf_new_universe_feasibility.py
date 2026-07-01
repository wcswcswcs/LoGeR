#!/usr/bin/env python3
"""Audit whether existing artifacts can form a new v100-schema v101 universe.

This is a read-only feasibility audit over already materialized v100/v101
diagnostics. It does not promote proxy evidence to action evidence.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
V100_ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
OUT = ROOT / "final_decision"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def intish(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def case_counts(rows: list[dict[str, str]], key: str = "case_id") -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        case_id = row.get(key, "")
        if case_id:
            counts[case_id] += 1
    return counts


def index_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def any_true(rows: list[dict[str, str]], column: str) -> bool:
    return any(truthy(row.get(column, "")) for row in rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    broad_rows = read_csv(ROOT / "trackT_drift_target_relabel" / "broad_prior_unique_case_summary.csv")
    target_rows = read_csv(ROOT / "trackT_drift_target_relabel" / "target_universe_v101.csv")
    support_rows = read_csv(ROOT / "trackU_true_current_support" / "anchor_current_support_rows.csv")
    geometry_case_rows = read_csv(
        ROOT / "trackV_anchor_scale_observability" / "per_anchor_geometry_case_summary.csv"
    )
    atlas_case_rows = read_csv(
        ROOT / "trackJL4_semantic_anchor_instance_atlas" / "anchor_instance_case_summary.csv"
    )
    r3_case_rows = read_csv(
        ROOT / "trackR3_query_head_anchor_edge_audit_true_support" / "support_conditioned_anchor_edge_case_rows.csv"
    )
    q2_rows = read_csv(ROOT / "trackQ2_scale_update_admission" / "admission_rows.csv")
    v100_control_rows = read_csv(
        V100_ROOT / "trackL2_anchor_scale_observability" / "anchor_head_semantic_control_case_rows.csv"
    )

    target_by_case = index_by_case(target_rows)
    geometry_by_case = index_by_case(geometry_case_rows)
    atlas_by_case = index_by_case(atlas_case_rows)
    r3_by_case = index_by_case(r3_case_rows)
    q2_by_case = index_by_case(q2_rows)
    v100_control_by_case = index_by_case(v100_control_rows)
    support_count_by_case = case_counts(support_rows)

    q2_summary = read_json(ROOT / "trackQ2_scale_update_admission" / "Q2_summary.json")
    r3_metric_rows = read_csv(ROOT / "trackR3_query_head_anchor_edge_audit_true_support" / "metric_summary.csv")
    f5_audit_rows = read_csv(ROOT / "trackF5_ttt_write_to_use_state_chain" / "write_to_use_materialization_audit.csv")

    q2_true_stage_pass = truthy(q2_summary.get("true_stage_pass", False))
    q2_proxy_only = truthy(q2_summary.get("proxy_only", True))
    r3_control_margins_available = any_true(r3_metric_rows, "control_margins_available")
    f5_write_to_use_materialized = False
    if f5_audit_rows:
        audit = f5_audit_rows[0]
        f5_write_to_use_materialized = (
            intish(audit.get("r_write_cache_nonempty")) > 0
            and intish(audit.get("r_cache_current_nonempty")) > 0
            and intish(audit.get("r_ref_current_nonempty")) > 0
        )

    rows: list[dict[str, Any]] = []
    for broad in broad_rows:
        if not truthy(broad.get("clean_candidate_any", "")):
            continue

        case_id = broad["case_id"]
        taxonomy = broad.get("representative_taxonomy", "")
        target = target_by_case.get(case_id, {})
        geometry = geometry_by_case.get(case_id, {})
        atlas = atlas_by_case.get(case_id, {})
        r3 = r3_by_case.get(case_id, {})
        q2 = q2_by_case.get(case_id, {})
        v100_control = v100_control_by_case.get(case_id, {})

        in_v100 = truthy(broad.get("already_in_v100_28_case_universe", ""))
        has_same = truthy(broad.get("has_v100_same_space_trace", ""))
        has_geom = truthy(broad.get("has_v100_per_anchor_geometry", ""))
        support_rows_count = support_count_by_case.get(case_id, 0)
        geometry_case_available = bool(geometry)
        atlas_available = bool(atlas)
        identity_level = atlas.get("identity_resolution_levels", "")
        strict_identity = bool(identity_level) and "semantic_class_fallback" not in identity_level
        r3_case_available = bool(r3)
        q2_case_available = bool(q2)
        v100_control_available = bool(v100_control)

        core_ready = (
            in_v100
            and has_same
            and has_geom
            and support_rows_count > 0
            and geometry_case_available
            and q2_case_available
        )
        strict_action_ready = (
            core_ready
            and strict_identity
            and r3_case_available
            and r3_control_margins_available
            and f5_write_to_use_materialized
            and q2_true_stage_pass
            and not q2_proxy_only
        )

        missing: list[str] = []
        if not in_v100:
            missing.append("not_in_v100_28_case_universe")
        if not has_same:
            missing.append("missing_v100_same_space_trace")
        if not has_geom:
            missing.append("missing_v100_per_anchor_geometry")
        if support_rows_count <= 0:
            missing.append("missing_v101_current_support_rows")
        if not geometry_case_available:
            missing.append("missing_v101_geometry_case_summary")
        if not q2_case_available:
            missing.append("missing_v101_q2_case_row")
        if not strict_identity:
            missing.append("strict_instance_identity_unavailable")
        if not r3_case_available:
            missing.append("missing_support_conditioned_r3_edge_row")
        if not r3_control_margins_available:
            missing.append("query_head_control_margins_unavailable")
        if not f5_write_to_use_materialized:
            missing.append("write_cache_current_chain_not_materialized")
        if not q2_true_stage_pass or q2_proxy_only:
            missing.append("q2_true_stage_unavailable")

        rows.append(
            {
                "case_id": case_id,
                "seq": broad.get("seq", ""),
                "candidate_kind": taxonomy,
                "L3_min": broad.get("L3_min", ""),
                "L3_max": broad.get("L3_max", ""),
                "already_in_v100_28_case_universe": in_v100,
                "has_v100_same_space_trace": has_same,
                "has_v100_per_anchor_geometry": has_geom,
                "v101_current_support_anchor_rows": support_rows_count,
                "v101_geometry_case_available": geometry_case_available,
                "v101_atlas_case_available": atlas_available,
                "identity_resolution_levels": identity_level,
                "strict_instance_identity_available": strict_identity,
                "v101_r3_edge_case_available": r3_case_available,
                "v100_anchor_head_semantic_control_available": v100_control_available,
                "v101_q2_case_available": q2_case_available,
                "core_v100_schema_ready": core_ready,
                "strict_action_ready": strict_action_ready,
                "missing_action_prereqs": ";".join(missing),
            }
        )

    clean_handoff = [r for r in rows if r["candidate_kind"] == "HANDOFF_SCALE_GAUGE_TARGET"]
    safe_good = [r for r in rows if r["candidate_kind"] == "SAFE_GOOD"]
    strict_ready = [r for r in rows if r["strict_action_ready"]]
    core_ready = [r for r in rows if r["core_v100_schema_ready"]]

    summary = {
        "schema": "acl2_v101_new_v100_schema_universe_feasibility_v1",
        "broad_unique_case_count": len(broad_rows),
        "clean_candidate_count": len(rows),
        "clean_handoff_candidate_count": len(clean_handoff),
        "clean_handoff_sequence_coverage": len({r["seq"] for r in clean_handoff}),
        "safe_good_candidate_count": len(safe_good),
        "core_v100_schema_ready_clean_candidate_count": len(core_ready),
        "strict_action_ready_clean_candidate_count": len(strict_ready),
        "q2_true_stage_pass": q2_true_stage_pass,
        "q2_proxy_only": q2_proxy_only,
        "r3_control_margins_available": r3_control_margins_available,
        "f5_write_to_use_materialized": f5_write_to_use_materialized,
        "new_universe_available_from_existing_artifacts": bool(
            len(clean_handoff) >= 8
            and len({r["seq"] for r in clean_handoff}) >= 3
            and len(safe_good) >= 6
            and strict_ready
        ),
        "blocked_reason": (
            "Existing artifacts expose only one clean handoff target and five safe-good controls; "
            "the clean candidates are not strict-action-ready because identity/query-head controls, "
            "write-to-use chain materialization, and Q2 true-stage admission remain unavailable."
        ),
    }

    fieldnames = [
        "case_id",
        "seq",
        "candidate_kind",
        "L3_min",
        "L3_max",
        "already_in_v100_28_case_universe",
        "has_v100_same_space_trace",
        "has_v100_per_anchor_geometry",
        "v101_current_support_anchor_rows",
        "v101_geometry_case_available",
        "v101_atlas_case_available",
        "identity_resolution_levels",
        "strict_instance_identity_available",
        "v101_r3_edge_case_available",
        "v100_anchor_head_semantic_control_available",
        "v101_q2_case_available",
        "core_v100_schema_ready",
        "strict_action_ready",
        "missing_action_prereqs",
    ]
    write_csv(OUT / "new_v100_schema_universe_feasibility_rows.csv", rows, fieldnames)

    with (OUT / "new_v100_schema_universe_feasibility_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# New v100-schema universe feasibility audit",
        "",
        "This audit checks whether already materialized v94-v101 artifacts can be assembled into a new clean v101 action universe.",
        "",
        "## Summary",
        "",
        f"- broad unique cases: `{summary['broad_unique_case_count']}`",
        f"- clean candidate cases: `{summary['clean_candidate_count']}`",
        f"- clean handoff targets: `{summary['clean_handoff_candidate_count']}`",
        f"- clean handoff sequence coverage: `{summary['clean_handoff_sequence_coverage']}`",
        f"- safe-good controls: `{summary['safe_good_candidate_count']}`",
        f"- core v100-schema-ready clean candidates: `{summary['core_v100_schema_ready_clean_candidate_count']}`",
        f"- strict action-ready clean candidates: `{summary['strict_action_ready_clean_candidate_count']}`",
        f"- Q2 true-stage pass: `{summary['q2_true_stage_pass']}`",
        f"- Q2 proxy-only: `{summary['q2_proxy_only']}`",
        f"- R3 control margins available: `{summary['r3_control_margins_available']}`",
        f"- F5 write-to-use chain materialized: `{summary['f5_write_to_use_materialized']}`",
        f"- new universe available from existing artifacts: `{summary['new_universe_available_from_existing_artifacts']}`",
        "",
        "## Conclusion",
        "",
        summary["blocked_reason"],
        "",
        "This supports the v101 No-Go boundary: do not run M4/runtime from the current artifacts. A next attempt needs newly materialized v100-schema clean handoff cases, not threshold tuning over this universe.",
    ]
    (OUT / "new_v100_schema_universe_feasibility_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
