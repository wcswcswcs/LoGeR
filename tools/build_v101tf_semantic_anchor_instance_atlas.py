#!/usr/bin/env python3
"""Build a diagnostic semantic anchor-instance atlas for ACL2 v101.

The atlas joins current support, per-anchor geometry observability, role, and
state rows.  It remains diagnostic because stable semantic instance/component
identity is not available.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
OUT = ROOT / "trackJL4_semantic_anchor_instance_atlas"
TRACK_U = ROOT / "trackU_true_current_support"
TRACK_V = ROOT / "trackV_anchor_scale_observability"
TRACK_W = ROOT / "trackW_anchor_memory_role"
TRACK_S2 = ROOT / "trackS2_anchor_state_estimator"
TRACK_T = ROOT / "trackT_drift_target_relabel"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id", "")), str(row.get("anchor_id", ""))


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        k = key(row)
        if k[0] and k[1] and k not in out:
            out[k] = row
    return out


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def main() -> None:
    support_rows = by_key(read_rows(TRACK_U / "anchor_current_support_rows.csv"))
    role_rows = by_key(read_rows(TRACK_W / "anchor_role_rows.csv"))
    state_rows = by_key(read_rows(TRACK_S2 / "anchor_state_rows.csv"))
    repaired_geom_rows = by_key(read_rows(TRACK_V / "per_anchor_geometry_observability_rows.csv"))
    target_rows = {row.get("case_id", ""): row for row in read_rows(TRACK_T / "target_universe_v101.csv")}
    all_keys = sorted(set(support_rows) | set(role_rows) | set(state_rows) | set(repaired_geom_rows))

    atlas_rows: list[dict[str, Any]] = []
    for case_id, anchor_id in all_keys:
        support = support_rows.get((case_id, anchor_id), {})
        role = role_rows.get((case_id, anchor_id), {})
        state = state_rows.get((case_id, anchor_id), {})
        geom = repaired_geom_rows.get((case_id, anchor_id), {})
        target = target_rows.get(case_id, {})
        identity_level = support.get("identity_resolution_level", "missing")
        component_id = ""
        anchor_instance_id = f"{case_id}:{anchor_id}"
        current_support = support.get("S_cur_combined", role.get("S_cur_combined", ""))
        scale_observability = geom.get("O_scale_repaired", role.get("O_scale", ""))
        same_space_state = state.get("state_status", "")
        query_usage = {
            "query_hit_max": support.get("query_hit_max", role.get("query_hit_max", "")),
            "R_same": support.get("R_same", role.get("R_same", "")),
        }
        ttt_write_use_chain = {
            "r_write_cache": state.get("r_write_cache", ""),
            "r_cache_current": state.get("r_cache_current", ""),
            "r_ref_current": state.get("r_ref_current", ""),
            "claim": "not_materialized_in_v101_atlas",
        }
        swa_handoff_effect = {
            "target_taxonomy": target.get("target_taxonomy", ""),
            "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
            "q2_or_runtime_effect": "not_run",
        }
        read_support_effect = {
            "allowed_READ_behavior": role.get("allowed_READ_behavior", ""),
            "read_provider_claim": "same-space support provider only; no READ full action",
        }
        atlas_rows.append(
            {
                "anchor_instance_id": anchor_instance_id,
                "case_id": case_id,
                "boundary_id": support.get("boundary_id", role.get("boundary_id", "")),
                "anchor_id": anchor_id,
                "semantic_label": support.get("semantic_label", role.get("semantic_label", "")),
                "component_id": component_id,
                "identity_resolution_level": identity_level,
                "target_taxonomy": target.get("target_taxonomy", ""),
                "memory_role": role.get("role", ""),
                "role_confidence": role.get("role_confidence", ""),
                "current_support": current_support,
                "scale_observability": scale_observability,
                "same_space_state": same_space_state,
                "K_anchor": state.get("K_anchor", ""),
                "query_head_usage": query_usage,
                "TTT_write_use_chain": ttt_write_use_chain,
                "SWA_handoff_effect": swa_handoff_effect,
                "READ_support_effect": read_support_effect,
                "support_source_flags": support.get("support_source_flags", ""),
                "support_quality": support.get("support_quality", ""),
                "geometry_source_level": geom.get("geometry_source_level", role.get("pointmap_depth_source_level", "")),
                "runtime_action_allowed": False,
                "claim_level": "diagnostic_anchor_instance_atlas_no_action",
            }
        )

    target_taxonomies = {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}
    target_atlas_rows = [row for row in atlas_rows if row.get("target_taxonomy") in target_taxonomies]
    target_anchor_keys = {
        key(row)
        for row in read_rows(TRACK_U / "anchor_current_support_rows.csv")
        if row.get("target_taxonomy") in target_taxonomies
    }
    atlas_target_keys = {(row["case_id"], row["anchor_id"]) for row in target_atlas_rows}
    target_coverage = len(atlas_target_keys & target_anchor_keys) / max(len(target_anchor_keys), 1)
    identity_counts = Counter(row["identity_resolution_level"] for row in atlas_rows)
    role_counts = Counter(row["memory_role"] for row in atlas_rows)
    taxonomy_counts = Counter(row["target_taxonomy"] for row in atlas_rows)
    per_case_rows = []
    for case_id in sorted({row["case_id"] for row in atlas_rows}):
        parts = [row for row in atlas_rows if row["case_id"] == case_id]
        per_case_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": target_rows.get(case_id, {}).get("target_taxonomy", ""),
                "anchor_instance_count": len(parts),
                "mean_current_support": mean([row["current_support"] for row in parts]),
                "mean_scale_observability": mean([row["scale_observability"] for row in parts]),
                "role_count": len({row["memory_role"] for row in parts if row["memory_role"]}),
                "identity_resolution_levels": ";".join(sorted({row["identity_resolution_level"] for row in parts})),
            }
        )

    distinguishes_instance = any(row["component_id"] for row in atlas_rows) and not all(
        row["identity_resolution_level"] == "semantic_class_fallback" for row in atlas_rows
    )
    records_role_transitions = False
    gate = target_coverage >= 0.90 and distinguishes_instance and records_role_transitions
    blockers = []
    if target_coverage < 0.90:
        blockers.append("Atlas target-anchor coverage is below 90%.")
    if not distinguishes_instance:
        blockers.append("Atlas does not distinguish semantic region label from stable instance/component id.")
    if not records_role_transitions:
        blockers.append("Role transitions across chunks are not materialized from current artifacts.")

    summary = {
        "schema": "acl2_v101_trackJL4_semantic_anchor_instance_atlas_v1",
        "status": "complete_diagnostic_blocked",
        "gate_pass": gate,
        "atlas_row_count": len(atlas_rows),
        "case_count": len({row["case_id"] for row in atlas_rows}),
        "target_anchor_row_count": len(target_anchor_keys),
        "target_anchor_coverage": target_coverage,
        "identity_resolution_level_counts": dict(identity_counts),
        "memory_role_counts": dict(role_counts),
        "target_taxonomy_counts": dict(taxonomy_counts),
        "distinguishes_region_label_from_instance": distinguishes_instance,
        "records_role_transitions_across_chunks": records_role_transitions,
        "runtime_action_allowed": False,
        "blockers": blockers,
        "claim": "Atlas rows were materialized for diagnostics, but instance/component identity and role transitions are insufficient for JL4 pass.",
    }

    write_rows(OUT / "anchor_instance_atlas.csv", atlas_rows)
    write_rows(OUT / "anchor_instance_case_summary.csv", per_case_rows)
    write_rows(OUT / "identity_resolution_gap_rows.csv", [row for row in atlas_rows if row["identity_resolution_level"] == "semantic_class_fallback"])
    write_rows(OUT / "role_transition_rows.csv", [])
    write_json(OUT / "JL4_summary.json", summary)
    write_json(OUT / "blocked_summary.json", {**summary, "run_allowed": False, "status": "complete_diagnostic_blocked"})
    write_rows(
        OUT / "gate_checks.csv",
        [
            {"gate": "target_anchor_coverage_ge_90pct", "pass": target_coverage >= 0.90, "observed": target_coverage},
            {"gate": "distinguishes_region_label_from_instance", "pass": distinguishes_instance, "observed": distinguishes_instance},
            {"gate": "records_role_transitions_across_chunks", "pass": records_role_transitions, "observed": records_role_transitions},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
    )
    write_text(
        OUT / "atlas_report.md",
        "# Track JL4 Semantic Anchor Instance Atlas\n\n"
        f"- Atlas rows: {summary['atlas_row_count']}\n"
        f"- Target anchor coverage: {summary['target_anchor_coverage']}\n"
        f"- Identity resolution levels: `{json.dumps(dict(identity_counts), sort_keys=True)}`\n"
        f"- Role counts: `{json.dumps(dict(role_counts), sort_keys=True)}`\n"
        f"- Gate pass: {summary['gate_pass']}\n\n"
        "Blockers:\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
    )
    write_text(OUT / "failure_report.md", "\n".join(f"- {item}" for item in blockers))
    write_text(
        OUT / "what_would_have_to_be_true_to_pass.md",
        "Stable component/instance ids must cover target anchors and role transitions across chunks must be materialized.",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
