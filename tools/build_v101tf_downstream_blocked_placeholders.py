#!/usr/bin/env python3
"""Write blocked placeholders for v101 downstream tracks.

These artifacts make non-execution explicit when upstream gates block N3/C5/F5
R3/M4/runtime/full-validation work.  They do not claim any downstream pass.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL_DECISION = ROOT / "final_decision/final_decision.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in keys})


TRACKS = [
    {
        "track": "N3",
        "dir": "trackN3_anchor_identity_graph_cleaned_targets",
        "planned_outputs": "anchor graph pattern rows, control margins, N3_summary.json",
        "requires": "Track T clean HANDOFF target universe and safe-good pool; Track U/V/S2/Q2 rows for S_cur/O_scale/K_anchor.",
    },
    {
        "track": "C5",
        "dir": "trackC5_identity_latent_gauge_with_support",
        "planned_outputs": "high_hit_anchor_groups.csv, harmful_safe_use_rows.csv, latent_support_interaction_metrics.csv, C5_summary.json",
        "requires": "Sequence-covered clean targets, true/support observability evidence, semantic/identity/query-head controls.",
    },
    {
        "track": "F5",
        "dir": "trackF5_ttt_write_to_use_state_chain",
        "planned_outputs": "write-to-cache/use risk rows and F5_summary.json",
        "requires": "Support-conditioned target/control universe and identity/semantic rotation controls.",
    },
    {
        "track": "R3",
        "dir": "trackR3_query_head_anchor_edge_audit_true_support",
        "planned_outputs": "edge audit rows, query-head/anchor random margins, R3_summary.json",
        "requires": "True-support/observability joined edge rows and clean target universe.",
    },
    {
        "track": "M4",
        "dir": "trackM4_state_machine_carrier_to_action_simulator",
        "planned_outputs": "action_family_rows.csv, simulator_metrics.csv, M4_summary.json",
        "requires": "Track T pass, Track Q2 true-stage pass or upstream action family pass, and usable support/observability controls.",
    },
    {
        "track": "DH4",
        "dir": "trackDH4_read_current_support_refresh_provider",
        "planned_outputs": "READ provider diagnostics and DH4_summary.json",
        "requires": "Current support use case that improves U/Q2 without promoting READ full action.",
    },
    {
        "track": "JL4",
        "dir": "trackJL4_semantic_anchor_instance_atlas",
        "planned_outputs": "anchor_instance_atlas.csv and JL4_summary.json",
        "requires": "Stable instance/component identity covering >=90% target anchors.",
    },
    {
        "track": "RuntimePilots",
        "dir": "runtime_pilots_or_blocked",
        "planned_outputs": "pilot run manifests, trace metrics, L3 deltas",
        "requires": "Track T pass, Track U pass, Track V pass or explicit proxy-only blocked path, Track Q2/M4 pass, and M4 simulator pass.",
    },
    {
        "track": "FullValidation",
        "dir": "full_validation_or_blocked",
        "planned_outputs": "full validation metrics and final method decision",
        "requires": "A 12-case runtime L3 pilot pass, good harm <=2%, and random/control margins.",
    },
]


def main() -> None:
    final = read_json(FINAL_DECISION)
    diagnostic_summary_paths = {
        "N3": ROOT / "trackN3_anchor_identity_graph_cleaned_targets/N3_summary.json",
        "C5": ROOT / "trackC5_identity_latent_gauge_with_support/C5_summary.json",
        "F5": ROOT / "trackF5_ttt_write_to_use_state_chain/F5_summary.json",
        "R3": ROOT / "trackR3_query_head_anchor_edge_audit_true_support/R3_summary.json",
        "DH4": ROOT / "trackDH4_read_current_support_refresh_provider/DH4_summary.json",
        "JL4": ROOT / "trackJL4_semantic_anchor_instance_atlas/JL4_summary.json",
    }
    diagnostic_summaries = {track: read_json(path) for track, path in diagnostic_summary_paths.items()}
    blockers = [
        "Track T gate_pass=false: clean HANDOFF target count/coverage and SAFE_GOOD count are insufficient.",
        "Track Q2 true_stage_pass=false: no true-stage admission gate is available.",
        "Track V gate_pass=false: per-anchor geometry materialized but strict observability/control gate is not passed.",
        "Track M4 run_allowed=false: runtime/full validation cannot start.",
    ]
    rows: list[dict[str, Any]] = []
    for item in TRACKS:
        out = ROOT / item["dir"]
        diagnostic_run = bool(diagnostic_summaries.get(item["track"]))
        if diagnostic_run:
            diag_summary = diagnostic_summaries[item["track"]]
            diag_path = diagnostic_summary_paths[item["track"]]
            diag_count_keys = [
                "atlas_row_count",
                "provider_row_count",
                "anchor_row_count",
                "anchor_group_row_count",
                "case_count",
                "joined_anchor_edge_row_count",
            ]
            diag_count = ""
            for key in diag_count_keys:
                if key in diag_summary:
                    diag_count = diag_summary.get(key, "")
                    break
            summary = {
                "schema": "acl2_v101_downstream_diagnostic_run_blocked_placeholder_v1",
                "track": item["track"],
                "status": "complete_diagnostic_blocked",
                "gate_pass": False,
                "run_allowed": False,
                "runtime_action_allowed": False,
                "full_validation_allowed": False,
                "planned_outputs": item["planned_outputs"],
                "requires": item["requires"],
                "blocking_upstream_gates": blockers,
                "diagnostic_summary_ref": str(diag_path),
                "diagnostic_row_count": diag_count,
                "diagnostic_target_anchor_coverage": diag_summary.get("target_anchor_coverage", ""),
                "diagnostic_blockers": "; ".join(diag_summary.get("blockers", diag_summary.get("failure_lines", []))),
                "final_decision_ref": str(FINAL_DECISION),
                "final_taxonomy": final.get("final_taxonomy", ""),
                "claim": f"{item['track']} diagnostic was run, but downstream/runtime action remains blocked.",
            }
            write_json(out / "blocked_summary.json", summary)
            write_rows(
                out / "not_run_manifest.csv",
                [
                    {
                        "track": item["track"],
                        "not_run": False,
                        "diagnostic_run": True,
                        "reason": "; ".join(diag_summary.get("blockers", diag_summary.get("failure_lines", []))),
                        "planned_outputs": item["planned_outputs"],
                    }
                ],
            )
            rows.append(
                {
                    "track": item["track"],
                    "dir": str(out),
                    "status": "complete_diagnostic_blocked",
                    "run_allowed": False,
                }
            )
            continue
        summary = {
            "schema": "acl2_v101_downstream_blocked_placeholder_v1",
            "track": item["track"],
            "status": "blocked_not_run",
            "gate_pass": False,
            "run_allowed": False,
            "runtime_action_allowed": False,
            "full_validation_allowed": False,
            "planned_outputs": item["planned_outputs"],
            "requires": item["requires"],
            "blocking_upstream_gates": blockers,
            "final_decision_ref": str(FINAL_DECISION),
            "final_taxonomy": final.get("final_taxonomy", ""),
            "claim": "This is a blocked placeholder only; no downstream experiment success is claimed.",
        }
        write_json(out / "blocked_summary.json", summary)
        write_rows(
            out / "gate_checks.csv",
            [
                {"gate": "trackT_gate_pass", "pass": final.get("trackT_gate_pass", False)},
                {"gate": "trackQ2_true_stage_pass", "pass": final.get("trackQ2_true_stage_pass", False)},
                {"gate": "trackV_gate_pass", "pass": final.get("trackV_gate_pass", False)},
                {"gate": "trackM4_run_allowed", "pass": final.get("trackM4_run_allowed", False)},
                {"gate": "runtime_action_allowed", "pass": final.get("runtime_action_allowed", False)},
            ],
        )
        write_rows(
            out / "not_run_manifest.csv",
            [
                {
                    "track": item["track"],
                    "not_run": True,
                    "reason": "; ".join(blockers),
                    "planned_outputs": item["planned_outputs"],
                }
            ],
        )
        write_text(out / "failure_report.md", "\n".join(f"- {blocker}" for blocker in blockers))
        write_text(out / "what_would_have_to_be_true_to_pass.md", item["requires"])
        rows.append({"track": item["track"], "dir": str(out), "status": "blocked_not_run", "run_allowed": False})
    write_rows(ROOT / "final_decision/downstream_blocked_manifest.csv", rows)
    print(json.dumps({"schema": "acl2_v101_downstream_blocked_manifest_v1", "blocked_track_count": len(rows), "tracks": [row["track"] for row in rows]}, sort_keys=True))


if __name__ == "__main__":
    main()
