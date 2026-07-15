#!/usr/bin/env python3
"""Build a post-final amendment for v118 R4 HS-HG follow-up runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
OUT_JSON = RESULT_ROOT / "V118_POSTFINAL_R4_AMENDMENT_SUMMARY.json"
OUT_CSV = RESULT_ROOT / "V118_POSTFINAL_BRANCH_STATUS_AMENDMENT.csv"
OUT_REPORT = RESULT_ROOT / "V118_POSTFINAL_R4_AMENDMENT_REPORT.md"

STAGE3_R4 = RESULT_ROOT / "stage3_r4_internal_reliability_repair/stage3_r4_internal_reliability_repair_summary.json"
STAGE4_SUMMARIES = [
    RESULT_ROOT / "stage4_r4_hs_hg_internal_only/summary/stage4_r4_hs_hg_internal_only_summary.json",
    RESULT_ROOT / "stage4_r4_hs_hg_sparse_internal/summary/stage4_r4_hs_hg_internal_only_summary.json",
    RESULT_ROOT / "stage4_r4_hs_hg_sparse_internal_tiny_tight/summary/stage4_r4_hs_hg_internal_only_summary.json",
    RESULT_ROOT / "stage4_r4_hs_hg_sparse_internal_tiny_tight_reverse/summary/stage4_r4_hs_hg_internal_only_summary.json",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv_rows(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def stage4_row(path: Path) -> dict[str, Any]:
    data = read_json(path)
    agg = data["aggregate"]
    return {
        "branch": "HS-HG",
        "variant": agg.get("candidate_name"),
        "control": agg.get("control"),
        "pilot_geometry_gate_pass": bool(agg.get("pilot_geometry_gate", {}).get("pass")),
        "median_full_ATE_rel_improvement": agg.get("median_full_ATE_rel_improvement"),
        "median_rolling_p90_rel_improvement": agg.get("median_rolling_p90_rel_improvement"),
        "max_full_ATE_harm_rel": agg.get("max_full_ATE_harm_rel"),
        "median_segment_scale_rel_improvement": agg.get("median_segment_scale_rel_improvement"),
        "segment_scale_not_worse_all": bool(agg.get("segment_scale_not_worse_all")),
        "improved_seq_count_full_ATE": agg.get("improved_seq_count_full_ATE"),
        "artifact": rel(path),
    }


def main() -> None:
    stage3 = read_json(STAGE3_R4)
    stage4_rows = [stage4_row(path) for path in STAGE4_SUMMARIES]
    any_geometry_pass = any(row["pilot_geometry_gate_pass"] for row in stage4_rows)
    hs_hg_status = "COMPLETE_NO_GO_AFTER_R4_STAGE4_VARIANTS"
    branch_rows = [
        {
            "branch": "HS-HG",
            "amended_status": hs_hg_status,
            "reason": "Stage3-R4 reopened HS-HG, but all Stage4-R4 HS-HG variants failed the geometry gate.",
            "ready_for_stage4_after_r4": True,
            "stage4_variant_count": len(stage4_rows),
            "stage4_geometry_pass_count": sum(1 for row in stage4_rows if row["pilot_geometry_gate_pass"]),
        }
    ]
    for branch in stage3.get("blocked_branches", []):
        branch_rows.append(
            {
                "branch": branch,
                "amended_status": "STRUCTURAL_BLOCKED_AFTER_R4_RECHECK",
                "reason": "Stage3-R4 still lacked operation-specific candidate/reliability or required partner branch.",
                "ready_for_stage4_after_r4": False,
                "stage4_variant_count": 0,
                "stage4_geometry_pass_count": 0,
            }
        )

    summary = {
        "schema": "acl2_v118tf_postfinal_r4_amendment_summary_v1",
        "previous_final_artifact": rel(RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json"),
        "stage3_r4_artifact": rel(STAGE3_R4),
        "decision": "V118_R4_HS_HG_STAGE4_NO_GO_REST_STRUCTURAL_BLOCKERS",
        "global_goal_achieved": False,
        "any_stage4_r4_geometry_pass": any_geometry_pass,
        "hs_hg_amended_status": hs_hg_status,
        "stage4_r4_variant_rows": stage4_rows,
        "branch_status_rows": branch_rows,
        "outputs": {
            "summary": rel(OUT_JSON),
            "branch_status_csv": rel(OUT_CSV),
            "report": rel(OUT_REPORT),
        },
        "boundary": "This amendment supersedes only the stale HS-HG structural-blocked wording from the earlier final closure. It does not claim v118 success; semantic-causality controls were not triggered because no HS-HG Stage4-R4 variant passed the geometry gate.",
    }
    write_json(OUT_JSON, summary)
    write_csv(OUT_CSV, branch_rows)

    lines = [
        "# ACL2 v118-TF Post-Final R4 Amendment",
        "",
        f"- decision: `{summary['decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- any_stage4_r4_geometry_pass: `{summary['any_stage4_r4_geometry_pass']}`",
        f"- hs_hg_amended_status: `{summary['hs_hg_amended_status']}`",
        "",
        "## Stage4-R4 HS-HG Variants",
        "",
        "| variant | control | gate pass | median full ATE rel | max harm | rolling p90 rel | segment scale rel | segment scale ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage4_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['control']}` | {row['pilot_geometry_gate_pass']} | "
            f"{row['median_full_ATE_rel_improvement']} | {row['max_full_ATE_harm_rel']} | "
            f"{row['median_rolling_p90_rel_improvement']} | {row['median_segment_scale_rel_improvement']} | "
            f"{row['segment_scale_not_worse_all']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        summary["boundary"],
    ]
    OUT_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R4",
            "surface_or_branch": "HS-HG",
            "status": summary["decision"],
            "artifact": rel(OUT_JSON),
            "notes": "HS-HG reopened by Stage3-R4 but closed as Stage4-R4 geometry No-Go after four variants; global v118 goal not achieved",
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
