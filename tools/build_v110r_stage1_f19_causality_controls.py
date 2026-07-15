#!/usr/bin/env python3
"""Build ACL2 v110R Stage1 F19 causality-control evidence.

This stage intentionally reuses completed v109 F19 same-count keyframe-control
evidence as an early fail-forward gate. It does not mark the larger v110
extended control family as complete unless those artifacts exist.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
OUT = RESULT_ROOT / "stage1_f19_causality_controls"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
V109 = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
V109_F19 = V109 / "stage2_role_specific_safety_candidates"
V109_F19_CONTROLS = V109 / "stage2_f19_keyframe_controls"

F19_POLICY = "F19_dynamic_or_special_admitted_high_risk_else_weak_context"
CONTROL_TOLERANCE = 0.005


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_harm(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max([max(0.0, -v) for v in vals], default=float("nan"))


def required_artifacts() -> list[dict[str, Any]]:
    artifacts = [
        ("stage0_summary", STAGE0 / "stage0_summary.json"),
        ("v109_f19_summary", V109_F19 / "role_specific_safety_candidate_summary.json"),
        ("v109_f19_metric_rows", V109_F19 / "full_metric_rows.csv"),
        ("v109_f19_control_summary", V109_F19_CONTROLS / "f19_keyframe_control_summary.json"),
        ("v109_f19_control_metric_rows", V109_F19_CONTROLS / "full_metric_rows.csv"),
        ("v109_f19_control_schedule_rows", V109_F19_CONTROLS / "action_config_rows.csv"),
    ]
    return [
        {
            "schema": "acl2_v110r_stage1_required_artifact_row_v1",
            "artifact_id": artifact_id,
            "path": rel(path),
            "exists": path.exists(),
            "row_count": len(read_csv(path)) if path.exists() and path.suffix == ".csv" else "",
        }
        for artifact_id, path in artifacts
    ]


def f19_rows() -> list[dict[str, str]]:
    return [row for row in read_csv(V109_F19 / "full_metric_rows.csv") if row.get("policy_id") == F19_POLICY]


def control_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V109_F19_CONTROLS / "full_metric_rows.csv"):
        out = {
            "schema": "acl2_v110r_stage1_f19_control_metric_row_v1",
            "evidence_scope": "v109_F19_plus_exact_count_keyframe_random_controls",
            "source": rel(V109_F19_CONTROLS / "full_metric_rows.csv"),
        }
        out.update(row)
        rows.append(out)
    return rows


def control_schedule_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V109_F19_CONTROLS / "action_config_rows.csv"):
        out = {
            "schema": "acl2_v110r_stage1_f19_control_schedule_row_v1",
            "evidence_scope": "v109_F19_plus_exact_count_keyframe_random_controls",
            "source": rel(V109_F19_CONTROLS / "action_config_rows.csv"),
        }
        out.update(row)
        rows.append(out)
    return rows


def per_sequence_rank_rows(f19_metric_rows: list[dict[str, str]], control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    f19_by_seq = {row.get("seq", ""): row for row in f19_metric_rows}
    controls_by_seq: dict[str, list[dict[str, Any]]] = {}
    for row in control_rows:
        controls_by_seq.setdefault(str(row.get("seq", "")), []).append(row)

    out: list[dict[str, Any]] = []
    for seq in sorted(f19_by_seq):
        f19 = f19_by_seq[seq]
        f19_rel = fnum(f19.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        seq_controls = sorted(
            controls_by_seq.get(seq, []),
            key=lambda row: fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline")),
            reverse=True,
        )
        for rank, row in enumerate(seq_controls, start=1):
            control_rel = fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
            gap = control_rel - f19_rel if math.isfinite(control_rel) and math.isfinite(f19_rel) else float("nan")
            out.append(
                {
                    "schema": "acl2_v110r_stage1_per_sequence_control_rank_row_v1",
                    "seq": seq,
                    "rank_by_control_full_rel": rank,
                    "policy_id": row.get("policy_id", ""),
                    "policy_family": row.get("policy_family", ""),
                    "control_full_ATE_sim3": row.get("full_ATE_sim3", ""),
                    "control_full_rel_improvement": control_rel,
                    "f19_full_ATE_sim3": f19.get("full_ATE_sim3", ""),
                    "f19_full_rel_improvement": f19_rel,
                    "gap_control_minus_f19_rel": gap,
                    "within_0p005_of_f19": math.isfinite(gap) and abs(gap) <= CONTROL_TOLERANCE,
                    "control_matches_f19_for_causality_gate": math.isfinite(gap) and gap >= -CONTROL_TOLERANCE,
                }
            )
    return out


def report_text(summary: dict[str, Any], rank_rows: list[dict[str, Any]]) -> str:
    best_same_seq_rows = summary.get("best_same_seq_rows", [])
    lines = [
        "# ACL2 v110R Stage1 F19 Control Distribution",
        "",
        "This artifact freezes completed v109 F19 same-count keyframe-control evidence for v110R Stage1.",
        "It is an early fail-forward gate, not a newly completed extended-control sweep.",
        "",
        "```text",
        f"stage1_pass={summary['stage1_pass']}",
        f"f19_semantic_causality_pass={summary['f19_semantic_causality_pass']}",
        f"taxonomy={summary['taxonomy']}",
        f"blocker={summary['blocker']}",
        f"f19_classification={summary['f19_classification']}",
        f"continue_to_stage2={summary['continue_to_stage2']}",
        f"stage1_extended_control_set_complete={summary['stage1_extended_control_set_complete']}",
        "```",
        "",
        "## F19 Frozen Geometry",
        "",
        "```text",
        f"median_full_rel={summary['f19_median_full_rel_improvement']}",
        f"mean_full_rel={summary['f19_mean_full_rel_improvement']}",
        f"improved_sequence_count={summary['f19_improved_sequence_count']}",
        f"max_harm={summary['f19_max_harm']}",
        "```",
        "",
        "## Best Same-Sequence Controls",
        "",
        "```text",
    ]
    for row in best_same_seq_rows:
        lines.append(
            "seq{seq} best_control={policy_id} control_rel={control_rel} "
            "f19_rel={f19_rel} gap_vs_f19={gap_vs_f19_rel}".format(**row)
        )
    lines += [
        "```",
        "",
        "## Interpretation",
        "",
        "Exact-selected-count random keyframe controls match F19 within the registered 0.005 tolerance on two sequences.",
        "Therefore v110R must keep F19 as a strong keyframe/cache schedule baseline rather than claim semantic content causality.",
        "The plan can still continue to A/B/E/F full-ATE candidate search because Stage0 permits those runtime surfaces.",
        "",
        "## Per-Sequence Ranked Controls",
        "",
        "| seq | rank | policy | control_rel | f19_rel | gap_control_minus_f19 | matches_gate |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rank_rows:
        lines.append(
            "| {seq} | {rank_by_control_full_rel} | {policy_id} | {control_full_rel_improvement} | "
            "{f19_full_rel_improvement} | {gap_control_minus_f19_rel} | {control_matches_f19_for_causality_gate} |".format(**row)
        )
    return "\n".join(lines)


def decision_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ACL2 v110R F19 Schedule Effect Dominates",
            "",
            "Decision: do not claim F19 semantic causality in v110R Stage1.",
            "",
            "```text",
            f"taxonomy={summary['taxonomy']}",
            f"blocker={summary['blocker']}",
            f"f19_semantic_causality_pass={summary['f19_semantic_causality_pass']}",
            f"f19_classification={summary['f19_classification']}",
            f"continue_to_stage2={summary['continue_to_stage2']}",
            "```",
            "",
            "F19 remains a real full-trajectory geometry lead, but the completed exact-count keyframe controls are sufficient to prevent a semantic-content claim.",
            "The next v110R work should proceed to A/B/E/F candidate generation and full KITTI 00/02 pilot runs.",
        ]
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_rows = required_artifacts()
    missing = [row["path"] for row in artifact_rows if not row["exists"]]

    f19_metric_rows = f19_rows()
    control_rows = control_metric_rows()
    schedule_rows = control_schedule_rows()
    rank_rows = per_sequence_rank_rows(f19_metric_rows, control_rows)
    control_summary = read_json(V109_F19_CONTROLS / "f19_keyframe_control_summary.json")

    f19_rels = [fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline")) for row in f19_metric_rows]
    f19_improved = sum(1 for val in f19_rels if math.isfinite(val) and val > 0)
    matching_gate_rows = [row for row in rank_rows if row["rank_by_control_full_rel"] == 1 and row["control_matches_f19_for_causality_gate"]]

    metric_complete = (
        not missing
        and len(f19_metric_rows) == 4
        and len(control_rows) == 12
        and bool(control_summary.get("metric_complete"))
        and bool(control_summary.get("all_action_fidelity"))
    )
    semantic_causality_pass = (
        metric_complete
        and bool(control_summary.get("f19_keyframe_control_supports_f19_causality"))
        and len(matching_gate_rows) == 0
    )

    summary: dict[str, Any] = {
        "schema": "acl2_v110r_stage1_f19_causality_controls_summary_v1",
        "stage1_pass": bool(semantic_causality_pass),
        "taxonomy": "F19_SCHEDULE_EFFECT_DOMINATES_EARLY_FAIL_FORWARD",
        "blocker": control_summary.get("blocker", "same_count_keyframe_control_matches_f19_on_multiple_sequences"),
        "evidence_scope": "v109_F19_plus_exact_count_keyframe_random_controls",
        "stage1_extended_control_set_complete": False,
        "stage1_extended_control_set_note": "Only completed v109 exact-selected-count keyframe random controls are frozen here; no new extended controls were run in this script.",
        "continue_to_stage2": metric_complete,
        "f19_semantic_causality_pass": bool(semantic_causality_pass),
        "f19_classification": "STRONG_KEYFRAME_CACHE_SCHEDULE_BASELINE",
        "metric_complete": metric_complete,
        "missing_required_artifacts": missing,
        "f19_policy_id": F19_POLICY,
        "f19_metric_row_count": len(f19_metric_rows),
        "control_metric_row_count": len(control_rows),
        "control_schedule_row_count": len(schedule_rows),
        "control_tolerance": CONTROL_TOLERANCE,
        "best_same_seq_control_match_f19_count": int(control_summary.get("best_same_seq_control_match_f19_count", len(matching_gate_rows))),
        "best_same_seq_rows": control_summary.get("best_same_seq_rows", []),
        "strongest_control_policy_id": control_summary.get("strongest_control_policy_id", ""),
        "strongest_control_median_full_rel_improvement": control_summary.get("strongest_control_median_full_rel_improvement", ""),
        "f19_median_full_rel_improvement": median(f19_rels),
        "f19_mean_full_rel_improvement": mean(f19_rels),
        "f19_improved_sequence_count": f19_improved,
        "f19_max_harm": max_harm(f19_rels),
        "outputs": {
            "required_artifact_rows": OUT / "required_artifact_rows.csv",
            "control_metric_rows": OUT / "control_metric_rows.csv",
            "control_schedule_rows": OUT / "control_schedule_rows.csv",
            "per_sequence_control_rank": OUT / "per_sequence_control_rank.csv",
            "f19_vs_control_distribution": OUT / "f19_vs_control_distribution.md",
            "decision": OUT / "F19_SCHEDULE_EFFECT_DOMINATES.md",
            "summary": OUT / "stage1_summary.json",
        },
    }

    write_csv(OUT / "required_artifact_rows.csv", artifact_rows)
    write_csv(OUT / "control_metric_rows.csv", control_rows)
    write_csv(OUT / "control_schedule_rows.csv", schedule_rows)
    write_csv(OUT / "per_sequence_control_rank.csv", rank_rows)
    write_text(OUT / "f19_vs_control_distribution.md", report_text(summary, rank_rows))
    write_text(OUT / "F19_SCHEDULE_EFFECT_DOMINATES.md", decision_text(summary))
    write_json(OUT / "stage1_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if metric_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
