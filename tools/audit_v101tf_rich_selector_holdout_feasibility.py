#!/usr/bin/env python3
"""Audit fresh-holdout feasibility for the v101 rich merge/gauge selector.

This script keeps the distinction between three evidence levels:

* measured labelled rows already used by the retrospective rich-selector screen;
* prior-measured but unlabelled support rows that were not in that screen;
* truly fresh unmeasured rows, which need a native-measurement stage before the
  fixed policy can even be evaluated.

It does not authorize runtime action.  It only writes reviewer-facing evidence
and a concrete next-stage command/spec.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
V94 = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DRY = ROOT / "outcomeD_merge_gauge_fresh_unlabelled_stage1_native_dryrun"
STAGE1 = ROOT / "outcomeD_merge_gauge_fresh_unlabelled_stage1_native_probe"

VARIANT_NATIVE = "native_actual"
FRESH_STAGE1_VARIANTS = "native_actual"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_unique_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    if line not in lines:
        lines.append(line)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def upsert_section(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    section = f"\n\n## {heading}\n\n{body.strip()}\n"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = f"\n## {heading}\n"
    if marker in text:
        prefix, rest = text.split(marker, 1)
        next_heading = rest.find("\n## ")
        if next_heading >= 0:
            text = prefix.rstrip() + section + rest[next_heading:]
        else:
            text = prefix.rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.lstrip() + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if math.isfinite(out) else default


def seq_str(value: Any) -> str:
    try:
        return f"{int(float(value)):02d}"
    except Exception:  # noqa: BLE001
        text = str(value)
        return text.zfill(2) if text.isdigit() else text


def compare(value: float, direction: str, threshold: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        return False
    if direction == "le":
        return value <= threshold
    if direction == "ge":
        return value >= threshold
    if direction == "lt":
        return value < threshold
    if direction == "gt":
        return value > threshold
    raise ValueError(f"unsupported direction: {direction}")


def median(values: list[float]) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def selected_text(pairs: list[str]) -> str:
    return ",".join(sorted(pairs))


def pick_best_policy(rows: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, str]:
    best_policy_id = str(summary.get("best_policy_id") or "")
    for row in rows:
        if row.get("policy_id") == best_policy_id:
            return row
    return rows[0] if rows else {}


def command_for(pairs: list[str], out_root: Path, *, dry_run: bool) -> str:
    args = [
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5",
        "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
        "tools/run_v94_merge_gauge_runtime_probe.py",
    ]
    if dry_run:
        args.append("--dry-run")
    args.extend(
        [
            "--out-root",
            str(out_root),
            "--target-pairs",
            ",".join(pairs),
            "--variants",
            FRESH_STAGE1_VARIANTS,
            "--gpus",
            "0,1,2,3,4,5",
        ]
    )
    return " ".join(args)


def label_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("case_label_offline_only", "")) for row in rows))


def failure_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("failure_type_primary", "")) for row in rows))


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)

    phase1 = read_rows(V94 / "phase1_boundary_failure_atlas/boundary_failure_rows.csv")
    phase5 = read_rows(V94 / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    phase6 = read_rows(V94 / "phase6_object_source_action_surface/action_surface_effect_rows.csv")
    candidate_rows = read_rows(FINAL / "merge_gauge_rich_selector_reentry_candidate_metrics.csv")
    rich_summary = read_json(FINAL / "merge_gauge_rich_selector_reentry_summary.json")
    best = pick_best_policy(candidate_rows, rich_summary)

    phase1_by_pair = {row["pair_id"]: row for row in phase1 if row.get("pair_id")}
    phase5_by_pair = {row["pair_id"]: row for row in phase5 if row.get("pair_id")}
    phase6_by_pair = {row["pair_id"]: row for row in phase6 if row.get("pair_id")}

    threshold1 = f(best.get("threshold1"))
    threshold2 = f(best.get("threshold2"))
    feature1 = best.get("feature1", "")
    feature2 = best.get("feature2", "")
    direction1 = best.get("direction1", "")
    direction2 = best.get("direction2", "")

    stage1_metric_rows = [
        row
        for row in read_rows(STAGE1 / "runtime_probe_metric_rows.csv")
        if row.get("variant") == VARIANT_NATIVE
    ]
    stage1_native_by_pair = {row.get("pair_id", ""): row for row in stage1_metric_rows if row.get("pair_id")}

    dry_manifest = read_json(DRY / "runtime_probe_manifest.json")
    stage1_manifest = read_json(STAGE1 / "runtime_probe_manifest.json")

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(phase1_by_pair):
        p1 = phase1_by_pair[pair_id]
        p5 = phase5_by_pair.get(pair_id, {})
        p6 = phase6_by_pair.get(pair_id, {})
        stage1_native = stage1_native_by_pair.get(pair_id, {})

        if pair_id in phase6_by_pair:
            evidence_role = "rich_screen_measured_labelled_or_control"
            native_source = "phase6_action_surface"
            native_value = f(p6.get("native_boundary_update_norm"))
        elif pair_id in phase5_by_pair:
            evidence_role = "prior_measured_not_rich_screen_unlabelled_support"
            native_source = "phase5_carrier_alignment"
            native_value = f(p5.get("native_boundary_update_norm"))
        elif pair_id in stage1_native_by_pair:
            evidence_role = "fresh_stage1_native_measured_unlabelled_support"
            native_source = "v101_fresh_stage1_native_probe"
            native_value = f(stage1_native.get("boundary_update_norm"))
        else:
            evidence_role = "fresh_unmeasured_unlabelled_support"
            native_source = "missing"
            native_value = math.nan

        feature2_value = f(p1.get("adjacent_log_scale_jump_offline"))
        f1_pass = compare(native_value, direction1, threshold1)
        f2_pass = compare(feature2_value, direction2, threshold2)
        full_evaluable = math.isfinite(native_value) and math.isfinite(feature2_value)
        selected = full_evaluable and f1_pass and f2_pass
        labelled_for_gate = str(p1.get("case_label_offline_only")) in {"bad", "good"}

        blockers: list[str] = []
        if not labelled_for_gate:
            blockers.append("unlabelled_support_no_bad_good_gate")
        if not math.isfinite(native_value):
            blockers.append("missing_native_boundary_update_norm")
        if pair_id in phase6_by_pair:
            blockers.append("already_used_by_rich_screen")
        elif pair_id in phase5_by_pair:
            blockers.append("prior_measured_not_fresh")

        rows.append(
            {
                "pair_id": pair_id,
                "seq": seq_str(p1.get("seq")),
                "case_label_offline_only": p1.get("case_label_offline_only", ""),
                "failure_type_primary": p1.get("failure_type_primary", ""),
                "evidence_role": evidence_role,
                "native_source": native_source,
                "labelled_for_bad_good_gate": labelled_for_gate,
                "fixed_policy_id": best.get("policy_id", ""),
                "feature1": feature1,
                "feature1_direction": direction1,
                "feature1_threshold": threshold1,
                "feature1_value": native_value if math.isfinite(native_value) else "",
                "feature1_pass": f1_pass,
                "feature2": feature2,
                "feature2_direction": direction2,
                "feature2_threshold": threshold2,
                "feature2_value": feature2_value if math.isfinite(feature2_value) else "",
                "feature2_prefilter_pass": f2_pass,
                "fixed_policy_full_evaluable": full_evaluable,
                "selected_by_fixed_policy": selected,
                "I_J_runtime_proxy_prior_if_available": p5.get("I_J_runtime_proxy", p6.get("I_J_runtime_proxy", "")),
                "W_good_runtime_proxy_prior_if_available": p5.get(
                    "W_good_runtime_proxy", p6.get("W_good_runtime_proxy", "")
                ),
                "stage1_native_returncode": stage1_native.get("returncode", ""),
                "stage1_native_curr_postmerge_sim3_rmse": stage1_native.get("curr_postmerge_sim3_rmse", ""),
                "stage1_native_curr_handoff_transfer_rmse": stage1_native.get(
                    "curr_handoff_transfer_rmse", ""
                ),
                "blocking_reasons": ";".join(blockers),
                "claim_level": "holdout_feasibility_no_action",
            }
        )

    phase1_rows = rows
    phase6_rows = [row for row in rows if row["evidence_role"] == "rich_screen_measured_labelled_or_control"]
    prior_not_screen_rows = [
        row
        for row in rows
        if row["evidence_role"] == "prior_measured_not_rich_screen_unlabelled_support"
    ]
    fresh_rows = [
        row
        for row in rows
        if row["evidence_role"] in {
            "fresh_unmeasured_unlabelled_support",
            "fresh_stage1_native_measured_unlabelled_support",
        }
    ]
    fresh_feature2_prefilter = [row for row in fresh_rows if row["feature2_prefilter_pass"]]
    prior_selected = [row for row in prior_not_screen_rows if row["selected_by_fixed_policy"]]
    fresh_selected = [row for row in fresh_rows if row["selected_by_fixed_policy"]]

    stage1_pairs = [row["pair_id"] for row in fresh_feature2_prefilter]
    prior_selected_ij = [f(row.get("I_J_runtime_proxy_prior_if_available")) for row in prior_selected]
    fresh_selected_native = [f(row.get("feature1_value")) for row in fresh_selected]

    summary = {
        "schema": "acl2_v101_rich_selector_holdout_feasibility_v1",
        "fixed_policy_id": best.get("policy_id", ""),
        "fixed_policy_threshold_source": "merge_gauge_rich_selector_reentry_candidate_metrics.csv best_policy_id",
        "feature1": feature1,
        "feature1_direction": direction1,
        "feature1_threshold": threshold1,
        "feature1_quantile": f(best.get("quantile1")),
        "feature2": feature2,
        "feature2_direction": direction2,
        "feature2_threshold": threshold2,
        "feature2_quantile": f(best.get("quantile2")),
        "phase1_pair_count": len(phase1_rows),
        "phase6_rich_screen_measured_pair_count": len(phase6_rows),
        "phase6_label_counts": label_counter(phase6_rows),
        "prior_measured_not_rich_screen_pair_count": len(prior_not_screen_rows),
        "prior_measured_not_rich_screen_label_counts": label_counter(prior_not_screen_rows),
        "prior_measured_not_rich_screen_failure_type_counts": failure_counter(prior_not_screen_rows),
        "prior_measured_not_rich_screen_fixed_policy_selected_count": len(prior_selected),
        "prior_measured_not_rich_screen_fixed_policy_selected_pairs": selected_text(
            [row["pair_id"] for row in prior_selected]
        ),
        "prior_measured_not_rich_screen_selected_median_I_J_runtime_proxy": median(prior_selected_ij),
        "fresh_unmeasured_or_stage1_pair_count": len(fresh_rows),
        "fresh_labelled_bad_good_holdout_pair_count": sum(1 for row in fresh_rows if row["labelled_for_bad_good_gate"]),
        "fresh_unlabelled_pair_count": sum(1 for row in fresh_rows if not row["labelled_for_bad_good_gate"]),
        "fresh_feature2_prefilter_pair_count": len(fresh_feature2_prefilter),
        "fresh_feature2_prefilter_pairs": selected_text(stage1_pairs),
        "fresh_stage1_native_dryrun_manifest_exists": bool(dry_manifest),
        "fresh_stage1_native_dryrun_target_count": dry_manifest.get("target_count", ""),
        "fresh_stage1_native_dryrun_job_count": dry_manifest.get("job_count", ""),
        "fresh_stage1_native_probe_manifest_exists": bool(stage1_manifest),
        "fresh_stage1_native_probe_completed_count": stage1_manifest.get("completed_count", ""),
        "fresh_stage1_native_probe_failed_count": stage1_manifest.get("failed_count", ""),
        "fresh_stage1_native_metric_row_count": len(stage1_metric_rows),
        "fresh_stage1_fixed_policy_evaluable_count": sum(
            1
            for row in fresh_rows
            if row["fixed_policy_full_evaluable"] and row["feature2_prefilter_pass"]
        ),
        "fresh_stage1_fixed_policy_selected_count": len(fresh_selected),
        "fresh_stage1_fixed_policy_selected_pairs": selected_text([row["pair_id"] for row in fresh_selected]),
        "fresh_stage1_selected_native_boundary_update_norm_median": median(fresh_selected_native),
        "predeclared_stage1_native_target_pairs": selected_text(stage1_pairs),
        "predeclared_stage1_native_dryrun_command": command_for(stage1_pairs, DRY, dry_run=True)
        if stage1_pairs
        else "",
        "predeclared_stage1_native_actual_command": command_for(stage1_pairs, STAGE1, dry_run=False)
        if stage1_pairs
        else "",
        "fresh_holdout_action_evaluable": False,
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "blocked_reason": (
            "No fresh labelled bad/good holdout exists in the available phase1-minus-measured universe. "
            "Fresh rows are unlabelled_support, so native measurement can only make the fixed policy "
            "evaluable; it cannot create a bad/good promotion gate."
        ),
        "claim": (
            "Holdout feasibility artifact only. Prior-measured support and fresh unlabelled native "
            "measurements do not authorize M4/runtime/full validation."
        ),
    }

    write_rows(FINAL / "rich_selector_holdout_feasibility_rows.csv", rows)
    write_json(FINAL / "rich_selector_holdout_feasibility_summary.json", summary)

    report = [
        "# Rich Selector Fresh-Holdout Feasibility",
        "",
        "This audit keeps the fixed rich-selector policy unchanged and checks whether the current v94/v101 artifact universe contains a legitimate fresh holdout.",
        "",
        "## Fixed Policy",
        "",
        f"- policy: `{summary['fixed_policy_id']}`",
        f"- `{feature1} {direction1} {threshold1}`",
        f"- `{feature2} {direction2} {threshold2}`",
        "",
        "## Evidence Universe",
        "",
        f"- phase1 pairs: `{summary['phase1_pair_count']}`",
        f"- already measured by rich/phase6 screen: `{summary['phase6_rich_screen_measured_pair_count']}`",
        f"- prior-measured but not rich-screen support: `{summary['prior_measured_not_rich_screen_pair_count']}`",
        f"- fresh unmeasured/stage1 support: `{summary['fresh_unmeasured_or_stage1_pair_count']}`",
        f"- fresh labelled bad/good holdout pairs: `{summary['fresh_labelled_bad_good_holdout_pair_count']}`",
        "",
        "## Prior-Measured Not-Screen Diagnostic",
        "",
        f"- selected by fixed policy: `{summary['prior_measured_not_rich_screen_fixed_policy_selected_count']}`",
        f"- selected pairs: `{summary['prior_measured_not_rich_screen_fixed_policy_selected_pairs']}`",
        f"- selected median I/J proxy: `{summary['prior_measured_not_rich_screen_selected_median_I_J_runtime_proxy']}`",
        "",
        "These rows are not a fresh holdout because they were already measured in phase5, and they are all `unlabelled_support`.",
        "",
        "## Fresh Stage1 Native Measurement",
        "",
        f"- feature2-prefilter fresh pairs: `{summary['fresh_feature2_prefilter_pair_count']}`",
        f"- feature2-prefilter pairs: `{summary['fresh_feature2_prefilter_pairs']}`",
        f"- dry-run target/job count: `{summary['fresh_stage1_native_dryrun_target_count']}` / `{summary['fresh_stage1_native_dryrun_job_count']}`",
        f"- actual stage1 completed/failed count: `{summary['fresh_stage1_native_probe_completed_count']}` / `{summary['fresh_stage1_native_probe_failed_count']}`",
        f"- actual stage1 metric rows: `{summary['fresh_stage1_native_metric_row_count']}`",
        f"- fixed-policy selected fresh rows after stage1: `{summary['fresh_stage1_fixed_policy_selected_count']}`",
        f"- selected fresh pairs: `{summary['fresh_stage1_fixed_policy_selected_pairs']}`",
        "",
        "## Conclusion",
        "",
        summary["blocked_reason"],
        "Therefore this branch remains diagnostic-only and does not authorize M4/runtime/full validation.",
    ]
    write_text(FINAL / "rich_selector_holdout_feasibility_report.md", "\n".join(report))

    write_text(
        STAGE1 / "failure_report.md",
        "Fresh stage1 native probe, if executed, only measures native boundary-update features for unlabelled support rows. It does not provide labelled bad/good holdout evidence and cannot pass a promotion gate by itself.",
    )
    write_text(
        STAGE1 / "what_would_have_to_be_true_to_pass.md",
        "A pass would require fresh labelled bad/good target/control rows, fixed-policy selection after native measurement, measured candidate/control variants, and a control-beating gate. The current rows are unlabelled_support.",
    )
    write_text(
        STAGE1 / "control_gap_report.md",
        "Control gap: no fresh labelled bad/good holdout exists, so selected/unselected rows cannot be scored as bad recall, good FPR, or safe-good protection.",
    )
    write_text(
        STAGE1 / "next_attempt_recommendation.md",
        "Acquire labelled clean handoff target/safe-good holdout rows before spending more runtime budget on promotion. Native-only stage1 can be reused as diagnostic premeasurement.",
    )
    write_rows(
        STAGE1 / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "rich_selector_fresh_stage1_native",
                "row_kind": "no_fresh_labelled_holdout",
                "fresh_labelled_bad_good_holdout_pair_count": summary[
                    "fresh_labelled_bad_good_holdout_pair_count"
                ],
                "fresh_stage1_fixed_policy_selected_count": summary[
                    "fresh_stage1_fixed_policy_selected_count"
                ],
                "claim_level": "diagnostic_no_action",
            }
        ],
    )
    write_text(
        DRY / "failure_report.md",
        "Fresh stage1 native dry-run manifest only; no measured outcome and no action authorization.",
    )
    write_text(
        DRY / "what_would_have_to_be_true_to_pass.md",
        "The actual stage1 native probe must complete first, then a labelled holdout and measured candidate/control stage are required before promotion.",
    )
    write_text(
        DRY / "control_gap_report.md",
        "Dry-run has no measured controls and no labels.",
    )
    write_text(
        DRY / "next_attempt_recommendation.md",
        "Use the dry-run only to reproduce command construction. Do not treat it as data.",
    )
    write_rows(
        DRY / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "rich_selector_fresh_stage1_native_dryrun",
                "row_kind": "dry_run_no_metrics",
                "dry_run_target_count": summary["fresh_stage1_native_dryrun_target_count"],
                "dry_run_job_count": summary["fresh_stage1_native_dryrun_job_count"],
                "claim_level": "command_feasibility_only",
            }
        ],
    )

    recommendation = (
        "Fresh-holdout feasibility audit found no fresh labelled bad/good holdout: "
        f"phase1 pairs={summary['phase1_pair_count']}, rich-screen measured pairs="
        f"{summary['phase6_rich_screen_measured_pair_count']}, prior-measured not-screen support="
        f"{summary['prior_measured_not_rich_screen_pair_count']}, fresh support="
        f"{summary['fresh_unmeasured_or_stage1_pair_count']}, fresh labelled holdout="
        f"{summary['fresh_labelled_bad_good_holdout_pair_count']}. "
        "The fixed policy can only be tested diagnostically on unlabelled support until a labelled "
        "clean handoff/safe-good holdout is acquired."
    )
    upsert_section(FINAL / "next_attempt_recommendation.md", "Rich Selector Fresh-Holdout Feasibility", recommendation)
    append_unique_line(
        FINAL / "remaining_blockers.md",
        "- Rich-selector fresh-holdout blocker: available fresh rows are unlabelled_support, so no fresh bad/good promotion gate exists.",
    )
    append_unique_line(
        FINAL / "failure_report.md",
        "- Rich-selector fresh-holdout feasibility: no fresh labelled bad/good holdout exists; stage1 native measurements remain diagnostic-only.",
    )
    append_unique_line(
        FINAL / "control_gap_report.md",
        "- Rich-selector fresh-holdout control gap: no labelled bad/good holdout rows outside prior measured screen.",
    )

    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
