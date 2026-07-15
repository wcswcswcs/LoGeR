#!/usr/bin/env python3
"""Build ACL2 v106 Stage3 memory role disambiguation diagnostics."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
STAGE1 = V106 / "stage1_selected_evidence_materialization"
STAGE2 = V106 / "stage2_moge_metric_verifier"
OUT = V106 / "stage3_memory_role_disambiguation"

VARIANTS = [
    "geometry_only",
    "semantic_only",
    "semantic_plus_geometry",
    "semantic_plus_geometry_plus_proxy",
]

RULE_PROFILE = "r4_context_semantic_before_metric_reject_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def quantiles(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "semantic_trust",
        "boundary_risk",
        "metric_consistency_score",
        "scale_observability_score",
        "local_window_support_score",
        "anchor_context_consistency_score",
        "lingbot_pose_residual_local",
        "trajectory_residual",
        "source_frame_age",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([as_float(row, key, 0.0) for row in rows], dtype=np.float64)
        out[f"{key}_q25"] = float(np.quantile(vals, 0.25))
        out[f"{key}_q50"] = float(np.quantile(vals, 0.50))
        out[f"{key}_q75"] = float(np.quantile(vals, 0.75))
    return out


def dynamic_or_boundary(row: dict[str, Any], q: dict[str, float]) -> bool:
    return (
        row.get("semantic_role") in {"dynamic_object", "object_boundary"}
        or as_float(row, "boundary_risk") >= q["boundary_risk_q75"]
    )


def update_role(role: str) -> bool:
    return role in {"SCALE_REFERENCE_EVIDENCE", "TRAJECTORY_MEMORY_EVIDENCE"}


def reference_update_role(role: str) -> bool:
    return role == "SCALE_REFERENCE_EVIDENCE"


def classify(row: dict[str, Any], q: dict[str, float], variant: str) -> tuple[str, str, bool]:
    semantic_role = row.get("semantic_role", "unknown")
    context_path = row.get("context_path", "")
    local_support = as_float(row, "local_window_support_score")
    anchor_score = as_float(row, "anchor_context_consistency_score")
    semantic_trust = as_float(row, "semantic_trust")
    boundary = as_float(row, "boundary_risk")
    metric = as_float(row, "metric_consistency_score")
    scale_obs = as_float(row, "scale_observability_score")
    residual = as_float(row, "lingbot_pose_residual_local")
    traj_residual = as_float(row, "trajectory_residual")
    source_age = abs(as_float(row, "source_frame_age"))
    uses_sem = variant in {"semantic_only", "semantic_plus_geometry", "semantic_plus_geometry_plus_proxy"}
    uses_geom = variant in {"geometry_only", "semantic_plus_geometry", "semantic_plus_geometry_plus_proxy"}
    uses_proxy = variant == "semantic_plus_geometry_plus_proxy"
    context_semantic = semantic_role in {"vegetation_or_weak_context", "road_or_ground", "sky_or_lowobs"}

    # R5 reject unreliable.
    if uses_sem and (
        semantic_role in {"dynamic_object", "object_boundary"}
        or (semantic_trust <= q["semantic_trust_q25"] and boundary >= q["boundary_risk_q50"])
    ):
        return "REJECT_UNRELIABLE", "R5_semantic_unreliable", False
    if uses_geom and residual >= q["lingbot_pose_residual_local_q75"]:
        return "REJECT_UNRELIABLE", "R5_high_pose_residual", False
    if uses_sem and uses_proxy and context_semantic:
        if local_support >= q["local_window_support_score_q50"] or context_path == "local_window_context":
            return "LOCAL_REGISTRATION_EVIDENCE", "R2_context_semantic_local_before_metric_reject", False
        return "CONTEXT_ONLY", "R4_context_semantic_before_metric_reject", False
    if uses_proxy and (boundary >= q["boundary_risk_q75"] or metric <= q["metric_consistency_score_q25"]):
        return "REJECT_UNRELIABLE", "R5_metric_boundary_or_metric_mismatch", False

    # R1 scale reference evidence.
    if uses_sem and uses_proxy:
        if (
            semantic_role == "stable_structure"
            and semantic_trust >= q["semantic_trust_q50"]
            and boundary <= q["boundary_risk_q50"]
            and metric >= q["metric_consistency_score_q50"]
            and scale_obs >= q["scale_observability_score_q50"]
            and context_path == "scale_reference_context"
        ):
            return "SCALE_REFERENCE_EVIDENCE", "R1_scale_reference_metric", False
    elif uses_sem and uses_geom:
        if (
            semantic_role == "stable_structure"
            and semantic_trust >= q["semantic_trust_q50"]
            and boundary <= q["boundary_risk_q50"]
            and anchor_score >= q["anchor_context_consistency_score_q50"]
            and context_path == "scale_reference_context"
        ):
            return "SCALE_REFERENCE_EVIDENCE", "R1_scale_reference_semgeom", False
    elif variant == "geometry_only":
        if anchor_score >= q["anchor_context_consistency_score_q75"] and residual <= q["lingbot_pose_residual_local_q50"]:
            return "SCALE_REFERENCE_EVIDENCE", "R1_geometry_anchor_reference", False
    elif variant == "semantic_only":
        if semantic_role == "stable_structure" and semantic_trust >= q["semantic_trust_q50"] and boundary <= q["boundary_risk_q50"]:
            return "SCALE_REFERENCE_EVIDENCE", "R1_semantic_stable_reference", False

    # R2 local registration evidence.
    if uses_geom and local_support >= q["local_window_support_score_q50"]:
        if context_path == "local_window_context" or anchor_score < q["anchor_context_consistency_score_q50"]:
            return "LOCAL_REGISTRATION_EVIDENCE", "R2_local_window_geometry", False
        if uses_sem and semantic_role in {"road_or_ground", "vegetation_or_weak_context"}:
            return "LOCAL_REGISTRATION_EVIDENCE", "R2_semantic_local_context", False
        if uses_proxy and scale_obs < q["scale_observability_score_q50"]:
            return "LOCAL_REGISTRATION_EVIDENCE", "R2_low_scale_observability", False

    # R3 trajectory memory candidate. Without true token age this stays diagnostic-only.
    if uses_sem and uses_proxy:
        if (
            semantic_role in {"stable_structure", "road_or_ground"}
            and metric >= q["metric_consistency_score_q50"]
            and scale_obs >= q["scale_observability_score_q50"]
            and traj_residual <= q["trajectory_residual_q50"]
            and source_age <= q["source_frame_age_q75"]
            and not dynamic_or_boundary(row, q)
        ):
            return "CONTEXT_ONLY", "R3_trajectory_candidate_metric_demoted_to_context", True

    # R4 context-only.
    if uses_sem and semantic_role in {"vegetation_or_weak_context", "road_or_ground", "sky_or_lowobs"}:
        return "CONTEXT_ONLY", "R4_semantic_context", False
    if uses_proxy and metric >= q["metric_consistency_score_q50"] and scale_obs < q["scale_observability_score_q50"]:
        return "CONTEXT_ONLY", "R4_metric_ok_low_scale_obs", False
    if uses_geom and residual <= q["lingbot_pose_residual_local_q50"]:
        return "CONTEXT_ONLY", "R4_low_residual_context", False

    return "CONTEXT_ONLY", "default_context_only", False


def build_rows() -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    enriched = read_csv(STAGE1 / "selected_evidence_enriched_rows.csv")
    proxy = {row["token_group_id"]: row for row in read_csv(STAGE2 / "moge_verifier_rows.csv")}
    rows: list[dict[str, Any]] = []
    joined: list[dict[str, Any]] = []
    for row in enriched:
        merged = dict(row)
        merged.update(proxy.get(row["token_group_id"], {}))
        joined.append(merged)
    q = quantiles(joined)
    stage2_summary = json.loads((STAGE2 / "stage2_summary.json").read_text(encoding="utf-8"))
    for row in joined:
        for variant in VARIANTS:
            role, rule_id, traj_proxy = classify(row, q, variant)
            label = row["label_type"]
            wrongly_used_scale_reference = label == "bad_selected" and role == "SCALE_REFERENCE_EVIDENCE"
            rows.append(
                {
                    "schema": "acl2_v106tf_stage3_memory_role_row_v1",
                    "classifier_variant": variant,
                    "seq_id": row["seq_id"],
                    "frame_id": row["frame_id"],
                    "head_id": row["head_id"],
                    "token_group_id": row["token_group_id"],
                    "label_type": label,
                    "memory_role": role,
                    "rule_id": rule_id,
                    "trajectory_candidate_proxy": traj_proxy,
                    "wrongly_used_scale_reference": wrongly_used_scale_reference,
                    "update_path_allowed": update_role(role),
                    "semantic_role": row.get("semantic_role", ""),
                    "context_path": row.get("context_path", ""),
                    "token_type": row.get("token_type", ""),
                    "semantic_trust": row.get("semantic_trust", ""),
                    "boundary_risk": row.get("boundary_risk", ""),
                    "metric_consistency_score": row.get("metric_consistency_score", ""),
                    "boundary_mismatch_score": row.get("boundary_mismatch_score", ""),
                    "scale_observability_score": row.get("scale_observability_score", ""),
                    "lingbot_pose_residual_local": row.get("lingbot_pose_residual_local", ""),
                    "trajectory_residual": row.get("trajectory_residual", ""),
                    "local_window_support_score": row.get("local_window_support_score", ""),
                    "anchor_context_consistency_score": row.get("anchor_context_consistency_score", ""),
                    "baseline_L3": row.get("baseline_L3", ""),
                    "action_L3": row.get("action_L3", ""),
                    "bad_improvement": row.get("bad_improvement", ""),
                    "good_harm": row.get("good_harm", ""),
                    "proxy_only": stage2_summary.get("proxy_only", True),
                    "moge_available": stage2_summary.get("moge_available", False),
                }
            )
    return rows, q, stage2_summary


def variant_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["classifier_variant"] == variant]
        good = [row for row in subset if row["label_type"] == "good_selected"]
        bad = [row for row in subset if row["label_type"] == "bad_selected"]
        update_good = [row for row in good if row["update_path_allowed"]]
        bad_caught = [
            row for row in bad
            if row["memory_role"] == "REJECT_UNRELIABLE" or row["wrongly_used_scale_reference"]
        ]
        good_local_context = [
            row for row in good if row["memory_role"] in {"LOCAL_REGISTRATION_EVIDENCE", "CONTEXT_ONLY"}
        ]
        good_not_update = [row for row in good if not row["update_path_allowed"]]
        good_no_reference = [row for row in good if not reference_update_role(row["memory_role"])]
        good_fpr = len(update_good) / len(good) if good else 0.0
        good_ref_fpr = len([row for row in good if reference_update_role(row["memory_role"])]) / len(good) if good else 0.0
        bad_coverage = len(bad_caught) / len(bad) if bad else 0.0
        good_local_context_frac = len(good_local_context) / len(good) if good else 0.0
        good_not_update_frac = len(good_not_update) / len(good) if good else 0.0
        good_no_reference_frac = len(good_no_reference) / len(good) if good else 0.0
        ba = 0.5 * (bad_coverage + good_not_update_frac)
        out.append(
            {
                "schema": "acl2_v106tf_stage3_semantic_increment_row_v1",
                "classifier_variant": variant,
                "bad_rows": len(bad),
                "good_rows": len(good),
                "good_update_false_positive_rows": len(update_good),
                "good_false_positive_rate": good_fpr,
                "good_scale_reference_false_positive_rate": good_ref_fpr,
                "bad_reject_or_reference_error_rows": len(bad_caught),
                "bad_reject_or_reference_error_frac": bad_coverage,
                "good_local_or_context_rows": len(good_local_context),
                "good_local_or_context_frac": good_local_context_frac,
                "good_not_update_rows": len(good_not_update),
                "good_not_update_frac": good_not_update_frac,
                "good_no_reference_update_rows": len(good_no_reference),
                "good_no_reference_update_frac": good_no_reference_frac,
                "balanced_accuracy_proxy": ba,
            }
        )
    base = next(row for row in out if row["classifier_variant"] == "geometry_only")
    for row in out:
        row["delta_safe_vs_geometry"] = float(base["good_false_positive_rate"] - row["good_false_positive_rate"])
        row["ba_gain_vs_geometry"] = float(row["balanced_accuracy_proxy"] - base["balanced_accuracy_proxy"])
        denom = max(float(base["good_false_positive_rate"]), 1e-9)
        row["good_harm_risk_reduction_vs_geometry"] = float(
            (base["good_false_positive_rate"] - row["good_false_positive_rate"]) / denom
        )
    return out


def role_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], int] = Counter(
        (row["classifier_variant"], row["label_type"], row["memory_role"]) for row in rows
    )
    out = []
    for (variant, label, role), count in sorted(grouped.items()):
        out.append(
            {
                "schema": "acl2_v106tf_stage3_role_case_summary_v1",
                "classifier_variant": variant,
                "label_type": label,
                "memory_role": role,
                "count": count,
            }
        )
    return out


def confusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["classifier_variant"] == variant]
        for label in sorted({row["label_type"] for row in subset}):
            label_rows = [row for row in subset if row["label_type"] == label]
            total = len(label_rows)
            for role, count in Counter(row["memory_role"] for row in label_rows).most_common():
                out.append(
                    {
                        "schema": "acl2_v106tf_stage3_role_confusion_matrix_v1",
                        "classifier_variant": variant,
                        "label_type": label,
                        "memory_role": role,
                        "count": count,
                        "fraction": count / total if total else 0.0,
                    }
                )
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = [
        "seq_id",
        "frame_id",
        "head_id",
        "token_group_id",
        "label_type",
        "memory_role",
        "rule_id",
        "semantic_role",
        "context_path",
        "metric_consistency_score",
        "scale_observability_score",
        "lingbot_pose_residual_local",
        "boundary_risk",
    ]
    lines = ["|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        vals = [str(row.get(key, "")) for key in keys]
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines) + "\n"


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, q, stage2 = build_rows()
    sem_rows = variant_metrics(rows)
    role_rows = role_summary(rows)
    confusion_rows = confusion(rows)
    target = next(row for row in sem_rows if row["classifier_variant"] == "semantic_plus_geometry_plus_proxy")
    stage1 = json.loads((STAGE1 / "materialization_summary.json").read_text(encoding="utf-8"))
    selected_good_pass = float(target["good_local_or_context_frac"]) >= 0.50
    selected_bad_pass = float(target["bad_reject_or_reference_error_frac"]) >= 0.50
    semantic_increment_pass = (
        float(target["delta_safe_vs_geometry"]) >= 0.10
        or float(target["good_harm_risk_reduction_vs_geometry"]) >= 0.50
        or float(target["ba_gain_vs_geometry"]) >= 0.05
    )
    hard_negative_rescue_pass = float(target["good_no_reference_update_frac"]) >= 0.50
    stage3_pass = (
        stage1.get("selected_evidence_row_coverage", 0.0) >= 0.80
        and selected_good_pass
        and selected_bad_pass
        and semantic_increment_pass
        and hard_negative_rescue_pass
    )
    stage2_moge_action_allowed = (
        bool(stage2.get("stage4_moge_based_action_promotion_allowed", False))
        and bool(stage2.get("moge_available", False))
        and not bool(stage2.get("moge_proxy_or_missing", True))
    )
    stage4_action_allowed = bool(stage3_pass and stage2_moge_action_allowed)
    metric_source_for_gate = (
        f"{stage2.get('moge_mode', 'moge2')}:{stage2.get('moge_region_scope', '')}"
        if stage2_moge_action_allowed
        else "lingbot_self_consistency_proxy"
    )
    summary = {
        "schema": "acl2_v106tf_stage3_memory_role_disambiguation_summary_v2",
        "rule_profile": RULE_PROFILE,
        "selected_evidence_materialized_rows": stage1.get("materialized_rows", 0),
        "selected_evidence_materialized_fraction": stage1.get("selected_evidence_row_coverage", 0.0),
        "moge_proxy_or_missing": stage2.get("moge_proxy_or_missing", True),
        "moge_available": stage2.get("moge_available", False),
        "stage2_moge_pass": stage2.get("stage2_moge_pass", False),
        "stage2_moge_based_action_promotion_allowed": stage2.get("stage4_moge_based_action_promotion_allowed", False),
        "metric_source_for_gate": metric_source_for_gate,
        "classifier_for_gate": "semantic_plus_geometry_plus_proxy",
        "selected_good_local_or_context_frac": target["good_local_or_context_frac"],
        "selected_good_not_update_frac": target["good_not_update_frac"],
        "hard_negative_no_reference_update_frac": target["good_no_reference_update_frac"],
        "selected_bad_reject_or_reference_error_frac": target["bad_reject_or_reference_error_frac"],
        "good_update_false_positive_frac": target["good_false_positive_rate"],
        "scale_reference_good_false_positive_frac": target["good_scale_reference_false_positive_rate"],
        "delta_safe_vs_geometry": target["delta_safe_vs_geometry"],
        "good_harm_risk_reduction_vs_geometry": target["good_harm_risk_reduction_vs_geometry"],
        "ba_gain_vs_geometry": target["ba_gain_vs_geometry"],
        "selected_good_gate_pass": selected_good_pass,
        "selected_bad_gate_pass": selected_bad_pass,
        "semantic_increment_gate_pass": semantic_increment_pass,
        "hard_negative_rescue_gate_pass": hard_negative_rescue_pass,
        "stage3_disambiguation_pass": stage3_pass,
        "stage4_action_allowed_with_moge_rule": stage4_action_allowed,
        "quantiles": q,
        "outputs": {
            "memory_role_rows": (OUT / "memory_role_rows.csv").relative_to(ROOT).as_posix(),
            "role_case_summary": (OUT / "role_case_summary.csv").relative_to(ROOT).as_posix(),
            "role_confusion_matrix": (OUT / "role_confusion_matrix.csv").relative_to(ROOT).as_posix(),
            "semantic_increment_rows": (OUT / "semantic_increment_rows.csv").relative_to(ROOT).as_posix(),
            "selected_bad_good_disambiguation_report": (OUT / "selected_bad_good_disambiguation_report.md").relative_to(ROOT).as_posix(),
            "stage3_summary": (OUT / "stage3_summary.json").relative_to(ROOT).as_posix(),
        },
    }
    write_csv(OUT / "memory_role_rows.csv", rows)
    write_csv(OUT / "role_case_summary.csv", role_rows)
    write_csv(OUT / "role_confusion_matrix.csv", confusion_rows)
    write_csv(OUT / "semantic_increment_rows.csv", sem_rows)
    (OUT / "stage3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target_rows = [row for row in rows if row["classifier_variant"] == "semantic_plus_geometry_plus_proxy"]
    target_good = [row for row in target_rows if row["label_type"] == "good_selected"]
    target_bad = [row for row in target_rows if row["label_type"] == "bad_selected"]
    interpretation = (
        "- Stage2 is real MoGe-backed for this gate; LingBot baseline depth is compared to MoGe-2 maps on traced attention top-k key patch regions.\n"
        "- `lingbot_vs_moge_point_error` remains blank because current LingBot artifacts expose depth maps, not point maps.\n"
        if stage2_moge_action_allowed
        else
        "- The metric column is proxy-only or missing-MoGe; do not claim MoGe verifier success.\n"
        "- If Stage4 is attempted from this output, it must not be promoted as a MoGe-backed action.\n"
    )
    report = f"""# Stage3 Selected Bad/Good Disambiguation Report

Gate classifier: `semantic_plus_geometry_plus_proxy`

MoGe status:
- moge_available: `{stage2.get("moge_available", False)}`
- moge_proxy_or_missing: `{stage2.get("moge_proxy_or_missing", True)}`
- stage2_moge_pass: `{stage2.get("stage2_moge_pass", False)}`
- metric_source_for_gate: `{metric_source_for_gate}`
- Stage4 action with MoGe-based rule allowed: `{stage4_action_allowed}`

Gate metrics:
- selected_good_local_or_context_frac: `{target["good_local_or_context_frac"]}`
- selected_good_not_update_frac: `{target["good_not_update_frac"]}`
- hard_negative_no_reference_update_frac: `{target["good_no_reference_update_frac"]}`
- selected_bad_reject_or_reference_error_frac: `{target["bad_reject_or_reference_error_frac"]}`
- good_update_false_positive_frac: `{target["good_false_positive_rate"]}`
- scale_reference_good_false_positive_frac: `{target["good_scale_reference_false_positive_rate"]}`
- delta_safe_vs_geometry: `{target["delta_safe_vs_geometry"]}`
- good_harm_risk_reduction_vs_geometry: `{target["good_harm_risk_reduction_vs_geometry"]}`
- ba_gain_vs_geometry: `{target["ba_gain_vs_geometry"]}`
- stage3_disambiguation_pass: `{stage3_pass}`

Interpretation:
- This is rule-based diagnostic, not a learned classifier.
{interpretation}

Selected good rows:

{markdown_table(target_good)}

Selected bad rows:

{markdown_table(target_bad)}
"""
    (OUT / "selected_bad_good_disambiguation_report.md").write_text(report, encoding="utf-8")
    failure_path = OUT / "stage3_disambiguation_failure.md"
    if not stage3_pass:
        failure_path.write_text(report, encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    summary = build()
    return 0 if summary["stage3_disambiguation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
