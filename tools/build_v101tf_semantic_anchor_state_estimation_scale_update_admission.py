#!/usr/bin/env python3
"""Build ACL2 v101 semantic-anchor state/admission diagnostics.

The builder is deliberately conservative.  It can materialize ledgers and
proxy/diagnostic rows from v100 evidence, but it never promotes proxy support,
proxy observability, or oracle labels into runtime permission.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
V100_ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V99_ROOT = Path("results/acl2_v99tf_semantic_anchor_identity_lifecycle_multiroute_memory_control")
V98_ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")

CASE_ROWS = V100_ROOT / "trackD4_read_current_support_provider/label_l3_hygiene_provenance_rows.csv"
Q_ROWS = V100_ROOT / "trackQ_chunk_update_admission/rows.csv"
L2_ROWS = V100_ROOT / "trackL2_anchor_scale_observability/rows.csv"
S_ROWS = V100_ROOT / "trackS_same_space_latent_state/same_space_anchor_rows.csv"

EPS = 1.0e-9


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            clean = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def median(values: list[Any]) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    return statistics.median(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mad(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    if not vals:
        return math.nan
    med = statistics.median(vals)
    return statistics.median(abs(v - med) for v in vals)


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def clip01(value: Any) -> float:
    fv = f(value)
    if not math.isfinite(fv):
        return math.nan
    return max(0.0, min(1.0, fv))


def norm01(value: Any, values: list[Any], *, invert: bool = False) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    fv = f(value)
    if not vals or not math.isfinite(fv):
        return math.nan
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= EPS:
        out = 0.5
    else:
        out = (fv - lo) / (hi - lo)
    return 1.0 - out if invert else out


def norm01_with_bounds(value: Any, lo: float, hi: float, *, invert: bool = False) -> float:
    fv = f(value)
    if not all(math.isfinite(v) for v in [fv, lo, hi]):
        return math.nan
    if hi - lo <= EPS:
        out = 0.5
    else:
        out = (fv - lo) / (hi - lo)
    out = max(0.0, min(1.0, out))
    return 1.0 - out if invert else out


def split_cases(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id", "")): row for row in rows if row.get("case_id")}


def case_seq(case_id: str) -> str:
    return str(case_id).split("_")[0]


def selected_metrics(rows: list[dict[str, Any]], selected: set[str], positives: set[str], negatives: set[str]) -> dict[str, Any]:
    cases = sorted((positives | negatives) & {str(row.get("case_id", "")) for row in rows})
    pos = positives & set(cases)
    neg = negatives & set(cases)
    tp = selected & pos
    fp = selected & neg
    missed = pos - selected
    recall = len(tp) / len(pos) if pos else math.nan
    fpr = len(fp) / len(neg) if neg else math.nan
    seq_counts = Counter(case_seq(case) for case in tp)
    max_frac = max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan
    l3_by_case = {str(row.get("case_id", "")): f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows}
    corr = pearson([1.0 if case in selected else 0.0 for case in cases], [l3_by_case.get(case) for case in cases])
    return {
        "available_case_count": len(cases),
        "positive_case_count": len(pos),
        "negative_case_count": len(neg),
        "selected_case_count": len(selected & set(cases)),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ((recall + (1.0 - fpr)) / 2.0) if math.isfinite(recall) and math.isfinite(fpr) else math.nan,
        "corr_L3": corr,
        "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": max_frac,
        "selected_positive_sequence_counts": dict(seq_counts),
        "true_positive_cases": ";".join(sorted(tp)),
        "false_positive_cases": ";".join(sorted(fp)),
        "missed_positive_cases": ";".join(sorted(missed)),
    }


def same_count_margin(rows: list[dict[str, Any]], selected: set[str], positives: set[str], negatives: set[str]) -> float:
    cases = sorted((positives | negatives) & {str(row.get("case_id", "")) for row in rows})
    if not cases:
        return math.nan
    actual = selected_metrics(rows, selected, positives, negatives)
    actual_signal = f(actual.get("bad_recall"), 0.0) - f(actual.get("good_FPR"), 0.0)
    count = len(selected & set(cases))
    controls = []
    for shift in range(len(cases)):
        control = set(cases[shift: shift + count])
        if len(control) < count:
            control.update(cases[: count - len(control)])
        metric = selected_metrics(rows, control, positives, negatives)
        controls.append(f(metric.get("bad_recall"), 0.0) - f(metric.get("good_FPR"), 0.0))
    return actual_signal - median(controls)


def artifact_row(name: str, path: Path, claim_level: str, required: bool = True) -> dict[str, Any]:
    exists = path.is_file()
    rows = read_rows(path) if exists and path.suffix == ".csv" else []
    json_payload = read_json(path) if exists and path.suffix == ".json" else {}
    return {
        "artifact": name,
        "path": str(path),
        "required": required,
        "exists": exists,
        "row_count": len(rows) if rows else "",
        "json_schema": json_payload.get("schema", ""),
        "claim_level": claim_level,
    }


def write_track_common(out: Path, *, failure: str, what_would: str, next_attempt: str, control_gap: str = "") -> None:
    write_text(out / "failure_report.md", failure)
    write_text(out / "what_would_have_to_be_true_to_pass.md", what_would)
    write_text(out / "next_attempt_recommendation.md", next_attempt)
    write_text(out / "control_gap_report.md", control_gap or "No additional control gap report was materialized for this track.")


def stage0() -> dict[str, Any]:
    out = ROOT / "stage0_v101_evidence_ledger"
    artifacts = [
        artifact_row("v100_final_decision", V100_ROOT / "final_decision/final_decision.json", "fact_lock"),
        artifact_row("v100_trackS_summary", V100_ROOT / "trackS_same_space_latent_state/summary.json", "instrumentation_fact"),
        artifact_row("v100_trackS_same_space_rows", S_ROWS, "instrumentation_rows"),
        artifact_row("v100_trackQ_summary", V100_ROOT / "trackQ_chunk_update_admission/summary.json", "proxy_diagnostic"),
        artifact_row("v100_trackQ_rows", Q_ROWS, "proxy_diagnostic_rows"),
        artifact_row("v100_trackL2_summary", V100_ROOT / "trackL2_anchor_scale_observability/summary.json", "proxy_diagnostic"),
        artifact_row("v100_trackN2_summary", V100_ROOT / "trackN2_anchor_identity_graph/summary.json", "diagnostic_no_go"),
        artifact_row("v100_trackO2_summary", V100_ROOT / "trackO2_freshness_current_support/summary.json", "diagnostic_no_go"),
        artifact_row("v100_trackC4_summary", V100_ROOT / "trackC4_identity_latent_gauge_ruler/summary.json", "diagnostic_no_go"),
        artifact_row("v100_trackF4_summary", V100_ROOT / "trackF4_ttt_write_to_use_same_space/summary.json", "diagnostic_no_go"),
        artifact_row("v100_trackR_summary", V100_ROOT / "trackR_edge_head_control_audit/summary.json", "diagnostic_no_go"),
        artifact_row("v100_trackR2_summary", V100_ROOT / "trackR2_anchor_edge_identity_control_audit/summary.json", "diagnostic_no_go"),
        artifact_row("v100_label_l3_hygiene", V100_ROOT / "trackD4_read_current_support_provider/label_l3_hygiene_provenance_summary.json", "target_hygiene"),
        artifact_row("v100_label_l3_hygiene_rows", CASE_ROWS, "target_hygiene_rows"),
        artifact_row("v100_oracle_l3_feasibility", V100_ROOT / "trackD4_read_current_support_provider/oracle_l3_gate_feasibility_summary.json", "oracle_diagnostic_only"),
        artifact_row("v99_graph_case_rows", V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/graph_case_rows.csv", "upstream_metadata"),
        artifact_row("v98_stage1_case_universe", V98_ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv", "upstream_metadata"),
        artifact_row("v98_stage7e_summary", V98_ROOT / "stage7e_ttt_stable_anchor_id_hook/summary.json", "upstream_diagnostic"),
        artifact_row("v98_stage7f_summary", V98_ROOT / "stage7f_prev_ttt_anchor_gate_action_pilot/summary.json", "upstream_no_go"),
        artifact_row("v98_stage7h_summary", V98_ROOT / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json", "upstream_no_go"),
    ]
    missing = [row for row in artifacts if row["required"] and not row["exists"]]
    v100_final = read_json(V100_ROOT / "final_decision/final_decision.json")
    track_s = read_json(V100_ROOT / "trackS_same_space_latent_state/summary.json")
    track_q = read_json(V100_ROOT / "trackQ_chunk_update_admission/summary.json")
    hygiene = read_json(V100_ROOT / "trackD4_read_current_support_provider/label_l3_hygiene_provenance_summary.json")
    runtime_claims = [
        bool(b(v100_final.get("runtime_action_allowed"))),
        bool(b(v100_final.get("runtime_action_pilot_run"))),
        bool(b(v100_final.get("full_method_success"))),
        bool(b(track_q.get("runtime_action_allowed"))),
    ]
    diagnostic_summaries = {
        name: read_json(V100_ROOT / f"{name}/summary.json")
        for name in [
            "trackN2_anchor_identity_graph",
            "trackO2_freshness_current_support",
            "trackC4_identity_latent_gauge_ruler",
            "trackF4_ttt_write_to_use_same_space",
            "trackR_edge_head_control_audit",
            "trackR2_anchor_edge_identity_control_audit",
            "trackL2_anchor_scale_observability",
            "trackQ_chunk_update_admission",
        ]
    }
    stage0_pass = (
        b(track_s.get("gate_pass"))
        and bool(diagnostic_summaries)
        and bool(track_q)
        and bool(hygiene)
        and not any(runtime_claims)
        and not missing
    )
    claim_rows = []
    for name, payload in diagnostic_summaries.items():
        claim_rows.append(
            {
                "track": name,
                "status": payload.get("status", ""),
                "gate_pass": b(payload.get("gate_pass")),
                "runtime_action_allowed": b(payload.get("runtime_action_allowed")),
                "claim_level": "blocked_or_proxy_diagnostic" if not b(payload.get("gate_pass")) else "diagnostic_only_pass",
                "blocker": payload.get("blocker", ""),
            }
        )
    do_not_repeat = [
        "weak-context READ skip strength sweep",
        "READ beta / T035/T045/T050 tiny sweep",
        "DG-Q90 per-head source-bias sweep",
        "Stage7f aggregate prev-anchor gate rho/min/layer sweep",
        "Stage7h query-soft ge75/ge90/rho sweep",
        "Track E source gate/source replace/merge alpha",
        "TTT no-write action from proxy write mass",
        "case-level R_same/query_hit threshold sweep",
        "freshness_score threshold sweep",
    ]
    summary = {
        "schema": "acl2_v101_stage0_evidence_ledger_v1",
        "status": "complete" if stage0_pass else "complete_with_missing_or_claim_gap",
        "gate_pass": stage0_pass,
        "v100_trackS_gate_pass": b(track_s.get("gate_pass")),
        "v100_best_canonical_space": track_s.get("best_canonical_space", ""),
        "v100_trackQ_proxy_only": b(track_q.get("proxy_only")),
        "v100_trackQ_best_balanced_accuracy": track_q.get("best_balanced_accuracy", math.nan),
        "v100_trackQ_best_good_FPR": track_q.get("best_good_FPR", math.nan),
        "label_l3_conflicts_loaded": bool(hygiene),
        "missing_required_artifact_count": len(missing),
        "missing_required_artifacts": missing,
        "runtime_claims_present": any(runtime_claims),
        "claim": "Stage0 is an evidence ledger only; no runtime or full validation success is claimed.",
    }
    write_json(out / "summary.json", summary)
    write_rows(out / "case_artifact_manifest.csv", artifacts)
    write_rows(out / "missing_artifact_report.csv", missing)
    write_rows(out / "claim_level_by_track.csv", claim_rows)
    write_text(
        out / "v100_fact_lock.md",
        "# v100 Fact Lock\n\n"
        f"- Track S gate pass: {summary['v100_trackS_gate_pass']}\n"
        f"- Best canonical space: {summary['v100_best_canonical_space']}\n"
        f"- v100 final taxonomy: {v100_final.get('final_taxonomy', '')}\n"
        f"- runtime_action_allowed: {v100_final.get('runtime_action_allowed', '')}\n"
        f"- full_method_success: {v100_final.get('full_method_success', '')}\n"
        f"- Track Q proxy best BA: {summary['v100_trackQ_best_balanced_accuracy']}\n"
        f"- Track Q proxy best good FPR: {summary['v100_trackQ_best_good_FPR']}\n",
    )
    write_text(
        out / "blocked_direction_list.md",
        "# Blocked Direction List\n\n"
        + "\n".join(f"- {row['track']}: {row['blocker']}" for row in claim_rows if row.get("blocker")),
    )
    write_text(
        out / "reusable_signal_list.md",
        "# Reusable Signal List\n\n"
        "- v100 S-B same-space anchor rows.\n"
        "- v100 Track Q composite admission proxy, diagnostic only.\n"
        "- v100 L2 geometry sidecar proxy rows, diagnostic only.\n"
        "- v100 label/L3 hygiene provenance and oracle feasibility diagnostics.\n",
    )
    write_text(out / "do_not_repeat_list.md", "# Do Not Repeat List\n\n" + "\n".join(f"- {item}" for item in do_not_repeat))
    write_track_common(
        out,
        failure=(
            "Stage0 passed as evidence ledger." if stage0_pass else "Stage0 did not fully pass; see missing_artifact_report.csv and claim_level_by_track.csv."
        ),
        what_would="Stage0 requires v100 TrackS, downstream fail summaries, Q proxy, label-L3 conflicts, old branch statuses, and no runtime pass claims.",
        next_attempt="Proceed to Track T target hygiene; do not run M4/runtime from v100 proxy evidence.",
        control_gap="Stage0 is a ledger; controls are inherited from each referenced diagnostic track.",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def target_taxonomy() -> dict[str, Any]:
    out = ROOT / "trackT_drift_target_relabel"
    case_rows = read_rows(CASE_ROWS)
    q_by_case = by_case(read_rows(Q_ROWS))
    if not case_rows:
        summary = {"schema": "acl2_v101_trackT_target_taxonomy_v1", "status": "blocked_missing_case_rows", "gate_pass": False}
        write_json(out / "target_taxonomy_summary.json", summary)
        write_track_common(
            out,
            failure=f"Missing required case rows: {CASE_ROWS}",
            what_would="Track T requires all 28 v100 cases with label/L3/hygiene provenance.",
            next_attempt="Regenerate v100 label_l3_hygiene_provenance_rows.csv before retrying Track T.",
        )
        return summary

    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in case_rows]
    good_rows = [row for row in case_rows if row.get("case_label") == "good"]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in good_rows]
    thresholds = {
        "median_good_L3": median(good_l3),
        "MAD_good_L3": mad(good_l3),
        "Q75_all_L3": quantile(l3_all, 0.75),
        "Q40_all_L3": quantile(l3_all, 0.40),
        "Q60_all_L3": quantile(l3_all, 0.60),
        "Q50_good_L3": quantile(good_l3, 0.50),
    }
    thresholds["L3_high"] = max(
        thresholds["median_good_L3"] + 2.0 * thresholds["MAD_good_L3"],
        thresholds["Q75_all_L3"],
    )
    thresholds["L3_low"] = thresholds["Q40_all_L3"]

    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    safe_good: list[dict[str, Any]] = []
    for raw in case_rows:
        case_id = str(raw.get("case_id", ""))
        qrow = q_by_case.get(case_id, {})
        label = str(raw.get("case_label", ""))
        failure = str(raw.get("failure_type", ""))
        failure_upper = failure.upper()
        l3 = f(raw.get("L3_handoff_transfer_penalty_proxy"))
        is_good = label == "good"
        is_non_good = label != "good"
        l3_high = math.isfinite(l3) and l3 >= thresholds["L3_high"]
        l3_low = math.isfinite(l3) and l3 <= thresholds["L3_low"]
        hygiene_excluded = b(raw.get("v98_hygiene_excluded_good_control"))
        lowobs_or_multimode = ("LOW_OBSERVABILITY" in failure_upper) or ("MULTIMODE_CONFLICT" in failure_upper)
        handoff_failure = ("HANDOFF_SCALE" in failure_upper) or ("HANDOFF_GAUGE" in failure_upper)
        reasons: list[str] = []
        if is_good and (hygiene_excluded or l3_high):
            taxonomy = "GOOD_HIGH_L3_CONTAMINATED"
            reasons.append("good_label_with_high_l3_or_v98_hygiene_exclusion")
        elif lowobs_or_multimode and l3_high:
            taxonomy = "MULTIMODE_LOWOBS_ABSTAIN"
            reasons.append("lowobs_or_multimode_high_l3")
        elif is_non_good and handoff_failure and l3_high:
            taxonomy = "HANDOFF_SCALE_GAUGE_TARGET"
            reasons.append("non_good_handoff_failure_and_l3_high")
        elif is_non_good and l3_low:
            taxonomy = "LOCAL_BAD_NOT_HANDOFF"
            reasons.append("non_good_l3_low")
        elif is_good and l3_low and not hygiene_excluded:
            taxonomy = "SAFE_GOOD"
            reasons.append("good_label_l3_low")
        elif is_non_good and handoff_failure:
            taxonomy = "AMBIGUOUS_SUPPORT"
            reasons.append("handoff_failure_but_l3_not_high")
        elif is_good:
            taxonomy = "AMBIGUOUS_SUPPORT"
            reasons.append("good_label_but_l3_not_low")
        else:
            taxonomy = "AMBIGUOUS_SUPPORT"
            reasons.append("mixed_or_unhandled_failure_type")
        row = {
            **raw,
            "target_taxonomy": taxonomy,
            "target_reason": ";".join(reasons),
            "L3_high_threshold": thresholds["L3_high"],
            "L3_low_threshold": thresholds["L3_low"],
            "is_L3_high": l3_high,
            "is_L3_low": l3_low,
            "lowobs_or_multimode": lowobs_or_multimode,
            "handoff_failure_mode": handoff_failure,
            "admission_decision_proxy": qrow.get("admission_decision_proxy", ""),
            "q_composite_delay_or_no_scale_proxy": qrow.get("q_composite_delay_or_no_scale_proxy", ""),
            "admission_block_score_proxy": qrow.get("admission_block_score_proxy", ""),
            "admission_allow_score_proxy": qrow.get("admission_allow_score_proxy", ""),
        }
        rows.append(row)
        if raw.get("label_l3_conflict"):
            conflicts.append(row)
        if taxonomy == "SAFE_GOOD":
            safe_good.append(row)
        if taxonomy == "AMBIGUOUS_SUPPORT":
            ambiguous.append(row)

    counts = Counter(row["target_taxonomy"] for row in rows)
    per_seq_rows = []
    for seq, parts in sorted(defaultdict(list, {seq: [r for r in rows if r.get("seq") == seq] for seq in sorted({r.get("seq") for r in rows})}).items()):
        c = Counter(row["target_taxonomy"] for row in parts)
        per_seq_rows.append({"seq": seq, **dict(c), "case_count": len(parts)})
    handoff_targets = [row for row in rows if row["target_taxonomy"] == "HANDOFF_SCALE_GAUGE_TARGET"]
    sequence_coverage = len({row.get("seq", "") for row in handoff_targets})
    gate = (
        len(rows) == 28
        and len(safe_good) >= 6
        and len(handoff_targets) >= 8
        and sequence_coverage >= 3
        and counts.get("GOOD_HIGH_L3_CONTAMINATED", 0) >= 1
    )
    summary = {
        "schema": "acl2_v101_trackT_target_taxonomy_v1",
        "status": "complete" if rows else "blocked_missing_rows",
        "gate_pass": gate,
        "case_count": len(rows),
        "thresholds": thresholds,
        "target_counts": dict(counts),
        "safe_good_count": len(safe_good),
        "handoff_scale_gauge_target_count": len(handoff_targets),
        "handoff_target_sequence_coverage": sequence_coverage,
        "good_high_l3_contaminated_cases": ";".join(row["case_id"] for row in rows if row["target_taxonomy"] == "GOOD_HIGH_L3_CONTAMINATED"),
        "handoff_target_cases": ";".join(row["case_id"] for row in handoff_targets),
        "safe_good_cases": ";".join(row["case_id"] for row in safe_good),
        "ambiguous_case_count": len(ambiguous),
        "runtime_action_allowed": False,
        "blocker": "" if gate else "Track T target universe did not satisfy safe-good / handoff-target count and sequence-coverage requirements.",
    }
    target_insufficient = [
        {
            "check": "SAFE_GOOD count >= 6",
            "observed": len(safe_good),
            "pass": len(safe_good) >= 6,
            "repair_direction": "Search/extend low-L3 good controls with v100 schema and same-space trace; do not use high-L3 contaminated good controls.",
        },
        {
            "check": "HANDOFF_SCALE_GAUGE_TARGET count >= 8",
            "observed": len(handoff_targets),
            "pass": len(handoff_targets) >= 8,
            "repair_direction": "Extend v95/v94 handoff cases while preserving v100 schema; do not force LOCAL_BAD into handoff target.",
        },
        {
            "check": "handoff target sequence coverage >= 3",
            "observed": sequence_coverage,
            "pass": sequence_coverage >= 3,
            "repair_direction": "Add handoff cases from additional sequences with same target taxonomy.",
        },
    ]
    write_rows(out / "target_universe_v101.csv", rows)
    write_rows(out / "label_l3_conflict_rows.csv", conflicts)
    write_rows(out / "per_sequence_target_distribution.csv", per_seq_rows)
    write_rows(out / "safe_good_controls.csv", safe_good)
    write_rows(out / "ambiguous_cases.csv", ambiguous)
    write_rows(out / "target_insufficient_checks.csv", target_insufficient)
    write_json(out / "target_taxonomy_summary.json", summary)
    write_text(
        out / "target_taxonomy_report.md",
        "# Track T Target Taxonomy Report\n\n"
        f"Gate pass: {gate}\n\n"
        f"Thresholds: `{json.dumps(thresholds, sort_keys=True)}`\n\n"
        f"Target counts: `{json.dumps(dict(counts), sort_keys=True)}`\n\n"
        f"HANDOFF target cases: {summary['handoff_target_cases'] or 'none'}\n\n"
        f"SAFE_GOOD cases: {summary['safe_good_cases'] or 'none'}\n\n"
        f"GOOD_HIGH_L3_CONTAMINATED cases: {summary['good_high_l3_contaminated_cases'] or 'none'}\n\n"
        "This taxonomy is data-driven and diagnostic. It does not relabel cases for runtime action.\n",
    )
    write_track_common(
        out,
        failure=summary["blocker"] or "Track T passed target hygiene gate.",
        what_would="Track T requires all 28 cases labelled, SAFE_GOOD>=6, HANDOFF_SCALE_GAUGE_TARGET>=8, handoff sequence coverage>=3, and contaminated good controls removed from the good pool.",
        next_attempt=(
            "Extend target universe from v95/v94 handoff cases with v100 schema and same-space trace, or split binary label target from L3 metric target before downstream action."
            if not gate
            else "Proceed to current-support and observability materialization; runtime remains blocked."
        ),
        control_gap="No runtime control is applicable at Track T. Main gap is target-universe size/coverage if gate fails.",
    )
    write_text(out / "visual_manifest.csv", "path,description\n")
    return summary


def build_support_rows(track_t_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackU_true_current_support"
    s_rows = [row for row in read_rows(S_ROWS) if row.get("canonical_space_name") == "S-B_preprojection_hidden"]
    case_meta = by_case(read_rows(Q_ROWS) or read_rows(L2_ROWS))
    target_meta = by_case(read_rows(ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"))
    rows = []
    for row in s_rows:
        case_id = str(row.get("case_id", ""))
        cmeta = case_meta.get(case_id, {})
        tmeta = target_meta.get(case_id, {})
        s_feat_vals = [
            math.exp(-max(0.0, f(row.get("R_ref_current"), 0.0))) if math.isfinite(f(row.get("R_ref_current"))) else math.nan,
            math.exp(-max(0.0, f(row.get("R_cache_current"), 0.0))) if math.isfinite(f(row.get("R_cache_current"))) else math.nan,
        ]
        s_feat = mean(s_feat_vals)
        s_read = clip01(f(row.get("query_hit_max"), 0.0))
        geom_support = clip01(cmeta.get("geometry_current_support_proxy"))
        semantic_support = clip01(row.get("source_label_mode_frac"))
        support_vals = [semantic_support, s_feat, geom_support]
        s_cur = mean(support_vals)
        flags = []
        if math.isfinite(semantic_support):
            flags.append("semantic_class_fallback")
        if math.isfinite(s_feat):
            flags.append("same_space_feature")
        if math.isfinite(geom_support):
            flags.append("case_level_geometry_sidecar_proxy")
        rows.append(
            {
                "case_id": case_id,
                "boundary_id": f"{row.get('source_chunk', '')}->{row.get('current_chunk', '')}",
                "anchor_id": row.get("anchor_id", ""),
                "semantic_label": row.get("semantic_class", ""),
                "target_taxonomy": tmeta.get("target_taxonomy", ""),
                "memory_role_candidate": row.get("anchor_role", ""),
                "S_sem": semantic_support,
                "S_vis": geom_support,
                "S_feat": s_feat,
                "S_overlap": clip01(cmeta.get("geometry_support_proxy_norm")),
                "S_read": s_read,
                "S_geom": geom_support,
                "S_cur_combined": s_cur,
                "support_source_flags": ";".join(flags),
                "support_quality": "proxy_current_support",
                "identity_resolution_level": "semantic_class_fallback",
                "proxy_only": True,
                "R_same": row.get("R_same", ""),
                "query_hit_max": row.get("query_hit_max", ""),
                "L3_handoff_transfer_penalty_proxy": tmeta.get("L3_handoff_transfer_penalty_proxy", cmeta.get("L3_handoff_transfer_penalty_proxy", "")),
            }
        )
    case_coverage = len({row["case_id"] for row in rows})
    target_cases = {row.get("case_id") for row in read_rows(ROOT / "trackT_drift_target_relabel/target_universe_v101.csv")}
    feature_frac = mean([1.0 if math.isfinite(f(row.get("S_feat"))) else 0.0 for row in rows])
    semantic_frac = mean([1.0 if math.isfinite(f(row.get("S_sem"))) else 0.0 for row in rows])
    non_feature_frac = mean([1.0 if ("case_level_geometry_sidecar_proxy" in str(row.get("support_source_flags"))) else 0.0 for row in rows])
    materialization_gate = (
        bool(rows)
        and case_coverage >= 24
        and (case_coverage / max(len(target_cases), 1)) >= 0.90
        and feature_frac >= 0.95
        and semantic_frac >= 0.80
        and non_feature_frac >= 0.80
    )
    summary = {
        "schema": "acl2_v101_trackU_true_current_support_v1",
        "status": "complete_proxy_materialization" if rows else "blocked_missing_same_space_rows",
        "gate_pass": materialization_gate,
        "true_current_support_strict_pass": False,
        "proxy_only": True,
        "anchor_support_row_count": len(rows),
        "case_coverage": case_coverage,
        "feature_support_available_frac": feature_frac,
        "semantic_support_available_frac": semantic_frac,
        "non_feature_support_available_frac": non_feature_frac,
        "identity_resolution_level": "semantic_class_fallback",
        "runtime_action_allowed": False,
        "blocker": (
            "" if materialization_gate else "Track U proxy support materialization did not meet coverage/source availability requirements."
        ),
        "note": "Rows use semantic-class fallback and case-level geometry sidecar proxies; no instance-level current support claim is made.",
    }
    write_rows(out / "anchor_current_support_rows.csv", rows)
    source_rows = [
        {"support_source": "semantic_class_fallback", "available_frac": semantic_frac},
        {"support_source": "same_space_feature", "available_frac": feature_frac},
        {"support_source": "case_level_geometry_sidecar_proxy", "available_frac": non_feature_frac},
    ]
    write_rows(out / "support_source_coverage.csv", source_rows)
    write_json(out / "current_support_summary.json", summary)
    write_text(out / "support_missing_report.md", "Instance-level semantic/component visibility was not available; semantic_class fallback was used for diagnostics only.")
    write_text(out / "support_visual_manifest.csv", "path,description\n")
    write_track_common(
        out,
        failure=summary["blocker"] or "Track U proxy materialization gate passed, but strict instance-level true support remains unavailable.",
        what_would="Track U strict pass requires stable instance/component visibility or geometry support at anchor level, not only case-level proxy support.",
        next_attempt="Search/materialize semantic component or pointmap support per anchor; keep READ as provider only until Q2/M4 gates pass.",
        control_gap="No semantic instance id or component tracklet ID was proven stable in this materialization.",
    )
    return summary


def build_observability_rows(track_t_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackV_anchor_scale_observability"
    support_rows = read_rows(ROOT / "trackU_true_current_support/anchor_current_support_rows.csv")
    case_meta = by_case(read_rows(Q_ROWS) or read_rows(L2_ROWS))
    target_meta = by_case(read_rows(ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"))
    rows = []
    for row in support_rows:
        case_id = row.get("case_id", "")
        cmeta = case_meta.get(case_id, {})
        tmeta = target_meta.get(case_id, {})
        geom_available = b(cmeta.get("geometry_sidecar_terms_available"))
        parallax_temporal_only = b(cmeta.get("parallax_proxy_is_temporal_frame_delta_only"))
        o_scale = clip01(cmeta.get("scale_observability_score"))
        rows.append(
            {
                **row,
                "O_scale": o_scale,
                "pointmap_depth_source_level": "geometry_sidecar_case_level" if geom_available else "missing",
                "anchor_depth_mean": cmeta.get("geometry_current_depth_mean", ""),
                "anchor_depth_std": cmeta.get("geometry_current_depth_spread", ""),
                "anchor_inverse_depth_std": "",
                "anchor_point_spread_svd_ratio": "",
                "anchor_point_count": cmeta.get("geometry_anchor_count", ""),
                "far_depth_fraction": "",
                "anchor_frame_span": cmeta.get("head_top1_abs_frame_delta_mean_max", ""),
                "anchor_top1_frame_delta": cmeta.get("head_top1_abs_frame_delta_mean_mean", ""),
                "anchor_cross_chunk_pixel_motion_proxy": cmeta.get("geometry_world_pair_distance_mean", ""),
                "temporal_proxy_only": parallax_temporal_only,
                "geometry_sidecar_terms_available": geom_available,
                "scale_observability_score_is_proxy": b(cmeta.get("scale_observability_score_is_proxy")),
                "target_taxonomy": tmeta.get("target_taxonomy", row.get("target_taxonomy", "")),
                "L3_handoff_transfer_penalty_proxy": tmeta.get("L3_handoff_transfer_penalty_proxy", row.get("L3_handoff_transfer_penalty_proxy", "")),
            }
        )
    target_rows = [row for row in rows if row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET"]
    target_available = mean([1.0 if math.isfinite(f(row.get("O_scale"))) else 0.0 for row in target_rows]) if target_rows else 0.0
    true_geometry_frac = mean([1.0 if b(row.get("geometry_sidecar_terms_available")) else 0.0 for row in rows])
    proxy_only = True
    diagnostic_allowed = bool(rows) and true_geometry_frac > 0.0
    gate = (
        bool(target_rows)
        and target_available >= 0.80
        and true_geometry_frac >= 0.80
        and not proxy_only
    )
    summary = {
        "schema": "acl2_v101_trackV_anchor_scale_observability_v1",
        "status": "complete_proxy_diagnostic" if rows else "blocked_missing_support_rows",
        "gate_pass": gate,
        "diagnostic_allowed": diagnostic_allowed,
        "proxy_only": proxy_only,
        "anchor_observability_row_count": len(rows),
        "handoff_target_anchor_row_count": len(target_rows),
        "handoff_target_observability_available_frac": target_available,
        "geometry_sidecar_available_frac": true_geometry_frac,
        "runtime_action_allowed": False,
        "blocker": "Track V has geometry sidecar proxy rows but lacks validated anchor-level true scale observability/control reruns.",
    }
    write_rows(out / "anchor_observability_rows.csv", rows)
    write_rows(out / "pointmap_depth_support_rows.csv", rows)
    write_rows(out / "parallax_support_rows.csv", rows)
    write_rows(out / "scale_mode_rows.csv", [])
    write_json(out / "observability_summary.json", summary)
    write_text(
        out / "observability_missing_report.md",
        "Per-anchor local scale-mode rows and validated parallax/depth controls are missing. Existing geometry sidecar terms are kept as diagnostic proxies.",
    )
    write_track_common(
        out,
        failure=summary["blocker"],
        what_would="Track V true pass requires validated anchor-level point/depth/parallax or local scale-mode observability for HANDOFF targets.",
        next_attempt="Materialize per-anchor point spread / scale-mode rows or rerun geometry sidecar controls; do not use temporal frame delta as true parallax.",
        control_gap="Missing local scale-mode controls and anchor-id/semantic query-head control reruns.",
    )
    return summary


def build_roles(track_t_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackW_anchor_memory_role"
    rows_in = read_rows(ROOT / "trackV_anchor_scale_observability/anchor_observability_rows.csv")
    s_cur_vals = [row.get("S_cur_combined") for row in rows_in]
    o_vals = [row.get("O_scale") for row in rows_in]
    r_vals = [row.get("R_same") for row in rows_in]
    s_med = median(s_cur_vals)
    o_med = median(o_vals)
    r_med = median(r_vals)
    role_rows = []
    for row in rows_in:
        s_cur = f(row.get("S_cur_combined"))
        o_scale = f(row.get("O_scale"))
        r_same = f(row.get("R_same"))
        target = row.get("target_taxonomy", "")
        if target == "MULTIMODE_LOWOBS_ABSTAIN":
            role = "dynamic_or_boundary_risk"
            reason = "target_multimode_lowobs"
        elif math.isfinite(s_cur) and math.isfinite(o_scale) and s_cur >= s_med and o_scale >= o_med and r_same <= r_med:
            role = "landmark"
            reason = "high_support_high_observability_low_residual"
        elif math.isfinite(s_cur) and s_cur >= s_med and (not math.isfinite(o_scale) or o_scale < o_med):
            role = "context_only"
            reason = "support_present_but_low_observability"
        elif math.isfinite(s_cur) and s_cur < s_med and math.isfinite(r_same) and r_same >= r_med:
            role = "stale_candidate"
            reason = "low_support_high_residual"
        elif math.isfinite(s_cur) and s_cur >= s_med:
            role = "local_recent"
            reason = "support_present"
        else:
            role = "context_only"
            reason = "fallback_low_confidence"
        role_rows.append(
            {
                **row,
                "role": role,
                "role_confidence": 0.75 if role != "context_only" else 0.55,
                "role_reason": reason,
                "allowed_READ_behavior": "context_read" if role in {"context_only", "dynamic_or_boundary_risk"} else "read_as_candidate_evidence",
                "allowed_SWA_behavior": "context_only" if role in {"context_only", "dynamic_or_boundary_risk", "stale_candidate"} else "candidate_handoff_evidence",
                "allowed_TTT_behavior": "metadata_only" if role in {"context_only", "dynamic_or_boundary_risk"} else "candidate_state_update",
                "forbidden_behavior": "runtime_scale_update_without_Q2_M4",
            }
        )
    target_roles = [row for row in role_rows if row.get("target_taxonomy") in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}]
    role_counts = Counter(row["role"] for row in role_rows)
    safe_roles = Counter(row["role"] for row in role_rows if row.get("target_taxonomy") == "SAFE_GOOD")
    handoff_roles = Counter(row["role"] for row in role_rows if row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET")
    gate = (
        bool(role_rows)
        and mean([1.0 if row.get("role") else 0.0 for row in role_rows]) >= 0.90
        and len(role_counts) >= 2
        and bool(target_roles)
        and safe_roles != handoff_roles
    )
    summary = {
        "schema": "acl2_v101_trackW_anchor_memory_role_v1",
        "status": "complete_proxy_role_assignment" if role_rows else "blocked_missing_observability_rows",
        "gate_pass": gate,
        "role_row_count": len(role_rows),
        "role_counts": dict(role_counts),
        "safe_good_role_counts": dict(safe_roles),
        "handoff_target_role_counts": dict(handoff_roles),
        "uses_S_cur_and_O_scale": True,
        "proxy_only": True,
        "runtime_action_allowed": False,
        "blocker": "" if gate else "Role assignment collapsed or lacks target-vs-safe interpretability.",
    }
    write_rows(out / "anchor_role_rows.csv", role_rows)
    write_rows(out / "role_transition_rows.csv", [])
    write_json(out / "role_summary.json", summary)
    write_text(out / "role_visual_manifest.csv", "path,description\n")
    write_track_common(
        out,
        failure=summary["blocker"] or "Track W proxy role assignment gate passed; runtime remains blocked.",
        what_would="Track W requires roles for >=90% target anchor rows, S_cur/O_scale usage, no collapse, and interpretable safe-vs-target differences.",
        next_attempt="Use instance-level current support and true observability before allowing role-conditioned action.",
        control_gap="Role classifier uses proxy S_cur/O_scale and lacks instance-level identity transition controls.",
    )
    return summary


def build_state_estimator(track_t_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackS2_anchor_state_estimator"
    rows_in = read_rows(ROOT / "trackW_anchor_memory_role/anchor_role_rows.csv")
    r_vals = [row.get("R_same") for row in rows_in]
    s_vals = [row.get("S_cur_combined") for row in rows_in]
    o_vals = [row.get("O_scale") for row in rows_in]
    r_finite = [f(v) for v in r_vals if math.isfinite(f(v))]
    r_lo = min(r_finite) if r_finite else math.nan
    r_hi = max(r_finite) if r_finite else math.nan
    r_med_raw = median(r_vals)
    s_med = median(s_vals)
    o_med = median(o_vals)
    r_norm_vals = [norm01_with_bounds(v, r_lo, r_hi) for v in r_vals]
    r_norm_med = median(r_norm_vals)
    rows = []
    for row in rows_in:
        r_norm = norm01_with_bounds(row.get("R_same"), r_lo, r_hi)
        s_cur = clip01(row.get("S_cur_combined"))
        o_scale = clip01(row.get("O_scale"))
        p_anchor = mean([r_norm, 1.0 - s_cur if math.isfinite(s_cur) else math.nan, 1.0 - o_scale if math.isfinite(o_scale) else math.nan])
        o_anchor = s_cur * o_scale if math.isfinite(s_cur) and math.isfinite(o_scale) else math.nan
        k_anchor = o_anchor / (o_anchor + p_anchor + EPS) if all(math.isfinite(v) for v in [o_anchor, p_anchor]) else math.nan
        if all(math.isfinite(v) for v in [s_cur, o_scale, r_norm]) and s_cur >= s_med and o_scale >= o_med and r_norm <= r_norm_med:
            status = "supported_consistent"
        elif all(math.isfinite(v) for v in [s_cur, o_scale, r_norm]) and s_cur < s_med and r_norm > r_norm_med:
            status = "unsupported_inconsistent"
        elif math.isfinite(s_cur) and s_cur >= s_med:
            status = "supported_inconsistent"
        else:
            status = "unsupported_consistent"
        rows.append(
            {
                **row,
                "r_write_cache": "",
                "r_cache_current": "",
                "r_ref_current": "",
                "P_anchor": p_anchor,
                "Q_anchor": r_norm,
                "O_anchor": o_anchor,
                "K_anchor": k_anchor,
                "state_status": status,
            }
        )
    k_vals = [f(row.get("K_anchor")) for row in rows if math.isfinite(f(row.get("K_anchor")))]
    status_counts = Counter(row["state_status"] for row in rows)
    status_case_l3: dict[str, list[float]] = defaultdict(list)
    seen_case_status: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("case_id", ""), row.get("state_status", ""))
        if key in seen_case_status:
            continue
        seen_case_status.add(key)
        status_case_l3[row.get("state_status", "")].append(f(row.get("L3_handoff_transfer_penalty_proxy")))
    supported_l3 = mean(status_case_l3.get("supported_consistent", []))
    unsupported_l3 = mean(status_case_l3.get("unsupported_inconsistent", []))
    gate = bool(rows) and k_vals and (max(k_vals) - min(k_vals) > 0.05) and math.isfinite(supported_l3) and math.isfinite(unsupported_l3) and supported_l3 < unsupported_l3
    summary = {
        "schema": "acl2_v101_trackS2_anchor_state_estimator_v1",
        "status": "complete_proxy_state_estimator" if rows else "blocked_missing_role_rows",
        "gate_pass": gate,
        "state_row_count": len(rows),
        "R_same_raw_median": r_med_raw,
        "R_same_norm_median": r_norm_med,
        "K_anchor_min": min(k_vals) if k_vals else math.nan,
        "K_anchor_max": max(k_vals) if k_vals else math.nan,
        "state_status_counts": dict(status_counts),
        "supported_consistent_mean_L3": supported_l3,
        "unsupported_inconsistent_mean_L3": unsupported_l3,
        "proxy_only": True,
        "runtime_action_allowed": False,
        "blocker": "" if gate else "State estimator did not prove supported_consistent lower-L3 than unsupported_inconsistent under proxy support/observability.",
    }
    write_rows(out / "anchor_state_rows.csv", rows)
    write_rows(out / "uncertainty_rows.csv", rows)
    write_rows(out / "gain_rows.csv", rows)
    write_json(out / "state_estimator_summary.json", summary)
    write_track_common(
        out,
        failure=summary["blocker"] or "Track S2 proxy state estimator gate passed; runtime remains blocked.",
        what_would="Track S2 requires non-collapsed K and state_status separation with controls in target universe.",
        next_attempt="Rerun S2 after true Track U/V materialization; add anchor-rotation controls for state_status.",
        control_gap="Anchor-rotation and same-count controls for state_status are not materialized in this proxy builder.",
    )
    return summary


def build_q2(track_t_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "trackQ2_scale_update_admission"
    target_rows = read_rows(ROOT / "trackT_drift_target_relabel/target_universe_v101.csv")
    q_rows = read_rows(Q_ROWS)
    q_by_case = by_case(q_rows)
    rows = []
    for t in target_rows:
        case_id = t.get("case_id", "")
        q = q_by_case.get(case_id, {})
        a_allow = f(q.get("admission_allow_score_proxy"))
        a_stale = f(q.get("stale_anchor_score_proxy"))
        block = f(q.get("admission_block_score_proxy"))
        if math.isfinite(a_allow) and math.isfinite(a_stale) and a_allow >= median([r.get("admission_allow_score_proxy") for r in q_rows]) and a_stale < median([r.get("stale_anchor_score_proxy") for r in q_rows]):
            decision = "ALLOW_UPDATE"
        elif math.isfinite(block) and block >= quantile([r.get("admission_block_score_proxy") for r in q_rows], 0.75):
            decision = "NO_SCALE_EVIDENCE"
        elif t.get("target_taxonomy") in {"GOOD_HIGH_L3_CONTAMINATED", "MULTIMODE_LOWOBS_ABSTAIN"}:
            decision = "CONTEXT_ONLY"
        else:
            decision = "DELAY_UPDATE"
        rows.append(
            {
                **t,
                "admission_decision": decision,
                "admission_score": a_allow,
                "A_allow_proxy": a_allow,
                "A_stale_proxy": a_stale,
                "admission_block_score_proxy": block,
                "fresh_supported_anchor_count": q.get("fresh_supported_anchor_count_proxy", ""),
                "fresh_supported_anchor_mass": q.get("fresh_supported_score_proxy", ""),
                "stale_unsupported_anchor_mass": q.get("stale_anchor_score_proxy", ""),
                "scale_observable_anchor_count": "",
                "context_only_anchor_mass": "",
                "no_scale_evidence_score": q.get("no_scale_evidence_proxy", ""),
                "proxy_only": True,
            }
        )
    positives = {row["case_id"] for row in rows if row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET"}
    negatives = {row["case_id"] for row in rows if row.get("target_taxonomy") == "SAFE_GOOD"}
    selected = {row["case_id"] for row in rows if row.get("admission_decision") in {"DELAY_UPDATE", "NO_SCALE_EVIDENCE"}}
    metrics = selected_metrics(rows, selected, positives, negatives)
    metrics["same_count_margin"] = same_count_margin(rows, selected, positives, negatives)
    proxy_stage_pass = (
        metrics["positive_case_count"] > 0
        and metrics["negative_case_count"] > 0
        and f(metrics.get("bad_recall")) >= 0.65
        and f(metrics.get("good_FPR")) <= 0.25
        and f(metrics.get("selected_positive_sequence_coverage")) >= 3
        and f(metrics.get("same_count_margin")) >= 0.05
        and (f(metrics.get("abs_corr_L3")) >= 0.50)
    )
    summary = {
        "schema": "acl2_v101_trackQ2_scale_update_admission_v1",
        "status": "complete_proxy_diagnostic" if rows else "blocked_missing_trackT_rows",
        "proxy_stage_pass": proxy_stage_pass,
        "true_stage_pass": False,
        "gate_pass": False,
        "proxy_only": True,
        "case_count": len(rows),
        "target_positive_count": len(positives),
        "safe_good_count": len(negatives),
        "admission_decision_counts": dict(Counter(row["admission_decision"] for row in rows)),
        "metrics": metrics,
        "runtime_action_allowed": False,
        "blocker": "Track Q2 true-stage blocked: current support / scale observability are proxy-only and Track T did not prove a sufficient clean target universe.",
    }
    fnfp_rows = []
    for row in rows:
        label = "positive" if row["case_id"] in positives else ("negative" if row["case_id"] in negatives else "excluded")
        sel = row["case_id"] in selected
        if label == "positive" and not sel:
            err = "missed_positive"
        elif label == "negative" and sel:
            err = "false_positive"
        else:
            err = ""
        if err:
            fnfp_rows.append({"case_id": row["case_id"], "error_type": err, **row})
    write_rows(out / "admission_rows.csv", rows)
    write_rows(out / "admission_rule_search.csv", [metrics])
    write_rows(out / "admission_metric_summary.csv", [metrics])
    write_rows(out / "false_positive_false_negative_rows.csv", fnfp_rows)
    write_json(out / "Q2_summary.json", summary)
    write_text(
        out / "admission_false_positive_negative_report.md",
        "# Q2 False Positive / False Negative Report\n\n"
        f"False positives: {metrics.get('false_positive_cases') or 'none'}\n\n"
        f"Missed positives: {metrics.get('missed_positive_cases') or 'none'}\n",
    )
    write_text(out / "admission_visual_manifest.csv", "path,description\n")
    write_track_common(
        out,
        failure=summary["blocker"],
        what_would="Track Q2 true-stage pass requires Track T clean targets, true current support/observability, controls, and either abs corr L3>=0.50 or documented improvement over v100 Q proxy with lower FPR.",
        next_attempt="Materialize true Track U/V or extend target universe before M4; do not run runtime from proxy Q2.",
        control_gap="Semantic rotation and anchor-id rotation controls are not available in proxy Q2.",
    )
    return summary


def final_decision(stage0_summary: dict[str, Any], track_t: dict[str, Any], track_u: dict[str, Any], track_v: dict[str, Any], track_w: dict[str, Any], track_s2: dict[str, Any], track_q2: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "final_decision"
    m4_allowed = (
        b(track_t.get("gate_pass"))
        and b(track_u.get("gate_pass"))
        and (b(track_v.get("gate_pass")) or (b(track_v.get("diagnostic_allowed")) and not b(track_q2.get("true_stage_pass"))))
        and b(track_q2.get("true_stage_pass"))
    )
    decision = {
        "schema": "acl2_v101_final_decision_v1",
        "stage0_gate_pass": b(stage0_summary.get("gate_pass")),
        "trackT_gate_pass": b(track_t.get("gate_pass")),
        "trackU_gate_pass": b(track_u.get("gate_pass")),
        "trackV_gate_pass": b(track_v.get("gate_pass")),
        "trackW_gate_pass": b(track_w.get("gate_pass")),
        "trackS2_gate_pass": b(track_s2.get("gate_pass")),
        "trackQ2_proxy_stage_pass": b(track_q2.get("proxy_stage_pass")),
        "trackQ2_true_stage_pass": b(track_q2.get("true_stage_pass")),
        "trackM4_run_allowed": m4_allowed,
        "trackM4_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_pilot_run": False,
        "full_validation_run": False,
        "full_method_success": False,
        "final_taxonomy": "V101_PROXY_DIAGNOSTIC_ACTION_BLOCKED",
        "primary_blocker": "M4/runtime blocked until Track T target hygiene and Q2 true-stage admission pass with true current support/observability.",
        "claim": "No runtime or full validation success is claimed.",
    }
    write_json(out / "final_decision.json", decision)
    write_json(out / "summary.json", decision)
    write_rows(out / "gate_checks.csv", [{"gate": key, "pass": value} for key, value in decision.items() if key.endswith("_pass") or key.endswith("_allowed")])
    write_text(out / "failure_report.md", decision["primary_blocker"])
    write_text(out / "what_would_have_to_be_true_to_pass.md", "Track T, true/proxy-appropriate U/V, Q2 true-stage, and M4 simulator must pass before runtime/full validation.")
    write_text(out / "final_report.md", "# ACL2 v101 Final Report\n\n" + json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_text(out / "visual_manifest.csv", "path,description\n")
    return decision


def main() -> None:
    stage0_summary = stage0()
    track_t = target_taxonomy()
    track_u = build_support_rows(track_t)
    track_v = build_observability_rows(track_t)
    track_w = build_roles(track_t)
    track_s2 = build_state_estimator(track_t)
    track_q2 = build_q2(track_t)
    decision = final_decision(stage0_summary, track_t, track_u, track_v, track_w, track_s2, track_q2)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
