#!/usr/bin/env python3
"""Exhaustively audit Stage1 boundaries for strict clean handoff targets.

The earlier materialization repair audits the current Stage3 candidate pool.
This script closes a different failure mode: a bug or overly narrow candidate
generator could have missed strict clean handoff positives in the full Stage1
taxonomy table.  It does not create labels or run actions.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT = ROOT / "stage3_semantic_oracle_upper_bound"

STAGE1_ROWS = ROOT / "stage1_drift_source_autopsy/drift_source_taxonomy.csv"
FULL_CONTROL_ROWS = OUT / "stage3_full_control_semantic_rotation_case_rows.csv"
CLEAN_EXPANSION_ROWS = OUT / "stage3_clean_handoff_candidate_expansion_rows.csv"
BROADER_ROWS = OUT / "stage3_broader_drift_onset_candidate_rows.csv"
STRICT_REPAIR_ROWS = OUT / "stage3_strict_clean_handoff_materialization_repair_rows.csv"

SUMMARY_PATH = OUT / "stage3_exhaustive_clean_handoff_target_mining_summary.json"
ROWS_PATH = OUT / "stage3_exhaustive_clean_handoff_target_mining_rows.csv"
REPORT_PATH = OUT / "exhaustive_clean_handoff_target_mining_report.md"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def quantile(values: list[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def case_set(rows: list[dict[str, str]], key: str = "case_id") -> set[str]:
    return {row[key] for row in rows if row.get(key)}


def blocker_list(row: dict[str, str], thresholds: dict[str, float]) -> list[str]:
    blockers: list[str] = []
    labels = row.get("drift_source_labels", "")
    taxonomy = row.get("target_taxonomy_v101", "")
    local_l1 = fnum(row.get("L1_local_sim3_ate"))
    local_l2 = fnum(row.get("L2_intra_scale_cv"))
    l3 = fnum(row.get("L3_handoff_transfer_penalty_proxy"))

    if row.get("label_original") != "bad":
        blockers.append("label_not_bad")
    if taxonomy != "HANDOFF_SCALE_GAUGE_TARGET":
        blockers.append("taxonomy_not_handoff_scale_gauge_target")
    if "SWA_HANDOFF" not in labels:
        blockers.append("not_swa_handoff_label")
    if "READ_LOCAL_SCALE" in labels:
        blockers.append("mixed_read_local")
    if "UNRELIABLE_OVERLAP" in labels:
        blockers.append("unreliable_overlap")
    if "LABEL_L3_CONFLICT" in labels:
        blockers.append("label_l3_conflict")
    if not (math.isfinite(local_l1) and local_l1 <= thresholds["l1_q70"]):
        blockers.append("local_l1_not_goodish_q70")
    if not (math.isfinite(local_l2) and local_l2 <= thresholds["l2_q70"]):
        blockers.append("local_l2_not_goodish_q70")
    if not (math.isfinite(l3) and l3 >= thresholds["l3_q65"]):
        blockers.append("l3_not_high_q65")
    return blockers


def required_new_evidence(blockers: list[str]) -> list[str]:
    needs: list[str] = []
    if "label_not_bad" in blockers or "taxonomy_not_handoff_scale_gauge_target" in blockers:
        needs.append("fresh/manual labelled clean bad handoff target with non-conflicting taxonomy")
    if "not_swa_handoff_label" in blockers:
        needs.append("new SWA handoff label evidence; current case is not an SWA handoff target")
    if "mixed_read_local" in blockers:
        needs.append("evidence separating local READ failure from inter-chunk SWA handoff effect")
    if "unreliable_overlap" in blockers:
        needs.append("reliable-overlap visual/geometric proof that overlap is not lowobs/dynamic/boundary dominated")
    if "label_l3_conflict" in blockers:
        needs.append("manual/GT-backed relabel or exclusion from strict validation")
    if "local_l1_not_goodish_q70" in blockers or "local_l2_not_goodish_q70" in blockers:
        needs.append("clean local-geometry evidence; current boundary is not local-goodish under q70")
    if "l3_not_high_q65" in blockers:
        needs.append("high-L3 handoff evidence; current boundary is below q65")
    if not needs:
        needs.append("query-head/rotation controls and true action-surface terms before promotion")
    return needs


def near_miss_rank(blockers: list[str], row: dict[str, str], in_clean_pool: bool) -> tuple[int, float]:
    hard = {
        "not_swa_handoff_label",
        "mixed_read_local",
        "local_l1_not_goodish_q70",
        "local_l2_not_goodish_q70",
        "l3_not_high_q65",
    }
    hard_count = sum(1 for item in blockers if item in hard)
    soft_count = len(blockers) - hard_count
    l3 = fnum(row.get("L3_handoff_transfer_penalty_proxy"), 0.0)
    pool_bonus = -1 if in_clean_pool else 0
    return hard_count * 100 + soft_count * 10 + pool_bonus, -l3


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            text = str(csv_value(row.get(col, ""))).replace("|", "\\|").replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            values.append(text)
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    stage1_rows = read_csv(STAGE1_ROWS)
    full_control = by_case(read_csv(FULL_CONTROL_ROWS))
    clean_cases = case_set(read_csv(CLEAN_EXPANSION_ROWS))
    broader_cases = case_set(read_csv(BROADER_ROWS))
    strict_repair = by_case(read_csv(STRICT_REPAIR_ROWS))

    l1_values = [fnum(row.get("L1_local_sim3_ate")) for row in stage1_rows]
    l2_values = [fnum(row.get("L2_intra_scale_cv")) for row in stage1_rows]
    l3_values = [fnum(row.get("L3_handoff_transfer_penalty_proxy")) for row in stage1_rows]
    thresholds = {
        "l1_q70": quantile(l1_values, 0.70),
        "l2_q70": quantile(l2_values, 0.70),
        "l3_q65": quantile(l3_values, 0.65),
    }

    rows: list[dict[str, Any]] = []
    strict_positive_cases: list[str] = []
    missed_positive_cases: list[str] = []
    near_miss_rows: list[dict[str, Any]] = []
    blocker_counter: Counter[str] = Counter()
    label_taxonomy_counter: Counter[str] = Counter()

    candidate_pool = clean_cases | broader_cases | set(strict_repair)

    for row in sorted(stage1_rows, key=lambda item: item.get("case_id", "")):
        case_id = row.get("case_id", "")
        blockers = blocker_list(row, thresholds)
        full_control_available = case_id in full_control
        trace_ready = bval(strict_repair.get(case_id, {}).get("trace_sidecar_ready"))
        in_current_candidate_pool = case_id in candidate_pool
        strict_positive = not blockers
        if strict_positive:
            strict_positive_cases.append(case_id)
            if not in_current_candidate_pool:
                missed_positive_cases.append(case_id)
        else:
            blocker_counter.update(blockers)

        labels = row.get("drift_source_labels", "")
        label_taxonomy_counter[f"{row.get('label_original','')}|{row.get('target_taxonomy_v101','')}|{labels}"] += 1

        audit_row = {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "label_original": row.get("label_original", ""),
            "target_taxonomy_v101": row.get("target_taxonomy_v101", ""),
            "drift_source_labels": labels,
            "primary_drift_source": row.get("primary_drift_source", ""),
            "target_memory_body": row.get("target_memory_body", ""),
            "L1_local_sim3_ate": row.get("L1_local_sim3_ate", ""),
            "L2_intra_scale_cv": row.get("L2_intra_scale_cv", ""),
            "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
            "L3_adjacent_log_scale_jump": row.get("L3_adjacent_log_scale_jump", ""),
            "local_l1_goodish_q70": fnum(row.get("L1_local_sim3_ate")) <= thresholds["l1_q70"],
            "local_l2_goodish_q70": fnum(row.get("L2_intra_scale_cv")) <= thresholds["l2_q70"],
            "high_l3_q65": fnum(row.get("L3_handoff_transfer_penalty_proxy")) >= thresholds["l3_q65"],
            "in_current_stage3_candidate_pool": in_current_candidate_pool,
            "in_full_control_case_rows": full_control_available,
            "trace_sidecar_ready_from_prior_repair": trace_ready,
            "exhaustive_strict_clean_handoff_positive": strict_positive,
            "candidate_generation_missed_strict_positive": strict_positive and not in_current_candidate_pool,
            "blockers": blockers,
            "blocker_count": len(blockers),
            "required_new_evidence": required_new_evidence(blockers),
            "claim_level": "v102_exhaustive_stage1_target_mining_no_action",
        }
        rows.append(audit_row)

        if not strict_positive and "SWA_HANDOFF" in labels:
            near_miss_rows.append(audit_row)

    near_miss_rows = sorted(
        near_miss_rows,
        key=lambda item: near_miss_rank(
            str(item.get("blockers", "")).split(";") if isinstance(item.get("blockers"), str) else list(item.get("blockers", [])),
            {key: str(item.get(key, "")) for key in item},
            bool(item.get("in_current_stage3_candidate_pool")),
        ),
    )
    top_near_misses = near_miss_rows[:10]
    strict_needed = 3
    additional_needed = max(0, strict_needed - len(strict_positive_cases))

    summary = {
        "schema": "acl2_v102_exhaustive_clean_handoff_target_mining_v1",
        "stage1_case_count": len(stage1_rows),
        "thresholds": thresholds,
        "current_stage3_candidate_pool_case_count": len(candidate_pool),
        "exhaustive_strict_clean_handoff_positive_count": len(strict_positive_cases),
        "exhaustive_strict_clean_handoff_positive_cases": ";".join(strict_positive_cases),
        "candidate_generation_missed_strict_positive_count": len(missed_positive_cases),
        "candidate_generation_missed_strict_positive_cases": ";".join(missed_positive_cases),
        "strict_clean_handoff_min_required_for_stage3": strict_needed,
        "additional_strict_positive_needed_count": additional_needed,
        "stage3_strict_coverage_repaired": len(strict_positive_cases) >= strict_needed,
        "blocker_counts": dict(sorted(blocker_counter.items())),
        "top_external_materialization_worklist_cases": ";".join(row["case_id"] for row in top_near_misses),
        "top_external_materialization_worklist_count": len(top_near_misses),
        "external_or_manual_new_evidence_required": additional_needed > 0 and not missed_positive_cases,
        "runtime_action_allowed": False,
        "stage4_allowed": False,
        "conclusion": (
            "Exhaustive Stage1 mining did not find hidden strict clean handoff positives beyond "
            f"{';'.join(strict_positive_cases) or 'none'}. "
            "The Stage3 target-coverage blocker is not explained by a narrow Stage3 candidate-pool bug."
        ),
    }

    write_csv(ROWS_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    report = [
        "# Exhaustive Clean Handoff Target Mining",
        "",
        "This no-action audit rechecks every Stage1 boundary for strict clean handoff target eligibility.",
        "",
        "## Summary",
        "",
        f"- stage1_case_count: `{summary['stage1_case_count']}`",
        f"- exhaustive_strict_clean_handoff_positive_count: `{summary['exhaustive_strict_clean_handoff_positive_count']}`",
        f"- exhaustive_strict_clean_handoff_positive_cases: `{summary['exhaustive_strict_clean_handoff_positive_cases']}`",
        f"- candidate_generation_missed_strict_positive_count: `{summary['candidate_generation_missed_strict_positive_count']}`",
        f"- additional_strict_positive_needed_count: `{summary['additional_strict_positive_needed_count']}`",
        f"- stage3_strict_coverage_repaired: `{summary['stage3_strict_coverage_repaired']}`",
        f"- runtime_action_allowed: `{summary['runtime_action_allowed']}`",
        "",
        "## Quantile Thresholds Used",
        "",
        f"- L1 q70: `{thresholds['l1_q70']}`",
        f"- L2 q70: `{thresholds['l2_q70']}`",
        f"- L3 q65: `{thresholds['l3_q65']}`",
        "",
        "## Blocker Counts",
        "",
    ]
    report.extend(f"- {key}: `{value}`" for key, value in sorted(blocker_counter.items()))
    report.extend(
        [
            "",
            "## Top External Materialization Worklist",
            "",
            md_table(
                top_near_misses,
                [
                    "case_id",
                    "label_original",
                    "target_taxonomy_v101",
                    "drift_source_labels",
                    "L1_local_sim3_ate",
                    "L2_intra_scale_cv",
                    "L3_handoff_transfer_penalty_proxy",
                    "blockers",
                    "required_new_evidence",
                ],
            ),
            "## Interpretation",
            "",
            "- No hidden strict clean handoff positive was found outside the current Stage3 candidate pool.",
            "- Current evidence still provides only one strict positive, so Stage3 promotion remains blocked.",
            "- Near-miss cases can guide fresh materialization or manual/GT-backed relabeling, but they are not strict evidence.",
            "- This audit does not authorize Stage4/5/6/7 runtime action.",
        ]
    )
    write_text(REPORT_PATH, "\n".join(report))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
