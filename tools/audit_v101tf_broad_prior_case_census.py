#!/usr/bin/env python3
"""Audit broad prior ACL2 case universes for v101 Track T extension.

The original v101 extension audit focused on v94/v95 case banks.  This script
widens the read-only census across prior ACL2 v94-v100 artifacts, using only
small case-level CSV tables.  It classifies candidate cases with the v101 Track
T thresholds and records whether they already have v100 same-space traces and
per-anchor geometry.  It does not promote any case to action.
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
OUT = ROOT / "trackT_drift_target_relabel"
V100_ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V100_CASE_ROWS = V100_ROOT / "trackD4_read_current_support_provider/label_l3_hygiene_provenance_rows.csv"
V100_SAME_SPACE_ROWS = V100_ROOT / "trackS_same_space_latent_state/same_space_anchor_rows.csv"
V100_GEOMETRY_EDGE_ROWS = V100_ROOT / "trackL2_anchor_scale_observability/geometry_edge_rows.csv"

VERSION_KEYS = ["acl2_v94", "acl2_v95", "acl2_v96", "acl2_v97", "acl2_v98", "acl2_v99", "acl2_v100"]
MAX_CASE_TABLE_ROWS = 1200


def is_allowed_case_universe_source(path: Path) -> bool:
    """Keep comparable case-level universes; exclude feature/action-pilot tables."""
    path_str = str(path)
    name = path.name
    parent = path.parent.name
    allowed_names = {
        "canonical_case_rows.csv",
        "good_controls.csv",
        "boundary_failure_rows.csv",
        "case_universe_rows.csv",
        "good_control_hygiene_rows.csv",
        "observability_rows.csv",
        "trace_semantic_anchor_rows.csv",
        "per_case_metrics.csv",
        "identity_case_rows.csv",
        "graph_case_rows.csv",
        "freshness_case_rows.csv",
        "label_l3_hygiene_provenance_rows.csv",
        "missed_positive_l3_case_rows.csv",
        "case_rows.csv",
        "semantic_evidence_rows.csv",
        "semantic_evidence_by_boundary.csv",
        "balanced_probe_set_rows.csv",
    }
    if name in allowed_names:
        return True
    if parent == "metric_suite" and name in {"rows.csv", "metric_rows.csv"}:
        return True
    if name == "rows.csv" and any(
        token in path_str
        for token in [
            "trackA_case_response_atlas",
            "trackI_scale_gauge_evidence_observatory",
            "trackD4_read_current_support_provider",
            "trackC4_identity_latent_gauge_ruler",
            "trackF4_ttt_write_to_use_same_space",
            "trackL2_anchor_scale_observability",
            "trackN2_anchor_identity_graph",
            "trackO2_freshness_current_support",
            "trackQ_chunk_update_admission",
        ]
    ):
        return True
    if parent in {
        "trackR_edge_head_control_audit",
        "trackR2_anchor_edge_identity_control_audit",
    } and name == "case_rows.csv":
        return True
    return False


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
            clean = {}
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


def median(values: list[Any]) -> float:
    vals = sorted(f(value) for value in values if math.isfinite(f(value)))
    return statistics.median(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(value) for value in values if math.isfinite(f(value)))
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
    vals = [f(value) for value in values if math.isfinite(f(value))]
    if not vals:
        return math.nan
    med = statistics.median(vals)
    return statistics.median(abs(value - med) for value in vals)


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label == "good":
        return "good"
    if label in {"bad", "non_good", "nongood", "non-good"}:
        return "non_good"
    if label == "unlabelled_support":
        return "unlabelled_support"
    if not label:
        return "missing"
    return label


def seq_from_case(case_id: str) -> str:
    return case_id.split("_", 1)[0] if "_" in case_id else ""


def source_version(path: Path) -> str:
    for part in path.parts:
        if part.startswith("acl2_v"):
            return part
    return ""


def pick_first(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""


def case_id_from_row(row: dict[str, str]) -> str:
    return pick_first(row, ["case_id", "pair_id"])


def l3_from_row(row: dict[str, str]) -> str:
    return pick_first(
        row,
        [
            "L3_handoff_transfer_penalty_proxy",
            "atlas_L3_handoff_transfer_penalty_proxy",
            "stage7e_L3",
            "v100_L3",
            "v99_L3",
            "v98_L3",
            "L3_J_handoff",
            "atlas_L3_J_handoff",
            "future_after_overlap",
        ],
    )


def failure_from_row(row: dict[str, str]) -> str:
    parts = [
        row.get("failure_type", ""),
        row.get("failure_type_primary", ""),
        row.get("failure_type_secondary", ""),
        row.get("action_response_label", ""),
        row.get("action_response_labels", ""),
        row.get("bucket", ""),
    ]
    return ";".join(part for part in parts if part)


def label_from_row(row: dict[str, str]) -> str:
    return pick_first([""], []) if False else pick_first(row, ["case_label", "case_label_offline_only", "raw_label", "v100_case_label", "v99_case_label", "v98_case_label"])


def count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            return sum(1 for _ in reader)
    except Exception:
        return -1


def discover_case_tables() -> tuple[list[Path], list[dict[str, Any]]]:
    roots = [path for path in Path("results").iterdir() if path.is_dir() and any(key in path.name for key in VERSION_KEYS)]
    used: list[Path] = []
    audit: list[dict[str, Any]] = []
    for root in sorted(roots):
        for path in sorted(root.rglob("*.csv")):
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader)
            except Exception as exc:  # noqa: BLE001
                audit.append({"source_path": str(path), "used": False, "skip_reason": f"read_error:{type(exc).__name__}"})
                continue
            header_set = set(header)
            row_count = count_rows(path)
            has_case = bool({"case_id", "pair_id"} & header_set)
            has_l3 = bool(
                {
                    "L3_handoff_transfer_penalty_proxy",
                    "atlas_L3_handoff_transfer_penalty_proxy",
                    "stage7e_L3",
                    "v100_L3",
                    "v99_L3",
                    "v98_L3",
                    "L3_J_handoff",
                    "atlas_L3_J_handoff",
                    "future_after_overlap",
                }
                & header_set
            )
            has_label_or_failure = bool(
                {
                    "case_label",
                    "case_label_offline_only",
                    "raw_label",
                    "failure_type",
                    "failure_type_primary",
                    "failure_type_secondary",
                    "action_response_label",
                    "action_response_labels",
                    "bucket",
                }
                & header_set
            )
            likely_edge_table = row_count > MAX_CASE_TABLE_ROWS and any(
                name in header_set
                for name in ["anchor_id", "head_idx", "layer", "swa_layer_idx", "trace_payload", "label_id"]
            )
            allowed_source = is_allowed_case_universe_source(path)
            used_flag = (
                has_case
                and has_l3
                and has_label_or_failure
                and row_count >= 0
                and row_count <= MAX_CASE_TABLE_ROWS
                and not likely_edge_table
                and allowed_source
            )
            reason = "used" if used_flag else "not_case_level_or_too_large"
            if not has_case:
                reason = "missing_case_id"
            elif not has_l3:
                reason = "missing_L3_field"
            elif not has_label_or_failure:
                reason = "missing_label_or_failure_field"
            elif row_count > MAX_CASE_TABLE_ROWS or likely_edge_table:
                reason = "excluded_large_edge_or_anchor_table"
            elif not allowed_source:
                reason = "excluded_non_universe_feature_or_action_table"
            audit.append(
                {
                    "source_path": str(path),
                    "version_root": source_version(path),
                    "row_count": row_count,
                    "used": used_flag,
                    "skip_reason": reason,
                    "allowed_case_universe_source": allowed_source,
                    "header_preview": ",".join(header[:24]),
                }
            )
            if used_flag:
                used.append(path)
    return used, audit


def classify(row: dict[str, Any], l3_high: float, l3_low: float) -> tuple[str, str]:
    label = str(row.get("label_norm", ""))
    failure = str(row.get("failure_type", "")).upper()
    l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
    l3_high_pass = math.isfinite(l3) and math.isfinite(l3_high) and l3 >= l3_high
    l3_low_pass = math.isfinite(l3) and math.isfinite(l3_low) and l3 <= l3_low
    handoff = ("HANDOFF_SCALE" in failure) or ("HANDOFF_GAUGE" in failure)
    lowobs_or_multimode = ("LOW_OBSERVABILITY" in failure) or ("MULTIMODE_CONFLICT" in failure) or ("SEM_MULTIMODE" in failure)
    hygiene_excluded = b(row.get("hygiene_excluded_good_control")) or str(row.get("good_control_hygiene_status", "")).lower() in {"hygiene_excluded", "excluded"}
    if label == "good" and (hygiene_excluded or l3_high_pass):
        return "GOOD_HIGH_L3_CONTAMINATED", "good_label_high_L3_or_hygiene_excluded"
    if lowobs_or_multimode and l3_high_pass:
        return "MULTIMODE_LOWOBS_ABSTAIN", "high_L3_lowobs_or_multimode"
    if label == "non_good" and handoff and l3_high_pass:
        return "HANDOFF_SCALE_GAUGE_TARGET", "non_good_handoff_high_L3"
    if label != "good" and l3_low_pass:
        return "LOCAL_BAD_NOT_HANDOFF", "non_good_or_unlabelled_low_L3"
    if label == "good" and l3_low_pass and not hygiene_excluded:
        return "SAFE_GOOD", "good_low_L3_not_hygiene_excluded"
    return "AMBIGUOUS_SUPPORT", "criteria_not_clean"


def main() -> None:
    v100_rows = read_rows(V100_CASE_ROWS)
    v100_case_set = {row.get("case_id", "") for row in v100_rows}
    same_space_cases = {
        row.get("case_id", "")
        for row in read_rows(V100_SAME_SPACE_ROWS)
        if row.get("canonical_space_name") == "S-B_preprojection_hidden"
    }
    geometry_cases = {row.get("case_id", "") for row in read_rows(V100_GEOMETRY_EDGE_ROWS)}
    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows if normalize_label(row.get("case_label")) == "good"]
    l3_high = max(median(good_l3) + 2.0 * mad(good_l3), quantile(l3_all, 0.75))
    l3_low = quantile(l3_all, 0.40)

    source_paths, source_audit = discover_case_tables()
    source_rows: list[dict[str, Any]] = []
    for path in source_paths:
        for raw in read_rows(path):
            case_id = case_id_from_row(raw)
            if not case_id:
                continue
            label_norm = normalize_label(label_from_row(raw))
            failure = failure_from_row(raw)
            l3 = l3_from_row(raw)
            prior_trace_available = bool(
                raw.get("trace_path")
                or raw.get("trace_payload")
                or raw.get("payload_count")
                or "trace" in path.name.lower()
                or "trace" in str(path.parent).lower()
            )
            prior_trace_hint = pick_first(raw, ["trace_path", "trace_payload", "trace_provenance", "payload_count", "source_artifact", "extension_sources"])
            row = {
                "source_path": str(path),
                "version_root": source_version(path),
                "case_id": case_id,
                "seq": pick_first(raw, ["seq"]) or seq_from_case(case_id),
                "prev_chunk": pick_first(raw, ["prev_chunk"]),
                "curr_chunk": pick_first(raw, ["curr_chunk"]),
                "label_raw": label_from_row(raw),
                "label_norm": label_norm,
                "failure_type": failure,
                "L3_handoff_transfer_penalty_proxy": l3,
                "good_control_hygiene_status": raw.get("good_control_hygiene_status", ""),
                "hygiene_excluded_good_control": raw.get("v98_hygiene_excluded_good_control", ""),
                "prior_trace_available": prior_trace_available,
                "prior_trace_hint": prior_trace_hint,
                "already_in_v100_28_case_universe": case_id in v100_case_set,
                "has_v100_same_space_trace": case_id in same_space_cases,
                "has_v100_per_anchor_geometry": case_id in geometry_cases,
            }
            taxonomy, reason = classify(row, l3_high, l3_low)
            failure_upper = failure.upper()
            row.update(
                {
                    "target_taxonomy_under_v101_rules": taxonomy,
                    "target_reason": reason,
                    "L3_high_threshold": l3_high,
                    "L3_low_threshold": l3_low,
                    "is_L3_high": math.isfinite(f(l3)) and f(l3) >= l3_high,
                    "is_L3_low": math.isfinite(f(l3)) and f(l3) <= l3_low,
                    "handoff_failure_mode": ("HANDOFF_SCALE" in failure_upper) or ("HANDOFF_GAUGE" in failure_upper),
                    "lowobs_or_multimode": ("LOW_OBSERVABILITY" in failure_upper) or ("MULTIMODE_CONFLICT" in failure_upper) or ("SEM_MULTIMODE" in failure_upper),
                    "clean_candidate": taxonomy in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"},
                    "usable_now_for_v101_action": taxonomy in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}
                    and case_id in same_space_cases
                    and case_id in geometry_cases,
                    "requires_v100_schema_trace_materialization": taxonomy in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}
                    and case_id not in same_space_cases,
                }
            )
            source_rows.append(row)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_case[str(row["case_id"])].append(row)

    unique_rows: list[dict[str, Any]] = []
    for case_id, rows in sorted(by_case.items()):
        clean_rows = [row for row in rows if row["clean_candidate"]]
        usable_rows = [row for row in rows if row["usable_now_for_v101_action"]]
        trace_need_rows = [row for row in rows if row["requires_v100_schema_trace_materialization"]]
        tax_counts = Counter(row["target_taxonomy_under_v101_rules"] for row in rows)
        labels = Counter(row["label_norm"] for row in rows)
        l3_values = [f(row["L3_handoff_transfer_penalty_proxy"]) for row in rows if math.isfinite(f(row["L3_handoff_transfer_penalty_proxy"]))]
        representative = clean_rows[0] if clean_rows else rows[0]
        unique_rows.append(
            {
                "case_id": case_id,
                "seq": representative.get("seq", seq_from_case(case_id)),
                "source_count": len({row["source_path"] for row in rows}),
                "version_roots": ";".join(sorted({row["version_root"] for row in rows if row["version_root"]})),
                "label_norm_counts": dict(labels),
                "taxonomy_counts": dict(tax_counts),
                "representative_taxonomy": representative.get("target_taxonomy_under_v101_rules", ""),
                "representative_reason": representative.get("target_reason", ""),
                "L3_min": min(l3_values) if l3_values else "",
                "L3_max": max(l3_values) if l3_values else "",
                "L3_values_unique": ";".join(str(value) for value in sorted(set(l3_values))),
                "failure_types": ";".join(sorted({str(row["failure_type"]) for row in rows if row["failure_type"]})),
                "already_in_v100_28_case_universe": case_id in v100_case_set,
                "has_v100_same_space_trace": case_id in same_space_cases,
                "has_v100_per_anchor_geometry": case_id in geometry_cases,
                "prior_trace_available_any": any(row["prior_trace_available"] for row in rows),
                "clean_candidate_any": bool(clean_rows),
                "usable_now_for_v101_action_any": bool(usable_rows),
                "requires_v100_schema_trace_materialization_any": bool(trace_need_rows),
                "clean_candidate_sources": ";".join(sorted({row["source_path"] for row in clean_rows})),
            }
        )

    clean_candidates = [row for row in source_rows if row["clean_candidate"]]
    clean_unique = [row for row in unique_rows if row["clean_candidate_any"]]
    new_clean_missing_trace = [
        row
        for row in unique_rows
        if row["clean_candidate_any"] and not row["already_in_v100_28_case_universe"] and not row["has_v100_same_space_trace"]
    ]
    new_clean_with_prior_trace = [row for row in new_clean_missing_trace if row["prior_trace_available_any"]]
    usable_now = [row for row in unique_rows if row["usable_now_for_v101_action_any"]]
    usable_new_now = [row for row in usable_now if not row["already_in_v100_28_case_universe"]]
    handoff_clean_unique = [row for row in clean_unique if "HANDOFF_SCALE_GAUGE_TARGET" in str(row["taxonomy_counts"])]
    safe_clean_unique = [row for row in clean_unique if "SAFE_GOOD" in str(row["taxonomy_counts"])]
    skip_counts = Counter(row.get("skip_reason", "") for row in source_audit)
    recheck_01005006 = next((row for row in unique_rows if row["case_id"] == "01_005_006"), {})

    summary = {
        "schema": "acl2_v101_trackT_broad_prior_case_census_v1",
        "source_file_count_used": len(source_paths),
        "source_file_count_scanned": len(source_audit),
        "source_skip_reason_counts": dict(skip_counts),
        "source_row_count": len(source_rows),
        "unique_case_count": len(unique_rows),
        "v100_case_count": len(v100_case_set),
        "v100_same_space_case_count": len(same_space_cases),
        "v100_geometry_case_count": len(geometry_cases),
        "L3_high_threshold": l3_high,
        "L3_low_threshold": l3_low,
        "clean_candidate_unique_count": len(clean_unique),
        "clean_handoff_candidate_unique_count": len(handoff_clean_unique),
        "clean_safe_good_candidate_unique_count": len(safe_clean_unique),
        "usable_now_for_v101_action_unique_count": len(usable_now),
        "usable_now_new_case_count": len(usable_new_now),
        "usable_now_new_cases": ";".join(sorted(row["case_id"] for row in usable_new_now)),
        "new_clean_candidate_missing_v100_trace_count": len(new_clean_missing_trace),
        "new_clean_candidate_missing_v100_trace_cases": ";".join(sorted(row["case_id"] for row in new_clean_missing_trace)),
        "new_clean_candidate_with_prior_trace_count": len(new_clean_with_prior_trace),
        "new_clean_candidate_with_prior_trace_cases": ";".join(sorted(row["case_id"] for row in new_clean_with_prior_trace)),
        "trace_materialization_recommended": bool(new_clean_missing_trace),
        "case_01_005_006_recheck": recheck_01005006,
        "runtime_action_allowed": False,
        "claim": "Broad prior-case census is diagnostic. Cases without v100 same-space/per-anchor geometry are not usable for v101 action.",
    }

    write_rows(OUT / "broad_prior_case_source_file_audit.csv", source_audit)
    write_rows(OUT / "broad_prior_case_source_rows.csv", source_rows)
    write_rows(OUT / "broad_prior_unique_case_summary.csv", unique_rows)
    write_rows(OUT / "broad_prior_clean_candidate_rows.csv", clean_candidates)
    write_rows(OUT / "broad_prior_trace_materialization_recommendations.csv", new_clean_missing_trace)
    write_json(OUT / "broad_prior_case_census_summary.json", summary)
    write_text(
        OUT / "broad_prior_case_census_report.md",
        "# Track T Broad Prior-Case Census\n\n"
        f"- Source files scanned: {summary['source_file_count_scanned']}\n"
        f"- Source files used: {summary['source_file_count_used']}\n"
        f"- Excluded non-universe feature/action tables: {skip_counts.get('excluded_non_universe_feature_or_action_table', 0)}\n"
        f"- Source rows: {summary['source_row_count']}\n"
        f"- Unique cases: {summary['unique_case_count']}\n"
        f"- Clean candidate unique cases: {summary['clean_candidate_unique_count']}\n"
        f"- Usable-now new cases: {summary['usable_now_new_case_count']} ({summary['usable_now_new_cases']})\n"
        f"- New clean candidates missing v100 trace: {summary['new_clean_candidate_missing_v100_trace_count']}\n"
        f"- New clean candidates with prior trace only: {summary['new_clean_candidate_with_prior_trace_count']}\n"
        f"- Runtime action allowed: {summary['runtime_action_allowed']}\n\n"
        "Source-filter note: READ cue, patch tensor, action-pilot, and other feature/action tables are excluded from target extension because their L3-like fields are not comparable case-universe L3 handoff penalties.\n\n"
        f"`01_005_006` recheck: `{json.dumps(recheck_01005006, ensure_ascii=False, sort_keys=True)}`\n\n"
        "Conclusion: this census can recommend future v100-schema trace materialization, but it cannot extend the current action-ready v101 target universe by itself.\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
