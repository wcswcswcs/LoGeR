#!/usr/bin/env python3
"""Build ACL2 v82 SWA adjacent-pair bank v2 from Phase1 stratified evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


DEFAULT_V81S_BANK = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS2_swa_good_bad_pair_bank/swa_good_bad_pair_bank.csv"
)
DEFAULT_PHASE1_PAIR_ROWS = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase1_overlap_quality_stratification/overlap_quality_by_pair.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase2_swa_pair_bank_v2"
)

OUTPUT_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "case_type",
    "base_case_type",
    "quality_type",
    "quality_source",
    "source_path",
    "future_after_overlap",
    "boundary_jump",
    "raw_overlap_residual",
    "overlap_scale_residual",
    "prev_to_curr_scale_jump",
    "prev_local_sim3",
    "curr_local_sim3",
    "high_conf_pair_count",
    "mixed_conf_pair_count",
    "zero_conf_pair_count",
    "high_res_low_conf_pair_count",
    "either_zero_ratio",
    "both_zero_ratio",
    "semantic_confidence_mean",
    "stable_overlap_mass",
    "harm_overlap_mass",
    "context_overlap_mass",
    "READ_used_stable_mass",
    "READ_used_harm_mass",
    "SWA_carried_stable_mass",
    "SWA_carried_harm_mass",
    "V_alignment_delta",
    "K_risk_delta",
    "same_object_overlap_ratio",
    "cross_object_boundary_ratio",
    "RADIO_temporal_stability",
    "has_radio",
    "target_reason",
    "J_mid",
    "artifact_quality_risk",
    "forbidden_as_stable_evidence",
    "carrier_missing_fields",
]

NON_RADIO_CARRIER_FIELDS = [
    "READ_used_stable_mass",
    "READ_used_harm_mass",
    "V_alignment_delta",
    "K_risk_delta",
    "same_object_overlap_ratio",
    "cross_object_boundary_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81s-bank", type=Path, default=DEFAULT_V81S_BANK)
    parser.add_argument("--phase1-pair-rows", type=Path, default=DEFAULT_PHASE1_PAIR_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key, "")
                    for key in fields
                }
            )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("seq", "")).zfill(2), safe_int(row.get("prev_chunk")), safe_int(row.get("curr_chunk")))


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_quality_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(key(row), []).append(row)
    return out


def choose_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    default = [row for row in rows if row.get("source") == "default"]
    minconf0 = [row for row in rows if row.get("source") == "minconf0"]
    for candidate_rows in (default, minconf0):
        for row in candidate_rows:
            if boolish(row.get("high_quality_usable")):
                return row
    for row in minconf0:
        if boolish(row.get("low_conf_stress_usable")):
            return row
    return default[0] if default else rows[0]


def classify_case(base_case: str, quality: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    quality_type = str(quality.get("quality_type", "unknown"))
    harm = safe_float(quality.get("harm_mass_by_bin")) or safe_float(row.get("harm_overlap_mass")) or 0.0
    stable = safe_float(quality.get("stable_overlap_mass")) or safe_float(row.get("stable_overlap_mass")) or 0.0
    if base_case == "bad":
        return "bad_lowconf" if quality_type == "low_conf_stress" else "bad_highconf"
    if quality_type == "low_conf_stress":
        return "good_lowconf_stress"
    if harm >= max(0.25, stable):
        return "false_positive_semantic"
    return "good_highconf"


def get_json_number_map_value(value: Any, key_name: str) -> float | None:
    if not isinstance(value, str) or not value.startswith("{"):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return safe_float(parsed.get(key_name))


def carrier_missing_fields(row: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in NON_RADIO_CARRIER_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            missing.append(field)
    return missing


def build_rows(bank_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality_index = build_quality_index(quality_rows)
    out: list[dict[str, Any]] = []
    for row in bank_rows:
        k = key(row)
        quality = choose_quality(quality_index.get(k, []))
        base_case = str(row.get("case_type", "unknown"))
        chosen_case = classify_case(base_case, quality, row)
        stable_by_bin = quality.get("stable_mass_by_bin")
        harm_by_bin = quality.get("harm_mass_by_bin")
        context_by_bin = quality.get("context_mass_by_bin")
        new_row: dict[str, Any] = {
            "seq": k[0],
            "prev_chunk": k[1],
            "curr_chunk": k[2],
            "case_type": chosen_case,
            "base_case_type": base_case,
            "quality_type": quality.get("quality_type", row.get("quality_type", "")),
            "quality_source": quality.get("source", ""),
            "source_path": quality.get("source_path", row.get("overlap_pair_file", "")),
            "future_after_overlap": row.get("future_after_overlap"),
            "boundary_jump": row.get("boundary_jump"),
            "raw_overlap_residual": row.get("raw_overlap_residual"),
            "overlap_scale_residual": quality.get("overlap_scale_residual", row.get("overlap_scale_residual")),
            "prev_to_curr_scale_jump": row.get("prev_to_curr_scale_jump"),
            "prev_local_sim3": row.get("prev_local_sim3"),
            "curr_local_sim3": row.get("curr_local_sim3"),
            "high_conf_pair_count": quality.get("high_conf_pair_count"),
            "mixed_conf_pair_count": quality.get("mixed_conf_pair_count"),
            "zero_conf_pair_count": quality.get("zero_conf_pair_count"),
            "high_res_low_conf_pair_count": quality.get("high_res_low_conf_pair_count"),
            "either_zero_ratio": quality.get("either_zero_ratio", row.get("either_zero_geometry_conf_ratio")),
            "both_zero_ratio": quality.get("both_zero_ratio", row.get("both_zero_geometry_conf_ratio")),
            "semantic_confidence_mean": quality.get("semantic_confidence_mean"),
            "stable_overlap_mass": get_json_number_map_value(stable_by_bin, "B0_high") or row.get("stable_overlap_mass"),
            "harm_overlap_mass": get_json_number_map_value(harm_by_bin, "B0_high") or row.get("harm_overlap_mass"),
            "context_overlap_mass": get_json_number_map_value(context_by_bin, "B0_high") or row.get("context_overlap_mass"),
            "READ_used_stable_mass": row.get("READ_used_stable_mass"),
            "READ_used_harm_mass": row.get("READ_used_harm_mass", ""),
            "SWA_carried_stable_mass": row.get("SWA_carried_stable_mass"),
            "SWA_carried_harm_mass": row.get("SWA_carried_harm_mass"),
            "V_alignment_delta": row.get("V_alignment_delta"),
            "K_risk_delta": row.get("K_risk_delta"),
            "same_object_overlap_ratio": row.get("same_object_overlap_ratio"),
            "cross_object_boundary_ratio": row.get("cross_object_boundary_ratio"),
            "RADIO_temporal_stability": row.get("RADIO_temporal_stability"),
            "has_radio": boolish(row.get("has_radio")),
            "target_reason": row.get("target_reason"),
            "J_mid": row.get("J_mid"),
            "artifact_quality_risk": boolish(row.get("artifact_quality_risk")) or boolish(quality.get("forbidden_as_stable_evidence")),
            "forbidden_as_stable_evidence": boolish(quality.get("forbidden_as_stable_evidence")),
        }
        missing = carrier_missing_fields(new_row)
        new_row["carrier_missing_fields"] = ";".join(missing)
        out.append(new_row)
    return out


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    case_counts = Counter(str(row.get("case_type")) for row in rows)
    base_counts = Counter(str(row.get("base_case_type")) for row in rows)
    quality_counts = Counter(str(row.get("quality_type")) for row in rows)
    seq_coverage = sorted({str(row.get("seq")) for row in rows})
    bad_rows = [row for row in rows if str(row.get("base_case_type")) == "bad"]
    good_rows = [row for row in rows if str(row.get("base_case_type")) == "good"]
    carrier_total = len(rows) * len(NON_RADIO_CARRIER_FIELDS)
    carrier_present = 0
    missing_counts: Counter[str] = Counter()
    for row in rows:
        missing = set(str(row.get("carrier_missing_fields", "")).split(";")) if row.get("carrier_missing_fields") else set()
        for field in NON_RADIO_CARRIER_FIELDS:
            if field in missing:
                missing_counts[field] += 1
            else:
                carrier_present += 1
    carrier_completeness = carrier_present / carrier_total if carrier_total else 0.0
    j_values = [safe_float(row.get("J_mid")) for row in rows if safe_float(row.get("J_mid")) is not None]
    gate_checks = {
        "bad_total_ge_12": len(bad_rows) >= 12,
        "good_or_false_positive_total_ge_12": len(good_rows) >= 12,
        "coverage_ge_3": len(seq_coverage) >= 3,
        "bad_highconf_represented": case_counts.get("bad_highconf", 0) > 0,
        "bad_lowconf_represented_if_available": case_counts.get("bad_lowconf", 0) > 0,
        "good_highconf_represented": case_counts.get("good_highconf", 0) > 0,
        "carrier_fields_completeness_ge_70pct": carrier_completeness >= 0.70,
        "radio_missing_marked_has_radio_false": all((row.get("RADIO_temporal_stability") not in ("", None)) or not boolish(row.get("has_radio")) for row in rows),
    }
    gate_pass = all(gate_checks.values())
    if not gate_checks["carrier_fields_completeness_ge_70pct"]:
        blocker = "carrier_fields_incomplete_run_phase3_direct_qkv_read_route_dump_before_action"
    else:
        blocker = ""
    return {
        "schema": "acl2_v82_swa_pair_bank_v2_summary_v1",
        "out_dir": str(out_dir),
        "rows": len(rows),
        "base_case_counts": dict(base_counts),
        "case_counts": dict(case_counts),
        "quality_counts": dict(quality_counts),
        "seq_coverage": seq_coverage,
        "bad_total": len(bad_rows),
        "good_or_false_positive_total": len(good_rows),
        "carrier_fields": NON_RADIO_CARRIER_FIELDS,
        "carrier_field_completeness": carrier_completeness,
        "carrier_missing_field_counts": dict(missing_counts),
        "j_mid_median": median(j_values) if j_values else None,
        "gate_checks": gate_checks,
        "phase2_gate_pass": gate_pass,
        "gate_blocker": blocker,
        "phase3_required_before_action": bool(blocker),
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# v82 SWA Pair Bank v2 Summary",
        "",
        f"- rows: `{summary.get('rows')}`",
        f"- phase2_gate_pass: `{summary.get('phase2_gate_pass')}`",
        f"- gate_blocker: `{summary.get('gate_blocker')}`",
        f"- seq_coverage: `{summary.get('seq_coverage')}`",
        f"- case_counts: `{summary.get('case_counts')}`",
        f"- quality_counts: `{summary.get('quality_counts')}`",
        f"- carrier_field_completeness: `{summary.get('carrier_field_completeness')}`",
        f"- carrier_missing_field_counts: `{summary.get('carrier_missing_field_counts')}`",
        "",
        "Interpretation:",
        "",
        "The v2 bank is usable as a confidence-stratified adjacent-pair case bank. It is not yet usable for SWA action because READ/SWA carrier fields are still missing. Per the v82 plan, Phase3 must dump true route/QKV/READ evidence before any action.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bank_rows = read_csv(args.v81s_bank)
    quality_rows = read_csv(args.phase1_pair_rows)
    rows = build_rows(bank_rows, quality_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "swa_pair_bank_v2.csv", rows, OUTPUT_FIELDS)
    summary = summarize(rows, args.out_dir)
    write_json(args.out_dir / "swa_pair_bank_v2_summary.json", summary)
    write_report(args.out_dir / "swa_pair_bank_v2_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
