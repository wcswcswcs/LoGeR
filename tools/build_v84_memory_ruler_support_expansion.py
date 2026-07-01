#!/usr/bin/env python3
"""Build ACL2 v84 Phase10 support-expansion pair bank.

The expansion keeps the labelled v82 24-row pair bank intact, then adds
observable adjacent overlap rows from the v82 overlap-quality stratification as
unlabelled support. Unlabelled rows are never treated as good cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from build_v84_ruler_candidate_universe import DEFAULT_DIRECT_ROOTS, discover_direct_paths, safe_int, seq_norm


DEFAULT_BASE_PAIR_BANK = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
)
DEFAULT_OVERLAP_QUALITY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase1_overlap_quality_stratification/overlap_quality_by_pair.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pair-bank", type=Path, default=DEFAULT_BASE_PAIR_BANK)
    parser.add_argument("--overlap-quality", type=Path, default=DEFAULT_OVERLAP_QUALITY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--direct-root", type=Path, action="append", default=None)
    parser.add_argument("--include-lowconf-stress", action="store_true")
    parser.add_argument("--max-extra-high-quality", type=int, default=999999)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def int_or_default(value: Any, default: int = -1) -> int:
    out = safe_int(value)
    return int(out) if out is not None else int(default)


def key(row: Mapping[str, Any]) -> tuple[str, int, int, str, str]:
    return (
        seq_norm(row.get("seq")),
        int_or_default(row.get("prev_chunk")),
        int_or_default(row.get("curr_chunk")),
        str(row.get("source") or row.get("quality_source") or ""),
        str(row.get("quality_type") or ""),
    )


def pair_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return seq_norm(row.get("seq")), int_or_default(row.get("prev_chunk")), int_or_default(row.get("curr_chunk"))


def overlap_to_support_row(row: Mapping[str, str], *, case_type: str, base_case_type: str) -> dict[str, Any]:
    quality_source = str(row.get("source") or "")
    return {
        "seq": seq_norm(row.get("seq")),
        "prev_chunk": safe_int(row.get("prev_chunk")),
        "curr_chunk": safe_int(row.get("curr_chunk")),
        "case_type": case_type,
        "base_case_type": base_case_type,
        "quality_type": row.get("quality_type", ""),
        "quality_source": quality_source,
        "source_path": row.get("source_path", ""),
        "future_after_overlap": "",
        "boundary_jump": "",
        "raw_overlap_residual": row.get("raw_residual_mean", ""),
        "overlap_scale_residual": row.get("overlap_scale_residual", ""),
        "prev_to_curr_scale_jump": "",
        "prev_local_sim3": "",
        "curr_local_sim3": "",
        "high_conf_pair_count": row.get("high_conf_pair_count", ""),
        "mixed_conf_pair_count": row.get("mixed_conf_pair_count", ""),
        "zero_conf_pair_count": row.get("zero_conf_pair_count", ""),
        "high_res_low_conf_pair_count": row.get("high_res_low_conf_pair_count", ""),
        "either_zero_ratio": row.get("either_zero_ratio", ""),
        "both_zero_ratio": row.get("both_zero_ratio", ""),
        "semantic_confidence_mean": row.get("semantic_confidence_mean", ""),
        "stable_overlap_mass": "",
        "harm_overlap_mass": "",
        "context_overlap_mass": "",
        "READ_used_stable_mass": "",
        "READ_used_harm_mass": "",
        "SWA_carried_stable_mass": "",
        "SWA_carried_harm_mass": "",
        "V_alignment_delta": "",
        "K_risk_delta": "",
        "same_object_overlap_ratio": "",
        "cross_object_boundary_ratio": "",
        "RADIO_temporal_stability": "",
        "has_radio": "False",
        "target_reason": (
            "phase10_unlabelled_support_expansion;"
            f"source={quality_source};quality={row.get('quality_type', '')};"
            "not_used_as_bad_good_label"
        ),
        "J_mid": "",
        "artifact_quality_risk": as_bool(row.get("forbidden_as_stable_evidence")),
        "forbidden_as_stable_evidence": row.get("forbidden_as_stable_evidence", ""),
        "carrier_missing_fields": "bad_good_label;future_after_overlap;boundary_jump;READ/SWA route true mass",
        "support_expansion_label_scope": "unlabelled_support_not_good",
        "seq01_sparse_support_flag": row.get("seq01_sparse_support_flag", ""),
        "high_quality_usable": row.get("high_quality_usable", ""),
        "low_conf_stress_usable": row.get("low_conf_stress_usable", ""),
    }


def main() -> None:
    args = parse_args()
    direct_roots = args.direct_root if args.direct_root is not None else DEFAULT_DIRECT_ROOTS
    read_paths, pca_paths = discover_direct_paths(direct_roots)
    base_rows = read_csv(args.base_pair_bank)
    overlap_rows = read_csv(args.overlap_quality)

    base_pair_keys = {pair_key(row) for row in base_rows}
    expanded_rows: list[dict[str, Any]] = []
    for row in base_rows:
        copied = dict(row)
        copied["support_expansion_label_scope"] = "labelled_v82_main_pair"
        copied["seq01_sparse_support_flag"] = ""
        copied["high_quality_usable"] = copied.get("quality_type") == "high_quality"
        copied["low_conf_stress_usable"] = copied.get("quality_type") == "low_conf_stress"
        expanded_rows.append(copied)

    observed_extra: list[dict[str, str]] = []
    missing_rows: list[dict[str, Any]] = []
    direct_availability_rows: list[dict[str, Any]] = []
    seen_extra_keys: set[tuple[str, int, int, str, str]] = set()

    for row in overlap_rows:
        seq = seq_norm(row.get("seq"))
        prev_chunk = int_or_default(row.get("prev_chunk"))
        curr_chunk = int_or_default(row.get("curr_chunk"))
        source = str(row.get("source") or "")
        quality_type = str(row.get("quality_type") or "")
        pair = (seq, prev_chunk, curr_chunk)
        read_available = bool(read_paths.get((seq, curr_chunk)))
        swa_available = bool(pca_paths.get((seq, curr_chunk)))
        source_path = Path(str(row.get("source_path") or ""))
        source_exists = source_path.is_file()
        direct_availability_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "source": source,
                "quality_type": quality_type,
                "read_available": read_available,
                "swa_available": swa_available,
                "source_exists": source_exists,
                "in_base_pair_bank": pair in base_pair_keys,
                "seq01_sparse_support_flag": row.get("seq01_sparse_support_flag", ""),
                "forbidden_as_stable_evidence": row.get("forbidden_as_stable_evidence", ""),
            }
        )

        high_quality_ok = (
            source == "default"
            and quality_type == "high_quality"
            and as_bool(row.get("high_quality_usable"))
            and source_exists
        )
        lowconf_ok = (
            args.include_lowconf_stress
            and source == "minconf0"
            and quality_type == "low_conf_stress"
            and as_bool(row.get("low_conf_stress_usable"))
            and source_exists
        )
        if not high_quality_ok and not lowconf_ok:
            continue
        if pair in base_pair_keys and high_quality_ok:
            continue
        if not (read_available and swa_available):
            missing_rows.append(
                {
                    "seq": seq,
                    "prev_chunk": prev_chunk,
                    "curr_chunk": curr_chunk,
                    "source": source,
                    "quality_type": quality_type,
                    "read_available": read_available,
                    "swa_available": swa_available,
                    "source_exists": source_exists,
                    "recommended_direct_chunk": curr_chunk,
                    "eligible_high_quality": high_quality_ok,
                    "eligible_lowconf_stress": lowconf_ok,
                    "seq01_sparse_support_flag": row.get("seq01_sparse_support_flag", ""),
                    "forbidden_as_stable_evidence": row.get("forbidden_as_stable_evidence", ""),
                }
            )
            continue
        extra_key = key(row)
        if extra_key in seen_extra_keys:
            continue
        if high_quality_ok and len(observed_extra) >= args.max_extra_high_quality:
            continue
        seen_extra_keys.add(extra_key)
        observed_extra.append(row)
        if high_quality_ok:
            case_type = "support_unlabelled_highconf"
            if seq == "01" and as_bool(row.get("seq01_sparse_support_flag")):
                case_type = "support_unlabelled_seq01_sparse_highconf"
            expanded_rows.append(overlap_to_support_row(row, case_type=case_type, base_case_type="unlabelled_support"))
        elif lowconf_ok:
            expanded_rows.append(
                overlap_to_support_row(row, case_type="support_unlabelled_lowconf_stress", base_case_type="stress_unlabelled")
            )

    missing_by_seq: dict[str, list[int]] = defaultdict(list)
    for row in missing_rows:
        if row.get("eligible_high_quality"):
            chunk = int(row["recommended_direct_chunk"])
            seq = str(row["seq"])
            if chunk not in missing_by_seq[seq]:
                missing_by_seq[seq].append(chunk)
    missing_by_seq = {seq: sorted(chunks) for seq, chunks in sorted(missing_by_seq.items())}

    rows_by_scope = Counter(row.get("support_expansion_label_scope", "") for row in expanded_rows)
    rows_by_case = Counter(row.get("case_type", "") for row in expanded_rows)
    seqs = sorted({str(row.get("seq")) for row in expanded_rows if row.get("seq")})
    high_quality_observed_rows = [
        row for row in expanded_rows if str(row.get("quality_source")) == "default" and str(row.get("quality_type")) == "high_quality"
    ]
    summary = {
        "schema": "acl2_v84_phase10_support_expansion_plan_v1",
        "base_pair_rows": len(base_rows),
        "expanded_pair_rows": len(expanded_rows),
        "rows_increase_ratio": len(expanded_rows) / max(len(base_rows), 1),
        "sequence_coverage": seqs,
        "sequence_coverage_count": len(seqs),
        "rows_by_scope": dict(rows_by_scope),
        "rows_by_case_type": dict(rows_by_case),
        "default_high_quality_observable_rows": len(high_quality_observed_rows),
        "missing_direct_hook_high_quality_chunks_by_seq": missing_by_seq,
        "pre_candidate_rows_2x_gate_pass": len(expanded_rows) >= 2 * len(base_rows),
        "notes": [
            "Unlabelled support rows expand observability only; they are not good/false-positive labels.",
            "seq01 sparse/minconf0 rows remain separated and forbidden as positive stable evidence when marked so.",
            "Direct READ/SWA availability is required before a row is added to the candidate bank.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "support_expansion_pair_bank.csv", expanded_rows)
    write_csv(args.out_dir / "missing_direct_hook_targets.csv", missing_rows)
    write_csv(args.out_dir / "direct_availability_by_overlap_row.csv", direct_availability_rows)
    write_json(args.out_dir / "support_expansion_plan.json", summary)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "expanded_pair_rows": len(expanded_rows),
                "rows_increase_ratio": summary["rows_increase_ratio"],
                "pre_candidate_rows_2x_gate_pass": summary["pre_candidate_rows_2x_gate_pass"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
