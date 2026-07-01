#!/usr/bin/env python3
"""Audit held-out coverage for the v80 low-support selected-write separator.

This read-only audit answers whether the chunk08 low-support selected-write
separator has enough cross-case / cross-sequence evidence to justify a held-out
runtime candidate. It does not create new model evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_LONG_CASES = REPORT_ROOT / "phase1_three_memory_case_bank" / "long_five_chunk_cases.csv"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_selected_write_low_support_coverage_20260622_2212"

SELECTED_WRITE_PATTERNS = [
    "phase9_seq01_ref055_v80_selected_write_support_maps*/chunk_*_selected_write_support_map_summary.json",
    "phase9_seq01_ref055_v80_selected_write_support_maps_chunk009_frame279_control_delta_full32/chunk_*_selected_write_support_map_summary.json",
    "phase10_seq00_chunk142_selected_write_low_support_map_20260622_2211/chunk_*_selected_write_support_map_summary.json",
]
SUPPORT_SUMMARY_PATTERNS = [
    "phase9_seq01_ref055_v80_error_semantic_support_maps*/chunk_*_support_map_summary.json",
    "phase9_out4_merge_overlap_error_semantic_support_seq00_chunk142/support_maps/support_map_summary.json",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--long-cases", type=Path, default=DEFAULT_LONG_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _seq_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("source_stage_c_masklet", "source_support_map", "support_path", "source_visual_root"):
        text = str(payload.get(key) or "")
        match = re.search(r"kitti_preprocess/([0-9]{2})/", text)
        if match:
            return match.group(1)
        match = re.search(r"seq([0-9]{2})", text)
        if match:
            return match.group(1)
    return None


def _selected_write_inventory(report_root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in SELECTED_WRITE_PATTERNS:
        paths.extend(report_root.glob(pattern))
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        payload = _read_json(path)
        seq = _seq_from_payload(payload)
        chunk = _safe_int(payload.get("chunk"))
        selected_runtime_mass = _safe_float(payload.get("selected_runtime_mass")) or 0.0
        selected_low_support_mass = _safe_float(payload.get("selected_low_support_mass")) or 0.0
        selected_low_support_frac = _safe_float(payload.get("selected_low_support_given_selected_runtime")) or 0.0
        out.append(
            {
                "artifact_kind": "selected_write_support",
                "path": path,
                "seq": seq,
                "chunk": chunk,
                "selected_runtime_mass": selected_runtime_mass,
                "selected_low_support_mass": selected_low_support_mass,
                "selected_low_support_given_selected_runtime": selected_low_support_frac,
                "runtime_low_support_ratio": _safe_float(payload.get("runtime_low_support_ratio")),
                "selected_write_low_support_flag": (
                    selected_runtime_mass > 0.0 and selected_low_support_mass > 0.0 and selected_low_support_frac >= 0.5
                ),
                "method_gate_claimed": bool(payload.get("method_gate_claimed")),
            }
        )
    return out


def _support_only_inventory(report_root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in SUPPORT_SUMMARY_PATTERNS:
        paths.extend(report_root.glob(pattern))
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        payload = _read_json(path)
        seq = _seq_from_payload(payload)
        chunk = _safe_int(payload.get("chunk"))
        ttt_evidence = payload.get("phase9_ttt_attribution_evidence") or {}
        out.append(
            {
                "artifact_kind": "semantic_support_only",
                "path": path,
                "seq": seq,
                "chunk": chunk,
                "score_mean": _safe_float(payload.get("score_mean")),
                "score_q50": _safe_float(payload.get("score_q50")),
                "semantic_explains_error_region": ttt_evidence.get("semantic_explains_error_region"),
                "semantic_ttt_positive_write_available": ttt_evidence.get("semantic_ttt_positive_write_available"),
                "ttt_writes_stable_carrier": ttt_evidence.get("ttt_writes_stable_carrier"),
                "random_control_separation": ttt_evidence.get("random_control_separation"),
                "recommendation": ttt_evidence.get("recommendation"),
            }
        )
    return out


def _case_coverage_rows(long_cases: list[dict[str, str]], selected_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for item in selected_inventory:
        seq = item.get("seq")
        if seq:
            by_seq.setdefault(str(seq), []).append(item)

    rows: list[dict[str, Any]] = []
    for case in long_cases:
        seq = str(case.get("seq") or "")
        start = _safe_int(case.get("chunk_start"))
        end = _safe_int(case.get("chunk_end"))
        if start is None or end is None:
            continue
        artifacts = [
            item for item in by_seq.get(seq, []) if item.get("chunk") is not None and start <= int(item["chunk"]) <= end
        ]
        positives = [item for item in artifacts if item.get("selected_write_low_support_flag")]
        rows.append(
            {
                "seq": seq,
                "chunk_start": start,
                "chunk_end": end,
                "case_type": case.get("case_type"),
                "case_rank": case.get("case_rank"),
                "J_long": case.get("J_long"),
                "artifact_count_in_window": len(artifacts),
                "artifact_chunks_in_window": [item["chunk"] for item in artifacts],
                "positive_separator_chunks_in_window": [item["chunk"] for item in positives],
                "case_has_any_selected_write_artifact": bool(artifacts),
                "case_has_positive_low_support_separator": bool(positives),
            }
        )
    return rows


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v80 Selected-Write Low-Support Coverage Audit",
        "",
        "This is a read-only held-out coverage audit. It does not claim method success.",
        "",
        "## Decision",
        "",
        f"- heldout_multi_case_gate: `{summary['heldout_multi_case_gate']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        "",
        "## Finding",
        "",
        summary["core_finding"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    selected_inventory = _selected_write_inventory(args.report_root)
    support_inventory = _support_only_inventory(args.report_root)
    long_cases = _read_csv(args.long_cases)
    case_rows = _case_coverage_rows(long_cases, selected_inventory)

    positive_selected = [item for item in selected_inventory if item.get("selected_write_low_support_flag")]
    positive_pairs = {
        (str(item.get("seq")), int(item.get("chunk")))
        for item in positive_selected
        if item.get("seq") and item.get("chunk") is not None
    }
    selected_seqs = sorted({str(item["seq"]) for item in selected_inventory if item.get("seq")})
    positive_seqs = sorted({str(item["seq"]) for item in positive_selected if item.get("seq")})
    support_low_without_ttt = [
        item
        for item in support_inventory
        if (item.get("score_mean") is not None and float(item["score_mean"]) < 0.5)
        and item.get("semantic_explains_error_region") is True
        and item.get("semantic_ttt_positive_write_available") is not True
        and (str(item.get("seq")), int(item.get("chunk") or -1)) not in positive_pairs
    ]
    positive_case_rows = [row for row in case_rows if row.get("case_has_positive_low_support_separator")]
    positive_case_seqs = sorted({str(row["seq"]) for row in positive_case_rows if row.get("seq")})
    positive_bad_case_rows = [row for row in positive_case_rows if row.get("case_type") == "bad"]
    positive_good_case_rows = [row for row in positive_case_rows if row.get("case_type") == "good"]
    heldout_multi_case_gate = (
        len(positive_case_seqs) >= 3
        and bool(positive_bad_case_rows)
        and bool(positive_good_case_rows)
        and any(seq != "01" for seq in positive_case_seqs)
    )
    coverage_blockers: list[str] = []
    if len(positive_case_seqs) < 3:
        coverage_blockers.append("positive_separator_case_seqs_lt_3")
    if not positive_good_case_rows:
        coverage_blockers.append("no_positive_good_case_coverage")
    if not any(seq != "01" for seq in positive_case_seqs):
        coverage_blockers.append("no_non_seq01_positive_case_coverage")
    if not positive_bad_case_rows:
        coverage_blockers.append("no_positive_bad_case_coverage")

    if positive_seqs and set(positive_seqs) - {"01"}:
        core_finding = (
            "Fresh selected-write low-support attribution now provides non-seq01 positive evidence, but the coverage "
            "is still not a held-out multi-case carrier: positive long-case coverage spans too few sequences and has "
            "no good-case safety coverage. Therefore it remains diagnostic and cannot be promoted."
        )
    else:
        core_finding = (
            "The low-support selected-write positive evidence is confined to seq01 selected-write artifacts in the "
            "current workspace. A seq00 support-only case also has low semantic support and explains an error region, "
            "but its TTT attribution explicitly reports semantic_ttt_positive_write_available=false. Therefore the "
            "separator is not yet a held-out multi-case carrier and cannot be promoted."
        )

    summary = {
        "schema": "acl2_v80_selected_write_low_support_coverage_audit_v1",
        "diagnostic_only": True,
        "selected_write_artifact_count": len(selected_inventory),
        "selected_write_artifact_seq_chunks": [
            {"seq": item.get("seq"), "chunk": item.get("chunk")} for item in selected_inventory
        ],
        "selected_write_artifact_seqs": selected_seqs,
        "selected_write_positive_seq_chunks": [
            {"seq": item.get("seq"), "chunk": item.get("chunk")} for item in positive_selected
        ],
        "selected_write_positive_seqs": positive_seqs,
        "long_case_rows": len(case_rows),
        "long_case_rows_with_any_selected_write_artifact": sum(
            1 for row in case_rows if row.get("case_has_any_selected_write_artifact")
        ),
        "long_case_rows_with_positive_low_support_separator": len(positive_case_rows),
        "long_positive_separator_case_seqs": positive_case_seqs,
        "long_positive_bad_case_rows": len(positive_bad_case_rows),
        "long_positive_good_case_rows": len(positive_good_case_rows),
        "support_only_low_semantic_error_no_ttt_write_seq_chunks": [
            {"seq": item.get("seq"), "chunk": item.get("chunk"), "score_mean": item.get("score_mean")}
            for item in support_low_without_ttt
        ],
        "heldout_multi_case_gate": heldout_multi_case_gate,
        "coverage_blockers": coverage_blockers,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "core_finding": core_finding,
        "next_action": (
            "Do not claim held-out coverage for the low-support selected-write separator. A real continuation would "
            "need positive coverage on additional Phase1 long cases, including good-case safety, or a different "
            "actuator that uses this separator without repeating the failed selected-write veto family."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "selected_write_low_support_artifact_inventory.csv", selected_inventory + support_inventory)
    _write_csv(args.out_dir / "selected_write_low_support_case_coverage.csv", case_rows)
    _write_json(args.out_dir / "selected_write_low_support_coverage_summary.json", summary)
    _write_report(args.out_dir / "selected_write_low_support_coverage_report.md", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
