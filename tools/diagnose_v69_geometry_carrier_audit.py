#!/usr/bin/env python3
"""Audit geometry-first overlap-pair carrier evidence for v69.

This is a diagnostic follow-up for the case where semantic controls match or
beat the semantic anchor rows. It summarizes geometry-only action-oracle CSVs
without promoting them to a v69 semantic-anchor success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--action-csv must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--action-csv label is empty")
    return label, Path(path)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _best_key(row: Mapping[str, Any]) -> Tuple[int, float, float, float, int, int]:
    return (
        1 if _bool(row.get("oracle_action_gate_pass")) else 0,
        _float(row.get("best_mechanism_improvement"), -1.0),
        _float(row.get("raw_overlap_improvement_ratio"), -1.0),
        -_float(row.get("delta_vs_baseline_global_ate"), 1e9),
        1 if _bool(row.get("safe_correction_pass")) else 0,
        1 if _bool(row.get("ate_guard_pass")) else 0,
    )


def _best(rows: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    items = [dict(row) for row in rows]
    if not items:
        return None
    return max(items, key=_best_key)


def _load_anchor_summary(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_csv(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[_int(row.get("chunk_id"))] = dict(row)
    return out


def _classify(best: Optional[Mapping[str, Any]], anchor_supported: bool) -> str:
    if best is None:
        return "missing_rows"
    if _bool(best.get("oracle_action_gate_pass")) and anchor_supported:
        return "gate_pass_anchor_supported"
    if _bool(best.get("oracle_action_gate_pass")):
        return "gate_pass_not_anchor_supported"
    reasons: List[str] = []
    if not _bool(best.get("safe_correction_pass")):
        reasons.append("unsafe_correction")
    if not _bool(best.get("raw_support_pass")):
        reasons.append("raw_support_below_gate")
    if not _bool(best.get("mechanism_pass")):
        reasons.append("mechanism_below_gate")
    if not _bool(best.get("ate_guard_pass")):
        reasons.append("ate_guard_fail")
    return ";".join(reasons) or "gate_false_other"


def _median(values: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(xs)) if xs else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-csv", action="append", type=_parse_label_path, required=True)
    parser.add_argument("--anchor-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-chunks", default="6,7,8,10,12,19,20,29,30,31,32")
    parser.add_argument("--min-positive-chunks", type=int, default=4)
    args = parser.parse_args()

    target_chunks = [int(x) for x in str(args.target_chunks).split(",") if x.strip()]
    target_set = set(target_chunks)
    anchor_summary = _load_anchor_summary(args.anchor_summary)

    rows: List[Dict[str, Any]] = []
    for source_label, path in args.action_csv:
        for row in _read_csv(path):
            item = dict(row)
            item["geometry_source_label"] = source_label
            item["geometry_source_csv"] = str(path)
            rows.append(item)
    if not rows:
        raise ValueError("no action rows loaded")

    chunk_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    union_gate_chunks = set()
    union_anchor_supported_gate_chunks = set()

    for source_label in sorted({str(row["geometry_source_label"]) for row in rows}):
        source_items = [row for row in rows if str(row["geometry_source_label"]) == source_label]
        target_best_improvements: List[float] = []
        gate_chunks = set()
        anchor_supported_gate_chunks = set()
        blocker_counter: Counter[str] = Counter()
        for chunk in target_chunks:
            sub = [row for row in source_items if _int(row.get("curr_chunk")) == chunk]
            best = _best(sub)
            aq = anchor_summary.get(chunk, {})
            anchor_supported = _bool(aq.get("anchor_bank_quality_pass")) and (
                _int(aq.get("valid_scale_anchor_count"), 0) > 0
                or _int(aq.get("valid_read_anchor_count"), 0) > 0
            )
            reason = _classify(best, anchor_supported)
            blocker_counter.update(reason.split(";"))
            if best is not None:
                target_best_improvements.append(max(0.0, _float(best.get("best_mechanism_improvement"), 0.0)))
                if _bool(best.get("oracle_action_gate_pass")):
                    gate_chunks.add(chunk)
                    union_gate_chunks.add(chunk)
                    if anchor_supported:
                        anchor_supported_gate_chunks.add(chunk)
                        union_anchor_supported_gate_chunks.add(chunk)
            chunk_rows.append({
                "source_label": source_label,
                "chunk_id": chunk,
                "rows": len(sub),
                "best_candidate": best.get("candidate", "") if best else "",
                "best_action_family": best.get("action_family", "") if best else "",
                "best_scope": best.get("scope", "") if best else "",
                "best_damping_alpha": _float(best.get("damping_alpha")) if best else "",
                "oracle_action_gate_pass": _bool(best.get("oracle_action_gate_pass")) if best else False,
                "anchor_supported": anchor_supported,
                "classification": reason,
                "best_mechanism_improvement": _float(best.get("best_mechanism_improvement")) if best else "",
                "raw_overlap_improvement_ratio": _float(best.get("raw_overlap_improvement_ratio")) if best else "",
                "delta_vs_baseline_global_ate": _float(best.get("delta_vs_baseline_global_ate")) if best else "",
                "safe_correction_pass": _bool(best.get("safe_correction_pass")) if best else False,
                "raw_support_pass": _bool(best.get("raw_support_pass")) if best else False,
                "mechanism_pass": _bool(best.get("mechanism_pass")) if best else False,
                "ate_guard_pass": _bool(best.get("ate_guard_pass")) if best else False,
                "anchor_valid_scale_count": aq.get("valid_scale_anchor_count", ""),
                "anchor_valid_read_count": aq.get("valid_read_anchor_count", ""),
                "anchor_reject_reason": aq.get("reject_reason", ""),
            })

        source_rows.append({
            "source_label": source_label,
            "target_gate_chunks": ";".join(str(c) for c in sorted(gate_chunks)),
            "target_gate_count": len(gate_chunks),
            "anchor_supported_target_gate_chunks": ";".join(str(c) for c in sorted(anchor_supported_gate_chunks)),
            "anchor_supported_target_gate_count": len(anchor_supported_gate_chunks),
            "median_target_best_mechanism_improvement": _median(target_best_improvements),
            "dominant_blockers": ";".join(
                f"{name}:{count}" for name, count in blocker_counter.most_common()
            ),
            "geometry_carrier_family_pass": len(gate_chunks) >= int(args.min_positive_chunks),
            "anchor_supported_family_pass": len(anchor_supported_gate_chunks) >= int(args.min_positive_chunks),
        })

    summary = {
        "schema": "acl2_v69_geometry_carrier_audit_v1",
        "action_csvs": [{"label": label, "path": str(path)} for label, path in args.action_csv],
        "anchor_summary": str(args.anchor_summary) if args.anchor_summary else None,
        "target_chunks": target_chunks,
        "rows": len(rows),
        "source_summary_csv": str(args.out_dir / "geometry_carrier_source_summary.csv"),
        "chunk_audit_csv": str(args.out_dir / "geometry_carrier_chunk_audit.csv"),
        "union_gate_chunks": sorted(union_gate_chunks),
        "union_gate_count": len(union_gate_chunks),
        "union_anchor_supported_gate_chunks": sorted(union_anchor_supported_gate_chunks),
        "union_anchor_supported_gate_count": len(union_anchor_supported_gate_chunks),
        "min_positive_chunks": int(args.min_positive_chunks),
        "geometry_carrier_family_pass": len(union_gate_chunks) >= int(args.min_positive_chunks),
        "anchor_supported_geometry_carrier_family_pass": len(union_anchor_supported_gate_chunks) >= int(args.min_positive_chunks),
        "decision": "diagnostic_only",
        "note": (
            "This audit follows the plan's random/shuffled-better branch: geometry-only evidence can guide a future "
            "geometry-first design, but it is not a semantic-anchor promotion."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "geometry_carrier_source_summary.csv", source_rows)
    _write_csv(args.out_dir / "geometry_carrier_chunk_audit.csv", chunk_rows)
    (args.out_dir / "geometry_carrier_audit_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
