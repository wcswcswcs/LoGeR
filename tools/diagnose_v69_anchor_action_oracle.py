#!/usr/bin/env python3
"""Summarize v69 Phase C anchor-action oracle rows and controls.

This tool is intentionally conservative. It can summarize action rows produced
by the existing materialized overlap-pair oracle, but it will not promote a
semantic-causal claim unless semantic rows beat available geometry controls and
the required shuffled/random controls are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _best_key(row: Mapping[str, Any]) -> Tuple[int, float, float, float]:
    return (
        1 if _bool(row.get("oracle_action_gate_pass")) else 0,
        _float(row.get("best_mechanism_improvement"), -1.0),
        _float(row.get("raw_overlap_improvement_ratio"), -1.0),
        -_float(row.get("delta_vs_baseline_global_ate"), 1e9),
    )


def _best(rows: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    items = [dict(row) for row in rows]
    if not items:
        return None
    return max(items, key=_best_key)


CONTROL_ALIASES: Mapping[str, str] = {
    "label_shuffled": "label_shuffled",
    "confidence_shuffled": "confidence_shuffled",
    "same_anchor_count_random": "same_anchor_count_random",
    "same_count_random": "same_anchor_count_random",
    "same_spatial_coverage_random": "same_spatial_coverage_random",
    "same_weight_distribution_random": "same_weight_distribution_random",
}


REQUIRED_CONTROLS = (
    "label_shuffled",
    "confidence_shuffled",
    "same_anchor_count_random",
    "same_spatial_coverage_random",
    "same_weight_distribution_random",
)


def _control_type(label: str) -> str:
    lowered = str(label).lower()
    for needle, canonical in CONTROL_ALIASES.items():
        if needle in lowered:
            return canonical
    return ""


def _load_anchor_quality(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_csv(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[_int(row.get("chunk_id"))] = dict(row)
    return out


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
    parser.add_argument("--semantic-filters", default="ground_static,vertical_static")
    parser.add_argument("--min-target-positive-chunks", type=int, default=4)
    parser.add_argument("--min-median-target-improvement", type=float, default=0.05)
    args = parser.parse_args()

    target_chunks = [int(x) for x in str(args.target_chunks).split(",") if x.strip()]
    target_set = set(target_chunks)
    semantic_filters = {x.strip() for x in str(args.semantic_filters).split(",") if x.strip()}
    anchor_quality = _load_anchor_quality(args.anchor_summary)

    rows: List[Dict[str, Any]] = []
    for source_label, path in args.action_csv:
        for row in _read_csv(path):
            item = dict(row)
            item["phaseC_source_label"] = source_label
            item["phaseC_source_csv"] = str(path)
            item["phaseC_control_type"] = _control_type(source_label)
            rows.append(item)
    if not rows:
        raise ValueError("no action rows loaded")

    group_rows: List[Dict[str, Any]] = []
    for source_label in sorted({str(row["phaseC_source_label"]) for row in rows}):
        source_rows = [row for row in rows if row["phaseC_source_label"] == source_label]
        for fit_filter in sorted({str(row.get("fit_semantic_filter", "")) for row in source_rows}):
            sub = [row for row in source_rows if str(row.get("fit_semantic_filter", "")) == fit_filter]
            gate_rows = [row for row in sub if _bool(row.get("oracle_action_gate_pass"))]
            best = _best(sub) or {}
            group_rows.append({
                "source_label": source_label,
                "fit_semantic_filter": fit_filter,
                "rows": len(sub),
                "oracle_gate_rows": len(gate_rows),
                "positive_chunks": ";".join(str(c) for c in sorted({_int(r.get("curr_chunk")) for r in gate_rows})),
                "target_positive_chunks": ";".join(str(c) for c in sorted({_int(r.get("curr_chunk")) for r in gate_rows if _int(r.get("curr_chunk")) in target_set})),
                "max_best_mechanism_improvement": _float(best.get("best_mechanism_improvement")),
                "max_raw_overlap_improvement_ratio": _float(best.get("raw_overlap_improvement_ratio")),
                "best_delta_vs_baseline_global_ate": _float(best.get("delta_vs_baseline_global_ate")),
                "best_candidate": best.get("candidate", ""),
            })

    base_rows = [row for row in rows if not str(row.get("phaseC_control_type", ""))]
    control_rows = [row for row in rows if str(row.get("phaseC_control_type", ""))]

    best_by_chunk_filter: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in base_rows:
        key = (_int(row.get("curr_chunk")), str(row.get("fit_semantic_filter", "")))
        old = best_by_chunk_filter.get(key)
        if old is None or _best_key(row) > _best_key(old):
            best_by_chunk_filter[key] = dict(row)

    best_control_by_chunk_filter_type: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for row in control_rows:
        control_type = str(row.get("phaseC_control_type", ""))
        key = (_int(row.get("curr_chunk")), str(row.get("fit_semantic_filter", "")), control_type)
        old = best_control_by_chunk_filter_type.get(key)
        if old is None or _best_key(row) > _best_key(old):
            best_control_by_chunk_filter_type[key] = dict(row)

    compare_rows: List[Dict[str, Any]] = []
    semantic_causal_chunks = set()
    action_positive_chunks = set()
    anchor_supported_action_positive_chunks = set()
    for chunk in target_chunks:
        geo = best_by_chunk_filter.get((chunk, "all"))
        geo_mech = _float(geo.get("best_mechanism_improvement")) if geo else float("nan")
        geo_delta = _float(geo.get("delta_vs_baseline_global_ate")) if geo else float("nan")
        aq = anchor_quality.get(chunk, {})
        anchor_supported = _bool(aq.get("anchor_bank_quality_pass")) and (
            _int(aq.get("valid_scale_anchor_count"), 0) > 0
            or _int(aq.get("valid_read_anchor_count"), 0) > 0
        )
        best_any = _best(row for row in base_rows if _int(row.get("curr_chunk")) == chunk)
        if best_any is not None and _bool(best_any.get("oracle_action_gate_pass")):
            action_positive_chunks.add(chunk)
            if anchor_supported:
                anchor_supported_action_positive_chunks.add(chunk)
        for fit_filter in sorted(semantic_filters):
            sem = best_by_chunk_filter.get((chunk, fit_filter))
            if sem is None:
                compare_rows.append({
                    "chunk_id": chunk,
                    "fit_semantic_filter": fit_filter,
                    "has_semantic_row": False,
                    "semantic_oracle_gate_pass": False,
                    "anchor_supported": anchor_supported,
                    "beats_geometry_mechanism": False,
                    "beats_geometry_ate": False,
                    "semantic_causal_positive_available_controls": False,
                })
                continue
            sem_mech = _float(sem.get("best_mechanism_improvement"))
            sem_delta = _float(sem.get("delta_vs_baseline_global_ate"))
            beats_geometry_mech = math.isfinite(sem_mech) and math.isfinite(geo_mech) and sem_mech > geo_mech
            beats_geometry_ate = math.isfinite(sem_delta) and math.isfinite(geo_delta) and sem_delta <= geo_delta
            semantic_gate = _bool(sem.get("oracle_action_gate_pass"))
            control_best = {
                control: best_control_by_chunk_filter_type.get((chunk, fit_filter, control))
                for control in REQUIRED_CONTROLS
            }
            missing_chunk_controls = [control for control, control_row in control_best.items() if control_row is None]
            control_mechs = [
                _float(control_row.get("best_mechanism_improvement"), -1.0)
                for control_row in control_best.values()
                if control_row is not None
            ]
            control_deltas = [
                _float(control_row.get("delta_vs_baseline_global_ate"), 1e9)
                for control_row in control_best.values()
                if control_row is not None
            ]
            max_control_mech = max(control_mechs) if control_mechs else float("nan")
            min_control_delta = min(control_deltas) if control_deltas else float("nan")
            beats_control_mechanism = bool(
                not missing_chunk_controls
                and math.isfinite(sem_mech)
                and math.isfinite(max_control_mech)
                and sem_mech > max_control_mech
            )
            beats_control_ate = bool(
                not missing_chunk_controls
                and math.isfinite(sem_delta)
                and math.isfinite(min_control_delta)
                and sem_delta <= min_control_delta
            )
            causal_available_controls = bool(
                semantic_gate
                and anchor_supported
                and beats_geometry_mech
                and beats_geometry_ate
                and beats_control_mechanism
                and beats_control_ate
            )
            if causal_available_controls:
                semantic_causal_chunks.add(chunk)
            aq = anchor_quality.get(chunk, {})
            compare_rows.append({
                "chunk_id": chunk,
                "fit_semantic_filter": fit_filter,
                "has_semantic_row": True,
                "semantic_candidate": sem.get("candidate", ""),
                "semantic_oracle_gate_pass": semantic_gate,
                "semantic_best_mechanism_improvement": sem_mech,
                "semantic_delta_vs_baseline_global_ate": sem_delta,
                "geometry_best_mechanism_improvement": geo_mech,
                "geometry_delta_vs_baseline_global_ate": geo_delta,
                "anchor_supported": anchor_supported,
                "beats_geometry_mechanism": beats_geometry_mech,
                "beats_geometry_ate": beats_geometry_ate,
                "missing_chunk_required_controls": ";".join(missing_chunk_controls),
                "max_required_control_mechanism": max_control_mech,
                "min_required_control_delta_vs_baseline_global_ate": min_control_delta,
                "beats_required_control_mechanism": beats_control_mechanism,
                "beats_required_control_ate": beats_control_ate,
                "semantic_causal_positive_available_controls": causal_available_controls,
                "anchor_valid_scale_count": aq.get("valid_scale_anchor_count", ""),
                "anchor_valid_read_count": aq.get("valid_read_anchor_count", ""),
                "anchor_quality_pass": aq.get("anchor_bank_quality_pass", ""),
            })

    target_best_improvements = []
    severe_regression_chunks = []
    for chunk in target_chunks:
        best = _best(row for row in base_rows if _int(row.get("curr_chunk")) == chunk)
        if best is None:
            target_best_improvements.append(0.0)
            continue
        target_best_improvements.append(max(0.0, _float(best.get("best_mechanism_improvement"), 0.0)))
        if _float(best.get("delta_vs_baseline_global_ate"), 0.0) > 0.30:
            severe_regression_chunks.append(chunk)

    median_target_improvement = _median(target_best_improvements)
    family_positive_pass = len(anchor_supported_action_positive_chunks) >= int(args.min_target_positive_chunks)
    family_median_pass = bool(
        median_target_improvement is not None
        and median_target_improvement >= float(args.min_median_target_improvement)
        and not severe_regression_chunks
    )
    available_controls = {
        "geometry_only_all_filter": any(str(row.get("fit_semantic_filter", "")) == "all" for row in base_rows),
        "label_shuffled": any(str(row.get("phaseC_control_type", "")) == "label_shuffled" for row in rows),
        "confidence_shuffled": any(str(row.get("phaseC_control_type", "")) == "confidence_shuffled" for row in rows),
        "same_anchor_count_random": any(str(row.get("phaseC_control_type", "")) == "same_anchor_count_random" for row in rows),
        "same_spatial_coverage_random": any(str(row.get("phaseC_control_type", "")) == "same_spatial_coverage_random" for row in rows),
        "same_weight_distribution_random": any(str(row.get("phaseC_control_type", "")) == "same_weight_distribution_random" for row in rows),
    }
    missing_controls = [name for name, ok in available_controls.items() if not ok]
    semantic_causal_gate_pass = bool(semantic_causal_chunks and not missing_controls)
    phaseC_gate_pass = bool((family_positive_pass or family_median_pass) and semantic_causal_gate_pass)

    summary = {
        "schema": "acl2_v69_anchor_action_oracle_summary_v1",
        "action_csvs": [{"label": label, "path": str(path)} for label, path in args.action_csv],
        "anchor_summary": str(args.anchor_summary) if args.anchor_summary else None,
        "target_chunks": target_chunks,
        "rows": len(rows),
        "base_rows": len(base_rows),
        "control_rows": len(control_rows),
        "group_summary_csv": str(args.out_dir / "anchor_action_oracle_by_filter.csv"),
        "chunk_compare_csv": str(args.out_dir / "anchor_action_oracle_chunk_compare.csv"),
        "action_positive_chunks_available_controls": sorted(action_positive_chunks),
        "anchor_supported_action_positive_chunks_available_controls": sorted(anchor_supported_action_positive_chunks),
        "semantic_causal_chunks_available_controls": sorted(semantic_causal_chunks),
        "median_target_best_mechanism_improvement": median_target_improvement,
        "severe_regression_chunks": severe_regression_chunks,
        "available_controls": available_controls,
        "missing_required_controls": missing_controls,
        "phaseC_gate": {
            "family_positive_chunks_ge_min": family_positive_pass,
            "median_target_improvement_ge_min_no_severe_regression": family_median_pass,
            "semantic_causal_positive_with_required_controls": semantic_causal_gate_pass,
        },
        "phaseC_gate_pass": phaseC_gate_pass,
        "decision": (
            "diagnostic_only"
            if not phaseC_gate_pass
            else "phaseC_pass_requires_review_before_online"
        ),
        "note": (
            "A geometry-only all-points control is available when fit_semantic_filter=all appears in non-control rows. "
            "Semantic-causal promotion requires semantic rows to beat geometry and all required shuffled/random "
            "controls on the same chunk/filter."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "anchor_action_oracle_by_filter.csv", group_rows)
    _write_csv(args.out_dir / "anchor_action_oracle_chunk_compare.csv", compare_rows)
    (args.out_dir / "anchor_action_oracle_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
