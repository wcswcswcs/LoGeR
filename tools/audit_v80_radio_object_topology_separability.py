#!/usr/bin/env python3
"""Audit RADIO object-topology separability for v80 seq01 canary chunks.

This is an offline diagnostic. It asks whether existing RADIO/RADSeg fields
can separate the one qscale chunk that helps both head-tail and future-overlap
from qscale chunks that only help head-tail while hurting future-overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_SIDECAR_ROOT = Path("results/kitti_preprocess/01/radseg_sidecar_chunks_slide336_stride224")
DEFAULT_DECISIONS = (
    REPORT_ROOT
    / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
    / "thingstuff_radio_qscale_ref055_canary5_decisions.csv"
)
DEFAULT_ALIGNMENT = (
    REPORT_ROOT
    / "phase9_seq01_non_gt_direction_recheck"
    / "non_gt_direction_recheck_rows.csv"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_radio_object_topology_separability_audit_20260622_2115"


RADIO_KEYS = (
    "object_boundary_score",
    "object_interior_score",
    "radio_confidence",
    "radio_static_score",
    "radio_dynamic_score",
    "radio_sky_context_score",
    "radio_lowtrust_score",
    "temporal_stability",
    "temporal_embedding_var",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR_ROOT)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--alignment-csv", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="6,7,8,10,12")
    parser.add_argument("--overlap-frames", type=int, default=3)
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _sidecar_path(root: Path, chunk: int) -> Path | None:
    matches = sorted(root.glob(f"chunk_{chunk:03d}_*/radio_sidecar.pt"))
    return matches[0] if matches else None


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.float().flatten()
    if flat.numel() == 0:
        return {"mean": math.nan, "q10": math.nan, "q50": math.nan, "q90": math.nan, "gt05": math.nan, "gt07": math.nan}
    return {
        "mean": float(flat.mean()),
        "q10": float(torch.quantile(flat, 0.10)),
        "q50": float(torch.quantile(flat, 0.50)),
        "q90": float(torch.quantile(flat, 0.90)),
        "gt05": float((flat > 0.5).float().mean()),
        "gt07": float((flat > 0.7).float().mean()),
    }


def _radio_features(path: Path, overlap_frames: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    frames = min(int(overlap_frames), int(payload.get("global_end_frame", 0)) - int(payload.get("global_start_frame", 0)))
    if frames <= 0:
        frames = int(overlap_frames)
    out: dict[str, Any] = {
        "radio_sidecar": str(path),
        "radio_format": payload.get("format"),
        "radio_source": payload.get("source"),
        "radio_patch_grid": payload.get("patch_grid"),
        "global_start_frame": payload.get("global_start_frame"),
        "global_end_frame": payload.get("global_end_frame"),
        "overlap_frames_used": frames,
    }
    for key in RADIO_KEYS:
        if key not in payload or not hasattr(payload[key], "shape"):
            out[f"{key}_available"] = False
            continue
        stats = _tensor_stats(payload[key][:frames])
        out[f"{key}_available"] = True
        for stat_key, stat_value in stats.items():
            out[f"{key}_{stat_key}"] = stat_value
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    out["object_boundary_contrast_mean"] = debug.get("object_boundary_contrast_mean")
    out["component_count_mean"] = debug.get("component_count_mean")
    return out


def _decision_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = int(row["chunk"])
        out[chunk] = {
            "qscale_head_tail_phaseE_chunk_pass": _bool_text(row.get("head_tail_phaseE_chunk_pass")),
            "qscale_overlap_phaseE_chunk_pass": _bool_text(row.get("overlap_phaseE_chunk_pass")),
            "qscale_head_tail_improvement_vs_baseline_ratio": _safe_float(row.get("head_tail_improvement_vs_baseline_ratio")),
            "qscale_overlap_improvement_vs_baseline_ratio": _safe_float(row.get("overlap_improvement_vs_baseline_ratio")),
        }
    return out


def _alignment_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = int(row["chunk"])
        out[chunk] = {
            "selected_runtime_mass": _safe_float(row.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(row.get("selected_low_support_mass")),
            "selected_write_interpretation": row.get("selected_write_interpretation") or row.get("interpretation"),
        }
    return out


def _classify(row: dict[str, Any]) -> str:
    head = bool(row.get("qscale_head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("qscale_overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "qscale_helpful_overlap_safe"
    if head and not overlap:
        return "qscale_head_tail_only_overlap_harm"
    return "qscale_not_helpful"


def _median(values: list[float]) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _select_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = [
        (
            "radio_lowtrust_mean_le_0p30",
            "Select qscale only when RADIO lowtrust mean in overlap frames is <= 0.30.",
            lambda r: float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.30,
        ),
        (
            "radio_boundary_lowtrust_guard",
            "Select qscale when RADIO boundary mean <= 0.59 and lowtrust mean <= 0.31.",
            lambda r: float(r.get("object_boundary_score_mean") or 1.0) <= 0.59
            and float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.31,
        ),
        (
            "radio_sky_context_lowtrust_guard",
            "Select qscale when sky-context mean >= 0.44 and lowtrust mean <= 0.31.",
            lambda r: float(r.get("radio_sky_context_score_mean") or 0.0) >= 0.44
            and float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.31,
        ),
        (
            "radio_temporal_stability_ge_0p975",
            "Select qscale when temporal stability mean is >= 0.975.",
            lambda r: float(r.get("temporal_stability_mean") or 0.0) >= 0.975,
        ),
        (
            "radio_dynamic_mean_le_0p03",
            "Select qscale when RADIO dynamic mean is <= 0.03.",
            lambda r: float(r.get("radio_dynamic_score_mean") or 1.0) <= 0.03,
        ),
        (
            "radio_interior_mean_ge_0p40",
            "Select qscale when object interior mean is >= 0.40.",
            lambda r: float(r.get("object_interior_score_mean") or 0.0) >= 0.40,
        ),
    ]

    audit_rows: list[dict[str, Any]] = []
    for rule_name, description, predicate in rules:
        selected = [row for row in rows if predicate(row)]
        selected_chunks = [int(row["chunk"]) for row in selected]
        head_tail_pass_chunks = [
            int(row["chunk"]) for row in selected if bool(row.get("qscale_head_tail_phaseE_chunk_pass"))
        ]
        overlap_pass_chunks = [int(row["chunk"]) for row in selected if bool(row.get("qscale_overlap_phaseE_chunk_pass"))]
        overlap_harm_chunks = [
            int(row["chunk"])
            for row in selected
            if (row.get("qscale_overlap_improvement_vs_baseline_ratio") is not None)
            and float(row["qscale_overlap_improvement_vs_baseline_ratio"]) < 0.0
        ]
        helpful_safe_chunks = [
            int(row["chunk"]) for row in selected if row.get("qscale_outcome_class") == "qscale_helpful_overlap_safe"
        ]
        false_positive_chunks = [
            int(row["chunk"])
            for row in selected
            if row.get("qscale_outcome_class") == "qscale_head_tail_only_overlap_harm"
        ]

        head_values: list[float] = []
        overlap_values: list[float] = []
        for row in rows:
            if int(row["chunk"]) in selected_chunks:
                head_values.append(float(row.get("qscale_head_tail_improvement_vs_baseline_ratio") or 0.0))
                overlap_values.append(float(row.get("qscale_overlap_improvement_vs_baseline_ratio") or 0.0))
            else:
                head_values.append(0.0)
                overlap_values.append(0.0)
        head_median = _median(head_values)
        overlap_median = _median(overlap_values)
        canary_rule_gate_pass = bool(
            (len(head_tail_pass_chunks) >= 4 and head_median >= 0.05)
            or (len(overlap_pass_chunks) >= 4 and overlap_median >= 0.05)
        )
        diagnostic_separates_chunk08_from_false_positive = bool(
            helpful_safe_chunks == [8] and not false_positive_chunks
        )
        audit_rows.append(
            {
                "rule": rule_name,
                "description": description,
                "selected_chunks": selected_chunks,
                "selected_count": len(selected_chunks),
                "helpful_safe_chunks": helpful_safe_chunks,
                "false_positive_overlap_harm_chunks": false_positive_chunks,
                "head_tail_pass_chunks": head_tail_pass_chunks,
                "overlap_pass_chunks": overlap_pass_chunks,
                "overlap_harm_chunks": overlap_harm_chunks,
                "head_tail_median_improvement_with_native_fallback": head_median,
                "overlap_median_improvement_with_native_fallback": overlap_median,
                "diagnostic_separates_chunk08_from_false_positive": diagnostic_separates_chunk08_from_false_positive,
                "canary_rule_gate_pass": canary_rule_gate_pass,
                "method_gate_claimed": False,
            }
        )
    return audit_rows


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 RADIO Object Topology Separability Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- diagnostic_only: `{summary['diagnostic_only']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- core_blocker: {summary['core_blocker']}",
        "",
        "## Chunk Features",
        "",
        "| chunk | class | boundary_mean | lowtrust_mean | sky_context_mean | temporal_stability_mean | selected_low_support_mass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    str(row["qscale_outcome_class"]),
                    f"{float(row.get('object_boundary_score_mean') or 0.0):.6f}",
                    f"{float(row.get('radio_lowtrust_score_mean') or 0.0):.6f}",
                    f"{float(row.get('radio_sky_context_score_mean') or 0.0):.6f}",
                    f"{float(row.get('temporal_stability_mean') or 0.0):.6f}",
                    str(row.get("selected_low_support_mass")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rule Audit", ""])
    lines.append("| rule | selected_chunks | separates_chunk08 | gate_pass | note |")
    lines.append("|---|---|---:|---:|---|")
    for row in rule_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rule"]),
                    json.dumps(row["selected_chunks"], ensure_ascii=False),
                    str(row["diagnostic_separates_chunk08_from_false_positive"]).lower(),
                    str(row["canary_rule_gate_pass"]).lower(),
                    "diagnostic only; native fallback median remains below v80 PhaseE gate",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    decisions = _decision_by_chunk(args.decisions_csv)
    alignment = _alignment_by_chunk(args.alignment_csv)

    rows: list[dict[str, Any]] = []
    missing_sidecars: list[int] = []
    for chunk in chunks:
        sidecar = _sidecar_path(args.sidecar_root, chunk)
        if sidecar is None:
            missing_sidecars.append(chunk)
            continue
        row: dict[str, Any] = {"chunk": chunk, **_radio_features(sidecar, args.overlap_frames)}
        row.update(decisions.get(chunk, {}))
        row.update(alignment.get(chunk, {}))
        row["qscale_outcome_class"] = _classify(row)
        rows.append(row)

    rule_rows = _select_rules(rows)
    separator_rules = [
        row["rule"] for row in rule_rows if bool(row["diagnostic_separates_chunk08_from_false_positive"])
    ]
    gate_pass_rules = [row["rule"] for row in rule_rows if bool(row["canary_rule_gate_pass"])]

    status = (
        "radio_topology_chunk08_only_fragile_diagnostic"
        if separator_rules and not gate_pass_rules
        else "no_radio_topology_separability"
        if not separator_rules
        else "unexpected_gate_pass_requires_manual_review"
    )
    summary = {
        "schema": "acl2_v80_radio_object_topology_separability_audit_v1",
        "status": status,
        "diagnostic_only": True,
        "chunks": chunks,
        "missing_sidecars": missing_sidecars,
        "radio_action_allowed_seq_scope": "seq01_only_from_phase0",
        "rule_count": len(rule_rows),
        "diagnostic_separator_rules": separator_rules,
        "deployable_gate_pass_rules": gate_pass_rules,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "core_blocker": (
            "RADIO lowtrust/boundary topology can isolate chunk08 from chunk10/chunk12 in this canary, "
            "but the margin is narrow, RADIO is only action-available for seq01, and selecting only chunk08 "
            "leaves native-fallback medians below the v80 PhaseE gate."
        ),
        "next_action": (
            "Do not promote a RADIO topology guard. If continuing, it must be validated on new RADIO-enabled "
            "chunks/sequences or combined with a genuinely non-local future-overlap carrier; current seq01 "
            "canary evidence is diagnostic only."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "radio_object_topology_separability_summary.json", summary)
    _write_csv(args.out_dir / "radio_object_topology_feature_rows.csv", rows)
    _write_csv(args.out_dir / "radio_object_topology_rule_audit.csv", rule_rows)
    _write_report(args.out_dir / "radio_object_topology_separability_report.md", summary, rows, rule_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
