#!/usr/bin/env python3
"""Audit whether the v80 RADIO topology clue covers extra seq01 chunks.

This is an offline diagnostic. It projects the chunk08-only RADIO topology
rules onto every existing seq01 RADIO sidecar chunk and overlays phase1
good/bad case membership. It does not infer qscale runtime outcomes for chunks
that have not been run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_SIDECAR_ROOT = Path("results/kitti_preprocess/01/radseg_sidecar_chunks_slide336_stride224")
DEFAULT_CASE_BANK = REPORT_ROOT / "phase1_three_memory_case_bank"
DEFAULT_DECISIONS = (
    REPORT_ROOT
    / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
    / "thingstuff_radio_qscale_ref055_canary5_decisions.csv"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_radio_guard_extra_chunk_coverage_20260622_2140"

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
    parser.add_argument("--case-bank-dir", type=Path, default=DEFAULT_CASE_BANK)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="")
    parser.add_argument("--seq", default="01")
    parser.add_argument("--overlap-frames", type=int, default=3)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
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


def _parse_chunks(text: str, sidecar_root: Path) -> list[int]:
    if text.strip():
        return sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    chunks: list[int] = []
    for path in sorted(sidecar_root.glob("chunk_*_*/radio_sidecar.pt")):
        chunk = _safe_int(path.parent.name.split("_")[1])
        if chunk is not None:
            chunks.append(chunk)
    return sorted(set(chunks))


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
    global_start = _safe_int(payload.get("global_start_frame")) or 0
    global_end = _safe_int(payload.get("global_end_frame")) or 0
    frames = min(int(overlap_frames), max(0, global_end - global_start))
    if frames <= 0:
        frames = int(overlap_frames)
    out: dict[str, Any] = {
        "radio_sidecar": str(path),
        "radio_format": payload.get("format"),
        "radio_source": payload.get("source"),
        "radio_patch_grid": payload.get("patch_grid"),
        "global_start_frame": global_start,
        "global_end_frame": global_end,
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
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "known_qscale_decision_available": True,
            "qscale_head_tail_phaseE_chunk_pass": _bool_text(row.get("head_tail_phaseE_chunk_pass")),
            "qscale_overlap_phaseE_chunk_pass": _bool_text(row.get("overlap_phaseE_chunk_pass")),
            "qscale_head_tail_improvement_vs_baseline_ratio": _safe_float(row.get("head_tail_improvement_vs_baseline_ratio")),
            "qscale_overlap_improvement_vs_baseline_ratio": _safe_float(row.get("overlap_improvement_vs_baseline_ratio")),
        }
    return out


def _classify(row: dict[str, Any]) -> str:
    if not row.get("known_qscale_decision_available"):
        return "unknown_not_run"
    head = bool(row.get("qscale_head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("qscale_overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "qscale_helpful_overlap_safe"
    if head and not overlap:
        return "qscale_head_tail_only_overlap_harm"
    return "qscale_not_helpful"


def _case_membership(case_bank_dir: Path, seq: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}

    def ensure(chunk: int) -> dict[str, Any]:
        return out.setdefault(
            chunk,
            {
                "mid_good_pairs": [],
                "mid_bad_pairs": [],
                "long_good_windows": [],
                "long_bad_windows": [],
            },
        )

    for row in _read_csv(case_bank_dir / "mid_adjacent_pair_cases.csv"):
        if row.get("seq") != seq:
            continue
        prev_chunk = _safe_int(row.get("prev_chunk"))
        curr_chunk = _safe_int(row.get("curr_chunk"))
        if prev_chunk is None or curr_chunk is None:
            continue
        case_type = row.get("case_type") or "unknown"
        label = f"{prev_chunk}-{curr_chunk}:rank{row.get('case_rank') or '?'}"
        key = "mid_good_pairs" if case_type == "good" else "mid_bad_pairs"
        ensure(prev_chunk)[key].append(label)
        ensure(curr_chunk)[key].append(label)

    for row in _read_csv(case_bank_dir / "long_five_chunk_cases.csv"):
        if row.get("seq") != seq:
            continue
        chunk_start = _safe_int(row.get("chunk_start"))
        chunk_end = _safe_int(row.get("chunk_end"))
        if chunk_start is None or chunk_end is None:
            continue
        case_type = row.get("case_type") or "unknown"
        label = f"{chunk_start}-{chunk_end}:rank{row.get('case_rank') or '?'}"
        key = "long_good_windows" if case_type == "good" else "long_bad_windows"
        for chunk in range(chunk_start, chunk_end + 1):
            ensure(chunk)[key].append(label)

    return out


def _rules() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    return [
        (
            "radio_lowtrust_mean_le_0p30",
            "Chunk08 separator: RADIO lowtrust mean in overlap frames <= 0.30.",
            lambda r: float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.30,
        ),
        (
            "radio_boundary_lowtrust_guard",
            "Chunk08 separator: boundary mean <= 0.59 and lowtrust mean <= 0.31.",
            lambda r: float(r.get("object_boundary_score_mean") or 1.0) <= 0.59
            and float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.31,
        ),
        (
            "radio_sky_context_lowtrust_guard",
            "Chunk08 separator: sky-context mean >= 0.44 and lowtrust mean <= 0.31.",
            lambda r: float(r.get("radio_sky_context_score_mean") or 0.0) >= 0.44
            and float(r.get("radio_lowtrust_score_mean") or 1.0) <= 0.31,
        ),
        (
            "radio_temporal_stability_ge_0p975",
            "Rejected earlier: temporal stability mean >= 0.975.",
            lambda r: float(r.get("temporal_stability_mean") or 0.0) >= 0.975,
        ),
        (
            "radio_dynamic_mean_le_0p03",
            "Rejected earlier: RADIO dynamic mean <= 0.03.",
            lambda r: float(r.get("radio_dynamic_score_mean") or 1.0) <= 0.03,
        ),
        (
            "radio_interior_mean_ge_0p40",
            "Rejected earlier: object interior mean >= 0.40.",
            lambda r: float(r.get("object_interior_score_mean") or 0.0) >= 0.40,
        ),
    ]


def _with_case_flags(row: dict[str, Any]) -> dict[str, Any]:
    row["phase1_mid_good_member"] = bool(row.get("mid_good_pairs"))
    row["phase1_mid_bad_member"] = bool(row.get("mid_bad_pairs"))
    row["phase1_long_good_member"] = bool(row.get("long_good_windows"))
    row["phase1_long_bad_member"] = bool(row.get("long_bad_windows"))
    row["phase1_any_good_member"] = bool(row["phase1_mid_good_member"] or row["phase1_long_good_member"])
    row["phase1_any_bad_member"] = bool(row["phase1_mid_bad_member"] or row["phase1_long_bad_member"])
    return row


def _rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    known_chunks = {int(row["chunk"]) for row in rows if row.get("known_qscale_decision_available")}
    for rule_name, description, predicate in _rules():
        selected = [row for row in rows if predicate(row)]
        selected_chunks = [int(row["chunk"]) for row in selected]
        selected_known_chunks = [chunk for chunk in selected_chunks if chunk in known_chunks]
        selected_unknown_chunks = [chunk for chunk in selected_chunks if chunk not in known_chunks]
        selected_known_safe = [
            int(row["chunk"]) for row in selected if row.get("qscale_outcome_class") == "qscale_helpful_overlap_safe"
        ]
        selected_known_harm = [
            int(row["chunk"])
            for row in selected
            if row.get("qscale_outcome_class") == "qscale_head_tail_only_overlap_harm"
        ]
        selected_good_case_chunks = [int(row["chunk"]) for row in selected if row.get("phase1_any_good_member")]
        selected_bad_case_chunks = [int(row["chunk"]) for row in selected if row.get("phase1_any_bad_member")]
        out.append(
            {
                "rule": rule_name,
                "description": description,
                "selected_chunks": selected_chunks,
                "selected_known_chunks": selected_known_chunks,
                "selected_unknown_chunks": selected_unknown_chunks,
                "selected_known_helpful_overlap_safe_chunks": selected_known_safe,
                "selected_known_overlap_harm_chunks": selected_known_harm,
                "selected_phase1_good_case_chunks": selected_good_case_chunks,
                "selected_phase1_bad_case_chunks": selected_bad_case_chunks,
                "selected_unknown_phase1_good_case_chunks": [
                    int(row["chunk"])
                    for row in selected
                    if (not row.get("known_qscale_decision_available")) and row.get("phase1_any_good_member")
                ],
                "selected_unknown_phase1_bad_case_chunks": [
                    int(row["chunk"])
                    for row in selected
                    if (not row.get("known_qscale_decision_available")) and row.get("phase1_any_bad_member")
                ],
                "preserves_chunk08_separator_on_known_canary": selected_known_safe == [8] and not selected_known_harm,
                "has_extra_runtime_coverage_candidate": bool(selected_unknown_chunks),
                "method_gate_claimed": False,
            }
        )
    return out


def _status(rule_rows: list[dict[str, Any]]) -> tuple[str, str]:
    separator_rows = [row for row in rule_rows if row["preserves_chunk08_separator_on_known_canary"]]
    if not separator_rows:
        return (
            "no_known_canary_separator_after_extra_overlay",
            "No RADIO guard keeps the known chunk08-vs-harm separation after overlay.",
        )
    extra_good_rows = [
        row
        for row in separator_rows
        if row.get("selected_unknown_phase1_good_case_chunks")
    ]
    if extra_good_rows:
        return (
            "radio_guard_has_extra_good_case_runtime_candidates",
            "At least one chunk08-preserving guard also selects unrun phase1 good-case chunks; run minimal runtime safety smoke before any promotion.",
        )
    extra_rows = [row for row in separator_rows if row.get("selected_unknown_chunks")]
    if extra_rows:
        return (
            "radio_guard_has_unlabeled_extra_candidates_only",
            "At least one chunk08-preserving guard selects extra unrun chunks, but none are in phase1 good cases.",
        )
    return (
        "radio_guard_chunk08_only_no_extra_coverage",
        "Chunk08-preserving guards do not select any additional unrun sidecar chunks; no fast runtime promotion path appears.",
    )


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 RADIO Guard Extra Chunk Coverage Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Chunk Overlay",
        "",
        "| chunk | qscale_class | mid_good | mid_bad | long_good | long_bad | lowtrust | boundary | sky | stability |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    str(row["qscale_outcome_class"]),
                    str(row["phase1_mid_good_member"]).lower(),
                    str(row["phase1_mid_bad_member"]).lower(),
                    str(row["phase1_long_good_member"]).lower(),
                    str(row["phase1_long_bad_member"]).lower(),
                    f"{float(row.get('radio_lowtrust_score_mean') or 0.0):.6f}",
                    f"{float(row.get('object_boundary_score_mean') or 0.0):.6f}",
                    f"{float(row.get('radio_sky_context_score_mean') or 0.0):.6f}",
                    f"{float(row.get('temporal_stability_mean') or 0.0):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rule Overlay", ""])
    lines.append("| rule | selected_chunks | known_safe | known_harm | extra_good | preserves_known_separator |")
    lines.append("|---|---|---|---|---|---:|")
    for row in rule_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rule"]),
                    json.dumps(row["selected_chunks"], ensure_ascii=False),
                    json.dumps(row["selected_known_helpful_overlap_safe_chunks"], ensure_ascii=False),
                    json.dumps(row["selected_known_overlap_harm_chunks"], ensure_ascii=False),
                    json.dumps(row["selected_unknown_phase1_good_case_chunks"], ensure_ascii=False),
                    str(row["preserves_chunk08_separator_on_known_canary"]).lower(),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks, args.sidecar_root)
    decisions = _decision_by_chunk(args.decisions_csv)
    membership = _case_membership(args.case_bank_dir, args.seq)

    rows: list[dict[str, Any]] = []
    missing_sidecars: list[int] = []
    for chunk in chunks:
        sidecar = _sidecar_path(args.sidecar_root, chunk)
        if sidecar is None:
            missing_sidecars.append(chunk)
            continue
        row: dict[str, Any] = {"seq": args.seq, "chunk": chunk, **_radio_features(sidecar, args.overlap_frames)}
        row.update(
            {
                "known_qscale_decision_available": False,
                "qscale_head_tail_phaseE_chunk_pass": None,
                "qscale_overlap_phaseE_chunk_pass": None,
                "qscale_head_tail_improvement_vs_baseline_ratio": None,
                "qscale_overlap_improvement_vs_baseline_ratio": None,
            }
        )
        row.update(decisions.get(chunk, {}))
        row.update(membership.get(chunk, {}))
        for key in ("mid_good_pairs", "mid_bad_pairs", "long_good_windows", "long_bad_windows"):
            row.setdefault(key, [])
        row["qscale_outcome_class"] = _classify(row)
        rows.append(_with_case_flags(row))

    rules = _rule_rows(rows)
    status, status_detail = _status(rules)
    separator_rules = [row["rule"] for row in rules if row["preserves_chunk08_separator_on_known_canary"]]
    extra_good_rule_rows = [row for row in rules if row["preserves_chunk08_separator_on_known_canary"] and row["selected_unknown_phase1_good_case_chunks"]]
    extra_good_chunks = sorted(
        {
            int(chunk)
            for row in extra_good_rule_rows
            for chunk in row["selected_unknown_phase1_good_case_chunks"]
        }
    )
    extra_good_case_chunks = [
        {
            "chunk": int(row["chunk"]),
            "mid_good_pairs": row["mid_good_pairs"],
            "long_good_windows": row["long_good_windows"],
            "lowtrust_mean": row.get("radio_lowtrust_score_mean"),
            "boundary_mean": row.get("object_boundary_score_mean"),
            "sky_context_mean": row.get("radio_sky_context_score_mean"),
            "temporal_stability_mean": row.get("temporal_stability_mean"),
        }
        for row in rows
        if int(row["chunk"]) in extra_good_chunks
    ]
    recommended_runtime_chunks = extra_good_chunks[:]
    next_action = (
        f"Run minimal RADIO qscale safety smoke on chunks {recommended_runtime_chunks} before promotion."
        if recommended_runtime_chunks
        else "Do not launch slow runtime from this guard alone; it has no extra phase1-good coverage candidate."
    )
    summary = {
        "schema": "acl2_v80_radio_guard_extra_chunk_coverage_v1",
        "status": status,
        "status_detail": status_detail,
        "seq": args.seq,
        "chunks": chunks,
        "missing_sidecars": missing_sidecars,
        "known_qscale_decision_chunks": sorted(decisions.keys()),
        "phase1_good_case_sidecar_chunks": sorted(
            {int(row["chunk"]) for row in rows if row["phase1_any_good_member"]}
        ),
        "phase1_bad_case_sidecar_chunks": sorted(
            {int(row["chunk"]) for row in rows if row["phase1_any_bad_member"]}
        ),
        "separator_rules_preserving_known_canary": separator_rules,
        "separator_rules_with_extra_good_candidates": [row["rule"] for row in extra_good_rule_rows],
        "extra_good_case_runtime_candidate_chunks": extra_good_chunks,
        "extra_good_case_runtime_candidate_details": extra_good_case_chunks,
        "recommended_runtime_chunks": recommended_runtime_chunks,
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "next_action": next_action,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "radio_guard_extra_chunk_summary.json", summary)
    _write_csv(args.out_dir / "radio_guard_extra_chunk_rows.csv", rows)
    _write_csv(args.out_dir / "radio_guard_extra_chunk_rule_overlay.csv", rules)
    _write_report(args.out_dir / "radio_guard_extra_chunk_report.md", summary, rows, rules)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
