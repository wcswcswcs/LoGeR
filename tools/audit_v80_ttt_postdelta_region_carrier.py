#!/usr/bin/env python3
"""Audit per-region TTT post-delta/support carriers for v80 seq01 canary.

This checks whether existing TTT spatial post-delta maps align with the v80
semantic/geometry support maps strongly enough to define a deployable action
rule. It is offline and uses already materialized tensors only.
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
DEFAULT_REDISCOVERY_ROWS = (
    REPORT_ROOT
    / "phase10_seq01_error_ttt_semantic_alignment_rediscovery_20260622_2030"
    / "canary_error_ttt_semantic_alignment_rows.csv"
)
DEFAULT_DECISIONS = (
    REPORT_ROOT
    / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
    / "thingstuff_radio_qscale_ref055_canary5_decisions.csv"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_ttt_postdelta_region_carrier_audit_20260622_2220"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rediscovery-rows", type=Path, default=DEFAULT_REDISCOVERY_ROWS)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunks", default="6,7,8,10,12")
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--top-frac", type=float, default=0.10)
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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _decision_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "head_tail_phaseE_chunk_pass": _bool_text(row.get("head_tail_phaseE_chunk_pass")),
            "overlap_phaseE_chunk_pass": _bool_text(row.get("overlap_phaseE_chunk_pass")),
            "head_tail_improvement_vs_baseline_ratio": _safe_float(row.get("head_tail_improvement_vs_baseline_ratio")),
            "overlap_improvement_vs_baseline_ratio": _safe_float(row.get("overlap_improvement_vs_baseline_ratio")),
        }
    return out


def _rediscovery_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = _safe_int(row.get("chunk"))
        if chunk is None:
            continue
        out[chunk] = {
            "source_post_delta_pt": row.get("source_post_delta_pt"),
            "source_support_map": row.get("source_support_map"),
            "support_score_mean": _safe_float(row.get("support_score_mean")),
            "support_low_proxy_1_minus_mean": _safe_float(row.get("support_low_proxy_1_minus_mean")),
            "selected_runtime_mass": _safe_float(row.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(row.get("selected_low_support_mass")),
            "rediscovery_interpretation": row.get("interpretation"),
        }
    return out


def _classify(row: dict[str, Any]) -> str:
    head = bool(row.get("head_tail_phaseE_chunk_pass"))
    overlap = bool(row.get("overlap_phaseE_chunk_pass"))
    if head and overlap:
        return "qscale_helpful_overlap_safe"
    if head and not overlap:
        return "qscale_head_tail_only_overlap_harm"
    return "qscale_not_helpful"


def _flatten_overlap_support(payload: dict[str, Any], frames: int) -> dict[str, torch.Tensor]:
    score = payload["score_overlap"].float().cpu()
    dq = payload["Dq_overlap"].float().cpu()
    ds = payload["Ds_overlap"].float().cpu()
    if score.ndim == 3:
        score = score[0]
    if dq.ndim == 3:
        dq = dq[0]
    if ds.ndim == 3:
        ds = ds[0]
    frames = min(frames, int(score.shape[0]))
    return {
        "support_score": score[:frames].reshape(-1),
        "Dq": dq[:frames].reshape(-1),
        "Ds": ds[:frames].reshape(-1),
    }


def _flatten_post_delta(payload: dict[str, Any], frames: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in (
        "committed_post_delta_norm_projection_patch",
        "native_delta_norm_projection_patch",
        "action_delta_norm_projection_patch",
    ):
        tensor = payload.get(name)
        if not hasattr(tensor, "shape"):
            continue
        arr = tensor.float().cpu()
        # Stored as layer/branch by frame by patch grid. Reduce layer/branch.
        if arr.ndim == 4:
            arr = arr.mean(dim=0)
        if arr.ndim != 3:
            continue
        frames_eff = min(frames, int(arr.shape[0]))
        out[name] = arr[:frames_eff].reshape(-1)
    for name in ("ttt_write_prior_patch", "D_tok_patch", "R_ttt_tok_patch"):
        tensor = payload.get(name)
        if not hasattr(tensor, "shape"):
            continue
        arr = tensor.float().cpu()
        if arr.ndim != 3:
            continue
        frames_eff = min(frames, int(arr.shape[0]))
        out[name] = arr[:frames_eff].reshape(-1)
    return out


def _corr(a: torch.Tensor, b: torch.Tensor) -> float | None:
    n = min(a.numel(), b.numel())
    if n < 2:
        return None
    x = a[:n].float()
    y = b[:n].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denom <= 1.0e-12:
        return None
    return float(torch.dot(x, y) / denom)


def _top_mask(values: torch.Tensor, top_frac: float) -> torch.Tensor:
    values = values.float()
    n = int(values.numel())
    if n <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    k = max(1, int(round(n * float(top_frac))))
    threshold = torch.topk(values, k).values.min()
    return values >= threshold


def _mean(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.float().mean())


def _region_features(row: dict[str, Any], overlap_frames: int, top_frac: float) -> dict[str, Any]:
    post_path = Path(str(row.get("source_post_delta_pt") or ""))
    support_path = Path(str(row.get("source_support_map") or ""))
    out: dict[str, Any] = {
        "post_delta_path": str(post_path),
        "support_map_path": str(support_path),
        "post_delta_exists": post_path.exists(),
        "support_map_exists": support_path.exists(),
    }
    if not post_path.exists() or not support_path.exists():
        return out
    post_payload = torch.load(post_path, map_location="cpu")
    support_payload = torch.load(support_path, map_location="cpu")
    if not isinstance(post_payload, dict) or not isinstance(support_payload, dict):
        out["error"] = "payload_not_dict"
        return out
    support = _flatten_overlap_support(support_payload, overlap_frames)
    delta = _flatten_post_delta(post_payload, overlap_frames)
    if "action_delta_norm_projection_patch" not in delta:
        out["error"] = "missing_action_delta_norm_projection_patch"
        return out

    n = min(delta["action_delta_norm_projection_patch"].numel(), support["support_score"].numel())
    if n <= 0:
        out["error"] = "empty_aligned_vectors"
        return out
    action = delta["action_delta_norm_projection_patch"][:n].float()
    native = delta.get("native_delta_norm_projection_patch", torch.zeros_like(action))[:n].float()
    committed = delta.get("committed_post_delta_norm_projection_patch", torch.zeros_like(action))[:n].float()
    support_score = support["support_score"][:n].float()
    low_support = 1.0 - support_score
    dq = support["Dq"][:n].float()
    ds = support["Ds"][:n].float()

    action_top = _top_mask(action, top_frac)
    native_top = _top_mask(native, top_frac)
    low_mask = support_score <= 0.50
    high_dq_mask = dq >= 0.50
    high_ds_mask = ds >= 0.50
    action_top_count = int(action_top.sum())
    native_top_count = int(native_top.sum())

    global_low = float(low_mask.float().mean())
    global_high_dq = float(high_dq_mask.float().mean())
    global_high_ds = float(high_ds_mask.float().mean())
    action_low_frac = float((action_top & low_mask).float().sum() / max(action_top_count, 1))
    action_high_dq_frac = float((action_top & high_dq_mask).float().sum() / max(action_top_count, 1))
    action_high_ds_frac = float((action_top & high_ds_mask).float().sum() / max(action_top_count, 1))
    native_low_frac = float((native_top & low_mask).float().sum() / max(native_top_count, 1))

    out.update(
        {
            "aligned_token_count": int(n),
            "top_frac": top_frac,
            "action_top_count": action_top_count,
            "action_delta_mean": _mean(action),
            "action_delta_q90": float(torch.quantile(action, 0.90)),
            "native_delta_mean": _mean(native),
            "committed_delta_mean": _mean(committed),
            "action_minus_native_delta_mean": float(action.mean() - native.mean()),
            "support_score_mean_aligned": _mean(support_score),
            "global_low_support_frac": global_low,
            "global_high_Dq_frac": global_high_dq,
            "global_high_Ds_frac": global_high_ds,
            "action_top_low_support_frac": action_low_frac,
            "action_top_high_Dq_frac": action_high_dq_frac,
            "action_top_high_Ds_frac": action_high_ds_frac,
            "native_top_low_support_frac": native_low_frac,
            "action_top_low_support_enrichment": action_low_frac / global_low if global_low > 0 else None,
            "action_top_high_Dq_enrichment": action_high_dq_frac / global_high_dq if global_high_dq > 0 else None,
            "action_top_high_Ds_enrichment": action_high_ds_frac / global_high_ds if global_high_ds > 0 else None,
            "corr_action_delta_low_support": _corr(action, low_support),
            "corr_action_delta_Dq": _corr(action, dq),
            "corr_action_delta_Ds": _corr(action, ds),
            "corr_action_delta_support_score": _corr(action, support_score),
        }
    )
    return out


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _safe_float(row.get(key))
    return default if value is None else value


def _median(values: list[float]) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _rules() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    return [
        (
            "postdelta_top_low_support_frac_ge_0p75",
            "Select qscale when >=75% of top action post-delta regions are low-support.",
            lambda r: _f(r, "action_top_low_support_frac") >= 0.75,
        ),
        (
            "postdelta_low_support_enrichment_ge_1p35",
            "Select qscale when top action post-delta low-support enrichment >=1.35.",
            lambda r: _f(r, "action_top_low_support_enrichment") >= 1.35,
        ),
        (
            "postdelta_corr_low_support_ge_0p25",
            "Select qscale when action post-delta positively correlates with low-support.",
            lambda r: _f(r, "corr_action_delta_low_support", -1.0) >= 0.25,
        ),
        (
            "postdelta_top_high_Dq_frac_ge_0p75",
            "Select qscale when top action post-delta regions are high Dq.",
            lambda r: _f(r, "action_top_high_Dq_frac") >= 0.75,
        ),
        (
            "postdelta_action_minus_native_delta_positive",
            "Select qscale when action post-delta mean exceeds native post-delta mean.",
            lambda r: _f(r, "action_minus_native_delta_mean", -1.0) > 0.0,
        ),
        (
            "postdelta_low_support_or_corr",
            "Select qscale when top low-support fraction is high or low-support correlation is positive.",
            lambda r: _f(r, "action_top_low_support_frac") >= 0.75
            or _f(r, "corr_action_delta_low_support", -1.0) >= 0.25,
        ),
    ]


def _rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule_name, description, predicate in _rules():
        selected = [row for row in rows if predicate(row)]
        selected_chunks = [int(row["chunk"]) for row in selected]
        head_tail_pass_chunks = [
            int(row["chunk"]) for row in selected if bool(row.get("head_tail_phaseE_chunk_pass"))
        ]
        overlap_pass_chunks = [int(row["chunk"]) for row in selected if bool(row.get("overlap_phaseE_chunk_pass"))]
        overlap_harm_chunks = [
            int(row["chunk"])
            for row in selected
            if (row.get("overlap_improvement_vs_baseline_ratio") is not None)
            and float(row["overlap_improvement_vs_baseline_ratio"]) < 0.0
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
                head_values.append(float(row.get("head_tail_improvement_vs_baseline_ratio") or 0.0))
                overlap_values.append(float(row.get("overlap_improvement_vs_baseline_ratio") or 0.0))
            else:
                head_values.append(0.0)
                overlap_values.append(0.0)
        head_median = _median(head_values)
        overlap_median = _median(overlap_values)
        canary_rule_gate_pass = bool(
            (len(head_tail_pass_chunks) >= 4 and head_median >= 0.05)
            or (len(overlap_pass_chunks) >= 4 and overlap_median >= 0.05)
        )
        out.append(
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
                "diagnostic_separates_chunk08_from_false_positive": helpful_safe_chunks == [8] and not false_positive_chunks,
                "canary_rule_gate_pass": canary_rule_gate_pass,
                "method_gate_claimed": False,
            }
        )
    return out


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 TTT Post-Delta Region Carrier Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- core_blocker: {summary['core_blocker']}",
        "",
        "## Chunk Region Features",
        "",
        "| chunk | class | top_low | enrich_low | corr_low | top_Dq | action_minus_native |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    str(row["qscale_outcome_class"]),
                    f"{_f(row, 'action_top_low_support_frac'):.6f}",
                    f"{_f(row, 'action_top_low_support_enrichment'):.6f}",
                    f"{_f(row, 'corr_action_delta_low_support'):.6f}",
                    f"{_f(row, 'action_top_high_Dq_frac'):.6f}",
                    f"{_f(row, 'action_minus_native_delta_mean'):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rule Audit", ""])
    lines.append("| rule | selected_chunks | separates_chunk08 | gate_pass | overlap_harm |")
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
                    json.dumps(row["overlap_harm_chunks"], ensure_ascii=False),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    decisions = _decision_by_chunk(args.decisions_csv)
    rediscovery = _rediscovery_by_chunk(args.rediscovery_rows)

    rows: list[dict[str, Any]] = []
    missing_chunks: list[int] = []
    for chunk in chunks:
        if chunk not in decisions or chunk not in rediscovery:
            missing_chunks.append(chunk)
            continue
        row: dict[str, Any] = {"chunk": chunk}
        row.update(decisions[chunk])
        row.update(rediscovery[chunk])
        row["qscale_outcome_class"] = _classify(row)
        row.update(_region_features(row, args.overlap_frames, args.top_frac))
        rows.append(row)

    rule_rows = _rule_rows(rows)
    gate_pass_rules = [row["rule"] for row in rule_rows if bool(row["canary_rule_gate_pass"])]
    separator_rules = [row["rule"] for row in rule_rows if bool(row["diagnostic_separates_chunk08_from_false_positive"])]
    harm_selecting_rules = [row["rule"] for row in rule_rows if row["false_positive_overlap_harm_chunks"]]
    if gate_pass_rules:
        status = "unexpected_postdelta_region_gate_pass_requires_runtime_review"
    elif separator_rules:
        status = "postdelta_region_chunk08_local_diagnostic_only"
    else:
        status = "no_postdelta_region_separability"

    core_blocker = (
        "Per-region post-delta/support top-region statistics do not separate chunk08 from other canary chunks. "
        "Low-support top-region rules select no chunks; the broad high-Dq rule selects non-helpful and "
        "overlap-harm chunks."
        if not separator_rules
        else (
            "Per-region post-delta/support alignment can isolate chunk08 in local diagnostics, but no rule provides "
            "deployable canary coverage; broader post-delta rules select non-helpful or overlap-harm chunks."
        )
    )
    summary = {
        "schema": "acl2_v80_ttt_postdelta_region_carrier_audit_v1",
        "status": status,
        "diagnostic_only": True,
        "chunks": chunks,
        "missing_chunks": missing_chunks,
        "row_count": len(rows),
        "rule_count": len(rule_rows),
        "top_frac": args.top_frac,
        "overlap_frames": args.overlap_frames,
        "diagnostic_separator_rules": separator_rules,
        "harm_selecting_rules": harm_selecting_rules,
        "deployable_gate_pass_rules": gate_pass_rules,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "core_blocker": core_blocker,
        "next_action": (
            "Do not launch runtime from post-delta/support region thresholds alone. A viable next carrier "
            "would need additional held-out coverage or a nonlocal future-overlap signal beyond current "
            "post-delta/support top-region statistics."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "ttt_postdelta_region_carrier_summary.json", summary)
    _write_csv(args.out_dir / "ttt_postdelta_region_feature_rows.csv", rows)
    _write_csv(args.out_dir / "ttt_postdelta_region_rule_audit.csv", rule_rows)
    _write_report(args.out_dir / "ttt_postdelta_region_carrier_report.md", summary, rows, rule_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
