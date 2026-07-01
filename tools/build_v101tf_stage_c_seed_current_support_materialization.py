#!/usr/bin/env python3
"""Materialize Stage-C seed current-support diagnostics for ACL2 v101.

This script consumes the no-action SWA raw transport traces produced by
``run_v101tf_stage_c_seed_bridge_target_traces.py``.  It builds component-like
rows keyed by ``seed_global_track_idx`` and records whether sampled current
query tokens and cache top-k source tokens share the same Stage-C seed.

The output is diagnostic only.  The trace is sampled, not a full anchor
visibility pass, and it does not include validated scale observability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACE_ROOT = ROOT / "stage_c_seed_bridge_target_traces"
TRACK_T = ROOT / "trackT_drift_target_relabel"
OUT = ROOT / "trackU_true_current_support"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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
            clean: dict[str, Any] = {}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--target-csv", type=Path, default=TRACK_T / "target_universe_v101.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--output-prefix", default="stage_c_seed_current_support")
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def counter_from_tensor(value: torch.Tensor) -> Counter[int]:
    flat = value.detach().cpu().long().reshape(-1)
    return Counter(int(item) for item in flat.tolist() if int(item) >= 0)


def parse_case(case_id: str) -> tuple[str, str]:
    parts = str(case_id).split("_")
    if len(parts) >= 3:
        return parts[0], f"{int(parts[1])}->{int(parts[2])}"
    return str(case_id).split("_", 1)[0], ""


def case_id_from_trace_path(path: Path) -> str:
    return path.parents[2].name


def summarize_taxonomy(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row.get("target_taxonomy", "") or "UNKNOWN")].append(row)
    out: list[dict[str, Any]] = []
    for taxonomy, rows in sorted(grouped.items()):
        out.append(
            {
                "target_taxonomy": taxonomy,
                "case_count": len(rows),
                "same_seed_true_total": sum(int(row.get("same_seed_true_count", 0) or 0) for row in rows),
                "current_seed_unique_mean": mean([row.get("current_seed_unique_count") for row in rows]),
                "cache_seed_unique_mean": mean([row.get("cache_seed_unique_count") for row in rows]),
                "same_seed_matched_unique_mean": mean([row.get("same_seed_matched_unique_count") for row in rows]),
                "matched_current_seed_recall_mean": mean([row.get("matched_current_seed_recall") for row in rows]),
                "same_seed_frac_over_current_slots_mean": mean(
                    [row.get("same_seed_frac_over_current_slots") for row in rows]
                ),
                "same_seed_frac_over_nonnegative_topk_mean": mean(
                    [row.get("same_seed_frac_over_nonnegative_topk") for row in rows]
                ),
                "L3_mean": mean([row.get("L3_handoff_transfer_penalty_proxy") for row in rows]),
            }
        )
    return out


def diagnostic_fpfn(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows = [
        row
        for row in case_rows
        if row.get("target_taxonomy") in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}
        and math.isfinite(f(row.get("same_seed_frac_over_current_slots")))
    ]
    positives = [row for row in eval_rows if row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET"]
    safe = [row for row in eval_rows if row.get("target_taxonomy") == "SAFE_GOOD"]
    if not positives or not safe:
        return [
            {
                "cue_name": "stage_c_seed_low_same_seed_support",
                "row_kind": "not_enough_target_or_safe_cases",
                "positive_case_count": len(positives),
                "safe_good_case_count": len(safe),
                "claim_level": "diagnostic_no_action",
            }
        ]
    ranked = sorted(eval_rows, key=lambda row: f(row.get("same_seed_frac_over_current_slots")))
    selected = {row["case_id"] for row in ranked[: len(positives)]}
    rows: list[dict[str, Any]] = []
    for row in eval_rows:
        if row["case_id"] in selected and row.get("target_taxonomy") == "SAFE_GOOD":
            row_kind = "false_positive_safe_good"
        elif row["case_id"] not in selected and row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET":
            row_kind = "missed_positive_handoff"
        elif row["case_id"] in selected:
            row_kind = "true_positive_handoff"
        else:
            row_kind = "true_negative_safe_good"
        rows.append(
            {
                "cue_name": "stage_c_seed_low_same_seed_support",
                "row_kind": row_kind,
                "case_id": row.get("case_id", ""),
                "seq": row.get("seq", ""),
                "target_taxonomy": row.get("target_taxonomy", ""),
                "same_seed_frac_over_current_slots": row.get("same_seed_frac_over_current_slots", ""),
                "matched_current_seed_recall": row.get("matched_current_seed_recall", ""),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "claim_level": "sampled_stage_c_seed_support_diagnostic_no_action",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    trace_root = args.trace_root
    out_dir = args.output_dir
    prefix = args.output_prefix
    target_by_case = {row.get("case_id", ""): row for row in read_rows(args.target_csv)}
    job_by_case = {row.get("case_id", ""): row for row in read_rows(trace_root / "job_results.csv")}
    trace_paths = sorted(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))

    seed_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for path in trace_paths:
        case_id = case_id_from_trace_path(path)
        target = target_by_case.get(case_id, {})
        job = job_by_case.get(case_id, {})
        seq, boundary_id = parse_case(case_id)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"case_id": case_id, "trace_payload_path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue

        sample = payload.get("sampled_query_stage_c_seed_global_track_idx")
        topk = payload.get("current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx")
        same = payload.get("current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx")
        scores = payload.get("current_Q_to_cache_K_topk_scores")
        if not (torch.is_tensor(sample) and torch.is_tensor(topk) and torch.is_tensor(same)):
            load_errors.append(
                {
                    "case_id": case_id,
                    "trace_payload_path": str(path),
                    "error": "missing_required_seed_tensors",
                }
            )
            continue

        sample = sample.detach().cpu().long()
        topk = topk.detach().cpu().long()
        same = same.detach().cpu().bool()
        # Trace top-k tensors are [batch, head, sampled_query, topk].
        # Sampled current seeds are [batch, sampled_query].
        current_expanded = sample[:, None, :, None].expand_as(topk)
        current_nonneg = current_expanded >= 0
        topk_nonneg = topk >= 0
        same_valid = same & current_nonneg & topk_nonneg

        current_counts = counter_from_tensor(sample)
        cache_counts = counter_from_tensor(topk)
        current_slot_counts = counter_from_tensor(current_expanded[current_nonneg])
        same_counts = counter_from_tensor(current_expanded[same_valid])
        all_seed_ids = sorted(set(current_counts) | set(cache_counts))
        score_sum: Counter[int] = Counter()
        score_count: Counter[int] = Counter()
        same_score_sum: Counter[int] = Counter()
        same_score_count: Counter[int] = Counter()
        if torch.is_tensor(scores):
            scores_cpu = scores.detach().cpu().float()
            for seed in all_seed_ids:
                seed_mask = topk == seed
                if bool(seed_mask.any()):
                    score_sum[seed] = float(scores_cpu[seed_mask].sum().item())
                    score_count[seed] = int(seed_mask.sum().item())
                same_seed_mask = same_valid & (current_expanded == seed)
                if bool(same_seed_mask.any()):
                    same_score_sum[seed] = float(scores_cpu[same_seed_mask].sum().item())
                    same_score_count[seed] = int(same_seed_mask.sum().item())

        for seed_id in all_seed_ids:
            current_count = int(current_counts.get(seed_id, 0))
            cache_count = int(cache_counts.get(seed_id, 0))
            same_count = int(same_counts.get(seed_id, 0))
            current_slot_count = int(current_slot_counts.get(seed_id, 0))
            if current_count and cache_count:
                status = "current_and_cache"
            elif current_count:
                status = "current_only"
            else:
                status = "cache_only"
            if same_count:
                quality = "sampled_current_cache_seed_match"
            elif current_count and cache_count:
                quality = "sampled_current_seed_seen_cache_no_same_topk_match"
            elif current_count:
                quality = "sampled_current_seed_only"
            else:
                quality = "cache_seed_only_no_current_sample"
            seed_rows.append(
                {
                    "case_id": case_id,
                    "seq": target.get("seq", seq),
                    "boundary_id": boundary_id,
                    "target_taxonomy": target.get("target_taxonomy", ""),
                    "stage_c_seed_global_track_idx": seed_id,
                    "seed_presence_status": status,
                    "current_sample_token_count": current_count,
                    "cache_topk_token_count": cache_count,
                    "current_topk_slot_count": current_slot_count,
                    "same_seed_topk_true_count": same_count,
                    "current_seed_topk_same_frac": safe_div(same_count, current_slot_count),
                    "cache_seed_same_frac": safe_div(same_count, cache_count),
                    "cache_topk_score_mean": safe_div(score_sum[seed_id], score_count[seed_id]),
                    "same_seed_topk_score_mean": safe_div(same_score_sum[seed_id], same_score_count[seed_id]),
                    "current_component_visible_in_sample": current_count > 0,
                    "cache_component_available_in_topk": cache_count > 0,
                    "current_cache_same_seed_supported": same_count > 0,
                    "support_quality": quality,
                    "support_source_flags": "stage_c_seed_global_track_idx;sampled_swa_raw_transport_topk",
                    "identity_resolution_level": "stage_c_seed_component_sampled_trace",
                    "proxy_only": True,
                    "strict_current_support_pass": False,
                    "runtime_action_allowed": False,
                    "trace_payload_path": str(path),
                }
            )

        sample_nonnegative_count = int((sample >= 0).sum().item())
        topk_nonnegative_count = int(topk_nonneg.sum().item())
        same_seed_true_count = int(same_valid.sum().item())
        current_seed_unique = len(current_counts)
        cache_seed_unique = len(cache_counts)
        matched_seed_unique = len([seed for seed, count in same_counts.items() if count > 0])
        case_rows.append(
            {
                "case_id": case_id,
                "seq": target.get("seq", seq),
                "boundary_id": boundary_id,
                "target_taxonomy": target.get("target_taxonomy", ""),
                "case_label": target.get("case_label", ""),
                "failure_type": target.get("failure_type", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
                "returncode": job.get("returncode", ""),
                "trace_payload_path": str(path),
                "sampled_query_count": payload.get("sampled_query_count", ""),
                "head_count": payload.get("head_count", ""),
                "topk": payload.get("topk_identity_topk", ""),
                "sample_nonnegative_count": sample_nonnegative_count,
                "topk_nonnegative_count": topk_nonnegative_count,
                "same_seed_true_count": same_seed_true_count,
                "current_seed_unique_count": current_seed_unique,
                "cache_seed_unique_count": cache_seed_unique,
                "current_cache_seed_overlap_unique_count": len(set(current_counts) & set(cache_counts)),
                "same_seed_matched_unique_count": matched_seed_unique,
                "matched_current_seed_recall": safe_div(matched_seed_unique, current_seed_unique),
                "matched_cache_seed_frac": safe_div(matched_seed_unique, cache_seed_unique),
                "same_seed_frac_payload_mean": payload.get(
                    "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_frac_mean",
                    "",
                ),
                "same_seed_frac_over_all_topk": safe_div(same_seed_true_count, int(topk.numel())),
                "same_seed_frac_over_nonnegative_topk": safe_div(same_seed_true_count, topk_nonnegative_count),
                "same_seed_frac_over_current_slots": safe_div(same_seed_true_count, int(current_nonneg.sum().item())),
                "stage_c_seed_current_support_available": sample_nonnegative_count > 0,
                "stage_c_seed_cache_support_available": topk_nonnegative_count > 0,
                "stage_c_seed_same_seed_support_available": same_seed_true_count > 0,
                "support_source_level": "sampled_swa_raw_transport_topk_seed_trace",
                "proxy_only": True,
                "strict_current_support_pass": False,
                "runtime_action_allowed": False,
            }
        )

    taxonomy_rows = summarize_taxonomy(case_rows)
    fpfn_rows = diagnostic_fpfn(case_rows)
    expected_cases = set(job_by_case) if job_by_case else set(target_by_case)
    observed_cases = {row["case_id"] for row in case_rows}
    handoff_cases = [row for row in case_rows if row.get("target_taxonomy") == "HANDOFF_SCALE_GAUGE_TARGET"]
    safe_cases = [row for row in case_rows if row.get("target_taxonomy") == "SAFE_GOOD"]
    target_eval_cases = handoff_cases + safe_cases
    seed_support_materialization_pass = (
        bool(expected_cases)
        and len(case_rows) == len(expected_cases)
        and not load_errors
        and bool(seed_rows)
        and all(row.get("stage_c_seed_current_support_available") is True for row in case_rows)
        and all(row.get("stage_c_seed_cache_support_available") is True for row in case_rows)
        and all(row.get("stage_c_seed_same_seed_support_available") is True for row in case_rows)
    )
    summary = {
        "schema": "acl2_v101_stage_c_seed_current_support_materialization_v1",
        "status": "complete_sampled_seed_support_diagnostic" if case_rows else "blocked_missing_target_traces",
        "seed_support_materialization_pass": seed_support_materialization_pass,
        "true_current_support_strict_pass": False,
        "stage_c_seed_support_discriminative_gate_pass": False,
        "proxy_only": True,
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "trace_root": str(trace_root),
        "target_csv": str(args.target_csv),
        "output_prefix": prefix,
        "target_case_count": len(expected_cases),
        "trace_payload_file_count": len(trace_paths),
        "case_count": len(case_rows),
        "missing_case_count": len(expected_cases - observed_cases),
        "missing_cases": ";".join(sorted(expected_cases - observed_cases)),
        "load_error_count": len(load_errors),
        "component_support_row_count": len(seed_rows),
        "case_sample_nonnegative_min": min([int(row["sample_nonnegative_count"]) for row in case_rows], default=0),
        "case_topk_nonnegative_min": min([int(row["topk_nonnegative_count"]) for row in case_rows], default=0),
        "same_seed_true_total": sum(int(row["same_seed_true_count"]) for row in case_rows),
        "same_seed_frac_over_current_slots_mean": mean(
            [row.get("same_seed_frac_over_current_slots") for row in case_rows]
        ),
        "matched_current_seed_recall_mean": mean([row.get("matched_current_seed_recall") for row in case_rows]),
        "taxonomy_summary": {row["target_taxonomy"]: row for row in taxonomy_rows},
        "handoff_target_case_count": len(handoff_cases),
        "safe_good_case_count": len(safe_cases),
        "eval_case_count": len(target_eval_cases),
        "same_seed_frac_corr_L3_all_cases": pearson(
            [row.get("same_seed_frac_over_current_slots") for row in case_rows],
            [row.get("L3_handoff_transfer_penalty_proxy") for row in case_rows],
        ),
        "same_seed_frac_corr_L3_handoff_vs_safe_only": pearson(
            [row.get("same_seed_frac_over_current_slots") for row in target_eval_cases],
            [row.get("L3_handoff_transfer_penalty_proxy") for row in target_eval_cases],
        ),
        "blocker": (
            "Materialized Stage-C seed sampled current/cache support, but this is not strict Track U: "
            "trace is sampled, visibility/overlap/scale observability remain incomplete, and Track T/Q2 true-stage are blocked."
        ),
        "claim": "Stage-C seed component support rows are diagnostic sampled trace evidence only; no runtime action is authorized.",
    }

    write_rows(out_dir / f"{prefix}_rows.csv", seed_rows)
    write_rows(out_dir / f"{prefix}_case_rows.csv", case_rows)
    write_rows(out_dir / f"{prefix}_taxonomy_summary.csv", taxonomy_rows)
    write_rows(out_dir / f"{prefix}_fpfn_rows.csv", fpfn_rows)
    write_rows(out_dir / f"{prefix}_load_errors.csv", load_errors)
    write_json(out_dir / f"{prefix}_summary.json", summary)
    write_text(
        out_dir / f"{prefix}_report.md",
        "# Stage-C Seed Current-Support Materialization\n\n"
        f"- Seed support materialization pass: {summary['seed_support_materialization_pass']}\n"
        f"- Strict current support pass: {summary['true_current_support_strict_pass']}\n"
        f"- Case count: {summary['case_count']}\n"
        f"- Component support rows: {summary['component_support_row_count']}\n"
        f"- Same-seed true total: {summary['same_seed_true_total']}\n"
        f"- Mean same-seed/current-slot fraction: {summary['same_seed_frac_over_current_slots_mean']}\n"
        f"- Mean matched current-seed recall: {summary['matched_current_seed_recall_mean']}\n"
        f"- Runtime action allowed: {summary['runtime_action_allowed']}\n\n"
        "This uses sampled no-action SWA trace payloads. It is useful as component-level provenance evidence, "
        "but it is not full visibility, geometry support, scale observability, Q2 true-stage, M4, or runtime success.\n",
    )
    write_text(
        out_dir / f"{prefix}_missing_report.md",
        "No target trace case is missing if missing_case_count is 0. Remaining missing evidence is full anchor visibility, "
        "boundary overlap support, and validated scale observability beyond sampled SWA top-k seed matches.",
    )
    write_text(
        out_dir / f"{prefix}_next_attempt_recommendation.md",
        "Join sampled Stage-C seed support with anchor-level same-space rows only when anchor_id to seed provenance is explicit; "
        "otherwise use these rows as a diagnostic component-support sidecar and prioritize full visibility/observability dumps.",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
