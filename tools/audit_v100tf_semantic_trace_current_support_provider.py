#!/usr/bin/env python3
"""Evaluate v100 trace-native semantic current-support fields.

Reads SWA raw transport traces that contain sampled query labels and cache
top-k labels in SemanticPriorGenerator's stable fine/group ID spaces.  The
audit is diagnostic-only; it does not authorize runtime action.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v100tf_same_space_semantic_anchor_latent_state_multiroute_memory_control import (
    case_id_from_trace,
    evaluate_pattern,
    f,
    load_v99_case_labels,
    write_json,
    write_rows,
)


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
DEFAULT_TRACE_ROOT = ROOT / "trackS_same_space_latent_state/probe28_semantic_trace_labels"
OUT_DIR = ROOT / "trackD4_read_current_support_provider"


def clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if torch.is_tensor(value):
        return clean(value.detach().cpu().tolist())
    return value


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else math.nan


def trace_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_v99_case_labels()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(trace_root.glob("**/swa_raw_transport_trace/*.pt")):
        case_id = case_id_from_trace(path)
        if case_id not in labels:
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append({"trace_payload": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"trace_payload": str(path), "error": "payload_not_dict"})
            continue
        same_fine = payload.get("current_Q_to_cache_K_topk_same_fine_label")
        same_group = payload.get("current_Q_to_cache_K_topk_same_group")
        stable_hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        stable_same_fine = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_fine_label")
        stable_same_group = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_group")
        query_fine = payload.get("sampled_query_fine_label_ids")
        cache_fine = payload.get("current_Q_to_cache_K_topk_cache_fine_label_ids")
        required = [same_fine, same_group, stable_hit, stable_same_fine, stable_same_group, query_fine, cache_fine]
        if not all(torch.is_tensor(item) for item in required):
            errors.append({"trace_payload": str(path), "error": "missing_semantic_trace_tensor"})
            continue
        same_fine = same_fine.detach().cpu().bool()
        same_group = same_group.detach().cpu().bool()
        stable_hit = stable_hit.detach().cpu().bool()
        stable_same_fine = stable_same_fine.detach().cpu().bool()
        stable_same_group = stable_same_group.detach().cpu().bool()
        query_fine = query_fine.detach().cpu().long()
        cache_fine = cache_fine.detach().cpu().long()
        stable_count = int(stable_hit.sum().item())
        stable_same_fine_count = int(stable_same_fine.sum().item())
        stable_same_group_count = int(stable_same_group.sum().item())
        valid_query = int((query_fine > 0).sum().item())
        valid_cache = int((cache_fine > 0).sum().item())
        rows.append({
            **labels[case_id],
            "trace_payload": str(path),
            "chunk_idx": payload.get("chunk_idx", ""),
            "swa_layer_idx": payload.get("swa_layer_idx", ""),
            "semantic_trace_fine_available": bool(payload.get("current_semantic_fine_trace_available")),
            "semantic_trace_group_available": bool(payload.get("current_semantic_group_trace_available")),
            "sampled_query_valid_fine_label_frac": ratio(valid_query, int(query_fine.numel())),
            "topk_cache_valid_fine_label_frac": ratio(valid_cache, int(cache_fine.numel())),
            "same_fine_topk_frac": float(same_fine.float().mean().item()) if same_fine.numel() else math.nan,
            "same_group_topk_frac": float(same_group.float().mean().item()) if same_group.numel() else math.nan,
            "stable_anchor_topk_hit_frac": float(stable_hit.float().mean().item()) if stable_hit.numel() else math.nan,
            "stable_anchor_same_fine_topk_frac": float(stable_same_fine.float().mean().item()) if stable_same_fine.numel() else math.nan,
            "stable_anchor_same_group_topk_frac": float(stable_same_group.float().mean().item()) if stable_same_group.numel() else math.nan,
            "stable_anchor_same_fine_given_stable_frac": ratio(stable_same_fine_count, stable_count),
            "stable_anchor_same_group_given_stable_frac": ratio(stable_same_group_count, stable_count),
            "stable_anchor_topk_hit_count": stable_count,
            "stable_anchor_same_fine_count": stable_same_fine_count,
            "stable_anchor_same_group_count": stable_same_group_count,
        })
    return rows, errors


def case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    fields = [
        "sampled_query_valid_fine_label_frac",
        "topk_cache_valid_fine_label_frac",
        "same_fine_topk_frac",
        "same_group_topk_frac",
        "stable_anchor_topk_hit_frac",
        "stable_anchor_same_fine_topk_frac",
        "stable_anchor_same_group_topk_frac",
        "stable_anchor_same_fine_given_stable_frac",
        "stable_anchor_same_group_given_stable_frac",
    ]
    for case_id, parts in sorted(grouped.items()):
        base = parts[0]
        stable_hit_total = sum(int(f(part.get("stable_anchor_topk_hit_count"), 0)) for part in parts)
        stable_same_fine_total = sum(int(f(part.get("stable_anchor_same_fine_count"), 0)) for part in parts)
        stable_same_group_total = sum(int(f(part.get("stable_anchor_same_group_count"), 0)) for part in parts)
        row: dict[str, Any] = {
            "case_id": case_id,
            "seq": base.get("seq", ""),
            "case_label": base.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy", math.nan),
            "trace_payload_count": len(parts),
            "trace_layer_count": len({str(part.get("swa_layer_idx", "")) for part in parts}),
            "stable_anchor_topk_hit_count_total": stable_hit_total,
            "stable_anchor_same_fine_count_total": stable_same_fine_total,
            "stable_anchor_same_group_count_total": stable_same_group_total,
            "stable_anchor_same_fine_given_stable_frac_weighted": ratio(stable_same_fine_total, stable_hit_total),
            "stable_anchor_same_group_given_stable_frac_weighted": ratio(stable_same_group_total, stable_hit_total),
        }
        for field in fields:
            row[field] = mean([f(part.get(field)) for part in parts])
        row["low_stable_anchor_fine_current_support_risk"] = (
            1.0 - row["stable_anchor_same_fine_given_stable_frac_weighted"]
            if math.isfinite(f(row.get("stable_anchor_same_fine_given_stable_frac_weighted")))
            else math.nan
        )
        row["low_stable_anchor_group_current_support_risk"] = (
            1.0 - row["stable_anchor_same_group_given_stable_frac_weighted"]
            if math.isfinite(f(row.get("stable_anchor_same_group_given_stable_frac_weighted")))
            else math.nan
        )
        out.append(row)
    return out


def score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("same_fine_topk_frac", "higher_bad"),
        ("same_fine_topk_frac", "lower_bad"),
        ("same_group_topk_frac", "higher_bad"),
        ("same_group_topk_frac", "lower_bad"),
        ("stable_anchor_topk_hit_frac", "higher_bad"),
        ("stable_anchor_same_fine_topk_frac", "higher_bad"),
        ("stable_anchor_same_fine_topk_frac", "lower_bad"),
        ("stable_anchor_same_group_topk_frac", "higher_bad"),
        ("stable_anchor_same_group_topk_frac", "lower_bad"),
        ("stable_anchor_same_fine_given_stable_frac_weighted", "higher_bad"),
        ("stable_anchor_same_fine_given_stable_frac_weighted", "lower_bad"),
        ("stable_anchor_same_group_given_stable_frac_weighted", "higher_bad"),
        ("stable_anchor_same_group_given_stable_frac_weighted", "lower_bad"),
        ("low_stable_anchor_fine_current_support_risk", "higher_bad"),
        ("low_stable_anchor_group_current_support_risk", "higher_bad"),
    ]
    metrics: list[dict[str, Any]] = []
    for field, direction in specs:
        metric = evaluate_pattern(rows, f"{field}_{direction}", field, direction)
        metric["field"] = field
        metric["direction"] = direction
        metric["gate_like"] = bool(
            f(metric.get("bad_recall")) >= 0.65
            and f(metric.get("good_FPR"), 1.0) <= 0.25
            and f(metric.get("abs_corr_L3")) >= 0.50
            and bool(metric.get("corr_direction_correct"))
            and int(f(metric.get("sequence_coverage"), 0)) >= 4
            and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
        )
        metrics.append(metric)
    metrics.sort(
        key=lambda row: (
            bool(row.get("gate_like")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("abs_corr_L3")),
        ),
        reverse=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rows, errors = trace_rows(args.trace_root)
    cases = case_rows(rows)
    metrics = score(cases)
    gate_like = [row for row in metrics if row.get("gate_like")]
    summary = {
        "trace_root": str(args.trace_root),
        "trace_payload_count": len(rows),
        "trace_read_error_count": len(errors),
        "case_count": len(cases),
        "metrics": len(metrics),
        "gate_like_count": len(gate_like),
        "best": metrics[0] if metrics else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "runtime_action_allowed": False,
        "note": "Trace-native semantic current-support provider audit; missing true rerun controls and no runtime action.",
    }
    out = args.out_dir
    write_rows(out / "semantic_trace_provider_trace_rows.csv", rows)
    write_rows(out / "semantic_trace_provider_case_rows.csv", cases)
    write_rows(out / "semantic_trace_provider_metrics.csv", metrics)
    write_rows(out / "semantic_trace_provider_read_errors.csv", errors)
    write_json(out / "semantic_trace_provider_summary.json", summary)
    report = [
        "# Semantic Trace Current-Support Provider Audit",
        "",
        f"- trace_root: `{args.trace_root}`",
        f"- trace_payload_count: `{len(rows)}`",
        f"- trace_read_error_count: `{len(errors)}`",
        f"- case_count: `{len(cases)}`",
        f"- gate_like_count: `{len(gate_like)}`",
        "",
        "## Best Metric",
        "",
        "```json",
        json.dumps(clean(metrics[0] if metrics else {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Best Gate-Like",
        "",
        "```json",
        json.dumps(clean(gate_like[0] if gate_like else {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "No runtime action is allowed from this audit alone.",
    ]
    (out / "semantic_trace_provider_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
