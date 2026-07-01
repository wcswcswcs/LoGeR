#!/usr/bin/env python3
"""Head/layer diagnostic for v100 trace-native semantic current support.

Case-level averages can hide query/head-local stale-anchor use.  This audit
keeps the SWA layer/head axes and derives head-local current-support risks from
trace-native fine/group label matches.  It remains diagnostic-only.
"""

from __future__ import annotations

import argparse
import csv
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
    quantile,
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


def top_mean(values: list[float], k: int) -> float:
    vals = sorted([float(v) for v in values if math.isfinite(float(v))], reverse=True)
    return mean(vals[: max(1, int(k))])


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else math.nan


def read_head_layer_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        stable_hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        same_fine = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_fine_label")
        same_group = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_group")
        all_same_fine = payload.get("current_Q_to_cache_K_topk_same_fine_label")
        all_same_group = payload.get("current_Q_to_cache_K_topk_same_group")
        if not all(torch.is_tensor(x) for x in [stable_hit, same_fine, same_group, all_same_fine, all_same_group]):
            errors.append({"trace_payload": str(path), "error": "missing_head_layer_semantic_tensor"})
            continue
        stable_hit = stable_hit.detach().cpu().bool()
        same_fine = same_fine.detach().cpu().bool()
        same_group = same_group.detach().cpu().bool()
        all_same_fine = all_same_fine.detach().cpu().bool()
        all_same_group = all_same_group.detach().cpu().bool()
        if stable_hit.ndim != 4:
            errors.append({"trace_payload": str(path), "error": f"bad_stable_hit_ndim_{stable_hit.ndim}"})
            continue
        _, head_count, _, _ = stable_hit.shape
        try:
            layer = int(payload.get("swa_layer_idx", -1))
        except Exception:
            layer = -1
        for head_idx in range(int(head_count)):
            stable_h = stable_hit[:, head_idx]
            fine_h = same_fine[:, head_idx]
            group_h = same_group[:, head_idx]
            all_fine_h = all_same_fine[:, head_idx]
            all_group_h = all_same_group[:, head_idx]
            stable_count = int(stable_h.sum().item())
            fine_count = int(fine_h.sum().item())
            group_count = int(group_h.sum().item())
            stable_frac = float(stable_h.float().mean().item()) if stable_h.numel() else math.nan
            fine_given = ratio(fine_count, stable_count)
            group_given = ratio(group_count, stable_count)
            low_fine_risk = stable_frac * (1.0 - fine_given) if math.isfinite(fine_given) else math.nan
            low_group_risk = stable_frac * (1.0 - group_given) if math.isfinite(group_given) else math.nan
            rows.append({
                **labels[case_id],
                "trace_payload": str(path),
                "chunk_idx": payload.get("chunk_idx", ""),
                "swa_layer_idx": layer,
                "head_idx": int(head_idx),
                "stable_anchor_topk_hit_frac": stable_frac,
                "stable_anchor_topk_hit_count": stable_count,
                "stable_anchor_same_fine_count": fine_count,
                "stable_anchor_same_group_count": group_count,
                "stable_anchor_same_fine_given_stable_frac": fine_given,
                "stable_anchor_same_group_given_stable_frac": group_given,
                "head_low_fine_current_support_risk": low_fine_risk,
                "head_low_group_current_support_risk": low_group_risk,
                "all_topk_same_fine_frac": float(all_fine_h.float().mean().item()) if all_fine_h.numel() else math.nan,
                "all_topk_same_group_frac": float(all_group_h.float().mean().item()) if all_group_h.numel() else math.nan,
            })
    return rows, errors


def aggregate_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        base = parts[0]
        stable = [f(row.get("stable_anchor_topk_hit_frac")) for row in parts]
        fine_given = [f(row.get("stable_anchor_same_fine_given_stable_frac")) for row in parts]
        group_given = [f(row.get("stable_anchor_same_group_given_stable_frac")) for row in parts]
        fine_risk = [f(row.get("head_low_fine_current_support_risk")) for row in parts]
        group_risk = [f(row.get("head_low_group_current_support_risk")) for row in parts]
        all_fine = [f(row.get("all_topk_same_fine_frac")) for row in parts]
        all_group = [f(row.get("all_topk_same_group_frac")) for row in parts]
        stable_counts = [f(row.get("stable_anchor_topk_hit_count")) for row in parts]
        stable_total = sum(v for v in stable_counts if math.isfinite(v))
        fine_total = sum(f(row.get("stable_anchor_same_fine_count")) for row in parts if math.isfinite(f(row.get("stable_anchor_same_fine_count"))))
        group_total = sum(f(row.get("stable_anchor_same_group_count")) for row in parts if math.isfinite(f(row.get("stable_anchor_same_group_count"))))
        high_stable_units = [
            row for row in parts
            if f(row.get("stable_anchor_topk_hit_frac")) >= quantile(stable, 0.75)
        ]
        low_fine_high_stable = [
            row for row in high_stable_units
            if f(row.get("stable_anchor_same_fine_given_stable_frac"), 1.0) <= quantile(fine_given, 0.25)
        ]
        row = {
            "case_id": case_id,
            "seq": base.get("seq", ""),
            "case_label": base.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy", math.nan),
            "head_layer_unit_count": len(parts),
            "stable_anchor_topk_hit_frac_mean": mean(stable),
            "stable_anchor_topk_hit_frac_max": max([v for v in stable if math.isfinite(v)], default=math.nan),
            "stable_anchor_topk_hit_frac_top3_mean": top_mean(stable, 3),
            "stable_anchor_same_fine_given_stable_frac_mean": mean(fine_given),
            "stable_anchor_same_fine_given_stable_frac_min": min([v for v in fine_given if math.isfinite(v)], default=math.nan),
            "stable_anchor_same_group_given_stable_frac_mean": mean(group_given),
            "stable_anchor_same_group_given_stable_frac_min": min([v for v in group_given if math.isfinite(v)], default=math.nan),
            "head_low_fine_current_support_risk_mean": mean(fine_risk),
            "head_low_fine_current_support_risk_max": max([v for v in fine_risk if math.isfinite(v)], default=math.nan),
            "head_low_fine_current_support_risk_top3_mean": top_mean(fine_risk, 3),
            "head_low_fine_current_support_risk_top5_mean": top_mean(fine_risk, 5),
            "head_low_group_current_support_risk_mean": mean(group_risk),
            "head_low_group_current_support_risk_max": max([v for v in group_risk if math.isfinite(v)], default=math.nan),
            "head_low_group_current_support_risk_top3_mean": top_mean(group_risk, 3),
            "all_topk_same_fine_frac_mean": mean(all_fine),
            "all_topk_same_fine_frac_min": min([v for v in all_fine if math.isfinite(v)], default=math.nan),
            "all_topk_same_group_frac_mean": mean(all_group),
            "all_topk_same_group_frac_min": min([v for v in all_group if math.isfinite(v)], default=math.nan),
            "stable_anchor_same_fine_given_stable_frac_weighted": ratio(fine_total, stable_total),
            "stable_anchor_same_group_given_stable_frac_weighted": ratio(group_total, stable_total),
            "low_fine_high_stable_head_layer_frac": ratio(len(low_fine_high_stable), len(parts)),
        }
        for layer in sorted({int(f(part.get("swa_layer_idx"), -1)) for part in parts}):
            layer_parts = [part for part in parts if int(f(part.get("swa_layer_idx"), -2)) == layer]
            lr = [f(part.get("head_low_fine_current_support_risk")) for part in layer_parts]
            row[f"layer{layer}_low_fine_risk_max"] = max([v for v in lr if math.isfinite(v)], default=math.nan)
            row[f"layer{layer}_low_fine_risk_mean"] = mean(lr)
        out.append(row)
    return out


def score(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = []
    for row in case_rows:
        for key in row:
            if key.startswith(("stable_", "head_", "all_topk_", "low_fine_", "layer")) and key not in fields:
                fields.append(key)
    metrics: list[dict[str, Any]] = []
    for field in fields:
        for direction in ("higher_bad", "lower_bad"):
            metric = evaluate_pattern(case_rows, f"{field}_{direction}", field, direction)
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
    head_rows, errors = read_head_layer_rows(args.trace_root)
    case_rows = aggregate_case_rows(head_rows)
    metrics = score(case_rows)
    gate_like = [row for row in metrics if row.get("gate_like")]
    summary = {
        "trace_root": str(args.trace_root),
        "head_layer_rows": len(head_rows),
        "case_count": len(case_rows),
        "read_error_count": len(errors),
        "metrics": len(metrics),
        "gate_like_count": len(gate_like),
        "best": metrics[0] if metrics else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "runtime_action_allowed": False,
        "note": "Head/layer trace-native semantic current-support diagnostic; no true rerun controls or runtime action.",
    }
    out = args.out_dir
    write_rows(out / "semantic_trace_head_layer_rows.csv", head_rows)
    write_rows(out / "semantic_trace_head_layer_case_rows.csv", case_rows)
    write_rows(out / "semantic_trace_head_layer_metrics.csv", metrics)
    write_rows(out / "semantic_trace_head_layer_read_errors.csv", errors)
    write_json(out / "semantic_trace_head_layer_summary.json", summary)
    report = [
        "# Semantic Trace Head/Layer Current-Support Audit",
        "",
        f"- trace_root: `{args.trace_root}`",
        f"- head_layer_rows: `{len(head_rows)}`",
        f"- read_error_count: `{len(errors)}`",
        f"- case_count: `{len(case_rows)}`",
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
        "No runtime action is allowed from this diagnostic-only audit.",
    ]
    (out / "semantic_trace_head_layer_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
