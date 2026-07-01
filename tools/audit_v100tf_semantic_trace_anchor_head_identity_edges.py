#!/usr/bin/env python3
"""Anchor/head identity-edge audit for v100 trace-native semantic support.

This keeps anchor_id, SWA layer, and head instead of only case or head/layer
aggregates.  It targets N2 Pattern N-E: query/head concentrated stale anchor
use.  Diagnostic-only; no runtime action is authorized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
OUT_DIR = ROOT / "trackN2_anchor_identity_graph"


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


def collect_anchor_head_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_v99_case_labels()
    accum: dict[tuple[str, int, int, int], dict[str, Any]] = {}
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
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        same_fine = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_fine_label")
        same_group = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_group")
        cache_fine = payload.get("current_Q_to_cache_K_topk_cache_fine_label_ids")
        if not all(torch.is_tensor(x) for x in [hit, anchor_ids, same_fine, same_group, cache_fine]):
            errors.append({"trace_payload": str(path), "error": "missing_anchor_head_tensor"})
            continue
        hit = hit.detach().cpu().bool()
        anchor_ids = anchor_ids.detach().cpu().long()
        same_fine = same_fine.detach().cpu().bool()
        same_group = same_group.detach().cpu().bool()
        cache_fine = cache_fine.detach().cpu().long()
        if hit.shape != anchor_ids.shape or hit.shape != same_fine.shape or hit.shape != same_group.shape:
            errors.append({"trace_payload": str(path), "error": "shape_mismatch"})
            continue
        try:
            layer = int(payload.get("swa_layer_idx", -1))
        except Exception:
            layer = -1
        nz = torch.nonzero(hit & (anchor_ids >= 0), as_tuple=False)
        for idx in nz.tolist():
            _b, head_idx, _q, _k = [int(v) for v in idx]
            anchor_id = int(anchor_ids[_b, head_idx, _q, _k].item())
            key = (case_id, layer, head_idx, anchor_id)
            if key not in accum:
                accum[key] = {
                    **labels[case_id],
                    "swa_layer_idx": layer,
                    "head_idx": head_idx,
                    "anchor_id": anchor_id,
                    "hit_count": 0,
                    "same_fine_count": 0,
                    "same_group_count": 0,
                    "source_fine_label_counts": Counter(),
                }
            row = accum[key]
            row["hit_count"] += 1
            row["same_fine_count"] += int(bool(same_fine[_b, head_idx, _q, _k].item()))
            row["same_group_count"] += int(bool(same_group[_b, head_idx, _q, _k].item()))
            row["source_fine_label_counts"][int(cache_fine[_b, head_idx, _q, _k].item())] += 1

    rows: list[dict[str, Any]] = []
    for row in accum.values():
        hit_count = int(row["hit_count"])
        same_fine_count = int(row["same_fine_count"])
        same_group_count = int(row["same_group_count"])
        label_counts: Counter[int] = row.pop("source_fine_label_counts")
        dominant_label, dominant_count = label_counts.most_common(1)[0] if label_counts else (-1, 0)
        same_fine_frac = ratio(same_fine_count, hit_count)
        same_group_frac = ratio(same_group_count, hit_count)
        row.update({
            "same_fine_frac": same_fine_frac,
            "same_group_frac": same_group_frac,
            "low_fine_support_risk": hit_count * (1.0 - same_fine_frac) if math.isfinite(same_fine_frac) else math.nan,
            "low_group_support_risk": hit_count * (1.0 - same_group_frac) if math.isfinite(same_group_frac) else math.nan,
            "dominant_source_fine_label": dominant_label,
            "dominant_source_fine_label_frac": ratio(dominant_count, hit_count),
        })
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("case_id")), int(f(r.get("swa_layer_idx"), -1)), int(f(r.get("head_idx"), -1)), -int(f(r.get("hit_count"), 0))))
    return rows, errors


def aggregate_case_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        base = parts[0]
        hit_counts = [f(row.get("hit_count")) for row in parts]
        fine = [f(row.get("same_fine_frac")) for row in parts]
        group = [f(row.get("same_group_frac")) for row in parts]
        fine_risk = [f(row.get("low_fine_support_risk")) for row in parts]
        group_risk = [f(row.get("low_group_support_risk")) for row in parts]
        label_counts = Counter(str(row.get("dominant_source_fine_label")) for row in parts)
        total_hits = sum(v for v in hit_counts if math.isfinite(v))
        top_anchor_hits = max([v for v in hit_counts if math.isfinite(v)], default=math.nan)
        high_hit_thr = quantile(hit_counts, 0.75)
        low_fine_thr = quantile(fine, 0.25)
        high_hit_low_fine = [
            row for row in parts
            if f(row.get("hit_count")) >= high_hit_thr and f(row.get("same_fine_frac"), 1.0) <= low_fine_thr
        ]
        row = {
            "case_id": case_id,
            "seq": base.get("seq", ""),
            "case_label": base.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy", math.nan),
            "anchor_head_edge_count": len(parts),
            "anchor_head_unique_anchor_count": len({str(row.get("anchor_id")) for row in parts}),
            "anchor_head_total_stable_hit_count": total_hits,
            "anchor_head_top_anchor_hit_count": top_anchor_hits,
            "anchor_head_top_anchor_hit_frac": ratio(top_anchor_hits, total_hits),
            "anchor_head_hit_count_top3_mean": top_mean(hit_counts, 3),
            "anchor_head_hit_count_top5_mean": top_mean(hit_counts, 5),
            "anchor_head_same_fine_frac_mean": mean(fine),
            "anchor_head_same_fine_frac_min": min([v for v in fine if math.isfinite(v)], default=math.nan),
            "anchor_head_same_group_frac_mean": mean(group),
            "anchor_head_low_fine_support_risk_max": max([v for v in fine_risk if math.isfinite(v)], default=math.nan),
            "anchor_head_low_fine_support_risk_top3_mean": top_mean(fine_risk, 3),
            "anchor_head_low_fine_support_risk_top5_mean": top_mean(fine_risk, 5),
            "anchor_head_low_group_support_risk_max": max([v for v in group_risk if math.isfinite(v)], default=math.nan),
            "anchor_head_low_group_support_risk_top3_mean": top_mean(group_risk, 3),
            "anchor_head_high_hit_low_fine_frac": ratio(len(high_hit_low_fine), len(parts)),
            "anchor_head_dominant_source_label_frac": (
                max(label_counts.values()) / len(parts) if parts and label_counts else math.nan
            ),
        }
        for label in ["1", "5", "15", "22", "23", "42"]:
            rows_l = [part for part in parts if str(part.get("dominant_source_fine_label")) == label]
            risks_l = [f(part.get("low_fine_support_risk")) for part in rows_l]
            row[f"anchor_head_label{label}_low_fine_risk_top3_mean"] = top_mean(risks_l, 3)
            row[f"anchor_head_label{label}_edge_frac"] = ratio(len(rows_l), len(parts))
        out.append(row)
    return out


def score(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = []
    for row in case_rows:
        for key in row:
            if key.startswith("anchor_head_") and key not in fields:
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
    edge_rows, errors = collect_anchor_head_rows(args.trace_root)
    case_rows = aggregate_case_rows(edge_rows)
    metrics = score(case_rows)
    gate_like = [row for row in metrics if row.get("gate_like")]
    summary = {
        "trace_root": str(args.trace_root),
        "anchor_head_edge_rows": len(edge_rows),
        "case_count": len(case_rows),
        "read_error_count": len(errors),
        "metrics": len(metrics),
        "gate_like_count": len(gate_like),
        "best": metrics[0] if metrics else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "runtime_action_allowed": False,
        "note": "Anchor/head identity-edge semantic support diagnostic; no true rerun controls or runtime action.",
    }
    out = args.out_dir
    write_rows(out / "semantic_trace_anchor_head_edge_rows.csv", edge_rows)
    write_rows(out / "semantic_trace_anchor_head_case_rows.csv", case_rows)
    write_rows(out / "semantic_trace_anchor_head_metrics.csv", metrics)
    write_rows(out / "semantic_trace_anchor_head_read_errors.csv", errors)
    write_json(out / "semantic_trace_anchor_head_summary.json", summary)
    report = [
        "# Semantic Trace Anchor/Head Identity Edge Audit",
        "",
        f"- trace_root: `{args.trace_root}`",
        f"- anchor_head_edge_rows: `{len(edge_rows)}`",
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
    (out / "semantic_trace_anchor_head_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
