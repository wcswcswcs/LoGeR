#!/usr/bin/env python3
"""Source-target semantic relation audit for v100 trace-native READ signals.

This tests the plan's D3-C fail-forward path: move from source-marginal
per-head carrier signals to source-target relations.  It reads only existing
SWA raw transport trace payloads and is diagnostic-only; no runtime action is
authorized by this script.
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

from loger.pipeline.video_masklet_frontend import (
    SEMANTIC_GROUP_LOW_VALUE_STUFF,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_NAMES,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
    SEMANTIC_GROUP_UNCERTAIN_REGION,
)
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

STABLE_GROUPS = {int(SEMANTIC_GROUP_STRUCTURE_ANCHOR), int(SEMANTIC_GROUP_STATIC_THING)}
RISK_QUERY_GROUPS = {
    int(SEMANTIC_GROUP_MOVABLE_THING),
    int(SEMANTIC_GROUP_LOW_VALUE_STUFF),
    int(SEMANTIC_GROUP_UNCERTAIN_REGION),
}
DYNAMIC_WEAK_SOURCE_GROUPS = {
    int(SEMANTIC_GROUP_MOVABLE_THING),
    int(SEMANTIC_GROUP_LOW_VALUE_STUFF),
    int(SEMANTIC_GROUP_UNCERTAIN_REGION),
}
VALID_GROUPS = sorted(STABLE_GROUPS | RISK_QUERY_GROUPS)


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


def bottom_mean(values: list[float], k: int) -> float:
    vals = sorted([float(v) for v in values if math.isfinite(float(v))])
    return mean(vals[: max(1, int(k))])


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else math.nan


def group_name(group_id: int) -> str:
    return str(SEMANTIC_GROUP_NAMES.get(int(group_id), f"group{int(group_id)}")).lower()


def frac(mask: torch.Tensor, denom: torch.Tensor) -> float:
    den = int(denom.sum().item())
    if den <= 0:
        return math.nan
    return float((mask & denom).sum().item()) / float(den)


def read_head_relation_rows(trace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

        query_group = payload.get("sampled_query_group_ids")
        source_group = payload.get("current_Q_to_cache_K_topk_cache_group_ids")
        query_fine = payload.get("sampled_query_fine_label_ids")
        source_fine = payload.get("current_Q_to_cache_K_topk_cache_fine_label_ids")
        stable_hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        same_group = payload.get("current_Q_to_cache_K_topk_same_group")
        same_fine = payload.get("current_Q_to_cache_K_topk_same_fine_label")
        stable_same_group = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_group")
        stable_same_fine = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_same_query_fine_label")
        topk_frames = payload.get("current_Q_to_cache_K_topk_cache_frames")
        sampled_query_indices = payload.get("sampled_query_indices")
        tensors = [query_group, source_group, query_fine, source_fine, stable_hit, same_group, same_fine]
        if not all(torch.is_tensor(x) for x in tensors):
            errors.append({"trace_payload": str(path), "error": "missing_source_target_tensor"})
            continue

        query_group = query_group.detach().cpu().long()
        source_group = source_group.detach().cpu().long()
        query_fine = query_fine.detach().cpu().long()
        source_fine = source_fine.detach().cpu().long()
        stable_hit = stable_hit.detach().cpu().bool()
        same_group = same_group.detach().cpu().bool()
        same_fine = same_fine.detach().cpu().bool()
        stable_same_group = stable_same_group.detach().cpu().bool() if torch.is_tensor(stable_same_group) else (stable_hit & same_group)
        stable_same_fine = stable_same_fine.detach().cpu().bool() if torch.is_tensor(stable_same_fine) else (stable_hit & same_fine)
        topk_frames = topk_frames.detach().cpu().long() if torch.is_tensor(topk_frames) else None
        sampled_query_indices = sampled_query_indices.detach().cpu().long() if torch.is_tensor(sampled_query_indices) else None

        if source_group.ndim != 4 or query_group.ndim != 2:
            errors.append({"trace_payload": str(path), "error": "bad_source_target_ndim"})
            continue
        if source_group.shape != stable_hit.shape or source_fine.shape != stable_hit.shape:
            errors.append({"trace_payload": str(path), "error": "source_stable_shape_mismatch"})
            continue
        if int(query_group.shape[0]) != int(source_group.shape[0]) or int(query_group.shape[1]) != int(source_group.shape[2]):
            errors.append({"trace_payload": str(path), "error": "query_source_shape_mismatch"})
            continue

        try:
            layer = int(payload.get("swa_layer_idx", -1))
        except Exception:
            layer = -1

        _, head_count, _, _ = source_group.shape
        qg_exp = query_group[:, None, :, None].expand_as(source_group)
        qf_exp = query_fine[:, None, :, None].expand_as(source_fine)
        valid_all = (source_group >= 0) & (qg_exp >= 0)
        valid_stable = valid_all & stable_hit
        same_frame = torch.zeros_like(valid_all)
        near_frame = torch.zeros_like(valid_all)
        far_frame = torch.zeros_like(valid_all)
        tokens_per_frame = int(payload.get("tokens_per_frame", 0) or 0)
        if (
            topk_frames is not None
            and sampled_query_indices is not None
            and topk_frames.shape == source_group.shape
            and tokens_per_frame > 0
            and int(sampled_query_indices.numel()) >= int(source_group.shape[2])
        ):
            q_idx = sampled_query_indices.reshape(-1)[: int(source_group.shape[2])]
            q_frames = torch.div(q_idx, int(tokens_per_frame), rounding_mode="floor").reshape(1, 1, -1, 1)
            q_frames = q_frames.expand_as(topk_frames)
            frame_delta_abs = (topk_frames - q_frames).abs()
            same_frame = frame_delta_abs == 0
            near_frame = frame_delta_abs <= 1
            far_frame = frame_delta_abs >= 4
        q_stable = torch.zeros_like(valid_all)
        q_risk = torch.zeros_like(valid_all)
        src_stable = torch.zeros_like(valid_all)
        src_dynamic_weak = torch.zeros_like(valid_all)
        for gid in STABLE_GROUPS:
            q_stable |= qg_exp == int(gid)
            src_stable |= source_group == int(gid)
        for gid in RISK_QUERY_GROUPS:
            q_risk |= qg_exp == int(gid)
        for gid in DYNAMIC_WEAK_SOURCE_GROUPS:
            src_dynamic_weak |= source_group == int(gid)
        source_group_mismatch = valid_all & (~same_group)
        source_fine_mismatch = valid_all & (~same_fine)

        for head_idx in range(int(head_count)):
            all_h = valid_all[:, head_idx]
            stable_h = valid_stable[:, head_idx]
            q_stable_h = q_stable[:, head_idx]
            q_risk_h = q_risk[:, head_idx]
            src_stable_h = src_stable[:, head_idx]
            src_dynamic_weak_h = src_dynamic_weak[:, head_idx]
            group_mismatch_h = source_group_mismatch[:, head_idx]
            fine_mismatch_h = source_fine_mismatch[:, head_idx]
            same_group_h = same_group[:, head_idx]
            same_fine_h = same_fine[:, head_idx]
            stable_same_group_h = stable_same_group[:, head_idx]
            stable_same_fine_h = stable_same_fine[:, head_idx]
            same_frame_h = same_frame[:, head_idx]
            near_frame_h = near_frame[:, head_idx]
            far_frame_h = far_frame[:, head_idx]
            sg_h = source_group[:, head_idx]
            qg_h = qg_exp[:, head_idx]

            stable_count = int(stable_h.sum().item())
            all_count = int(all_h.sum().item())
            row: dict[str, Any] = {
                **labels[case_id],
                "trace_payload": str(path),
                "chunk_idx": payload.get("chunk_idx", ""),
                "swa_layer_idx": layer,
                "head_idx": int(head_idx),
                "all_topk_count": all_count,
                "stable_hit_count": stable_count,
                "stable_hit_frac": ratio(stable_count, all_count),
                "src_marginal_stablehit_source_stable_frac": frac(src_stable_h, stable_h),
                "src_marginal_stablehit_source_dynamicweak_frac": frac(src_dynamic_weak_h, stable_h),
                "src_marginal_alltopk_source_stable_frac": frac(src_stable_h, all_h),
                "src_marginal_alltopk_source_dynamicweak_frac": frac(src_dynamic_weak_h, all_h),
                "src_marginal_alltopk_sameframe_frac": frac(same_frame_h, all_h),
                "src_marginal_alltopk_nearframe_frac": frac(near_frame_h, all_h),
                "src_marginal_alltopk_farframe_frac": frac(far_frame_h, all_h),
                "src_marginal_stablehit_sameframe_frac": frac(same_frame_h, stable_h),
                "src_marginal_stablehit_nearframe_frac": frac(near_frame_h, stable_h),
                "src_marginal_stablehit_farframe_frac": frac(far_frame_h, stable_h),
                "src_tgt_stablehit_same_group_frac": frac(stable_same_group_h, stable_h),
                "src_tgt_stablehit_same_fine_frac": frac(stable_same_fine_h, stable_h),
                "src_tgt_stablehit_group_mismatch_frac": frac(group_mismatch_h, stable_h),
                "src_tgt_stablehit_fine_mismatch_frac": frac(fine_mismatch_h, stable_h),
                "src_tgt_stablehit_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h, stable_h),
                "src_tgt_stablehit_qrisk_dynamicweak_source_frac": frac(q_risk_h & src_dynamic_weak_h, stable_h),
                "src_tgt_stablehit_qrisk_stable_source_frac": frac(q_risk_h & src_stable_h, stable_h),
                "src_tgt_stablehit_qstable_dynamicweak_source_frac": frac(q_stable_h & src_dynamic_weak_h, stable_h),
                "src_tgt_stablehit_qstable_stable_samegroup_frac": frac(q_stable_h & src_stable_h & same_group_h, stable_h),
                "src_tgt_stablehit_qstable_stable_mismatch_frac": frac(q_stable_h & src_stable_h & group_mismatch_h, stable_h),
                "src_tgt_alltopk_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h, all_h),
                "src_tgt_alltopk_qrisk_dynamicweak_source_frac": frac(q_risk_h & src_dynamic_weak_h, all_h),
                "src_tgt_alltopk_qstable_dynamicweak_source_frac": frac(q_stable_h & src_dynamic_weak_h, all_h),
                "src_tgt_sameframe_alltopk_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & same_frame_h, all_h),
                "src_tgt_nearframe_alltopk_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & near_frame_h, all_h),
                "src_tgt_farframe_alltopk_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & far_frame_h, all_h),
                "src_tgt_sameframe_stablehit_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & same_frame_h, stable_h),
                "src_tgt_nearframe_stablehit_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & near_frame_h, stable_h),
                "src_tgt_farframe_stablehit_qrisk_stale_source_frac": frac(q_risk_h & group_mismatch_h & far_frame_h, stable_h),
                "src_tgt_sameframe_alltopk_lowstuff_uncertain_frac": frac(
                    (qg_h == int(SEMANTIC_GROUP_LOW_VALUE_STUFF))
                    & (sg_h == int(SEMANTIC_GROUP_UNCERTAIN_REGION))
                    & same_frame_h,
                    all_h,
                ),
                "src_tgt_nearframe_alltopk_lowstuff_uncertain_frac": frac(
                    (qg_h == int(SEMANTIC_GROUP_LOW_VALUE_STUFF))
                    & (sg_h == int(SEMANTIC_GROUP_UNCERTAIN_REGION))
                    & near_frame_h,
                    all_h,
                ),
                "src_tgt_farframe_alltopk_lowstuff_uncertain_frac": frac(
                    (qg_h == int(SEMANTIC_GROUP_LOW_VALUE_STUFF))
                    & (sg_h == int(SEMANTIC_GROUP_UNCERTAIN_REGION))
                    & far_frame_h,
                    all_h,
                ),
            }
            for q_gid in VALID_GROUPS:
                for s_gid in VALID_GROUPS:
                    q_name = group_name(q_gid)
                    s_name = group_name(s_gid)
                    pair_mask = (qg_h == int(q_gid)) & (sg_h == int(s_gid))
                    row[f"src_tgt_stablehit_q{q_gid}_{q_name}_s{s_gid}_{s_name}_frac"] = frac(pair_mask, stable_h)
                    row[f"src_tgt_alltopk_q{q_gid}_{q_name}_s{s_gid}_{s_name}_frac"] = frac(pair_mask, all_h)
            rows.append(row)
    return rows, errors


def aggregate_case_rows(head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        grouped[str(row.get("case_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        if not parts:
            continue
        base = parts[0]
        row: dict[str, Any] = {
            "case_id": case_id,
            "seq": base.get("seq", ""),
            "case_label": base.get("case_label", ""),
            "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy", math.nan),
            "source_target_head_layer_count": len(parts),
            "source_target_stable_hit_total": sum(f(part.get("stable_hit_count"), 0.0) for part in parts),
            "source_target_stable_hit_frac_mean": mean([f(part.get("stable_hit_frac")) for part in parts]),
            "source_target_stable_hit_frac_top3_mean": top_mean([f(part.get("stable_hit_frac")) for part in parts], 3),
        }
        field_names: list[str] = []
        for part in parts:
            for key in part:
                if key.startswith(("src_tgt_", "src_marginal_")) and key not in field_names:
                    field_names.append(key)
        for field in field_names:
            values = [f(part.get(field)) for part in parts]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_max"] = max([v for v in values if math.isfinite(v)], default=math.nan)
            row[f"{field}_top3_mean"] = top_mean(values, 3)
            if field.endswith(("same_group_frac", "same_fine_frac", "samegroup_frac")):
                row[f"{field}_bottom3_mean"] = bottom_mean(values, 3)
        for layer in sorted({int(f(part.get("swa_layer_idx"), -1)) for part in parts}):
            layer_parts = [part for part in parts if int(f(part.get("swa_layer_idx"), -2)) == layer]
            risk_values = [f(part.get("src_tgt_stablehit_qrisk_stale_source_frac")) for part in layer_parts]
            row[f"src_tgt_layer{layer}_qrisk_stale_source_frac_top3_mean"] = top_mean(risk_values, 3)
        out.append(row)
    return out


def score(case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_fields: list[str] = []
    pair_fields: list[str] = []
    for row in case_rows:
        for key in row:
            if key.startswith("src_marginal_") and key not in source_fields:
                source_fields.append(key)
            elif key.startswith("src_tgt_") and key not in pair_fields:
                pair_fields.append(key)
    source_metrics: list[dict[str, Any]] = []
    for field in source_fields:
        for direction in ("higher_bad", "lower_bad"):
            metric = evaluate_pattern(case_rows, f"{field}_{direction}", field, direction)
            metric["field"] = field
            metric["direction"] = direction
            metric["is_source_marginal"] = True
            source_metrics.append(metric)
    best_source_ba = max([f(row.get("balanced_accuracy")) for row in source_metrics], default=math.nan)
    best_source = max(
        source_metrics,
        key=lambda row: (
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("abs_corr_L3")),
        ),
        default={},
    )

    pair_metrics: list[dict[str, Any]] = []
    for field in pair_fields:
        for direction in ("higher_bad", "lower_bad"):
            metric = evaluate_pattern(case_rows, f"{field}_{direction}", field, direction)
            metric["field"] = field
            metric["direction"] = direction
            metric["is_source_target_pairwise"] = True
            metric["source_marginal_best_BA"] = best_source_ba
            metric["source_marginal_best_field"] = best_source.get("field", "")
            metric["source_marginal_best_direction"] = best_source.get("direction", "")
            add_ba = f(metric.get("balanced_accuracy")) - best_source_ba if math.isfinite(best_source_ba) else math.nan
            metric["pairwise_add_BA_over_source_marginal"] = add_ba
            metric["d3c_gate_like"] = bool(
                f(metric.get("bad_recall")) >= 0.65
                and f(metric.get("good_FPR"), 1.0) <= 0.25
                and math.isfinite(add_ba)
                and add_ba >= 0.05
            )
            metric["strict_gate_like"] = bool(
                metric["d3c_gate_like"]
                and f(metric.get("abs_corr_L3")) >= 0.50
                and bool(metric.get("corr_direction_correct"))
                and int(f(metric.get("sequence_coverage"), 0)) >= 4
                and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
            )
            metric["gate_like"] = bool(metric["strict_gate_like"])
            pair_metrics.append(metric)

    source_metrics.sort(
        key=lambda row: (
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("abs_corr_L3")),
        ),
        reverse=True,
    )
    pair_metrics.sort(
        key=lambda row: (
            bool(row.get("strict_gate_like")),
            bool(row.get("d3c_gate_like")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("abs_corr_L3")),
            f(row.get("pairwise_add_BA_over_source_marginal")),
        ),
        reverse=True,
    )
    return pair_metrics, source_metrics


def write_report(path: Path, summary: dict[str, Any], pair_metrics: list[dict[str, Any]], source_metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_defs = ", ".join(f"{gid}={group_name(gid)}" for gid in VALID_GROUPS)
    best = pair_metrics[0] if pair_metrics else {}
    best_source = source_metrics[0] if source_metrics else {}
    lines = [
        "# Semantic Trace Source-Target Pairwise Diagnostic",
        "",
        "Diagnostic-only. This does not authorize M3, E4, runtime action, or full validation.",
        "",
        f"- trace_root: `{summary.get('trace_root')}`",
        f"- group ids: {group_defs}",
        f"- stable query/source groups: {sorted(STABLE_GROUPS)}",
        f"- risk query groups: {sorted(RISK_QUERY_GROUPS)}",
        f"- dynamic/weak source groups: {sorted(DYNAMIC_WEAK_SOURCE_GROUPS)}",
        "",
        "## Summary",
        "",
        f"- head_relation_rows: `{summary.get('head_relation_rows')}`",
        f"- case_count: `{summary.get('case_count')}`",
        f"- read_error_count: `{summary.get('read_error_count')}`",
        f"- pair_metric_count: `{summary.get('pair_metric_count')}`",
        f"- source_marginal_metric_count: `{summary.get('source_marginal_metric_count')}`",
        f"- d3c_gate_like_count: `{summary.get('d3c_gate_like_count')}`",
        f"- strict_gate_like_count: `{summary.get('strict_gate_like_count')}`",
        "",
        "## Best Source-Marginal Baseline",
        "",
        "```json",
        json.dumps(clean(best_source), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Best Source-Target Pairwise Cue",
        "",
        "```json",
        json.dumps(clean(best), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    head_rows, errors = read_head_relation_rows(args.trace_root)
    case_rows = aggregate_case_rows(head_rows)
    pair_metrics, source_metrics = score(case_rows)
    d3c_gate_like = [row for row in pair_metrics if row.get("d3c_gate_like")]
    strict_gate_like = [row for row in pair_metrics if row.get("strict_gate_like")]
    summary = {
        "trace_root": str(args.trace_root),
        "head_relation_rows": len(head_rows),
        "case_count": len(case_rows),
        "read_error_count": len(errors),
        "pair_metric_count": len(pair_metrics),
        "source_marginal_metric_count": len(source_metrics),
        "d3c_gate_like_count": len(d3c_gate_like),
        "strict_gate_like_count": len(strict_gate_like),
        "best": pair_metrics[0] if pair_metrics else {},
        "best_d3c_gate_like": d3c_gate_like[0] if d3c_gate_like else {},
        "best_strict_gate_like": strict_gate_like[0] if strict_gate_like else {},
        "best_source_marginal": source_metrics[0] if source_metrics else {},
        "runtime_action_allowed": False,
        "note": "D3-C source-target pairwise diagnostic over trace-native semantic groups; no true rerun controls or runtime action.",
    }

    out = args.out_dir
    write_rows(out / "semantic_trace_source_target_head_rows.csv", head_rows)
    write_rows(out / "semantic_trace_source_target_case_rows.csv", case_rows)
    write_rows(out / "semantic_trace_source_target_pair_metrics.csv", pair_metrics)
    write_rows(out / "semantic_trace_source_target_source_marginal_metrics.csv", source_metrics)
    write_rows(out / "semantic_trace_source_target_read_errors.csv", errors)
    write_json(out / "semantic_trace_source_target_summary.json", clean(summary))
    write_report(out / "semantic_trace_source_target_report.md", summary, pair_metrics, source_metrics)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
