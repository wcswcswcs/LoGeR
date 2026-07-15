#!/usr/bin/env python3
"""Build ACL2 v106R Stage1 memory operation discovery artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
STAGE0 = V106R / "stage0_v105_evidence_freeze"
STAGE1 = V106R / "stage1_memory_operation_map"

FRAME_ROWS = V105 / "stage3_lingbot_oracle/frame_semantic_geometry_rows.csv"
HEAD_ROWS = V105 / "stage4_lingbot_headlocal_trace/headlocal_frame_head_features.csv"
BASELINE_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
LOCAL_WINDOW_ROWS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/local_window_rows.csv"
STAGE0_FACTS = STAGE0 / "v105_known_facts.json"

EPS = 1e-12


FEATURE_DEFS = [
    {
        "column": "scale_reference_context_attention_frac",
        "operation_type": "readout",
        "context_role": "anchor_context",
        "token_type": "image_patch",
        "feature_family": "anchor_context_read_mass",
        "semantic_role": "internal_context_attention",
        "interpretation": "Anchor/scale-reference context read mass.",
    },
    {
        "column": "local_window_context_attention_frac",
        "operation_type": "readout",
        "context_role": "local_pose_reference_window",
        "token_type": "image_patch",
        "feature_family": "local_window_read_mass",
        "semantic_role": "internal_context_attention",
        "interpretation": "Local reference window read mass.",
    },
    {
        "column": "current_or_latest_frame_attention_frac",
        "operation_type": "readout",
        "context_role": "current_frame",
        "token_type": "image_patch",
        "feature_family": "current_frame_read_mass",
        "semantic_role": "internal_context_attention",
        "interpretation": "Current/latest frame read mass.",
    },
    {
        "column": "semantic_scale_reference_attention_frac",
        "operation_type": "readout",
        "context_role": "anchor_context",
        "token_type": "image_patch",
        "feature_family": "semantic_scale_reference_read_mass",
        "semantic_role": "scale_reference_evidence",
        "interpretation": "Semantic mass assigned to scale-reference evidence in anchor context.",
    },
    {
        "column": "semantic_local_registration_attention_frac",
        "operation_type": "readout",
        "context_role": "local_pose_reference_window",
        "token_type": "image_patch",
        "feature_family": "semantic_local_registration_read_mass",
        "semantic_role": "local_registration_evidence",
        "interpretation": "Semantic mass assigned to local-registration evidence.",
    },
    {
        "column": "semantic_context_only_attention_frac",
        "operation_type": "readout",
        "context_role": "unknown",
        "token_type": "image_patch",
        "feature_family": "semantic_context_only_read_mass",
        "semantic_role": "context_only_candidate",
        "interpretation": "Semantic mass assigned to context-only evidence.",
    },
    {
        "column": "semantic_reject_unreliable_attention_frac",
        "operation_type": "readout",
        "context_role": "unknown",
        "token_type": "image_patch",
        "feature_family": "semantic_reject_unreliable_read_mass",
        "semantic_role": "reject_unreliable_candidate",
        "interpretation": "Semantic mass assigned to unreliable/reject evidence.",
    },
    {
        "column": "scale_context_reject_attention_frac",
        "operation_type": "readout",
        "context_role": "anchor_context",
        "token_type": "image_patch",
        "feature_family": "scale_context_reject_read_mass",
        "semantic_role": "reject_unreliable_in_anchor_context",
        "interpretation": "Rejected/unreliable evidence still read through scale context.",
    },
    {
        "column": "scale_context_structure_attention_frac",
        "operation_type": "readout",
        "context_role": "anchor_context",
        "token_type": "image_patch",
        "feature_family": "scale_context_structure_read_mass",
        "semantic_role": "stable_structure_in_anchor_context",
        "interpretation": "Stable-structure evidence read through scale context.",
    },
    {
        "column": "local_context_reject_attention_frac",
        "operation_type": "readout",
        "context_role": "local_pose_reference_window",
        "token_type": "image_patch",
        "feature_family": "local_context_reject_read_mass",
        "semantic_role": "reject_unreliable_in_local_context",
        "interpretation": "Rejected/unreliable evidence still read through local context.",
    },
    {
        "column": "head_trace_topk_attention_sum",
        "operation_type": "readout",
        "context_role": "unknown",
        "token_type": "special_unknown",
        "feature_family": "head_total_topk_attention_mass",
        "semantic_role": "internal_attention_volume",
        "interpretation": "Head total traced top-k attention mass.",
    },
]

ROTATED_COLUMN = {
    "scale_reference_context_attention_frac": "local_window_context_attention_frac",
    "local_window_context_attention_frac": "current_or_latest_frame_attention_frac",
    "current_or_latest_frame_attention_frac": "scale_reference_context_attention_frac",
    "semantic_scale_reference_attention_frac": "semantic_local_registration_attention_frac",
    "semantic_local_registration_attention_frac": "semantic_context_only_attention_frac",
    "semantic_context_only_attention_frac": "semantic_reject_unreliable_attention_frac",
    "semantic_reject_unreliable_attention_frac": "semantic_scale_reference_attention_frac",
    "scale_context_reject_attention_frac": "local_context_reject_attention_frac",
    "scale_context_structure_attention_frac": "scale_context_reject_attention_frac",
    "local_context_reject_attention_frac": "scale_context_reject_attention_frac",
    "head_trace_topk_attention_sum": "head_trace_topk_attention_sum",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= EPS or vy <= EPS:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    thresholds: list[float] = []
    for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        thresholds.append(ordered[idx])
    return sorted(set(thresholds))


def topk_mask(values: list[float], direction: str, count: int) -> list[bool]:
    if count <= 0:
        return [False] * len(values)
    reverse = direction == "ge"
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    chosen = set(order[: min(count, len(values))])
    return [idx in chosen for idx in range(len(values))]


def selected_metrics(mask: list[bool], bad: list[bool], good: list[bool]) -> dict[str, float]:
    selected = sum(mask)
    bad_total = sum(bad)
    good_total = sum(good)
    selected_bad = sum(1 for keep, flag in zip(mask, bad) if keep and flag)
    selected_good = sum(1 for keep, flag in zip(mask, good) if keep and flag)
    bad_recall = selected_bad / bad_total if bad_total else 0.0
    good_fpr = selected_good / good_total if good_total else 0.0
    balanced = (bad_recall + (1.0 - good_fpr)) / 2.0
    return {
        "selected_rows": selected,
        "selected_bad_rows": selected_bad,
        "selected_good_rows": selected_good,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": balanced,
    }


def best_threshold(values: list[float], bad: list[bool], good: list[bool]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for direction in ("ge", "le"):
        for threshold in quantile_thresholds(values):
            mask = [(value >= threshold) if direction == "ge" else (value <= threshold) for value in values]
            metrics = selected_metrics(mask, bad, good)
            if metrics["selected_rows"] == 0:
                continue
            random_bad_recall = metrics["selected_rows"] / len(values) if values else 0.0
            metrics["same_count_random_margin"] = metrics["bad_recall"] - random_bad_recall
            metrics["threshold_direction"] = direction
            metrics["threshold_value"] = threshold
            score = (
                metrics["balanced_accuracy"],
                metrics["same_count_random_margin"],
                metrics["bad_recall"],
                -metrics["good_FPR"],
            )
            if best is None or score > best["_score"]:
                best = {**metrics, "_score": score}
    if best is None:
        return {
            "selected_rows": 0,
            "selected_bad_rows": 0,
            "selected_good_rows": 0,
            "bad_recall": 0.0,
            "good_FPR": 0.0,
            "balanced_accuracy": 0.0,
            "same_count_random_margin": 0.0,
            "threshold_direction": "ge",
            "threshold_value": 0.0,
        }
    best.pop("_score", None)
    return best


def label_name(row: dict[str, str]) -> str:
    if parse_bool(row.get("bad_label")):
        return "bad"
    if parse_bool(row.get("good_label")):
        return "good"
    return "neutral"


def dominant_label_and_purity(row: dict[str, str]) -> tuple[str, float]:
    patch_count = fnum(row, "patch_count", 0.0)
    labels = row.get("top_labels", "")
    if patch_count <= 0 or not labels:
        return "", 0.0
    best_label = ""
    best_count = 0.0
    for item in labels.split(";"):
        if ":" not in item:
            continue
        name, count = item.rsplit(":", 1)
        try:
            count_value = float(count)
        except ValueError:
            continue
        if count_value > best_count:
            best_label = name
            best_count = count_value
    return best_label, best_count / patch_count if patch_count else 0.0


def build_frame_index() -> dict[tuple[str, str], dict[str, Any]]:
    frames = {}
    for row in read_csv(FRAME_ROWS):
        seq = row["seq"]
        sample_position = row["sample_position"]
        dominant_label, purity = dominant_label_and_purity(row)
        semantic_confidence = fnum(row, "semantic_confidence_mean", 0.0)
        semantic_trust = semantic_confidence * purity * purity
        residual = fnum(row, "sim3_residual_m", 0.0)
        seq_error_p75 = fnum(row, "seq_error_p75", 0.0)
        metric_consistency = 1.0 / (1.0 + residual / (seq_error_p75 + EPS)) if seq_error_p75 > 0 else 0.0
        scale_obs = min(
            1.0,
            0.5 * fnum(row, "scale_reference_patch_frac", 0.0)
            + 0.5 * fnum(row, "semantic_scale_reference_attention_frac", 0.0),
        )
        geometry_support = min(
            1.0,
            0.5 * fnum(row, "scale_reference_patch_frac", 0.0)
            + 0.5 * fnum(row, "local_registration_patch_frac", 0.0)
            + 0.5 * (1.0 - fnum(row, "reject_unreliable_patch_frac", 1.0)),
        )
        frames[(seq, sample_position)] = {
            **row,
            "dominant_semantic_label": dominant_label,
            "patch_purity": purity,
            "semantic_trust": semantic_trust,
            "metric_consistency_score": metric_consistency,
            "scale_observability_score": scale_obs,
            "geometry_support_score": geometry_support,
        }
    return frames


def window_metric_index() -> dict[str, list[dict[str, Any]]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not LOCAL_WINDOW_ROWS.exists():
        return by_seq
    for row in read_csv(LOCAL_WINDOW_ROWS):
        seq = row["seq"]
        by_seq[seq].append(
            {
                **row,
                "frame_start_i": int(float(row["frame_start"])),
                "frame_end_i": int(float(row["frame_end"])),
                "window_index_i": int(float(row["window_index"])),
                "handoff_transfer_penalty_f": fnum(row, "handoff_transfer_penalty", 0.0),
                "handoff_available": row.get("handoff_transfer_penalty", "") not in {"", None},
                "adjacent_log_scale_jump_f": fnum(row, "adjacent_log_scale_jump", 0.0),
                "local_sim3_ate_rmse_m_f": fnum(row, "local_sim3_ate_rmse_m", 0.0),
            }
        )
    for rows in by_seq.values():
        rows.sort(key=lambda item: item["window_index_i"])
    return by_seq


def nearby_window_metrics(index: dict[str, list[dict[str, Any]]], seq: str, frame_id: str) -> dict[str, Any]:
    try:
        frame = int(float(frame_id))
    except ValueError:
        return {
            "local_window_index": "",
            "L3_handoff_metric_nearby": "",
            "rolling_drift_metric_nearby": "",
            "adjacent_log_scale_jump_nearby": "",
            "local_window_ATE_nearby": "",
        }
    rows = index.get(seq, [])
    matched: dict[str, Any] | None = None
    for row in rows:
        if int(row["frame_start_i"]) <= frame <= int(row["frame_end_i"]):
            matched = row
            break
    if matched is None:
        return {
            "local_window_index": "",
            "L3_handoff_metric_nearby": "",
            "rolling_drift_metric_nearby": "",
            "adjacent_log_scale_jump_nearby": "",
            "local_window_ATE_nearby": "",
        }
    idx = int(matched["window_index_i"])
    recent = [
        row["handoff_transfer_penalty_f"]
        for row in rows
        if row["handoff_available"] and idx - 4 <= int(row["window_index_i"]) <= idx
    ]
    rolling = sum(recent) / len(recent) if recent else 0.0
    return {
        "local_window_index": idx,
        "L3_handoff_metric_nearby": matched["handoff_transfer_penalty_f"] if matched["handoff_available"] else "",
        "rolling_drift_metric_nearby": rolling,
        "adjacent_log_scale_jump_nearby": matched["adjacent_log_scale_jump_f"] if matched.get("adjacent_log_scale_jump", "") not in {"", None} else "",
        "local_window_ATE_nearby": matched["local_sim3_ate_rmse_m_f"],
    }


def lever_id(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["operation_type"]),
            str(row["context_role"]),
            str(row["token_type"]),
            str(row["feature_family"]),
        ]
    )


def operation_rows() -> list[dict[str, Any]]:
    frames = build_frame_index()
    windows = window_metric_index()
    rows: list[dict[str, Any]] = []
    for head in read_csv(HEAD_ROWS):
        key = (head["seq"], head["sample_position"])
        frame = frames.get(key)
        if frame is None:
            continue
        window_metrics = nearby_window_metrics(windows, head["seq"], frame.get("original_frame", ""))
        for feature in FEATURE_DEFS:
            value = fnum(head, feature["column"], 0.0)
            rows.append(
                {
                    "schema": "acl2_v106r_stage1_memory_operation_row_v1",
                    "seq_id": head["seq"],
                    "frame_id": frame.get("original_frame", ""),
                    "window_id": frame.get("semantic_chunk", ""),
                    "boundary_id": f"{head['seq']}:{head['sample_position']}",
                    "sample_position": head["sample_position"],
                    "operation_type": feature["operation_type"],
                    "context_role": feature["context_role"],
                    "token_type": feature["token_type"],
                    "head_id": head["head_idx"],
                    "source_frame_age": "",
                    "source_frame_age_available": False,
                    "attention_mass": value,
                    "feature_family": feature["feature_family"],
                    "feature_column": feature["column"],
                    "semantic_role": feature["semantic_role"],
                    "dominant_semantic_label": frame["dominant_semantic_label"],
                    "semantic_confidence": frame.get("semantic_confidence_mean", ""),
                    "patch_purity": frame["patch_purity"],
                    "semantic_trust": frame["semantic_trust"],
                    "geometry_support_score": frame["geometry_support_score"],
                    "metric_consistency_score": frame["metric_consistency_score"],
                    "scale_observability_score": frame["scale_observability_score"],
                    "L3_handoff_metric_nearby": window_metrics["L3_handoff_metric_nearby"],
                    "L3_metric_source": "stage1_full_local_window_rows_handoff_transfer_penalty",
                    "rolling_drift_metric_nearby": window_metrics["rolling_drift_metric_nearby"],
                    "rolling_metric_source": "stage1_full_local_window_rows_recent5_handoff_transfer_penalty_mean",
                    "adjacent_log_scale_jump_nearby": window_metrics["adjacent_log_scale_jump_nearby"],
                    "local_window_ATE_nearby": window_metrics["local_window_ATE_nearby"],
                    "local_window_index": window_metrics["local_window_index"],
                    "good_or_bad_label": label_name(frame),
                    "bad_label": frame.get("bad_label", ""),
                    "good_label": frame.get("good_label", ""),
                    "trace_scope": "v105_seq00_seq02_trace32_head_resolved",
                    "source_artifact": rel(HEAD_ROWS),
                    "frame_source_artifact": rel(FRAME_ROWS),
                }
            )
    return rows


def deterministic_rotated_values(group: list[dict[str, Any]], column: str) -> list[float]:
    rows_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for head in read_csv(HEAD_ROWS):
        rows_by_identity[(head["seq"], head["sample_position"], head["head_idx"])] = head
    values: list[float] = []
    for row in group:
        identity = (str(row["seq_id"]), str(row["sample_position"]), str(row["head_id"]))
        values.append(fnum(rows_by_identity.get(identity, {}), column, 0.0))
    return values


def semantic_shuffled_values(values: list[float], rows: list[dict[str, Any]]) -> list[float]:
    by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_seq[str(row["seq_id"])].append(idx)
    shuffled = values[:]
    for seq, indices in by_seq.items():
        if not indices:
            continue
        shift = 7 % len(indices)
        seq_values = [values[idx] for idx in indices]
        for out_idx, source_value in zip(indices, seq_values[shift:] + seq_values[:shift]):
            shuffled[out_idx] = source_value
    return shuffled


def rank_levers(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_lever: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lever[lever_id(row)].append(row)

    rank_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    feature_by_lever = {f"{d['operation_type']}|{d['context_role']}|{d['token_type']}|{d['feature_family']}": d for d in FEATURE_DEFS}
    for lever, group in sorted(by_lever.items()):
        values = [float(row["attention_mass"]) for row in group]
        l3 = [fnum(row, "L3_handoff_metric_nearby", 0.0) for row in group]
        rolling = [fnum(row, "rolling_drift_metric_nearby", 0.0) for row in group]
        bad = [parse_bool(row.get("bad_label")) for row in group]
        good = [parse_bool(row.get("good_label")) for row in group]
        best = best_threshold(values, bad, good)
        mask = [(v >= best["threshold_value"]) if best["threshold_direction"] == "ge" else (v <= best["threshold_value"]) for v in values]
        feature_def = feature_by_lever[lever]

        rotated_column = ROTATED_COLUMN.get(str(feature_def["column"]), str(feature_def["column"]))
        rotated_values = deterministic_rotated_values(group, rotated_column)
        rotated_mask = topk_mask(rotated_values, str(best["threshold_direction"]), int(best["selected_rows"]))
        rotated_metrics = selected_metrics(rotated_mask, bad, good)

        shuffled_values = semantic_shuffled_values(values, group)
        shuffled_mask = topk_mask(shuffled_values, str(best["threshold_direction"]), int(best["selected_rows"]))
        shuffled_metrics = selected_metrics(shuffled_mask, bad, good)

        selected_bad_by_seq: dict[str, int] = defaultdict(int)
        selected_by_seq: dict[str, int] = defaultdict(int)
        for keep, row, is_bad in zip(mask, group, bad):
            if not keep:
                continue
            selected_by_seq[str(row["seq_id"])] += 1
            if is_bad:
                selected_bad_by_seq[str(row["seq_id"])] += 1
        total_selected_bad = sum(selected_bad_by_seq.values())
        positive_sequence_max_frac = (
            max(selected_bad_by_seq.values()) / total_selected_bad if total_selected_bad else 0.0
        )
        selected_sequence_coverage = sum(1 for value in selected_by_seq.values() if value > 0)

        rank_row = {
            "schema": "acl2_v106r_stage1_memory_lever_rank_row_v1",
            "lever_id": lever,
            "operation_type": feature_def["operation_type"],
            "context_role": feature_def["context_role"],
            "token_type": feature_def["token_type"],
            "feature_family": feature_def["feature_family"],
            "feature_column": feature_def["column"],
            "case_count": len(group),
            "sequence_coverage": len({row["seq_id"] for row in group}),
            "selected_sequence_coverage": selected_sequence_coverage,
            "bad_recall": best["bad_recall"],
            "good_FPR": best["good_FPR"],
            "balanced_accuracy": best["balanced_accuracy"],
            "selected_rows": best["selected_rows"],
            "selected_bad_rows": best["selected_bad_rows"],
            "selected_good_rows": best["selected_good_rows"],
            "threshold_direction": best["threshold_direction"],
            "threshold_value": best["threshold_value"],
            "abs_corr_L3": abs(pearson(values, l3)),
            "signed_corr_L3": pearson(values, l3),
            "abs_corr_rolling": abs(pearson(values, rolling)),
            "signed_corr_rolling": pearson(values, rolling),
            "same_count_random_margin": best["same_count_random_margin"],
            "context_role_rotation_margin": best["bad_recall"] - rotated_metrics["bad_recall"],
            "semantic_shuffle_margin": best["bad_recall"] - shuffled_metrics["bad_recall"],
            "positive_sequence_max_frac": positive_sequence_max_frac,
            "operation_interpretation": feature_def["interpretation"],
            "trace_scope": "v105_seq00_seq02_trace32_head_resolved",
        }
        rank_rows.append(rank_row)

        for seq in sorted({row["seq_id"] for row in group}):
            idxs = [idx for idx, row in enumerate(group) if row["seq_id"] == seq]
            seq_mask = [mask[idx] for idx in idxs]
            seq_bad = [bad[idx] for idx in idxs]
            seq_good = [good[idx] for idx in idxs]
            seq_values = [values[idx] for idx in idxs]
            seq_l3 = [l3[idx] for idx in idxs]
            seq_metrics = selected_metrics(seq_mask, seq_bad, seq_good)
            split_rows.append(
                {
                    "schema": "acl2_v106r_stage1_memory_lever_sequence_split_row_v1",
                    "lever_id": lever,
                    "seq_id": seq,
                    "case_count": len(idxs),
                    "selected_rows": seq_metrics["selected_rows"],
                    "selected_bad_rows": seq_metrics["selected_bad_rows"],
                    "selected_good_rows": seq_metrics["selected_good_rows"],
                    "bad_recall": seq_metrics["bad_recall"],
                    "good_FPR": seq_metrics["good_FPR"],
                    "balanced_accuracy": seq_metrics["balanced_accuracy"],
                    "abs_corr_L3": abs(pearson(seq_values, seq_l3)),
                    "signed_corr_L3": pearson(seq_values, seq_l3),
                }
            )

    rank_rows.sort(
        key=lambda row: (
            max(float(row["abs_corr_L3"]), float(row["abs_corr_rolling"])),
            float(row["same_count_random_margin"]),
            float(row["balanced_accuracy"]),
        ),
        reverse=True,
    )
    return rank_rows, split_rows


def stage0_facts() -> dict[str, Any]:
    if STAGE0_FACTS.exists():
        return json.loads(STAGE0_FACTS.read_text(encoding="utf-8"))
    return {}


def write_report(path: Path, summary: dict[str, Any], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage1 Memory Operation Discovery Report",
        "",
        f"- stage1_discovery_pass: `{summary['stage1_discovery_pass']}`",
        f"- stage1_promotion_ready: `{summary['stage1_promotion_ready']}`",
        f"- targeted_trace_required_before_action: `{summary['targeted_trace_required_before_action']}`",
        f"- memory_operation_rows: `{summary['memory_operation_rows']}`",
        f"- lever_count: `{summary['lever_count']}`",
        f"- levers_sequence_coverage_ge2: `{summary['levers_sequence_coverage_ge2']}`",
        f"- max_abs_corr_L3: `{summary['max_abs_corr_L3']}`",
        f"- max_abs_corr_rolling: `{summary['max_abs_corr_rolling']}`",
        f"- max_same_count_random_margin: `{summary['max_same_count_random_margin']}`",
        "",
        "Trace coverage boundary:",
        "",
        "- Current Stage1 rows are derived from v105 seq00/seq02 trace32 head-resolved artifacts.",
        "- This is valid for discovery ranking but is not sufficient for action promotion or full KITTI ATE claims.",
        "- Full KITTI 00/01/02/05 targeted trace remains required before Stage4/Stage5 runtime promotion.",
        "",
        "Top levers:",
        "",
    ]
    for row in top_rows[:8]:
        lines.append(
            "- "
            f"{row['lever_id']}: abs_corr_L3={row['abs_corr_L3']}, "
            f"bad_recall={row['bad_recall']}, good_FPR={row['good_FPR']}, "
            f"same_count_random_margin={row['same_count_random_margin']}, "
            f"selected_sequence_coverage={row['selected_sequence_coverage']}"
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            "Stage1 can only be treated as a no-action discovery map at this point. "
            "The next plan-consistent step is semantic increment mapping on the same fixed universe, "
            "plus targeted trace supplementation before any action surface promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_targeted_trace_requirement(path: Path) -> None:
    lines = [
        "# Targeted Trace Requirement Before Action Promotion",
        "",
        "Reason:",
        "",
        "- v105 trace-derived Stage1 evidence covers only seq00/seq02 trace32.",
        "- v106R full validation scope is KITTI 00/01/02/05.",
        "- The current rows are discovery-only and cannot justify runtime action or full ATE claims.",
        "",
        "Required supplementation before Stage4/Stage5:",
        "",
        "- selected high-L3 windows from KITTI 00/01/02/05",
        "- selected safe-good low-drift windows",
        "- head-resolved GCA trace for top suspect windows",
        "- no-action parity for any new trace mode",
        "",
        "Required gates:",
        "",
        "- pose/depth/intrinsics/confidence parity pass",
        "- trace_error_rows=0",
        "- trace row count > 0",
        "- context_role resolved ratio >= 0.90",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_no_lever(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage1 No Memory Lever Found",
        "",
        f"- memory_operation_rows: `{summary['memory_operation_rows']}`",
        f"- lever_count: `{summary['lever_count']}`",
        f"- levers_sequence_coverage_ge2: `{summary['levers_sequence_coverage_ge2']}`",
        f"- max_abs_corr_L3: `{summary['max_abs_corr_L3']}`",
        f"- max_abs_corr_rolling: `{summary['max_abs_corr_rolling']}`",
        f"- max_same_count_random_margin: `{summary['max_same_count_random_margin']}`",
        "",
        "No Stage1 discovery lever passed the required association gates. Action stages must stop.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    STAGE1.mkdir(parents=True, exist_ok=True)
    rows = operation_rows()
    rank_rows, split_rows = rank_levers(rows)
    facts = stage0_facts()

    trace_sequences = set(facts.get("stage2_trace_sequences", []))
    full_sequences = set(facts.get("lingbot_full_kitti_sequences", []))
    trace32_only = trace_sequences != full_sequences
    trace_parity_blocker = not bool(facts.get("stage2_trace_parity_pass", False))

    max_abs_corr_l3 = max((float(row["abs_corr_L3"]) for row in rank_rows), default=0.0)
    max_abs_corr_rolling = max((float(row["abs_corr_rolling"]) for row in rank_rows), default=0.0)
    max_random_margin = max((float(row["same_count_random_margin"]) for row in rank_rows), default=0.0)
    levers_seq_ge2 = sum(1 for row in rank_rows if int(row["sequence_coverage"]) >= 2)
    discovery_pass = (
        levers_seq_ge2 >= 3
        and max(max_abs_corr_l3, max_abs_corr_rolling) >= 0.45
        and max_random_margin >= 0.05
        and not trace_parity_blocker
    )
    promotion_ready = discovery_pass and not trace32_only

    summary = {
        "schema": "acl2_v106r_stage1_summary_v1",
        "stage1_discovery_pass": discovery_pass,
        "stage1_promotion_ready": promotion_ready,
        "targeted_trace_required_before_action": bool(trace32_only),
        "trace_parity_blocker": trace_parity_blocker,
        "trace_sequences": sorted(trace_sequences),
        "full_kitti_sequences": sorted(full_sequences),
        "memory_operation_rows": len(rows),
        "lever_count": len(rank_rows),
        "levers_sequence_coverage_ge2": levers_seq_ge2,
        "max_abs_corr_L3": max_abs_corr_l3,
        "max_abs_corr_rolling": max_abs_corr_rolling,
        "max_same_count_random_margin": max_random_margin,
        "operation_types_present": sorted({row["operation_type"] for row in rows}),
        "missing_operation_types": ["update", "retention", "initialization", "budget_eviction"],
        "outputs": {
            "memory_operation_rows": rel(STAGE1 / "memory_operation_rows.csv"),
            "memory_lever_rank": rel(STAGE1 / "memory_lever_rank.csv"),
            "memory_lever_sequence_split": rel(STAGE1 / "memory_lever_sequence_split.csv"),
            "memory_lever_report": rel(STAGE1 / "memory_lever_report.md"),
            "targeted_trace_requirement": rel(STAGE1 / "targeted_trace_requirement.md"),
        },
    }

    write_csv(STAGE1 / "memory_operation_rows.csv", rows)
    write_csv(STAGE1 / "memory_lever_rank.csv", rank_rows)
    write_csv(STAGE1 / "memory_lever_sequence_split.csv", split_rows)
    write_report(STAGE1 / "memory_lever_report.md", summary, rank_rows)
    write_targeted_trace_requirement(STAGE1 / "targeted_trace_requirement.md")
    if not discovery_pass:
        write_no_lever(STAGE1 / "stage1_no_memory_lever_found.md", summary)
        summary["outputs"]["stage1_no_memory_lever_found"] = rel(STAGE1 / "stage1_no_memory_lever_found.md")

    (STAGE1 / "stage1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
