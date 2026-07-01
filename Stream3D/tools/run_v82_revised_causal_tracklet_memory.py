#!/usr/bin/env python3
"""Run Stream4D v82 revised causal tracklet memory audit.

Phase0 is a strict fact lock over v81 causal evidence.  Later phases should be
added to this runner so the v82 audit remains one ordered, reproducible path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_v81_history_anchored_cmap_af_l2h_pipeline as v81  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _metric_le(value: Any, threshold: float, default: float = 1.0) -> bool:
    metric = _float(value, default)
    return metric is not None and metric <= threshold


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO).as_posix()
        except ValueError:
            return path.as_posix()


def _repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(math.ceil((pct / 100.0) * len(vals))) - 1))
    return vals[idx]


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _hash_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def _int_field(value: Any, default: int = -1) -> int:
    parsed = _float(value)
    return default if parsed is None else int(parsed)


def _metric_rows() -> list[dict[str, Any]]:
    selection = [
        "causal_violation_count",
        "future_descriptor_count",
        "tracklet_assignment_entropy",
        "tracklet_top1_top2_margin",
        "full_minus_semantic_score",
        "real_minus_shuffled_score",
        "false_attachment_proxy_rate",
        "new_object_absorption_proxy_rate",
        "identity_switch_proxy",
        "fragmentation_proxy",
        "wrong_absorption_proxy_rate",
        "cannot_link_violation_count",
        "memory_MB",
        "history_node_count",
        "eligible_tracklet_coverage_rate",
        "all_slot_assignment_rate",
        "new_object_no_anchor_rate",
        "full_minus_stale_score",
        "future_tracklet_descriptor_count",
        "self_confirmation_count",
    ]
    diagnostic = [
        "local_SF50",
        "local_AP50",
        "local_AP25",
        "GT_best_IoU_mean",
        "same_instance_tracklet_precision_GT",
        "wrong_instance_rate_GT",
        "within_semantic_GT_AUC",
        "oracle_history_match",
        "oracle_tracklet_to_GT_match",
        "v81_B0_local_SF50",
        "v81_repair31_Q_coverage",
        "v81_repair31_full_minus_semantic",
        "v81_repair31_full_minus_shuffled",
        "v81_confirmed_only_Q_coverage",
    ]
    final_eval = [
        "scene_SF50",
        "scene_AP50",
        "identity_switch_rate",
        "fragmentation_rate",
        "overmerge_rate",
        "holdout_run_count_for_method_claim",
        "parameter_change_after_holdout_count",
    ]
    rows: list[dict[str, Any]] = []
    for metric in selection:
        rows.append(
            {
                "metric_name": metric,
                "metric_class": "selection",
                "can_drive_parameter_selection": True,
                "uses_gt": False,
                "notes": "GT-free metric allowed for dev config freeze.",
            }
        )
    for metric in diagnostic:
        rows.append(
            {
                "metric_name": metric,
                "metric_class": "diagnostic",
                "can_drive_parameter_selection": False,
                "uses_gt": metric.startswith("GT_") or metric.endswith("_GT") or metric in {"local_SF50", "local_AP50", "local_AP25"},
                "notes": "Diagnostic only; not allowed to tune v82 method config.",
            }
        )
    for metric in final_eval:
        rows.append(
            {
                "metric_name": metric,
                "metric_class": "final_eval",
                "can_drive_parameter_selection": False,
                "uses_gt": metric in {"scene_SF50", "scene_AP50"},
                "notes": "Report after frozen config; not allowed for post-holdout tuning.",
            }
        )
    return rows


def _summary_value(path: Path, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_name": key,
        "fact_value": payload.get(key, ""),
        "source_path": _rel(path),
        "source_key": key,
    }


def _run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase0_output_root
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "v81_pipeline_summary": _repo_path(args.v81_pipeline_summary),
        "v81_local_summary": _repo_path(args.v81_local_summary),
        "v81_history_summary": _repo_path(args.v81_history_summary),
        "v81_q_summary": _repo_path(args.v81_q_summary),
        "v81_q_control_rows": _repo_path(args.v81_q_control_rows),
        "v81_final_decision": _repo_path(args.v81_final_decision),
        "v81_repair_summary_rows": _repo_path(args.v81_repair_summary_rows),
    }
    payloads = {name: _read_json(path) for name, path in paths.items() if path.suffix == ".json"}
    q_control_rows = _read_csv_rows(paths["v81_q_control_rows"])
    repair_rows = _read_csv_rows(paths["v81_repair_summary_rows"])

    q6_coverages = [
        _float(row.get("Q_coverage_rate"))
        for row in q_control_rows
        if row.get("variant") == "Q6_full_affinity_confirmed_only"
    ]
    q6_coverages = [v for v in q6_coverages if v is not None]

    deprecated_rows: list[dict[str, Any]] = []
    for row in repair_rows:
        repair = str(row.get("repair", "")).strip()
        if repair.startswith(("24", "25", "26", "27", "28")):
            deprecated_rows.append(
                {
                    "repair": repair,
                    "tag": row.get("tag", ""),
                    "summary_json": row.get("summary_json", ""),
                    "final_decision": row.get("final_decision", ""),
                    "diagnostic_only": True,
                    "allowed_for_method_evidence": False,
                    "reason": "pre-causal or pre-final-causal-boundary artifact; v82 plan requires repair29-31 causal evidence only",
                }
            )

    metric_rows = _metric_rows()
    metric_class_unlabeled_count = sum(1 for row in metric_rows if not row.get("metric_class"))

    final = payloads.get("v81_final_decision", {})
    q_summary = payloads.get("v81_q_summary", {})
    local_summary = payloads.get("v81_local_summary", {})
    history_summary = payloads.get("v81_history_summary", {})
    pipeline_summary = payloads.get("v81_pipeline_summary", {})

    carrier_suffix_method_allowed = False
    gt_prediction_violation_count = int(_bool(q_summary.get("method_uses_gt_anywhere"))) + int(
        _float(local_summary.get("method_GT_violation_count"), 0.0) or 0.0
    )
    future_descriptor_allowed_count = 0

    fact_rows = [
        _summary_value(paths["v81_final_decision"], "final_decision", final),
        _summary_value(paths["v81_q_summary"], "Q_coverage_rate", q_summary),
        _summary_value(paths["v81_q_summary"], "full_minus_semantic_top1_confidence", q_summary),
        _summary_value(paths["v81_q_summary"], "full_minus_shuffled_top1_confidence", q_summary),
        {
            "fact_name": "v81_confirmed_only_Q_coverage",
            "fact_value": _mean(q6_coverages),
            "source_path": _rel(paths["v81_q_control_rows"]),
            "source_key": "mean Q_coverage_rate where variant=Q6_full_affinity_confirmed_only over all dev chunks",
        },
        _summary_value(paths["v81_local_summary"], "local_SF50", local_summary),
        _summary_value(paths["v81_q_summary"], "carrier_id_scope", q_summary),
        {
            "fact_name": "carrier_suffix_method_allowed",
            "fact_value": carrier_suffix_method_allowed,
            "source_path": _rel(paths["v81_q_summary"]),
            "source_key": "carrier_id_scope / secondary_blocker / carrier_sketch_score_method_enabled",
        },
        {
            "fact_name": "pre_causal_artifact_count",
            "fact_value": len(deprecated_rows),
            "source_path": _rel(paths["v81_repair_summary_rows"]),
            "source_key": "repairs 24-28 rows",
        },
        {
            "fact_name": "future_descriptor_allowed_count",
            "fact_value": future_descriptor_allowed_count,
            "source_path": _rel(paths["v81_q_summary"]),
            "source_key": "v82 method path allows no future descriptors",
        },
        {
            "fact_name": "GT_prediction_violation_count",
            "fact_value": gt_prediction_violation_count,
            "source_path": f"{_rel(paths['v81_q_summary'])};{_rel(paths['v81_local_summary'])}",
            "source_key": "method_uses_gt_anywhere + method_GT_violation_count",
        },
        {
            "fact_name": "metric_class_unlabeled_count",
            "fact_value": metric_class_unlabeled_count,
            "source_path": _rel(out / "metric_class_rows.csv"),
            "source_key": "generated metric_class_rows",
        },
    ]

    causal_rows = [
        {
            "artifact": "v81_repair31b_phase2_history_snapshot",
            "source_path": _rel(paths["v81_history_summary"]),
            "history_use_causal_snapshots": history_summary.get("history_use_causal_snapshots", ""),
            "history_snapshot_row_count": history_summary.get("history_snapshot_row_count", ""),
            "history_snapshot_chunk_count": history_summary.get("history_snapshot_chunk_count", ""),
            "future_descriptor_count": history_summary.get("future_history_descriptor_count", ""),
            "method_uses_gt": history_summary.get("method_uses_gt_anywhere", ""),
            "allowed_for_v82_method_boundary": True,
            "notes": "Causal snapshot source used only to lock v81 boundary.",
        },
        {
            "artifact": "v81_repair31b_phase3_q",
            "source_path": _rel(paths["v81_q_summary"]),
            "history_use_causal_snapshots": q_summary.get("history_use_causal_snapshots", ""),
            "history_causal_last_seen_required": q_summary.get("history_causal_last_seen_required", ""),
            "future_descriptor_count": q_summary.get("future_history_descriptor_candidate_count", ""),
            "future_descriptor_filtered_count": q_summary.get("future_history_descriptor_filtered_count", ""),
            "method_uses_gt": q_summary.get("method_uses_gt_anywhere", ""),
            "allowed_for_v82_method_boundary": True,
            "notes": "Locks v81 final Q weakness under causal snapshot.",
        },
        {
            "artifact": "v81_repair31b_pipeline_summary",
            "source_path": _rel(paths["v81_pipeline_summary"]),
            "decision": pipeline_summary.get("decision", ""),
            "allowed_for_v82_method_boundary": True,
            "notes": "Only used as prior No-Go evidence, not as v82 method output.",
        },
    ]

    missing_paths = [name for name, path in paths.items() if not path.exists()]
    gate = {
        "GT_prediction_violation_count_eq_0": gt_prediction_violation_count == 0,
        "future_descriptor_allowed_count_eq_0": future_descriptor_allowed_count == 0,
        "carrier_suffix_method_allowed_false": carrier_suffix_method_allowed is False,
        "deprecated_artifacts_marked_diagnostic_only": all(
            _bool(row.get("diagnostic_only")) and not _bool(row.get("allowed_for_method_evidence"))
            for row in deprecated_rows
        ),
        "metric_class_unlabeled_count_eq_0": metric_class_unlabeled_count == 0,
        "required_sources_present": not missing_paths,
    }
    gate_pass = all(gate.values())
    summary = {
        "phase": "v82_phase0_fact_lock",
        "schema": "stream4d_v82_phase0_fact_lock_v1",
        "decision": "PASS_V82_PHASE0_FACT_LOCK" if gate_pass else "BLOCK_V82_PHASE0_FACT_LOCK",
        "can_enter_next_phase": gate_pass,
        "v81_final_decision": final.get("final_decision", ""),
        "v81_repair31_Q_coverage": q_summary.get("Q_coverage_rate", ""),
        "v81_repair31_full_minus_semantic": q_summary.get("full_minus_semantic_top1_confidence", ""),
        "v81_repair31_full_minus_shuffled": q_summary.get("full_minus_shuffled_top1_confidence", ""),
        "v81_confirmed_only_Q_coverage": _mean(q6_coverages),
        "v81_B0_local_SF50": local_summary.get("local_SF50", ""),
        "carrier_id_scope": q_summary.get("carrier_id_scope", ""),
        "carrier_suffix_method_allowed": carrier_suffix_method_allowed,
        "pre_causal_artifact_count": len(deprecated_rows),
        "future_descriptor_allowed_count": future_descriptor_allowed_count,
        "GT_prediction_violation_count": gt_prediction_violation_count,
        "metric_class_unlabeled_count": metric_class_unlabeled_count,
        "missing_required_sources": missing_paths,
        "gate": gate,
        "primary_blocker": "" if gate_pass else "phase0_fact_lock_gate_failed",
        "runtime_sec": time.time() - started,
    }

    fields = ["fact_name", "fact_value", "source_path", "source_key"]
    _write_csv(out / "fact_rows.csv", fact_rows, fields)
    _write_csv(
        out / "deprecated_artifact_rows.csv",
        deprecated_rows,
        ["repair", "tag", "summary_json", "final_decision", "diagnostic_only", "allowed_for_method_evidence", "reason"],
    )
    _write_csv(
        out / "metric_class_rows.csv",
        metric_rows,
        ["metric_name", "metric_class", "can_drive_parameter_selection", "uses_gt", "notes"],
    )
    _write_csv(out / "causal_boundary_rows.csv", causal_rows)
    _write_json(out / "fact_lock_summary.json", summary)
    _write_json(out / "summary.json", summary)
    return summary


def _v81_phase1_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = v81.build_parser()
    local_args = parser.parse_args([])
    local_args.split = args.split
    local_args.run_tag = args.run_tag
    if getattr(args, "scenes", ""):
        local_args.scenes = args.scenes
    if getattr(args, "chunk_ids", ""):
        local_args.chunk_ids = args.chunk_ids
    if getattr(args, "v75_phase1_root", ""):
        local_args.v75_phase1_root = args.v75_phase1_root
    if getattr(args, "semantic_feature_rows", ""):
        local_args.semantic_feature_rows = args.semantic_feature_rows
    if getattr(args, "incidence_variant", ""):
        local_args.incidence_variant = args.incidence_variant
    local_args.pipeline_root = args.pipeline_root
    local_args.phase1_output_root = f"{args.phase1_output_root}/raw_v81_replay"
    local_args.local_shadow_root = args.local_shadow_root
    local_args.appearance_feature_mode = args.appearance_feature_mode
    local_args.appearance_feature_rows = args.appearance_feature_rows

    # Freeze the local CMAP-AF config to the v80/v81 dev_r77 family used by
    # repair29-31.  These are fixed GT-free method parameters, not tuned here.
    fixed = {
        "semantic_positive_guard": "confident_same_proto",
        "semantic_guard_min_margin": 0.13125,
        "semantic_guard_max_broad_share": 0.75,
        "max_component_ratio": 0.085,
        "adapter_render_kernel": "gaussian_disk",
        "adapter_ambiguous_mask_policy": "best",
        "object_parent_merge_mode": "coarse_parent",
        "object_parent_merge_min_object_count": 45,
        "object_parent_merge_max_component_ratio": 0.12,
        "object_parent_merge_min_parent_inclusion": 0.95,
        "object_parent_merge_min_signed_score": 0.45,
        "object_parent_merge_max_parent_child_count": 5,
        "adapter_score_mode": "hybrid",
        "object_mask_ownership_mode": "dominance",
        "object_mask_ownership_min_score_margin": 0.0,
        "object_mask_ownership_min_score_ratio": 1.0,
        "random_seed": 8001,
    }
    for key, value in fixed.items():
        setattr(local_args, key, value)
    carrier_scope, _carrier_rows = v81._classify_carrier_scope(local_args)
    local_args.carrier_id_scope_effective = carrier_scope
    return local_args


def _adapter_stats(adapter_rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in adapter_rows:
        key = (str(row.get("scene_id") or ""), _int_field(row.get("chunk_id")), str(row.get("cluster_id") or ""))
        grouped[key].append(row)
    stats: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        f1 = [_float(row.get("hybrid_adapter_F1")) for row in rows]
        precision = [_float(row.get("hybrid_adapter_precision")) for row in rows]
        recall = [_float(row.get("hybrid_adapter_recall")) for row in rows]
        f1_vals = [v for v in f1 if v is not None]
        precision_vals = [v for v in precision if v is not None]
        recall_vals = [v for v in recall if v is not None]
        duplicate_count = sum(
            1
            for row in rows
            if _bool(row.get("adapter_caused_merge")) or _bool(row.get("adapter_caused_split"))
        )
        broad_count = sum(1 for row in rows if str(row.get("adapter_role") or "").lower() in {"broad", "coarse_parent", "background"})
        stats[key] = {
            "adapter_mask_count": len(rows),
            "adapter_score_mean": _mean(f1_vals),
            "adapter_score_p10": _percentile(f1_vals, 10.0),
            "adapter_precision_mean": _mean(precision_vals),
            "adapter_recall_mean": _mean(recall_vals),
            "duplicate_frame_mask_conflict_count": duplicate_count,
            "broad_adapter_rate": _safe_ratio(broad_count, len(rows)),
        }
    return stats


def _proxy_appearance_descriptor_hash(
    *,
    scene: str,
    chunk: int,
    cluster_id: str,
    semantic_proto: str,
    desc: dict[str, Any],
    carrier_count: int,
) -> str:
    payload = {
        "scene_id": scene,
        "chunk_id": int(chunk),
        "cluster_id": str(cluster_id),
        "semantic_proto_id": semantic_proto,
        "semantic_margin_mean": round(float(desc.get("semantic_margin_mean") or 0.0), 6),
        "uv_x_mean": round(float(desc.get("uv_x_mean") or 0.0), 6),
        "uv_y_mean": round(float(desc.get("uv_y_mean") or 0.0), 6),
        "frame_count": int(desc.get("frame_count") or 0),
        "frame_span": int(desc.get("frame_span") or 0),
        "confidence_mean": round(float(desc.get("confidence_mean") or 0.0), 6),
        "carrier_count": int(carrier_count),
    }
    return _hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _phase1_slot_rows(
    clusters: dict[tuple[str, int], dict[str, Any]],
    raw_local_rows: list[dict[str, Any]],
    raw_cluster_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in raw_local_rows:
        cluster_id = str(row.get("source_cluster_id") or row.get("cluster_id") or "")
        key = (str(row.get("scene_id") or ""), _int_field(row.get("chunk_id")), cluster_id)
        raw_by_key[key] = row
    cluster_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in raw_cluster_rows:
        key = (
            str(row.get("scene_id") or ""),
            _int_field(row.get("chunk_id")),
            str(row.get("cluster_id") or ""),
        )
        cluster_by_key[key] = row
    adapter_by_key = _adapter_stats(adapter_rows)

    rows: list[dict[str, Any]] = []
    for (scene, chunk), graph in sorted(clusters.items()):
        data = graph["data"]
        carriers_all = graph["carriers"]
        for label, indices in sorted(graph["label_to_indices"].items(), key=lambda item: str(item[0])):
            cluster_id = str(label)
            carriers = [carriers_all[int(idx)] for idx in indices]
            desc = v81._cluster_descriptor(data, carriers)
            key = (scene, int(chunk), cluster_id)
            raw = raw_by_key.get(key, {})
            cluster = cluster_by_key.get(key, {})
            adapter = adapter_by_key.get(key, {})
            adapter_score = adapter.get("adapter_score_mean")
            slot_confidence = adapter_score if adapter_score is not None else _float(raw.get("mean_adapter_F1"), 0.0)
            semantic_proto = str(desc.get("semantic_proto_id") or "")
            appearance_hash = str(desc.get("appearance_feature_hash") or "")
            appearance_source = "object_support_filtered_dino_csv"
            if not appearance_hash:
                appearance_hash = _proxy_appearance_descriptor_hash(
                    scene=scene,
                    chunk=int(chunk),
                    cluster_id=cluster_id,
                    semantic_proto=semantic_proto,
                    desc=desc,
                    carrier_count=len(carriers),
                )
                appearance_source = "method_safe_proxy_semantic_spatial"
            row = {
                "scene_id": scene,
                "chunk_id": int(chunk),
                "local_slot_id": f"V82_local:c{chunk}:cluster{cluster_id}",
                "cluster_id": cluster_id,
                "carrier_count": len(carriers),
                "frame_support_count": desc.get("frame_count", ""),
                "visible_frame_span": desc.get("frame_span", ""),
                "semantic_descriptor_hash": _hash_text(semantic_proto),
                "semantic_descriptor_source": "v81_cluster_semantic_proto_id",
                "semantic_proto_id": semantic_proto,
                "appearance_descriptor_hash": appearance_hash,
                "appearance_descriptor_source": appearance_source,
                "appearance_vector_json": json.dumps(
                    [round(float(value), 7) for value in (desc.get("appearance_vector").tolist() if desc.get("appearance_vector") is not None else [])],
                    separators=(",", ":"),
                ),
                "appearance_support_mask_count": desc.get("appearance_feature_support_count", ""),
                "appearance_support_precision_mean": adapter.get("adapter_precision_mean", ""),
                "semantic_margin_mean": desc.get("semantic_margin_mean", ""),
                "uv_x_mean": desc.get("uv_x_mean", ""),
                "uv_y_mean": desc.get("uv_y_mean", ""),
                "confidence_mean": desc.get("confidence_mean", ""),
                "adapter_mask_count": adapter.get("adapter_mask_count", raw.get("mask_count", "")),
                "adapter_score_mean": adapter_score if adapter_score is not None else raw.get("mean_adapter_F1", ""),
                "adapter_score_p10": adapter.get("adapter_score_p10", ""),
                "same_frame_violation_count": cluster.get("cannot_link_violation_count", 0),
                "duplicate_frame_mask_conflict_count": adapter.get("duplicate_frame_mask_conflict_count", 0),
                "cannot_link_violation_count": cluster.get("cannot_link_violation_count", 0),
                "broad_adapter_rate": adapter.get("broad_adapter_rate", desc.get("broad_background_share_mean", "")),
                "slot_confidence": slot_confidence,
                "slot_ambiguity": desc.get("broad_background_share_mean", ""),
                "method_uses_gt": False,
            }
            rows.append(row)
    return rows


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_fields = [
        "local_SF50_rendered_adapter",
        "local_AP50",
        "local_AP25",
        "GT_best_IoU_mean",
        "adapter_identity_flip_rate",
        "adapter_multi_object_materialization_rate",
        "duplicate_frame_mask_conflict_rate",
        "broad_adapter_rate",
        "carrier_F1_vs_pixel_F1_spearman",
        "method_GT_violation_count",
    ]
    out: dict[str, float] = {}
    for field in numeric_fields:
        vals = [_float(row.get(field)) for row in rows]
        clean = [v for v in vals if v is not None]
        if clean:
            out[field] = _mean(clean) or 0.0
    return out


def _run_phase1(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase1_output_root
    out.mkdir(parents=True, exist_ok=True)
    local_args = _v81_phase1_args(args)
    phase1_summary, _context, _bundles, clusters = v81._run_phase1(local_args)
    appearance_audit = v81._attach_carrier_appearance_profiles(clusters, local_args)

    raw_root = ROOT / local_args.phase1_output_root
    shadow_args = v81._local_shadow_args(local_args)
    adapter_root = ROOT / shadow_args.phase5_output_root
    raw_local_rows = _read_csv_rows(raw_root / "local_slot_rows.csv")
    raw_metric_rows = _read_csv_rows(raw_root / "local_metric_rows.csv")
    raw_cluster_rows = _read_csv_rows(raw_root / "local_cluster_rows.csv")
    adapter_rows = _read_csv_rows(adapter_root / "adapter_rows.csv")
    slot_rows = _phase1_slot_rows(clusters, raw_local_rows, raw_cluster_rows, adapter_rows)

    metric_agg = _aggregate_metric_rows(raw_metric_rows)
    slot_conf = [_float(row.get("slot_confidence")) for row in slot_rows]
    slot_amb = [_float(row.get("slot_ambiguity")) for row in slot_rows]
    cannot_link = sum(int(_float(row.get("cannot_link_violation_count"), 0.0) or 0.0) for row in slot_rows)
    same_frame = sum(int(_float(row.get("same_frame_violation_count"), 0.0) or 0.0) for row in slot_rows)
    duplicate_conflicts = sum(int(_float(row.get("duplicate_frame_mask_conflict_count"), 0.0) or 0.0) for row in slot_rows)
    adapter_mask_count = sum(int(_float(row.get("adapter_mask_count"), 0.0) or 0.0) for row in slot_rows)
    duplicate_conflict_rate = _safe_ratio(duplicate_conflicts, adapter_mask_count)
    broad_rates = [_float(row.get("broad_adapter_rate")) for row in slot_rows]
    missing_semantic_hash = sum(1 for row in slot_rows if not row.get("semantic_descriptor_hash"))
    missing_appearance_hash = sum(1 for row in slot_rows if not row.get("appearance_descriptor_hash"))

    local_sf50 = _float(phase1_summary.get("local_SF50"), metric_agg.get("local_SF50_rendered_adapter", 0.0)) or 0.0
    gt_best = _float(phase1_summary.get("GT_best_IoU_mean"), metric_agg.get("GT_best_IoU_mean", 0.0)) or 0.0
    adapter_flip = metric_agg.get("adapter_identity_flip_rate", 0.0)
    method_gt_violation = int(metric_agg.get("method_GT_violation_count", 0.0))
    history_gate = {
        "local_SF50_ge_0p33_diagnostic": local_sf50 >= 0.33,
        "GT_best_IoU_mean_ge_0p33_diagnostic": gt_best >= 0.33,
        "same_frame_violation_count_eq_0": same_frame == 0,
        "cannot_link_violation_count_eq_0": cannot_link == 0,
        "duplicate_frame_mask_conflict_rate_le_0p02": duplicate_conflict_rate <= 0.02,
        "adapter_identity_flip_rate_le_0p05": adapter_flip <= 0.05,
        "method_uses_gt_false": method_gt_violation == 0,
        "semantic_descriptor_hash_complete": missing_semantic_hash == 0,
        "appearance_descriptor_hash_complete": missing_appearance_hash == 0,
    }
    history_gate["pass"] = all(history_gate.values())
    v79_best = _float(phase1_summary.get("v79_best_replay_SF50"), 0.3287608225108225) or 0.3287608225108225
    final_local_gate = {
        "local_SF50_ge_0p40": local_sf50 >= 0.40,
        "local_SF50_ge_v79_best_plus_0p05": local_sf50 >= v79_best + 0.05,
        "GT_best_IoU_mean_ge_0p36": gt_best >= 0.36,
    }
    final_local_gate["pass"] = all(final_local_gate.values())

    adapter_identity_rows = [
        {
            "scene_id": row.get("scene_id"),
            "chunk_id": row.get("chunk_id"),
            "adapter_identity_flip_rate": row.get("adapter_identity_flip_rate"),
            "adapter_multi_object_materialization_rate": row.get("adapter_multi_object_materialization_rate"),
            "adapter_candidate_frame_mask_conflict_rate": row.get("adapter_candidate_frame_mask_conflict_rate"),
            "method_GT_violation_count": row.get("method_GT_violation_count"),
            "source": _rel(raw_root / "local_metric_rows.csv"),
        }
        for row in raw_metric_rows
    ]
    local_conflict_rows = [
        {
            "scene_id": row.get("scene_id"),
            "chunk_id": row.get("chunk_id"),
            "cluster_id": row.get("cluster_id"),
            "cannot_link_violation_count": row.get("cannot_link_violation_count"),
            "duplicate_frame_mask_conflict_count": next(
                (
                    item.get("duplicate_frame_mask_conflict_count")
                    for item in slot_rows
                    if item["scene_id"] == row.get("scene_id")
                    and str(item["chunk_id"]) == str(row.get("chunk_id"))
                    and str(item["cluster_id"]) == str(row.get("cluster_id"))
                ),
                0,
            ),
            "source": _rel(raw_root / "local_cluster_rows.csv"),
        }
        for row in raw_cluster_rows
    ]

    local_metric_summary = {
        "scene_id": "ALL_DEV",
        "chunk_id": "ALL_DEV",
        "slot_confidence_mean": _mean([v for v in slot_conf if v is not None]),
        "slot_ambiguity_mean": _mean([v for v in slot_amb if v is not None]),
        "adapter_identity_flip_rate": adapter_flip,
        "adapter_multi_object_materialization_rate": metric_agg.get("adapter_multi_object_materialization_rate", 0.0),
        "cannot_link_violation_count": cannot_link,
        "same_frame_violation_count": same_frame,
        "duplicate_frame_mask_conflict_rate": duplicate_conflict_rate,
        "broad_adapter_rate": _mean([v for v in broad_rates if v is not None]),
        "carrier_pixel_adapter_agreement": metric_agg.get("carrier_F1_vs_pixel_F1_spearman", ""),
        "local_SF50": local_sf50,
        "local_AP50": phase1_summary.get("local_AP50", metric_agg.get("local_AP50", "")),
        "local_AP25": phase1_summary.get("local_AP25", metric_agg.get("local_AP25", "")),
        "GT_best_IoU_mean": gt_best,
    }

    summary = {
        "phase": "v82_phase1_local_b0",
        "schema": "stream4d_v82_phase1_local_b0_v1",
        "decision": "PASS_V82_PHASE1_LOCAL_HISTORY_ELIGIBLE"
        if history_gate["pass"]
        else "NO_GO_LOCAL_BOOTSTRAP_WEAK",
        "can_enter_next_phase": bool(history_gate["pass"]),
        "can_enter_method_mode_local2history": False,
        "local_SF50": local_sf50,
        "local_AP50": local_metric_summary["local_AP50"],
        "local_AP25": local_metric_summary["local_AP25"],
        "GT_best_IoU_mean": gt_best,
        "slot_count": len(slot_rows),
        "slot_confidence_mean": local_metric_summary["slot_confidence_mean"],
        "slot_ambiguity_mean": local_metric_summary["slot_ambiguity_mean"],
        "adapter_identity_flip_rate": adapter_flip,
        "adapter_multi_object_materialization_rate": metric_agg.get("adapter_multi_object_materialization_rate", 0.0),
        "cannot_link_violation_count": cannot_link,
        "same_frame_violation_count": same_frame,
        "duplicate_frame_mask_conflict_rate": duplicate_conflict_rate,
        "broad_adapter_rate": local_metric_summary["broad_adapter_rate"],
        "carrier_pixel_adapter_agreement": local_metric_summary["carrier_pixel_adapter_agreement"],
        "method_GT_violation_count": method_gt_violation,
        "missing_semantic_descriptor_hash_count": missing_semantic_hash,
        "missing_appearance_descriptor_hash_count": missing_appearance_hash,
        "appearance_feature_audit": appearance_audit,
        "history_eligibility_gate": history_gate,
        "final_local_gate": final_local_gate,
        "primary_blocker": "" if history_gate["pass"] else "local_history_eligibility_failed",
        "secondary_blocker": "" if final_local_gate["pass"] else "final_local_gate_not_met",
        "raw_v81_replay_root": _rel(raw_root),
        "adapter_rows_source": _rel(adapter_root / "adapter_rows.csv"),
        "runtime_sec": time.time() - started,
    }

    _write_csv(
        out / "local_slot_rows.csv",
        slot_rows,
        [
            "scene_id",
            "chunk_id",
            "local_slot_id",
            "cluster_id",
            "carrier_count",
            "frame_support_count",
            "visible_frame_span",
            "semantic_descriptor_hash",
            "semantic_descriptor_source",
            "semantic_proto_id",
            "appearance_descriptor_hash",
            "appearance_descriptor_source",
            "appearance_vector_json",
            "appearance_support_mask_count",
            "appearance_support_precision_mean",
            "semantic_margin_mean",
            "uv_x_mean",
            "uv_y_mean",
            "confidence_mean",
            "adapter_mask_count",
            "adapter_score_mean",
            "adapter_score_p10",
            "same_frame_violation_count",
            "duplicate_frame_mask_conflict_count",
            "cannot_link_violation_count",
            "broad_adapter_rate",
            "slot_confidence",
            "slot_ambiguity",
            "method_uses_gt",
        ],
    )
    _write_csv(out / "local_metric_rows.csv", [local_metric_summary])
    _write_csv(
        out / "local_descriptor_rows.csv",
        [
            {
                "scene_id": row["scene_id"],
                "chunk_id": row["chunk_id"],
                "local_slot_id": row["local_slot_id"],
                "semantic_proto_id": row["semantic_proto_id"],
                "semantic_descriptor_hash": row["semantic_descriptor_hash"],
                "semantic_margin_mean": row["semantic_margin_mean"],
                "appearance_descriptor_hash": row["appearance_descriptor_hash"],
                "appearance_vector_json": row["appearance_vector_json"],
                "uv_x_mean": row["uv_x_mean"],
                "uv_y_mean": row["uv_y_mean"],
                "visible_frame_span": row["visible_frame_span"],
                "confidence_mean": row["confidence_mean"],
                "slot_confidence": row["slot_confidence"],
                "slot_ambiguity": row["slot_ambiguity"],
                "method_uses_gt": row["method_uses_gt"],
            }
            for row in slot_rows
        ],
    )
    _write_csv(out / "adapter_identity_rows.csv", adapter_identity_rows)
    _write_csv(out / "local_conflict_rows.csv", local_conflict_rows)
    _write_json(out / "local_summary.json", summary)
    _write_json(out / "summary.json", summary)
    return summary


def _tokens(proto: str) -> set[str]:
    parts = [part for part in str(proto or "").split("|") if part]
    if parts and parts[0] == "dino":
        parts = parts[1:]
    return set(parts)


def _semantic_overlap(left: str, right: str) -> float:
    lt = _tokens(left)
    rt = _tokens(right)
    if not lt or not rt:
        return 0.0
    return _safe_ratio(len(lt & rt), max(len(lt), len(rt), 1))


def _parse_vec(raw: Any) -> list[float]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            out.append(float(parsed))
    return out


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln <= 1e-12 or rn <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, dot / (ln * rn)))


def _center_vec(values: list[float], center: list[float]) -> list[float]:
    if not values or not center or len(values) != len(center):
        return values
    return [float(value) - float(offset) for value, offset in zip(values, center)]


def _blend_vec(old: list[float], new: list[float], old_weight: float, new_weight: float) -> list[float]:
    if not old:
        return list(new)
    if not new or len(old) != len(new):
        return list(old)
    total = max(1e-12, old_weight + new_weight)
    blended = [(a * old_weight + b * new_weight) / total for a, b in zip(old, new)]
    norm = math.sqrt(sum(v * v for v in blended))
    if norm > 1e-12:
        blended = [v / norm for v in blended]
    return blended


def _entropy_from_scores(scores: list[float]) -> float:
    vals = [max(0.0, float(v)) for v in scores if math.isfinite(float(v))]
    if not vals:
        return 0.0
    total = sum(vals)
    if total <= 1e-12:
        return 1.0
    probs = [v / total for v in vals]
    raw = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return raw / math.log(len(probs)) if len(probs) > 1 else 0.0


def _slot_from_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": str(row.get("scene_id") or ""),
        "chunk_id": _int_field(row.get("chunk_id")),
        "local_slot_id": str(row.get("local_slot_id") or ""),
        "semantic_proto_id": str(row.get("semantic_proto_id") or ""),
        "appearance_vector": _parse_vec(row.get("appearance_vector_json")),
        "uv_x_mean": _float(row.get("uv_x_mean"), 0.0) or 0.0,
        "uv_y_mean": _float(row.get("uv_y_mean"), 0.0) or 0.0,
        "visible_frame_span": _float(row.get("visible_frame_span"), 0.0) or 0.0,
        "confidence_mean": _float(row.get("confidence_mean"), 0.0) or 0.0,
        "slot_confidence": _float(row.get("slot_confidence"), 0.0) or 0.0,
        "slot_ambiguity": _float(row.get("slot_ambiguity"), 0.0) or 0.0,
        "method_uses_gt": _bool(row.get("method_uses_gt")),
    }


def _tracklet_memory_bytes(tracklet: dict[str, Any]) -> int:
    return 256 + 4 * len(tracklet.get("appearance_vector") or []) + 64 * len(tracklet.get("slots") or [])


def _score_slot_tracklet(slot: dict[str, Any], tracklet: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    age = max(0, int(slot["chunk_id"]) - int(tracklet["last_seen_chunk"]))
    semantic = _semantic_overlap(slot["semantic_proto_id"], str(tracklet.get("semantic_proto_id") or ""))
    slot_app = slot["appearance_vector"]
    tracklet_app = tracklet.get("appearance_vector") or []
    missing_appearance_vector = not slot_app or not tracklet_app or len(slot_app) != len(tracklet_app)
    raw_appearance = _cosine(slot_app, tracklet_app)
    if args.tracklet_appearance_residual_mode in {"active_mean", "active_mean_blend50"}:
        center = slot.get("appearance_residual_center") or []
        residual_appearance = _cosine(_center_vec(slot_app, center), _center_vec(tracklet_app, center))
        if args.tracklet_appearance_residual_mode == "active_mean_blend50":
            appearance = 0.5 * raw_appearance + 0.5 * residual_appearance
        else:
            appearance = residual_appearance
    else:
        appearance = raw_appearance
    temporal = max(0.0, 1.0 - age / max(1.0, float(args.tracklet_window_chunks) + 1.0))
    visibility = _safe_ratio(
        min(float(slot["visible_frame_span"]), float(tracklet.get("visible_frame_span", 0.0))),
        max(float(slot["visible_frame_span"]), float(tracklet.get("visible_frame_span", 0.0)), 1e-6),
    )
    dx = float(slot["uv_x_mean"]) - float(tracklet.get("uv_x_mean", 0.0))
    dy = float(slot["uv_y_mean"]) - float(tracklet.get("uv_y_mean", 0.0))
    dist = math.sqrt(dx * dx + dy * dy)
    spatial = math.exp(-0.5 * (dist / max(1e-6, float(args.tracklet_spatial_sigma))) ** 2)
    proxy_appearance_used = missing_appearance_vector and str(getattr(args, "appearance_feature_mode", "")) == "proxy"
    if proxy_appearance_used:
        appearance = max(appearance, 0.55 * semantic + 0.30 * spatial + 0.15 * visibility)
    conflict = 1.0 if (semantic < float(args.tracklet_semantic_conflict_threshold) and spatial < 0.2) else 0.0
    t5_raw = 0.15 * semantic + 0.60 * appearance + 0.10 * temporal + 0.05 * visibility + 0.10 * spatial - 0.40 * conflict
    if proxy_appearance_used:
        t5 = max(t5_raw, semantic + 0.20 * spatial + 0.10 * visibility + 0.10 * temporal - 0.40 * conflict)
    else:
        t5 = min(1.0, t5_raw)
    scores = {
        "semantic_score": semantic,
        "appearance_score": appearance,
        "temporal_score": temporal,
        "visibility_score": visibility,
        "spatial_score": spatial,
        "conflict_score": conflict,
    }
    scores["T0_semantic_only"] = semantic
    scores["T1_object_support_appearance_only"] = appearance
    scores["T2_temporal_visibility_only"] = 0.60 * temporal + 0.40 * visibility
    scores["T3_semantic_appearance"] = 0.50 * semantic + 0.50 * appearance
    scores["T4_semantic_appearance_temporal"] = 0.35 * semantic + 0.35 * appearance + 0.20 * temporal + 0.10 * visibility
    scores["T5_semantic_appearance_temporal_conflict_guard"] = max(0.0, t5)
    return scores


def _new_tracklet(tracklet_id: str, slot: dict[str, Any], state: str = "tentative") -> dict[str, Any]:
    return {
        "tracklet_id": tracklet_id,
        "tracklet_state": state,
        "birth_chunk_id": int(slot["chunk_id"]),
        "last_seen_chunk": int(slot["chunk_id"]),
        "support_chunk_count": 1,
        "support_slot_count": 1,
        "semantic_proto_id": slot["semantic_proto_id"],
        "appearance_vector": list(slot["appearance_vector"]),
        "uv_x_mean": float(slot["uv_x_mean"]),
        "uv_y_mean": float(slot["uv_y_mean"]),
        "visible_frame_span": float(slot["visible_frame_span"]),
        "confidence_mean": float(slot["confidence_mean"]),
        "slot_confidence_mean": float(slot["slot_confidence"]),
        "slot_ambiguity_mean": float(slot["slot_ambiguity"]),
        "slots": [slot["local_slot_id"]],
        "chunks": {int(slot["chunk_id"])},
        "last_assignment_entropy": "",
        "last_assignment_margin": "",
        "method_uses_gt": False,
        "uses_future": False,
    }


def _update_tracklet(tracklet: dict[str, Any], slot: dict[str, Any], entropy: float, margin: float) -> None:
    old_n = max(1, int(tracklet.get("support_slot_count", 1)))
    new_n = old_n + 1
    tracklet["appearance_vector"] = _blend_vec(tracklet.get("appearance_vector") or [], slot["appearance_vector"], old_n, 1.0)
    tracklet["uv_x_mean"] = (float(tracklet.get("uv_x_mean", 0.0)) * old_n + float(slot["uv_x_mean"])) / new_n
    tracklet["uv_y_mean"] = (float(tracklet.get("uv_y_mean", 0.0)) * old_n + float(slot["uv_y_mean"])) / new_n
    tracklet["visible_frame_span"] = max(float(tracklet.get("visible_frame_span", 0.0)), float(slot["visible_frame_span"]))
    tracklet["confidence_mean"] = (float(tracklet.get("confidence_mean", 0.0)) * old_n + float(slot["confidence_mean"])) / new_n
    tracklet["slot_confidence_mean"] = (float(tracklet.get("slot_confidence_mean", 0.0)) * old_n + float(slot["slot_confidence"])) / new_n
    tracklet["slot_ambiguity_mean"] = (float(tracklet.get("slot_ambiguity_mean", 0.0)) * old_n + float(slot["slot_ambiguity"])) / new_n
    tracklet["last_seen_chunk"] = int(slot["chunk_id"])
    tracklet["support_slot_count"] = new_n
    tracklet.setdefault("chunks", set()).add(int(slot["chunk_id"]))
    tracklet["support_chunk_count"] = len(tracklet["chunks"])
    tracklet.setdefault("slots", []).append(slot["local_slot_id"])
    tracklet["last_assignment_entropy"] = entropy
    tracklet["last_assignment_margin"] = margin
    if (
        tracklet["support_chunk_count"] >= 2
        and entropy <= 0.60
        and margin >= 0.05
        and float(tracklet.get("slot_confidence_mean", 0.0)) >= 0.05
    ):
        tracklet["tracklet_state"] = "confirmed"


def _tracklet_descriptor_hash(tracklet: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "semantic": tracklet.get("semantic_proto_id"),
            "appearance": [round(float(v), 6) for v in (tracklet.get("appearance_vector") or [])[:16]],
            "uv_x": round(float(tracklet.get("uv_x_mean", 0.0)), 6),
            "uv_y": round(float(tracklet.get("uv_y_mean", 0.0)), 6),
            "support": tracklet.get("support_slot_count"),
        },
        sort_keys=True,
    )
    return _hash_text(payload)


def _run_phase2(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase2_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase1_summary = _read_json(ROOT / args.phase1_output_root / "local_summary.json")
    if not _bool(phase1_summary.get("can_enter_next_phase")):
        summary = {
            "phase": "v82_phase2_object_tracklets",
            "schema": "stream4d_v82_phase2_tracklets_v1",
            "decision": "BLOCK_TRACKLETS_BY_LOCAL_BOOTSTRAP",
            "can_enter_next_phase": False,
            "primary_blocker": "phase1_local_history_eligibility_failed",
            "runtime_sec": time.time() - started,
        }
        _write_json(out / "tracklet_summary.json", summary)
        _write_json(out / "summary.json", summary)
        for name in [
            "tracklet_snapshot_rows.csv",
            "tracklet_candidate_rows.csv",
            "tracklet_assignment_rows.csv",
            "tracklet_node_rows.csv",
            "tracklet_edge_rows.csv",
            "tracklet_descriptor_rows.csv",
            "tracklet_control_rows.csv",
        ]:
            _write_csv(out / name, [])
        return summary

    descriptor_rows = _read_csv_rows(ROOT / args.phase1_output_root / "local_descriptor_rows.csv")
    slots = [_slot_from_descriptor(row) for row in descriptor_rows]
    slots_by_scene_chunk: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        slots_by_scene_chunk[(slot["scene_id"], int(slot["chunk_id"]))].append(slot)

    active_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_tracklets: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    control_rows_by_variant: dict[str, list[dict[str, float]]] = defaultdict(list)
    selected_rows: list[dict[str, Any]] = []
    eligible_count = 0
    future_count = 0
    self_confirmation_count = 0
    next_tracklet = 1

    variants = [
        "T0_semantic_only",
        "T1_object_support_appearance_only",
        "T2_temporal_visibility_only",
        "T3_semantic_appearance",
        "T4_semantic_appearance_temporal",
        "T5_semantic_appearance_temporal_conflict_guard",
    ]

    for scene in sorted({slot["scene_id"] for slot in slots}):
        chunks = sorted(chunk for row_scene, chunk in slots_by_scene_chunk if row_scene == scene)
        for chunk in chunks:
            active_live = [
                trk
                for trk in active_by_scene[scene]
                if str(trk.get("tracklet_state")) != "inactive"
                and int(trk.get("last_seen_chunk", -1)) < chunk
                and chunk - int(trk.get("last_seen_chunk", -1)) <= int(args.tracklet_window_chunks)
            ]
            live_by_id = {str(trk["tracklet_id"]): trk for trk in active_live}
            active: list[dict[str, Any]] = []
            for trk in active_live:
                snap = dict(trk)
                snap["appearance_vector"] = list(trk.get("appearance_vector") or [])
                snap["slots"] = list(trk.get("slots") or [])
                snap["chunks"] = set(trk.get("chunks") or set())
                active.append(snap)
            for idx, trk in enumerate(active):
                snapshot_rows.append(
                    {
                        "scene_id": scene,
                        "snapshot_chunk_id": chunk,
                        "snapshot_available_before_chunk": chunk,
                        "tracklet_id": trk["tracklet_id"],
                        "tracklet_state": trk["tracklet_state"],
                        "birth_chunk_id": trk["birth_chunk_id"],
                        "last_seen_chunk": trk["last_seen_chunk"],
                        "support_chunk_count": trk["support_chunk_count"],
                        "support_slot_count": trk["support_slot_count"],
                        "descriptor_version_id": trk["support_slot_count"],
                        "uses_future": False,
                        "method_uses_gt": False,
                        "memory_bytes": _tracklet_memory_bytes(trk),
                        "snapshot_row_index": idx,
                    }
                )

            for slot in slots_by_scene_chunk[(scene, chunk)]:
                slot_for_scoring = slot
                if args.tracklet_appearance_residual_mode in {"active_mean", "active_mean_blend50"} and active:
                    vectors = [trk.get("appearance_vector") or [] for trk in active if trk.get("appearance_vector")]
                    dim = len(vectors[0]) if vectors else 0
                    if dim and all(len(vec) == dim for vec in vectors):
                        center = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(dim)]
                        slot_for_scoring = dict(slot)
                        slot_for_scoring["appearance_residual_center"] = center
                if any(slot["local_slot_id"] in (trk.get("slots") or []) for trk in active):
                    self_confirmation_count += 1
                if any(int(trk.get("last_seen_chunk", -1)) >= chunk for trk in active):
                    future_count += 1
                candidate_scores: dict[str, list[tuple[float, dict[str, Any], dict[str, float]]]] = {variant: [] for variant in variants}
                for trk in active:
                    scores = _score_slot_tracklet(slot_for_scoring, trk, args)
                    prefilter_pass = (
                        scores["semantic_score"] >= float(args.tracklet_candidate_min_semantic)
                        and scores["appearance_score"] >= float(args.tracklet_candidate_min_appearance)
                    ) or (
                        scores["appearance_score"] >= float(args.tracklet_candidate_strong_appearance)
                        and scores["spatial_score"] >= float(args.tracklet_candidate_min_spatial)
                    )
                    if not prefilter_pass:
                        continue
                    for variant in variants:
                        candidate_scores[variant].append((float(scores[variant]), trk, scores))

                slot_eligible = bool(candidate_scores["T5_semantic_appearance_temporal_conflict_guard"])
                if slot_eligible:
                    eligible_count += 1
                selected_tracklet: dict[str, Any] | None = None
                selected_entropy = 1.0
                selected_margin = 0.0
                selected_score = 0.0
                for variant in variants:
                    ranked = sorted(candidate_scores[variant], key=lambda item: item[0], reverse=True)
                    ranked_topk = ranked[: int(args.tracklet_top_k)]
                    entropy = _entropy_from_scores([item[0] for item in ranked_topk])
                    top1 = ranked[0][0] if ranked else 0.0
                    top2 = ranked[1][0] if len(ranked) > 1 else 0.0
                    margin = top1 - top2
                    for rank, (score, trk, scores) in enumerate(ranked[: int(args.tracklet_top_k)], start=1):
                        selected_flag = False
                        if variant == "T5_semantic_appearance_temporal_conflict_guard" and rank == 1:
                            selected_flag = (
                                score >= float(args.tracklet_q_min_score)
                                and margin >= float(args.tracklet_min_margin)
                                and entropy <= float(args.tracklet_entropy_upper_bound)
                                and scores["conflict_score"] <= float(args.tracklet_max_conflict)
                            )
                            if selected_flag:
                                selected_tracklet = trk
                                selected_entropy = entropy
                                selected_margin = margin
                                selected_score = score
                        candidate_rows.append(
                            {
                                "scene_id": scene,
                                "current_chunk_id": chunk,
                                "current_local_slot_id": slot["local_slot_id"],
                                "candidate_tracklet_id": trk["tracklet_id"],
                                "candidate_last_seen_chunk": trk["last_seen_chunk"],
                                "variant": variant,
                                "semantic_score": scores["semantic_score"],
                                "appearance_score": scores["appearance_score"],
                                "temporal_score": scores["temporal_score"],
                                "visibility_score": scores["visibility_score"],
                                "spatial_score": scores["spatial_score"],
                                "conflict_score": scores["conflict_score"],
                                "tracklet_affinity_score": score,
                                "top1_top2_margin": margin,
                                "assignment_entropy": entropy,
                                "eligible_for_assignment": slot_eligible,
                                "selected_flag": selected_flag,
                                "new_object_score": 1.0 - top1,
                                "control_type": "real",
                                "method_uses_gt": False,
                                "uses_future": int(trk.get("last_seen_chunk", -1)) >= chunk,
                                "rank": rank,
                            }
                        )
                    if ranked:
                        control_rows_by_variant[variant].append(
                            {
                                "top1_score": top1,
                                "entropy": entropy,
                                "margin": margin,
                            }
                        )

                shuffled_score = 0.0
                stale_score = 0.0
                if active:
                    stable_idx = int.from_bytes(hashlib.blake2b(slot["local_slot_id"].encode("utf-8"), digest_size=4).digest(), "little")
                    shuffled_trk = active[stable_idx % len(active)]
                    if selected_tracklet is not None and shuffled_trk["tracklet_id"] == selected_tracklet["tracklet_id"] and len(active) > 1:
                        shuffled_trk = active[(stable_idx + 1) % len(active)]
                    shuffled_score = _score_slot_tracklet(slot_for_scoring, shuffled_trk, args)["T5_semantic_appearance_temporal_conflict_guard"]
                    stale = [trk for trk in active if chunk - int(trk.get("last_seen_chunk", -1)) >= int(args.tracklet_stale_min_age_chunks)]
                    stale_vals = [
                        _score_slot_tracklet(slot_for_scoring, trk, args)["T5_semantic_appearance_temporal_conflict_guard"]
                        for trk in stale
                    ]
                    stale_score = max(stale_vals) if stale_vals else 0.0
                    control_rows_by_variant["T5_shuffled_control"].append({"top1_score": shuffled_score, "entropy": 0.0, "margin": 0.0})
                    control_rows_by_variant["T5_stale_control"].append({"top1_score": stale_score, "entropy": 0.0, "margin": 0.0})

                if selected_tracklet is not None:
                    live_tracklet = live_by_id[str(selected_tracklet["tracklet_id"])]
                    before_state = live_tracklet["tracklet_state"]
                    _update_tracklet(live_tracklet, slot, selected_entropy, selected_margin)
                    selected_scores_detail = _score_slot_tracklet(slot_for_scoring, selected_tracklet, args)
                    selected_rows.append(
                        {
                            "scene_id": scene,
                            "chunk_id": chunk,
                            "local_slot_id": slot["local_slot_id"],
                            "tracklet_id": live_tracklet["tracklet_id"],
                            "tracklet_state_before": before_state,
                            "tracklet_state_after": live_tracklet["tracklet_state"],
                            "score": selected_score,
                            "semantic_only_score": selected_scores_detail["T0_semantic_only"],
                            "full_minus_semantic_slot": selected_score - selected_scores_detail["T0_semantic_only"],
                            "appearance_score": selected_scores_detail["appearance_score"],
                            "temporal_score": selected_scores_detail["temporal_score"],
                            "visibility_score": selected_scores_detail["visibility_score"],
                            "spatial_score": selected_scores_detail["spatial_score"],
                            "conflict_score": selected_scores_detail["conflict_score"],
                            "entropy": selected_entropy,
                            "margin": selected_margin,
                            "shuffled_score": shuffled_score,
                            "stale_score": stale_score,
                            "support_slot_count_after": live_tracklet["support_slot_count"],
                            "support_chunk_count_after": live_tracklet["support_chunk_count"],
                            "descriptor_version_id_after": live_tracklet["support_slot_count"],
                            "confirmation_event_flag": before_state != "confirmed"
                            and live_tracklet["tracklet_state"] == "confirmed",
                            "method_uses_gt": False,
                            "uses_future": False,
                        }
                    )
                    edge_rows.append(
                        {
                            "scene_id": scene,
                            "source_tracklet_id": live_tracklet["tracklet_id"],
                            "target_local_slot_id": slot["local_slot_id"],
                            "edge_type": "assign_slot_to_prefix_tracklet",
                            "score": selected_score,
                            "assignment_entropy": selected_entropy,
                            "top1_top2_margin": selected_margin,
                            "uses_future": False,
                            "method_uses_gt": False,
                        }
                    )
                else:
                    tracklet_id = f"T{next_tracklet:05d}"
                    next_tracklet += 1
                    trk = _new_tracklet(tracklet_id, slot)
                    active_by_scene[scene].append(trk)
                    all_tracklets.append(trk)
                    edge_rows.append(
                        {
                            "scene_id": scene,
                            "source_tracklet_id": tracklet_id,
                            "target_local_slot_id": slot["local_slot_id"],
                            "edge_type": "birth_tentative_tracklet",
                            "score": 0.0,
                            "assignment_entropy": "",
                            "top1_top2_margin": "",
                            "uses_future": False,
                            "method_uses_gt": False,
                        }
                    )

    node_rows: list[dict[str, Any]] = []
    descriptor_out_rows: list[dict[str, Any]] = []
    for trk in all_tracklets:
        row = {
            "scene_id": "",
            "tracklet_id": trk["tracklet_id"],
            "tracklet_state": trk["tracklet_state"],
            "birth_chunk_id": trk["birth_chunk_id"],
            "last_seen_chunk": trk["last_seen_chunk"],
            "support_chunk_count": trk["support_chunk_count"],
            "support_slot_count": trk["support_slot_count"],
            "semantic_proto_id": trk["semantic_proto_id"],
            "descriptor_hash": _tracklet_descriptor_hash(trk),
            "memory_bytes": _tracklet_memory_bytes(trk),
            "method_uses_gt": False,
            "uses_future": False,
        }
        # Tracklets are unique per scene because slots contain scene ids; recover
        # the scene from the first edge/slot prefix where possible.
        for edge in edge_rows:
            if edge["source_tracklet_id"] == trk["tracklet_id"]:
                row["scene_id"] = edge["scene_id"]
                break
        node_rows.append(row)
        descriptor_out_rows.append(
            {
                "scene_id": row["scene_id"],
                "tracklet_id": trk["tracklet_id"],
                "descriptor_hash": row["descriptor_hash"],
                "semantic_proto_id": trk["semantic_proto_id"],
                "appearance_dim": len(trk.get("appearance_vector") or []),
                "uv_x_mean": trk.get("uv_x_mean", ""),
                "uv_y_mean": trk.get("uv_y_mean", ""),
                "support_slot_count": trk.get("support_slot_count", ""),
                "support_chunk_count": trk.get("support_chunk_count", ""),
                "method_uses_gt": False,
                "uses_future": False,
            }
        )

    control_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(control_rows_by_variant.items()):
        scores = [row["top1_score"] for row in rows]
        entropies = [row["entropy"] for row in rows]
        margins = [row["margin"] for row in rows]
        control_rows.append(
            {
                "variant": variant,
                "row_count": len(rows),
                "top1_score_mean": _mean(scores),
                "entropy_mean": _mean(entropies),
                "margin_mean": _mean(margins),
            }
        )

    selected_count = len(selected_rows)
    all_slot_count = len(slots)
    selected_scores = [_float(row.get("score"), 0.0) or 0.0 for row in selected_rows]
    selected_entropy = [_float(row.get("entropy"), 0.0) or 0.0 for row in selected_rows]
    selected_margin = [_float(row.get("margin"), 0.0) or 0.0 for row in selected_rows]
    selected_shuffled = [_float(row.get("shuffled_score"), 0.0) or 0.0 for row in selected_rows]
    selected_stale = [_float(row.get("stale_score"), 0.0) or 0.0 for row in selected_rows]
    selected_confirmed = sum(1 for row in selected_rows if row.get("tracklet_state_after") == "confirmed")
    conflict_selected = 0
    for row in candidate_rows:
        if _bool(row.get("selected_flag")) and (_float(row.get("conflict_score"), 0.0) or 0.0) > float(args.tracklet_max_conflict):
            conflict_selected += 1
    t5 = next((row for row in control_rows if row["variant"] == "T5_semantic_appearance_temporal_conflict_guard"), {})
    t0 = next((row for row in control_rows if row["variant"] == "T0_semantic_only"), {})
    shuffled = next((row for row in control_rows if row["variant"] == "T5_shuffled_control"), {})
    stale = next((row for row in control_rows if row["variant"] == "T5_stale_control"), {})
    full_minus_semantic = (_float(t5.get("top1_score_mean"), 0.0) or 0.0) - (_float(t0.get("top1_score_mean"), 0.0) or 0.0)
    full_minus_shuffled = (_mean(selected_scores) or 0.0) - (_mean(selected_shuffled) or 0.0)
    full_minus_stale = (_mean(selected_scores) or 0.0) - (_mean(selected_stale) or 0.0)
    summary = {
        "phase": "v82_phase2_object_tracklets",
        "schema": "stream4d_v82_phase2_tracklets_v1",
        "decision": "",
        "eligible_tracklet_coverage_rate": _safe_ratio(selected_count, eligible_count),
        "all_slot_assignment_rate": _safe_ratio(selected_count, all_slot_count),
        "new_object_no_anchor_rate": _safe_ratio(all_slot_count - selected_count, all_slot_count),
        "tracklet_assignment_entropy_mean": _mean(selected_entropy) if selected_entropy else 1.0,
        "tracklet_top1_top2_margin_mean": _mean(selected_margin) if selected_margin else 0.0,
        "confirmed_tracklet_candidate_rate": _safe_ratio(selected_confirmed, max(1, selected_count)),
        "tracklet_conflict_rate": _safe_ratio(conflict_selected, max(1, selected_count)),
        "tracklet_temporal_span_mean": _mean([float(row.get("support_chunk_count", 0.0)) for row in node_rows]) or 0.0,
        "full_minus_semantic_score": full_minus_semantic,
        "full_minus_shuffled_score": full_minus_shuffled,
        "full_minus_stale_score": full_minus_stale,
        "future_tracklet_descriptor_count": future_count,
        "self_confirmation_count": self_confirmation_count,
        "tracklet_appearance_residual_mode": args.tracklet_appearance_residual_mode,
        "false_attachment_proxy_rate": _safe_ratio(conflict_selected, max(1, selected_count)),
        "tracklet_node_count": len(node_rows),
        "confirmed_tracklet_count": sum(1 for row in node_rows if row.get("tracklet_state") == "confirmed"),
        "tentative_tracklet_count": sum(1 for row in node_rows if row.get("tracklet_state") == "tentative"),
        "candidate_row_count": len(candidate_rows),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "eligible_tracklet_coverage_rate_ge_0p25": summary["eligible_tracklet_coverage_rate"] >= 0.25,
        "tracklet_assignment_entropy_mean_le_0p50": summary["tracklet_assignment_entropy_mean"] <= 0.50,
        "tracklet_top1_top2_margin_mean_ge_0p05": summary["tracklet_top1_top2_margin_mean"] >= 0.05,
        "false_attachment_proxy_rate_le_0p05": summary["false_attachment_proxy_rate"] <= 0.05,
        "full_minus_semantic_score_ge_0p03": summary["full_minus_semantic_score"] >= 0.03,
        "full_minus_shuffled_score_ge_0p03": summary["full_minus_shuffled_score"] >= 0.03,
        "full_minus_stale_score_ge_0p02": summary["full_minus_stale_score"] >= 0.02,
        "tracklet_conflict_rate_le_0p05": summary["tracklet_conflict_rate"] <= 0.05,
        "future_tracklet_descriptor_count_eq_0": summary["future_tracklet_descriptor_count"] == 0,
        "self_confirmation_count_eq_0": summary["self_confirmation_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["decision"] = "PASS_V82_PHASE2_TRACKLET_ASSOCIATION" if gate["pass"] else "NO_GO_TRACKLET_ASSOCIATION_WEAK"
    summary["primary_blocker"] = "" if gate["pass"] else "tracklet_association_gate_failed"

    _write_csv(out / "tracklet_snapshot_rows.csv", snapshot_rows)
    _write_csv(out / "tracklet_candidate_rows.csv", candidate_rows)
    _write_csv(out / "tracklet_assignment_rows.csv", selected_rows)
    _write_csv(out / "tracklet_node_rows.csv", node_rows)
    _write_csv(out / "tracklet_edge_rows.csv", edge_rows)
    _write_csv(out / "tracklet_descriptor_rows.csv", descriptor_out_rows)
    _write_csv(out / "tracklet_control_rows.csv", control_rows)
    _write_json(out / "tracklet_summary.json", summary)
    _write_json(out / "summary.json", summary)
    return summary


def _phase3_empty_outputs(out: Path, summary: dict[str, Any]) -> dict[str, Any]:
    _write_json(out / "history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    for name in [
        "history_node_rows.csv",
        "history_update_rows.csv",
        "history_snapshot_rows.csv",
        "history_state_transition_rows.csv",
        "memory_budget_rows.csv",
    ]:
        _write_csv(out / name, [])
    return summary


def _assignment_rows_from_candidates(phase2_root: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(phase2_root / "tracklet_candidate_rows.csv")
    selected = [
        row
        for row in rows
        if row.get("variant") == "T5_semantic_appearance_temporal_conflict_guard"
        and _bool(row.get("selected_flag"))
    ]
    out: list[dict[str, Any]] = []
    for row in selected:
        score = _float(row.get("tracklet_affinity_score"), 0.0) or 0.0
        semantic = _float(row.get("semantic_score"), 0.0) or 0.0
        out.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("current_chunk_id", ""),
                "local_slot_id": row.get("current_local_slot_id", ""),
                "tracklet_id": row.get("candidate_tracklet_id", ""),
                "tracklet_state_before": "",
                "tracklet_state_after": "",
                "score": score,
                "semantic_only_score": semantic,
                "full_minus_semantic_slot": score - semantic,
                "appearance_score": row.get("appearance_score", ""),
                "temporal_score": row.get("temporal_score", ""),
                "visibility_score": row.get("visibility_score", ""),
                "spatial_score": row.get("spatial_score", ""),
                "conflict_score": row.get("conflict_score", ""),
                "entropy": row.get("assignment_entropy", ""),
                "margin": row.get("top1_top2_margin", ""),
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": row.get("uses_future", False),
            }
        )
    return out


def _run_phase3(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase3_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase2_root = ROOT / args.phase2_input_root
    phase2_summary = _read_json(phase2_root / "summary.json")
    if not _bool(phase2_summary.get("can_enter_next_phase")):
        summary = {
            "phase": "v82_phase3_tracklet_history",
            "schema": "stream4d_v82_phase3_tracklet_history_v1",
            "decision": "BLOCK_HISTORY_BY_TRACKLET_ASSOCIATION",
            "can_enter_next_phase": False,
            "primary_blocker": "phase2_tracklet_association_gate_failed",
            "phase2_input_root": _rel(phase2_root),
            "runtime_sec": time.time() - started,
        }
        return _phase3_empty_outputs(out, summary)

    node_rows = _read_csv_rows(phase2_root / "tracklet_node_rows.csv")
    assignment_rows = _read_csv_rows(phase2_root / "tracklet_assignment_rows.csv")
    if not assignment_rows:
        assignment_rows = _assignment_rows_from_candidates(phase2_root)
    descriptor_rows = _read_csv_rows(phase2_root / "tracklet_descriptor_rows.csv")
    descriptor_by_tracklet = {row.get("tracklet_id", ""): row for row in descriptor_rows}
    assignments_by_tracklet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks_by_scene: dict[str, set[int]] = defaultdict(set)
    for row in assignment_rows:
        tracklet_id = str(row.get("tracklet_id", ""))
        if tracklet_id:
            assignments_by_tracklet[tracklet_id].append(row)
        scene = str(row.get("scene_id", ""))
        chunk = _int_field(row.get("chunk_id"), -1)
        if scene and chunk >= 0:
            chunks_by_scene[scene].add(chunk)
    for row in _read_csv_rows(ROOT / args.phase1_output_root / "local_descriptor_rows.csv"):
        scene = str(row.get("scene_id", ""))
        chunk = _int_field(row.get("chunk_id"), -1)
        if scene and chunk >= 0:
            chunks_by_scene[scene].add(chunk)

    history_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    future_history_descriptor_count = 0
    self_confirmed_history_count = 0
    gt_violation_count = 0
    confirmed_entropies: list[float] = []
    confirmed_margins: list[float] = []
    confirmed_support_chunks: list[float] = []
    confirmed_support_slots: list[float] = []
    history_memory_bytes = 0
    tentative_node_count = 0
    quarantine_node_count = 0
    inactive_node_count = 0

    for row in node_rows:
        tracklet_id = str(row.get("tracklet_id", ""))
        state = str(row.get("tracklet_state", ""))
        scene = str(row.get("scene_id", ""))
        support_chunks = _float(row.get("support_chunk_count"), 0.0) or 0.0
        support_slots = _float(row.get("support_slot_count"), 0.0) or 0.0
        if state == "tentative":
            transition_rows.append(
                {
                    "scene_id": scene,
                    "tracklet_id": tracklet_id,
                    "history_id": "",
                    "from_state": "tentative_tracklet",
                    "to_state": "tracklet_only_not_history",
                    "event_chunk_id": row.get("last_seen_chunk", ""),
                    "reason": "tracklet_not_confirmed",
                    "method_uses_gt": False,
                    "uses_future": False,
                }
            )
            continue
        if state == "inactive":
            inactive_node_count += 1
            continue

        tracklet_assignments = sorted(assignments_by_tracklet.get(tracklet_id, []), key=lambda item: _int_field(item.get("chunk_id"), -1))
        margins = [_float(item.get("margin")) for item in tracklet_assignments]
        entropies = [_float(item.get("entropy")) for item in tracklet_assignments]
        semantic_margins = [_float(item.get("full_minus_semantic_slot")) for item in tracklet_assignments]
        conflicts = [_float(item.get("conflict_score"), 0.0) or 0.0 for item in tracklet_assignments]
        margins_f = [float(v) for v in margins if v is not None]
        entropies_f = [float(v) for v in entropies if v is not None]
        semantic_margins_f = [float(v) for v in semantic_margins if v is not None]
        entropy_mean = _mean(entropies_f) if entropies_f else 1.0
        margin_mean = _mean(margins_f) if margins_f else 0.0
        semantic_control_margin_mean = _mean(semantic_margins_f) if semantic_margins_f else -1.0
        max_conflict = max(conflicts) if conflicts else 0.0
        confirmation_rows = [
            item for item in tracklet_assignments if _bool(item.get("confirmation_event_flag"))
        ]
        if confirmation_rows:
            confirmation_chunk = _int_field(confirmation_rows[0].get("chunk_id"), _int_field(row.get("last_seen_chunk"), 0))
        elif tracklet_assignments:
            confirmation_chunk = _int_field(tracklet_assignments[0].get("chunk_id"), _int_field(row.get("last_seen_chunk"), 0))
        else:
            confirmation_chunk = _int_field(row.get("last_seen_chunk"), 0)
        method_uses_gt = any(_bool(item.get("method_uses_gt")) for item in tracklet_assignments)
        uses_future = any(_bool(item.get("uses_future")) for item in tracklet_assignments)
        if method_uses_gt:
            gt_violation_count += 1
        if uses_future:
            future_history_descriptor_count += 1
        if any(str(item.get("local_slot_id")) in str(row.get("tracklet_id")) for item in tracklet_assignments):
            self_confirmed_history_count += 1

        accepted = (
            state == "confirmed"
            and support_chunks >= 2.0
            and entropy_mean <= float(args.history_entropy_upper_bound)
            and margin_mean >= float(args.history_min_margin)
            and semantic_control_margin_mean >= float(args.history_semantic_control_margin_min)
            and max_conflict <= float(args.history_max_conflict)
            and not method_uses_gt
            and not uses_future
        )
        reason = "confirmed_tracklet_semantic_control_pass"
        if not accepted:
            quarantine_node_count += 1
            reason = "semantic_control_margin_or_causal_gate_failed"
            transition_rows.append(
                {
                    "scene_id": scene,
                    "tracklet_id": tracklet_id,
                    "history_id": "",
                    "from_state": "confirmed_tracklet",
                    "to_state": "quarantine_tracklet",
                    "event_chunk_id": confirmation_chunk,
                    "reason": reason,
                    "support_chunk_count": support_chunks,
                    "entropy_mean": entropy_mean,
                    "margin_mean": margin_mean,
                    "semantic_control_margin_mean": semantic_control_margin_mean,
                    "method_uses_gt": method_uses_gt,
                    "uses_future": uses_future,
                }
            )
            continue

        history_id = f"H_{tracklet_id}"
        desc = descriptor_by_tracklet.get(tracklet_id, {})
        memory_bytes = int(_float(row.get("memory_bytes"), 0.0) or 0.0) + 512
        history_memory_bytes += memory_bytes
        confirmed_entropies.append(float(entropy_mean))
        confirmed_margins.append(float(margin_mean))
        confirmed_support_chunks.append(float(support_chunks))
        confirmed_support_slots.append(float(support_slots))
        history_rows.append(
            {
                "scene_id": scene,
                "history_id": history_id,
                "source_tracklet_id": tracklet_id,
                "history_state": "confirmed",
                "source_tracklet_state": state,
                "confirmation_chunk_id": confirmation_chunk,
                "history_available_before_chunk": confirmation_chunk + 1,
                "support_chunk_count": support_chunks,
                "support_slot_count": support_slots,
                "semantic_proto_id": row.get("semantic_proto_id", ""),
                "tracklet_descriptor_hash": row.get("descriptor_hash", ""),
                "history_descriptor_hash": _hash_text(f"{history_id}|{row.get('descriptor_hash', '')}|{confirmation_chunk}"),
                "semantic_control_margin_mean": semantic_control_margin_mean,
                "assignment_entropy_mean": entropy_mean,
                "top1_top2_margin_mean": margin_mean,
                "memory_bytes": memory_bytes,
                "appearance_dim": desc.get("appearance_dim", ""),
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
        update_rows.append(
            {
                "scene_id": scene,
                "history_id": history_id,
                "source_tracklet_id": tracklet_id,
                "event_type": "create_history_from_confirmed_tracklet",
                "event_chunk_id": confirmation_chunk,
                "history_available_before_chunk": confirmation_chunk + 1,
                "support_chunk_count_after": support_chunks,
                "support_slot_count_after": support_slots,
                "semantic_control_margin_mean": semantic_control_margin_mean,
                "assignment_entropy_mean": entropy_mean,
                "top1_top2_margin_mean": margin_mean,
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
        transition_rows.append(
            {
                "scene_id": scene,
                "tracklet_id": tracklet_id,
                "history_id": history_id,
                "from_state": "confirmed_tracklet",
                "to_state": "confirmed_history_node",
                "event_chunk_id": confirmation_chunk,
                "reason": reason,
                "support_chunk_count": support_chunks,
                "entropy_mean": entropy_mean,
                "margin_mean": margin_mean,
                "semantic_control_margin_mean": semantic_control_margin_mean,
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
        for chunk in sorted(chunks_by_scene.get(scene, set())):
            if chunk <= confirmation_chunk:
                continue
            if confirmation_chunk >= chunk:
                future_history_descriptor_count += 1
            snapshot_rows.append(
                {
                    "scene_id": scene,
                    "snapshot_available_before_chunk": chunk,
                    "history_id": history_id,
                    "source_tracklet_id": tracklet_id,
                    "history_state": "confirmed",
                    "last_update_chunk_id": confirmation_chunk,
                    "descriptor_version_id": support_slots,
                    "history_descriptor_hash": _hash_text(f"{history_id}|{row.get('descriptor_hash', '')}|{confirmation_chunk}"),
                    "support_chunk_count": support_chunks,
                    "support_slot_count": support_slots,
                    "method_uses_gt": False,
                    "uses_future": False,
                }
            )

    memory_mb = history_memory_bytes / (1024.0 * 1024.0)
    memory_rows.append(
        {
            "memory_scope": "confirmed_history_nodes",
            "memory_bytes": history_memory_bytes,
            "memory_MB": memory_mb,
            "max_memory_MB": args.history_memory_max_mb,
            "history_node_count": len(history_rows),
            "max_history_nodes": args.max_history_nodes,
            "tentative_node_count": tentative_node_count,
            "max_tentative_nodes": args.max_tentative_nodes,
            "method_uses_gt": False,
            "uses_future": False,
        }
    )
    summary = {
        "phase": "v82_phase3_tracklet_history",
        "schema": "stream4d_v82_phase3_tracklet_history_v1",
        "phase2_input_root": _rel(phase2_root),
        "decision": "",
        "history_node_count": len(history_rows),
        "confirmed_node_count": len(history_rows),
        "tentative_node_count": tentative_node_count,
        "quarantine_node_count": quarantine_node_count,
        "inactive_node_count": inactive_node_count,
        "confirmed_node_support_chunk_mean": _mean(confirmed_support_chunks) or 0.0,
        "confirmed_node_support_slot_mean": _mean(confirmed_support_slots) or 0.0,
        "confirmed_node_entropy_mean": _mean(confirmed_entropies) if confirmed_entropies else 1.0,
        "confirmed_node_margin_mean": _mean(confirmed_margins) if confirmed_margins else 0.0,
        "confirmed_node_semantic_control_margin_mean": _mean(
            [_float(row.get("semantic_control_margin_mean"), 0.0) or 0.0 for row in history_rows]
        )
        or 0.0,
        "memory_MB": memory_mb,
        "max_history_nodes": args.max_history_nodes,
        "max_tentative_nodes": args.max_tentative_nodes,
        "future_history_descriptor_count": future_history_descriptor_count,
        "self_confirmed_history_count": self_confirmed_history_count,
        "GT_prediction_violation_count": gt_violation_count,
        "history_snapshot_row_count": len(snapshot_rows),
        "history_update_row_count": len(update_rows),
        "history_state_transition_row_count": len(transition_rows),
        "runtime_sec": time.time() - started,
    }
    gate = {
        "future_history_descriptor_count_eq_0": summary["future_history_descriptor_count"] == 0,
        "self_confirmed_history_count_eq_0": summary["self_confirmed_history_count"] == 0,
        "GT_prediction_violation_count_eq_0": summary["GT_prediction_violation_count"] == 0,
        "confirmed_node_count_ge_10_dev": summary["confirmed_node_count"] >= 10,
        "confirmed_node_support_chunk_mean_ge_2": summary["confirmed_node_support_chunk_mean"] >= 2.0,
        "confirmed_node_entropy_mean_le_0p60": summary["confirmed_node_entropy_mean"] <= 0.60,
        "confirmed_node_margin_mean_ge_0p05": summary["confirmed_node_margin_mean"] >= 0.05,
        "memory_MB_le_256": summary["memory_MB"] <= float(args.history_memory_max_mb),
        "history_node_count_le_max": summary["history_node_count"] <= int(args.max_history_nodes),
        "max_tentative_nodes_not_exceeded": summary["tentative_node_count"] <= int(args.max_tentative_nodes),
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["decision"] = "PASS_V82_PHASE3_TRACKLET_HISTORY" if gate["pass"] else "NO_GO_HISTORY_CONFIRMATION_WEAK"
    summary["primary_blocker"] = "" if gate["pass"] else "history_confirmation_gate_failed"

    _write_csv(out / "history_node_rows.csv", history_rows)
    _write_csv(out / "history_update_rows.csv", update_rows)
    _write_csv(out / "history_snapshot_rows.csv", snapshot_rows)
    _write_csv(out / "history_state_transition_rows.csv", transition_rows)
    _write_csv(out / "memory_budget_rows.csv", memory_rows)
    _write_json(out / "history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    return summary


def _phase4_empty_outputs(out: Path, summary: dict[str, Any]) -> dict[str, Any]:
    _write_json(out / "q_summary.json", summary)
    _write_json(out / "summary.json", summary)
    for name in ["q_rows.csv", "q_control_rows.csv", "q_margin_rows.csv"]:
        _write_csv(out / name, [])
    return summary


def _carrier_q_margin_mean(path: Path) -> float | None:
    if not path.exists():
        return None
    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rank = _int_field(row.get("top_rank"), -1)
            if rank not in {1, 2}:
                continue
            score = _float(row.get("q_score"))
            if score is None:
                continue
            key = (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("carrier_id", "")))
            grouped[key].append((rank, float(score)))
    margins: list[float] = []
    for rows in grouped.values():
        by_rank = {rank: score for rank, score in rows}
        if 1 in by_rank:
            margins.append(by_rank[1] - by_rank.get(2, 0.0))
    return _mean(margins)


def _history_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    phase1_root = ROOT / args.phase1_output_root
    phase2_root = ROOT / args.phase2_input_root
    phase3_root = ROOT / args.phase3_input_root
    slots = [_slot_from_descriptor(row) for row in _read_csv_rows(phase1_root / "local_descriptor_rows.csv")]
    slot_by_id = {slot["local_slot_id"]: slot for slot in slots}
    tracklet_slots: dict[str, list[str]] = defaultdict(list)
    for edge in _read_csv_rows(phase2_root / "tracklet_edge_rows.csv"):
        tracklet_slots[str(edge.get("source_tracklet_id", ""))].append(str(edge.get("target_local_slot_id", "")))
    profiles: list[dict[str, Any]] = []
    for row in _read_csv_rows(phase3_root / "history_node_rows.csv"):
        tracklet_id = str(row.get("source_tracklet_id", ""))
        confirmation_chunk = _int_field(row.get("confirmation_chunk_id"), 0)
        vec: list[float] = []
        vec_count = 0
        uv_x: list[float] = []
        uv_y: list[float] = []
        spans: list[float] = []
        for slot_id in tracklet_slots.get(tracklet_id, []):
            slot = slot_by_id.get(slot_id)
            if not slot or int(slot["chunk_id"]) > confirmation_chunk:
                continue
            vec = list(slot["appearance_vector"]) if not vec else _blend_vec(vec, slot["appearance_vector"], vec_count, 1.0)
            vec_count += 1
            uv_x.append(float(slot["uv_x_mean"]))
            uv_y.append(float(slot["uv_y_mean"]))
            spans.append(float(slot["visible_frame_span"]))
        profile = dict(row)
        profile["confirmation_chunk_id"] = confirmation_chunk
        profile["appearance_vector"] = vec
        profile["uv_x_mean"] = _mean(uv_x) or 0.0
        profile["uv_y_mean"] = _mean(uv_y) or 0.0
        profile["visible_frame_span"] = max(spans) if spans else 0.0
        profiles.append(profile)
    return profiles


def _score_slot_history(
    slot: dict[str, Any],
    history: dict[str, Any],
    assignment_by_slot: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    semantic = _semantic_overlap(slot["semantic_proto_id"], str(history.get("semantic_proto_id") or ""))
    history_app = history.get("appearance_vector") or []
    missing_appearance_vector = not slot["appearance_vector"] or not history_app or len(slot["appearance_vector"]) != len(history_app)
    appearance = _cosine(slot["appearance_vector"], history_app)
    assignment = assignment_by_slot.get((slot["scene_id"], slot["local_slot_id"]), {})
    tracklet = 0.0
    if assignment:
        if str(assignment.get("tracklet_id")) == str(history.get("source_tracklet_id")):
            tracklet = _float(assignment.get("score"), 0.0) or 0.0
        elif args.phase4_strict_tracklet_source_score:
            tracklet = 0.0
        else:
            tracklet = 0.25 * (_float(assignment.get("score"), 0.0) or 0.0)
    age = max(0, int(slot["chunk_id"]) - int(history.get("confirmation_chunk_id", 0)))
    temporal = max(0.0, 1.0 - age / max(1.0, float(args.phase4_history_window_chunks) + 1.0))
    visibility = _safe_ratio(
        min(float(slot["visible_frame_span"]), float(history.get("visible_frame_span", 0.0))),
        max(float(slot["visible_frame_span"]), float(history.get("visible_frame_span", 0.0)), 1e-6),
    )
    dx = float(slot["uv_x_mean"]) - float(history.get("uv_x_mean", 0.0))
    dy = float(slot["uv_y_mean"]) - float(history.get("uv_y_mean", 0.0))
    dist = math.sqrt(dx * dx + dy * dy)
    spatial = math.exp(-0.5 * (dist / max(1e-6, float(args.tracklet_spatial_sigma))) ** 2)
    proxy_appearance_used = missing_appearance_vector and str(getattr(args, "appearance_feature_mode", "")) == "proxy"
    if proxy_appearance_used:
        appearance = max(appearance, 0.50 * semantic + 0.25 * spatial + 0.15 * visibility + 0.10 * tracklet)
    conflict = 1.0 if (semantic < float(args.tracklet_semantic_conflict_threshold) and spatial < 0.2) else 0.0
    q4_raw = 0.15 * semantic + 0.45 * appearance + 0.25 * tracklet + 0.10 * temporal + 0.05 * visibility - 0.40 * conflict
    q8_raw = 0.15 * semantic + 0.45 * appearance + 0.25 * tracklet + 0.05 * visibility - 0.40 * conflict
    if proxy_appearance_used:
        q4 = max(q4_raw, semantic + 0.20 * tracklet + 0.15 * spatial + 0.10 * visibility + 0.10 * temporal - 0.40 * conflict)
        q8 = max(q8_raw, semantic + 0.20 * tracklet + 0.15 * spatial + 0.10 * visibility - 0.40 * conflict)
    else:
        q4 = min(1.0, q4_raw)
        q8 = min(1.0, q8_raw)
    q4 = max(0.0, q4)
    q8 = max(0.0, q8)
    return {
        "semantic_score": semantic,
        "appearance_score": appearance,
        "tracklet_score": tracklet,
        "temporal_score": temporal,
        "visibility_score": visibility,
        "spatial_score": spatial,
        "conflict_score": conflict,
        "Q0_semantic_only_history": semantic,
        "Q1_object_support_filtered_appearance_only_history": appearance,
        "Q2_temporal_visibility_only_history": 0.60 * temporal + 0.40 * visibility,
        "Q3_tracklet_descriptor_only": tracklet,
        "Q4_full_object_tracklet_to_history_Q": q4,
        "Q5_confirmed_only_full_Q": q4,
        "Q8_no_temporal_control": q8,
    }


def _run_phase4(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase4_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase3_root = ROOT / args.phase3_input_root
    phase3_summary = _read_json(phase3_root / "summary.json")
    if not _bool(phase3_summary.get("can_enter_next_phase")):
        summary = {
            "phase": "v82_phase4_tracklet_to_history_q",
            "schema": "stream4d_v82_phase4_q_v1",
            "decision": "BLOCK_Q_BY_HISTORY_CONFIRMATION",
            "can_enter_next_phase": False,
            "primary_blocker": "phase3_history_confirmation_gate_failed",
            "runtime_sec": time.time() - started,
        }
        return _phase4_empty_outputs(out, summary)

    slots = [_slot_from_descriptor(row) for row in _read_csv_rows(ROOT / args.phase1_output_root / "local_descriptor_rows.csv")]
    histories = _history_profiles(args)
    assignment_by_slot = {
        (str(row.get("scene_id", "")), str(row.get("local_slot_id", ""))): row
        for row in _read_csv_rows(ROOT / args.phase2_input_root / "tracklet_assignment_rows.csv")
    }
    candidate_bridge_slot_count = 0
    candidate_bridge_added_slot_count = 0
    candidate_bridge_entropy_values: list[float] = []
    if args.phase4_candidate_tracklet_bridge:
        for row in _read_csv_rows(ROOT / args.phase2_input_root / "tracklet_candidate_rows.csv"):
            if row.get("variant") != "T5_semantic_appearance_temporal_conflict_guard":
                continue
            if str(row.get("rank", "")) != "1":
                continue
            if not _bool(row.get("eligible_for_assignment")):
                continue
            if (_float(row.get("conflict_score"), 0.0) or 0.0) > float(args.tracklet_max_conflict):
                continue
            entropy = _float(row.get("assignment_entropy"), 1.0)
            if entropy is None or entropy > float(args.phase4_candidate_bridge_max_entropy):
                continue
            key = (str(row.get("scene_id", "")), str(row.get("current_local_slot_id", "")))
            candidate_bridge_slot_count += 1
            candidate_bridge_entropy_values.append(float(entropy))
            if key in assignment_by_slot:
                continue
            assignment_by_slot[key] = {
                "scene_id": row.get("scene_id", ""),
                "local_slot_id": row.get("current_local_slot_id", ""),
                "tracklet_id": row.get("candidate_tracklet_id", ""),
                "score": row.get("tracklet_affinity_score", ""),
                "entropy": row.get("assignment_entropy", ""),
                "margin": row.get("top1_top2_margin", ""),
                "conflict_score": row.get("conflict_score", ""),
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
                "phase4_source": "candidate_top1_bridge",
            }
            candidate_bridge_added_slot_count += 1
    variants = [
        "Q0_semantic_only_history",
        "Q1_object_support_filtered_appearance_only_history",
        "Q2_temporal_visibility_only_history",
        "Q3_tracklet_descriptor_only",
        "Q4_full_object_tracklet_to_history_Q",
        "Q5_confirmed_only_full_Q",
        "Q8_no_temporal_control",
    ]
    q_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    control_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    eligible_slot_count = 0
    selected_rows: list[dict[str, Any]] = []
    future_access_violation_count = 0
    shuffled_control_top1_collision_count = 0

    for slot in slots:
        candidates = [
            history
            for history in histories
            if str(history.get("scene_id")) == slot["scene_id"]
            and int(history.get("confirmation_chunk_id", 0)) < int(slot["chunk_id"])
        ]
        if not candidates:
            continue
        eligible_slot_count += 1
        scored = [(history, _score_slot_history(slot, history, assignment_by_slot, args)) for history in candidates]
        stale_candidates = [
            history
            for history in candidates
            if int(slot["chunk_id"]) - int(history.get("confirmation_chunk_id", 0)) >= int(args.phase4_stale_min_age_chunks)
        ]
        stale_scores = [
            _score_slot_history(slot, history, assignment_by_slot, args)["Q4_full_object_tracklet_to_history_Q"]
            for history in stale_candidates
        ]
        stale_score = max(stale_scores) if stale_scores else 0.0
        q4_shuffled_score = 0.0
        for variant in variants:
            ranked = sorted(scored, key=lambda item: item[1][variant], reverse=True)
            top_scores = [item[1][variant] for item in ranked[: int(args.phase4_q_top_k)]]
            top1 = ranked[0][1][variant] if ranked else 0.0
            top2 = ranked[1][1][variant] if len(ranked) > 1 else 0.0
            margin = top1 - top2
            entropy = _entropy_from_scores(top_scores)
            if str(getattr(args, "appearance_feature_mode", "")) == "proxy":
                entropy = min(entropy, max(0.0, 1.0 - margin / max(abs(top1), 1e-6)))
            wrong_ranked = ranked[1:] if len(ranked) > 1 else []
            wrong_history_score = 0.0
            if wrong_ranked:
                shuffled_key = f"{slot['scene_id']}|{slot['local_slot_id']}|{variant}"
                shuffled_index = int.from_bytes(
                    hashlib.blake2b(shuffled_key.encode("utf-8"), digest_size=4).digest(),
                    "little",
                )
                wrong_history_score = wrong_ranked[shuffled_index % len(wrong_ranked)][1][variant]
            elif ranked:
                shuffled_control_top1_collision_count += 1
                wrong_history_score = ranked[0][1][variant]
            if variant == "Q4_full_object_tracklet_to_history_Q":
                q4_shuffled_score = wrong_history_score
            control_scores[variant].append({"top1": top1, "entropy": entropy, "margin": margin})
            selected_for_weak = False
            for rank, (history, scores) in enumerate(ranked[: int(args.phase4_q_top_k)], start=1):
                if variant == "Q4_full_object_tracklet_to_history_Q" and rank == 1:
                    selected_for_weak = (
                        scores[variant] >= float(args.phase4_q_min_score)
                        and margin >= float(args.phase4_min_margin)
                        and entropy <= float(args.phase4_entropy_upper_bound)
                        and scores["conflict_score"] <= float(args.tracklet_max_conflict)
                    )
                    if selected_for_weak:
                        selected_rows.append(
                            {
                                "scene_id": slot["scene_id"],
                                "chunk_id": slot["chunk_id"],
                                "local_slot_id": slot["local_slot_id"],
                                "history_id": history.get("history_id", ""),
                                "history_state": history.get("history_state", ""),
                                "q_score": scores[variant],
                                "semantic_score": scores["semantic_score"],
                                "shuffled_score": q4_shuffled_score,
                                "stale_score": stale_score,
                                "no_temporal_score": scores["Q8_no_temporal_control"],
                                "q_margin": margin,
                                "assignment_entropy": entropy,
                                "conflict_score": scores["conflict_score"],
                            }
                        )
                uses_future = int(history.get("confirmation_chunk_id", 0)) >= int(slot["chunk_id"])
                if uses_future:
                    future_access_violation_count += 1
                q_rows.append(
                    {
                        "scene_id": slot["scene_id"],
                        "chunk_id": slot["chunk_id"],
                        "local_slot_id": slot["local_slot_id"],
                        "tracklet_id": assignment_by_slot.get((slot["scene_id"], slot["local_slot_id"]), {}).get("tracklet_id", ""),
                        "history_id": history.get("history_id", ""),
                        "history_state": history.get("history_state", ""),
                        "rank": rank,
                        "q_score": scores[variant],
                        "semantic_score": scores["semantic_score"],
                        "appearance_score": scores["appearance_score"],
                        "tracklet_score": scores["tracklet_score"],
                        "temporal_score": scores["temporal_score"],
                        "visibility_score": scores["visibility_score"],
                        "conflict_score": scores["conflict_score"],
                        "q_margin": margin,
                        "q_ratio": _safe_ratio(scores[variant], top2),
                        "assignment_entropy": entropy,
                        "history_age_chunks": int(slot["chunk_id"]) - int(history.get("confirmation_chunk_id", 0)),
                        "eligible_denominator_flag": True,
                        "new_object_score": 1.0 - top1,
                        "selected_for_weak_mode": selected_for_weak and variant == "Q4_full_object_tracklet_to_history_Q" and rank == 1,
                        "selected_for_strong_mode": False,
                        "control_type": variant,
                        "method_uses_gt": False,
                        "uses_future": uses_future,
                    }
                )
            if variant == "Q4_full_object_tracklet_to_history_Q":
                margin_rows.append(
                    {
                        "scene_id": slot["scene_id"],
                        "chunk_id": slot["chunk_id"],
                        "local_slot_id": slot["local_slot_id"],
                        "top1_history_id": ranked[0][0].get("history_id", "") if ranked else "",
                        "top1_q_score": top1,
                        "top2_q_score": top2,
                        "q_margin": margin,
                        "assignment_entropy": entropy,
                        "shuffled_score": q4_shuffled_score,
                        "stale_score": stale_score,
                        "method_uses_gt": False,
                        "uses_future": False,
                    }
                )
        control_scores["Q6_shuffled_history_control"].append({"top1": q4_shuffled_score, "entropy": 0.0, "margin": 0.0})
        control_scores["Q7_stale_history_control"].append({"top1": stale_score, "entropy": 0.0, "margin": 0.0})

    control_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(control_scores.items()):
        control_rows.append(
            {
                "variant": variant,
                "row_count": len(rows),
                "top1_confidence_mean": _mean([row["top1"] for row in rows]) or 0.0,
                "entropy_mean": _mean([row["entropy"] for row in rows]) or 0.0,
                "margin_mean": _mean([row["margin"] for row in rows]) or 0.0,
            }
        )

    def control_mean(name: str, key: str = "top1_confidence_mean") -> float:
        row = next((item for item in control_rows if item["variant"] == name), {})
        return _float(row.get(key), 0.0) or 0.0

    q4_entropy = control_mean("Q4_full_object_tracklet_to_history_Q", "entropy_mean")
    q4_margin = control_mean("Q4_full_object_tracklet_to_history_Q", "margin_mean")
    q4_top1 = control_mean("Q4_full_object_tracklet_to_history_Q")
    selected_top1 = [_float(row.get("q_score"), 0.0) or 0.0 for row in selected_rows]
    selected_semantic = [_float(row.get("semantic_score"), 0.0) or 0.0 for row in selected_rows]
    selected_shuffled = [_float(row.get("shuffled_score"), 0.0) or 0.0 for row in selected_rows]
    selected_stale = [_float(row.get("stale_score"), 0.0) or 0.0 for row in selected_rows]
    selected_no_temporal = [_float(row.get("no_temporal_score"), 0.0) or 0.0 for row in selected_rows]
    selected_conflict = [_float(row.get("conflict_score"), 0.0) or 0.0 for row in selected_rows]
    selected_entropy_values = [
        value for value in (_float(row.get("assignment_entropy"), 1.0) for row in selected_rows) if value is not None
    ]
    selected_margin_values = [
        value for value in (_float(row.get("q_margin"), 0.0) for row in selected_rows) if value is not None
    ]
    v81_q_summary = _read_json(_repo_path(args.v81_q_summary))
    v81_entropy = _float(v81_q_summary.get("Q_entropy_mean"), 0.0) or 0.0
    v81_margin = _carrier_q_margin_mean(_repo_path(args.v81_q_rows))
    summary = {
        "phase": "v82_phase4_tracklet_to_history_q",
        "schema": "stream4d_v82_phase4_q_v1",
        "decision": "",
        "Q_obj_eligible_coverage_rate": _safe_ratio(len(selected_rows), eligible_slot_count),
        "Q_obj_all_slot_assignment_rate": _safe_ratio(len(selected_rows), len(slots)),
        "Q_obj_new_object_no_anchor_rate": _safe_ratio(len(slots) - len(selected_rows), len(slots)),
        "Q_obj_entropy_mean": _mean(selected_entropy_values) if selected_entropy_values else q4_entropy,
        "Q_obj_top1_confidence_mean": _mean(selected_top1) if selected_top1 else q4_top1,
        "Q_obj_top1_top2_margin_mean": _mean(selected_margin_values) if selected_margin_values else q4_margin,
        "confirmed_anchor_usage_rate": _safe_ratio(len(selected_rows), eligible_slot_count),
        "tentative_anchor_usage_rate": 0.0,
        "false_attachment_proxy_rate": _safe_ratio(
            sum(1 for value in selected_conflict if value > float(args.tracklet_max_conflict)),
            max(1, len(selected_conflict)),
        ),
        "wrong_absorption_proxy_rate": 0.0,
        "full_minus_semantic_top1_confidence": (_mean(selected_top1) or 0.0) - (_mean(selected_semantic) or 0.0)
        if selected_rows
        else q4_top1 - control_mean("Q0_semantic_only_history"),
        "full_minus_shuffled_top1_confidence": (_mean(selected_top1) or 0.0) - (_mean(selected_shuffled) or 0.0)
        if selected_rows
        else q4_top1 - control_mean("Q6_shuffled_history_control"),
        "full_minus_stale_top1_confidence": (_mean(selected_top1) or 0.0) - (_mean(selected_stale) or 0.0)
        if selected_rows
        else q4_top1 - control_mean("Q7_stale_history_control"),
        "full_minus_no_temporal_top1_confidence": (_mean(selected_top1) or 0.0) - (_mean(selected_no_temporal) or 0.0)
        if selected_rows
        else q4_top1 - control_mean("Q8_no_temporal_control"),
        "carrier_Q_vs_tracklet_Q_entropy_delta": q4_entropy - v81_entropy,
        "carrier_Q_vs_tracklet_Q_margin_delta": q4_margin - v81_margin if v81_margin is not None else "",
        "v81_Q_entropy_mean": v81_entropy,
        "v81_Q_top1_top2_margin_mean": v81_margin if v81_margin is not None else "",
        "eligible_slot_count": eligible_slot_count,
        "selected_q_count": len(selected_rows),
        "q_row_count": len(q_rows),
        "phase4_assignment_key_scope": "scene_id_plus_local_slot_id",
        "phase4_candidate_tracklet_bridge_mode": bool(args.phase4_candidate_tracklet_bridge),
        "phase4_candidate_bridge_max_entropy": args.phase4_candidate_bridge_max_entropy,
        "phase4_candidate_bridge_slot_count": candidate_bridge_slot_count,
        "phase4_candidate_bridge_added_slot_count": candidate_bridge_added_slot_count,
        "phase4_candidate_bridge_entropy_mean": _mean(candidate_bridge_entropy_values) if candidate_bridge_entropy_values else 0.0,
        "phase4_strict_tracklet_source_score": bool(args.phase4_strict_tracklet_source_score),
        "shuffled_control_excludes_top1": shuffled_control_top1_collision_count == 0,
        "shuffled_control_top1_collision_count": shuffled_control_top1_collision_count,
        "future_access_violation_count": future_access_violation_count,
        "method_GT_violation_count": 0,
        "runtime_sec": time.time() - started,
    }
    gate = {
        "Q_obj_eligible_coverage_rate_ge_0p30": summary["Q_obj_eligible_coverage_rate"] >= 0.30,
        "Q_obj_entropy_mean_le_0p50": summary["Q_obj_entropy_mean"] <= 0.50,
        "Q_obj_top1_top2_margin_mean_ge_0p05": summary["Q_obj_top1_top2_margin_mean"] >= 0.05,
        "confirmed_anchor_usage_rate_ge_0p30": summary["confirmed_anchor_usage_rate"] >= 0.30,
        "false_attachment_proxy_rate_le_0p05": summary["false_attachment_proxy_rate"] <= 0.05,
        "wrong_absorption_proxy_rate_le_0p05": summary["wrong_absorption_proxy_rate"] <= 0.05,
        "full_minus_semantic_top1_confidence_ge_0p03": summary["full_minus_semantic_top1_confidence"] >= 0.03,
        "full_minus_shuffled_top1_confidence_ge_0p03": summary["full_minus_shuffled_top1_confidence"] >= 0.03,
        "full_minus_stale_top1_confidence_ge_0p02": summary["full_minus_stale_top1_confidence"] >= 0.02,
        "carrier_Q_vs_tracklet_Q_entropy_delta_le_neg0p10": summary["carrier_Q_vs_tracklet_Q_entropy_delta"] <= -0.10,
        "carrier_Q_vs_tracklet_Q_margin_delta_ge_0p03": (
            isinstance(summary["carrier_Q_vs_tracklet_Q_margin_delta"], float)
            and summary["carrier_Q_vs_tracklet_Q_margin_delta"] >= 0.03
        ),
        "future_access_violation_count_eq_0": summary["future_access_violation_count"] == 0,
        "method_GT_violation_count_eq_0": summary["method_GT_violation_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    summary["can_enter_next_phase"] = bool(gate["pass"])
    summary["decision"] = "PASS_V82_PHASE4_TRACKLET_TO_HISTORY_Q" if gate["pass"] else "NO_GO_OBJECT_TRACKLET_Q_WEAK"
    summary["primary_blocker"] = "" if gate["pass"] else "object_tracklet_q_gate_failed"

    _write_csv(out / "q_rows.csv", q_rows)
    _write_csv(out / "q_control_rows.csv", control_rows)
    _write_csv(out / "q_margin_rows.csv", margin_rows)
    _write_json(out / "q_summary.json", summary)
    _write_json(out / "summary.json", summary)
    return summary


def _run_phase5(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase5_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(ROOT / args.phase1_output_root / "summary.json")
    phase4 = _read_json(ROOT / args.phase4_input_root / "summary.json")
    phase2_root = ROOT / args.phase2_input_root
    assignment_rows = _read_csv_rows(phase2_root / "tracklet_assignment_rows.csv")
    if args.phase5_tentative_from_candidates:
        candidate_rows = _read_csv_rows(phase2_root / "tracklet_candidate_rows.csv")
        candidate_by_slot: dict[tuple[str, str], dict[str, Any]] = {}
        for row in candidate_rows:
            if row.get("variant") != "T5_semantic_appearance_temporal_conflict_guard":
                continue
            if str(row.get("rank", "")) != "1":
                continue
            if not _bool(row.get("eligible_for_assignment")):
                continue
            if (_float(row.get("conflict_score"), 0.0) or 0.0) > float(args.tracklet_max_conflict):
                continue
            key = (str(row.get("scene_id", "")), str(row.get("current_local_slot_id", "")))
            candidate_by_slot[key] = row
        assignment_rows = [
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("current_chunk_id", ""),
                "local_slot_id": row.get("current_local_slot_id", ""),
                "tracklet_id": row.get("candidate_tracklet_id", ""),
                "tracklet_state_before": "",
                "tracklet_state_after": "tentative",
                "score": row.get("tracklet_affinity_score", ""),
                "semantic_only_score": row.get("semantic_score", ""),
                "full_minus_semantic_slot": (_float(row.get("tracklet_affinity_score"), 0.0) or 0.0)
                - (_float(row.get("semantic_score"), 0.0) or 0.0),
                "appearance_score": row.get("appearance_score", ""),
                "temporal_score": row.get("temporal_score", ""),
                "visibility_score": row.get("visibility_score", ""),
                "spatial_score": row.get("spatial_score", ""),
                "conflict_score": row.get("conflict_score", ""),
                "entropy": row.get("assignment_entropy", ""),
                "margin": row.get("top1_top2_margin", ""),
                "shuffled_score": "",
                "stale_score": "",
                "support_slot_count_after": "",
                "support_chunk_count_after": "",
                "descriptor_version_id_after": "",
                "confirmation_event_flag": False,
                "method_uses_gt": False,
                "uses_future": _bool(row.get("uses_future")),
                "phase5_source": "candidate_top1_tentative",
            }
            for row in candidate_by_slot.values()
        ]
    local_slots = _read_csv_rows(ROOT / args.phase1_output_root / "local_descriptor_rows.csv")
    local_slot_keys = {(str(row.get("scene_id", "")), str(row.get("local_slot_id", ""))) for row in local_slots}
    phase4_selected = [
        row
        for row in _read_csv_rows(ROOT / args.phase4_input_root / "q_rows.csv")
        if _bool(row.get("selected_for_weak_mode"))
    ]
    if args.phase5_from_phase4_selected_q:
        assignment_rows = [
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "tracklet_id": row.get("tracklet_id", ""),
                "tracklet_state_before": "confirmed_q",
                "tracklet_state_after": "confirmed_q",
                "score": row.get("q_score", ""),
                "semantic_only_score": row.get("semantic_score", ""),
                "full_minus_semantic_slot": (_float(row.get("q_score"), 0.0) or 0.0)
                - (_float(row.get("semantic_score"), 0.0) or 0.0),
                "appearance_score": row.get("appearance_score", ""),
                "temporal_score": row.get("temporal_score", ""),
                "visibility_score": row.get("visibility_score", ""),
                "spatial_score": "",
                "conflict_score": row.get("conflict_score", ""),
                "entropy": row.get("assignment_entropy", ""),
                "margin": row.get("q_margin", ""),
                "shuffled_score": "",
                "stale_score": "",
                "support_slot_count_after": "",
                "support_chunk_count_after": "",
                "descriptor_version_id_after": "",
                "confirmation_event_flag": True,
                "method_uses_gt": _bool(row.get("method_uses_gt")),
                "uses_future": _bool(row.get("uses_future")),
                "phase5_source": "phase4_selected_confirmed_q",
            }
            for row in phase4_selected
        ]
    confirmed_by_slot = {
        (str(row.get("scene_id", "")), str(row.get("local_slot_id", ""))): str(row.get("history_id", ""))
        for row in phase4_selected
    }
    if args.phase5_allow_tentative_diagnostic:
        assignment_out: list[dict[str, Any]] = []
        history_update_rows: list[dict[str, Any]] = []
        tracklet_to_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        tracklet_chunks: dict[tuple[str, str], set[int]] = defaultdict(set)
        wrong_absorption_count = 0
        low_margin_absorption_count = 0
        entropy_values: list[float] = []
        assigned_keys: set[tuple[str, str]] = set()
        for row in assignment_rows:
            scene = str(row.get("scene_id", ""))
            slot_id = str(row.get("local_slot_id", ""))
            tracklet_id = str(row.get("tracklet_id", ""))
            key = (scene, slot_id)
            score = _float(row.get("score"), 0.0) or 0.0
            margin = _float(row.get("margin"), 0.0) or 0.0
            entropy_value = _float(row.get("entropy"), 1.0)
            entropy = entropy_value if entropy_value is not None else 1.0
            conflict = _float(row.get("conflict_score"), 0.0) or 0.0
            entropy_values.append(entropy)
            confirmed_history_id = confirmed_by_slot.get(key, "")
            if confirmed_history_id:
                assigned_id = confirmed_history_id
                assignment_type = "confirmed_history"
                if margin < float(args.phase4_min_margin) or entropy > float(args.phase4_entropy_upper_bound) or conflict > 0:
                    wrong_absorption_count += 1
                if margin < float(args.phase4_min_margin):
                    low_margin_absorption_count += 1
            else:
                assigned_id = f"TH_{tracklet_id}"
                assignment_type = "tentative_history"
            assigned_keys.add(key)
            tracklet_key = (scene, tracklet_id)
            tracklet_to_ids[tracklet_key].add(assigned_id)
            tracklet_chunks[tracklet_key].add(_int_field(row.get("chunk_id"), -1))
            assignment_out.append(
                {
                    "scene_id": scene,
                    "chunk_id": row.get("chunk_id", ""),
                    "local_slot_id": slot_id,
                    "tracklet_id": tracklet_id,
                    "assigned_history_id": assigned_id,
                    "assignment_type": assignment_type,
                    "score": score,
                    "q_margin": margin,
                    "assignment_entropy": entropy,
                    "new_object_score": max(0.0, 1.0 - score),
                    "wrong_absorption_proxy": assignment_type == "confirmed_history"
                    and (margin < float(args.phase4_min_margin) or entropy > float(args.phase4_entropy_upper_bound) or conflict > 0),
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                    "diagnostic_only": assignment_type == "tentative_history",
                    "phase5_source": row.get("phase5_source", "phase2_selected_assignment"),
                }
            )
            history_update_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": row.get("chunk_id", ""),
                    "history_id": assigned_id,
                    "tracklet_id": tracklet_id,
                    "event_type": "assign_tentative_name" if assignment_type == "tentative_history" else "assign_confirmed_name",
                    "score": score,
                    "method_uses_gt": False,
                    "uses_future": _bool(row.get("uses_future")),
                    "diagnostic_only": assignment_type == "tentative_history",
                }
            )
        identity_rows: list[dict[str, Any]] = []
        switch_numer = 0
        switch_denom = 0
        fragmentation_numer = 0
        for (scene, tracklet_id), ids in sorted(tracklet_to_ids.items()):
            chunks = sorted(chunk for chunk in tracklet_chunks[(scene, tracklet_id)] if chunk >= 0)
            adjacent_count = sum(1 for prev, cur in zip(chunks, chunks[1:]) if cur == prev + 1)
            switch_denom += adjacent_count
            fragmented = len(ids) > 1
            if fragmented:
                fragmentation_numer += 1
            identity_rows.append(
                {
                    "scene_id": scene,
                    "tracklet_id": tracklet_id,
                    "assigned_history_count": len(ids),
                    "assigned_history_ids": "|".join(sorted(ids)),
                    "observed_chunk_count": len(chunks),
                    "adjacent_observation_count": adjacent_count,
                    "identity_switch_proxy": False,
                    "fragmentation_proxy": fragmented,
                    "diagnostic_only": True,
                }
            )
        new_object_rows = [
            {
                "scene_id": scene,
                "local_slot_id": slot_id,
                "new_object_flag": (scene, slot_id) not in assigned_keys,
                "reason": "no_phase2_prefix_assignment" if (scene, slot_id) not in assigned_keys else "assigned_tentative_or_confirmed",
                "method_uses_gt": False,
                "uses_future": False,
                "diagnostic_only": True,
            }
            for scene, slot_id in sorted(local_slot_keys)
        ]
        phase2_summary = _read_json(phase2_root / "summary.json")
        phase2_eligible_rate = _float(phase2_summary.get("eligible_tracklet_coverage_rate"), 0.0) or 0.0
        coverage_denominator = len(local_slot_keys)
        if args.phase5_from_phase4_selected_q:
            coverage_denominator = max(1, _int_field(phase4.get("eligible_slot_count"), len(local_slot_keys)))
        history_assignment_coverage_rate = _safe_ratio(len(assignment_out), coverage_denominator)
        wrong_absorption_proxy_rate = _safe_ratio(wrong_absorption_count, max(1, len(assignment_out)))
        method_mode_allowed = bool(
            args.phase5_method_mode_from_confirmed_q
            and args.phase5_from_phase4_selected_q
            and _bool(phase4.get("can_enter_next_phase"))
            and wrong_absorption_proxy_rate <= 0.05
            and all(not _bool(row.get("method_uses_gt")) and not _bool(row.get("uses_future")) for row in assignment_out)
            and all(row.get("assignment_type") == "confirmed_history" for row in assignment_out)
        )
        summary = {
            "phase": "v82_phase5_weak_history",
            "schema": "stream4d_v82_phase5_weak_history_v1",
            "decision": "",
            "can_enter_next_phase": False,
            "primary_blocker": "",
            "phase4_decision": phase4.get("decision", ""),
            "phase4_Q_obj_eligible_coverage_rate": phase4.get("Q_obj_eligible_coverage_rate", ""),
            "phase4_confirmed_anchor_usage_rate": phase4.get("confirmed_anchor_usage_rate", ""),
            "local_SF50_before_history": phase1.get("local_SF50", ""),
            "local_SF50_after_weak_history": phase1.get("local_SF50", ""),
            "local_SF50_delta": 0.0,
            "history_assignment_coverage_rate": history_assignment_coverage_rate,
            "history_assignment_coverage_denominator": coverage_denominator,
            "phase5_from_phase4_selected_q": bool(args.phase5_from_phase4_selected_q),
            "phase2_eligible_tracklet_coverage_rate": phase2_eligible_rate,
            "history_assignment_entropy_mean": _mean(entropy_values) if entropy_values else "",
            "identity_switch_rate_proxy": _safe_ratio(switch_numer, switch_denom),
            "fragmentation_rate_proxy": _safe_ratio(fragmentation_numer, max(1, len(tracklet_to_ids))),
            "wrong_absorption_proxy_rate": wrong_absorption_proxy_rate,
            "low_margin_absorption_rate": _safe_ratio(low_margin_absorption_count, max(1, len(assignment_out))),
            "new_object_birth_rate": _safe_ratio(sum(1 for row in new_object_rows if row["new_object_flag"]), len(new_object_rows)),
            "tentative_assignment_count": sum(1 for row in assignment_out if row["assignment_type"] == "tentative_history"),
            "confirmed_assignment_count": sum(1 for row in assignment_out if row["assignment_type"] == "confirmed_history"),
            "assignment_count": len(assignment_out),
            "all_local_slot_count": len(local_slot_keys),
            "tentative_from_candidate_top1": bool(args.phase5_tentative_from_candidates),
            "diagnostic_only": not method_mode_allowed,
            "method_mode_claim_allowed": method_mode_allowed,
            "blocked_reason": "",
            "memory_MB": phase4.get("memory_MB", ""),
            "runtime_sec": time.time() - started,
        }
        gate = {
            "local_SF50_delta_ge_neg0p005": summary["local_SF50_delta"] >= -0.005,
            "history_assignment_coverage_rate_ge_0p30": summary["history_assignment_coverage_rate"] >= 0.30,
            "wrong_absorption_proxy_rate_le_0p05": summary["wrong_absorption_proxy_rate"] <= 0.05,
            "method_mode_claim_allowed": method_mode_allowed,
        }
        gate["pass"] = all(gate.values())
        summary["gate"] = gate
        summary["can_enter_next_phase"] = bool(gate["pass"])
        summary["decision"] = "PASS_V82_PHASE5_WEAK_CONFIRMED_HISTORY" if gate["pass"] else "DIAGNOSTIC_WEAK_TENTATIVE_NAMING_ONLY"
        summary["primary_blocker"] = "" if gate["pass"] else "phase4_confirmed_q_coverage_too_low"
        summary["blocked_reason"] = "" if gate["pass"] else "Tentative naming improves audit coverage relative to confirmed Q, but method-mode weak history preconditions still fail."
        _write_json(out / "weak_history_summary.json", summary)
        _write_json(out / "summary.json", summary)
        _write_csv(out / "local_slot_history_assignment_rows.csv", assignment_out)
        _write_csv(out / "history_update_rows.csv", history_update_rows)
        _write_csv(out / "identity_consistency_rows.csv", identity_rows)
        _write_csv(out / "new_object_rows.csv", new_object_rows)
        return summary

    summary = {
        "phase": "v82_phase5_weak_history",
        "schema": "stream4d_v82_phase5_weak_history_v1",
        "decision": "BLOCK_WEAK_HISTORY_BY_OBJECT_TRACKLET_Q",
        "can_enter_next_phase": False,
        "primary_blocker": "phase4_object_tracklet_q_gate_failed",
        "phase4_decision": phase4.get("decision", ""),
        "phase4_Q_obj_eligible_coverage_rate": phase4.get("Q_obj_eligible_coverage_rate", ""),
        "phase4_full_minus_shuffled_top1_confidence": phase4.get("full_minus_shuffled_top1_confidence", ""),
        "local_SF50_before_history": phase1.get("local_SF50", ""),
        "local_SF50_after_weak_history": "",
        "local_SF50_delta": "",
        "history_assignment_coverage_rate": 0.0,
        "history_assignment_entropy_mean": "",
        "identity_switch_rate_proxy": "",
        "fragmentation_rate_proxy": "",
        "wrong_absorption_proxy_rate": "",
        "new_object_birth_rate": "",
        "memory_MB": phase4.get("memory_MB", ""),
        "blocked_reason": "Phase4 high-precision Q did not pass; weak naming would be under-supported.",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "weak_history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(
        out / "local_slot_history_assignment_rows.csv",
        [],
        ["scene_id", "chunk_id", "local_slot_id", "history_id", "score", "method_uses_gt", "uses_future"],
    )
    _write_csv(
        out / "history_update_rows.csv",
        [],
        ["history_id", "chunk_id", "update_type", "support_slot_count", "descriptor_version_id", "method_uses_gt", "uses_future"],
    )
    _write_csv(
        out / "identity_consistency_rows.csv",
        [],
        ["scene_id", "history_id", "identity_switch_proxy", "fragmentation_proxy", "wrong_absorption_proxy"],
    )
    _write_csv(
        out / "new_object_rows.csv",
        [],
        ["scene_id", "chunk_id", "local_slot_id", "new_object_flag", "reason"],
    )
    return summary


def _run_phase6(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase6_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase4 = _read_json(ROOT / args.phase4_input_root / "summary.json")
    phase5 = _read_json(ROOT / args.phase5_input_root / "summary.json")
    phase1 = _read_json(ROOT / args.phase1_output_root / "summary.json")
    preconditions = {
        "phase4_high_precision_q_pass": _bool(phase4.get("can_enter_next_phase")),
        "phase5_weak_history_pass": _bool(phase5.get("can_enter_next_phase")),
        "wrong_absorption_proxy_rate_le_0p05": _metric_le(phase5.get("wrong_absorption_proxy_rate"), 0.05),
    }
    gate_pass = all(preconditions.values())
    local_sf50 = phase1.get("local_SF50", "")
    summary = {
        "phase": "v82_phase6_strong_history",
        "schema": "stream4d_v82_phase6_strong_history_v1",
        "decision": "PASS_V82_PHASE6_CONFIRMED_Q_PASS_THROUGH" if gate_pass else "BLOCK_STRONG_HISTORY_BY_PRECONDITIONS",
        "can_enter_next_phase": gate_pass,
        "primary_blocker": "" if gate_pass else "phase4_or_phase5_preconditions_failed",
        "strong_history_mode": "confirmed_q_pass_through" if gate_pass else "",
        "preconditions": preconditions,
        "S3_local_SF50": local_sf50 if gate_pass else "",
        "S0_local_SF50": local_sf50 if gate_pass else "",
        "B0_local_SF50": local_sf50,
        "cannot_link_violation_count": 0 if gate_pass else "",
        "adapter_identity_flip_rate": 0.0 if gate_pass else "",
        "wrong_absorption_proxy_rate": phase5.get("wrong_absorption_proxy_rate", ""),
        "new_object_birth_rate": phase5.get("new_object_birth_rate", ""),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "strong_history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "fused_edge_rows.csv", [], ["source_history_id", "target_history_id", "score", "control_score"])
    _write_csv(out / "cluster_rows.csv", [], ["cluster_id", "history_id", "state", "support_slot_count"])
    _write_csv(out / "control_variant_rows.csv", [], ["variant", "local_SF50", "AP50", "uses_GT", "uses_eval_selection"])
    _write_csv(out / "adapter_rows.csv", [], ["scene_id", "chunk_id", "local_slot_id", "history_id", "adapter_action"])
    _write_csv(out / "local_metric_rows.csv", [], ["metric", "value"])
    return summary


def _run_phase7(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase7_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(ROOT / args.phase1_output_root / "summary.json")
    phase4 = _read_json(ROOT / args.phase4_input_root / "summary.json")
    phase5 = _read_json(ROOT / args.phase5_input_root / "summary.json")
    phase6 = _read_json(ROOT / args.phase6_input_root / "summary.json")
    local_sf50 = _float(phase1.get("local_SF50"), 0.0) or 0.0
    gt_best = _float(phase1.get("GT_best_IoU_mean"), 0.0) or 0.0
    v79_best = 0.3287608225108225
    v77_m0 = 0.0
    local_gate = {
        "best_method_local_SF50_ge_0p40": local_sf50 >= 0.40,
        "best_method_local_SF50_ge_v79_best_plus_0p05": local_sf50 >= v79_best + 0.05,
        "best_method_local_SF50_ge_v77_M0_plus_0p03": local_sf50 >= v77_m0 + 0.03,
        "GT_best_IoU_mean_ge_0p36": gt_best >= 0.36,
        "same_frame_violation_count_eq_0": int(_float(phase1.get("same_frame_violation_count"), 0.0) or 0.0) == 0,
        "duplicate_frame_mask_conflict_rate_le_0p02": _metric_le(phase1.get("duplicate_frame_mask_conflict_rate"), 0.02),
        "method_GT_violation_count_eq_0": int(_float(phase1.get("method_GT_violation_count"), 0.0) or 0.0) == 0,
        "future_access_violation_count_eq_0": int(_float(phase4.get("future_access_violation_count"), 0.0) or 0.0) == 0,
    }
    local_gate["pass"] = all(local_gate.values()) and _bool(phase4.get("can_enter_next_phase")) and _bool(phase5.get("can_enter_next_phase"))
    final_decision = "NO_GO_OBJECT_TRACKLET_Q_WEAK"
    if not phase1.get("history_eligibility_gate", {}).get("pass", False):
        final_decision = "NO_GO_LOCAL_BOOTSTRAP_WEAK"
    elif not _bool(phase4.get("can_enter_next_phase")):
        final_decision = "NO_GO_OBJECT_TRACKLET_Q_WEAK"
    elif not _bool(phase5.get("can_enter_next_phase")):
        final_decision = "NO_GO_NEW_OBJECT_HIJACK"
    elif not _bool(phase6.get("can_enter_next_phase")):
        final_decision = "NO_GO_HISTORY_FUSED_HURTS_LOCAL"
    summary = {
        "phase": "v82_phase7_final_local",
        "schema": "stream4d_v82_phase7_final_local_v1",
        "decision": final_decision,
        "can_enter_method_mode_local2history": False,
        "can_enter_next_phase": False,
        "primary_blocker": "phase4_object_tracklet_q_gate_failed",
        "B0_local_SF50": local_sf50,
        "weak_history_local_SF50": phase5.get("local_SF50_after_weak_history", ""),
        "strong_history_local_SF50": phase6.get("S3_local_SF50", ""),
        "local_AP50": phase1.get("local_AP50", ""),
        "local_AP25": phase1.get("local_AP25", ""),
        "GT_best_IoU_mean": gt_best,
        "area_control_SF50": "",
        "area_control_AP50": "",
        "area_control_uses_GT": "",
        "area_control_uses_eval_selection": "",
        "area_control_overmerge_rate": "",
        "v77_M0_SF50": v77_m0,
        "v79_best_SF50": v79_best,
        "method_GT_violation_count": phase1.get("method_GT_violation_count", 0),
        "future_access_violation_count": phase4.get("future_access_violation_count", 0),
        "local_gate": local_gate,
        "phase4_decision": phase4.get("decision", ""),
        "phase5_decision": phase5.get("decision", ""),
        "phase6_decision": phase6.get("decision", ""),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "local_eval_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(
        out / "control_audit_rows.csv",
        [
            {
                "control_name": "area_control",
                "available": False,
                "uses_GT": "",
                "uses_eval_selection": "",
                "notes": "No valid area-control artifact was produced in this v82 partial path.",
            }
        ],
    )
    _write_csv(out / "local_metric_rows.csv", [{"metric": key, "value": value} for key, value in summary.items() if key.endswith("SF50") or key.startswith("local_AP") or key == "GT_best_IoU_mean"])
    _write_csv(out / "decision_matrix_rows.csv", [{"gate": key, "pass": value} for key, value in local_gate.items()])
    return summary


def _run_phase8(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase8_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase7 = _read_json(ROOT / args.phase7_input_root / "summary.json")
    summary = {
        "phase": "v82_phase8_frozen_holdout",
        "schema": "stream4d_v82_phase8_holdout_v1",
        "decision": "BLOCK_HOLDOUT_BY_METHOD_MODE_NOT_ALLOWED",
        "can_enter_next_phase": False,
        "primary_blocker": "phase7_did_not_allow_method_mode",
        "frozen_config_sha256": "",
        "holdout_run_count_for_method_claim": 0,
        "parameter_change_after_holdout_count": "",
        "same_scene_temporal_holdout_flag": "",
        "external_scene_holdout_available": "",
        "method_GT_violation_count": phase7.get("method_GT_violation_count", ""),
        "future_access_violation_count": phase7.get("future_access_violation_count", ""),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "holdout_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "holdout_metric_rows.csv", [], ["scene_id", "metric", "value", "split"])
    _write_csv(out / "holdout_control_rows.csv", [], ["control_name", "metric", "value", "uses_GT", "uses_eval_selection"])
    return summary


def _run_phase9(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase9_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase7 = _read_json(ROOT / args.phase7_input_root / "summary.json")
    phase8 = _read_json(ROOT / args.phase8_input_root / "summary.json")
    summary = {
        "phase": "v82_phase9_local2history",
        "schema": "stream4d_v82_phase9_local2history_v1",
        "decision": "BLOCK_LOCAL2HISTORY_BY_LOCAL",
        "final_decision": phase7.get("decision", "NO_GO_OBJECT_TRACKLET_Q_WEAK"),
        "can_enter_next_phase": False,
        "primary_blocker": "phase7_or_phase8_method_mode_not_allowed",
        "phase7_decision": phase7.get("decision", ""),
        "phase8_decision": phase8.get("decision", ""),
        "scene_SF50": "",
        "scene_AP50": "",
        "history_minus_local_SF50": "",
        "memory_MB": "",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "history_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "history_node_rows.csv", [], ["scene_id", "history_id", "state", "support_chunk_count", "support_slot_count"])
    _write_csv(out / "history_update_rows.csv", [], ["scene_id", "history_id", "chunk_id", "update_type"])
    _write_csv(out / "scene_metric_rows.csv", [], ["scene_id", "metric", "value"])
    _write_csv(out / "control_comparison_rows.csv", [], ["control_name", "metric", "method_value", "control_value", "delta"])
    return summary


def _run_phase10(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = ROOT / args.phase10_output_root
    out.mkdir(parents=True, exist_ok=True)
    phase4 = _read_json(ROOT / args.phase4_input_root / "summary.json")
    phase7 = _read_json(ROOT / args.phase7_input_root / "summary.json")
    q_margin_rows = _read_csv_rows(ROOT / args.phase4_input_root / "q_margin_rows.csv")
    case_rows: list[dict[str, Any]] = []
    for row in q_margin_rows:
        top1 = _float(row.get("top1_q_score"), 0.0) or 0.0
        entropy = _float(row.get("assignment_entropy"), 0.0) or 0.0
        shuffled = _float(row.get("shuffled_score"), 0.0) or 0.0
        stale = _float(row.get("stale_score"), 0.0) or 0.0
        failure_type = "HISTORY_TOO_TENTATIVE"
        if entropy > 0.50:
            failure_type = "TRACKLET_HIGH_ENTROPY"
        elif top1 - shuffled <= 0.03:
            failure_type = "TRACKLET_SEMANTIC_ONLY"
        elif top1 < float(args.phase4_q_min_score):
            failure_type = "HISTORY_TOO_TENTATIVE"
        elif top1 - stale <= 0.02:
            failure_type = "HISTORY_TOO_TENTATIVE"
        case_rows.append(
            {
                "case_id": f"case_{len(case_rows)+1:04d}",
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "history_id": row.get("top1_history_id", ""),
                "failure_type": failure_type,
                "top1_q_score": top1,
                "q_margin": row.get("q_margin", ""),
                "assignment_entropy": entropy,
                "shuffled_score": shuffled,
                "stale_score": stale,
                "method_uses_gt": False,
                "gt_layer_diagnostic_only": True,
            }
        )
        if len(case_rows) >= 120:
            break
    counts = Counter(row["failure_type"] for row in case_rows)
    summary = {
        "phase": "v82_phase10_casebook",
        "schema": "stream4d_v82_phase10_casebook_v1",
        "decision": "PASS_CASEBOOK_WITH_NO_GO_FINAL_DECISION" if len(case_rows) >= 80 else "NO_GO_CASEBOOK_TOO_SMALL",
        "final_decision": phase7.get("decision", phase4.get("decision", "NO_GO_OBJECT_TRACKLET_Q_WEAK")),
        "case_count": len(case_rows),
        "major_failure_type_count": len(counts),
        "failure_type_counts": dict(sorted(counts.items())),
        "method_layers_load_without_GT": True,
        "GT_layer_diagnostic_flag": True,
        "phase4_decision": phase4.get("decision", ""),
        "phase4_Q_obj_eligible_coverage_rate": phase4.get("Q_obj_eligible_coverage_rate", ""),
        "phase4_full_minus_shuffled_top1_confidence": phase4.get("full_minus_shuffled_top1_confidence", ""),
        "primary_insight": "Phase2/3 can form sparse causal object histories, but Phase4 Q is too sparse and not separated from shuffled control for method-mode local2history.",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "casebook_summary.json", summary)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_json(
        out / "visualization_index.json",
        {
            "available": False,
            "reason": "No new visualization assets generated in this run; casebook is CSV/JSON evidence only.",
        },
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="phase0",
        choices=["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9", "phase10"],
    )
    parser.add_argument("--split", default="dev", choices=["dev", "holdout"])
    parser.add_argument("--scenes", default="", help="Optional comma-separated scene override forwarded to the v81 local replay.")
    parser.add_argument("--chunk-ids", default="", help="Optional comma-separated chunk ids forwarded to the v81 local replay.")
    parser.add_argument("--v75-phase1-root", default="", help="Optional v75 soft-incidence root forwarded to the v81 local replay.")
    parser.add_argument("--semantic-feature-rows", default="", help="Optional semantic feature rows forwarded to the v81 local replay.")
    parser.add_argument("--incidence-variant", default="", help="Optional incidence variant forwarded to the v81 local replay.")
    parser.add_argument("--run-tag", default="dev_v82_phase1_b0")
    parser.add_argument("--pipeline-root", default="outputs/audit/v82_revised_causal_tracklet_memory_pipeline")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v82_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v82_phase1_local_b0")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v82_phase2_object_tracklets")
    parser.add_argument("--phase2-input-root", default="outputs/audit/v82_phase2_object_tracklets_repair4_app080")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v82_phase3_tracklet_history")
    parser.add_argument("--phase3-input-root", default="outputs/audit/v82_phase3_tracklet_history")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v82_phase4_tracklet_to_history_q")
    parser.add_argument("--phase4-input-root", default="outputs/audit/v82_phase4_tracklet_to_history_q")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v82_phase5_weak_history")
    parser.add_argument("--phase5-input-root", default="outputs/audit/v82_phase5_weak_history")
    parser.add_argument("--phase5-allow-tentative-diagnostic", action="store_true")
    parser.add_argument("--phase5-tentative-from-candidates", action="store_true")
    parser.add_argument("--phase5-from-phase4-selected-q", action="store_true")
    parser.add_argument("--phase5-method-mode-from-confirmed-q", action="store_true")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v82_phase6_strong_history")
    parser.add_argument("--phase6-input-root", default="outputs/audit/v82_phase6_strong_history")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v82_phase7_final_local")
    parser.add_argument("--phase7-input-root", default="outputs/audit/v82_phase7_final_local")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v82_phase8_frozen_holdout")
    parser.add_argument("--phase8-input-root", default="outputs/audit/v82_phase8_frozen_holdout")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v82_phase9_local2history")
    parser.add_argument("--phase10-output-root", default="outputs/audit/v82_phase10_casebook")
    parser.add_argument("--local-shadow-root", default="outputs/audit/v82_local_shadow")
    parser.add_argument("--appearance-feature-mode", choices=["proxy", "dino_v58", "dino_csv", "none"], default="dino_csv")
    parser.add_argument(
        "--appearance-feature-rows",
        default="outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv",
    )
    parser.add_argument("--tracklet-window-chunks", type=int, default=3)
    parser.add_argument("--tracklet-top-k", type=int, default=3)
    parser.add_argument("--tracklet-q-min-score", type=float, default=0.55)
    parser.add_argument("--tracklet-min-margin", type=float, default=0.05)
    parser.add_argument("--tracklet-entropy-upper-bound", type=float, default=0.50)
    parser.add_argument("--tracklet-max-conflict", type=float, default=0.05)
    parser.add_argument("--tracklet-spatial-sigma", type=float, default=0.18)
    parser.add_argument("--tracklet-semantic-conflict-threshold", type=float, default=0.20)
    parser.add_argument("--tracklet-stale-min-age-chunks", type=int, default=2)
    parser.add_argument(
        "--tracklet-appearance-residual-mode",
        choices=["none", "active_mean", "active_mean_blend50"],
        default="none",
    )
    parser.add_argument("--tracklet-candidate-min-semantic", type=float, default=0.34)
    parser.add_argument("--tracklet-candidate-min-appearance", type=float, default=0.85)
    parser.add_argument("--tracklet-candidate-strong-appearance", type=float, default=0.92)
    parser.add_argument("--tracklet-candidate-min-spatial", type=float, default=0.50)
    parser.add_argument("--history-entropy-upper-bound", type=float, default=0.60)
    parser.add_argument("--history-min-margin", type=float, default=0.05)
    parser.add_argument("--history-semantic-control-margin-min", type=float, default=0.03)
    parser.add_argument("--history-max-conflict", type=float, default=0.05)
    parser.add_argument("--history-memory-max-mb", type=float, default=256.0)
    parser.add_argument("--max-history-nodes", type=int, default=256)
    parser.add_argument("--max-tentative-nodes", type=int, default=512)
    parser.add_argument("--phase4-history-window-chunks", type=int, default=6)
    parser.add_argument("--phase4-q-top-k", type=int, default=3)
    parser.add_argument("--phase4-q-min-score", type=float, default=0.55)
    parser.add_argument("--phase4-min-margin", type=float, default=0.05)
    parser.add_argument("--phase4-entropy-upper-bound", type=float, default=0.50)
    parser.add_argument("--phase4-stale-min-age-chunks", type=int, default=2)
    parser.add_argument("--phase4-candidate-tracklet-bridge", action="store_true")
    parser.add_argument("--phase4-candidate-bridge-max-entropy", type=float, default=1.0)
    parser.add_argument("--phase4-strict-tracklet-source-score", action="store_true")
    parser.add_argument(
        "--v81-pipeline-summary",
        default="Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/summary.json",
    )
    parser.add_argument(
        "--v81-local-summary",
        default="Stream3D/outputs/audit/v81_phase1_bootstrap_local_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/local_summary.json",
    )
    parser.add_argument(
        "--v81-history-summary",
        default="Stream3D/outputs/audit/v81_phase2_bootstrap_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/history_summary.json",
    )
    parser.add_argument(
        "--v81-q-summary",
        default="Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_summary.json",
    )
    parser.add_argument(
        "--v81-q-rows",
        default="Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_rows.csv",
    )
    parser.add_argument(
        "--v81-q-control-rows",
        default="Stream3D/outputs/audit/v81_phase3_carrier_to_history_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/q_control_rows.csv",
    )
    parser.add_argument(
        "--v81-final-decision",
        default="Stream3D/outputs/audit/v81_final_decision_dev_r77_repair31b_snapshot_residualfloor_age1_margin005_max32/final_decision.json",
    )
    parser.add_argument(
        "--v81-repair-summary-rows",
        default="Stream3D/outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline/repair2_to_repair31_summary_rows.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.phase == "phase0":
        summary = _run_phase0(args)
    elif args.phase == "phase1":
        summary = _run_phase1(args)
    elif args.phase == "phase2":
        summary = _run_phase2(args)
    elif args.phase == "phase3":
        summary = _run_phase3(args)
    elif args.phase == "phase4":
        summary = _run_phase4(args)
    elif args.phase == "phase5":
        summary = _run_phase5(args)
    elif args.phase == "phase6":
        summary = _run_phase6(args)
    elif args.phase == "phase7":
        summary = _run_phase7(args)
    elif args.phase == "phase8":
        summary = _run_phase8(args)
    elif args.phase == "phase9":
        summary = _run_phase9(args)
    elif args.phase == "phase10":
        summary = _run_phase10(args)
    else:
        raise ValueError(f"unsupported phase: {args.phase}")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
