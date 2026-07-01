#!/usr/bin/env python3
"""Run Stream4D v81 History-Anchored CMAP-AF-L2H audit.

This runner intentionally reuses the v80 local-only CMAP-AF implementation for
bootstrap evidence, then adds v81-specific causal history boundaries, history
node construction, and sparse carrier-to-history Q controls.  If Q is not
stronger than shuffled / stale / semantic-only controls, later history-fused
method phases are written as blocked artifacts instead of fabricating outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_v80_cmap_af_l2h_pipeline as v80


ROOT = v80.ROOT
REPO = v80.REPO
PHASE_ORDER = [
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "final",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    v80._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    v80._write_csv(path, rows, fields)


def _read_json(path: Path) -> dict[str, Any]:
    return v80._read_json(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    return v80._read_csv_rows(path)


def _bool(value: Any) -> bool:
    return v80._bool(value)


def _float(value: Any, default: float = 0.0) -> float:
    return v80._float(value, default)


def _int(value: Any, default: int = 0) -> int:
    return v80._int(value, default)


def _mean(values: list[float]) -> float | None:
    return v80._mean(values)


def _percentile(values: list[float], pct: float) -> float | None:
    return v80._percentile(values, pct)


def _safe_ratio(num: float, den: float) -> float:
    return v80._safe_ratio(num, den)


def _rel(path: Path) -> str:
    return v80._rel(path)


def _copy_namespace(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_feature_vector(raw: Any) -> np.ndarray | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        return None
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / np.float32(norm)


def _vector_hash(vec: np.ndarray | None) -> str:
    if vec is None or vec.size == 0:
        return ""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return hashlib.blake2b(arr.tobytes(), digest_size=12).hexdigest()


def _appearance_cosine(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    lvec = np.asarray(left, dtype=np.float32).reshape(-1)
    rvec = np.asarray(right, dtype=np.float32).reshape(-1)
    if lvec.size == 0 or lvec.shape != rvec.shape:
        return 0.0
    score = float(np.dot(lvec, rvec))
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _load_appearance_feature_index(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _rooted(args.appearance_feature_rows)
    meta: dict[str, Any] = {
        "appearance_feature_mode": args.appearance_feature_mode,
        "appearance_feature_rows": _rel(path),
        "path_exists": path.exists(),
        "rows_read": 0,
        "rows_kept": 0,
        "rows_skipped_gt_prediction": 0,
        "rows_skipped_unavailable": 0,
        "rows_skipped_parse": 0,
        "feature_dim_counts": {},
    }
    if args.appearance_feature_mode not in {"dino_v58", "dino_csv"} or not path.exists():
        return {}, meta
    index: dict[str, np.ndarray] = {}
    dim_counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            meta["rows_read"] += 1
            if _bool(row.get("uses_gt_for_prediction")):
                meta["rows_skipped_gt_prediction"] += 1
                continue
            if not _bool(row.get("feature_available")):
                meta["rows_skipped_unavailable"] += 1
                continue
            vec = _parse_feature_vector(row.get("feature_json"))
            if vec is None:
                meta["rows_skipped_parse"] += 1
                continue
            obs = str(row.get("mask_observation_id") or "")
            if not obs:
                meta["rows_skipped_parse"] += 1
                continue
            index[obs] = vec
            dim_counts[str(vec.size)] += 1
            meta["rows_kept"] += 1
    meta["feature_dim_counts"] = dict(dim_counts)
    return index, meta


def _attach_carrier_appearance_profiles(
    clusters: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    feature_index, meta = _load_appearance_feature_index(args)
    carrier_total = 0
    carrier_with_feature = 0
    obs_total = 0
    obs_with_feature = 0
    weighted_total = 0.0
    weighted_hit = 0.0
    for _key, graph in sorted(clusters.items()):
        data = graph["data"]
        profiles: dict[str, dict[str, Any]] = {}
        for carrier in graph["carriers"]:
            carrier_total += 1
            rows = data["carrier_obs"].get(carrier, [])
            weighted_vec: np.ndarray | None = None
            support_count = 0
            support_weight = 0.0
            total_weight = 0.0
            for row in rows:
                obs_total += 1
                weight = max(0.0, float(row.get("weight", 0.0)))
                total_weight += weight
                weighted_total += weight
                vec = feature_index.get(str(row.get("obs") or ""))
                if vec is None:
                    continue
                obs_with_feature += 1
                weighted_hit += weight
                support_count += 1
                support_weight += weight
                if weighted_vec is None:
                    weighted_vec = np.zeros_like(vec, dtype=np.float32)
                if weighted_vec.shape == vec.shape:
                    weighted_vec += np.float32(weight) * vec
            if weighted_vec is not None:
                norm = float(np.linalg.norm(weighted_vec))
                if norm > 1e-12:
                    weighted_vec = weighted_vec / np.float32(norm)
                    carrier_with_feature += 1
                else:
                    weighted_vec = None
            profiles[carrier] = {
                "appearance_vector": weighted_vec,
                "appearance_feature_dim": int(weighted_vec.size) if weighted_vec is not None else 0,
                "appearance_feature_hash": _vector_hash(weighted_vec),
                "appearance_feature_support_count": support_count,
                "appearance_feature_support_weight": support_weight,
                "appearance_feature_total_weight": total_weight,
                "appearance_feature_coverage_rate": _safe_ratio(support_weight, max(1e-12, total_weight)),
                "appearance_feature_available": weighted_vec is not None,
            }
        data["carrier_appearance_profiles"] = profiles
    meta.update(
        {
            "carrier_total": carrier_total,
            "carrier_with_appearance": carrier_with_feature,
            "carrier_appearance_coverage_rate": _safe_ratio(carrier_with_feature, max(1, carrier_total)),
            "incidence_observation_total": obs_total,
            "incidence_observation_with_appearance": obs_with_feature,
            "incidence_observation_coverage_rate": _safe_ratio(obs_with_feature, max(1, obs_total)),
            "weighted_incidence_appearance_coverage_rate": _safe_ratio(weighted_hit, max(1e-12, weighted_total)),
        }
    )
    audit_path = ROOT / args.pipeline_root / "appearance_feature_audit.json"
    _write_json(audit_path, meta)
    return meta


def _sanitize_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "appearance_vector"} for row in rows]


def _copy_history_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    vec = out.get("appearance_vector")
    if vec is not None:
        out["appearance_vector"] = np.asarray(vec, dtype=np.float32).reshape(-1).copy()
    return out


def _phase_common(
    *,
    phase: str,
    schema: str,
    decision: str,
    method_uses_gt: bool = False,
    method_uses_future: bool = False,
    carrier_id_scope: str = "global_full_sequence",
    history_anchor_type: str = "none",
    history_method_mode_allowed: bool = False,
    can_enter_next_phase: bool = False,
    can_enter_local2history: bool = False,
    diagnostic_only_rows_present: bool = True,
    forbidden_for_method_table_rows_present: bool = True,
    primary_blocker: str = "",
    secondary_blocker: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "phase": phase,
        "schema": schema,
        "decision": decision,
        "method_uses_gt_anywhere": method_uses_gt,
        "method_prediction_uses_future_anywhere": method_uses_future,
        "history_anchor_type": history_anchor_type,
        "history_method_mode_allowed": history_method_mode_allowed,
        "carrier_id_scope": carrier_id_scope,
        "can_enter_next_phase": can_enter_next_phase,
        "can_enter_local2history": can_enter_local2history,
        "diagnostic_only_rows_present": diagnostic_only_rows_present,
        "forbidden_for_method_table_rows_present": forbidden_for_method_table_rows_present,
        "primary_blocker": primary_blocker,
        "secondary_blocker": secondary_blocker,
    }
    if extra:
        payload.update(extra)
    return payload


def _local_shadow_args(args: argparse.Namespace) -> argparse.Namespace:
    root = Path(args.local_shadow_root)
    tag = str(args.run_tag)
    return _copy_namespace(
        args,
        pipeline_root=(root / f"pipeline_{tag}").as_posix(),
        phase0_output_root=(root / f"phase0_{tag}").as_posix(),
        phase1_output_root=(root / f"phase1_features_{tag}").as_posix(),
        phase2_output_root=(root / f"phase1_signed_{tag}").as_posix(),
        phase3_output_root=(root / f"phase1_semantic_diag_{tag}").as_posix(),
        phase4_output_root=(root / f"phase1_scale_clusters_{tag}").as_posix(),
        phase5_output_root=(root / f"phase1_adapter_{tag}").as_posix(),
        phase6_output_root=(root / f"phase1_control_{tag}").as_posix(),
        phase7_output_root=(root / f"phase1_holdout_block_{tag}").as_posix(),
        phase8_output_root=(root / f"phase1_history_descriptor_block_{tag}").as_posix(),
        phase9_output_root=(root / f"phase1_l2h_block_{tag}").as_posix(),
        final_output_root=(root / f"phase1_final_{tag}").as_posix(),
    )


def _classify_carrier_scope(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    path = ROOT / args.v75_phase1_root / "incidence_rows.csv"
    rows: list[dict[str, Any]] = []
    scope = "unknown"
    if not path.exists():
        return scope, rows
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        has_global = "carrier_global_id" in fields
        has_local = "carrier_id" in fields
        sample = []
        for _idx, row in zip(range(2000), reader):
            if str(row.get("membership_variant") or "") != str(args.incidence_variant):
                continue
            sample.append(row)
            if len(sample) >= 256:
                break
    if has_global:
        scope = "global_full_sequence"
    elif has_local:
        scope = "chunk_local"
    rows.append(
        {
            "artifact_path": _rel(path),
            "carrier_id_field": "carrier_global_id" if has_global else ("carrier_id" if has_local else ""),
            "carrier_id_scope": scope,
            "allowed_for_method_carrier_sketch": scope == "online_streaming",
            "allowed_for_diagnostic_carrier_sketch": True,
            "sampled_row_count": len(sample),
            "notes": "carrier_global_id is treated conservatively as global_full_sequence; carrier sketch is diagnostic-only until an online-streaming id policy is verified.",
        }
    )
    return scope, rows


def _run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase0_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    carrier_scope, carrier_rows = _classify_carrier_scope(args)
    sources = v80._source_paths(args)
    artifact_rows = [
        {
            "artifact_path": _rel(sources["v75_incidence_rows"]),
            "artifact_role": "local_soft_incidence_method_input",
            "allowed_for_method": True,
            "allowed_for_diagnostic": True,
            "uses_future": False,
            "uses_gt_for_prediction": False,
            "carrier_id_scope": carrier_scope,
            "history_anchor_type": "none",
            "metric_class": "",
            "notes": "Used only for current active split chunks. carrier sketch is diagnostic-only when carrier_id_scope is not online_streaming.",
        },
        {
            "artifact_path": _rel(sources["v71_semantic_rows"]),
            "artifact_role": "frozen_semantic_descriptor_method_input",
            "allowed_for_method": True,
            "allowed_for_diagnostic": True,
            "uses_future": False,
            "uses_gt_for_prediction": False,
            "carrier_id_scope": "",
            "history_anchor_type": "none",
            "metric_class": "",
            "notes": "Frozen DINO prototype rows joined by current mask_observation_id.",
        },
        {
            "artifact_path": _rel(_rooted(args.appearance_feature_rows)),
            "artifact_role": "frozen_appearance_descriptor_method_input",
            "allowed_for_method": args.appearance_feature_mode in {"dino_v58", "dino_csv"},
            "allowed_for_diagnostic": True,
            "uses_future": False,
            "uses_gt_for_prediction": False,
            "carrier_id_scope": "",
            "history_anchor_type": "none",
            "metric_class": "",
            "notes": "Optional frozen DINO feature_json rows. Rows marked uses_gt_for_prediction are skipped by the v81 loader.",
        },
        {
            "artifact_path": _rel(sources["v79_sweep"]),
            "artifact_role": "prior_baseline_diagnostic_reference",
            "allowed_for_method": False,
            "allowed_for_diagnostic": True,
            "uses_future": False,
            "uses_gt_for_prediction": False,
            "carrier_id_scope": "",
            "history_anchor_type": "none",
            "metric_class": "diagnostic_metric",
            "notes": "Used as prior fact/report anchor, not as a method selection signal for v81 output.",
        },
    ]
    metric_rows = [
        {"metric_name": "LCC", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": "GT-free graph collapse guard."},
        {"metric_name": "cannot_link_violation_count", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": "GT-free current evidence guard."},
        {"metric_name": "Q_coverage_rate", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": "History anchor coverage after warmup."},
        {"metric_name": "Q_entropy_mean", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": "History uncertainty gate."},
        {"metric_name": "local_SF50", "metric_class": "final_eval_metric", "can_drive_parameter_selection": False, "notes": "Adapter/frozen eval only; not used after holdout for tuning."},
        {"metric_name": "same_instance_history_topk_recall_diagnostic", "metric_class": "diagnostic_metric", "can_drive_parameter_selection": False, "notes": "GT diagnostic only if produced."},
    ]
    gt_prediction_violation_count = sum(1 for row in artifact_rows if _bool(row["allowed_for_method"]) and _bool(row["uses_gt_for_prediction"]))
    future_access_violation_count = sum(1 for row in artifact_rows if _bool(row["allowed_for_method"]) and _bool(row["uses_future"]))
    oracle_method_count = sum(1 for row in artifact_rows if _bool(row["allowed_for_method"]) and str(row["history_anchor_type"]) == "oracle")
    unlabeled = sum(1 for row in metric_rows if not str(row["metric_class"]).strip())
    carrier_method_unknown = carrier_scope == "unknown"
    gate = {
        "GT_prediction_violation_count_eq_0": gt_prediction_violation_count == 0,
        "future_access_violation_count_eq_0": future_access_violation_count == 0,
        "history_anchor_oracle_method_count_eq_0": oracle_method_count == 0,
        "metric_class_unlabeled_count_eq_0": unlabeled == 0,
        "carrier_id_scope_not_unknown_for_method_carrier_sketch": not carrier_method_unknown,
    }
    gate["pass"] = all(gate.values())
    summary = _phase_common(
        phase="v81_phase0_fact_causality",
        schema="stream4d_v81_phase0_fact_causality_v1",
        decision="PASS_V81_PHASE0_FACT_CAUSALITY" if gate["pass"] else "NO_GO_PHASE0_CAUSALITY_BOUNDARY",
        method_uses_gt=gt_prediction_violation_count > 0,
        method_uses_future=future_access_violation_count > 0,
        carrier_id_scope=carrier_scope,
        history_anchor_type="none",
        history_method_mode_allowed=False,
        can_enter_next_phase=gate["pass"],
        can_enter_local2history=False,
        primary_blocker="" if gate["pass"] else "phase0_boundary_violation",
        secondary_blocker="" if carrier_scope == "online_streaming" else "carrier_sketch_diagnostic_only_until_online_streaming_id_policy",
        extra={
            "GT_prediction_violation_count": gt_prediction_violation_count,
            "future_access_violation_count": future_access_violation_count,
            "history_anchor_oracle_method_count": oracle_method_count,
            "metric_class_unlabeled_count": unlabeled,
            "local2history_method_mode_allowed": False,
            "gate": gate,
            "runtime_sec": time.time() - started,
        },
    )
    _write_json(output_root / "history_boundary_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "input_artifact_rows.csv", artifact_rows)
    _write_csv(output_root / "carrier_id_scope_rows.csv", carrier_rows)
    _write_csv(output_root / "metric_class_rows.csv", metric_rows)
    return summary


def _write_alias_csv(source: Path, target: Path, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = _read_csv_rows(source)
    if extra:
        rows = [{**row, **extra} for row in rows]
    _write_csv(target, rows)
    return rows


def _run_phase1(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase1_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    local_args = _local_shadow_args(args)
    incidence = v80._load_incidence(local_args)
    feature_summary, bundles = v80._run_phase1(local_args, incidence)
    _signed_summary, signed_graphs = v80._run_phase2(local_args, bundles)
    cluster_summary, clusters = v80._run_phase4(local_args, signed_graphs, bundles)
    adapter_summary, _eval_by_chunk = v80._run_phase5(local_args, clusters)

    shadow = ROOT / local_args.phase5_output_root
    phase4_shadow = ROOT / local_args.phase4_output_root
    local_metric_rows = _write_alias_csv(shadow / "local_metric_rows.csv", output_root / "local_metric_rows.csv", {"variant": "L3_v80_signed_scale_gated_local_only"})
    local_cluster_rows = _write_alias_csv(phase4_shadow / "carrier_cluster_rows.csv", output_root / "local_cluster_rows.csv", {"variant": "L3_v80_signed_scale_gated_local_only"})
    local_slot_rows = _write_alias_csv(shadow / "local_slot_rows.csv", output_root / "local_slot_rows.csv", {"variant": "L3_v80_signed_scale_gated_local_only"})

    v79 = _read_json(ROOT / args.v79_sweep_root / "sweep_summary.json")
    v79_best = _float((v79.get("best_variant") or {}).get("local_SF50"), _float(adapter_summary.get("v79_best_replay_SF50"), 0.3287608225108225))
    variant_rows = [
        {
            "variant": "L0_v79_best_prior_replay_reference",
            "source": _rel(ROOT / args.v79_sweep_root / "sweep_summary.json"),
            "local_SF50": v79_best,
            "imported_prior_reference": True,
            "notes": "Prior v79 fact reference only; current v81 Phase1 metrics come from L3 run in this command.",
        },
        {
            "variant": "L3_v80_signed_scale_gated_local_only",
            "source": _rel(shadow / "adapter_summary.json"),
            "local_SF50": adapter_summary.get("local_SF50_rendered_adapter", ""),
            "local_AP50": adapter_summary.get("local_AP50", ""),
            "local_AP25": adapter_summary.get("local_AP25", ""),
            "GT_best_IoU_mean": adapter_summary.get("GT_best_IoU_mean", ""),
            "imported_prior_reference": False,
            "notes": "Current v81 bootstrap local replay using the v80 local-only implementation.",
        },
    ]
    _write_csv(output_root / "local_variant_rows.csv", variant_rows)

    case_rows = []
    for row in local_metric_rows:
        case_rows.append(
            {
                "scene_id": row.get("scene_id"),
                "chunk_id": row.get("chunk_id"),
                "variant": row.get("variant"),
                "local_SF50": row.get("local_SF50_rendered_adapter") or row.get("local_SF50_carrier_adapter"),
                "local_AP50": row.get("local_AP50"),
                "GT_best_IoU_mean": row.get("GT_best_IoU_mean"),
                "duplicate_frame_mask_conflict_rate": row.get("duplicate_frame_mask_conflict_rate"),
                "same_frame_violation_count": row.get("same_frame_violation_count"),
            }
        )
    _write_csv(output_root / "local_case_rows.csv", case_rows)

    local_sf50 = _float(adapter_summary.get("local_SF50_rendered_adapter"))
    gt_best = _float(adapter_summary.get("GT_best_IoU_mean"))
    duplicate_conflict = _float(adapter_summary.get("duplicate_frame_mask_conflict_rate"))
    same_frame_v = _int(adapter_summary.get("same_frame_violation_count"), 0)
    method_gt_v = _int(adapter_summary.get("method_GT_violation_count"), 0)
    gate = {
        "local_SF50_ge_0p33": local_sf50 >= 0.33,
        "GT_best_IoU_mean_ge_0p33": gt_best >= 0.33,
        "same_frame_violation_count_eq_0": same_frame_v == 0,
        "duplicate_frame_mask_conflict_rate_le_0p02": duplicate_conflict <= 0.02,
        "method_GT_violation_count_eq_0": method_gt_v == 0,
    }
    gate["pass"] = all(gate.values())
    final_local_gate = {
        "local_SF50_ge_0p40": local_sf50 >= 0.40,
        "local_SF50_ge_v79_plus_0p05": local_sf50 >= v79_best + 0.05,
        "GT_best_IoU_mean_ge_0p36": gt_best >= 0.36,
    }
    summary = _phase_common(
        phase="v81_phase1_bootstrap_local",
        schema="stream4d_v81_phase1_bootstrap_local_v1",
        decision="PASS_V81_PHASE1_BOOTSTRAP_LOCAL_HISTORY_ELIGIBLE" if gate["pass"] else "NO_GO_BOOTSTRAP_LOCAL_WEAK",
        method_uses_gt=False,
        method_uses_future=False,
        carrier_id_scope=args.carrier_id_scope_effective,
        history_anchor_type="none",
        history_method_mode_allowed=False,
        can_enter_next_phase=gate["pass"],
        can_enter_local2history=False,
        diagnostic_only_rows_present=True,
        forbidden_for_method_table_rows_present=True,
        primary_blocker="" if gate["pass"] else "bootstrap_local_below_history_eligibility",
        secondary_blocker="" if all(final_local_gate.values()) else "final_local_gate_not_met",
        extra={
            "local_SF50": local_sf50,
            "local_AP50": adapter_summary.get("local_AP50", ""),
            "local_AP25": adapter_summary.get("local_AP25", ""),
            "GT_best_IoU_mean": gt_best,
            "same_frame_violation_count": same_frame_v,
            "duplicate_frame_mask_conflict_rate": duplicate_conflict,
            "single_frame_slot_rate": adapter_summary.get("single_frame_slot_rate_object", ""),
            "cluster_count": cluster_summary.get("cluster_count_object", ""),
            "largest_cluster_ratio": cluster_summary.get("largest_cluster_ratio_object", ""),
            "within_semantic_instance_AUC_diagnostic": "",
            "method_GT_violation_count": method_gt_v,
            "v79_best_replay_SF50": v79_best,
            "history_eligibility_gate": gate,
            "final_local_gate": final_local_gate,
            "v80_feature_summary": feature_summary.get("decision"),
            "v80_cluster_summary": cluster_summary.get("decision"),
            "v80_adapter_summary": adapter_summary.get("decision"),
            "runtime_sec": time.time() - started,
        },
    )
    _write_json(output_root / "local_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, {"incidence": incidence, "feature_summary": feature_summary, "adapter_summary": adapter_summary}, bundles, clusters


def _carrier_proto(data: dict[str, Any], carrier: str) -> tuple[str, float, float]:
    profile = data.get("carrier_semantic_profiles", {}).get(carrier, {})
    return (
        str(profile.get("primary_proto") or ""),
        _float(profile.get("primary_margin"), 0.0),
        _float(profile.get("broad_background_share"), 0.0),
    )


def _cluster_descriptor(data: dict[str, Any], carriers: list[str]) -> dict[str, Any]:
    proto_scores: dict[str, float] = defaultdict(float)
    margins: list[float] = []
    broad: list[float] = []
    frames: set[int] = set()
    uvx: list[float] = []
    uvy: list[float] = []
    conf: list[float] = []
    appearance_vec: np.ndarray | None = None
    appearance_support_count = 0
    appearance_support_weight = 0.0
    appearance_total_weight = 0.0
    appearance_carrier_hits = 0
    appearance_profiles = data.get("carrier_appearance_profiles", {})
    suffixes = sorted({suffix for suffix in (_carrier_suffix(carrier) for carrier in carriers) if suffix >= 0})
    for carrier in carriers:
        proto, margin, broad_share = _carrier_proto(data, carrier)
        if proto:
            proto_scores[proto] += max(0.01, margin)
        margins.append(margin)
        broad.append(broad_share)
        frames.update(int(frame) for frame in data["carrier_frames"].get(carrier, set()))
        for row in data["carrier_obs"].get(carrier, []):
            uvx.append(float(row["uv_x"]))
            uvy.append(float(row["uv_y"]))
            conf.append(float(row["confidence"]))
        app = appearance_profiles.get(carrier, {})
        appearance_total_weight += _float(app.get("appearance_feature_total_weight"), 0.0)
        vec = app.get("appearance_vector")
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                weight = max(1e-6, _float(app.get("appearance_feature_support_weight"), 0.0))
                if appearance_vec is None:
                    appearance_vec = np.zeros_like(arr, dtype=np.float32)
                if appearance_vec.shape == arr.shape:
                    appearance_vec += np.float32(weight) * arr
                    appearance_support_count += _int(app.get("appearance_feature_support_count"), 0)
                    appearance_support_weight += weight
                    appearance_carrier_hits += 1
    if appearance_vec is not None:
        norm = float(np.linalg.norm(appearance_vec))
        appearance_vec = appearance_vec / np.float32(norm) if norm > 1e-12 else None
    proto = max(proto_scores, key=proto_scores.get) if proto_scores else ""
    return {
        "semantic_proto_id": proto,
        "semantic_margin_mean": _mean(margins) or 0.0,
        "broad_background_share_mean": _mean(broad) or 0.0,
        "frame_count": len(frames),
        "frame_span": (max(frames) - min(frames) + 1) if frames else 0,
        "uv_x_mean": _mean(uvx) or 0.0,
        "uv_y_mean": _mean(uvy) or 0.0,
        "confidence_mean": _mean(conf) or 0.0,
        "carrier_suffixes": ",".join(str(value) for value in suffixes),
        "carrier_suffix_count": len(suffixes),
        "appearance_vector": appearance_vec,
        "appearance_feature_norm": float(np.linalg.norm(appearance_vec)) if appearance_vec is not None else 0.0,
        "appearance_feature_dim": int(appearance_vec.size) if appearance_vec is not None else 0,
        "appearance_feature_hash": _vector_hash(appearance_vec),
        "appearance_feature_support_count": appearance_support_count,
        "appearance_feature_support_weight": appearance_support_weight,
        "appearance_feature_total_weight": appearance_total_weight,
        "appearance_carrier_coverage_rate": _safe_ratio(appearance_carrier_hits, max(1, len(carriers))),
        "appearance_feature_coverage_rate": _safe_ratio(appearance_support_weight, max(1e-12, appearance_total_weight)),
    }


def _uv_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    dx = _float(left.get("uv_x_mean"), 0.0) - _float(right.get("uv_x_mean"), 0.0)
    dy = _float(left.get("uv_y_mean"), 0.0) - _float(right.get("uv_y_mean"), 0.0)
    return float(math.sqrt(dx * dx + dy * dy))


def _carrier_suffix(carrier_id: str) -> int:
    raw = str(carrier_id).split(":")[-1]
    try:
        return int(raw) % 10000000
    except ValueError:
        return -1


def _parse_suffixes(raw: Any) -> set[int]:
    out: set[int] = set()
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _proto_tokens(proto: str) -> set[str]:
    parts = [part for part in str(proto or "").split("|") if part]
    if parts and parts[0] == "dino":
        parts = parts[1:]
    return set(parts)


def _semantic_score(left_proto: str, right_proto: str, args: argparse.Namespace) -> float:
    if not left_proto or not right_proto:
        return 0.0
    if str(getattr(args, "history_semantic_score_mode", "exact")) == "token_overlap":
        left = _proto_tokens(left_proto)
        right = _proto_tokens(right_proto)
        if not left or not right:
            return 0.0
        return _safe_ratio(len(left & right), max(len(left), len(right), 1))
    return 1.0 if left_proto == right_proto else 0.0


def _coalesce_history_nodes(
    node_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Merge per-cluster births into causal history nodes using past chunks only."""

    active_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    support_chunks: dict[str, set[int]] = {}
    alias_ids_by_history: dict[str, list[str]] = defaultdict(list)
    for row in sorted(node_rows, key=lambda item: (str(item.get("scene_id")), _int(item.get("source_chunk_id")), str(item.get("history_id")))):
        scene = str(row.get("scene_id") or "")
        chunk = _int(row.get("source_chunk_id"), -1)
        proto = str(row.get("semantic_proto_id") or "")
        best: dict[str, Any] | None = None
        best_score = float("-inf")
        for candidate in active_by_scene.get(scene, []):
            if str(candidate.get("state")) in {"inactive", "quarantine"}:
                continue
            semantic_match_score = _semantic_score(proto, str(candidate.get("semantic_proto_id") or ""), args)
            if semantic_match_score < float(args.history_semantic_match_threshold):
                continue
            age = chunk - _int(candidate.get("last_seen_chunk"), chunk)
            if age <= 0 or age > int(args.history_merge_max_age_chunks):
                continue
            dist = _uv_distance(row, candidate)
            spatial = math.exp(-0.5 * (dist / max(1e-6, float(args.history_spatial_sigma))) ** 2)
            age_decay = max(0.0, 1.0 - age / max(1.0, float(args.history_merge_max_age_chunks) + 1.0))
            appearance = _appearance_cosine(row.get("appearance_vector"), candidate.get("appearance_vector"))
            mode = str(args.history_coalescing_mode)
            if mode == "spatial":
                if dist > float(args.history_merge_uv_threshold):
                    continue
                score = 0.75 * spatial + 0.25 * age_decay
            elif mode == "appearance":
                if appearance < float(args.history_merge_min_appearance_score):
                    continue
                score = 0.70 * appearance + 0.20 * semantic_match_score + 0.10 * age_decay
            elif mode == "spatial_appearance":
                if dist > float(args.history_merge_uv_threshold):
                    continue
                if appearance < float(args.history_merge_min_appearance_score):
                    continue
                score = 0.45 * appearance + 0.25 * spatial + 0.20 * semantic_match_score + 0.10 * age_decay
            else:
                continue
            if score > best_score:
                best_score = score
                best = candidate
                row["coalesce_appearance_score"] = appearance
                row["coalesce_spatial_score"] = spatial
                row["coalesce_age_decay"] = age_decay
                row["coalesce_semantic_score"] = semantic_match_score
        if best is None or best_score < float(args.history_merge_min_score):
            if str(row.get("state")) == "confirmed":
                row["state"] = "tentative"
                row["demotion_reason"] = "no_spatial_temporal_prior_match"
            row["support_chunk_count"] = 1
            support_chunks[str(row["history_id"])] = {chunk}
            active_by_scene[scene].append(row)
            continue

        source_id = str(row["history_id"])
        target_id = str(best["history_id"])
        old_weight = max(1.0, _float(best.get("carrier_count"), 1.0))
        new_weight = max(1.0, _float(row.get("carrier_count"), 1.0))
        total_weight = old_weight + new_weight
        for field in ["uv_x_mean", "uv_y_mean", "semantic_margin_mean", "ambiguity_score"]:
            best[field] = (
                _float(best.get(field), 0.0) * old_weight
                + _float(row.get(field), 0.0) * new_weight
            ) / total_weight
        best_vec = best.get("appearance_vector")
        row_vec = row.get("appearance_vector")
        if best_vec is not None or row_vec is not None:
            left = np.asarray(best_vec, dtype=np.float32).reshape(-1) if best_vec is not None else None
            right = np.asarray(row_vec, dtype=np.float32).reshape(-1) if row_vec is not None else None
            merged: np.ndarray | None = None
            if left is not None and right is not None and left.shape == right.shape:
                merged = left * np.float32(old_weight) + right * np.float32(new_weight)
            elif left is not None:
                merged = left
            elif right is not None:
                merged = right
            if merged is not None:
                norm = float(np.linalg.norm(merged))
                merged = merged / np.float32(norm) if norm > 1e-12 else None
            best["appearance_vector"] = merged
            best["appearance_feature_norm"] = float(np.linalg.norm(merged)) if merged is not None else 0.0
            best["appearance_feature_dim"] = int(merged.size) if merged is not None else 0
            best["appearance_feature_hash"] = _vector_hash(merged)
        best["appearance_feature_support_count"] = _int(best.get("appearance_feature_support_count"), 0) + _int(row.get("appearance_feature_support_count"), 0)
        best["appearance_feature_support_weight"] = _float(best.get("appearance_feature_support_weight"), 0.0) + _float(row.get("appearance_feature_support_weight"), 0.0)
        best["appearance_feature_total_weight"] = _float(best.get("appearance_feature_total_weight"), 0.0) + _float(row.get("appearance_feature_total_weight"), 0.0)
        best["appearance_carrier_coverage_rate"] = _safe_ratio(
            _float(best.get("appearance_carrier_coverage_rate"), 0.0) * old_weight
            + _float(row.get("appearance_carrier_coverage_rate"), 0.0) * new_weight,
            total_weight,
        )
        best["appearance_feature_coverage_rate"] = _safe_ratio(
            _float(best.get("appearance_feature_support_weight"), 0.0),
            max(1e-12, _float(best.get("appearance_feature_total_weight"), 0.0)),
        )
        best["carrier_count"] = _int(best.get("carrier_count"), 0) + _int(row.get("carrier_count"), 0)
        best["carrier_sketch_size"] = _int(best.get("carrier_sketch_size"), 0) + _int(row.get("carrier_sketch_size"), 0)
        suffixes = _parse_suffixes(best.get("carrier_suffixes")) | _parse_suffixes(row.get("carrier_suffixes"))
        best["carrier_suffixes"] = ",".join(str(value) for value in sorted(suffixes))
        best["carrier_suffix_count"] = len(suffixes)
        best["frame_count"] = max(_int(best.get("frame_count"), 0), _int(row.get("frame_count"), 0))
        best["frame_span"] = max(_int(best.get("frame_span"), 0), _int(row.get("frame_span"), 0))
        best["confidence"] = min(1.0, max(_float(best.get("confidence"), 0.0), _float(row.get("confidence"), 0.0)) + 0.03)
        best["last_seen_chunk"] = max(_int(best.get("last_seen_chunk"), chunk), chunk)
        best["version_id"] = _int(best.get("version_id"), 1) + 1
        support_chunks.setdefault(target_id, {_int(best.get("source_chunk_id"), chunk)}).add(chunk)
        best["support_chunk_count"] = len(support_chunks[target_id])
        alias_ids_by_history[target_id].append(source_id)
        best["child_ids"] = ",".join(alias_ids_by_history[target_id])
        if (
            int(best["support_chunk_count"]) >= int(args.history_confirm_min_support_chunks)
            and _float(best.get("confidence"), 0.0) >= float(args.history_confirm_confidence)
        ):
            best["state"] = "confirmed"

        row["state"] = "inactive"
        row["alias_of_history_id"] = target_id
        row["inactive_age"] = 0
        row["eviction_reason"] = "coalesced_into_prior_history_node"
        for update in update_rows:
            if str(update.get("history_id")) == source_id:
                update["assigned_history_id"] = target_id
                update["update_action"] = "coalesce_update_existing_history"
                update["state_after"] = str(best.get("state"))
                update["coalesce_score"] = best_score
                update["coalesce_appearance_score"] = row.get("coalesce_appearance_score", "")
                update["coalesce_spatial_score"] = row.get("coalesce_spatial_score", "")
                update["coalesce_semantic_score"] = row.get("coalesce_semantic_score", "")
                break


def _build_causal_history_snapshots(
    birth_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Replay online memory updates and emit M_{r-1} snapshots per scene/chunk."""

    rows_by_scene_chunk: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in birth_rows:
        rows_by_scene_chunk[(str(row.get("scene_id") or ""), _int(row.get("source_chunk_id"), -1))].append(
            _copy_history_row(row)
        )

    active_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    support_chunks: dict[str, set[int]] = {}
    alias_ids_by_history: dict[str, list[str]] = defaultdict(list)
    snapshot_rows: list[dict[str, Any]] = []

    def emit_snapshot(scene: str, chunk: int) -> None:
        active = [row for row in active_by_scene.get(scene, []) if str(row.get("state")) != "inactive"]
        for idx, row in enumerate(active):
            snap = _copy_history_row(row)
            snap["snapshot_chunk_id"] = chunk
            snap["snapshot_available_before_chunk"] = chunk
            snap["snapshot_row_index"] = idx
            snap["snapshot_memory_mode"] = "causal_online_replay"
            snapshot_rows.append(snap)

    def prune_scene(scene: str) -> None:
        active = active_by_scene.get(scene, [])
        tentative = [row for row in active if str(row.get("state")) == "tentative"]
        excess = len(tentative) - int(args.max_tentative_nodes)
        if excess <= 0:
            return
        prune_order = sorted(
            tentative,
            key=lambda row: (
                _float(row.get("confidence"), 0.0),
                _int(row.get("frame_count"), 0),
                _int(row.get("carrier_count"), 0),
                -_float(row.get("ambiguity_score"), 0.0),
            ),
        )
        prune_ids = {str(row.get("history_id")) for row in prune_order[:excess]}
        for row in active:
            if str(row.get("history_id")) in prune_ids:
                row["state"] = "inactive"
                row["inactive_age"] = 0
                row["eviction_reason"] = "online_tentative_budget_prune_low_confidence"

    def merge_into(best: dict[str, Any], row: dict[str, Any], chunk: int) -> None:
        source_id = str(row["history_id"])
        target_id = str(best["history_id"])
        old_weight = max(1.0, _float(best.get("carrier_count"), 1.0))
        new_weight = max(1.0, _float(row.get("carrier_count"), 1.0))
        total_weight = old_weight + new_weight
        for field in ["uv_x_mean", "uv_y_mean", "semantic_margin_mean", "ambiguity_score"]:
            best[field] = (
                _float(best.get(field), 0.0) * old_weight
                + _float(row.get(field), 0.0) * new_weight
            ) / total_weight
        best_vec = best.get("appearance_vector")
        row_vec = row.get("appearance_vector")
        if best_vec is not None or row_vec is not None:
            left = np.asarray(best_vec, dtype=np.float32).reshape(-1) if best_vec is not None else None
            right = np.asarray(row_vec, dtype=np.float32).reshape(-1) if row_vec is not None else None
            merged: np.ndarray | None = None
            if left is not None and right is not None and left.shape == right.shape:
                merged = left * np.float32(old_weight) + right * np.float32(new_weight)
            elif left is not None:
                merged = left.copy()
            elif right is not None:
                merged = right.copy()
            if merged is not None:
                norm = float(np.linalg.norm(merged))
                merged = merged / np.float32(norm) if norm > 1e-12 else None
            best["appearance_vector"] = merged
            best["appearance_feature_norm"] = float(np.linalg.norm(merged)) if merged is not None else 0.0
            best["appearance_feature_dim"] = int(merged.size) if merged is not None else 0
            best["appearance_feature_hash"] = _vector_hash(merged)
        best["appearance_feature_support_count"] = _int(best.get("appearance_feature_support_count"), 0) + _int(row.get("appearance_feature_support_count"), 0)
        best["appearance_feature_support_weight"] = _float(best.get("appearance_feature_support_weight"), 0.0) + _float(row.get("appearance_feature_support_weight"), 0.0)
        best["appearance_feature_total_weight"] = _float(best.get("appearance_feature_total_weight"), 0.0) + _float(row.get("appearance_feature_total_weight"), 0.0)
        best["appearance_carrier_coverage_rate"] = _safe_ratio(
            _float(best.get("appearance_carrier_coverage_rate"), 0.0) * old_weight
            + _float(row.get("appearance_carrier_coverage_rate"), 0.0) * new_weight,
            total_weight,
        )
        best["appearance_feature_coverage_rate"] = _safe_ratio(
            _float(best.get("appearance_feature_support_weight"), 0.0),
            max(1e-12, _float(best.get("appearance_feature_total_weight"), 0.0)),
        )
        best["carrier_count"] = _int(best.get("carrier_count"), 0) + _int(row.get("carrier_count"), 0)
        best["carrier_sketch_size"] = _int(best.get("carrier_sketch_size"), 0) + _int(row.get("carrier_sketch_size"), 0)
        suffixes = _parse_suffixes(best.get("carrier_suffixes")) | _parse_suffixes(row.get("carrier_suffixes"))
        best["carrier_suffixes"] = ",".join(str(value) for value in sorted(suffixes))
        best["carrier_suffix_count"] = len(suffixes)
        best["frame_count"] = max(_int(best.get("frame_count"), 0), _int(row.get("frame_count"), 0))
        best["frame_span"] = max(_int(best.get("frame_span"), 0), _int(row.get("frame_span"), 0))
        best["confidence"] = min(1.0, max(_float(best.get("confidence"), 0.0), _float(row.get("confidence"), 0.0)) + 0.03)
        best["last_seen_chunk"] = max(_int(best.get("last_seen_chunk"), chunk), chunk)
        best["version_id"] = _int(best.get("version_id"), 1) + 1
        support_chunks.setdefault(target_id, {_int(best.get("source_chunk_id"), chunk)}).add(chunk)
        best["support_chunk_count"] = len(support_chunks[target_id])
        alias_ids_by_history[target_id].append(source_id)
        best["child_ids"] = ",".join(alias_ids_by_history[target_id])
        if (
            int(best["support_chunk_count"]) >= int(args.history_confirm_min_support_chunks)
            and _float(best.get("confidence"), 0.0) >= float(args.history_confirm_confidence)
        ):
            best["state"] = "confirmed"

    for scene in sorted({scene for scene, _chunk in rows_by_scene_chunk}):
        chunks = sorted(chunk for row_scene, chunk in rows_by_scene_chunk if row_scene == scene)
        for chunk in chunks:
            emit_snapshot(scene, chunk)
            for row in sorted(rows_by_scene_chunk[(scene, chunk)], key=lambda item: str(item.get("history_id"))):
                proto = str(row.get("semantic_proto_id") or "")
                best: dict[str, Any] | None = None
                best_score = float("-inf")
                for candidate in active_by_scene.get(scene, []):
                    if str(candidate.get("state")) in {"inactive", "quarantine"}:
                        continue
                    semantic_match_score = _semantic_score(proto, str(candidate.get("semantic_proto_id") or ""), args)
                    if semantic_match_score < float(args.history_semantic_match_threshold):
                        continue
                    age = chunk - _int(candidate.get("last_seen_chunk"), chunk)
                    if age <= 0 or age > int(args.history_merge_max_age_chunks):
                        continue
                    dist = _uv_distance(row, candidate)
                    spatial = math.exp(-0.5 * (dist / max(1e-6, float(args.history_spatial_sigma))) ** 2)
                    age_decay = max(0.0, 1.0 - age / max(1.0, float(args.history_merge_max_age_chunks) + 1.0))
                    appearance = _appearance_cosine(row.get("appearance_vector"), candidate.get("appearance_vector"))
                    mode = str(args.history_coalescing_mode)
                    if mode == "spatial":
                        if dist > float(args.history_merge_uv_threshold):
                            continue
                        score = 0.75 * spatial + 0.25 * age_decay
                    elif mode == "appearance":
                        if appearance < float(args.history_merge_min_appearance_score):
                            continue
                        score = 0.70 * appearance + 0.20 * semantic_match_score + 0.10 * age_decay
                    elif mode == "spatial_appearance":
                        if dist > float(args.history_merge_uv_threshold):
                            continue
                        if appearance < float(args.history_merge_min_appearance_score):
                            continue
                        score = 0.45 * appearance + 0.25 * spatial + 0.20 * semantic_match_score + 0.10 * age_decay
                    else:
                        continue
                    if score > best_score:
                        best_score = score
                        best = candidate
                if best is None or best_score < float(args.history_merge_min_score):
                    if str(row.get("state")) == "confirmed":
                        row["state"] = "tentative"
                        row["demotion_reason"] = "no_spatial_temporal_prior_match"
                    row["support_chunk_count"] = 1
                    support_chunks[str(row["history_id"])] = {chunk}
                    active_by_scene[scene].append(row)
                    continue
                merge_into(best, row, chunk)
                row["state"] = "inactive"
                row["alias_of_history_id"] = str(best.get("history_id"))
                row["inactive_age"] = 0
                row["eviction_reason"] = "coalesced_into_prior_history_node"
            prune_scene(scene)
    return snapshot_rows


def _run_phase2(
    args: argparse.Namespace,
    phase1: dict[str, Any],
    clusters: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase2_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not bool((phase1.get("history_eligibility_gate") or {}).get("pass")):
        summary = _blocked_phase(
            output_root,
            "v81_phase2_bootstrap_history",
            "stream4d_v81_phase2_history_v1",
            "BLOCK_HISTORY_BY_BOOTSTRAP_LOCAL_WEAK",
            args,
        )
        _write_csv(output_root / "history_node_rows.csv", [])
        _write_csv(output_root / "history_update_rows.csv", [])
        return summary, []

    node_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    proto_seen_by_scene: dict[str, Counter[str]] = defaultdict(Counter)
    next_id = 1
    for (scene, chunk), graph in sorted(clusters.items()):
        data = graph["data"]
        carriers_all = graph["carriers"]
        chunk_proto_counts: Counter[str] = Counter()
        for label, indices in sorted(graph["label_to_indices"].items()):
            carriers = [carriers_all[int(idx)] for idx in indices]
            desc = _cluster_descriptor(data, carriers)
            proto = desc["semantic_proto_id"]
            confidence = min(
                1.0,
                0.25
                + 0.10 * min(4, len(carriers))
                + 0.08 * min(3, desc["frame_count"])
                + 0.25 * desc["semantic_margin_mean"]
                + 0.10 * desc["confidence_mean"],
            )
            if desc["broad_background_share_mean"] > 0.80 or len(carriers) > max(1, int(args.max_history_cluster_carriers)):
                state = "quarantine"
            elif proto and proto_seen_by_scene[scene][proto] > 0 and confidence >= float(args.history_confirm_confidence):
                state = "confirmed"
            else:
                state = "tentative"
            history_id = f"H{next_id:05d}"
            next_id += 1
            row = {
                "scene_id": scene,
                "history_id": history_id,
                "version_id": 1,
                "state": state,
                "source_chunk_id": chunk,
                "source_local_slot_id": f"V81_bootstrap:c{chunk}:cluster{label}",
                "semantic_proto_id": proto,
                "carrier_sketch_size": len(carriers),
                "carrier_sketch_allowed_for_method": args.carrier_id_scope_effective == "online_streaming",
                "appearance_feature_norm": desc["appearance_feature_norm"],
                "appearance_feature_dim": desc["appearance_feature_dim"],
                "appearance_feature_hash": desc["appearance_feature_hash"],
                "appearance_feature_support_count": desc["appearance_feature_support_count"],
                "appearance_feature_support_weight": desc["appearance_feature_support_weight"],
                "appearance_feature_total_weight": desc["appearance_feature_total_weight"],
                "appearance_feature_coverage_rate": desc["appearance_feature_coverage_rate"],
                "appearance_carrier_coverage_rate": desc["appearance_carrier_coverage_rate"],
                "appearance_descriptor_source": args.appearance_feature_mode,
                "appearance_vector": desc["appearance_vector"],
                "motion_summary_valid": desc["frame_count"] > 0,
                "scale_role": "object",
                "confidence": confidence,
                "ambiguity_score": desc["broad_background_share_mean"],
                "last_seen_chunk": chunk,
                "parent_id": "",
                "child_ids": "",
                "semantic_margin_mean": desc["semantic_margin_mean"],
                "frame_count": desc["frame_count"],
                "frame_span": desc["frame_span"],
                "uv_x_mean": desc["uv_x_mean"],
                "uv_y_mean": desc["uv_y_mean"],
                "carrier_count": len(carriers),
                "carrier_suffixes": desc["carrier_suffixes"],
                "carrier_suffix_count": desc["carrier_suffix_count"],
            }
            node_rows.append(row)
            update_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "local_cluster_id": label,
                    "history_id": history_id,
                    "update_action": f"create_{state}",
                    "state_after": state,
                    "confidence": confidence,
                    "uses_gt_for_prediction": False,
                }
            )
            if proto:
                chunk_proto_counts[proto] += 1
        proto_seen_by_scene[scene].update(chunk_proto_counts)

    birth_rows = [_copy_history_row(row) for row in node_rows]
    snapshot_rows = _build_causal_history_snapshots(birth_rows, args) if bool(args.history_use_causal_snapshots) else []

    if str(args.history_coalescing_mode) in {"spatial", "appearance", "spatial_appearance"}:
        _coalesce_history_nodes(node_rows, update_rows, args)

    tentative_rows = [row for row in node_rows if row["state"] == "tentative"]
    if len(tentative_rows) > int(args.max_tentative_nodes):
        excess = len(tentative_rows) - int(args.max_tentative_nodes)
        prune_order = sorted(
            tentative_rows,
            key=lambda row: (
                _float(row.get("confidence"), 0.0),
                _int(row.get("frame_count"), 0),
                _int(row.get("carrier_count"), 0),
                -_float(row.get("ambiguity_score"), 0.0),
            ),
        )
        prune_ids = {str(row["history_id"]) for row in prune_order[:excess]}
        for row in node_rows:
            if str(row["history_id"]) in prune_ids:
                row["state"] = "inactive"
                row["inactive_age"] = 0
                row["eviction_reason"] = "tentative_budget_prune_low_confidence_single_chunk"
        for row in update_rows:
            if str(row["history_id"]) in prune_ids:
                row["update_action"] = "create_then_prune_inactive"
                row["state_after"] = "inactive"
                row["eviction_reason"] = "tentative_budget_prune_low_confidence_single_chunk"

    active_rows = [row for row in node_rows if row["state"] != "inactive"]
    node_count = len(active_rows)
    confirmed = sum(1 for row in active_rows if row["state"] == "confirmed")
    tentative = sum(1 for row in active_rows if row["state"] == "tentative")
    quarantine = sum(1 for row in active_rows if row["state"] == "quarantine")
    inactive = sum(1 for row in node_rows if row["state"] == "inactive")
    semantic_cov = _safe_ratio(sum(1 for row in active_rows if row["semantic_proto_id"]), max(1, node_count))
    appearance_cov = _safe_ratio(sum(1 for row in active_rows if _float(row["appearance_feature_norm"]) > 0.0), max(1, node_count))
    appearance_support_count = sum(_int(row.get("appearance_feature_support_count"), 0) for row in active_rows)
    appearance_feature_coverage_mean = _mean([_float(row.get("appearance_feature_coverage_rate"), 0.0) for row in active_rows]) or 0.0
    appearance_carrier_coverage_mean = _mean([_float(row.get("appearance_carrier_coverage_rate"), 0.0) for row in active_rows]) or 0.0
    carrier_cov = _safe_ratio(sum(1 for row in active_rows if _int(row["carrier_sketch_size"]) > 0), max(1, node_count))
    appearance_bytes = sum(_int(row.get("appearance_feature_dim"), 0) * 4 for row in active_rows)
    memory_mb = node_count * 1.25 / 1024.0 + appearance_bytes / (1024.0 * 1024.0)
    carrier_scope_online = args.carrier_id_scope_effective == "online_streaming"
    gate = {
        "history_node_count_le_max_history_nodes": node_count <= int(args.max_history_nodes),
        "tentative_node_count_le_max_tentative_nodes": tentative <= int(args.max_tentative_nodes),
        "carrier_sketch_coverage_rate_ge_0p70_if_online": (carrier_cov >= 0.70) if carrier_scope_online else True,
        "semantic_descriptor_coverage_rate_ge_0p95": semantic_cov >= 0.95,
        "history_descriptor_contains_local_mask_hash_as_primary_id_false": True,
        "memory_MB_le_budget": memory_mb <= float(args.memory_budget_mb),
    }
    gate["pass"] = all(gate.values())
    summary = _phase_common(
        phase="v81_phase2_bootstrap_history",
        schema="stream4d_v81_phase2_history_v1",
        decision="PASS_V81_PHASE2_BOOTSTRAP_HISTORY" if gate["pass"] else "NO_GO_HISTORY_DESCRIPTOR_WEAK",
        carrier_id_scope=args.carrier_id_scope_effective,
        history_anchor_type="tentative_confirmed_from_bootstrap_local",
        history_method_mode_allowed=gate["pass"],
        can_enter_next_phase=gate["pass"],
        can_enter_local2history=False,
        primary_blocker="" if gate["pass"] else "history_descriptor_or_budget_gate_failed",
        secondary_blocker="" if carrier_scope_online else "carrier_sketch_diagnostic_only_non_online_scope",
        extra={
            "history_node_count": node_count,
            "raw_history_node_row_count": len(node_rows),
            "confirmed_node_count": confirmed,
            "tentative_node_count": tentative,
            "quarantine_node_count": quarantine,
            "inactive_node_count": inactive,
            "carrier_sketch_coverage_rate": carrier_cov,
            "semantic_descriptor_coverage_rate": semantic_cov,
            "appearance_descriptor_coverage_rate": appearance_cov,
            "appearance_feature_support_count": appearance_support_count,
            "appearance_feature_coverage_rate_mean": appearance_feature_coverage_mean,
            "appearance_carrier_coverage_rate_mean": appearance_carrier_coverage_mean,
            "appearance_descriptor_source": args.appearance_feature_mode,
            "appearance_feature_audit": getattr(args, "appearance_audit_summary", {}),
            "history_confidence_mean": _mean([_float(row["confidence"]) for row in node_rows]) or 0.0,
            "history_entropy_mean": "",
            "memory_MB": memory_mb,
            "history_descriptor_contains_local_mask_hash_as_primary_id": False,
            "history_use_causal_snapshots": bool(args.history_use_causal_snapshots),
            "history_snapshot_row_count": len(snapshot_rows),
            "history_snapshot_chunk_count": len({(str(row.get("scene_id")), _int(row.get("snapshot_chunk_id"), -1)) for row in snapshot_rows}),
            "history_rows_for_phase3_source": "history_snapshot_rows" if bool(args.history_use_causal_snapshots) else "history_node_rows",
            "gate": gate,
            "runtime_sec": time.time() - started,
        },
    )
    _write_csv(output_root / "history_node_rows.csv", _sanitize_history_rows(node_rows))
    _write_csv(output_root / "history_update_rows.csv", update_rows)
    _write_csv(output_root / "history_snapshot_rows.csv", _sanitize_history_rows(snapshot_rows))
    _write_json(output_root / "history_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    history_rows_for_phase3 = snapshot_rows if bool(args.history_use_causal_snapshots) else node_rows
    return summary, history_rows_for_phase3


def _q_entropy(scores: list[float]) -> float:
    vals = [max(0.0, float(score)) for score in scores]
    total = sum(vals)
    if total <= 0.0 or len(vals) <= 1:
        return 0.0
    probs = [v / total for v in vals if v > 0.0]
    return float(-sum(p * math.log(p) for p in probs) / math.log(len(vals)))


def _carrier_descriptor(data: dict[str, Any], carrier: str) -> dict[str, Any]:
    rows = data["carrier_obs"].get(carrier, [])
    uvx = [float(row["uv_x"]) for row in rows]
    uvy = [float(row["uv_y"]) for row in rows]
    conf = [float(row["confidence"]) for row in rows]
    frames = {int(frame) for frame in data["carrier_frames"].get(carrier, set())}
    app = data.get("carrier_appearance_profiles", {}).get(carrier, {})
    return {
        "uv_x_mean": _mean(uvx) or 0.0,
        "uv_y_mean": _mean(uvy) or 0.0,
        "confidence_mean": _mean(conf) or 0.0,
        "frame_count": len(frames),
        "frame_span": (max(frames) - min(frames) + 1) if frames else 0,
        "carrier_suffix": _carrier_suffix(carrier),
        "appearance_vector": app.get("appearance_vector"),
        "appearance_feature_norm": 1.0 if app.get("appearance_vector") is not None else 0.0,
        "appearance_feature_support_count": _int(app.get("appearance_feature_support_count"), 0),
        "appearance_feature_coverage_rate": _float(app.get("appearance_feature_coverage_rate"), 0.0),
    }


def _score_history(
    carrier_proto: str,
    carrier_margin: float,
    carrier_broad: float,
    carrier_desc: dict[str, Any],
    history: dict[str, Any],
    chunk_id: int,
    *,
    variant: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, float]:
    full_q_variants = {
        "Q4_full_carrier_to_history_affinity",
        "Q5_full_affinity_tentative_disabled",
        "Q6_full_affinity_confirmed_only",
    }
    h_proto = str(history.get("semantic_proto_id") or "")
    dist = _uv_distance(carrier_desc, history)
    spatial_score = math.exp(-0.5 * (dist / max(1e-6, float(args.history_spatial_sigma))) ** 2)
    legacy_affinity = str(args.history_affinity_mode) == "legacy"
    history_state = str(history.get("state") or "")
    semantic_match_threshold = float(args.history_semantic_match_threshold)
    residual_floor_score = 0.0
    if history_state == "confirmed":
        state_weight = float(args.history_confirmed_score_weight)
    elif history_state == "tentative":
        state_weight = float(args.history_tentative_score_weight)
    elif history_state == "quarantine":
        state_weight = float(args.history_quarantine_score_weight)
    else:
        state_weight = 0.0
    if variant == "Q0_semantic_only_history_affinity":
        semantic_score = _semantic_score(carrier_proto, h_proto, args)
        appearance_score = 0.0
        temporal_score = 0.0
    elif variant == "Q1_appearance_only_history_affinity":
        semantic_score = 0.0
        appearance_score = _appearance_cosine(carrier_desc.get("appearance_vector"), history.get("appearance_vector"))
        temporal_score = 0.0
    elif variant == "Q2_carrier_suffix_diagnostic":
        semantic_score = 0.0
        appearance_score = 0.0
        temporal_score = 0.0
    elif variant == "Q4_full_carrier_to_history_affinity_shuffled":
        semantic_score = rng.random()
        appearance_score = semantic_score
        temporal_score = 0.0
    elif variant == "Q4_full_carrier_to_history_affinity_stale":
        semantic_score = _semantic_score(carrier_proto, h_proto, args)
        appearance_score = semantic_score * 0.50
        temporal_score = 0.0
    else:
        semantic_score = _semantic_score(carrier_proto, h_proto, args)
        if str(args.appearance_feature_mode) in {"dino_v58", "dino_csv"}:
            appearance_score = _appearance_cosine(carrier_desc.get("appearance_vector"), history.get("appearance_vector"))
        elif legacy_affinity:
            appearance_score = semantic_score * max(0.0, min(1.0, carrier_margin + _float(history.get("semantic_margin_mean"), 0.0)))
        else:
            margin_score = math.sqrt(
                max(0.0, min(1.0, carrier_margin))
                * max(0.0, min(1.0, _float(history.get("semantic_margin_mean"), 0.0)))
            )
            appearance_score = semantic_score * (0.70 * spatial_score + 0.30 * margin_score)
        age = max(0, int(chunk_id) - _int(history.get("last_seen_chunk"), int(chunk_id)))
        temporal_score = max(0.0, 1.0 - age / max(1.0, float(args.history_temporal_decay_chunks)))
    carrier_sketch_score = 0.0
    if variant == "Q2_carrier_suffix_diagnostic":
        suffix = _int(carrier_desc.get("carrier_suffix"), -1)
        carrier_sketch_score = 1.0 if suffix >= 0 and suffix in _parse_suffixes(history.get("carrier_suffixes")) else 0.0
    scale_score = 1.0 if str(history.get("scale_role") or "object") == "object" else 0.0
    conflict_score = 0.0
    if carrier_broad > 0.80 or str(history.get("state")) == "quarantine":
        conflict_score = max(conflict_score, 0.5)
    if carrier_proto and h_proto and semantic_score < semantic_match_threshold:
        conflict_score = max(conflict_score, 0.25)
    if not legacy_affinity and semantic_score >= semantic_match_threshold and spatial_score < float(args.history_spatial_conflict_threshold):
        conflict_score = max(conflict_score, 0.35)
    if legacy_affinity:
        q = (
            0.55 * semantic_score
            + 0.25 * appearance_score
            + 0.00 * carrier_sketch_score
            + 0.15 * temporal_score
            + 0.05 * scale_score
            - 0.35 * conflict_score
        )
    else:
        q = (
            0.30 * semantic_score
            + 0.35 * appearance_score
            + 0.00 * carrier_sketch_score
            + 0.25 * temporal_score
            + 0.05 * scale_score
            + 0.05 * _float(history.get("confidence"), 0.0)
            - 0.40 * conflict_score
        )
        q *= state_weight
        if (
            str(getattr(args, "history_full_score_calibration", "weighted")) == "confirmed_semantic_residual_floor"
            and variant in full_q_variants
            and history_state == "confirmed"
            and semantic_score >= semantic_match_threshold
        ):
            app_residual = max(0.0, appearance_score - semantic_score)
            temporal_residual = max(0.0, temporal_score - 0.5) * max(0.0, min(1.0, appearance_score))
            confidence_residual = max(0.0, _float(history.get("confidence"), 0.0) - 0.5)
            residual_floor_score = min(
                1.0,
                semantic_score
                + 0.50 * app_residual
                + 0.05 * temporal_residual
                + 0.02 * confidence_residual
                - 0.40 * conflict_score,
            )
            q = max(q, residual_floor_score)
    if variant == "Q0_semantic_only_history_affinity":
        q = semantic_score
    elif variant == "Q1_appearance_only_history_affinity":
        q = appearance_score
    elif variant == "Q2_carrier_suffix_diagnostic":
        q = carrier_sketch_score
    semantic_gate_suppressed = (
        bool(getattr(args, "history_full_require_semantic_match", False))
        and variant in full_q_variants
        and semantic_score < semantic_match_threshold
    )
    if semantic_gate_suppressed:
        q = 0.0
    return {
        "q_score": max(0.0, q),
        "semantic_score": semantic_score,
        "appearance_score": appearance_score,
        "carrier_sketch_score": carrier_sketch_score,
        "temporal_score": temporal_score,
        "spatial_score": spatial_score,
        "history_state_weight": state_weight,
        "scale_score": scale_score,
        "conflict_score": conflict_score,
        "semantic_residual_floor_score": residual_floor_score,
        "history_full_score_calibration": str(getattr(args, "history_full_score_calibration", "weighted")),
        "semantic_gate_suppressed": float(semantic_gate_suppressed),
        "semantic_match_for_method": float(semantic_score >= semantic_match_threshold),
    }


def _run_phase3(
    args: argparse.Namespace,
    phase2: dict[str, Any],
    history_rows: list[dict[str, Any]],
    clusters: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase3_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not bool((phase2.get("gate") or {}).get("pass")):
        summary = _blocked_phase(
            output_root,
            "v81_phase3_carrier_to_history",
            "stream4d_v81_phase3_q_v1",
            "BLOCK_Q_BY_HISTORY_DESCRIPTOR_WEAK",
            args,
        )
        _write_csv(output_root / "q_rows.csv", [])
        _write_csv(output_root / "q_control_rows.csv", [])
        return summary

    variants = [
        "Q0_semantic_only_history_affinity",
        "Q1_appearance_only_history_affinity",
        "Q2_carrier_suffix_diagnostic",
        "Q4_full_carrier_to_history_affinity",
        "Q5_full_affinity_tentative_disabled",
        "Q6_full_affinity_confirmed_only",
        "Q4_full_carrier_to_history_affinity_shuffled",
        "Q4_full_carrier_to_history_affinity_stale",
    ]
    rng = random.Random(int(args.random_seed) + 8103)
    histories_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history_rows:
        histories_by_scene[str(row.get("scene_id"))].append(row)

    q_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    variant_metric: dict[str, dict[str, list[float]]] = {
        variant: defaultdict(list) for variant in variants
    }
    for (scene, chunk), graph in sorted(clusters.items()):
        data = graph["data"]
        scene_histories = histories_by_scene.get(scene, [])
        snapshot_histories = [
            row for row in scene_histories
            if str(row.get("snapshot_memory_mode") or "") == "causal_online_replay"
        ]
        if snapshot_histories:
            raw_histories = [
                row for row in snapshot_histories
                if _int(row.get("snapshot_chunk_id"), -1) == int(chunk)
                and str(row.get("state")) != "inactive"
            ]
        else:
            raw_histories = [
                row for row in scene_histories
                if _int(row.get("source_chunk_id"), 10**9) < int(chunk)
                and str(row.get("state")) != "inactive"
            ]
        future_descriptor_rows = [
            row for row in raw_histories
            if _int(row.get("last_seen_chunk"), _int(row.get("source_chunk_id"), -1)) >= int(chunk)
        ]
        if bool(getattr(args, "history_require_causal_last_seen", False)):
            histories = [
                row for row in raw_histories
                if _int(row.get("last_seen_chunk"), _int(row.get("source_chunk_id"), -1)) < int(chunk)
            ]
        else:
            histories = list(raw_histories)
        after_warmup = int(chunk) >= int(args.history_warmup_chunks)
        for variant in variants:
            candidate_histories = list(histories)
            if variant in {"Q5_full_affinity_tentative_disabled", "Q6_full_affinity_confirmed_only"}:
                candidate_histories = [row for row in candidate_histories if str(row.get("state")) == "confirmed"]
            carrier_count = 0
            covered = 0
            confirmed_used = 0
            tentative_used = 0
            quarantine_used = 0
            top1_scores: list[float] = []
            raw_top1_scores: list[float] = []
            raw_top1_positive = 0
            raw_qmin_pass = 0
            raw_entropy_pass = 0
            candidate_eval_count = 0
            semantic_gate_suppressed = 0
            entropies: list[float] = []
            false_attach_proxy = 0
            no_anchor = 0
            for carrier in graph["carriers"]:
                carrier_count += 1
                proto, margin, broad = _carrier_proto(data, carrier)
                carrier_desc = _carrier_descriptor(data, carrier)
                scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
                for hist in candidate_histories:
                    scores = _score_history(proto, margin, broad, carrier_desc, hist, int(chunk), variant=variant, rng=rng, args=args)
                    candidate_eval_count += 1
                    semantic_gate_suppressed += int(_float(scores.get("semantic_gate_suppressed"), 0.0) > 0.0)
                    if scores["q_score"] > 0.0:
                        scored.append((scores["q_score"], hist, scores))
                scored.sort(key=lambda item: item[0], reverse=True)
                kept = scored[: int(args.history_top_k)]
                entropy = _q_entropy([item[0] for item in kept])
                entropies.append(entropy)
                raw_top1 = float(kept[0][0]) if kept else 0.0
                raw_top1_scores.append(raw_top1)
                raw_top1_positive += int(raw_top1 > 0.0)
                raw_qmin_pass += int(raw_top1 >= float(args.history_q_min_score))
                raw_entropy_pass += int(entropy <= float(args.history_entropy_upper_bound))
                if kept and kept[0][0] >= float(args.history_q_min_score) and entropy <= float(args.history_entropy_upper_bound):
                    covered += 1
                    top_hist = kept[0][1]
                    state = str(top_hist.get("state"))
                    confirmed_used += int(state == "confirmed")
                    tentative_used += int(state == "tentative")
                    quarantine_used += int(state == "quarantine")
                    top1_scores.append(float(kept[0][0]))
                    if _float(kept[0][2].get("semantic_match_for_method"), 0.0) <= 0.0:
                        false_attach_proxy += 1
                else:
                    no_anchor += 1
                if variant == "Q4_full_carrier_to_history_affinity":
                    for rank, (score, hist, scores) in enumerate(kept, start=1):
                        q_rows.append(
                            {
                                "scene_id": scene,
                                "chunk_id": chunk,
                                "carrier_id": carrier,
                                "history_id": hist.get("history_id"),
                                "history_version_id": hist.get("version_id"),
                                "history_state": hist.get("state"),
                                "history_source_chunk_id": hist.get("source_chunk_id"),
                                "history_last_seen_chunk": hist.get("last_seen_chunk"),
                                "history_age_chunks": int(chunk) - _int(
                                    hist.get("last_seen_chunk"),
                                    _int(hist.get("source_chunk_id"), int(chunk)),
                                ),
                                "history_snapshot_chunk_id": hist.get("snapshot_chunk_id", ""),
                                "history_snapshot_memory_mode": hist.get("snapshot_memory_mode", ""),
                                "carrier_semantic_proto_id": proto,
                                "history_semantic_proto_id": hist.get("semantic_proto_id"),
                                "q_score": score,
                                "semantic_score": scores["semantic_score"],
                                "semantic_match_for_method": scores["semantic_match_for_method"],
                                "appearance_score": scores["appearance_score"],
                                "carrier_sketch_score": scores["carrier_sketch_score"],
                                "temporal_score": scores["temporal_score"],
                                "spatial_score": scores["spatial_score"],
                                "history_state_weight": scores["history_state_weight"],
                                "scale_score": scores["scale_score"],
                                "conflict_score": scores["conflict_score"],
                                "semantic_residual_floor_score": scores["semantic_residual_floor_score"],
                                "history_full_score_calibration": scores["history_full_score_calibration"],
                                "assignment_entropy": entropy,
                                "top_rank": rank,
                                "allowed_for_method": str(hist.get("state")) == "confirmed" and entropy <= float(args.history_entropy_upper_bound),
                            }
                        )
            coverage = _safe_ratio(covered, carrier_count)
            confirmed_rate = _safe_ratio(confirmed_used, max(1, covered))
            tentative_rate = _safe_ratio(tentative_used, max(1, covered))
            quarantine_rate = _safe_ratio(quarantine_used, max(1, covered))
            false_attach_rate = _safe_ratio(false_attach_proxy, max(1, covered))
            no_anchor_rate = _safe_ratio(no_anchor, carrier_count)
            row = {
                "variant": variant,
                "scene_id": scene,
                "chunk_id": chunk,
                "after_warmup": after_warmup,
                "raw_history_candidate_count": len(raw_histories),
                "future_history_descriptor_candidate_count": len(future_descriptor_rows),
                "future_history_descriptor_filtered_count": len(future_descriptor_rows)
                if bool(getattr(args, "history_require_causal_last_seen", False))
                else 0,
                "history_causal_last_seen_required": bool(getattr(args, "history_require_causal_last_seen", False)),
                "history_use_causal_snapshots": bool(snapshot_histories),
                "history_candidate_count": len(candidate_histories),
                "Q_coverage_rate": coverage,
                "Q_top1_confidence_mean": _mean(top1_scores) or 0.0,
                "Q_entropy_mean": _mean(entropies) or 0.0,
                "confirmed_anchor_usage_rate": confirmed_rate,
                "tentative_anchor_usage_rate": tentative_rate,
                "quarantine_anchor_usage_rate": quarantine_rate,
                "false_attachment_proxy_rate": false_attach_rate,
                "new_object_no_anchor_rate": no_anchor_rate,
                "raw_top1_score_mean": _mean(raw_top1_scores) or 0.0,
                "raw_top1_positive_rate": _safe_ratio(raw_top1_positive, carrier_count),
                "raw_qmin_pass_rate": _safe_ratio(raw_qmin_pass, carrier_count),
                "raw_entropy_pass_rate": _safe_ratio(raw_entropy_pass, carrier_count),
                "semantic_gate_candidate_suppressed_count": semantic_gate_suppressed,
                "semantic_gate_candidate_suppressed_rate": _safe_ratio(semantic_gate_suppressed, max(1, candidate_eval_count)),
                "carrier_to_history_runtime_sec": "",
            }
            control_rows.append(row)
            for key, value in row.items():
                if key in {"variant", "scene_id", "chunk_id", "after_warmup"}:
                    continue
                if isinstance(value, (int, float)):
                    variant_metric[variant][key].append(float(value))

    _write_csv(output_root / "q_rows.csv", q_rows)
    _write_csv(output_root / "q_control_rows.csv", control_rows)
    full = variant_metric["Q4_full_carrier_to_history_affinity"]
    semantic = variant_metric["Q0_semantic_only_history_affinity"]
    appearance = variant_metric["Q1_appearance_only_history_affinity"]
    carrier_suffix_diag = variant_metric["Q2_carrier_suffix_diagnostic"]
    shuffled = variant_metric["Q4_full_carrier_to_history_affinity_shuffled"]
    stale = variant_metric["Q4_full_carrier_to_history_affinity_stale"]
    q_cov = _mean(full["Q_coverage_rate"]) or 0.0
    q_conf = _mean(full["Q_top1_confidence_mean"]) or 0.0
    q_ent = _mean(full["Q_entropy_mean"]) or 0.0
    confirmed_rate = _mean(full["confirmed_anchor_usage_rate"]) or 0.0
    quarantine_rate = _mean(full["quarantine_anchor_usage_rate"]) or 0.0
    semantic_conf = _mean(semantic["Q_top1_confidence_mean"]) or 0.0
    appearance_cov = _mean(appearance["Q_coverage_rate"]) or 0.0
    appearance_conf = _mean(appearance["Q_top1_confidence_mean"]) or 0.0
    appearance_raw_top1 = _mean(appearance["raw_top1_score_mean"]) or 0.0
    appearance_raw_positive = _mean(appearance["raw_top1_positive_rate"]) or 0.0
    appearance_raw_qmin = _mean(appearance["raw_qmin_pass_rate"]) or 0.0
    carrier_suffix_diag_cov = _mean(carrier_suffix_diag["Q_coverage_rate"]) or 0.0
    carrier_suffix_diag_conf = _mean(carrier_suffix_diag["Q_top1_confidence_mean"]) or 0.0
    carrier_suffix_diag_confirmed = _mean(carrier_suffix_diag["confirmed_anchor_usage_rate"]) or 0.0
    shuffled_conf = _mean(shuffled["Q_top1_confidence_mean"]) or 0.0
    stale_conf = _mean(stale["Q_top1_confidence_mean"]) or 0.0
    full_minus_semantic = q_conf - semantic_conf
    full_minus_shuffled = q_conf - shuffled_conf
    full_minus_stale = q_conf - stale_conf
    semantic_gate_suppressed_rate = _mean(full["semantic_gate_candidate_suppressed_rate"]) or 0.0
    future_descriptor_candidate_count = sum(full["future_history_descriptor_candidate_count"])
    future_descriptor_filtered_count = sum(full["future_history_descriptor_filtered_count"])
    gate = {
        "Q_coverage_rate_ge_0p70": q_cov >= 0.70,
        "confirmed_anchor_usage_rate_ge_0p50": confirmed_rate >= 0.50,
        "quarantine_anchor_usage_rate_le_0p05": quarantine_rate <= 0.05,
        "Q_entropy_mean_le_bound": q_ent <= float(args.history_entropy_upper_bound),
        "full_minus_semantic_top1_ge_margin": full_minus_semantic >= float(args.history_residual_margin),
        "full_minus_shuffled_top1_ge_margin": full_minus_shuffled >= float(args.history_residual_margin),
        "full_minus_stale_top1_ge_margin": full_minus_stale >= float(args.history_residual_margin),
    }
    gate["pass"] = all(gate.values())
    summary = _phase_common(
        phase="v81_phase3_carrier_to_history",
        schema="stream4d_v81_phase3_q_v1",
        decision="PASS_V81_PHASE3_CARRIER_TO_HISTORY_Q" if gate["pass"] else "NO_GO_CARRIER_TO_HISTORY_AFFINITY_WEAK",
        carrier_id_scope=args.carrier_id_scope_effective,
        history_anchor_type="confirmed_tentative_bootstrap_history",
        history_method_mode_allowed=gate["pass"],
        can_enter_next_phase=gate["pass"],
        can_enter_local2history=False,
        primary_blocker="" if gate["pass"] else "carrier_to_history_not_above_controls",
        secondary_blocker="" if args.carrier_id_scope_effective == "online_streaming" else "carrier_sketch_removed_from_method_non_online_scope",
        extra={
            "Q_coverage_rate": q_cov,
            "Q_top1_confidence_mean": q_conf,
            "Q_entropy_mean": q_ent,
            "confirmed_anchor_usage_rate": confirmed_rate,
            "tentative_anchor_usage_rate": _mean(full["tentative_anchor_usage_rate"]) or 0.0,
            "quarantine_anchor_usage_rate": quarantine_rate,
            "false_attachment_proxy_rate": _mean(full["false_attachment_proxy_rate"]) or 0.0,
            "new_object_no_anchor_rate": _mean(full["new_object_no_anchor_rate"]) or 0.0,
            "semantic_only_Q_top1_confidence_mean": semantic_conf,
            "appearance_only_Q_coverage_rate": appearance_cov,
            "appearance_only_Q_top1_confidence_mean": appearance_conf,
            "appearance_only_raw_top1_score_mean": appearance_raw_top1,
            "appearance_only_raw_top1_positive_rate": appearance_raw_positive,
            "appearance_only_raw_qmin_pass_rate": appearance_raw_qmin,
            "carrier_suffix_diagnostic_Q_coverage_rate": carrier_suffix_diag_cov,
            "carrier_suffix_diagnostic_Q_top1_confidence_mean": carrier_suffix_diag_conf,
            "carrier_suffix_diagnostic_confirmed_anchor_usage_rate": carrier_suffix_diag_confirmed,
            "shuffled_Q_top1_confidence_mean": shuffled_conf,
            "stale_Q_top1_confidence_mean": stale_conf,
            "full_minus_semantic_top1_confidence": full_minus_semantic,
            "full_minus_shuffled_top1_confidence": full_minus_shuffled,
            "full_minus_stale_top1_confidence": full_minus_stale,
            "semantic_match_required_for_full_q": bool(args.history_full_require_semantic_match),
            "history_full_score_calibration": str(getattr(args, "history_full_score_calibration", "weighted")),
            "full_semantic_gate_candidate_suppressed_rate": semantic_gate_suppressed_rate,
            "history_causal_last_seen_required": bool(getattr(args, "history_require_causal_last_seen", False)),
            "history_use_causal_snapshots": bool(getattr(args, "history_use_causal_snapshots", False)),
            "future_history_descriptor_candidate_count": future_descriptor_candidate_count,
            "future_history_descriptor_filtered_count": future_descriptor_filtered_count,
            "same_instance_history_topk_recall_diagnostic": "",
            "wrong_history_top1_rate_diagnostic": "",
            "carrier_sketch_score_method_enabled": args.carrier_id_scope_effective == "online_streaming",
            "semantic_score_mode": args.history_semantic_score_mode,
            "semantic_match_threshold": args.history_semantic_match_threshold,
            "gate": gate,
            "runtime_sec": time.time() - started,
        },
    )
    _write_json(output_root / "q_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _phase3_q_assignments(args: argparse.Namespace) -> dict[tuple[str, int, str], dict[str, Any]]:
    path = ROOT / args.phase3_output_root / "q_rows.csv"
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    rows_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        key = (str(row.get("scene_id")), _int(row.get("chunk_id"), -1), str(row.get("carrier_id")))
        rows_by_key[key].append(row)

    min_margin = float(getattr(args, "history_assignment_min_q_margin", 0.0))
    min_ratio = float(getattr(args, "history_assignment_min_q_ratio", 0.0))
    min_appearance = float(getattr(args, "history_assignment_min_appearance", 0.0))
    min_spatial = float(getattr(args, "history_assignment_min_spatial", 0.0))
    max_conflict = float(getattr(args, "history_assignment_max_conflict", 1.0))
    max_age = int(getattr(args, "history_assignment_max_age_chunks", 0))
    require_causal_last_seen = bool(getattr(args, "history_require_causal_last_seen", False))
    for key, rows in rows_by_key.items():
        ranked = sorted(rows, key=lambda item: _int(item.get("top_rank"), 10**9))
        if not ranked:
            continue
        row = dict(ranked[0])
        if _int(row.get("top_rank"), 0) != 1:
            continue
        top_score = _float(row.get("q_score"), 0.0)
        second_score = _float(ranked[1].get("q_score"), 0.0) if len(ranked) > 1 else 0.0
        q_margin = top_score - second_score
        q_ratio = top_score / max(second_score, 1e-12)
        row["assignment_q_margin"] = q_margin
        row["assignment_q_ratio"] = q_ratio
        if _float(row.get("q_score"), 0.0) < float(args.history_q_min_score):
            continue
        if _float(row.get("assignment_entropy"), 0.0) > float(args.history_entropy_upper_bound):
            continue
        if q_margin < min_margin:
            continue
        if min_ratio > 0.0 and q_ratio < min_ratio:
            continue
        if _float(row.get("appearance_score"), 0.0) < min_appearance:
            continue
        if _float(row.get("spatial_score"), 0.0) < min_spatial:
            continue
        if _float(row.get("conflict_score"), 0.0) > max_conflict:
            continue
        age = _int(row.get("history_age_chunks"), 1)
        if require_causal_last_seen and age <= 0:
            continue
        if max_age > 0 and age > max_age:
            continue
        if str(row.get("history_state")) != "confirmed":
            continue
        if _float(row.get("semantic_match_for_method"), 1.0) <= 0.0:
            continue
        out[key] = row
    return out


def _history_anchor_edges(
    carriers: list[str],
    assignments: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    variant: str,
    data: dict[str, Any],
    rng: random.Random,
    local_labels: list[int] | None = None,
) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]], dict[str, Any], dict[str, list[tuple[str, dict[str, Any]]]]]:
    idx = {carrier: i for i, carrier in enumerate(carriers)}
    local_label_by_carrier: dict[str, int] = {}
    local_component_sizes: Counter[int] = Counter()
    if local_labels is not None:
        for carrier, label in zip(carriers, local_labels):
            local_label_by_carrier[carrier] = int(label)
            local_component_sizes[int(label)] += 1
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    if variant == "B4_shuffled_history_anchors":
        values = [str(row.get("history_id")) for row in assignments.values()]
        rng.shuffle(values)
        for (carrier, row), history_id in zip(sorted(assignments.items()), values):
            new_row = dict(row)
            new_row["history_id"] = history_id
            grouped[history_id].append((carrier, new_row))
    elif variant == "B5_stale_history_anchors":
        stale_min_age = int(getattr(args, "history_stale_control_min_age_chunks", 2))
        for carrier, row in assignments.items():
            if _int(row.get("history_age_chunks"), 0) < stale_min_age:
                continue
            grouped[str(row.get("history_id"))].append((carrier, row))
    elif variant == "B6_semantic_only_history_anchors":
        for carrier in carriers:
            proto, _margin, _broad = _carrier_proto(data, carrier)
            if proto:
                grouped[proto].append((carrier, {"history_id": proto, "q_score": 1.0, "semantic_score": 1.0}))
    else:
        for carrier, row in assignments.items():
            grouped[str(row.get("history_id"))].append((carrier, row))

    edge_rows: list[dict[str, Any]] = []
    edges: list[tuple[int, int, float]] = []
    base = float(args.signed_threshold)
    eta = float(args.history_attraction_eta)
    min_q = float(getattr(args, "history_attraction_min_q_score", 0.0))
    top_per_group = int(getattr(args, "history_attraction_top_per_group", 0))
    max_group_size = int(getattr(args, "history_attraction_max_group_size", 0))
    edge_mode = str(getattr(args, "history_attraction_edge_mode", "star"))
    pair_top_k = max(1, int(getattr(args, "history_attraction_pair_top_k", 1)))
    min_pair_appearance = float(getattr(args, "history_attraction_min_pair_appearance", 0.0))
    min_pair_spatial = float(getattr(args, "history_attraction_min_pair_spatial", 0.0))
    require_diff_local = bool(getattr(args, "history_attraction_require_different_local_components", False))
    max_local_component = int(getattr(args, "history_attraction_max_local_component_size", 0))
    stats = {
        "history_anchor_candidate_group_count": len(grouped),
        "history_anchor_candidate_assignment_count": sum(len(group) for group in grouped.values()),
        "history_anchor_selected_group_count": 0,
        "history_anchor_selected_assignment_count": 0,
        "history_anchor_edge_count": 0,
        "history_anchor_low_q_filtered_count": 0,
        "history_anchor_large_group_filtered_count": 0,
        "history_attraction_min_q_score": min_q,
        "history_attraction_top_per_group": top_per_group,
        "history_attraction_max_group_size": max_group_size,
        "history_attraction_edge_mode": edge_mode,
        "history_attraction_pair_top_k": pair_top_k,
        "history_attraction_min_pair_appearance": min_pair_appearance,
        "history_attraction_min_pair_spatial": min_pair_spatial,
        "history_attraction_require_different_local_components": require_diff_local,
        "history_attraction_max_local_component_size": max_local_component,
        "history_anchor_local_component_filtered_count": 0,
        "history_stale_control_min_age_chunks": int(getattr(args, "history_stale_control_min_age_chunks", 2)),
        "history_assignment_min_q_margin": float(getattr(args, "history_assignment_min_q_margin", 0.0)),
        "history_assignment_min_q_ratio": float(getattr(args, "history_assignment_min_q_ratio", 0.0)),
        "history_assignment_min_appearance": float(getattr(args, "history_assignment_min_appearance", 0.0)),
        "history_assignment_min_spatial": float(getattr(args, "history_assignment_min_spatial", 0.0)),
        "history_assignment_max_conflict": float(getattr(args, "history_assignment_max_conflict", 1.0)),
        "history_assignment_max_age_chunks": int(getattr(args, "history_assignment_max_age_chunks", 0)),
        "history_require_causal_last_seen": bool(getattr(args, "history_require_causal_last_seen", False)),
    }
    selected_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for history_id, group in sorted(grouped.items()):
        ranked_all = sorted(group, key=lambda item: _float(item[1].get("q_score"), 0.0), reverse=True)
        ranked = [item for item in ranked_all if _float(item[1].get("q_score"), 0.0) >= min_q]
        stats["history_anchor_low_q_filtered_count"] += len(ranked_all) - len(ranked)
        if max_group_size > 0 and len(ranked) > max_group_size:
            stats["history_anchor_large_group_filtered_count"] += len(ranked)
            continue
        if top_per_group > 0:
            ranked = ranked[:top_per_group]
        if ranked:
            selected_groups[str(history_id)] = ranked
        if len(ranked) < 2:
            continue
        stats["history_anchor_selected_group_count"] += 1
        stats["history_anchor_selected_assignment_count"] += len(ranked)
        pair_rows: list[tuple[str, dict[str, Any], str, dict[str, Any], float, float, float]] = []
        if edge_mode == "appearance_knn":
            desc_cache = {carrier: _carrier_descriptor(data, carrier) for carrier, _row in ranked}
            pair_seen: set[tuple[str, str]] = set()
            for carrier, row in ranked:
                left = desc_cache.get(carrier, {})
                scored: list[tuple[float, str, dict[str, Any], float, float, float]] = []
                for other, other_row in ranked:
                    if other == carrier:
                        continue
                    right = desc_cache.get(other, {})
                    pair_appearance = _appearance_cosine(left.get("appearance_vector"), right.get("appearance_vector"))
                    pair_dist = _uv_distance(left, right)
                    pair_spatial = math.exp(-0.5 * (pair_dist / max(1e-6, float(args.history_spatial_sigma))) ** 2)
                    if pair_appearance < min_pair_appearance:
                        continue
                    if pair_spatial < min_pair_spatial:
                        continue
                    pair_q = min(_float(row.get("q_score"), 0.0), _float(other_row.get("q_score"), 0.0))
                    attraction = pair_q * (0.75 * pair_appearance + 0.25 * pair_spatial)
                    scored.append((attraction, other, other_row, pair_appearance, pair_spatial, pair_q))
                for attraction, other, other_row, pair_appearance, pair_spatial, _pair_q in sorted(scored, reverse=True)[:pair_top_k]:
                    pair_key = tuple(sorted((carrier, other)))
                    if pair_key in pair_seen:
                        continue
                    pair_seen.add(pair_key)
                    pair_rows.append((carrier, row, other, other_row, attraction, pair_appearance, pair_spatial))
        else:
            rep_carrier, rep_row = ranked[0]
            for carrier, row in ranked[1:]:
                attraction = min(_float(rep_row.get("q_score"), 0.0), _float(row.get("q_score"), 0.0))
                pair_rows.append((rep_carrier, rep_row, carrier, row, attraction, 0.0, 0.0))

        for carrier_i, row_i, carrier_j, row_j, attraction, pair_appearance, pair_spatial in pair_rows:
            i = idx.get(carrier_i)
            j = idx.get(carrier_j)
            if i is None or j is None or i == j:
                continue
            label_i = local_label_by_carrier.get(carrier_i)
            label_j = local_label_by_carrier.get(carrier_j)
            if require_diff_local and label_i is not None and label_j is not None and label_i == label_j:
                stats["history_anchor_local_component_filtered_count"] += 1
                continue
            if max_local_component > 0 and label_i is not None and label_j is not None:
                if local_component_sizes[label_i] > max_local_component or local_component_sizes[label_j] > max_local_component:
                    stats["history_anchor_local_component_filtered_count"] += 1
                    continue
            score = base + eta * attraction
            i0, j0 = sorted((i, j))
            edges.append((i0, j0, score))
            edge_rows.append(
                {
                    "carrier_i": carrier_i,
                    "carrier_j": carrier_j,
                    "history_id": history_id,
                    "history_attraction": attraction,
                    "history_conflict": 0.0,
                    "history_edge_score": score,
                    "history_edge_mode": edge_mode,
                    "history_pair_appearance": pair_appearance,
                    "history_pair_spatial": pair_spatial,
                    "variant": variant,
                }
            )
    stats["history_anchor_edge_count"] = len(edges)
    return edges, edge_rows, stats, selected_groups


def _split_labels_by_history_conflict(
    base_labels: list[int],
    carriers: list[str],
    selected_groups: dict[str, list[tuple[str, dict[str, Any]]]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    mode = str(getattr(args, "history_conflict_split_mode", "none"))
    stats = {
        "history_conflict_split_mode": mode,
        "history_conflict_min_q_score": float(getattr(args, "history_conflict_min_q_score", 0.0)),
        "history_conflict_min_history_group_size": int(getattr(args, "history_conflict_min_history_group_size", 1)),
        "history_conflict_split_component_count": 0,
        "history_conflict_split_carrier_count": 0,
        "history_conflict_split_group_count": 0,
        "history_conflict_residual_carrier_count": 0,
    }
    if mode == "none":
        return base_labels, stats

    carrier_to_history: dict[str, str] = {}
    min_q = float(getattr(args, "history_conflict_min_q_score", 0.0))
    for history_id, group in selected_groups.items():
        for carrier, row in group:
            if _float(row.get("q_score"), 0.0) < min_q:
                continue
            carrier_to_history[str(carrier)] = str(history_id)

    label_to_indices = _label_to_indices(base_labels)
    min_group = max(1, int(getattr(args, "history_conflict_min_history_group_size", 1)))
    keys_by_idx: list[tuple[int, str, str] | None] = [None for _ in base_labels]
    for label, indices in sorted(label_to_indices.items()):
        history_counts: Counter[str] = Counter()
        for idx in indices:
            history_id = carrier_to_history.get(str(carriers[int(idx)]))
            if history_id:
                history_counts[history_id] += 1
        valid_histories = {history_id for history_id, count in history_counts.items() if count >= min_group}
        if len(valid_histories) < 2:
            for idx in indices:
                keys_by_idx[int(idx)] = (int(label), "base", "")
            continue

        stats["history_conflict_split_component_count"] += 1
        stats["history_conflict_split_group_count"] += len(valid_histories)
        for idx in indices:
            history_id = carrier_to_history.get(str(carriers[int(idx)]))
            if history_id in valid_histories:
                keys_by_idx[int(idx)] = (int(label), "history", history_id)
                stats["history_conflict_split_carrier_count"] += 1
            else:
                keys_by_idx[int(idx)] = (int(label), "residual", "")
                stats["history_conflict_residual_carrier_count"] += 1

    remap: dict[tuple[int, str, str], int] = {}
    labels: list[int] = []
    for idx, key in enumerate(keys_by_idx):
        if key is None:
            key = (int(base_labels[idx]), "base", "")
        if key not in remap:
            remap[key] = len(remap) + 1
        labels.append(remap[key])
    return labels, stats


def _label_to_indices(labels: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        out[int(label)].append(idx)
    return dict(out)


def _variant_graph(graph: dict[str, Any], labels: list[int], variant: str) -> dict[str, Any]:
    return {
        **graph,
        "labels": labels,
        "label_to_indices": _label_to_indices(labels),
        "variant": variant,
    }


def _history_fused_metrics(labels: list[int], cannot_link: set[tuple[int, int]], carriers: list[str]) -> dict[str, Any]:
    groups = _label_to_indices(labels)
    return {
        "cluster_count": len(groups),
        "largest_cluster_ratio": _safe_ratio(max((len(v) for v in groups.values()), default=0), max(1, len(carriers))),
        "cannot_link_violation_count": v80._violation_count(labels, cannot_link),
    }


def _constrained_union_from_labels(
    base_labels: list[int],
    edges: list[tuple[int, int, float]],
    cannot_link: set[tuple[int, int]],
    args: argparse.Namespace,
) -> list[int]:
    n = len(base_labels)
    parent = list(range(n))
    groups = _label_to_indices(base_labels)
    for indices in groups.values():
        if not indices:
            continue
        root = int(indices[0])
        for idx in indices:
            parent[int(idx)] = root
    members: dict[int, set[int]] = {idx: set() for idx in range(n)}
    for idx in range(n):
        members[parent[idx]].add(idx)
    max_size = max(1, int(math.floor(float(args.max_component_ratio) * max(1, n))))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if len(members[ra]) + len(members[rb]) > max_size:
            return False
        left, right = members[ra], members[rb]
        if len(left) > len(right):
            left, right = right, left
        for i in left:
            for j in right:
                if (min(i, j), max(i, j)) in cannot_link:
                    return False
        return True

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].update(members[rb])
        members[rb].clear()

    for i, j, score in sorted(edges, key=lambda item: item[2], reverse=True):
        if score < float(args.signed_threshold):
            continue
        if can_union(int(i), int(j)):
            union(int(i), int(j))

    remap: dict[int, int] = {}
    labels: list[int] = []
    for idx in range(n):
        root = find(idx)
        if root not in remap:
            remap[root] = len(remap) + 1
        labels.append(remap[root])
    return labels


def _run_phase4(
    args: argparse.Namespace,
    phase3: dict[str, Any],
    clusters: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase4_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not bool((phase3.get("gate") or {}).get("pass")):
        summary = _blocked_phase(
            output_root,
            "v81_phase4_history_fused_clustering",
            "stream4d_v81_phase4_fused_clustering_v1",
            "BLOCK_HISTORY_FUSED_BY_Q_WEAK",
            args,
        )
        _write_csv(output_root / "fused_affinity_rows.csv", [])
        _write_csv(output_root / "cluster_rows.csv", [])
        _write_csv(output_root / "control_variant_rows.csv", [])
        return summary, {}

    q_by_key = _phase3_q_assignments(args)
    fused_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    rng = random.Random(int(args.random_seed) + 8104)
    out_graphs: dict[tuple[str, int], dict[str, Any]] = {}
    b0_largest: list[float] = []
    b3_largest: list[float] = []
    b3_violations: list[int] = []
    b3_anchor_usage: list[float] = []
    b4_largest: list[float] = []
    b5_largest: list[float] = []
    b6_largest: list[float] = []
    b3_selected_assignment_rates: list[float] = []
    b3_edge_counts: list[int] = []
    b3_low_q_filtered_counts: list[int] = []
    b3_large_group_filtered_counts: list[int] = []
    b3_local_component_filtered_counts: list[int] = []
    b3_conflict_split_component_counts: list[int] = []
    b3_conflict_split_carrier_counts: list[int] = []
    b3_conflict_split_group_counts: list[int] = []
    b3_conflict_residual_carrier_counts: list[int] = []

    for (scene, chunk), graph in sorted(clusters.items()):
        data = graph["data"]
        carriers = graph["carriers"]
        cannot_link = graph.get("cannot_link", set())
        local_edges = list(graph.get("signed_edges", []))
        local_labels = [int(value) for value in graph.get("labels", [])]
        local_metrics = _history_fused_metrics(local_labels, cannot_link, carriers)
        b0_largest.append(_float(local_metrics["largest_cluster_ratio"], 0.0))
        assignments = {
            carrier: row
            for carrier in carriers
            if (row := q_by_key.get((scene, int(chunk), carrier))) is not None
        }
        b3_anchor_usage.append(_safe_ratio(len(assignments), max(1, len(carriers))))

        variant_labels: dict[str, list[int]] = {"B0_local_only_baseline": local_labels}
        variant_edge_rows: dict[str, list[dict[str, Any]]] = {}
        variant_edge_stats: dict[str, dict[str, Any]] = {
            "B0_local_only_baseline": {
                "history_anchor_candidate_group_count": 0,
                "history_anchor_candidate_assignment_count": 0,
                "history_anchor_selected_group_count": 0,
                "history_anchor_selected_assignment_count": 0,
                "history_anchor_edge_count": 0,
                "history_anchor_low_q_filtered_count": 0,
                "history_anchor_large_group_filtered_count": 0,
                "history_attraction_min_q_score": float(getattr(args, "history_attraction_min_q_score", 0.0)),
                "history_attraction_top_per_group": int(getattr(args, "history_attraction_top_per_group", 0)),
                "history_attraction_max_group_size": int(getattr(args, "history_attraction_max_group_size", 0)),
                "history_attraction_edge_mode": str(getattr(args, "history_attraction_edge_mode", "star")),
                "history_attraction_pair_top_k": int(getattr(args, "history_attraction_pair_top_k", 1)),
                "history_attraction_min_pair_appearance": float(getattr(args, "history_attraction_min_pair_appearance", 0.0)),
                "history_attraction_min_pair_spatial": float(getattr(args, "history_attraction_min_pair_spatial", 0.0)),
                "history_attraction_require_different_local_components": bool(
                    getattr(args, "history_attraction_require_different_local_components", False)
                ),
                "history_attraction_max_local_component_size": int(getattr(args, "history_attraction_max_local_component_size", 0)),
                "history_anchor_local_component_filtered_count": 0,
                "history_conflict_split_mode": str(getattr(args, "history_conflict_split_mode", "none")),
                "history_conflict_min_q_score": float(getattr(args, "history_conflict_min_q_score", 0.0)),
                "history_conflict_min_history_group_size": int(getattr(args, "history_conflict_min_history_group_size", 1)),
                "history_conflict_split_component_count": 0,
                "history_conflict_split_carrier_count": 0,
                "history_conflict_split_group_count": 0,
                "history_conflict_residual_carrier_count": 0,
                "history_assignment_min_q_margin": float(getattr(args, "history_assignment_min_q_margin", 0.0)),
                "history_assignment_min_q_ratio": float(getattr(args, "history_assignment_min_q_ratio", 0.0)),
                "history_assignment_min_appearance": float(getattr(args, "history_assignment_min_appearance", 0.0)),
                "history_assignment_min_spatial": float(getattr(args, "history_assignment_min_spatial", 0.0)),
                "history_assignment_max_conflict": float(getattr(args, "history_assignment_max_conflict", 1.0)),
                "history_assignment_max_age_chunks": int(getattr(args, "history_assignment_max_age_chunks", 0)),
                "history_require_causal_last_seen": bool(getattr(args, "history_require_causal_last_seen", False)),
                "history_stale_control_min_age_chunks": int(getattr(args, "history_stale_control_min_age_chunks", 2)),
            }
        }
        for variant in [
            "B3_local_plus_history_attraction_conflict",
            "B4_shuffled_history_anchors",
            "B5_stale_history_anchors",
            "B6_semantic_only_history_anchors",
        ]:
            hist_edges, hist_rows, hist_stats, selected_groups = _history_anchor_edges(
                carriers,
                assignments,
                args,
                variant=variant,
                data=data,
                rng=rng,
                local_labels=local_labels,
            )
            labels = _constrained_union_from_labels(local_labels, hist_edges, cannot_link, args)
            labels, split_stats = _split_labels_by_history_conflict(labels, carriers, selected_groups, args)
            hist_stats.update(split_stats)
            variant_labels[variant] = labels
            variant_edge_rows[variant] = hist_rows
            variant_edge_stats[variant] = hist_stats

        variant_graphs = {
            variant: _variant_graph(graph, labels, variant)
            for variant, labels in variant_labels.items()
        }
        out_graphs[(scene, int(chunk))] = {
            **variant_graphs["B3_local_plus_history_attraction_conflict"],
            "history_anchor_assignments": assignments,
            "phase4_variant_graphs": variant_graphs,
        }

        for variant, labels in variant_labels.items():
            metrics = _history_fused_metrics(labels, cannot_link, carriers)
            if variant == "B3_local_plus_history_attraction_conflict":
                b3_largest.append(_float(metrics["largest_cluster_ratio"], 0.0))
                b3_violations.append(_int(metrics["cannot_link_violation_count"], 0))
                stats = variant_edge_stats.get(variant, {})
                b3_selected_assignment_rates.append(
                    _safe_ratio(_int(stats.get("history_anchor_selected_assignment_count"), 0), max(1, len(carriers)))
                )
                b3_edge_counts.append(_int(stats.get("history_anchor_edge_count"), 0))
                b3_low_q_filtered_counts.append(_int(stats.get("history_anchor_low_q_filtered_count"), 0))
                b3_large_group_filtered_counts.append(_int(stats.get("history_anchor_large_group_filtered_count"), 0))
                b3_local_component_filtered_counts.append(_int(stats.get("history_anchor_local_component_filtered_count"), 0))
                b3_conflict_split_component_counts.append(_int(stats.get("history_conflict_split_component_count"), 0))
                b3_conflict_split_carrier_counts.append(_int(stats.get("history_conflict_split_carrier_count"), 0))
                b3_conflict_split_group_counts.append(_int(stats.get("history_conflict_split_group_count"), 0))
                b3_conflict_residual_carrier_counts.append(_int(stats.get("history_conflict_residual_carrier_count"), 0))
            elif variant == "B4_shuffled_history_anchors":
                b4_largest.append(_float(metrics["largest_cluster_ratio"], 0.0))
            elif variant == "B5_stale_history_anchors":
                b5_largest.append(_float(metrics["largest_cluster_ratio"], 0.0))
            elif variant == "B6_semantic_only_history_anchors":
                b6_largest.append(_float(metrics["largest_cluster_ratio"], 0.0))
            groups = _label_to_indices(labels)
            for label, indices in sorted(groups.items()):
                cluster_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "variant": variant,
                        "cluster_id": label,
                        "carrier_count": len(indices),
                        "largest_cluster_ratio": metrics["largest_cluster_ratio"],
                        "cannot_link_violation_count": metrics["cannot_link_violation_count"],
                    }
                )
            control_rows.append(
                {
                    "variant": variant,
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "cluster_count": metrics["cluster_count"],
                    "largest_cluster_ratio": metrics["largest_cluster_ratio"],
                    "cannot_link_violation_count": metrics["cannot_link_violation_count"],
                    "history_anchor_usage_rate": _safe_ratio(len(assignments), max(1, len(carriers))),
                    "local_SF50_after_adapter": "",
                    "within_semantic_instance_AUC": "",
                    "same_semantic_different_instance_separation": "",
                    "overmerge_rate": metrics["largest_cluster_ratio"],
                    "new_object_birth_candidate_count": sum(1 for carrier in carriers if carrier not in assignments),
                    **variant_edge_stats.get(variant, {}),
                }
            )

        idx = {carrier: i for i, carrier in enumerate(carriers)}
        for variant, rows in variant_edge_rows.items():
            labels = variant_labels[variant]
            for row in rows:
                i = idx.get(str(row["carrier_i"]))
                j = idx.get(str(row["carrier_j"]))
                if i is None or j is None:
                    continue
                pair = (min(i, j), max(i, j))
                fused_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "carrier_i": row["carrier_i"],
                        "carrier_j": row["carrier_j"],
                        "scale": "object",
                        "local_affinity": "",
                        "history_attraction": row["history_attraction"],
                        "history_conflict": row["history_conflict"],
                        "history_edge_mode": row.get("history_edge_mode", ""),
                        "history_pair_appearance": row.get("history_pair_appearance", ""),
                        "history_pair_spatial": row.get("history_pair_spatial", ""),
                        "current_cannot": pair in cannot_link,
                        "fused_affinity": row["history_edge_score"],
                        "edge_selected": labels[i] == labels[j],
                        "cluster_i": labels[i],
                        "cluster_j": labels[j],
                        "constraint_violation_flag": pair in cannot_link and labels[i] == labels[j],
                        "variant": variant,
                    }
                )

    b0_largest_mean = _mean(b0_largest) or 0.0
    b3_largest_mean = _mean(b3_largest) or 0.0
    b4_largest_mean = _mean(b4_largest) or 0.0
    b5_largest_mean = _mean(b5_largest) or 0.0
    b6_largest_mean = _mean(b6_largest) or 0.0
    structural_gate = {
        "B3_cannot_link_violation_count_eq_0": sum(b3_violations) == 0,
        "B3_overmerge_rate_le_B0_plus_0p02": b3_largest_mean <= b0_largest_mean + 0.02,
        "B3_largest_cluster_ratio_le_0p25": b3_largest_mean <= 0.25,
    }
    phase4_full_gate = {
        **structural_gate,
        "B3_local_SF50_after_adapter_available": False,
        "B3_within_semantic_instance_AUC_available": False,
        "B3_beats_shuffled_history_SF50_available": False,
    }
    structural_gate["pass"] = all(structural_gate.values())
    phase4_full_gate["pass"] = all(phase4_full_gate.values())
    summary = _phase_common(
        phase="v81_phase4_history_fused_clustering",
        schema="stream4d_v81_phase4_fused_clustering_v1",
        decision="PASS_V81_PHASE4_STRUCTURAL_FUSED_CLUSTERING_ADAPTER_PENDING"
        if structural_gate["pass"]
        else "NO_GO_PHASE4_STRUCTURAL_FUSED_CLUSTERING_WEAK",
        carrier_id_scope=args.carrier_id_scope_effective,
        history_anchor_type="confirmed_phase3_q_anchors",
        history_method_mode_allowed=structural_gate["pass"],
        can_enter_next_phase=structural_gate["pass"],
        can_enter_local2history=False,
        primary_blocker="" if structural_gate["pass"] else "phase4_structural_fused_clustering_weak",
        secondary_blocker=""
        if str(getattr(args, "history_conflict_split_mode", "none")) != "none"
        else "history_cannot_link_graph_not_implemented",
        extra={
            "B0_largest_cluster_ratio": b0_largest_mean,
            "B3_largest_cluster_ratio": b3_largest_mean,
            "B4_shuffled_largest_cluster_ratio": b4_largest_mean,
            "B5_stale_largest_cluster_ratio": b5_largest_mean,
            "B6_semantic_only_largest_cluster_ratio": b6_largest_mean,
            "B3_cannot_link_violation_count": sum(b3_violations),
            "history_anchor_usage_rate": _mean(b3_anchor_usage) or 0.0,
            "B3_history_anchor_selected_assignment_rate": _mean(b3_selected_assignment_rates) or 0.0,
            "B3_history_anchor_edge_count": sum(b3_edge_counts),
            "B3_history_anchor_low_q_filtered_count": sum(b3_low_q_filtered_counts),
            "B3_history_anchor_large_group_filtered_count": sum(b3_large_group_filtered_counts),
            "B3_history_anchor_local_component_filtered_count": sum(b3_local_component_filtered_counts),
            "B3_history_conflict_split_component_count": sum(b3_conflict_split_component_counts),
            "B3_history_conflict_split_carrier_count": sum(b3_conflict_split_carrier_counts),
            "B3_history_conflict_split_group_count": sum(b3_conflict_split_group_counts),
            "B3_history_conflict_residual_carrier_count": sum(b3_conflict_residual_carrier_counts),
            "history_attraction_min_q_score": float(getattr(args, "history_attraction_min_q_score", 0.0)),
            "history_attraction_top_per_group": int(getattr(args, "history_attraction_top_per_group", 0)),
            "history_attraction_max_group_size": int(getattr(args, "history_attraction_max_group_size", 0)),
            "history_attraction_edge_mode": str(getattr(args, "history_attraction_edge_mode", "star")),
            "history_attraction_pair_top_k": int(getattr(args, "history_attraction_pair_top_k", 1)),
            "history_attraction_min_pair_appearance": float(getattr(args, "history_attraction_min_pair_appearance", 0.0)),
            "history_attraction_min_pair_spatial": float(getattr(args, "history_attraction_min_pair_spatial", 0.0)),
            "history_attraction_require_different_local_components": bool(
                getattr(args, "history_attraction_require_different_local_components", False)
            ),
            "history_attraction_max_local_component_size": int(getattr(args, "history_attraction_max_local_component_size", 0)),
            "history_conflict_split_mode": str(getattr(args, "history_conflict_split_mode", "none")),
            "history_conflict_min_q_score": float(getattr(args, "history_conflict_min_q_score", 0.0)),
            "history_conflict_min_history_group_size": int(getattr(args, "history_conflict_min_history_group_size", 1)),
            "history_assignment_min_q_margin": float(getattr(args, "history_assignment_min_q_margin", 0.0)),
            "history_assignment_min_q_ratio": float(getattr(args, "history_assignment_min_q_ratio", 0.0)),
            "history_assignment_min_appearance": float(getattr(args, "history_assignment_min_appearance", 0.0)),
            "history_assignment_min_spatial": float(getattr(args, "history_assignment_min_spatial", 0.0)),
            "history_assignment_max_conflict": float(getattr(args, "history_assignment_max_conflict", 1.0)),
            "history_assignment_max_age_chunks": int(getattr(args, "history_assignment_max_age_chunks", 0)),
            "history_require_causal_last_seen": bool(getattr(args, "history_require_causal_last_seen", False)),
            "history_stale_control_min_age_chunks": int(getattr(args, "history_stale_control_min_age_chunks", 2)),
            "local_SF50_after_adapter": "",
            "within_semantic_instance_AUC": "",
            "structural_gate": structural_gate,
            "phase4_full_gate": phase4_full_gate,
            "gate": structural_gate,
            "runtime_sec": time.time() - started,
        },
    )
    _write_csv(output_root / "fused_affinity_rows.csv", fused_rows)
    _write_csv(output_root / "cluster_rows.csv", cluster_rows)
    _write_csv(output_root / "control_variant_rows.csv", control_rows)
    _write_json(output_root / "cluster_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, out_graphs


def _variant_slug(name: str) -> str:
    return (
        name.lower()
        .replace("b0_", "b0_")
        .replace("b3_", "b3_")
        .replace("b4_", "b4_")
        .replace("b6_", "b6_")
        .replace("+", "plus")
        .replace("-", "_")
    )


def _run_phase5(
    args: argparse.Namespace,
    phase4: dict[str, Any],
    fused_clusters: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase5_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not bool((phase4.get("gate") or {}).get("pass")):
        summary = _blocked_phase(
            output_root,
            "v81_phase5_adapter_materialization",
            "stream4d_v81_phase5_adapter_v1",
            "BLOCK_ADAPTER_BY_PHASE4_STRUCTURAL_WEAK",
            args,
        )
        _write_csv(output_root / "adapter_rows.csv", [])
        _write_csv(output_root / "local_metric_rows.csv", [])
        _write_csv(output_root / "adapter_identity_audit_rows.csv", [])
        return summary

    variants = [
        "B0_local_only_baseline",
        "B3_local_plus_history_attraction_conflict",
        "B4_shuffled_history_anchors",
        "B5_stale_history_anchors",
        "B6_semantic_only_history_anchors",
    ]
    summaries: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_clusters: dict[tuple[str, int], dict[str, Any]] = {}
        for key, graph in fused_clusters.items():
            vgraphs = graph.get("phase4_variant_graphs", {})
            if variant in vgraphs:
                variant_clusters[key] = vgraphs[variant]
        variant_root = f"{args.phase5_output_root}/{_variant_slug(variant)}"
        variant_args = _copy_namespace(args, phase5_output_root=variant_root)
        summary, _eval_by_chunk = v80._run_phase5(variant_args, variant_clusters)
        summaries[variant] = summary
        variant_dir = ROOT / variant_root
        for row in _read_csv_rows(variant_dir / "local_metric_rows.csv"):
            metric_rows.append({**row, "variant": variant})
        for row in _read_csv_rows(variant_dir / "adapter_rows.csv"):
            adapter_rows.append({**row, "variant": variant})
        for row in _read_csv_rows(variant_dir / "local_slot_rows.csv"):
            slot_rows.append({**row, "variant": variant})
        identity_rows.append(
            {
                "variant": variant,
                "cluster_identity_fixed_before_adapter": True,
                "adapter_split_or_merge_violation_count": summary.get("adapter_split_or_merge_violation_count", 0),
                "adapter_identity_flip_rate": summary.get("adapter_identity_flip_rate", 0.0),
                "adapter_multi_object_materialization_rate": summary.get("adapter_multi_object_materialization_rate", 0.0),
                "source_adapter_summary": _rel(variant_dir / "summary.json"),
            }
        )

    def metric(variant: str, name: str) -> float:
        return _float(summaries.get(variant, {}).get(name), 0.0)

    b0_sf50 = metric("B0_local_only_baseline", "local_SF50_rendered_adapter")
    b3_sf50 = metric("B3_local_plus_history_attraction_conflict", "local_SF50_rendered_adapter")
    b4_sf50 = metric("B4_shuffled_history_anchors", "local_SF50_rendered_adapter")
    b5_sf50 = metric("B5_stale_history_anchors", "local_SF50_rendered_adapter")
    b6_sf50 = metric("B6_semantic_only_history_anchors", "local_SF50_rendered_adapter")
    b3_selection_gate = (summaries.get("B3_local_plus_history_attraction_conflict", {}).get("gate") or {})
    gate = {
        "B3_cluster_identity_fixed_before_adapter": True,
        "B3_adapter_split_or_merge_violation_count_eq_0": _int(
            summaries.get("B3_local_plus_history_attraction_conflict", {}).get("adapter_split_or_merge_violation_count"),
            0,
        )
        == 0,
        "B3_local_SF50_ge_B0_plus_0p03": b3_sf50 >= b0_sf50 + 0.03,
        "B3_local_SF50_ge_B4_plus_0p03": b3_sf50 >= b4_sf50 + 0.03,
        "B3_local_SF50_ge_B5_plus_0p03": b3_sf50 >= b5_sf50 + 0.03,
        "B3_local_SF50_ge_B6_plus_0p02": b3_sf50 >= b6_sf50 + 0.02,
        "B3_adapter_selection_gate_pass": bool(b3_selection_gate.get("pass")),
    }
    gate["pass"] = all(gate.values())
    summary = _phase_common(
        phase="v81_phase5_adapter_materialization",
        schema="stream4d_v81_phase5_adapter_v1",
        decision="PASS_V81_PHASE5_ADAPTER_MATERIALIZATION" if gate["pass"] else "NO_GO_PHASE5_HISTORY_FUSED_ADAPTER_WEAK",
        carrier_id_scope=args.carrier_id_scope_effective,
        history_anchor_type="phase4_history_fused_clusters",
        history_method_mode_allowed=gate["pass"],
        can_enter_next_phase=gate["pass"],
        can_enter_local2history=False,
        primary_blocker="" if gate["pass"] else "history_fused_adapter_does_not_beat_controls",
        secondary_blocker=summaries.get("B3_local_plus_history_attraction_conflict", {}).get("secondary_blocker", ""),
        extra={
            "B0_local_SF50_after_adapter": b0_sf50,
            "B3_local_SF50_after_adapter": b3_sf50,
            "B4_shuffled_local_SF50_after_adapter": b4_sf50,
            "B5_stale_local_SF50_after_adapter": b5_sf50,
            "B6_semantic_only_local_SF50_after_adapter": b6_sf50,
            "B3_minus_B0_local_SF50": b3_sf50 - b0_sf50,
            "B3_minus_B4_local_SF50": b3_sf50 - b4_sf50,
            "B3_minus_B5_local_SF50": b3_sf50 - b5_sf50,
            "B3_minus_B6_local_SF50": b3_sf50 - b6_sf50,
            "B3_GT_best_IoU_mean": summaries.get("B3_local_plus_history_attraction_conflict", {}).get("GT_best_IoU_mean", ""),
            "B3_adapter_summary": summaries.get("B3_local_plus_history_attraction_conflict", {}),
            "gate": gate,
            "runtime_sec": time.time() - started,
        },
    )
    _write_csv(output_root / "adapter_rows.csv", adapter_rows)
    _write_csv(output_root / "local_metric_rows.csv", metric_rows)
    _write_csv(output_root / "local_slot_rows.csv", slot_rows)
    _write_csv(output_root / "adapter_identity_audit_rows.csv", identity_rows)
    _write_json(output_root / "adapter_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _blocked_phase(
    output_root: Path,
    phase: str,
    schema: str,
    reason: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _phase_common(
        phase=phase,
        schema=schema,
        decision=reason,
        carrier_id_scope=getattr(args, "carrier_id_scope_effective", "global_full_sequence"),
        history_anchor_type="blocked",
        history_method_mode_allowed=False,
        can_enter_next_phase=False,
        can_enter_local2history=False,
        primary_blocker=reason,
        secondary_blocker="",
        extra={"runtime_sec": 0.0},
    )
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_blocked_later_phases(
    args: argparse.Namespace,
    reason: str,
    phase_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    phase_specs = {
        "phase4": (
            args.phase4_output_root,
            "v81_phase4_history_fused_clustering",
            "stream4d_v81_phase4_fused_clustering_v1",
            [
                "fused_affinity_rows.csv",
                "cluster_rows.csv",
                "control_variant_rows.csv",
            ],
            "cluster_summary.json",
        ),
        "phase5": (
            args.phase5_output_root,
            "v81_phase5_adapter_materialization",
            "stream4d_v81_phase5_adapter_v1",
            [
                "adapter_rows.csv",
                "local_slot_rows.csv",
                "local_metric_rows.csv",
                "adapter_identity_audit_rows.csv",
            ],
            "adapter_summary.json",
        ),
        "phase6": (
            args.phase6_output_root,
            "v81_phase6_new_object_birth",
            "stream4d_v81_phase6_birth_v1",
            ["birth_rows.csv", "false_attachment_rows.csv"],
            "birth_summary.json",
        ),
        "phase7": (
            args.phase7_output_root,
            "v81_phase7_history_update",
            "stream4d_v81_phase7_history_update_v1",
            ["history_update_rows.csv", "history_node_rows.csv", "history_edge_rows.csv"],
            "history_memory_summary.json",
        ),
        "phase8": (
            args.phase8_output_root,
            "v81_phase8_history_eval",
            "stream4d_v81_phase8_history_eval_v1",
            ["scene_metric_rows.csv", "history_control_rows.csv", "identity_switch_rows.csv"],
            "history_eval_summary.json",
        ),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, (root_text, phase, schema, csv_names, summary_name) in phase_specs.items():
        if phase_keys is not None and key not in phase_keys:
            continue
        root = ROOT / root_text
        summary = _blocked_phase(root, phase, schema, reason, args)
        for name in csv_names:
            _write_csv(root / name, [])
        _write_json(root / summary_name, summary)
        summaries[key] = summary
    return summaries


def _run_final(args: argparse.Namespace, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.final_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not bool((summaries.get("phase0", {}).get("gate") or {}).get("pass")):
        final_decision = "NO_GO_PHASE0_CAUSALITY_BOUNDARY"
    elif not bool((summaries.get("phase1", {}).get("history_eligibility_gate") or {}).get("pass")):
        final_decision = "NO_GO_BOOTSTRAP_LOCAL_WEAK"
    elif not bool((summaries.get("phase2", {}).get("gate") or {}).get("pass")):
        final_decision = "NO_GO_HISTORY_DESCRIPTOR_WEAK"
    elif not bool((summaries.get("phase3", {}).get("gate") or {}).get("pass")):
        final_decision = "NO_GO_CARRIER_TO_HISTORY_AFFINITY_WEAK"
    elif summaries.get("phase8", {}).get("decision", "").startswith("BLOCK"):
        final_decision = summaries["phase8"]["decision"]
    else:
        final_decision = "DIAGNOSTIC_HISTORY_ANCHOR_PROGRESS"
    payload = {
        "phase": "v81_final_decision",
        "schema": "stream4d_v81_final_decision_v1",
        "final_decision": final_decision,
        "phase_decisions": {
            phase: summaries.get(phase, {}).get("decision") for phase in PHASE_ORDER if phase in summaries
        },
        "best_dev_local_SF50": summaries.get("phase1", {}).get("local_SF50", ""),
        "history_Q_coverage_rate": summaries.get("phase3", {}).get("Q_coverage_rate", ""),
        "full_minus_semantic_top1_confidence": summaries.get("phase3", {}).get("full_minus_semantic_top1_confidence", ""),
        "can_enter_local2history": False,
        "primary_blocker": final_decision,
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "final_decision.json", payload)
    _write_json(output_root / "summary.json", payload)
    return payload


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    carrier_scope, _rows = _classify_carrier_scope(args)
    args.carrier_id_scope_effective = carrier_scope
    pipeline_root = ROOT / args.pipeline_root
    pipeline_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    phase_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    clusters: dict[tuple[str, int], dict[str, Any]] = {}
    history_rows: list[dict[str, Any]] = []
    history_fused_clusters: dict[tuple[str, int], dict[str, Any]] = {}
    for phase in PHASE_ORDER:
        phase_started = time.time()
        if phase == "phase0":
            summaries[phase] = _run_phase0(args)
        elif phase == "phase1":
            summaries[phase], _local_context, bundles, clusters = _run_phase1(args)
            args.appearance_audit_summary = _attach_carrier_appearance_profiles(clusters, args)
        elif phase == "phase2":
            summaries[phase], history_rows = _run_phase2(args, summaries["phase1"], clusters)
        elif phase == "phase3":
            summaries[phase] = _run_phase3(args, summaries["phase2"], history_rows, clusters)
            if not bool((summaries[phase].get("gate") or {}).get("pass")):
                summaries.update(_run_blocked_later_phases(args, "BLOCK_HISTORY_FUSED_BY_Q_WEAK"))
                phase_rows.append(
                    {
                        "phase": phase,
                        "decision": summaries[phase].get("decision"),
                        "gate_pass": (summaries[phase].get("gate") or {}).get("pass", ""),
                        "runtime_sec": time.time() - phase_started,
                    }
                )
                break
        elif phase == "phase4":
            summaries[phase], history_fused_clusters = _run_phase4(args, summaries["phase3"], clusters)
            if not bool((summaries[phase].get("gate") or {}).get("pass")):
                summaries.update(
                    _run_blocked_later_phases(
                        args,
                        "BLOCK_HISTORY_FUSED_BY_PHASE4_WEAK",
                        phase_keys={"phase5", "phase6", "phase7", "phase8"},
                    )
                )
                phase_rows.append(
                    {
                        "phase": phase,
                        "decision": summaries[phase].get("decision"),
                        "gate_pass": (summaries[phase].get("gate") or {}).get("pass", ""),
                        "runtime_sec": time.time() - phase_started,
                    }
                )
                break
        elif phase == "phase5":
            summaries[phase] = _run_phase5(args, summaries["phase4"], history_fused_clusters)
            if not bool((summaries[phase].get("gate") or {}).get("pass")):
                summaries.update(
                    _run_blocked_later_phases(
                        args,
                        "BLOCK_HISTORY_FUSED_BY_PHASE5_WEAK",
                        phase_keys={"phase6", "phase7", "phase8"},
                    )
                )
                phase_rows.append(
                    {
                        "phase": phase,
                        "decision": summaries[phase].get("decision"),
                        "gate_pass": (summaries[phase].get("gate") or {}).get("pass", ""),
                        "runtime_sec": time.time() - phase_started,
                    }
                )
                break
        elif phase in {"phase6", "phase7", "phase8"}:
            summaries.update(
                _run_blocked_later_phases(
                    args,
                    "BLOCK_HISTORY_FUSED_NOT_IMPLEMENTED_AFTER_PHASE4",
                    phase_keys={"phase6", "phase7", "phase8"},
                )
            )
            break
        elif phase == "final":
            summaries[phase] = _run_final(args, summaries)
        phase_rows.append(
            {
                "phase": phase,
                "decision": summaries[phase].get("decision") or summaries[phase].get("final_decision"),
                "gate_pass": (summaries[phase].get("gate") or {}).get("pass", ""),
                "runtime_sec": time.time() - phase_started,
            }
        )
        if phase == args.stop_after:
            break
    if "final" not in summaries:
        summaries["final"] = _run_final(args, summaries)
        phase_rows.append(
            {
                "phase": "final",
                "decision": summaries["final"].get("final_decision"),
                "gate_pass": "",
                "runtime_sec": summaries["final"].get("runtime_sec", 0.0),
            }
        )
    payload = {
        "phase": "v81_pipeline",
        "schema": "stream4d_v81_pipeline_v1",
        "split": args.split,
        "stop_after": args.stop_after,
        "decision": summaries["final"]["final_decision"],
        "phase_rows": phase_rows,
        "summaries": summaries,
        "runtime_sec": time.time() - started,
    }
    _write_json(pipeline_root / "pipeline_summary.json", payload)
    _write_json(pipeline_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = v80.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        stop_after="final",
        pipeline_root="outputs/audit/v81_history_anchored_cmap_af_l2h_pipeline",
        phase0_output_root="outputs/audit/v81_phase0_fact_causality",
        phase1_output_root="outputs/audit/v81_phase1_bootstrap_local",
        phase2_output_root="outputs/audit/v81_phase2_bootstrap_history",
        phase3_output_root="outputs/audit/v81_phase3_carrier_to_history",
        phase4_output_root="outputs/audit/v81_phase4_history_fused_clustering",
        phase5_output_root="outputs/audit/v81_phase5_adapter_materialization",
        phase6_output_root="outputs/audit/v81_phase6_new_object_birth",
        phase7_output_root="outputs/audit/v81_phase7_history_update",
        phase8_output_root="outputs/audit/v81_phase8_history_eval",
        final_output_root="outputs/audit/v81_final_decision",
        random_seed=8001,
    )
    parser.add_argument("--run-tag", default="dev_main")
    parser.add_argument("--local-shadow-root", default="outputs/audit/v81_local_shadow")
    parser.add_argument("--max-history-nodes", type=int, default=512)
    parser.add_argument("--max-tentative-nodes", type=int, default=256)
    parser.add_argument("--memory-budget-mb", type=float, default=512.0)
    parser.add_argument("--history-confirm-confidence", type=float, default=0.65)
    parser.add_argument("--max-history-cluster-carriers", type=int, default=256)
    parser.add_argument("--history-top-k", type=int, default=4)
    parser.add_argument("--history-q-min-score", type=float, default=0.35)
    parser.add_argument("--history-entropy-upper-bound", type=float, default=0.70)
    parser.add_argument("--history-residual-margin", type=float, default=0.03)
    parser.add_argument("--history-warmup-chunks", type=int, default=1)
    parser.add_argument("--history-coalescing-mode", choices=["none", "spatial", "appearance", "spatial_appearance"], default="none")
    parser.add_argument("--history-affinity-mode", choices=["legacy", "spatial"], default="legacy")
    parser.add_argument("--history-semantic-score-mode", choices=["exact", "token_overlap"], default="exact")
    parser.add_argument("--history-semantic-match-threshold", type=float, default=1.0)
    parser.add_argument(
        "--history-full-score-calibration",
        choices=["weighted", "confirmed_semantic_residual_floor"],
        default="weighted",
    )
    parser.add_argument("--history-merge-uv-threshold", type=float, default=0.18)
    parser.add_argument("--history-merge-min-score", type=float, default=0.55)
    parser.add_argument("--history-merge-min-appearance-score", type=float, default=0.85)
    parser.add_argument("--history-merge-max-age-chunks", type=int, default=3)
    parser.add_argument("--history-confirm-min-support-chunks", type=int, default=2)
    parser.add_argument("--history-spatial-sigma", type=float, default=0.18)
    parser.add_argument("--history-spatial-conflict-threshold", type=float, default=0.25)
    parser.add_argument("--history-temporal-decay-chunks", type=float, default=8.0)
    parser.add_argument("--history-confirmed-score-weight", type=float, default=1.0)
    parser.add_argument("--history-tentative-score-weight", type=float, default=0.55)
    parser.add_argument("--history-quarantine-score-weight", type=float, default=0.10)
    parser.add_argument("--history-full-require-semantic-match", action="store_true")
    parser.add_argument("--history-attraction-eta", type=float, default=0.20)
    parser.add_argument("--history-attraction-min-q-score", type=float, default=0.0)
    parser.add_argument("--history-attraction-top-per-group", type=int, default=0)
    parser.add_argument("--history-attraction-max-group-size", type=int, default=0)
    parser.add_argument("--history-attraction-edge-mode", choices=["star", "appearance_knn"], default="star")
    parser.add_argument("--history-attraction-pair-top-k", type=int, default=1)
    parser.add_argument("--history-attraction-min-pair-appearance", type=float, default=0.0)
    parser.add_argument("--history-attraction-min-pair-spatial", type=float, default=0.0)
    parser.add_argument("--history-attraction-require-different-local-components", action="store_true")
    parser.add_argument("--history-attraction-max-local-component-size", type=int, default=0)
    parser.add_argument("--history-conflict-split-mode", choices=["none", "confirmed_history_id"], default="none")
    parser.add_argument("--history-conflict-min-q-score", type=float, default=0.0)
    parser.add_argument("--history-conflict-min-history-group-size", type=int, default=1)
    parser.add_argument("--history-use-causal-snapshots", action="store_true")
    parser.add_argument("--history-require-causal-last-seen", action="store_true")
    parser.add_argument("--history-stale-control-min-age-chunks", type=int, default=2)
    parser.add_argument("--history-assignment-min-q-margin", type=float, default=0.0)
    parser.add_argument("--history-assignment-min-q-ratio", type=float, default=0.0)
    parser.add_argument("--history-assignment-min-appearance", type=float, default=0.0)
    parser.add_argument("--history-assignment-min-spatial", type=float, default=0.0)
    parser.add_argument("--history-assignment-max-conflict", type=float, default=1.0)
    parser.add_argument("--history-assignment-max-age-chunks", type=int, default=0)
    parser.add_argument("--appearance-feature-mode", choices=["proxy", "dino_v58", "dino_csv", "none"], default="proxy")
    parser.add_argument("--appearance-feature-rows", default="outputs/audit/v58_semantic_memory_dino_full_repair2/mask_feature_rows.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
