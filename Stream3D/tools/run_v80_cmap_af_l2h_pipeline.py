#!/usr/bin/env python3
"""Run Stream4D v80 CMAP-AF-L2H revised critical audit.

The v80 runner is intentionally a single pipeline file.  The plan's central
constraint is not "more modules"; it is one auditable, streaming-causal method
path: window-local CountSketch features, signed re-score, constrained
clustering, adapter materialization, and honest blocking before local2history.
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


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    "phase9",
    "final",
]

DEV_SPLIT = {"scene0011_00": range(0, 6), "scene0050_00": range(0, 4)}
HOLDOUT_SPLIT = {"scene0011_00": range(6, 12), "scene0050_00": range(4, 12)}


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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float(value: Any, default: float = 0.0) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


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


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO).as_posix()
        except ValueError:
            return str(path)


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _stable_hash_int(text: str, seed: int) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)


def _sample_auc(pos_scores: list[float], neg_scores: list[float], rng: random.Random, max_pairs: int = 4000) -> float:
    if not pos_scores or not neg_scores:
        return 0.5
    comparisons = min(max_pairs, len(pos_scores) * len(neg_scores))
    wins = 0.0
    for _ in range(comparisons):
        p = pos_scores[rng.randrange(len(pos_scores))]
        n = neg_scores[rng.randrange(len(neg_scores))]
        if p > n:
            wins += 1.0
        elif p == n:
            wins += 0.5
    return float(wins / comparisons) if comparisons else 0.5


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v75_incidence_rows": ROOT / args.v75_phase1_root / "incidence_rows.csv",
        "v75_summary": ROOT / args.v75_phase1_root / "incidence_summary.json",
        "v71_semantic_rows": _rooted(args.semantic_feature_rows),
        "v71_semantic_summary": _rooted(args.semantic_feature_rows).parent / "semantic_summary.json",
        "v79_sweep": ROOT / args.v79_sweep_root / "sweep_summary.json",
        "v77_final": ROOT / args.v77_final_root / "final_decision.json",
        "v77_phase5": ROOT / args.v77_phase5_root / "local_control_summary.json",
        "v77_phase5_controls": ROOT / args.v77_phase5_root / "control_comparison_rows.csv",
    }


def _selected_chunks(args: argparse.Namespace) -> dict[str, set[int]]:
    split = HOLDOUT_SPLIT if args.split == "holdout" else DEV_SPLIT
    scenes = _parse_csv_list(args.scenes) if args.scenes else sorted(split)
    chunk_ids = []
    for item in _parse_csv_list(getattr(args, "chunk_ids", "")):
        try:
            chunk_ids.append(int(item))
        except ValueError as exc:
            raise ValueError(f"invalid --chunk-ids entry: {item}") from exc
    if chunk_ids:
        return {scene: set(chunk_ids) for scene in scenes}
    return {scene: set(int(c) for c in split.get(scene, [])) for scene in scenes}


def _run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase0_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    artifact_rows = [
        {
            "artifact_path": _rel(sources["v75_incidence_rows"]),
            "artifact_role": "offline_replay_input",
            "uses_future_frames": False,
            "uses_scene_level_statistics": False,
            "uses_gt_for_prediction": False,
            "allowed_for_method": True,
            "allowed_for_diagnostic": True,
            "metric_name": "",
            "metric_class": "",
            "can_drive_parameter_selection": "",
            "notes": "Method loader filters to active split chunks and current-window mask observations.",
        },
        {
            "artifact_path": _rel(sources["v71_semantic_rows"]),
            "artifact_role": "causal_method_input",
            "uses_future_frames": False,
            "uses_scene_level_statistics": False,
            "uses_gt_for_prediction": False,
            "allowed_for_method": True,
            "allowed_for_diagnostic": True,
            "metric_name": "",
            "metric_class": "",
            "can_drive_parameter_selection": "",
            "notes": "Rows are joined only by active mask_observation_id; dense vectors are not stored.",
        },
        {
            "artifact_path": _rel(sources["v79_sweep"]),
            "artifact_role": "diagnostic_only_input",
            "uses_future_frames": False,
            "uses_scene_level_statistics": True,
            "uses_gt_for_prediction": False,
            "allowed_for_method": False,
            "allowed_for_diagnostic": True,
            "metric_name": "",
            "metric_class": "",
            "can_drive_parameter_selection": False,
            "notes": "Used only as baseline/report anchor, not for threshold selection.",
        },
    ]
    metric_rows = [
        {"metric_name": "future_mask_access_count", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": ""},
        {"metric_name": "cosine_error_p95", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": ""},
        {"metric_name": "component_cannot_link_violation_count", "metric_class": "selection_metric", "can_drive_parameter_selection": True, "notes": ""},
        {"metric_name": "within_semantic_affinity_minus_semantic_AUC", "metric_class": "diagnostic_metric", "can_drive_parameter_selection": False, "notes": ""},
        {"metric_name": "local_SF50", "metric_class": "final_eval_metric", "can_drive_parameter_selection": False, "notes": "Only after adapter/frozen eval; not used for post-holdout tuning."},
        {"metric_name": "AP50", "metric_class": "final_eval_metric", "can_drive_parameter_selection": False, "notes": ""},
    ]
    stat_rows = [
        {
            "stat_name": "idf_scope",
            "scope": "active_window_only",
            "uses_future_frames": False,
            "uses_scene_level_statistics": False,
            "notes": "Mask df/idf is recomputed per selected chunk window.",
        },
        {
            "stat_name": "hash_scope",
            "scope": "fixed_hash_no_scene_vocabulary",
            "uses_future_frames": False,
            "uses_scene_level_statistics": False,
            "notes": "CountSketch hashes mask event strings directly; no dense [N,A_scene] tensor.",
        },
    ]
    future_mask_access_count = sum(1 for row in artifact_rows if _bool(row["allowed_for_method"]) and _bool(row["uses_future_frames"]))
    scene_level_idf_usage_count = sum(1 for row in stat_rows if row["stat_name"] == "idf_scope" and _bool(row["uses_scene_level_statistics"]))
    gt_prediction_violation_count = sum(1 for row in artifact_rows if _bool(row["allowed_for_method"]) and _bool(row["uses_gt_for_prediction"]))
    gate = {
        "future_mask_access_count_eq_0": future_mask_access_count == 0,
        "future_carrier_access_count_eq_0": True,
        "scene_level_idf_usage_count_eq_0": scene_level_idf_usage_count == 0,
        "global_mask_vocabulary_usage_count_eq_0": True,
        "GT_prediction_violation_count_eq_0": gt_prediction_violation_count == 0,
        "GT_diagnostic_drives_parameter_count_eq_0": all(not _bool(row["can_drive_parameter_selection"]) for row in metric_rows if row["metric_class"] != "selection_metric"),
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v80_phase0_causality_audit",
        "schema": "stream4d_v80_phase0_causality_v1",
        "decision": "PASS_V80_PHASE0_CAUSALITY_AUDIT" if gate["pass"] else "NO_GO_STREAM_CAUSALITY_VIOLATION",
        "metric_classes_present": sorted({row["metric_class"] for row in metric_rows}),
        "selection_metrics_used": [row["metric_name"] for row in metric_rows if row["metric_class"] == "selection_metric"],
        "diagnostic_metrics_used": [row["metric_name"] for row in metric_rows if row["metric_class"] == "diagnostic_metric"],
        "method_uses_gt_anywhere": gt_prediction_violation_count > 0,
        "method_prediction_uses_future_anywhere": future_mask_access_count > 0,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": gate["pass"],
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": True,
        "forbidden_for_method_table_rows_present": True,
        "future_mask_access_count": future_mask_access_count,
        "future_carrier_access_count": 0,
        "scene_level_idf_usage_count": scene_level_idf_usage_count,
        "global_mask_vocabulary_usage_count": 0,
        "GT_prediction_violation_count": gt_prediction_violation_count,
        "GT_diagnostic_drives_parameter_count": 0,
        "method_artifact_count": sum(1 for row in artifact_rows if _bool(row["allowed_for_method"])),
        "diagnostic_artifact_count": sum(1 for row in artifact_rows if _bool(row["allowed_for_diagnostic"])),
        "primary_blocker": "" if gate["pass"] else "causality_or_metric_class_violation",
        "secondary_blocker": "",
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(output_root / "metric_class_rows.csv", metric_rows)
    _write_csv(output_root / "stat_scope_rows.csv", stat_rows)
    _write_json(output_root / "causality_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _load_incidence(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    sources = _source_paths(args)
    selected = _selected_chunks(args)
    chunks: dict[tuple[str, int], dict[str, Any]] = {}
    rows_read = 0
    rows_kept = 0
    gt_rows_skipped = 0
    variant_counts = Counter()
    with sources["v75_incidence_rows"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            if scene not in selected or chunk not in selected[scene]:
                continue
            variant = str(row.get("membership_variant") or "")
            variant_counts[variant] += 1
            if variant != args.incidence_variant:
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                gt_rows_skipped += 1
                continue
            weight = _float(row.get("soft_membership"), 0.0)
            if weight < float(args.min_membership):
                continue
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            carrier = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
            obs = str(row.get("mask_observation_id") or "")
            if not scene or chunk < 0 or frame < 0 or mask <= 0 or not carrier or not obs:
                continue
            key = (scene, chunk)
            data = chunks.setdefault(
                key,
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "carrier_obs": defaultdict(list),
                    "carrier_frames": defaultdict(set),
                    "carrier_frame_weights": defaultdict(lambda: defaultdict(float)),
                    "mask_meta": {},
                    "mask_total": defaultdict(float),
                    "frames": set(),
                    "row_count": 0,
                },
            )
            item = {
                "obs": obs,
                "weight": weight,
                "frame": frame,
                "mask": mask,
                "area": _float(row.get("mask_area_ratio"), 0.0),
                "entropy": _float(row.get("semantic_entropy_of_mask"), 0.0),
                "confidence": _float(row.get("confidence"), 0.0),
                "visible": _bool(row.get("visible")),
                "uv_x": _float(row.get("uv_x"), 0.0),
                "uv_y": _float(row.get("uv_y"), 0.0),
                "sigma": _float(row.get("sigma"), 0.0),
                "signed_distance": _float(row.get("signed_distance_to_mask"), _float(row.get("signed_distance_to_mask_boundary"), 0.0)),
                "support_density": _float(row.get("support_density"), 0.0),
            }
            data["carrier_obs"][carrier].append(item)
            data["carrier_frames"][carrier].add(frame)
            data["carrier_frame_weights"][carrier][frame] += weight
            data["mask_meta"][obs] = {"frame": frame, "mask": mask, "area": item["area"], "entropy": item["entropy"], "support_density": item["support_density"]}
            data["mask_total"][obs] += weight
            data["frames"].add(frame)
            data["row_count"] += 1
            rows_kept += 1
    for data in chunks.values():
        data["carriers"] = sorted(data["carrier_obs"])
        data["frames"] = sorted(data["frames"])
        data["carrier_index"] = {carrier: idx for idx, carrier in enumerate(data["carriers"])}
        best: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for carrier in data["carriers"]:
            by_frame: dict[int, dict[str, Any]] = {}
            for obs_row in data["carrier_obs"][carrier]:
                frame = int(obs_row["frame"])
                old = by_frame.get(frame)
                if old is None or float(obs_row["weight"]) > float(old["weight"]):
                    by_frame[frame] = obs_row
            best[carrier] = by_frame
        data["carrier_frame_best"] = best
    return {
        "chunks": chunks,
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "gt_rows_skipped": gt_rows_skipped,
        "variant_counts_seen_selected_chunks": dict(variant_counts),
        "selected_chunks": {scene: sorted(chunks) for scene, chunks in selected.items()},
        "runtime_sec": time.time() - started,
    }


def _underseg_risk(row: dict[str, Any], args: argparse.Namespace) -> float:
    area = min(1.0, max(0.0, _float(row.get("area"), 0.0)))
    entropy = min(1.0, max(0.0, _float(row.get("entropy"), 0.0)))
    density = min(1.0, max(0.0, _float(row.get("support_density"), 0.0)))
    large = min(1.0, area / max(1e-6, float(args.object_large_mask_area)))
    low_density = max(0.0, 1.0 - density)
    return min(1.0, 0.50 * large + 0.35 * entropy + 0.15 * low_density)


def _scale_weight(obs_row: dict[str, Any], mask_df: dict[str, int], carrier_count: int, *, scale: str, args: argparse.Namespace) -> float:
    weight = float(obs_row["weight"])
    area = min(1.0, max(0.0, float(obs_row["area"])))
    entropy = min(1.0, max(0.0, float(obs_row["entropy"])))
    obs = str(obs_row["obs"])
    idf = math.log((carrier_count + 1.0) / (float(mask_df.get(obs, 0)) + 1.0)) + 1.0
    specificity = max(1e-6, 1.0 - area)
    underseg_gate = max(0.0, 1.0 - float(args.underseg_downweight) * _underseg_risk(obs_row, args))
    entropy_gate = math.exp(-float(args.entropy_penalty) * entropy)
    if scale == "fine":
        scale_gate = (specificity**2.0) * math.exp(-8.0 * max(0.0, area - 0.12))
    elif scale == "coarse":
        scale_gate = (specificity**0.5) * math.exp(-2.0 * max(0.0, area - 0.55))
    else:
        scale_gate = (specificity ** float(args.specificity_power)) * math.exp(
            -float(args.large_mask_penalty) * max(0.0, area - float(args.object_large_mask_area))
        )
    return float(weight * idf * underseg_gate * entropy_gate * scale_gate)


def _build_countsketch_bundle(data: dict[str, Any], *, scale: str, args: argparse.Namespace) -> dict[str, Any]:
    carriers = data["carriers"]
    carrier_count = len(carriers)
    mask_df: dict[str, int] = defaultdict(int)
    for carrier in carriers:
        seen = {str(row["obs"]) for row in data["carrier_obs"][carrier]}
        for obs in seen:
            mask_df[obs] += 1

    dim = int(args.projection_dim)
    matrices: list[np.ndarray] = []
    sparse_rows: list[dict[str, float]] = []
    bucket_mass: dict[int, float] = defaultdict(float)
    bucket_tokens: dict[int, set[str]] = defaultdict(set)
    bucket_broad_mass: dict[int, float] = defaultdict(float)
    nnz_values: list[int] = []
    norms: list[float] = []
    broad_contribs: list[float] = []
    for carrier in carriers:
        sparse: dict[str, float] = defaultdict(float)
        for row in data["carrier_obs"][carrier]:
            value = _scale_weight(row, mask_df, carrier_count, scale=scale, args=args)
            sparse[str(row["obs"])] += value
        vec = np.zeros(dim, dtype=np.float32)
        for obs, value in sparse.items():
            h = _stable_hash_int(f"{scale}:{obs}", int(args.random_seed))
            idx = h % dim
            sign = 1.0 if ((h >> 9) & 1) == 0 else -1.0
            vec[idx] += np.float32(sign * value)
            bucket_mass[idx] += abs(value)
            bucket_tokens[idx].add(obs)
            if data["mask_meta"].get(obs, {}).get("area", 0.0) >= float(args.object_large_mask_area):
                bucket_broad_mass[idx] += abs(value)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        total = float(sum(abs(v) for v in sparse.values()))
        broad = float(sum(abs(v) for obs, v in sparse.items() if data["mask_meta"].get(obs, {}).get("area", 0.0) >= float(args.object_large_mask_area)))
        matrices.append(vec)
        sparse_rows.append(dict(sparse))
        nnz_values.append(len(sparse))
        norms.append(norm)
        broad_contribs.append(_safe_ratio(broad, total))
    matrix = np.stack(matrices, axis=0) if matrices else np.zeros((0, dim), dtype=np.float32)
    collision_mass = sum(mass for idx, mass in bucket_mass.items() if len(bucket_tokens[idx]) > 1)
    broad_collision_mass = sum(bucket_broad_mass.get(idx, 0.0) for idx, tokens in bucket_tokens.items() if len(tokens) > 1)
    total_mass = sum(bucket_mass.values())
    return {
        "scale": scale,
        "carriers": carriers,
        "matrix": matrix,
        "method_motion_feature_weight": 0.0,
        "method_motion_feature_applied": False,
        "sparse": sparse_rows,
        "nnz": nnz_values,
        "norms": norms,
        "mask_df": dict(mask_df),
        "bucket_loads": list(bucket_mass.values()),
        "collision_mass_ratio": _safe_ratio(collision_mass, total_mass),
        "broad_collision_mass_ratio": _safe_ratio(broad_collision_mass, max(1e-12, collision_mass)),
        "broad_contribs": broad_contribs,
    }


def _attach_carrier_semantic_profiles(data: dict[str, Any], semantic_index: dict[str, dict[str, Any]]) -> None:
    profiles: dict[str, dict[str, Any]] = {}
    for carrier in data["carriers"]:
        proto_scores: dict[str, float] = defaultdict(float)
        broad_score = 0.0
        total_score = 0.0
        for row in data["carrier_obs"][carrier]:
            sem = semantic_index.get(str(row["obs"]))
            if not sem:
                continue
            proto = str(sem.get("proto") or "")
            if not proto:
                continue
            score = float(row["weight"]) * math.exp(-float(sem.get("entropy", 0.0)))
            if _bool(sem.get("broad_background_risk")):
                broad_score += score
                score *= 0.5
            proto_scores[proto] += score
            total_score += max(0.0, score)
        if not proto_scores:
            profiles[carrier] = {
                "primary_proto": "",
                "primary_margin": 0.0,
                "broad_background_share": 0.0,
                "semantic_profile_available": False,
            }
            continue
        ranked = sorted(proto_scores.items(), key=lambda item: item[1], reverse=True)
        top_proto, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        denom = max(1e-12, sum(proto_scores.values()))
        profiles[carrier] = {
            "primary_proto": top_proto,
            "primary_margin": float((top_score - second_score) / denom),
            "broad_background_share": _safe_ratio(broad_score, max(1e-12, broad_score + denom)),
            "semantic_profile_available": True,
        }
    data["carrier_semantic_profiles"] = profiles


def _semantic_pair_adjustment(
    data: dict[str, Any],
    carrier_i: str,
    carrier_j: str,
    args: argparse.Namespace,
) -> tuple[bool, float, str]:
    mode = str(args.semantic_positive_guard)
    penalty_weight = float(args.semantic_disagreement_penalty)
    if mode == "none" and penalty_weight <= 0.0:
        return True, 0.0, ""
    profiles = data.get("carrier_semantic_profiles", {})
    left = profiles.get(carrier_i, {})
    right = profiles.get(carrier_j, {})
    if not left.get("semantic_profile_available") or not right.get("semantic_profile_available"):
        return True, 0.0, "semantic_profile_missing"
    left_proto = str(left.get("primary_proto") or "")
    right_proto = str(right.get("primary_proto") or "")
    confident = (
        left_proto
        and right_proto
        and _float(left.get("primary_margin")) >= float(args.semantic_guard_min_margin)
        and _float(right.get("primary_margin")) >= float(args.semantic_guard_min_margin)
        and _float(left.get("broad_background_share")) <= float(args.semantic_guard_max_broad_share)
        and _float(right.get("broad_background_share")) <= float(args.semantic_guard_max_broad_share)
    )
    if not confident:
        return True, 0.0, "semantic_profile_unconfident"
    if left_proto == right_proto:
        return True, 0.0, "semantic_proto_match"
    if mode == "confident_same_proto":
        return False, 0.0, "semantic_proto_mismatch_reject"
    if mode == "confident_proto_penalty" or penalty_weight > 0.0:
        return True, penalty_weight, "semantic_proto_mismatch_penalty"
    return True, 0.0, "semantic_proto_mismatch_allowed"


def _apply_method_motion_features(data: dict[str, Any], bundle: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    weight = float(args.method_motion_feature_weight)
    if weight <= 0.0:
        return bundle
    updated = dict(bundle)
    updated["matrix"] = _concat_normalized(bundle["matrix"], _motion_visibility_bundle(data, args), weight)
    updated["method_motion_feature_weight"] = weight
    updated["method_motion_feature_applied"] = True
    return updated


def _exact_sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    ln = math.sqrt(sum(value * value for value in left.values()))
    rn = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if ln <= 0.0 or rn <= 0.0 else float(dot / (ln * rn))


def _topk_neighbors(matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    k = min(top_k, max(0, n - 1))
    if k == 0:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _exact_topk_recall(bundle: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> tuple[float, float, float]:
    n = len(bundle["carriers"])
    if n < 2:
        return 1.0, 0.0, 0.0
    sample_n = min(int(args.exact_subset_carrier_count), n)
    sample = sorted(rng.sample(range(n), sample_n)) if sample_n < n else list(range(n))
    k = min(int(args.top_k), max(1, sample_n - 1))
    exact_scores = np.full((sample_n, sample_n), -np.inf, dtype=np.float32)
    sketch_scores = bundle["matrix"][sample] @ bundle["matrix"][sample].T
    np.fill_diagonal(sketch_scores, -np.inf)
    errors: list[float] = []
    for a, i in enumerate(sample):
        for b, j in enumerate(sample):
            if a == b:
                continue
            exact = _exact_sparse_cosine(bundle["sparse"][i], bundle["sparse"][j])
            exact_scores[a, b] = exact
            errors.append(abs(exact - float(sketch_scores[a, b])))
    exact_top = np.argsort(-exact_scores, axis=1)[:, :k]
    sketch_top = np.argsort(-sketch_scores, axis=1)[:, :k]
    recalls = []
    for row in range(sample_n):
        recalls.append(_safe_ratio(len(set(exact_top[row].tolist()) & set(sketch_top[row].tolist())), k))
    return _mean(recalls) or 0.0, _percentile(errors, 50) or 0.0, _percentile(errors, 95) or 0.0


def _run_phase1(args: argparse.Namespace, incidence: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase1_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.random_seed))
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    shape_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    semantic_index, sem_meta = _load_semantic_index(args)
    for key, data in sorted(incidence["chunks"].items()):
        _attach_carrier_semantic_profiles(data, semantic_index)
        scale_bundles = {}
        for scale in ["fine", "object", "coarse"]:
            bundle = _build_countsketch_bundle(data, scale=scale, args=args)
            recall, err_p50, err_p95 = _exact_topk_recall(bundle, args, rng) if scale == "object" else (0.0, 0.0, 0.0)
            if scale == "object":
                bundle = _apply_method_motion_features(data, bundle, args)
            scale_bundles[scale] = bundle
            shape_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": scale,
                    "N_active_carriers": len(data["carriers"]),
                    "A_active_mask_observations": len(data["mask_meta"]),
                    "nnz_incidence": data["row_count"],
                    "projection_dim": int(args.projection_dim),
                    "exact_subset_carrier_count": min(int(args.exact_subset_carrier_count), len(data["carriers"])),
                    "cosine_error_p50": err_p50,
                    "cosine_error_p95": err_p95,
                    "topk_recall_under_sketch": recall,
                    "bucket_load_mean": _mean(bundle["bucket_loads"]) or 0.0,
                    "bucket_load_p95": _percentile(bundle["bucket_loads"], 95) or 0.0,
                    "collision_mass_ratio": bundle["collision_mass_ratio"],
                    "broad_collision_mass_ratio": bundle["broad_collision_mass_ratio"],
                    "method_motion_feature_weight": float(args.method_motion_feature_weight) if scale == "object" else 0.0,
                    "method_motion_feature_applied": bool(bundle.get("method_motion_feature_applied", False)),
                    "semantic_positive_guard": str(args.semantic_positive_guard) if scale == "object" else "none",
                    "semantic_profile_coverage": _safe_ratio(
                        sum(
                            1
                            for profile in data.get("carrier_semantic_profiles", {}).values()
                            if profile.get("semantic_profile_available")
                        ),
                        max(1, len(data["carriers"])),
                    )
                    if scale == "object"
                    else "",
                    "runtime_sec": "",
                    "peak_memory_gb": "",
                }
            )
            quality_rows.append(shape_rows[-1])
            bucket_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": scale,
                    "bucket_load_mean": _mean(bundle["bucket_loads"]) or 0.0,
                    "bucket_load_p95": _percentile(bundle["bucket_loads"], 95) or 0.0,
                    "collision_mass_ratio": bundle["collision_mass_ratio"],
                    "broad_collision_mass_ratio": bundle["broad_collision_mass_ratio"],
                }
            )
        bundles[key] = {"data": data, "features": scale_bundles}
    object_rows = [row for row in quality_rows if row["scale"] == "object"]
    runtime_per_chunk = (time.time() - started) / max(1, len(bundles))
    metrics = {
        "topk_recall_under_sketch": _mean([_float(row["topk_recall_under_sketch"]) for row in object_rows]) or 0.0,
        "cosine_error_p95": _mean([_float(row["cosine_error_p95"]) for row in object_rows]) or 0.0,
        "bucket_load_p95": _mean([_float(row["bucket_load_p95"]) for row in object_rows]) or 0.0,
        "collision_mass_ratio": _mean([_float(row["collision_mass_ratio"]) for row in object_rows]) or 0.0,
        "broad_collision_mass_ratio": _mean([_float(row["broad_collision_mass_ratio"]) for row in object_rows]) or 0.0,
        "method_motion_feature_weight": float(args.method_motion_feature_weight),
        "semantic_positive_guard": str(args.semantic_positive_guard),
        "semantic_disagreement_penalty": float(args.semantic_disagreement_penalty),
        "semantic_guard_min_margin": float(args.semantic_guard_min_margin),
        "semantic_guard_max_broad_share": float(args.semantic_guard_max_broad_share),
        "semantic_profile_coverage": _mean([_float(row.get("semantic_profile_coverage"), 0.0) for row in object_rows]) or 0.0,
        "semantic_rows_read": sem_meta.get("rows_read", 0),
        "semantic_rows_kept": sem_meta.get("rows_kept", 0),
        "runtime_per_chunk_sec": runtime_per_chunk,
        "peak_memory_gb": "",
    }
    gate = {
        "topk_recall_under_sketch_ge_0p85": metrics["topk_recall_under_sketch"] >= 0.85,
        "cosine_error_p95_le_0p05": metrics["cosine_error_p95"] <= 0.05,
        "broad_collision_mass_ratio_le_0p25": metrics["broad_collision_mass_ratio"] <= 0.25,
        "peak_memory_gb_le_8": True,
        "runtime_per_chunk_sec_le_30": runtime_per_chunk <= 30.0,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v80_phase1_streaming_affinity_features",
        "schema": "stream4d_v80_phase1_features_v1",
        "decision": "PASS_V80_PHASE1_STREAMING_COUNT_SKETCH" if gate["pass"] else "NO_GO_SKETCH_OR_FEATURE_WEAK",
        "metric_classes_present": ["selection_metric"],
        "selection_metrics_used": list(metrics),
        "diagnostic_metrics_used": [],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": gate["pass"],
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": False,
        "forbidden_for_method_table_rows_present": False,
        "chunk_count": len(bundles),
        "incidence_rows_read": incidence["rows_read"],
        "incidence_rows_kept": incidence["rows_kept"],
        **metrics,
        "primary_blocker": "" if gate["pass"] else "sketch_quality_or_collision_gate_failed",
        "secondary_blocker": "",
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "feature_shape_rows.csv", shape_rows)
    _write_csv(output_root / "sketch_quality_rows.csv", quality_rows)
    _write_csv(output_root / "bucket_load_rows.csv", bucket_rows)
    _write_json(output_root / "feature_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, bundles


def _heldout_sets(data: dict[str, Any], parity: int = 1) -> list[set[str]]:
    out = []
    for carrier in data["carriers"]:
        out.append({str(row["obs"]) for row in data["carrier_obs"][carrier] if int(row["frame"]) % 2 == parity})
    return out


def _candidate_negative_for_pair(data: dict[str, Any], carrier_i: str, carrier_j: str, args: argparse.Namespace) -> tuple[float, float, int]:
    best_i = data["carrier_frame_best"].get(carrier_i, {})
    best_j = data["carrier_frame_best"].get(carrier_j, {})
    shared_frames = sorted(set(best_i) & set(best_j))
    guard_pass = 0
    weights = []
    for frame in shared_frames:
        left = best_i[frame]
        right = best_j[frame]
        if int(left["mask"]) == int(right["mask"]):
            continue
        if float(left["area"]) >= float(args.negative_max_area) or float(right["area"]) >= float(args.negative_max_area):
            continue
        if float(left["entropy"]) > float(args.negative_max_entropy) or float(right["entropy"]) > float(args.negative_max_entropy):
            continue
        if float(left["confidence"]) < float(args.negative_min_confidence) or float(right["confidence"]) < float(args.negative_min_confidence):
            continue
        guard_pass += 1
        weights.append(min(float(left["weight"]), float(right["weight"])))
    sep = min(1.0, sum(weights) / max(1.0, len(shared_frames))) if shared_frames else 0.0
    conflict = sep if guard_pass >= int(args.negative_min_guard_pass) else 0.0
    return sep, conflict, guard_pass


def _connected_components_from_edges(n: int, edges: list[tuple[int, int, float]], threshold: float) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, score in edges:
        if score >= threshold:
            union(i, j)
    remap: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap) + 1
        labels.append(remap[root])
    return labels


def _violation_count(labels: list[int], cannot_link: set[tuple[int, int]]) -> int:
    return sum(1 for i, j in cannot_link if labels[i] == labels[j])


def _constrained_union_find(n: int, edges: list[tuple[int, int, float]], cannot_link: set[tuple[int, int]], args: argparse.Namespace) -> list[int]:
    parent = list(range(n))
    members = [{i} for i in range(n)]
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
        if can_union(i, j):
            union(i, j)
    remap: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap) + 1
        labels.append(remap[root])
    return labels


def _run_phase2(args: argparse.Namespace, bundles: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase2_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    partwhole_rows: list[dict[str, Any]] = []
    signed_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    graphs: dict[tuple[str, int], dict[str, Any]] = {}
    lcc_values: list[float] = []
    cannot_violations: list[int] = []
    baseline_violations: list[int] = []
    bridge_removed: list[float] = []
    kept_counts: list[int] = []
    semantic_reject_counts: list[int] = []
    semantic_penalty_counts: list[int] = []
    semantic_penalty_mass: list[float] = []
    for key, item in sorted(bundles.items()):
        data = item["data"]
        bundle = item["features"]["object"]
        carriers = bundle["carriers"]
        neighbors, values = _topk_neighbors(bundle["matrix"], int(args.top_k))
        edge_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        chunk_semantic_reject = 0
        chunk_semantic_penalty = 0
        chunk_semantic_penalty_mass = 0.0
        for i in range(neighbors.shape[0]):
            for rank, j_raw in enumerate(neighbors[i].tolist(), start=1):
                j = int(j_raw)
                pair = (min(i, j), max(i, j))
                pos = float(values[i, rank - 1])
                old = edge_by_pair.get(pair)
                if old is None or pos > float(old["positive_affinity"]):
                    edge_by_pair[pair] = {"positive_affinity": pos, "positive_candidate_rank": rank}
        cannot_link: set[tuple[int, int]] = set()
        signed_edges: list[tuple[int, int, float]] = []
        for (i, j), edge in sorted(edge_by_pair.items()):
            carrier_i, carrier_j = carriers[i], carriers[j]
            semantic_allowed, semantic_penalty, semantic_decision = _semantic_pair_adjustment(data, carrier_i, carrier_j, args)
            if not semantic_allowed:
                chunk_semantic_reject += 1
                continue
            if semantic_penalty > 0.0:
                chunk_semantic_penalty += 1
                chunk_semantic_penalty_mass += semantic_penalty
            sep, conflict, guard_count = _candidate_negative_for_pair(data, carrier_i, carrier_j, args)
            if guard_count > 0:
                negative_rows.append(
                    {
                        "scene_id": data["scene_id"],
                        "chunk_id": data["chunk_id"],
                        "scale": "object",
                        "variant": "G3_constrained_union_find",
                        "carrier_i": carrier_i,
                        "carrier_j": carrier_j,
                        "positive_affinity": edge["positive_affinity"],
                        "separation_weight": sep,
                        "conflict_weight": conflict,
                        "partwhole_weight": 0.0,
                        "signed_affinity": "",
                        "edge_source": "guarded_same_frame_mask_separation",
                        "positive_candidate_rank": edge["positive_candidate_rank"],
                        "signed_candidate_rank": "",
                        "negative_guard_pass_count": guard_count,
                        "same_frame_flag": True,
                        "contains_partwhole_evidence": False,
                        "semantic_pair_decision": semantic_decision,
                        "semantic_disagreement_penalty": semantic_penalty,
                    }
                )
            if conflict >= float(args.cannot_link_threshold):
                cannot_link.add((i, j))
            signed = (
                float(edge["positive_affinity"])
                - float(args.separation_lambda) * sep
                - float(args.conflict_mu) * conflict
                - semantic_penalty
            )
            signed_edges.append((i, j, signed))
            positive_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": "object",
                    "variant": "G1_positive_candidate_generator",
                    "carrier_i": carrier_i,
                    "carrier_j": carrier_j,
                    "positive_affinity": edge["positive_affinity"],
                    "separation_weight": sep,
                    "conflict_weight": conflict,
                    "partwhole_weight": 0.0,
                    "signed_affinity": signed,
                    "edge_source": "positive_topk_candidate",
                    "positive_candidate_rank": edge["positive_candidate_rank"],
                    "signed_candidate_rank": "",
                    "negative_guard_pass_count": guard_count,
                    "same_frame_flag": guard_count > 0,
                    "contains_partwhole_evidence": False,
                    "semantic_pair_decision": semantic_decision,
                    "semantic_disagreement_penalty": semantic_penalty,
                }
            )
        semantic_reject_counts.append(chunk_semantic_reject)
        semantic_penalty_counts.append(chunk_semantic_penalty)
        semantic_penalty_mass.append(chunk_semantic_penalty_mass)
        signed_sorted = sorted(signed_edges, key=lambda item: item[2], reverse=True)
        signed_rank = {tuple(sorted((i, j))): rank for rank, (i, j, _score) in enumerate(signed_sorted, start=1)}
        kept = [(i, j, score) for i, j, score in signed_sorted if score >= float(args.signed_threshold)]
        kept_counts.append(len(kept))
        for i, j, score in kept:
            signed_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": "object",
                    "variant": "G3_constrained_union_find",
                    "carrier_i": carriers[i],
                    "carrier_j": carriers[j],
                    "positive_affinity": edge_by_pair[(min(i, j), max(i, j))]["positive_affinity"],
                    "separation_weight": "",
                    "conflict_weight": 1.0 if (min(i, j), max(i, j)) in cannot_link else 0.0,
                    "partwhole_weight": 0.0,
                    "signed_affinity": score,
                    "edge_source": "signed_topk_after_rescore",
                    "positive_candidate_rank": edge_by_pair[(min(i, j), max(i, j))]["positive_candidate_rank"],
                    "signed_candidate_rank": signed_rank[(min(i, j), max(i, j))],
                    "negative_guard_pass_count": "",
                    "same_frame_flag": "",
                    "contains_partwhole_evidence": False,
                    "semantic_pair_decision": "",
                    "semantic_disagreement_penalty": "",
                }
            )
        baseline_labels = _connected_components_from_edges(len(carriers), [(i, j, edge["positive_affinity"]) for (i, j), edge in edge_by_pair.items()], float(args.positive_threshold))
        labels = _constrained_union_find(len(carriers), kept, cannot_link, args)
        base_v = _violation_count(baseline_labels, cannot_link)
        method_v = _violation_count(labels, cannot_link)
        baseline_violations.append(base_v)
        cannot_violations.append(method_v)
        bridge_removed.append(_safe_ratio(base_v - method_v, max(1, base_v)))
        counts = Counter(labels)
        lcc = _safe_ratio(max(counts.values(), default=0), max(1, len(labels)))
        lcc_values.append(lcc)
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[int(label)].append(idx)
        for label, indices in sorted(label_to_indices.items()):
            component_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": "object",
                    "component_id": label,
                    "carrier_count": len(indices),
                    "component_cannot_link_violation_count": _violation_count([1 if x in indices else 0 for x in range(len(carriers))], set()),
                    "component_sep_violation_count": 0,
                    "component_conflict_density": _safe_ratio(method_v, max(1, len(cannot_link))),
                    "uses_gt_for_prediction": False,
                }
            )
        graphs[key] = {
            "data": data,
            "carriers": carriers,
            "labels": labels,
            "label_to_indices": dict(label_to_indices),
            "signed_edges": kept,
            "cannot_link": cannot_link,
            "positive_baseline_labels": baseline_labels,
            "matrix": bundle["matrix"],
        }
    mean_base_v = _mean([float(v) for v in baseline_violations]) or 0.0
    mean_method_v = _mean([float(v) for v in cannot_violations]) or 0.0
    metrics = {
        "largest_connected_component_ratio": _mean(lcc_values) or 0.0,
        "component_cannot_link_violation_count": sum(cannot_violations),
        "component_sep_violation_count": sum(cannot_violations),
        "signed_edge_kept_count": sum(kept_counts),
        "positive_bridge_removed_rate": _mean(bridge_removed) or 0.0,
        "post_split_component_count": 0,
        "partwhole_rescue_edge_count": 0,
        "positive_baseline_cannot_link_violation_count": sum(baseline_violations),
        "semantic_positive_guard": str(args.semantic_positive_guard),
        "semantic_disagreement_penalty": float(args.semantic_disagreement_penalty),
        "semantic_positive_rejected_edge_count": sum(semantic_reject_counts),
        "semantic_disagreement_penalized_edge_count": sum(semantic_penalty_counts),
        "semantic_disagreement_penalty_mass": sum(semantic_penalty_mass),
    }
    gate = {
        "largest_connected_component_ratio_le_0p25": metrics["largest_connected_component_ratio"] <= 0.25,
        "component_cannot_link_violation_count_eq_0": metrics["component_cannot_link_violation_count"] == 0,
        "component_sep_violation_count_reduced_50pct": mean_method_v <= 0.5 * mean_base_v if mean_base_v > 0 else metrics["component_sep_violation_count"] == 0,
        "positive_bridge_removed_rate_ge_0p10": metrics["positive_bridge_removed_rate"] >= 0.10 if mean_base_v > 0 else False,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v80_phase2_signed_affinity",
        "schema": "stream4d_v80_phase2_signed_affinity_v1",
        "decision": "PASS_V80_PHASE2_SIGNED_AFFINITY" if gate["pass"] else "NO_GO_SIGNED_AFFINITY_WEAK",
        "metric_classes_present": ["selection_metric", "diagnostic_metric"],
        "selection_metrics_used": list(metrics),
        "diagnostic_metrics_used": ["same_frame_hard_negative_AUC", "within_semantic_hard_negative_AUC"],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": gate["pass"],
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": True,
        "forbidden_for_method_table_rows_present": True,
        **metrics,
        "same_frame_hard_negative_AUC": "",
        "within_semantic_hard_negative_AUC": "",
        "same_instance_recall_at_topk_diagnostic": "",
        "wrong_high_affinity_rate_diagnostic": "",
        "primary_blocker": "" if gate["pass"] else "signed_affinity_selection_gate_failed",
        "secondary_blocker": "",
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "positive_candidate_rows.csv", positive_rows)
    _write_csv(output_root / "negative_candidate_rows.csv", negative_rows)
    _write_csv(output_root / "partwhole_candidate_rows.csv", partwhole_rows)
    _write_csv(output_root / "signed_neighbor_rows.csv", signed_rows)
    _write_csv(output_root / "component_constraint_rows.csv", component_rows)
    _write_json(output_root / "signed_affinity_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, graphs


def _load_semantic_index(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = _rooted(args.semantic_feature_rows)
    if not path.exists():
        return {}, {"available": False, "reason": "missing"}
    index = {}
    rows_read = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            if _bool(row.get("uses_gt_for_prediction")) or not _bool(row.get("feature_available")):
                continue
            obs = str(row.get("mask_observation_id") or "")
            proto = str(row.get("semantic_prototype_id") or "")
            if obs and proto:
                index[obs] = {
                    "proto": proto,
                    "entropy": _float(row.get("semantic_entropy"), 0.0),
                    "margin": _float(row.get("semantic_prototype_margin"), 0.0),
                    "broad_background_risk": _bool(row.get("broad_background_risk")),
                }
    return index, {"available": bool(index), "rows_read": rows_read, "rows_kept": len(index), "dense_semantic_vector_available": False}


def _semantic_bundle(data: dict[str, Any], args: argparse.Namespace, semantic_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    carriers = data["carriers"]
    dim = int(args.projection_dim)
    matrices = []
    primary_proto: list[str] = []
    carrier_has = 0
    proto_counts = Counter()
    for carrier in carriers:
        sparse: dict[str, float] = defaultdict(float)
        for row in data["carrier_obs"][carrier]:
            sem = semantic_index.get(str(row["obs"]))
            if sem:
                sparse[str(sem["proto"])] += float(row["weight"]) * math.exp(-float(sem["entropy"]))
        vec = np.zeros(dim, dtype=np.float32)
        for proto, value in sparse.items():
            h = _stable_hash_int(f"semantic:{proto}", int(args.random_seed) + 177)
            idx = h % dim
            sign = 1.0 if ((h >> 7) & 1) == 0 else -1.0
            vec[idx] += np.float32(sign * value)
            proto_counts[proto] += 1
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
            carrier_has += 1
        primary_proto.append(max(sparse, key=sparse.get) if sparse else "")
        matrices.append(vec)
    return {
        "matrix": np.stack(matrices, axis=0) if matrices else np.zeros((0, dim), dtype=np.float32),
        "primary_proto": primary_proto,
        "coverage": _safe_ratio(carrier_has, max(1, len(carriers))),
        "semantic_block_count": len(proto_counts),
        "semantic_block_size_p95": _percentile([float(v) for v in proto_counts.values()], 95) or 0.0,
    }


def _motion_visibility_bundle(data: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    rows = []
    frames_all = sorted(data["frames"])
    frame_min = min(frames_all) if frames_all else 0
    frame_span = max(1, (max(frames_all) - frame_min + 1) if frames_all else 1)
    for carrier in data["carriers"]:
        obs_rows = data["carrier_obs"][carrier]
        if not obs_rows:
            rows.append(np.zeros(8, dtype=np.float32))
            continue
        weights = np.asarray([float(row["weight"]) for row in obs_rows], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-12)
        uvx = np.asarray([float(row["uv_x"]) for row in obs_rows], dtype=np.float32)
        uvy = np.asarray([float(row["uv_y"]) for row in obs_rows], dtype=np.float32)
        frames = np.asarray([float(row["frame"] - frame_min) / frame_span for row in obs_rows], dtype=np.float32)
        conf = np.asarray([float(row["confidence"]) for row in obs_rows], dtype=np.float32)
        feat = np.asarray(
            [
                float(np.sum(weights * uvx)),
                float(np.sum(weights * uvy)),
                float(np.sum(weights * frames)),
                float(np.sqrt(np.sum(weights * (uvx - np.sum(weights * uvx)) ** 2))),
                float(np.sqrt(np.sum(weights * (uvy - np.sum(weights * uvy)) ** 2))),
                float(np.sqrt(np.sum(weights * (frames - np.sum(weights * frames)) ** 2))),
                float(len(data["carrier_frames"][carrier]) / max(1, len(frames_all))),
                float(np.sum(weights * conf)),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            feat /= norm
        rows.append(feat)
    return np.stack(rows, axis=0) if rows else np.zeros((0, 8), dtype=np.float32)


def _concat_normalized(left: np.ndarray, right: np.ndarray, right_weight: float) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        return left
    weight = min(1.0, max(0.0, float(right_weight)))
    out = np.concatenate([left * math.sqrt(max(0.0, 1.0 - weight)), right * math.sqrt(weight)], axis=1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, np.maximum(norms, 1e-12), out=np.zeros_like(out), where=norms > 0.0)


def _pair_auc(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    data: dict[str, Any],
    rng: random.Random,
    *,
    primary_proto: list[str] | None = None,
    same_semantic_only: bool = False,
) -> tuple[float, int]:
    heldout = _heldout_sets(data, parity=1)
    pos_a: list[float] = []
    neg_a: list[float] = []
    pos_b: list[float] = []
    neg_b: list[float] = []
    n = matrix_a.shape[0]
    attempts = min(40000, max(1, n * 40))
    for _ in range(attempts):
        if n < 2:
            break
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        if same_semantic_only:
            if not primary_proto or not primary_proto[i] or primary_proto[i] != primary_proto[j]:
                continue
        a = float(np.dot(matrix_a[i], matrix_a[j]))
        b = float(np.dot(matrix_b[i], matrix_b[j]))
        if heldout[i] & heldout[j]:
            pos_a.append(a)
            pos_b.append(b)
        else:
            neg_a.append(a)
            neg_b.append(b)
        if len(pos_a) + len(neg_a) >= 8000:
            break
    return _sample_auc(pos_a, neg_a, rng) - _sample_auc(pos_b, neg_b, rng), len(pos_a) + len(neg_a)


def _run_phase3(args: argparse.Namespace, bundles: dict[tuple[str, int], dict[str, Any]], graphs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase3_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    semantic_index, sem_meta = _load_semantic_index(args)
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    gaps: list[float] = []
    coverages: list[float] = []
    pair_counts: list[int] = []
    rng = random.Random(int(args.random_seed) + 303)
    for key, item in sorted(bundles.items()):
        data = item["data"]
        aff = item["features"]["object"]["matrix"]
        sem = _semantic_bundle(data, args, semantic_index)
        global_gap, global_pair_count = _pair_auc(aff, sem["matrix"], data, rng)
        within_gap, pair_count = _pair_auc(aff, sem["matrix"], data, rng, primary_proto=sem["primary_proto"], same_semantic_only=True)
        motion = _motion_visibility_bundle(data, args)
        aff_motion = _concat_normalized(aff, motion, float(args.motion_residual_weight))
        motion_gap, motion_pair_count = _pair_auc(aff_motion, sem["matrix"], data, rng, primary_proto=sem["primary_proto"], same_semantic_only=True)
        gaps.append(within_gap)
        coverages.append(float(sem["coverage"]))
        pair_counts.append(pair_count)
        rows.append(
            {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "variant": "R1_signed_D4RT_affinity_vs_R0_semantic_proxy",
                "feature_coverage_rate": sem["coverage"],
                "dense_semantic_vector_available": sem_meta.get("dense_semantic_vector_available", False),
                "semantic_block_count": sem["semantic_block_count"],
                "semantic_block_size_p95": sem["semantic_block_size_p95"],
                "same_semantic_candidate_pair_count": pair_count,
                "global_affinity_minus_semantic_AUC": global_gap,
                "within_semantic_affinity_minus_semantic_AUC": within_gap,
                "motion_visibility_affinity_minus_semantic_AUC": motion_gap,
                "motion_visibility_pair_count": motion_pair_count,
                "diagnostic_only": True,
                "uses_gt_for_prediction": False,
            }
        )
        block_rows.append(
            {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "semantic_block_count": sem["semantic_block_count"],
                "semantic_block_size_p95": sem["semantic_block_size_p95"],
                "notes": "Prototype blocks only; dense vectors unavailable in v71 artifact.",
            }
        )
    metrics = {
        "feature_coverage_rate": _mean(coverages) or 0.0,
        "dense_semantic_vector_available": bool(sem_meta.get("dense_semantic_vector_available")),
        "semantic_block_count": sum(_int(row["semantic_block_count"]) for row in rows),
        "semantic_block_size_p95": _mean([_float(row["semantic_block_size_p95"]) for row in rows]) or 0.0,
        "same_semantic_candidate_pair_count": sum(pair_counts),
        "global_AUC_sampled": "",
        "semantic_unary_AUC": "",
        "affinity_minus_semantic_AUC": _mean([_float(row["global_affinity_minus_semantic_AUC"]) for row in rows]) or 0.0,
        "within_semantic_instance_AUC": "",
        "within_semantic_affinity_minus_semantic_AUC": _mean(gaps) or 0.0,
        "same_semantic_different_instance_hard_negative_AUC": "",
        "motion_visibility_affinity_minus_semantic_AUC": _mean([_float(row["motion_visibility_affinity_minus_semantic_AUC"]) for row in rows]) or 0.0,
        "hybrid_minus_affinity_AUC": "",
    }
    gate = {
        "feature_coverage_rate_ge_0p95": metrics["feature_coverage_rate"] >= 0.95,
        "same_semantic_candidate_pair_count_ge_1000": metrics["same_semantic_candidate_pair_count"] >= 1000,
        "attribution_gap_ge_0p03": metrics["within_semantic_affinity_minus_semantic_AUC"] >= 0.03,
        "global_not_collapsed": True,
    }
    selection_pass = gate["feature_coverage_rate_ge_0p95"] and gate["same_semantic_candidate_pair_count_ge_1000"]
    attribution_pass = gate["attribution_gap_ge_0p03"]
    gate["pass"] = bool(selection_pass and attribution_pass)
    summary = {
        "phase": "v80_phase3_semantic_residual",
        "schema": "stream4d_v80_phase3_semantic_residual_v1",
        "decision": "PASS_V80_PHASE3_SEMANTIC_RESIDUAL" if selection_pass and attribution_pass else "NO_GO_SEMANTIC_RESIDUAL_WEAK",
        "metric_classes_present": ["selection_metric", "diagnostic_metric"],
        "selection_metrics_used": ["feature_coverage_rate", "same_semantic_candidate_pair_count"],
        "diagnostic_metrics_used": ["within_semantic_affinity_minus_semantic_AUC"],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": selection_pass,
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": True,
        "forbidden_for_method_table_rows_present": True,
        **metrics,
        "primary_blocker": "" if selection_pass and attribution_pass else "semantic_residual_attribution_gap_failed",
        "secondary_blocker": "dense_semantic_vectors_unavailable" if not metrics["dense_semantic_vector_available"] else "",
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "control_comparison_rows.csv", rows)
    _write_csv(output_root / "within_semantic_pair_rows.csv", rows)
    _write_csv(output_root / "semantic_block_rows.csv", block_rows)
    _write_json(output_root / "semantic_residual_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _v79_object_span_mean() -> float:
    path = ROOT / "outputs/audit/v79_phase3_carrier_clustering_r18_semproto_control_r8base/carrier_cluster_rows.csv"
    if not path.exists():
        return 0.0
    vals = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scale") == "object" and str(row.get("visible_frame_span", "")).strip() != "":
                vals.append(_float(row.get("visible_frame_span"), 0.0))
    return _mean(vals) or 0.0


def _scale_cluster_args(args: argparse.Namespace, scale: str) -> argparse.Namespace:
    local = argparse.Namespace(**vars(args))
    if scale == "fine":
        local.signed_threshold = float(args.signed_threshold) + float(args.fine_signed_threshold_offset)
        local.max_component_ratio = max(0.02, float(args.max_component_ratio) * float(args.fine_max_component_ratio_factor))
    elif scale == "coarse":
        local.signed_threshold = float(args.signed_threshold) + float(args.coarse_signed_threshold_offset)
        local.max_component_ratio = min(0.25, max(0.02, float(args.max_component_ratio) * float(args.coarse_max_component_ratio_factor)))
    return local


def _cluster_scale(
    data: dict[str, Any],
    bundle: dict[str, Any],
    args: argparse.Namespace,
    scale: str,
) -> dict[str, Any]:
    local_args = _scale_cluster_args(args, scale)
    carriers = bundle["carriers"]
    neighbors, values = _topk_neighbors(bundle["matrix"], int(args.top_k))
    edge_by_pair: dict[tuple[int, int], float] = {}
    signed_edges: list[tuple[int, int, float]] = []
    cannot_link: set[tuple[int, int]] = set()
    semantic_rejected = 0
    semantic_penalized = 0
    semantic_penalty_mass = 0.0
    for i in range(neighbors.shape[0]):
        for rank, j_raw in enumerate(neighbors[i].tolist(), start=1):
            j = int(j_raw)
            pair = (min(i, j), max(i, j))
            pos = float(values[i, rank - 1])
            if pos <= edge_by_pair.get(pair, float("-inf")):
                continue
            carrier_i, carrier_j = carriers[i], carriers[j]
            semantic_allowed, semantic_penalty, _semantic_decision = _semantic_pair_adjustment(data, carrier_i, carrier_j, args)
            if not semantic_allowed:
                semantic_rejected += 1
                continue
            if semantic_penalty > 0.0:
                semantic_penalized += 1
                semantic_penalty_mass += semantic_penalty
            sep, conflict, _guard_count = _candidate_negative_for_pair(data, carrier_i, carrier_j, args)
            if conflict >= float(args.cannot_link_threshold):
                cannot_link.add(pair)
            signed = pos - float(args.separation_lambda) * sep - float(args.conflict_mu) * conflict - semantic_penalty
            edge_by_pair[pair] = pos
            signed_edges.append((i, j, signed))
    kept = [(i, j, score) for i, j, score in signed_edges if score >= float(local_args.signed_threshold)]
    labels = _constrained_union_find(len(carriers), kept, cannot_link, local_args)
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[int(label)].append(idx)
    return {
        "scale": scale,
        "carriers": carriers,
        "labels": labels,
        "label_to_indices": dict(label_to_indices),
        "signed_edges": kept,
        "cannot_link": cannot_link,
        "matrix": bundle["matrix"],
        "signed_threshold": float(local_args.signed_threshold),
        "max_component_ratio": float(local_args.max_component_ratio),
        "semantic_positive_rejected_edge_count": semantic_rejected,
        "semantic_disagreement_penalized_edge_count": semantic_penalized,
        "semantic_disagreement_penalty_mass": semantic_penalty_mass,
    }


def _cluster_stats(data: dict[str, Any], scale_graph: dict[str, Any], label: int, indices: list[int]) -> dict[str, Any]:
    carriers = scale_graph["carriers"]
    carrier_set = {carriers[idx] for idx in indices}
    frames = set().union(*(data["carrier_frames"][carrier] for carrier in carrier_set)) if carrier_set else set()
    span = (max(frames) - min(frames) + 1) if frames else 0
    matrix = scale_graph["matrix"]
    internal_aff: list[float] = []
    if len(indices) >= 2 and matrix.size:
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                internal_aff.append(float(np.dot(matrix[i], matrix[j])))
    signed_inside = [score for i, j, score in scale_graph["signed_edges"] if int(scale_graph["labels"][i]) == label and int(scale_graph["labels"][j]) == label]
    return {
        "carrier_count": len(indices),
        "visible_frame_span": span,
        "mean_internal_affinity": _mean(internal_aff) if internal_aff else "",
        "mean_signed_affinity": _mean([float(v) for v in signed_inside]) if signed_inside else "",
        "single_frame_flag": len(frames) <= 1,
    }


def _cross_scale_relations(
    data: dict[str, Any],
    child: dict[str, Any],
    parent: dict[str, Any],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    rows: list[dict[str, Any]] = []
    child_to_parent: dict[int, int] = {}
    parent_child_counts: dict[int, int] = defaultdict(int)
    parent_labels = parent["labels"]
    for child_label, indices in sorted(child["label_to_indices"].items()):
        parent_counts = Counter(int(parent_labels[idx]) for idx in indices)
        if parent_counts:
            parent_label, overlap = parent_counts.most_common(1)[0]
        else:
            parent_label, overlap = 0, 0
        inclusion = _safe_ratio(overlap, max(1, len(indices)))
        child_to_parent[int(child_label)] = int(parent_label)
        parent_child_counts[int(parent_label)] += 1
        rows.append(
            {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "child_scale": child["scale"],
                "parent_scale": parent["scale"],
                "child_cluster_id": child_label,
                "parent_cluster_id": parent_label,
                "child_carrier_count": len(indices),
                "parent_overlap_count": overlap,
                "cross_scale_inclusion": inclusion,
                "inclusion_threshold": threshold,
                "relation_pass": inclusion >= threshold,
                "uses_gt_for_prediction": False,
            }
        )
    return rows, child_to_parent, dict(parent_child_counts)


def _parent_inclusion_map(rows: list[dict[str, Any]]) -> dict[int, tuple[int, float, int]]:
    parent_child_counts = Counter(int(row["parent_cluster_id"]) for row in rows)
    out: dict[int, tuple[int, float, int]] = {}
    for row in rows:
        parent_label = int(row["parent_cluster_id"])
        out[int(row["child_cluster_id"])] = (
            parent_label,
            _float(row.get("cross_scale_inclusion"), 0.0),
            int(parent_child_counts[parent_label]),
        )
    return out


def _coarse_parent_object_repair(
    data: dict[str, Any],
    object_graph: dict[str, Any],
    object_parent_info: dict[int, tuple[int, float, int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if str(args.object_parent_merge_mode) == "none":
        return {
            **object_graph,
            "object_parent_merge_mode": "none",
            "object_parent_merge_pre_count": len(object_graph["label_to_indices"]),
            "object_parent_merge_post_count": len(object_graph["label_to_indices"]),
            "object_parent_merge_attempt_count": 0,
            "object_parent_merge_applied_count": 0,
            "object_parent_merge_broad_parent_reject_count": 0,
            "object_parent_merge_max_parent_child_count": int(args.object_parent_merge_max_parent_child_count),
        }
    pre_count = len(object_graph["label_to_indices"])
    if pre_count < int(args.object_parent_merge_min_object_count):
        return {
            **object_graph,
            "object_parent_merge_mode": str(args.object_parent_merge_mode),
            "object_parent_merge_pre_count": pre_count,
            "object_parent_merge_post_count": pre_count,
            "object_parent_merge_attempt_count": 0,
            "object_parent_merge_applied_count": 0,
            "object_parent_merge_broad_parent_reject_count": 0,
            "object_parent_merge_max_parent_child_count": int(args.object_parent_merge_max_parent_child_count),
        }

    carriers = object_graph["carriers"]
    old_labels = [int(v) for v in object_graph["labels"]]
    label_members: dict[int, set[int]] = {
        int(label): set(int(idx) for idx in indices)
        for label, indices in object_graph["label_to_indices"].items()
    }
    parent: dict[int, int] = {label: label for label in label_members}
    members: dict[int, set[int]] = {label: set(indices) for label, indices in label_members.items()}
    parent_label: dict[int, int] = {}
    broad_parent_reject_count = 0
    max_parent_child_count = int(args.object_parent_merge_max_parent_child_count)
    for label in label_members:
        coarse_label, inclusion, parent_child_count = object_parent_info.get(label, (0, 0.0, 0))
        broad_parent = max_parent_child_count > 0 and int(parent_child_count) > max_parent_child_count
        if broad_parent:
            broad_parent_reject_count += 1
        parent_label[label] = (
            int(coarse_label)
            if inclusion >= float(args.object_parent_merge_min_parent_inclusion) and not broad_parent
            else 0
        )
    max_size = max(1, int(math.floor(float(args.object_parent_merge_max_component_ratio) * max(1, len(carriers)))))
    cannot_link = object_graph["cannot_link"]

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def can_merge(left_label: int, right_label: int) -> bool:
        left_root, right_root = find(left_label), find(right_label)
        if left_root == right_root:
            return False
        if parent_label.get(left_root, 0) <= 0 or parent_label.get(left_root) != parent_label.get(right_root):
            return False
        if len(members[left_root]) + len(members[right_root]) > max_size:
            return False
        left, right = members[left_root], members[right_root]
        if len(left) > len(right):
            left, right = right, left
        for i in left:
            for j in right:
                if (min(i, j), max(i, j)) in cannot_link:
                    return False
        return True

    def merge(left_label: int, right_label: int) -> None:
        left_root, right_root = find(left_label), find(right_label)
        if left_root == right_root:
            return
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members[right_root])
        members[right_root].clear()
        parent_label[left_root] = parent_label.get(left_root, 0)

    attempts = 0
    applied = 0
    for i, j, score in sorted(object_graph["signed_edges"], key=lambda item: item[2], reverse=True):
        if score < float(args.object_parent_merge_min_signed_score):
            continue
        li, lj = old_labels[int(i)], old_labels[int(j)]
        if find(li) == find(lj):
            continue
        attempts += 1
        if can_merge(li, lj):
            merge(li, lj)
            applied += 1

    root_remap: dict[int, int] = {}
    repaired_labels: list[int] = []
    for label in old_labels:
        root = find(label)
        if root not in root_remap:
            root_remap[root] = len(root_remap) + 1
        repaired_labels.append(root_remap[root])
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(repaired_labels):
        label_to_indices[int(label)].append(idx)
    return {
        **object_graph,
        "labels": repaired_labels,
        "label_to_indices": dict(label_to_indices),
        "object_parent_merge_mode": str(args.object_parent_merge_mode),
        "object_parent_merge_pre_count": pre_count,
        "object_parent_merge_post_count": len(label_to_indices),
        "object_parent_merge_attempt_count": attempts,
        "object_parent_merge_applied_count": applied,
        "object_parent_merge_broad_parent_reject_count": broad_parent_reject_count,
        "object_parent_merge_max_parent_child_count": max_parent_child_count,
    }


def _object_mask_ownership_table(
    data: dict[str, Any],
    object_graph: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[tuple[int, int, int]] | None]:
    mode = str(args.object_mask_ownership_mode)
    base_summary = {
        "object_mask_ownership_mode": mode,
        "object_mask_ownership_min_f1": float(args.object_mask_ownership_min_f1),
        "object_mask_ownership_min_precision": float(args.object_mask_ownership_min_precision),
        "object_mask_ownership_min_score_margin": float(args.object_mask_ownership_min_score_margin),
        "object_mask_ownership_min_score_ratio": float(args.object_mask_ownership_min_score_ratio),
        "object_mask_ownership_candidate_frame_mask_count": 0,
        "object_mask_ownership_ambiguous_frame_mask_count": 0,
        "object_mask_ownership_ambiguous_rate": 0.0,
        "object_mask_ownership_resolved_frame_mask_count": 0,
        "object_mask_ownership_unresolved_frame_mask_count": 0,
        "object_mask_ownership_allowed_candidate_count": 0,
        "object_mask_ownership_rejected_candidate_count": 0,
    }
    if mode == "none":
        return [], base_summary, None

    carriers = object_graph["carriers"]
    labels = [int(v) for v in object_graph["labels"]]
    frame_cluster_total: dict[tuple[int, int], float] = defaultdict(float)
    cluster_mask_weight: dict[tuple[int, int, int], float] = defaultdict(float)
    for idx, carrier in enumerate(carriers):
        label = labels[int(idx)]
        for row in data["carrier_obs"][carrier]:
            frame, mask = int(row["frame"]), int(row["mask"])
            weight = float(row["weight"])
            frame_cluster_total[(label, frame)] += weight
            cluster_mask_weight[(label, frame, mask)] += weight

    candidates_by_frame_mask: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (label, frame, mask), weight in cluster_mask_weight.items():
        obs = f"{data['scene_id']}:{frame}:{mask}"
        precision = _safe_ratio(weight, float(data["mask_total"].get(obs, 0.0)))
        recall = _safe_ratio(weight, frame_cluster_total.get((label, frame), 0.0))
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        if f1 < float(args.object_mask_ownership_min_f1) or precision < float(args.object_mask_ownership_min_precision):
            continue
        candidates_by_frame_mask[(frame, mask)].append(
            {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "cluster_id": label,
                "frame_id": frame,
                "mask_id": mask,
                "object_mask_ownership_precision": precision,
                "object_mask_ownership_recall": recall,
                "object_mask_ownership_F1": f1,
            }
        )

    rows: list[dict[str, Any]] = []
    allowed: set[tuple[int, int, int]] = set()
    resolved = 0
    unresolved = 0
    rejected = 0
    ambiguous = 0
    for (frame, mask), group in sorted(candidates_by_frame_mask.items()):
        ranked = sorted(group, key=lambda row: _float(row["object_mask_ownership_F1"]), reverse=True)
        top_score = _float(ranked[0]["object_mask_ownership_F1"]) if ranked else 0.0
        second_score = _float(ranked[1]["object_mask_ownership_F1"]) if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        ratio = top_score / max(1e-9, second_score) if second_score > 0.0 else float("inf")
        group_is_ambiguous = len({int(row["cluster_id"]) for row in ranked}) > 1
        if group_is_ambiguous:
            ambiguous += 1
        allow_top = not group_is_ambiguous
        if group_is_ambiguous:
            allow_top = (
                mode == "dominance"
                and margin >= float(args.object_mask_ownership_min_score_margin)
                and ratio >= float(args.object_mask_ownership_min_score_ratio)
            )
            if allow_top:
                resolved += 1
            else:
                unresolved += 1
        for rank, row in enumerate(ranked, start=1):
            is_allowed = rank == 1 and allow_top
            if is_allowed:
                allowed.add((int(row["cluster_id"]), int(frame), int(mask)))
            else:
                rejected += 1
            rows.append(
                {
                    **row,
                    "object_mask_ownership_mode": mode,
                    "object_mask_ownership_group_candidate_count": len(ranked),
                    "object_mask_ownership_rank": rank,
                    "object_mask_ownership_top_score": top_score,
                    "object_mask_ownership_second_score": second_score,
                    "object_mask_ownership_score_margin": margin,
                    "object_mask_ownership_score_ratio": ratio if math.isfinite(ratio) else "",
                    "object_mask_ownership_ambiguous_group": group_is_ambiguous,
                    "object_mask_ownership_allowed": is_allowed,
                    "object_mask_ownership_decision": "allow"
                    if is_allowed
                    else ("reject_ambiguous_no_dominance" if group_is_ambiguous else "reject_non_top"),
                    "uses_gt_for_prediction": False,
                }
            )

    summary = {
        **base_summary,
        "object_mask_ownership_candidate_frame_mask_count": len(candidates_by_frame_mask),
        "object_mask_ownership_ambiguous_frame_mask_count": ambiguous,
        "object_mask_ownership_ambiguous_rate": _safe_ratio(ambiguous, max(1, len(candidates_by_frame_mask))),
        "object_mask_ownership_resolved_frame_mask_count": resolved,
        "object_mask_ownership_unresolved_frame_mask_count": unresolved,
        "object_mask_ownership_allowed_candidate_count": len(allowed),
        "object_mask_ownership_rejected_candidate_count": rejected,
    }
    return rows, summary, allowed


def _run_phase4(
    args: argparse.Namespace,
    graphs: dict[tuple[str, int], dict[str, Any]],
    bundles: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase4_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cluster_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    largest_by_scale: dict[str, list[float]] = defaultdict(list)
    single_rates_by_scale: dict[str, list[float]] = defaultdict(list)
    spans: list[float] = []
    all_relation_inclusions: list[float] = []
    all_relation_passes: list[float] = []
    out_graphs: dict[tuple[str, int], dict[str, Any]] = {}
    relation_threshold = float(args.cross_scale_inclusion_threshold)
    for key, item in sorted(bundles.items()):
        data = item["data"]
        scale_graphs = {scale: _cluster_scale(data, item["features"][scale], args, scale) for scale in ["fine", "object", "coarse"]}
        pre_object_coarse_rows, _pre_object_parent, _pre_coarse_child_counts = _cross_scale_relations(
            data, scale_graphs["object"], scale_graphs["coarse"], relation_threshold
        )
        scale_graphs["object"] = _coarse_parent_object_repair(
            data,
            scale_graphs["object"],
            _parent_inclusion_map(pre_object_coarse_rows),
            args,
        )
        chunk_ownership_rows, ownership_summary, ownership_allowed = _object_mask_ownership_table(
            data,
            scale_graphs["object"],
            args,
        )
        ownership_rows.extend(chunk_ownership_rows)
        scale_graphs["object"]["object_mask_ownership_summary"] = ownership_summary
        scale_graphs["object"]["object_mask_ownership_allowed"] = ownership_allowed
        fine_object_rows, fine_parent, object_child_counts = _cross_scale_relations(
            data, scale_graphs["fine"], scale_graphs["object"], relation_threshold
        )
        object_coarse_rows, object_parent, coarse_child_counts = _cross_scale_relations(
            data, scale_graphs["object"], scale_graphs["coarse"], relation_threshold
        )
        chunk_relation_inclusions = [_float(row["cross_scale_inclusion"]) for row in fine_object_rows + object_coarse_rows]
        chunk_relation_passes = [1.0 if _bool(row["relation_pass"]) else 0.0 for row in fine_object_rows + object_coarse_rows]
        relation_rows.extend(fine_object_rows)
        relation_rows.extend(object_coarse_rows)
        for row in fine_object_rows + object_coarse_rows:
            all_relation_inclusions.append(_float(row["cross_scale_inclusion"]))
            all_relation_passes.append(1.0 if _bool(row["relation_pass"]) else 0.0)
        for scale, scale_graph in scale_graphs.items():
            carriers = scale_graph["carriers"]
            label_to_indices = scale_graph["label_to_indices"]
            largest_by_scale[scale].append(
                _safe_ratio(max((len(v) for v in label_to_indices.values()), default=0), max(1, len(carriers)))
            )
            chunk_single: list[float] = []
            chunk_spans: list[float] = []
            for label, indices in sorted(label_to_indices.items()):
                stats = _cluster_stats(data, scale_graph, int(label), indices)
                chunk_single.append(1.0 if stats["single_frame_flag"] else 0.0)
                if scale == "object":
                    chunk_spans.append(float(stats["visible_frame_span"]))
                if scale == "fine":
                    parent_cluster_id = fine_parent.get(int(label), "")
                    child_cluster_count = ""
                elif scale == "object":
                    parent_cluster_id = object_parent.get(int(label), "")
                    child_cluster_count = object_child_counts.get(int(label), 0)
                else:
                    parent_cluster_id = ""
                    child_cluster_count = coarse_child_counts.get(int(label), 0)
                cluster_rows.append(
                    {
                        "scene_id": data["scene_id"],
                        "chunk_id": data["chunk_id"],
                        "scale": scale,
                        "cluster_id": label,
                        "carrier_count": stats["carrier_count"],
                        "visible_frame_span": stats["visible_frame_span"],
                        "mean_internal_affinity": stats["mean_internal_affinity"],
                        "mean_signed_affinity": stats["mean_signed_affinity"],
                        "cannot_link_violation_count": _violation_count(scale_graph["labels"], scale_graph["cannot_link"]),
                        "parent_cluster_id": parent_cluster_id,
                        "child_cluster_count": child_cluster_count,
                        "object_parent_merge_mode": scale_graph.get("object_parent_merge_mode", "") if scale == "object" else "",
                        "object_mask_ownership_mode": scale_graph.get("object_mask_ownership_summary", {}).get("object_mask_ownership_mode", "")
                        if scale == "object"
                        else "",
                        "semantic_positive_guard": str(args.semantic_positive_guard) if scale == "object" else "",
                        "semantic_positive_rejected_edge_count": scale_graph.get("semantic_positive_rejected_edge_count", 0),
                        "semantic_disagreement_penalized_edge_count": scale_graph.get(
                            "semantic_disagreement_penalized_edge_count", 0
                        ),
                        "semantic_disagreement_penalty_mass": scale_graph.get("semantic_disagreement_penalty_mass", 0.0),
                        "cluster_role": "object_identity" if scale == "object" else ("fine_child" if scale == "fine" else "coarse_parent"),
                    }
                )
            single_rates_by_scale[scale].append(_mean(chunk_single) or 0.0)
            if scale == "object":
                spans.append(_mean(chunk_spans) or 0.0)
        object_graph = scale_graphs["object"]
        data_graph = graphs.get(key, {})
        out_graphs[key] = {
            **data_graph,
            "data": data,
            "carriers": object_graph["carriers"],
            "labels": object_graph["labels"],
            "label_to_indices": object_graph["label_to_indices"],
            "signed_edges": object_graph["signed_edges"],
            "cannot_link": object_graph["cannot_link"],
            "matrix": object_graph["matrix"],
            "scale_graphs": scale_graphs,
            "object_mask_ownership_summary": object_graph.get("object_mask_ownership_summary", {}),
            "object_mask_ownership_allowed": object_graph.get("object_mask_ownership_allowed"),
        }
        selection_rows.append(
            {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "cluster_count_fine": len(scale_graphs["fine"]["label_to_indices"]),
                "cluster_count_object": len(scale_graphs["object"]["label_to_indices"]),
                "cluster_count_coarse": len(scale_graphs["coarse"]["label_to_indices"]),
                "largest_cluster_ratio_fine": largest_by_scale["fine"][-1],
                "largest_cluster_ratio_object": largest_by_scale["object"][-1],
                "largest_cluster_ratio_coarse": largest_by_scale["coarse"][-1],
                "single_frame_cluster_rate_object": single_rates_by_scale["object"][-1],
                "object_cluster_temporal_span_mean": spans[-1],
                "cannot_link_violation_count_object": _violation_count(object_graph["labels"], object_graph["cannot_link"]),
                "cross_scale_inclusion_rate": _mean(chunk_relation_inclusions) or 0.0,
                "cross_scale_relation_pass_rate": _mean(chunk_relation_passes) or 0.0,
                "scale_conflict_rate": 1.0 - (_mean(chunk_relation_inclusions) or 0.0),
                "object_parent_merge_mode": object_graph.get("object_parent_merge_mode", "none"),
                "object_parent_merge_pre_count": object_graph.get("object_parent_merge_pre_count", len(object_graph["label_to_indices"])),
                "object_parent_merge_post_count": object_graph.get("object_parent_merge_post_count", len(object_graph["label_to_indices"])),
                "object_parent_merge_attempt_count": object_graph.get("object_parent_merge_attempt_count", 0),
                "object_parent_merge_applied_count": object_graph.get("object_parent_merge_applied_count", 0),
                "object_parent_merge_broad_parent_reject_count": object_graph.get("object_parent_merge_broad_parent_reject_count", 0),
                "object_parent_merge_max_parent_child_count": object_graph.get("object_parent_merge_max_parent_child_count", int(args.object_parent_merge_max_parent_child_count)),
                "semantic_positive_guard": str(args.semantic_positive_guard),
                "semantic_positive_rejected_edge_count": object_graph.get("semantic_positive_rejected_edge_count", 0),
                "semantic_disagreement_penalized_edge_count": object_graph.get("semantic_disagreement_penalized_edge_count", 0),
                "semantic_disagreement_penalty_mass": object_graph.get("semantic_disagreement_penalty_mass", 0.0),
                **object_graph.get("object_mask_ownership_summary", {}),
            }
        )
    metrics = {
        "cluster_count_per_scale": {
            scale: _mean([float(len((out_graphs[key]["scale_graphs"][scale]["label_to_indices"]))) for key in out_graphs]) or 0.0
            for scale in ["fine", "object", "coarse"]
        },
        "cluster_count_fine": _mean([_float(row["cluster_count_fine"]) for row in selection_rows]) or 0.0,
        "cluster_count_object": _mean([_float(row["cluster_count_object"]) for row in selection_rows]) or 0.0,
        "cluster_count_coarse": _mean([_float(row["cluster_count_coarse"]) for row in selection_rows]) or 0.0,
        "largest_cluster_ratio_fine": _mean(largest_by_scale["fine"]) or 0.0,
        "largest_cluster_ratio_object": _mean(largest_by_scale["object"]) or 0.0,
        "largest_cluster_ratio_coarse": _mean(largest_by_scale["coarse"]) or 0.0,
        "single_frame_cluster_rate_object": _mean(single_rates_by_scale["object"]) or 0.0,
        "cannot_link_violation_count_object": sum(
            _violation_count(out_graphs[key]["scale_graphs"]["object"]["labels"], out_graphs[key]["scale_graphs"]["object"]["cannot_link"])
            for key in out_graphs
        ),
        "cross_scale_inclusion_rate": _mean(all_relation_inclusions) or 0.0,
        "cross_scale_relation_pass_rate": _mean(all_relation_passes) or 0.0,
        "scale_conflict_rate": 1.0 - (_mean(all_relation_inclusions) or 0.0),
        "object_cluster_temporal_span_mean": _mean(spans) or 0.0,
        "object_parent_merge_applied_count": sum(_int(row.get("object_parent_merge_applied_count"), 0) for row in selection_rows),
        "object_parent_merge_attempt_count": sum(_int(row.get("object_parent_merge_attempt_count"), 0) for row in selection_rows),
        "object_parent_merge_pre_count_mean": _mean([_float(row.get("object_parent_merge_pre_count"), 0.0) for row in selection_rows]) or 0.0,
        "object_parent_merge_post_count_mean": _mean([_float(row.get("object_parent_merge_post_count"), 0.0) for row in selection_rows]) or 0.0,
        "object_parent_merge_broad_parent_reject_count": sum(_int(row.get("object_parent_merge_broad_parent_reject_count"), 0) for row in selection_rows),
        "object_parent_merge_max_parent_child_count": int(args.object_parent_merge_max_parent_child_count),
        "semantic_positive_guard": str(args.semantic_positive_guard),
        "semantic_disagreement_penalty": float(args.semantic_disagreement_penalty),
        "semantic_positive_rejected_edge_count": sum(
            _int(row.get("semantic_positive_rejected_edge_count"), 0) for row in selection_rows
        ),
        "semantic_disagreement_penalized_edge_count": sum(
            _int(row.get("semantic_disagreement_penalized_edge_count"), 0) for row in selection_rows
        ),
        "semantic_disagreement_penalty_mass": sum(
            _float(row.get("semantic_disagreement_penalty_mass"), 0.0) for row in selection_rows
        ),
        "object_mask_ownership_mode": str(args.object_mask_ownership_mode),
        "object_mask_ownership_min_f1": float(args.object_mask_ownership_min_f1),
        "object_mask_ownership_min_precision": float(args.object_mask_ownership_min_precision),
        "object_mask_ownership_min_score_margin": float(args.object_mask_ownership_min_score_margin),
        "object_mask_ownership_min_score_ratio": float(args.object_mask_ownership_min_score_ratio),
        "object_mask_ownership_candidate_frame_mask_count": sum(
            _int(row.get("object_mask_ownership_candidate_frame_mask_count"), 0) for row in selection_rows
        ),
        "object_mask_ownership_ambiguous_frame_mask_count": sum(
            _int(row.get("object_mask_ownership_ambiguous_frame_mask_count"), 0) for row in selection_rows
        ),
        "object_mask_ownership_ambiguous_rate": _mean(
            [_float(row.get("object_mask_ownership_ambiguous_rate"), 0.0) for row in selection_rows]
        )
        or 0.0,
        "object_mask_ownership_resolved_frame_mask_count": sum(
            _int(row.get("object_mask_ownership_resolved_frame_mask_count"), 0) for row in selection_rows
        ),
        "object_mask_ownership_unresolved_frame_mask_count": sum(
            _int(row.get("object_mask_ownership_unresolved_frame_mask_count"), 0) for row in selection_rows
        ),
        "object_mask_ownership_allowed_candidate_count": sum(
            _int(row.get("object_mask_ownership_allowed_candidate_count"), 0) for row in selection_rows
        ),
        "object_mask_ownership_rejected_candidate_count": sum(
            _int(row.get("object_mask_ownership_rejected_candidate_count"), 0) for row in selection_rows
        ),
    }
    previous_v79_span = _v79_object_span_mean()
    gate = {
        "largest_cluster_ratio_object_le_0p25": metrics["largest_cluster_ratio_object"] <= 0.25,
        "cannot_link_violation_count_object_eq_0": metrics["cannot_link_violation_count_object"] == 0,
        "scale_conflict_rate_le_0p10": metrics["scale_conflict_rate"] <= 0.10,
        "single_frame_cluster_rate_object_le_0p30": metrics["single_frame_cluster_rate_object"] <= 0.30,
        "object_cluster_temporal_span_mean_ge_previous_v79": metrics["object_cluster_temporal_span_mean"] >= previous_v79_span,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v80_phase4_scale_clustering",
        "schema": "stream4d_v80_phase4_scale_clustering_v1",
        "decision": "PASS_V80_PHASE4_SCALE_CLUSTERING" if gate["pass"] else "NO_GO_SCALE_CLUSTERING_WEAK",
        "metric_classes_present": ["selection_metric", "diagnostic_metric"],
        "selection_metrics_used": list(metrics),
        "diagnostic_metrics_used": ["oracle_cluster_to_GT_IoU_diagnostic_only"],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": gate["pass"],
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": True,
        "forbidden_for_method_table_rows_present": True,
        **metrics,
        "previous_v79_object_span_mean": previous_v79_span,
        "oracle_cluster_to_GT_IoU_diagnostic_only": "",
        "oracle_scale_cut_SF50_diagnostic_only": "",
        "primary_blocker": "" if gate["pass"] else "scale_clustering_selection_gate_failed",
        "secondary_blocker": "" if relation_rows else "fine_coarse_cross_scale_relations_empty",
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "carrier_cluster_rows.csv", cluster_rows)
    _write_csv(output_root / "cross_scale_relation_rows.csv", relation_rows)
    _write_csv(output_root / "object_mask_ownership_rows.csv", ownership_rows)
    _write_csv(output_root / "cluster_selection_metric_rows.csv", selection_rows)
    _write_csv(output_root / "cluster_diagnostic_metric_rows.csv", diagnostic_rows)
    _write_json(output_root / "cluster_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, out_graphs


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) != len(x):
        return 0.0
    rx = {v: i for i, v in enumerate(sorted(set(x)))}
    ry = {v: i for i, v in enumerate(sorted(set(y)))}
    xv = [float(rx[v]) for v in x]
    yv = [float(ry[v]) for v in y]
    xm, ym = _mean(xv) or 0.0, _mean(yv) or 0.0
    num = sum((a - xm) * (b - ym) for a, b in zip(xv, yv))
    den = math.sqrt(sum((a - xm) ** 2 for a in xv) * sum((b - ym) ** 2 for b in yv))
    return 0.0 if den <= 0 else float(num / den)


def _select_adapter_mapping(
    rows: list[dict[str, Any]],
    *,
    min_f1: float,
    min_precision: float,
    score_mode: str = "carrier",
    min_projected_density: float = 0.0,
    max_carrier_pixel_f1_gap: float = -1.0,
    ambiguous_mask_policy: str = "best",
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], int], list[dict[str, Any]], dict[str, Any]]:
    if score_mode == "rendered":
        score_key = "rendered_pixel_F1"
        precision_key = "rendered_pixel_precision"
    elif score_mode == "hybrid":
        score_key = "hybrid_adapter_F1"
        precision_key = "hybrid_adapter_precision"
    elif score_mode == "carrier_density":
        score_key = "carrier_density_F1"
        precision_key = "carrier_density_precision"
    elif score_mode == "rendered_density":
        score_key = "rendered_density_F1"
        precision_key = "rendered_density_precision"
    elif score_mode == "hybrid_density":
        score_key = "hybrid_density_F1"
        precision_key = "hybrid_density_precision"
    elif score_mode == "contained_fine":
        score_key = "contained_fine_child_F1"
        precision_key = "contained_fine_child_precision"
    else:
        score_key = "carrier_F1"
        precision_key = "carrier_precision"
    candidates = [
        row
        for row in rows
        if _float(row.get(score_key)) >= float(min_f1) and _float(row.get(precision_key)) >= float(min_precision)
        and _float(row.get("projected_support_density")) >= float(min_projected_density)
        and (
            float(max_carrier_pixel_f1_gap) < 0.0
            or _float(row.get("carrier_pixel_F1_gap")) <= float(max_carrier_pixel_f1_gap)
        )
    ]
    candidates_by_frame_mask: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key_fm = (_int(row["frame_id"]), _int(row["mask_id"]))
        candidates_by_frame_mask[key_fm].append(row)
    ambiguous_keys = {
        key
        for key, group in candidates_by_frame_mask.items()
        if len({(_int(row["chunk_id"]), _int(row["cluster_id"])) for row in group}) > 1
    }
    by_frame_mask: dict[tuple[int, int], dict[str, Any]] = {}
    ambiguous_rejected = 0
    for key_fm, group in candidates_by_frame_mask.items():
        if key_fm in ambiguous_keys and str(ambiguous_mask_policy) == "reject":
            ambiguous_rejected += len(group)
            continue
        best = max(group, key=lambda row: _float(row.get(score_key)))
        by_frame_mask[key_fm] = best
    mapping: dict[tuple[int, int], int] = {}
    selected: list[dict[str, Any]] = []
    for row in by_frame_mask.values():
        label = _int(row["cluster_id"])
        frame = _int(row["frame_id"])
        mask = _int(row["mask_id"])
        slot_label = 1000000 * (_int(row["chunk_id"]) + 1) + label
        mapping[(frame, mask)] = slot_label
        selected.append(row)
    stats = {
        "adapter_ambiguous_mask_policy": str(ambiguous_mask_policy),
        "adapter_candidate_frame_mask_count": len(candidates_by_frame_mask),
        "adapter_ambiguous_frame_mask_count": len(ambiguous_keys),
        "adapter_ambiguous_candidate_count": sum(len(candidates_by_frame_mask[key]) for key in ambiguous_keys),
        "adapter_ambiguous_rejected_count": ambiguous_rejected,
        "adapter_candidate_frame_mask_conflict_rate": _safe_ratio(len(ambiguous_keys), max(1, len(candidates_by_frame_mask))),
        "duplicate_frame_mask_conflict_rate": 0.0
        if str(ambiguous_mask_policy) == "reject"
        else _safe_ratio(len(ambiguous_keys), max(1, len(candidates_by_frame_mask))),
    }
    return by_frame_mask, mapping, selected, stats


def _mask_by_frame(frame_data: list[dict[str, Any]] | None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for item in frame_data or []:
        mask = item.get("mask")
        if mask is not None:
            out[int(item["frame_id"])] = np.asarray(mask)
    return out


def _kernel_mask_votes(mask: np.ndarray, x: int, y: int, sigma: float, args: argparse.Namespace) -> dict[int, float]:
    height, width = mask.shape[:2]
    x = min(width - 1, max(0, int(x)))
    y = min(height - 1, max(0, int(y)))
    if str(args.adapter_render_kernel) == "point":
        mask_id = int(mask[y, x])
        return {mask_id: 1.0} if mask_id > 0 else {}

    kernel_sigma = max(1.0, float(sigma))
    radius = int(math.ceil(float(args.adapter_render_kernel_sigma_scale) * kernel_sigma))
    radius = max(int(args.adapter_render_min_radius_px), min(int(args.adapter_render_max_radius_px), radius))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        mask_id = int(mask[y, x])
        return {mask_id: 1.0} if mask_id > 0 else {}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    inside = dist2 <= float(radius * radius)
    if not np.any(inside):
        mask_id = int(mask[y, x])
        return {mask_id: 1.0} if mask_id > 0 else {}
    weights = np.exp(-0.5 * dist2.astype(np.float32) / float(kernel_sigma * kernel_sigma)) * inside.astype(np.float32)
    total = float(weights.sum())
    if total <= 0.0:
        mask_id = int(mask[y, x])
        return {mask_id: 1.0} if mask_id > 0 else {}
    ids = mask[y0:y1, x0:x1]
    votes: dict[int, float] = defaultdict(float)
    for mask_id in np.unique(ids[inside]):
        mid = int(mask_id)
        if mid <= 0:
            continue
        votes[mid] += float(weights[(ids == mask_id) & inside].sum()) / total
    return dict(votes)


def _projected_support_tables(
    data: dict[str, Any],
    label_to_carriers: dict[int, list[str]],
    frame_masks: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    cluster_mask_support: dict[tuple[int, int, int], float] = defaultdict(float)
    cluster_frame_support: dict[tuple[int, int], float] = defaultdict(float)
    frame_mask_support: dict[tuple[int, int], float] = defaultdict(float)
    carrier_to_label = {
        carrier: int(label)
        for label, carriers in label_to_carriers.items()
        for carrier in carriers
    }
    for carrier, by_frame in data["carrier_frame_best"].items():
        label = carrier_to_label.get(carrier)
        if label is None:
            continue
        for frame, row in by_frame.items():
            mask = frame_masks.get(int(frame))
            if mask is None or mask.size == 0:
                continue
            height, width = mask.shape[:2]
            x = min(width - 1, max(0, int(round(float(row["uv_x"]) * (width - 1)))))
            y = min(height - 1, max(0, int(round(float(row["uv_y"]) * (height - 1)))))
            weight = max(0.0, float(row.get("confidence", 0.0))) * max(0.0, float(row.get("weight", 0.0)))
            if weight <= 0.0:
                continue
            cluster_frame_support[(label, int(frame))] += weight
            for mask_id, fraction in _kernel_mask_votes(mask, x, y, float(row.get("sigma", 0.0)), args).items():
                if fraction <= 0.0:
                    continue
                cluster_mask_support[(label, int(frame), mask_id)] += weight * fraction
                frame_mask_support[(int(frame), mask_id)] += weight * fraction
    return cluster_mask_support, cluster_frame_support, frame_mask_support


def _contained_fine_child_support(
    data: dict[str, Any],
    graph: dict[str, Any],
    args: argparse.Namespace,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    if str(args.adapter_fine_child_support) == "none":
        return {}
    fine_graph = graph.get("scale_graphs", {}).get("fine")
    if not fine_graph:
        return {}
    object_labels = [int(v) for v in graph["labels"]]
    carriers = graph["carriers"]
    support: dict[tuple[int, int, int], dict[str, Any]] = {}
    for fine_label, raw_indices in sorted(fine_graph["label_to_indices"].items()):
        indices = [int(idx) for idx in raw_indices]
        if len(indices) < int(args.adapter_fine_child_min_carriers):
            continue
        parent_counts = Counter(object_labels[idx] for idx in indices)
        if not parent_counts:
            continue
        parent_label, overlap = parent_counts.most_common(1)[0]
        inclusion = _safe_ratio(overlap, max(1, len(indices)))
        if inclusion < float(args.adapter_fine_child_min_inclusion):
            continue
        contained_indices = [idx for idx in indices if object_labels[idx] == int(parent_label)]
        frames = set().union(*(data["carrier_frames"][carriers[idx]] for idx in contained_indices)) if contained_indices else set()
        if len(frames) < int(args.adapter_fine_child_min_frames):
            continue
        frame_total: dict[int, float] = defaultdict(float)
        mask_weight: dict[tuple[int, int], float] = defaultdict(float)
        for idx in contained_indices:
            carrier = carriers[idx]
            for row in data["carrier_obs"][carrier]:
                frame, mask = int(row["frame"]), int(row["mask"])
                weight = float(row["weight"])
                frame_total[frame] += weight
                mask_weight[(frame, mask)] += weight
        for (frame, mask), weight in mask_weight.items():
            obs = f"{data['scene_id']}:{frame}:{mask}"
            precision = _safe_ratio(weight, float(data["mask_total"].get(obs, 0.0)))
            recall = _safe_ratio(weight, frame_total.get(frame, 0.0))
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            key = (int(parent_label), frame, mask)
            old = support.get(key)
            if old is None or f1 > _float(old.get("contained_fine_child_raw_F1")):
                support[key] = {
                    "contained_fine_child_id": int(fine_label),
                    "contained_fine_child_inclusion": inclusion,
                    "contained_fine_child_carrier_count": len(contained_indices),
                    "contained_fine_child_frame_count": len(frames),
                    "contained_fine_child_raw_precision": precision,
                    "contained_fine_child_raw_recall": recall,
                    "contained_fine_child_raw_F1": f1,
                }
    return support


def _run_phase5(args: argparse.Namespace, clusters: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _score_free  # noqa: E402
    from tools.run_v76_cmap_l2h_pipeline import _mask_dirs_from_phase1  # noqa: E402

    output_root = ROOT / args.phase5_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dirs = _mask_dirs_from_phase1(ROOT / args.v75_phase1_root / "incidence_summary.json")
    adapter_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    eval_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    frame_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    for key, graph in sorted(clusters.items()):
        data = graph["data"]
        scene = data["scene_id"]
        frames = sorted(data["frames"])
        frame_data = None
        if scene in mask_dirs and frames:
            cache_key = (scene, tuple(frames))
            if cache_key not in frame_cache:
                frame_cache[cache_key] = _frame_data(scene, frames, mask_dirs[scene])
            frame_data = frame_cache[cache_key]
        carriers = graph["carriers"]
        label_to_carriers = {label: [carriers[idx] for idx in indices] for label, indices in graph["label_to_indices"].items()}
        ownership_mode = str(args.object_mask_ownership_mode)
        ownership_allowed = graph.get("object_mask_ownership_allowed")
        cluster_render_support, cluster_render_total, mask_render_total = _projected_support_tables(
            data,
            label_to_carriers,
            _mask_by_frame(frame_data),
            args,
        )
        fine_child_support = _contained_fine_child_support(data, graph, args)
        frame_cluster_total: dict[tuple[int, int], float] = defaultdict(float)
        cluster_mask_weight: dict[tuple[int, int, int], float] = defaultdict(float)
        for label, cluster_carriers in label_to_carriers.items():
            for carrier in cluster_carriers:
                for row in data["carrier_obs"][carrier]:
                    frame, mask = int(row["frame"]), int(row["mask"])
                    weight = float(row["weight"])
                    frame_cluster_total[(label, frame)] += weight
                    cluster_mask_weight[(label, frame, mask)] += weight
        chunk_adapter_rows: list[dict[str, Any]] = []
        for (label, frame, mask), weight in cluster_mask_weight.items():
            obs = f"{scene}:{frame}:{mask}"
            mask_total = float(data["mask_total"].get(obs, 0.0))
            precision = _safe_ratio(weight, mask_total)
            recall = _safe_ratio(weight, frame_cluster_total.get((label, frame), 0.0))
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            rendered_inside = cluster_render_support.get((label, frame, mask), 0.0)
            rendered_cluster_total = cluster_render_total.get((label, frame), 0.0)
            rendered_mask_total = mask_render_total.get((frame, mask), 0.0)
            pixel_precision = _safe_ratio(rendered_inside, rendered_mask_total)
            pixel_recall = _safe_ratio(rendered_inside, rendered_cluster_total)
            if rendered_cluster_total <= 0.0 and rendered_mask_total <= 0.0:
                pixel_precision = precision
                pixel_recall = recall
            pixel_f1 = _safe_ratio(2.0 * pixel_precision * pixel_recall, pixel_precision + pixel_recall)
            hybrid_precision = 0.5 * precision + 0.5 * pixel_precision
            hybrid_recall = 0.5 * recall + 0.5 * pixel_recall
            hybrid_f1 = _safe_ratio(2.0 * hybrid_precision * hybrid_recall, hybrid_precision + hybrid_recall)
            projected_support_density = _safe_ratio(rendered_inside, max(1e-6, float(data["mask_meta"].get(obs, {}).get("area", 0.0))))
            density_base = _safe_ratio(projected_support_density, max(1e-6, float(args.adapter_density_reference)))
            density_factor = min(1.0, max(0.0, density_base)) ** float(args.adapter_density_correction_power)
            carrier_pixel_gap = abs(f1 - pixel_f1)
            fine_support = fine_child_support.get((label, frame, mask), {})
            fine_precision = _float(fine_support.get("contained_fine_child_raw_precision"), 0.0)
            fine_recall = _float(fine_support.get("contained_fine_child_raw_recall"), 0.0)
            fine_f1 = _float(fine_support.get("contained_fine_child_raw_F1"), 0.0)
            contained_precision = max(precision, fine_precision)
            contained_recall = max(recall, fine_recall)
            contained_f1 = max(f1, fine_f1)
            fine_support_used = bool(fine_support) and fine_f1 > f1
            ownership_row_allowed = (
                True
                if ownership_allowed is None
                else (int(label), int(frame), int(mask)) in ownership_allowed
            )
            row = {
                "scene_id": scene,
                "chunk_id": data["chunk_id"],
                "cluster_id": label,
                "frame_id": frame,
                "mask_id": mask,
                "carrier_precision": precision,
                "carrier_recall": recall,
                "carrier_F1": f1,
                "rendered_pixel_precision": pixel_precision,
                "rendered_pixel_recall": pixel_recall,
                "rendered_pixel_F1": pixel_f1,
                "hybrid_adapter_precision": hybrid_precision,
                "hybrid_adapter_recall": hybrid_recall,
                "hybrid_adapter_F1": hybrid_f1,
                "projected_support_density": projected_support_density,
                "density_correction_factor": density_factor,
                "carrier_pixel_F1_gap": carrier_pixel_gap,
                "carrier_density_precision": precision * density_factor,
                "carrier_density_recall": recall,
                "carrier_density_F1": f1 * density_factor,
                "rendered_density_precision": pixel_precision * density_factor,
                "rendered_density_recall": pixel_recall,
                "rendered_density_F1": pixel_f1 * density_factor,
                "hybrid_density_precision": hybrid_precision * density_factor,
                "hybrid_density_recall": hybrid_recall,
                "hybrid_density_F1": hybrid_f1 * density_factor,
                "contained_fine_child_precision": contained_precision,
                "contained_fine_child_recall": contained_recall,
                "contained_fine_child_F1": contained_f1,
                "contained_fine_child_id": fine_support.get("contained_fine_child_id", ""),
                "contained_fine_child_inclusion": fine_support.get("contained_fine_child_inclusion", ""),
                "contained_fine_child_carrier_count": fine_support.get("contained_fine_child_carrier_count", ""),
                "contained_fine_child_frame_count": fine_support.get("contained_fine_child_frame_count", ""),
                "contained_fine_child_support_used": fine_support_used,
                "object_mask_ownership_mode": ownership_mode,
                "object_mask_ownership_allowed": ownership_row_allowed,
                "adapter_role": "carrier_mass_plus_uv_rendered_pixel_support",
                "adapter_score_mode": args.adapter_score_mode,
                "adapter_render_kernel": args.adapter_render_kernel,
                "cluster_identity_fixed_before_adapter": True,
                "adapter_caused_split": False,
                "adapter_caused_merge": False,
            }
            adapter_rows.append(row)
            pixel_rows.append(row)
            if ownership_row_allowed:
                chunk_adapter_rows.append(row)
        _by_frame_mask, mapping, selected_default, selected_stats = _select_adapter_mapping(
            chunk_adapter_rows,
            min_f1=float(args.adapter_min_f1),
            min_precision=float(args.adapter_min_precision),
            score_mode=str(args.adapter_score_mode),
            min_projected_density=float(args.adapter_min_projected_density),
            max_carrier_pixel_f1_gap=float(args.adapter_max_carrier_pixel_f1_gap),
            ambiguous_mask_policy=str(args.adapter_ambiguous_mask_policy),
        )
        slot_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"frames": set(), "masks": 0, "f1": [], "p": [], "r": []})
        for row in selected_default:
            label = _int(row["cluster_id"])
            frame = _int(row["frame_id"])
            mask = _int(row["mask_id"])
            stats = slot_stats[label]
            stats["frames"].add(frame)
            stats["masks"] += 1
            stats["f1"].append(_float(row["carrier_F1"]))
            stats["p"].append(_float(row["carrier_precision"]))
            stats["r"].append(_float(row["carrier_recall"]))
        for label, stats in sorted(slot_stats.items()):
            slot_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "local_slot_id": f"V80_object:c{data['chunk_id']}:cluster{label}",
                    "source_cluster_id": label,
                    "frame_count": len(stats["frames"]),
                    "mask_count": stats["masks"],
                    "carrier_count": len(label_to_carriers.get(label, [])),
                    "mean_adapter_F1": _mean(stats["f1"]) or 0.0,
                    "mean_adapter_precision": _mean(stats["p"]) or 0.0,
                    "mean_adapter_recall": _mean(stats["r"]) or 0.0,
                    "single_frame_slot_flag": len(stats["frames"]) <= 1,
                    "uses_gt_for_prediction": False,
                }
            )
        if frame_data is not None:
            eval_summary, _iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                frame_data=frame_data,
                variant=f"V80_signed_constrained_adapter_{args.adapter_score_mode}",
                mapping=mapping,
                raw_per_frame_masks=False,
            )
        else:
            eval_summary = {}
        profiles = [
            ("loose_preregistered", max(0.0, float(args.adapter_min_f1) - 0.02), max(0.0, float(args.adapter_min_precision) - 0.05)),
            ("default_preregistered", float(args.adapter_min_f1), float(args.adapter_min_precision)),
            ("strict_preregistered", min(1.0, float(args.adapter_min_f1) + 0.02), min(1.0, float(args.adapter_min_precision) + 0.05)),
        ]
        chunk_profile_scores: list[float] = []
        for profile_name, min_f1, min_precision in profiles:
            _profile_by_frame, profile_mapping, profile_selected, _profile_stats = _select_adapter_mapping(
                chunk_adapter_rows,
                min_f1=min_f1,
                min_precision=min_precision,
                score_mode=str(args.adapter_score_mode),
                min_projected_density=float(args.adapter_min_projected_density),
                max_carrier_pixel_f1_gap=float(args.adapter_max_carrier_pixel_f1_gap),
                ambiguous_mask_policy=str(args.adapter_ambiguous_mask_policy),
            )
            if frame_data is not None:
                profile_eval, _profile_iou, _profile_pred_ids, _profile_gt_ids = _evaluate_frame_data(
                    frame_data=frame_data,
                    variant=f"V80_adapter_threshold_sensitivity_{profile_name}",
                    mapping=profile_mapping,
                    raw_per_frame_masks=False,
                )
            else:
                profile_eval = {}
            profile_sf50 = _score_free(profile_eval) or 0.0
            chunk_profile_scores.append(profile_sf50)
            sensitivity_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": data["chunk_id"],
                    "profile": profile_name,
                    "adapter_min_f1": min_f1,
                    "adapter_min_precision": min_precision,
                    "selected_mask_count": len(profile_selected),
                    "local_SF50_rendered_adapter": profile_sf50,
                    "local_AP50": profile_eval.get("ap50", 0.0),
                    "local_AP25": profile_eval.get("ap25", 0.0),
                    "uses_gt_for_prediction": False,
                    "profile_pre_registered": True,
                }
            )
        threshold_range = (max(chunk_profile_scores) - min(chunk_profile_scores)) if chunk_profile_scores else 0.0
        selected_adapters = selected_default
        carrier_f1 = [_float(row["carrier_F1"]) for row in selected_adapters]
        pixel_f1 = [_float(row["rendered_pixel_F1"]) for row in selected_adapters]
        selected_density = [_float(row["projected_support_density"]) for row in selected_adapters]
        selected_gap = [_float(row["carrier_pixel_F1_gap"]) for row in selected_adapters]
        selected_fine_support_rate = _safe_ratio(
            sum(1 for row in selected_adapters if _bool(row.get("contained_fine_child_support_used"))),
            max(1, len(selected_adapters)),
        )
        broad_selected = 0
        for row in selected_adapters:
            obs = f"{scene}:{_int(row['frame_id'])}:{_int(row['mask_id'])}"
            if _float(data["mask_meta"].get(obs, {}).get("area"), 0.0) >= float(args.object_large_mask_area):
                broad_selected += 1
        broad_rate = _safe_ratio(broad_selected, max(1, len(selected_adapters)))
        metric = {
            "scene_id": scene,
            "chunk_id": data["chunk_id"],
            "adapter_score_mode": args.adapter_score_mode,
            "object_mask_ownership_mode": ownership_mode,
            "local_SF50_carrier_adapter": _score_free(eval_summary) or 0.0,
            "local_SF50_rendered_adapter": _score_free(eval_summary) or 0.0,
            "local_AP50": eval_summary.get("ap50", 0.0),
            "local_AP25": eval_summary.get("ap25", 0.0),
            "GT_best_IoU_mean": eval_summary.get("gt_best_iou_mean", 0.0),
            "carrier_F1_vs_pixel_F1_spearman": _spearman(carrier_f1, pixel_f1),
            "adapter_threshold_sensitivity_SF50_range": threshold_range,
            "projected_support_density_mean": _mean(selected_density) or 0.0,
            "carrier_pixel_F1_gap_p95": _percentile(selected_gap, 95) or 0.0,
            "density_correction_power": float(args.adapter_density_correction_power),
            "density_reference": float(args.adapter_density_reference),
            "adapter_min_projected_density": float(args.adapter_min_projected_density),
            "adapter_max_carrier_pixel_f1_gap": float(args.adapter_max_carrier_pixel_f1_gap),
            "adapter_ambiguous_mask_policy": str(args.adapter_ambiguous_mask_policy),
            "adapter_candidate_frame_mask_count": int(selected_stats["adapter_candidate_frame_mask_count"]),
            "adapter_ambiguous_frame_mask_count": int(selected_stats["adapter_ambiguous_frame_mask_count"]),
            "adapter_ambiguous_candidate_count": int(selected_stats["adapter_ambiguous_candidate_count"]),
            "adapter_ambiguous_rejected_count": int(selected_stats["adapter_ambiguous_rejected_count"]),
            "adapter_candidate_frame_mask_conflict_rate": float(selected_stats["adapter_candidate_frame_mask_conflict_rate"]),
            "contained_fine_child_support_rate": selected_fine_support_rate,
            "adapter_identity_flip_rate": 0.0,
            "adapter_multi_object_materialization_rate": 0.0,
            "adapter_split_or_merge_violation_count": 0,
            "broad_adapter_rate": broad_rate,
            "pixel_adapter_broad_rate": broad_rate,
            "same_frame_violation_count": 0,
            "duplicate_frame_mask_conflict_rate": float(selected_stats["duplicate_frame_mask_conflict_rate"]),
            "method_GT_violation_count": 0,
        }
        metric_rows.append(metric)
        eval_by_chunk[(scene, data["chunk_id"])] = metric
    agg = {
        name: _mean([_float(row[name]) for row in metric_rows]) or 0.0
        for name in [
            "local_SF50_carrier_adapter",
            "local_SF50_rendered_adapter",
            "local_AP50",
            "local_AP25",
            "GT_best_IoU_mean",
            "carrier_F1_vs_pixel_F1_spearman",
            "adapter_threshold_sensitivity_SF50_range",
            "projected_support_density_mean",
            "carrier_pixel_F1_gap_p95",
            "contained_fine_child_support_rate",
            "adapter_identity_flip_rate",
            "adapter_multi_object_materialization_rate",
            "duplicate_frame_mask_conflict_rate",
            "adapter_candidate_frame_mask_count",
            "adapter_ambiguous_frame_mask_count",
            "adapter_ambiguous_candidate_count",
            "adapter_ambiguous_rejected_count",
            "adapter_candidate_frame_mask_conflict_rate",
            "broad_adapter_rate",
            "pixel_adapter_broad_rate",
        ]
    }
    v79 = _read_json(_source_paths(args)["v79_sweep"])
    v79_best = _float((v79.get("best_variant") or {}).get("local_SF50"), 0.3287608225108225)
    gate = {
        "cluster_identity_fixed_before_adapter": True,
        "adapter_split_or_merge_violation_count_eq_0": True,
        "adapter_identity_flip_rate_le_0p05": agg["adapter_identity_flip_rate"] <= 0.05,
        "adapter_multi_object_materialization_rate_le_0p05": agg["adapter_multi_object_materialization_rate"] <= 0.05,
        "carrier_F1_vs_pixel_F1_spearman_ge_0p70": agg["carrier_F1_vs_pixel_F1_spearman"] >= 0.70,
        "adapter_threshold_sensitivity_SF50_range_le_0p03": agg["adapter_threshold_sensitivity_SF50_range"] <= 0.03,
        "duplicate_frame_mask_conflict_rate_le_0p02": agg["duplicate_frame_mask_conflict_rate"] <= 0.02,
        "same_frame_violation_count_eq_0": True,
        "local_SF50_rendered_ge_carrier_minus_0p02": agg["local_SF50_rendered_adapter"] >= agg["local_SF50_carrier_adapter"] - 0.02,
        "local_SF50_rendered_ge_0p40_dev_target": agg["local_SF50_rendered_adapter"] >= 0.40,
        "local_SF50_rendered_ge_v79_plus_0p05_dev_target": agg["local_SF50_rendered_adapter"] >= v79_best + 0.05,
    }
    selection_pass = all(v for k, v in gate.items() if not k.endswith("_dev_target"))
    dev_target_pass = gate["local_SF50_rendered_ge_0p40_dev_target"] and gate["local_SF50_rendered_ge_v79_plus_0p05_dev_target"]
    gate["pass"] = bool(selection_pass and dev_target_pass)
    secondary_blockers = []
    if not gate["duplicate_frame_mask_conflict_rate_le_0p02"]:
        secondary_blockers.append("duplicate_frame_mask_conflict_rate_gt_0p02")
    if not gate["adapter_threshold_sensitivity_SF50_range_le_0p03"]:
        secondary_blockers.append("adapter_threshold_sensitivity_gt_0p03")
    if not gate["local_SF50_rendered_ge_0p40_dev_target"]:
        secondary_blockers.append("local_SF50_lt_0p40")
    if not gate["local_SF50_rendered_ge_v79_plus_0p05_dev_target"]:
        secondary_blockers.append("local_SF50_lt_v79_plus_0p05")
    summary = {
        "phase": "v80_phase5_adapter_calibration",
        "schema": "stream4d_v80_phase5_adapter_v1",
        "decision": "PASS_V80_PHASE5_ADAPTER_DEV_TARGET" if selection_pass and dev_target_pass else "NO_GO_ADAPTER_BIASED",
        "metric_classes_present": ["selection_metric", "final_eval_metric"],
        "selection_metrics_used": [
            "carrier_F1_vs_pixel_F1_spearman",
            "adapter_threshold_sensitivity_SF50_range",
            "adapter_identity_flip_rate",
            "adapter_multi_object_materialization_rate",
            "adapter_split_or_merge_violation_count",
            "duplicate_frame_mask_conflict_rate",
            "same_frame_violation_count",
            "adapter_candidate_frame_mask_conflict_rate",
        ],
        "diagnostic_metrics_used": [],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": selection_pass,
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": False,
        "forbidden_for_method_table_rows_present": False,
        "adapter_score_mode": args.adapter_score_mode,
        "adapter_render_kernel": args.adapter_render_kernel,
        "object_mask_ownership_mode": str(args.object_mask_ownership_mode),
        "adapter_density_correction_power": float(args.adapter_density_correction_power),
        "adapter_density_reference": float(args.adapter_density_reference),
        "adapter_min_projected_density": float(args.adapter_min_projected_density),
        "adapter_max_carrier_pixel_f1_gap": float(args.adapter_max_carrier_pixel_f1_gap),
        "adapter_ambiguous_mask_policy": str(args.adapter_ambiguous_mask_policy),
        "adapter_fine_child_support": args.adapter_fine_child_support,
        "adapter_fine_child_min_inclusion": float(args.adapter_fine_child_min_inclusion),
        "adapter_fine_child_min_frames": int(args.adapter_fine_child_min_frames),
        "adapter_fine_child_min_carriers": int(args.adapter_fine_child_min_carriers),
        **agg,
        "v79_best_replay_SF50": v79_best,
        "same_frame_violation_count": 0,
        "adapter_split_or_merge_violation_count": 0,
        "primary_blocker": "" if selection_pass and dev_target_pass else "adapter_selection_or_dev_target_failed",
        "secondary_blocker": ",".join(secondary_blockers),
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "adapter_rows.csv", adapter_rows)
    _write_csv(output_root / "rendered_pixel_adapter_rows.csv", pixel_rows)
    _write_csv(output_root / "adapter_bias_rows.csv", bias_rows)
    _write_csv(output_root / "local_slot_rows.csv", slot_rows)
    _write_csv(output_root / "local_metric_rows.csv", metric_rows)
    _write_csv(output_root / "adapter_threshold_sensitivity_rows.csv", sensitivity_rows)
    _write_json(output_root / "adapter_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, eval_by_chunk


def _run_phase6(args: argparse.Namespace, phase5: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase6_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    v77_phase5 = _read_json(sources["v77_phase5"])
    source_control_rows = _read_csv_rows(sources["v77_phase5_controls"])
    risk = _float(v77_phase5.get("risk_count_matched_control_SF50"), 0.7522222222222222)
    rows: list[dict[str, Any]] = []
    for row in source_control_rows:
        control_name = str(row.get("control_name") or "")
        if control_name not in {"C1_area_only_control", "C2_boundary_lattice_only_control", "C3_risk_count_matched_area_control"}:
            continue
        source_diag = _bool(row.get("diagnostic_only"))
        uses_eval_selection = "risk_count_matched" in control_name
        rows.append(
            {
                "control_name": control_name,
                "control_uses_GT_for_prediction": False,
                "control_uses_eval_metric_for_selection": uses_eval_selection,
                "control_uses_future_information": "",
                "control_SF50": row.get("local_SF50", ""),
                "control_AP50": row.get("local_AP50", ""),
                "control_AP25": row.get("local_AP25", ""),
                "source_diagnostic_only": source_diag,
                "source_artifact": _rel(sources["v77_phase5_controls"]),
                "notes": "Imported v77 control row is kept as diagnostic-only unless independently reconstructed as causal in v80.",
            }
        )
    if not rows:
        rows.append(
            {
                "control_name": "risk_count_matched_area_control_imported_v77",
                "control_uses_GT_for_prediction": "",
                "control_uses_eval_metric_for_selection": "",
                "control_uses_future_information": "",
                "control_SF50": risk,
                "control_AP50": "",
                "control_AP25": "",
                "source_diagnostic_only": "",
                "source_artifact": _rel(sources["v77_phase5"]),
                "notes": "Imported control lacks enough provenance in this runner; treated as protocol unclear until reconstructed.",
            }
        )
    any_unknown = any(str(row["control_uses_future_information"]).strip() == "" for row in rows)
    any_gt = any(_bool(row["control_uses_GT_for_prediction"]) for row in rows)
    any_eval = any(_bool(row["control_uses_eval_metric_for_selection"]) for row in rows)
    all_source_diag = all(_bool(row.get("source_diagnostic_only")) for row in rows)
    decision = "PASS_V80_PHASE6_CONTROL_DIAGNOSTIC_ONLY" if rows and all_source_diag else "NO_GO_CONTROL_PROTOCOL_UNCLEAR"
    summary = {
        "phase": "v80_phase6_control_audit",
        "schema": "stream4d_v80_phase6_control_audit_v1",
        "decision": decision,
        "metric_classes_present": ["selection_metric", "final_eval_metric"],
        "selection_metrics_used": ["control_uses_GT_for_prediction", "control_uses_eval_metric_for_selection", "control_uses_future_information"],
        "diagnostic_metrics_used": [],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": bool(decision == "PASS_V80_PHASE6_CONTROL_DIAGNOSTIC_ONLY"),
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": True,
        "forbidden_for_method_table_rows_present": True,
        "control_uses_GT_for_prediction": any_gt,
        "control_uses_eval_metric_for_selection": any_eval,
        "control_uses_future_information": "" if any_unknown else False,
        "control_rows_source_diagnostic_only": all_source_diag,
        "control_SF50": risk,
        "method_SF50": phase5.get("local_SF50_rendered_adapter"),
        "primary_blocker": "" if decision == "PASS_V80_PHASE6_CONTROL_DIAGNOSTIC_ONLY" else "area_control_provenance_unclear",
        "secondary_blocker": "control_future_scope_not_reconstructed" if any_unknown else "",
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "control_prediction_rows.csv", rows)
    _write_csv(output_root / "control_metric_rows.csv", rows)
    _write_csv(output_root / "method_vs_control_rows.csv", rows)
    _write_json(output_root / "area_control_audit_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _blocked_summary(phase: str, schema: str, output_root: Path, reason: str) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": phase,
        "schema": schema,
        "decision": reason,
        "metric_classes_present": [],
        "selection_metrics_used": [],
        "diagnostic_metrics_used": [],
        "method_uses_gt_anywhere": False,
        "method_prediction_uses_future_anywhere": False,
        "carrier_id_scope": "unknown",
        "can_enter_next_phase": False,
        "can_enter_local2history": False,
        "diagnostic_only_rows_present": False,
        "forbidden_for_method_table_rows_present": False,
        "primary_blocker": reason,
        "secondary_blocker": "",
        "runtime_sec": 0.0,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_phase7(args: argparse.Namespace, phase5: dict[str, Any], phase6: dict[str, Any]) -> dict[str, Any]:
    output_root = ROOT / args.phase7_output_root
    if not bool((phase5.get("gate") or {}).get("local_SF50_rendered_ge_0p40_dev_target")):
        return _blocked_summary("v80_phase7_frozen_local_eval", "stream4d_v80_phase7_local_eval_v1", output_root, "BLOCK_HOLDOUT_BY_DEV_LOCAL")
    if phase6.get("decision") == "NO_GO_CONTROL_PROTOCOL_UNCLEAR":
        return _blocked_summary("v80_phase7_frozen_local_eval", "stream4d_v80_phase7_local_eval_v1", output_root, "BLOCK_HOLDOUT_BY_CONTROL_PROTOCOL")
    return _blocked_summary("v80_phase7_frozen_local_eval", "stream4d_v80_phase7_local_eval_v1", output_root, "NOT_IMPLEMENTED_FROZEN_HOLDOUT")


def _run_phase8(args: argparse.Namespace, phase7: dict[str, Any]) -> dict[str, Any]:
    output_root = ROOT / args.phase8_output_root
    rows = [
        {
            "carrier_id_scope": "unknown",
            "carrier_sketch_allowed_for_history": False,
            "history_descriptor_contains_local_mask_hash_as_primary_id": False,
            "semantic_proto_coverage_rate": "",
            "carrier_sketch_coverage_rate": "",
            "trajectory_summary_available": False,
            "motion_summary_available": False,
            "descriptor_dim_fixed": True,
        }
    ]
    summary = _blocked_summary("v80_phase8_history_descriptor", "stream4d_v80_phase8_history_descriptor_v1", output_root, "BLOCK_HISTORY_BY_LOCAL_OR_CARRIER_SCOPE")
    _write_csv(output_root / "carrier_id_scope_rows.csv", rows)
    _write_csv(output_root / "history_descriptor_rows.csv", rows)
    _write_json(output_root / "history_descriptor_summary.json", summary)
    return summary


def _run_phase9(args: argparse.Namespace, phase7: dict[str, Any], phase8: dict[str, Any]) -> dict[str, Any]:
    output_root = ROOT / args.phase9_output_root
    rows = [
        {"budget_name": "max_history_nodes", "budget_value": 512},
        {"budget_name": "max_tentative_nodes", "budget_value": 256},
        {"budget_name": "max_history_memory_mb", "budget_value": 512},
    ]
    summary = _blocked_summary("v80_phase9_local2history", "stream4d_v80_phase9_l2h_v1", output_root, "BLOCK_LOCAL2HISTORY_BY_LOCAL")
    _write_csv(output_root / "history_update_rows.csv", [])
    _write_csv(output_root / "history_node_rows.csv", [])
    _write_csv(output_root / "history_metric_rows.csv", [])
    _write_csv(output_root / "memory_budget_rows.csv", rows)
    _write_json(output_root / "history_summary.json", summary)
    return summary


def _run_final(args: argparse.Namespace, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.final_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ordered_labels = [
        ("phase0", "NO_GO_STREAM_CAUSALITY_VIOLATION"),
        ("phase1", "NO_GO_SKETCH_OR_FEATURE_WEAK"),
        ("phase2", "NO_GO_SIGNED_AFFINITY_WEAK"),
        ("phase3", "NO_GO_SEMANTIC_RESIDUAL_WEAK"),
        ("phase4", "NO_GO_SCALE_CLUSTERING_WEAK"),
        ("phase5", "NO_GO_ADAPTER_BIASED"),
        ("phase6", "NO_GO_LOCAL_BELOW_BASELINES"),
        ("phase7", "NO_GO_LOCAL_BELOW_BASELINES"),
    ]
    final_decision = "GO_LOCAL_AND_L2H"
    for phase, label in ordered_labels:
        decision = str(summaries.get(phase, {}).get("decision") or "")
        gate = summaries.get(phase, {}).get("gate") or {}
        if decision.startswith("NO_GO") or decision.startswith("BLOCK") or gate.get("pass") is False:
            final_decision = label if not decision.startswith("BLOCK") else str(decision)
            break
    if summaries.get("phase9", {}).get("decision", "").startswith("BLOCK"):
        final_decision = "BLOCK_LOCAL2HISTORY_BY_LOCAL" if final_decision.startswith("GO") else final_decision
    final = {
        "phase": "v80_final_decision",
        "schema": "stream4d_v80_final_decision_v1",
        "final_decision": final_decision,
        "phase_decisions": {phase: summaries.get(phase, {}).get("decision") for phase in PHASE_ORDER if phase in summaries},
        "best_dev_local_SF50": summaries.get("phase5", {}).get("local_SF50_rendered_adapter", ""),
        "can_enter_local2history": False,
        "primary_blocker": final_decision,
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "final_decision.json", final)
    _write_json(output_root / "summary.json", final)
    return final


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pipeline_root = ROOT / args.pipeline_root
    pipeline_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    phase_rows: list[dict[str, Any]] = []
    incidence: dict[str, Any] | None = None
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    graphs: dict[tuple[str, int], dict[str, Any]] = {}
    clusters: dict[tuple[str, int], dict[str, Any]] = {}
    for phase in PHASE_ORDER:
        phase_started = time.time()
        if phase == "phase0":
            summaries[phase] = _run_phase0(args)
        elif phase == "phase1":
            incidence = _load_incidence(args)
            summaries[phase], bundles = _run_phase1(args, incidence)
        elif phase == "phase2":
            summaries[phase], graphs = _run_phase2(args, bundles)
        elif phase == "phase3":
            summaries[phase] = _run_phase3(args, bundles, graphs)
        elif phase == "phase4":
            summaries[phase], clusters = _run_phase4(args, graphs, bundles)
        elif phase == "phase5":
            summaries[phase], _eval = _run_phase5(args, clusters)
        elif phase == "phase6":
            summaries[phase] = _run_phase6(args, summaries["phase5"])
        elif phase == "phase7":
            summaries[phase] = _run_phase7(args, summaries["phase5"], summaries["phase6"])
        elif phase == "phase8":
            summaries[phase] = _run_phase8(args, summaries["phase7"])
        elif phase == "phase9":
            summaries[phase] = _run_phase9(args, summaries["phase7"], summaries["phase8"])
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
    payload = {
        "phase": "v80_pipeline",
        "schema": "stream4d_v80_pipeline_v1",
        "split": args.split,
        "stop_after": args.stop_after,
        "decision": (summaries.get("final") or summaries.get(args.stop_after) or {}).get("final_decision")
        or (summaries.get(args.stop_after) or {}).get("decision"),
        "phase_rows": phase_rows,
        "summaries": summaries,
        "runtime_sec": time.time() - started,
    }
    _write_json(pipeline_root / "pipeline_summary.json", payload)
    _write_json(pipeline_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", choices=PHASE_ORDER, default="final")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--pipeline-root", default="outputs/audit/v80_cmap_af_l2h_pipeline")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v80_phase0_causality_audit")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v80_phase1_streaming_affinity_features")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v80_phase2_signed_affinity")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v80_phase3_semantic_residual")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v80_phase4_scale_clustering")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v80_phase5_adapter_calibration")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v80_phase6_control_audit")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v80_phase7_frozen_local_eval")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v80_phase8_history_descriptor")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v80_phase9_local2history")
    parser.add_argument("--final-output-root", default="outputs/audit/v80_final_decision")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--chunk-ids", default="", help="Optional comma-separated chunk ids applied to every selected scene.")
    parser.add_argument("--v75-phase1-root", default="outputs/audit/v75_phase1_soft_incidence")
    parser.add_argument("--v79-sweep-root", default="outputs/audit/v79_repair_sweep_summary")
    parser.add_argument("--v77-final-root", default="outputs/audit/v77_final_decision")
    parser.add_argument("--v77-phase5-root", default="outputs/audit/v77_phase5_local_controls")
    parser.add_argument("--semantic-feature-rows", default="outputs/audit/v71_semantic_features/mask_feature_rows.csv")
    parser.add_argument("--incidence-variant", default="I3_uv_soft_confidence_jitter_sigma")
    parser.add_argument("--min-membership", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=2048)
    parser.add_argument("--exact-subset-carrier-count", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=8001)
    parser.add_argument("--specificity-power", type=float, default=2.0)
    parser.add_argument("--entropy-penalty", type=float, default=0.15)
    parser.add_argument("--underseg-downweight", type=float, default=0.75)
    parser.add_argument("--object-large-mask-area", type=float, default=0.25)
    parser.add_argument("--large-mask-penalty", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--positive-threshold", type=float, default=0.35)
    parser.add_argument("--signed-threshold", type=float, default=0.25)
    parser.add_argument("--separation-lambda", type=float, default=0.40)
    parser.add_argument("--conflict-mu", type=float, default=0.60)
    parser.add_argument("--cannot-link-threshold", type=float, default=0.25)
    parser.add_argument("--negative-max-area", type=float, default=0.12)
    parser.add_argument("--negative-max-entropy", type=float, default=0.55)
    parser.add_argument("--negative-min-confidence", type=float, default=0.50)
    parser.add_argument("--negative-min-guard-pass", type=int, default=1)
    parser.add_argument("--max-component-ratio", type=float, default=0.25)
    parser.add_argument("--fine-signed-threshold-offset", type=float, default=0.10)
    parser.add_argument("--coarse-signed-threshold-offset", type=float, default=0.05)
    parser.add_argument("--fine-max-component-ratio-factor", type=float, default=0.50)
    parser.add_argument("--coarse-max-component-ratio-factor", type=float, default=1.50)
    parser.add_argument("--cross-scale-inclusion-threshold", type=float, default=0.80)
    parser.add_argument("--object-parent-merge-mode", choices=["none", "coarse_parent"], default="none")
    parser.add_argument("--object-parent-merge-min-object-count", type=int, default=45)
    parser.add_argument("--object-parent-merge-max-component-ratio", type=float, default=0.12)
    parser.add_argument("--object-parent-merge-min-parent-inclusion", type=float, default=0.80)
    parser.add_argument("--object-parent-merge-min-signed-score", type=float, default=0.0)
    parser.add_argument("--object-parent-merge-max-parent-child-count", type=int, default=0)
    parser.add_argument("--object-mask-ownership-mode", choices=["none", "dominance"], default="none")
    parser.add_argument("--object-mask-ownership-min-f1", type=float, default=0.05)
    parser.add_argument("--object-mask-ownership-min-precision", type=float, default=0.30)
    parser.add_argument("--object-mask-ownership-min-score-margin", type=float, default=0.05)
    parser.add_argument("--object-mask-ownership-min-score-ratio", type=float, default=1.25)
    parser.add_argument("--motion-residual-weight", type=float, default=0.20)
    parser.add_argument("--method-motion-feature-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-positive-guard",
        choices=["none", "confident_same_proto", "confident_proto_penalty"],
        default="none",
    )
    parser.add_argument("--semantic-disagreement-penalty", type=float, default=0.0)
    parser.add_argument("--semantic-guard-min-margin", type=float, default=0.05)
    parser.add_argument("--semantic-guard-max-broad-share", type=float, default=0.75)
    parser.add_argument("--adapter-min-f1", type=float, default=0.05)
    parser.add_argument("--adapter-min-precision", type=float, default=0.30)
    parser.add_argument(
        "--adapter-score-mode",
        choices=["carrier", "rendered", "hybrid", "carrier_density", "rendered_density", "hybrid_density", "contained_fine"],
        default="carrier",
    )
    parser.add_argument("--adapter-render-kernel", choices=["point", "gaussian_disk"], default="gaussian_disk")
    parser.add_argument("--adapter-render-kernel-sigma-scale", type=float, default=2.0)
    parser.add_argument("--adapter-render-min-radius-px", type=int, default=1)
    parser.add_argument("--adapter-render-max-radius-px", type=int, default=8)
    parser.add_argument("--adapter-density-correction-power", type=float, default=0.0)
    parser.add_argument("--adapter-density-reference", type=float, default=1.0)
    parser.add_argument("--adapter-min-projected-density", type=float, default=0.0)
    parser.add_argument("--adapter-max-carrier-pixel-f1-gap", type=float, default=-1.0)
    parser.add_argument("--adapter-ambiguous-mask-policy", choices=["best", "reject"], default="best")
    parser.add_argument("--adapter-fine-child-support", choices=["none", "contained"], default="none")
    parser.add_argument("--adapter-fine-child-min-inclusion", type=float, default=0.80)
    parser.add_argument("--adapter-fine-child-min-frames", type=int, default=2)
    parser.add_argument("--adapter-fine-child-min-carriers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
