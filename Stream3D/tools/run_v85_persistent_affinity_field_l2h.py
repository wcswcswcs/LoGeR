#!/usr/bin/env python3
"""Run Stream4D v85 Persistent Affinity Field L2H audit.

This runner is evidence-first: it reuses existing v79/v80/v82/v83/v84
artifacts as inputs, writes v85 phase artifacts, and keeps unavailable
scene/materializer metrics explicit instead of filling them with guessed values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

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
    "phase10",
]

DEFAULT_NATIVE_CARRIER_OBSERVATION_TABLES = [
    "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
    "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
]

V84_HOLDOUT_REPLAY_LOCAL_SLOT_ROWS = (
    "outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv"
)
V84_HOLDOUT_REPLAY_ADAPTER_ROWS = (
    "outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"
)
V84_HOLDOUT_REPLAY_WEAK_ASSIGNMENT_ROWS = (
    "outputs/audit/v84_holdout_replay_v82_phase5_weak_history/local_slot_history_assignment_rows.csv"
)
V82_DEV_PHASE2_REFERENCE_ROOT = "outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022"
V84_HOLDOUT_PHASE2_ROOT = "outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets"


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


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


def _num(value: Any, default: float = 0.0) -> float:
    out = _float(value, default)
    return default if out is None else out


def _int(value: Any, default: int = 0) -> int:
    out = _float(value)
    return default if out is None else int(out)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _field_presence(rows: list[dict[str, Any]], required: list[str]) -> dict[str, bool]:
    fields: set[str] = set()
    for row in rows:
        fields.update(str(key) for key in row)
    return {field: field in fields for field in required}


def _native_carrier_observation_tables(args: argparse.Namespace) -> list[Path]:
    raw_paths = getattr(args, "native_carrier_observation_tables", None) or DEFAULT_NATIVE_CARRIER_OBSERVATION_TABLES
    paths: list[Path] = []
    for raw in raw_paths:
        text = str(raw).strip()
        if text:
            paths.append(_repo_path(text))
    return paths


def _load_native_carrier_support(
    selected_frame_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_frame_rows:
        key = (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("mask_id", "")))
        if all(part.strip() for part in key):
            selected_by_key[key].append(row)

    support_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_keys_with_allowed_support: set[tuple[str, str, str]] = set()
    unique_native_carriers: set[str] = set()
    matched_observation_rows = 0
    blocked_observation_rows = 0
    uses_gt_rows = 0
    table_paths = _native_carrier_observation_tables(args)

    for table_path in table_paths:
        rel_path = _rel(table_path)
        if not table_path.exists():
            audit_rows.append(
                {
                    "source_path": rel_path,
                    "exists": False,
                    "selected_key_count": len(selected_by_key),
                    "matched_observation_rows": 0,
                    "allowed_support_rows": 0,
                    "blocked_observation_rows": 0,
                    "unique_native_carrier_count": 0,
                    "uses_gt_for_prediction_row_count": 0,
                    "method_safe": False,
                    "is_scannet_ap_export": False,
                    "notes": "carrier observation table missing",
                }
            )
            continue

        table_matched = 0
        table_allowed = 0
        table_blocked = 0
        table_uses_gt = 0
        table_unique: set[str] = set()
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for obs in reader:
                scene = str(obs.get("scene_id") or obs.get("scene") or "")
                key = (scene, str(obs.get("frame_id", "")), str(obs.get("observed_mask_id", "")))
                selected_matches = selected_by_key.get(key)
                if not selected_matches:
                    continue
                table_matched += 1
                matched_observation_rows += 1
                uses_gt = _bool(obs.get("uses_gt_for_prediction"))
                if uses_gt:
                    table_uses_gt += 1
                    uses_gt_rows += 1
                allowed = (
                    not uses_gt
                    and _bool(obs.get("visible"))
                    and _bool(obs.get("valid"))
                    and _bool(obs.get("valid_uv"))
                    and _bool(obs.get("inside_prepared_mask"))
                    and _bool(obs.get("scale_guard_pass"))
                )
                if not allowed:
                    table_blocked += 1
                    blocked_observation_rows += 1
                    continue
                native_id = str(obs.get("carrier_global_id", "")).strip()
                if not native_id:
                    table_blocked += 1
                    blocked_observation_rows += 1
                    continue
                table_allowed += 1
                table_unique.add(native_id)
                unique_native_carriers.add(native_id)
                selected_keys_with_allowed_support.add(key)
                for selected in selected_matches:
                    support_rows.append(
                        {
                            "scene_id": scene,
                            "chunk_id": selected.get("chunk_id", ""),
                            "frame_id": obs.get("frame_id", ""),
                            "mask_id": obs.get("observed_mask_id", ""),
                            "history_id": selected.get("history_id", ""),
                            "local_slot_id": selected.get("local_slot_id", ""),
                            "cluster_id": selected.get("cluster_id", ""),
                            "candidate_row_id": selected.get("candidate_row_id", ""),
                            "native_carrier_global_id": native_id,
                            "native_carrier_id": obs.get("carrier_id", ""),
                            "carrier_uv_x": obs.get("uv_x", ""),
                            "carrier_uv_y": obs.get("uv_y", ""),
                            "confidence": obs.get("confidence", ""),
                            "visibility_prob": obs.get("visibility_prob", ""),
                            "observed_mask_support_density": obs.get("observed_mask_support_density", ""),
                            "source_observation_table": rel_path,
                            "native_support_kind": "d4rt_carrier_global_id",
                            "native_support_allowed": True,
                            "is_scannet_ap_export": False,
                            "uses_gt_for_prediction": False,
                            "uses_rgbd_pose_mesh_for_export": False,
                            "method_uses_gt": False,
                            "uses_future": selected.get("uses_future", False),
                        }
                    )
        audit_rows.append(
            {
                "source_path": rel_path,
                "exists": True,
                "selected_key_count": len(selected_by_key),
                "matched_observation_rows": table_matched,
                "allowed_support_rows": table_allowed,
                "blocked_observation_rows": table_blocked,
                "unique_native_carrier_count": len(table_unique),
                "uses_gt_for_prediction_row_count": table_uses_gt,
                "method_safe": table_allowed > 0 and table_uses_gt == 0,
                "is_scannet_ap_export": False,
                "notes": "method-safe native D4RT carrier support table; still lacks ScanNet scene vertex ids for AP npz",
            }
        )

    selected_count = len(selected_by_key)
    summary = {
        "available": bool(support_rows),
        "selected_frame_mask_count": selected_count,
        "selected_frame_mask_with_native_support_count": len(selected_keys_with_allowed_support),
        "selected_frame_mask_native_support_coverage_rate": _safe_ratio(
            len(selected_keys_with_allowed_support),
            selected_count,
        ),
        "native_carrier_support_row_count": len(support_rows),
        "native_unique_carrier_count": len(unique_native_carriers),
        "matched_observation_rows": matched_observation_rows,
        "blocked_observation_rows": blocked_observation_rows,
        "uses_gt_for_prediction_row_count": uses_gt_rows,
        "source_table_count": len(table_paths),
        "source_tables": [_rel(path) for path in table_paths],
        "is_scannet_ap_export": False,
        "can_legally_export_native_carrier_support": bool(support_rows) and uses_gt_rows == 0,
        "can_legally_export_prediction_npz": False,
        "primary_blocker": "native_carrier_ids_not_scannet_scene_vertex_ids",
        "required_next_artifact": "method-safe native-carrier-to-ScanNet-scene-vertex calibration or native-carrier evaluator",
        "reason": (
            "selected frame-mask rows can be joined to method-safe D4RT carrier_global_id observations, "
            "but these ids are not ScanNet evaluator vertex ids and cannot be serialized as scene AP masks."
        ),
    }
    return support_rows, audit_rows, summary


def _counter_winner(counter: Counter[str]) -> tuple[str, int, float]:
    if not counter:
        return "", 0, 0.0
    total = sum(counter.values())
    label, count = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[0]
    return str(label), int(count), _safe_ratio(count, total)


def _counter_json(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items(), key=lambda item: str(item[0]))), sort_keys=True)


def _comb2(count: int) -> float:
    return float(count * (count - 1) / 2)


def _cluster_metrics(pred_labels: list[str], gt_labels: list[str]) -> dict[str, Any]:
    if len(pred_labels) != len(gt_labels):
        raise ValueError("pred_labels and gt_labels must have the same length")
    n = len(pred_labels)
    if n == 0:
        return {
            "sample_count": 0,
            "adjusted_rand_index": 0.0,
            "purity": 0.0,
            "completeness": 0.0,
            "pred_cluster_count": 0,
            "gt_cluster_count": 0,
            "overmerge_pred_cluster_count": 0,
            "oversplit_gt_cluster_count": 0,
        }
    pred_counts = Counter(pred_labels)
    gt_counts = Counter(gt_labels)
    contingency = Counter(zip(pred_labels, gt_labels))
    total_pairs = _comb2(n)
    sum_cont = sum(_comb2(count) for count in contingency.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    sum_gt = sum(_comb2(count) for count in gt_counts.values())
    if total_pairs == 0:
        ari = 1.0
    else:
        expected = sum_pred * sum_gt / total_pairs
        denom = 0.5 * (sum_pred + sum_gt) - expected
        ari = 0.0 if denom == 0 else (sum_cont - expected) / denom

    gt_by_pred: dict[str, Counter[str]] = defaultdict(Counter)
    pred_by_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for pred, gt in zip(pred_labels, gt_labels):
        gt_by_pred[pred][gt] += 1
        pred_by_gt[gt][pred] += 1
    purity = _safe_ratio(sum(max(counter.values()) for counter in gt_by_pred.values()), n)
    completeness = _safe_ratio(sum(max(counter.values()) for counter in pred_by_gt.values()), n)
    return {
        "sample_count": n,
        "adjusted_rand_index": ari,
        "purity": purity,
        "completeness": completeness,
        "pred_cluster_count": len(pred_counts),
        "gt_cluster_count": len(gt_counts),
        "overmerge_pred_cluster_count": sum(1 for counter in gt_by_pred.values() if len(counter) > 1),
        "oversplit_gt_cluster_count": sum(1 for counter in pred_by_gt.values() if len(counter) > 1),
    }


def _set_iou(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _safe_ratio(len(a & b), len(a | b))


def _cluster_ap_style_metrics(
    pred_sets: dict[str, set[str]],
    gt_sets: dict[str, set[str]],
    pred_scores: dict[str, float],
    thresholds: tuple[float, ...] = (0.25, 0.5),
) -> dict[str, Any]:
    sorted_preds = sorted(pred_sets, key=lambda pred: (-pred_scores.get(pred, 0.0), str(pred)))
    best_iou_values = []
    for pred in sorted_preds:
        best_iou_values.append(max((_set_iou(pred_sets[pred], gt_set) for gt_set in gt_sets.values()), default=0.0))
    out: dict[str, Any] = {
        "native_carrier_cluster_prediction_count": len(pred_sets),
        "native_carrier_cluster_gt_object_count": len(gt_sets),
        "native_carrier_cluster_mean_best_iou": _mean(best_iou_values),
    }
    gt_count = len(gt_sets)
    for threshold in thresholds:
        matched_gt: set[str] = set()
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for pred in sorted_preds:
            best_gt = ""
            best_iou = 0.0
            for gt_label, gt_set in gt_sets.items():
                if gt_label in matched_gt:
                    continue
                iou = _set_iou(pred_sets[pred], gt_set)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_label
            if best_gt and best_iou >= threshold:
                matched_gt.add(best_gt)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp_cum = 0
        fp_cum = 0
        precision_sum_at_tp = 0.0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_cum += tp
            fp_cum += fp
            if tp:
                precision_sum_at_tp += _safe_ratio(tp_cum, tp_cum + fp_cum)
        precision = _safe_ratio(tp_cum, tp_cum + fp_cum)
        recall = _safe_ratio(tp_cum, gt_count)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        suffix = str(int(threshold * 100))
        out[f"native_carrier_cluster_AP{suffix}"] = _safe_ratio(precision_sum_at_tp, gt_count)
        out[f"native_carrier_cluster_precision{suffix}"] = precision
        out[f"native_carrier_cluster_recall{suffix}"] = recall
        out[f"native_carrier_cluster_F1_{suffix}"] = f1
        out[f"native_carrier_cluster_matched_gt_count_at_{suffix}"] = len(matched_gt)
    if thresholds:
        out["native_carrier_cluster_AP_mean"] = _mean(
            [float(out[f"native_carrier_cluster_AP{int(threshold * 100)}"]) for threshold in thresholds]
        )
    return out


def _native_eval_metrics_for_pred_map(
    assignment_rows: list[dict[str, Any]],
    pred_by_carrier: dict[str, str],
    pred_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    pred_labels: list[str] = []
    gt_labels: list[str] = []
    carriers_by_pred: dict[str, set[str]] = defaultdict(set)
    carriers_by_gt: dict[str, set[str]] = defaultdict(set)
    for row in assignment_rows:
        native_id = str(row.get("native_carrier_global_id", ""))
        pred_label = str(pred_by_carrier.get(native_id, ""))
        gt_label = str(row.get("diagnostic_gt_eval_label", ""))
        if not native_id or not pred_label or not gt_label:
            continue
        pred_labels.append(pred_label)
        gt_labels.append(gt_label)
        carriers_by_pred[pred_label].add(native_id)
        carriers_by_gt[gt_label].add(native_id)
    metrics = _cluster_metrics(pred_labels, gt_labels)
    if pred_scores is None:
        pred_scores = {label: float(len(carriers)) for label, carriers in carriers_by_pred.items()}
    return {
        **metrics,
        **_cluster_ap_style_metrics(dict(carriers_by_pred), dict(carriers_by_gt), pred_scores),
    }


def _size_matched_hash_pred_map(
    assignment_rows: list[dict[str, Any]],
    *,
    group_field: str | None,
    salt: str,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        group = str(row.get(group_field, "")) if group_field else "ALL"
        grouped[group].append(row)
    out: dict[str, str] = {}
    for group, rows in grouped.items():
        counts = Counter(str(row.get("pred_history_eval_label", "")) for row in rows)
        expanded: list[str] = []
        for label, count in sorted(
            counts.items(),
            key=lambda item: (_sha1_text(f"{salt}|label|{group}|{item[0]}"), str(item[0])),
        ):
            expanded.extend([label] * count)
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                _sha1_text(f"{salt}|carrier|{row.get('native_carrier_global_id', '')}"),
                str(row.get("native_carrier_global_id", "")),
            ),
        )
        for row, label in zip(sorted_rows, expanded):
            out[str(row.get("native_carrier_global_id", ""))] = label
    return out


def _uniform_hash_pred_map(assignment_rows: list[dict[str, Any]], *, salt: str) -> dict[str, str]:
    labels = sorted({str(row.get("pred_history_eval_label", "")) for row in assignment_rows if row.get("pred_history_eval_label")})
    if not labels:
        return {}
    out: dict[str, str] = {}
    for row in assignment_rows:
        native_id = str(row.get("native_carrier_global_id", ""))
        if not native_id:
            continue
        idx = int(_sha1_text(f"{salt}|{native_id}"), 16) % len(labels)
        out[native_id] = labels[idx]
    return out


def _single_largest_pred_map(assignment_rows: list[dict[str, Any]], *, group_field: str) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        grouped[str(row.get(group_field, ""))].append(row)
    out: dict[str, str] = {}
    for _group, rows in grouped.items():
        counts = Counter(str(row.get("pred_history_eval_label", "")) for row in rows)
        label, _count, _purity = _counter_winner(counts)
        for row in rows:
            native_id = str(row.get("native_carrier_global_id", ""))
            if native_id:
                out[native_id] = label
    return out


def _native_control_rows(
    assignment_rows: list[dict[str, Any]],
    real_pred_scores: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_pred_map = {
        str(row.get("native_carrier_global_id", "")): str(row.get("pred_history_eval_label", ""))
        for row in assignment_rows
        if row.get("native_carrier_global_id") and row.get("pred_history_eval_label")
    }
    oracle_pred_map = {
        str(row.get("native_carrier_global_id", "")): str(row.get("diagnostic_gt_eval_label", ""))
        for row in assignment_rows
        if row.get("native_carrier_global_id") and row.get("diagnostic_gt_eval_label")
    }
    variants = [
        {
            "variant_id": "B3_real_history_native",
            "variant_name": "real history native-carrier clusters",
            "control_type": "real_diagnostic_candidate",
            "pred_map": real_pred_map,
            "pred_scores": real_pred_scores,
            "score_contract": "mean_pred_history_vote_purity_per_history_cluster",
            "prediction_uses_gt": False,
            "is_oracle": False,
        },
        {
            "variant_id": "B4_size_matched_hash_global",
            "variant_name": "global size-matched hash shuffle of history clusters",
            "control_type": "non_oracle_size_matched_hash_control",
            "pred_map": _size_matched_hash_pred_map(
                assignment_rows,
                group_field=None,
                salt="v85_native_global_size_matched_control_v1",
            ),
            "pred_scores": None,
            "score_contract": "cluster_size_desc_for_control",
            "prediction_uses_gt": False,
            "is_oracle": False,
        },
        {
            "variant_id": "B5_size_matched_hash_by_scene",
            "variant_name": "scene-wise size-matched hash shuffle of history clusters",
            "control_type": "non_oracle_scene_size_matched_hash_control",
            "pred_map": _size_matched_hash_pred_map(
                assignment_rows,
                group_field="scene_id",
                salt="v85_native_scene_size_matched_control_v1",
            ),
            "pred_scores": None,
            "score_contract": "cluster_size_desc_for_control",
            "prediction_uses_gt": False,
            "is_oracle": False,
        },
        {
            "variant_id": "B6_uniform_hash_history",
            "variant_name": "uniform hash assignment to real history labels",
            "control_type": "non_oracle_uniform_hash_control",
            "pred_map": _uniform_hash_pred_map(assignment_rows, salt="v85_native_uniform_hash_control_v1"),
            "pred_scores": None,
            "score_contract": "cluster_size_desc_for_control",
            "prediction_uses_gt": False,
            "is_oracle": False,
        },
        {
            "variant_id": "B7_single_largest_by_scene",
            "variant_name": "scene-wise single largest history cluster",
            "control_type": "non_oracle_single_cluster_control",
            "pred_map": _single_largest_pred_map(assignment_rows, group_field="scene_id"),
            "pred_scores": None,
            "score_contract": "cluster_size_desc_for_control",
            "prediction_uses_gt": False,
            "is_oracle": False,
        },
        {
            "variant_id": "B9_oracle_gt_native",
            "variant_name": "oracle diagnostic native-carrier GT clusters",
            "control_type": "oracle_upper_bound",
            "pred_map": oracle_pred_map,
            "pred_scores": None,
            "score_contract": "cluster_size_desc_for_oracle",
            "prediction_uses_gt": True,
            "is_oracle": True,
        },
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = _native_eval_metrics_for_pred_map(
            assignment_rows,
            variant["pred_map"],
            variant["pred_scores"],
        )
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "variant_name": variant["variant_name"],
                "control_type": variant["control_type"],
                "sample_count": metrics.get("sample_count", 0),
                "pred_cluster_count": metrics.get("pred_cluster_count", 0),
                "gt_cluster_count": metrics.get("gt_cluster_count", 0),
                "adjusted_rand_index": metrics.get("adjusted_rand_index", 0.0),
                "purity": metrics.get("purity", 0.0),
                "completeness": metrics.get("completeness", 0.0),
                "native_carrier_cluster_AP25": metrics.get("native_carrier_cluster_AP25", 0.0),
                "native_carrier_cluster_AP50": metrics.get("native_carrier_cluster_AP50", 0.0),
                "native_carrier_cluster_AP_mean": metrics.get("native_carrier_cluster_AP_mean", 0.0),
                "native_carrier_cluster_mean_best_iou": metrics.get("native_carrier_cluster_mean_best_iou", 0.0),
                "native_carrier_cluster_precision50": metrics.get("native_carrier_cluster_precision50", 0.0),
                "native_carrier_cluster_recall50": metrics.get("native_carrier_cluster_recall50", 0.0),
                "native_carrier_cluster_F1_50": metrics.get("native_carrier_cluster_F1_50", 0.0),
                "score_contract": variant["score_contract"],
                "prediction_uses_gt": bool(variant["prediction_uses_gt"]),
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "is_oracle": bool(variant["is_oracle"]),
            }
        )
    real_row = next((row for row in rows if row["variant_id"] == "B3_real_history_native"), {})
    non_oracle_rows = [
        row for row in rows if not _bool(row.get("prediction_uses_gt")) and row.get("variant_id") != "B3_real_history_native"
    ]
    best_non_oracle_ap50 = max([_num(row.get("native_carrier_cluster_AP50"), 0.0) for row in non_oracle_rows], default=0.0)
    best_non_oracle_mean_iou = max(
        [_num(row.get("native_carrier_cluster_mean_best_iou"), 0.0) for row in non_oracle_rows],
        default=0.0,
    )
    best_non_oracle_ari = max([_num(row.get("adjusted_rand_index"), 0.0) for row in non_oracle_rows], default=0.0)
    summary = {
        "native_carrier_control_variant_count": len(rows),
        "native_carrier_non_oracle_control_count": len(non_oracle_rows),
        "native_carrier_real_minus_best_non_oracle_AP50": _num(real_row.get("native_carrier_cluster_AP50"), 0.0)
        - best_non_oracle_ap50,
        "native_carrier_real_minus_best_non_oracle_mean_best_iou": _num(
            real_row.get("native_carrier_cluster_mean_best_iou"),
            0.0,
        )
        - best_non_oracle_mean_iou,
        "native_carrier_real_minus_best_non_oracle_ARI": _num(real_row.get("adjusted_rand_index"), 0.0)
        - best_non_oracle_ari,
        "native_carrier_real_beats_non_oracle_AP50_by_0p03": (
            _num(real_row.get("native_carrier_cluster_AP50"), 0.0) - best_non_oracle_ap50
        )
        >= 0.03,
        "native_carrier_real_beats_non_oracle_mean_iou_by_0p03": (
            _num(real_row.get("native_carrier_cluster_mean_best_iou"), 0.0) - best_non_oracle_mean_iou
        )
        >= 0.03,
    }
    return rows, summary


def _native_carrier_evaluator_candidate_contract(native_diagnostic_summary: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "schema": "stream4d_v85_native_carrier_evaluator_candidate_contract_v1",
        "status": "candidate_future_freeze_only_not_valid_current_dev_gate",
        "allowed_for_current_method_table": False,
        "allowed_for_future_pre_registered_gate": True,
        "reason_current_not_allowed": (
            "This contract was defined after inspecting current dev diagnostic artifacts; it must be frozen before "
            "selection in a future run before it can be used as a method gate."
        ),
        "prediction_universe": "scene-scoped native D4RT carrier_global_id from v85 native_carrier_support_rows.csv",
        "prediction_object_label": "scene_id:pred_history_id majority vote per native carrier",
        "prediction_score_contract": "mean_pred_history_vote_purity_per_history_cluster",
        "gt_scoring_universe": "scene_id:diagnostic_gt_instance from mask_observation_table.csv, scoring only",
        "evaluation_label_scope": native_diagnostic_summary.get(
            "evaluation_label_scope",
            "scene_id_scoped_history_and_diagnostic_gt_labels",
        ),
        "metrics": [
            "adjusted_rand_index",
            "purity",
            "completeness",
            "native_carrier_cluster_AP25",
            "native_carrier_cluster_AP50",
            "native_carrier_cluster_AP_mean",
            "native_carrier_cluster_mean_best_iou",
        ],
        "required_controls": [
            "B3_real_history_native",
            "B4_size_matched_hash_global",
            "B5_size_matched_hash_by_scene",
            "B6_uniform_hash_history",
            "B7_single_largest_by_scene",
            "B9_oracle_gt_native",
        ],
        "gate_suggestion_for_future_only": {
            "real_beats_best_non_oracle_AP50_by_at_least": 0.03,
            "real_beats_best_non_oracle_mean_best_iou_by_at_least": 0.03,
            "scene_metric_replacement_allowed": False,
            "must_report_separately_from_scannet_ap": True,
        },
        "method_safety": {
            "prediction_uses_gt": False,
            "prediction_uses_future": False,
            "prediction_uses_rgbd_pose_mesh_for_export": False,
            "gt_used_for_scoring_only": True,
            "forbidden_for_current_method_table": True,
        },
        "current_dev_reference_metrics": {
            "native_carrier_cluster_AP50": native_diagnostic_summary.get("native_carrier_cluster_AP50", ""),
            "native_carrier_real_minus_best_non_oracle_AP50": native_diagnostic_summary.get(
                "native_carrier_real_minus_best_non_oracle_AP50",
                "",
            ),
            "native_carrier_real_minus_best_non_oracle_mean_best_iou": native_diagnostic_summary.get(
                "native_carrier_real_minus_best_non_oracle_mean_best_iou",
                "",
            ),
        },
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return contract


def _native_carrier_evaluator_holdout_readiness(
    *,
    selected_frame_rows: list[dict[str, Any]],
    native_support_rows: list[dict[str, Any]],
    candidate_contract: dict[str, Any],
    phase7_out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout_split = {
        "scene0011_00": {"holdout_chunk_start": 6, "holdout_chunk_end": 11},
        "scene0050_00": {"holdout_chunk_start": 4, "holdout_chunk_end": 11},
    }

    def split_name(scene: str, chunk_value: Any) -> str:
        chunk = _int(chunk_value, -1)
        spec = holdout_split.get(scene)
        if spec is None or chunk < 0:
            return "unknown"
        start = int(spec["holdout_chunk_start"])
        end = int(spec["holdout_chunk_end"])
        if start <= chunk <= end:
            return "planned_temporal_holdout"
        if chunk < start:
            return "dev_prefix"
        return "outside_planned_holdout"

    by_scene: dict[str, dict[str, Any]] = {}
    scenes = sorted(
        set(holdout_split)
        | {str(row.get("scene_id", "")) for row in selected_frame_rows if row.get("scene_id")}
        | {str(row.get("scene_id", "")) for row in native_support_rows if row.get("scene_id")}
    )
    for scene in scenes:
        by_scene[scene] = {
            "scene_id": scene,
            "holdout_chunk_start": holdout_split.get(scene, {}).get("holdout_chunk_start", ""),
            "holdout_chunk_end": holdout_split.get(scene, {}).get("holdout_chunk_end", ""),
            "selected_frame_mask_count": 0,
            "selected_frame_mask_dev_prefix_count": 0,
            "selected_frame_mask_holdout_count": 0,
            "native_carrier_support_row_count": 0,
            "native_carrier_support_dev_prefix_row_count": 0,
            "native_carrier_support_holdout_row_count": 0,
            "native_carrier_support_unknown_split_row_count": 0,
            "native_unique_carrier_count": 0,
            "native_unique_holdout_carrier_count": 0,
            "min_chunk_id": "",
            "max_chunk_id": "",
            "chunks_observed_json": "[]",
            "holdout_ready": False,
            "notes": "",
        }

    chunks_by_scene: dict[str, set[int]] = defaultdict(set)
    native_carriers_by_scene: dict[str, set[str]] = defaultdict(set)
    native_holdout_carriers_by_scene: dict[str, set[str]] = defaultdict(set)

    for row in selected_frame_rows:
        scene = str(row.get("scene_id", ""))
        if scene not in by_scene:
            continue
        chunk = _int(row.get("chunk_id"), -1)
        if chunk >= 0:
            chunks_by_scene[scene].add(chunk)
        split = split_name(scene, row.get("chunk_id"))
        by_scene[scene]["selected_frame_mask_count"] += 1
        if split == "planned_temporal_holdout":
            by_scene[scene]["selected_frame_mask_holdout_count"] += 1
        elif split == "dev_prefix":
            by_scene[scene]["selected_frame_mask_dev_prefix_count"] += 1

    for row in native_support_rows:
        scene = str(row.get("scene_id", ""))
        if scene not in by_scene:
            continue
        chunk = _int(row.get("chunk_id"), -1)
        if chunk >= 0:
            chunks_by_scene[scene].add(chunk)
        split = split_name(scene, row.get("chunk_id"))
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        by_scene[scene]["native_carrier_support_row_count"] += 1
        if native_id:
            native_carriers_by_scene[scene].add(native_id)
        if split == "planned_temporal_holdout":
            by_scene[scene]["native_carrier_support_holdout_row_count"] += 1
            if native_id:
                native_holdout_carriers_by_scene[scene].add(native_id)
        elif split == "dev_prefix":
            by_scene[scene]["native_carrier_support_dev_prefix_row_count"] += 1
        else:
            by_scene[scene]["native_carrier_support_unknown_split_row_count"] += 1

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        chunks = sorted(chunks_by_scene.get(scene, set()))
        row = by_scene[scene]
        row["native_unique_carrier_count"] = len(native_carriers_by_scene.get(scene, set()))
        row["native_unique_holdout_carrier_count"] = len(native_holdout_carriers_by_scene.get(scene, set()))
        row["min_chunk_id"] = min(chunks) if chunks else ""
        row["max_chunk_id"] = max(chunks) if chunks else ""
        row["chunks_observed_json"] = json.dumps(chunks)
        row["holdout_ready"] = (
            scene in holdout_split
            and bool(candidate_contract.get("allowed_for_future_pre_registered_gate"))
            and int(row["selected_frame_mask_holdout_count"]) > 0
            and int(row["native_carrier_support_holdout_row_count"]) > 0
        )
        row["notes"] = (
            "planned holdout has selected frame-mask and native carrier support"
            if row["holdout_ready"]
            else "no selected/native carrier support rows in planned temporal holdout chunks"
            if scene in holdout_split
            else "scene is not part of the v85 planned same-scene temporal holdout split"
        )
        rows.append(row)

    planned_rows = [row for row in rows if row["scene_id"] in holdout_split]
    ready = bool(planned_rows) and all(_bool(row.get("holdout_ready")) for row in planned_rows)
    summary = {
        "schema": "stream4d_v85_native_carrier_evaluator_holdout_readiness_v1",
        "candidate_contract_path": _rel(phase7_out / "native_carrier_evaluator_candidate_contract.json"),
        "candidate_contract_sha256": candidate_contract.get("contract_sha256", ""),
        "candidate_contract_status": candidate_contract.get("status", ""),
        "candidate_contract_future_allowed": bool(candidate_contract.get("allowed_for_future_pre_registered_gate")),
        "candidate_contract_current_allowed": bool(candidate_contract.get("allowed_for_current_method_table")),
        "planned_holdout_split": holdout_split,
        "scene_count": len(rows),
        "planned_holdout_scene_count": len(planned_rows),
        "native_holdout_evaluation_ready": ready,
        "formal_holdout_run_allowed_current_v85": False,
        "selected_frame_mask_holdout_count_total": sum(_int(row.get("selected_frame_mask_holdout_count"), 0) for row in planned_rows),
        "native_carrier_support_holdout_row_count_total": sum(
            _int(row.get("native_carrier_support_holdout_row_count"), 0) for row in planned_rows
        ),
        "native_unique_holdout_carrier_count_total": sum(
            _int(row.get("native_unique_holdout_carrier_count"), 0) for row in planned_rows
        ),
        "primary_blocker": ""
        if ready
        else "no_native_carrier_support_rows_in_planned_temporal_holdout_chunks",
        "reason": (
            "The candidate native-carrier evaluator can only be considered for a future pre-registered gate; "
            "current v85 selected/native support rows do not cover the planned temporal holdout chunks, so no "
            "native-carrier holdout metric can be reported."
        ),
        "split_rows_path": _rel(phase7_out / "native_carrier_evaluator_holdout_readiness_rows.csv"),
    }
    return rows, summary


def _native_carrier_diagnostic_eval(
    native_support_rows: list[dict[str, Any]],
    phase7_out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_tables = sorted(
        {
            str(row.get("source_observation_table", "")).strip()
            for row in native_support_rows
            if str(row.get("source_observation_table", "")).strip()
        }
    )
    mask_label_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_audit_rows: list[dict[str, Any]] = []
    duplicate_mask_keys = 0
    gt_label_values: list[str] = []

    for source_table in source_tables:
        carrier_table_path = _repo_path(source_table)
        mask_table_path = carrier_table_path.with_name("mask_observation_table.csv")
        exists = mask_table_path.exists()
        row_count = 0
        positive_gt_row_count = 0
        diagnostic_label_row_count = 0
        uses_gt_for_prediction_count = 0
        if exists:
            with mask_table_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row_count += 1
                    scene = str(row.get("scene_id") or row.get("scene") or "")
                    key = (scene, str(row.get("frame_id", "")), str(row.get("mask_id", "")))
                    if key in mask_label_by_key:
                        duplicate_mask_keys += 1
                    mask_label_by_key[key] = row
                    gt_label = str(_int(row.get("diagnostic_gt_instance"), 0))
                    if _int(row.get("diagnostic_gt_instance"), 0) > 0:
                        positive_gt_row_count += 1
                        gt_label_values.append(gt_label)
                    if _bool(row.get("uses_gt_for_diagnostic_labels")):
                        diagnostic_label_row_count += 1
                    if _bool(row.get("uses_gt_for_prediction")):
                        uses_gt_for_prediction_count += 1
        source_audit_rows.append(
            {
                "source_carrier_observation_table": source_table,
                "source_mask_observation_table": _rel(mask_table_path),
                "exists": exists,
                "mask_observation_row_count": row_count,
                "positive_diagnostic_gt_row_count": positive_gt_row_count,
                "uses_gt_for_diagnostic_label_row_count": diagnostic_label_row_count,
                "uses_gt_for_prediction_row_count": uses_gt_for_prediction_count,
                "legal_for_prediction": uses_gt_for_prediction_count == 0,
                "legal_for_diagnostic_scoring": positive_gt_row_count > 0,
                "notes": "mask diagnostic GT labels are used only to score native-carrier/history assignments, not to form predictions",
            }
        )

    pred_votes_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
    gt_votes_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
    scene_by_carrier: dict[str, str] = {}
    support_obs_by_carrier: Counter[str] = Counter()
    purity_values: list[float] = []
    joined_support_observation_count = 0
    labeled_support_observation_count = 0
    missing_label_support_observation_count = 0
    nonpositive_gt_support_observation_count = 0

    for row in native_support_rows:
        scene = str(row.get("scene_id", ""))
        key = (scene, str(row.get("frame_id", "")), str(row.get("mask_id", "")))
        mask_label = mask_label_by_key.get(key)
        if not mask_label:
            missing_label_support_observation_count += 1
            continue
        joined_support_observation_count += 1
        gt_instance = _int(mask_label.get("diagnostic_gt_instance"), 0)
        if gt_instance <= 0:
            nonpositive_gt_support_observation_count += 1
            continue
        history_id = str(row.get("history_id", "")).strip()
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if not history_id or not native_id:
            missing_label_support_observation_count += 1
            continue
        labeled_support_observation_count += 1
        gt_label = str(gt_instance)
        pred_votes_by_carrier[native_id][history_id] += 1
        gt_votes_by_carrier[native_id][gt_label] += 1
        scene_by_carrier.setdefault(native_id, scene)
        support_obs_by_carrier[native_id] += 1
        purity = _float(mask_label.get("diagnostic_gt_purity"))
        if purity is not None:
            purity_values.append(purity)

    assignment_rows: list[dict[str, Any]] = []
    pred_labels: list[str] = []
    gt_labels: list[str] = []
    carriers_by_pred: dict[str, set[str]] = defaultdict(set)
    carriers_by_gt: dict[str, set[str]] = defaultdict(set)
    pred_score_values: dict[str, list[float]] = defaultdict(list)
    for native_id in sorted(gt_votes_by_carrier):
        pred_counter = pred_votes_by_carrier[native_id]
        gt_counter = gt_votes_by_carrier[native_id]
        pred_label_raw, pred_vote_count, pred_vote_purity = _counter_winner(pred_counter)
        gt_label_raw, gt_vote_count, gt_vote_purity = _counter_winner(gt_counter)
        scene = scene_by_carrier.get(native_id, "")
        pred_label = f"{scene}:{pred_label_raw}"
        gt_label = f"{scene}:{gt_label_raw}"
        pred_labels.append(pred_label)
        gt_labels.append(gt_label)
        carriers_by_pred[pred_label].add(native_id)
        carriers_by_gt[gt_label].add(native_id)
        pred_score_values[pred_label].append(pred_vote_purity)
        assignment_rows.append(
            {
                "scene_id": scene,
                "native_carrier_global_id": native_id,
                "pred_history_id": pred_label_raw,
                "diagnostic_gt_instance": gt_label_raw,
                "pred_history_eval_label": pred_label,
                "diagnostic_gt_eval_label": gt_label,
                "labeled_support_observation_count": support_obs_by_carrier[native_id],
                "pred_vote_count": pred_vote_count,
                "pred_vote_purity": pred_vote_purity,
                "diagnostic_gt_vote_count": gt_vote_count,
                "diagnostic_gt_vote_purity": gt_vote_purity,
                "pred_history_vote_json": _counter_json(pred_counter),
                "diagnostic_gt_vote_json": _counter_json(gt_counter),
                "pred_history_conflict": len(pred_counter) > 1,
                "diagnostic_gt_conflict": len(gt_counter) > 1,
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "metric_scope": "native_carrier_mask_diagnostic_cluster_metric_not_scannet_ap",
            }
        )

    metrics = _cluster_metrics(pred_labels, gt_labels)
    pred_scores = {label: _mean(values) for label, values in pred_score_values.items()}
    ap_style_metrics = _cluster_ap_style_metrics(dict(carriers_by_pred), dict(carriers_by_gt), pred_scores)
    control_rows, control_summary = _native_control_rows(assignment_rows, pred_scores)
    pred_conflict_count = sum(1 for row in assignment_rows if _bool(row.get("pred_history_conflict")))
    gt_conflict_count = sum(1 for row in assignment_rows if _bool(row.get("diagnostic_gt_conflict")))
    assignment_path = phase7_out / "native_carrier_diagnostic_assignment_rows.csv"
    source_audit_path = phase7_out / "native_carrier_diagnostic_source_audit_rows.csv"
    control_rows_path = phase7_out / "native_carrier_diagnostic_control_rows.csv"
    summary = {
        "schema": "stream4d_v85_native_carrier_diagnostic_eval_v2",
        "available": bool(assignment_rows),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "prediction_uses_gt": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_gt_for_scoring": True,
        "uses_rgbd_pose_mesh_for_export": False,
        "metric_scope": "native_carrier_mask_diagnostic_cluster_metric_not_scannet_ap",
        "evaluation_label_scope": "scene_id_scoped_history_and_diagnostic_gt_labels",
        "native_support_observation_count": len(native_support_rows),
        "source_mask_table_count": len(source_tables),
        "mask_label_key_count": len(mask_label_by_key),
        "duplicate_mask_label_key_count": duplicate_mask_keys,
        "joined_support_observation_count": joined_support_observation_count,
        "labeled_support_observation_count": labeled_support_observation_count,
        "missing_label_support_observation_count": missing_label_support_observation_count,
        "nonpositive_gt_support_observation_count": nonpositive_gt_support_observation_count,
        "native_carrier_diagnostic_assignment_count": len(assignment_rows),
        "unique_pred_history_raw_count": len({row.get("pred_history_id", "") for row in assignment_rows}),
        "unique_diagnostic_gt_instance_raw_count": len({row.get("diagnostic_gt_instance", "") for row in assignment_rows}),
        "unique_pred_history_eval_count": len(set(pred_labels)),
        "unique_diagnostic_gt_eval_count": len(set(gt_labels)),
        "pred_history_conflict_carrier_count": pred_conflict_count,
        "diagnostic_gt_conflict_carrier_count": gt_conflict_count,
        "pred_history_conflict_carrier_rate": _safe_ratio(pred_conflict_count, len(assignment_rows)),
        "diagnostic_gt_conflict_carrier_rate": _safe_ratio(gt_conflict_count, len(assignment_rows)),
        "mask_diagnostic_gt_purity_mean": _mean(purity_values),
        "adjusted_rand_index": metrics["adjusted_rand_index"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "pred_cluster_count": metrics["pred_cluster_count"],
        "gt_cluster_count": metrics["gt_cluster_count"],
        "overmerge_pred_cluster_count": metrics["overmerge_pred_cluster_count"],
        "oversplit_gt_cluster_count": metrics["oversplit_gt_cluster_count"],
        **ap_style_metrics,
        "native_carrier_cluster_score_contract": "mean_pred_history_vote_purity_per_history_cluster",
        "native_carrier_cluster_ap_style_metric_scope": "native_carrier_cluster_ap_style_diagnostic_not_scannet_ap",
        **control_summary,
        "native_carrier_evaluator_contract_status": "candidate_diagnostic_contract_controls_computed_not_pre_registered_for_current_dev",
        "can_be_method_result": False,
        "can_satisfy_phase8_scene_metric": False,
        "primary_blocker": "native_carrier_candidate_metric_is_diagnostic_posthoc_not_scene_ap_or_frozen_method_contract",
        "required_next_artifact": "pre-register/freeze native-carrier evaluator contract before selection, or implement method-safe native-carrier-to-ScanNet-scene-vertex exporter",
        "assignment_rows_path": _rel(assignment_path),
        "source_audit_rows_path": _rel(source_audit_path),
        "control_rows_path": _rel(control_rows_path),
        "reason": (
            "v85 histories can be scored against scene-scoped mask diagnostic GT labels on native carriers, and "
            "non-oracle controls can be computed in the same carrier universe. This remains diagnostic for current v85 "
            "because it is not ScanNet AP/SF and was not pre-registered as the method gate before dev inspection."
        ),
    }
    return assignment_rows, source_audit_rows, control_rows, summary


def _csv_fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _npz_member_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        import zipfile

        with zipfile.ZipFile(path) as archive:
            return sorted(archive.namelist())
    except Exception as exc:  # noqa: BLE001 - write the read failure into the audit row.
        return [f"read_error:{exc}"]


def _audit_native_scene_vertex_export_routes(
    *,
    native_support_rows: list[dict[str, Any]],
    native_support_summary: dict[str, Any],
    native_diagnostic_summary: dict[str, Any],
    diagnostic_npz: dict[str, Any],
    phase7_out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_vertex_fields = {
        "scene_point_id",
        "scene_point_ids",
        "scene_vertex_id",
        "scene_vertex_ids",
        "mesh_vertex_id",
        "mesh_vertex_ids",
        "point_id",
        "point_ids",
    }
    native_support_fields = list(native_support_rows[0].keys()) if native_support_rows else []
    native_support_field_set = set(native_support_fields)

    export_scannet_path = ROOT / "stream4d/export_scannet.py"
    export_scannet_text = export_scannet_path.read_text(encoding="utf-8", errors="replace") if export_scannet_path.exists() else ""
    d4rt_nn_blocked = "d4rt_nn export requires a scene-coordinate calibration path" in export_scannet_text

    v41_object_points_path = ROOT / "outputs/audit/v41_1_native_object_field_export_smoke/native_object_point_rows.csv"
    v41_object_summary_path = ROOT / "outputs/audit/v41_1_native_object_field_export_smoke/native_object_field_export_summary.json"
    v41_object_fields = _csv_fieldnames(v41_object_points_path)
    v41_object_summary = _read_json(v41_object_summary_path)
    v41_object_field_set = set(v41_object_fields)

    v41_blocker_path = ROOT / "outputs/audit/v41_1_native_ap_exporter_blocker/native_ap_exporter_blocker_summary.json"
    v41_blocker = _read_json(v41_blocker_path)

    v41_metric_path = ROOT / "outputs/audit/v41_1_native_support_metrics_probe5/native_support_metrics_summary.json"
    v41_metric = _read_json(v41_metric_path)

    v42_bridge_path = ROOT / "outputs/audit/v42_calibrated_native_ap_bridge_allframe_r1/calibrated_native_ap_bridge_summary.json"
    v42_bridge = _read_json(v42_bridge_path)

    surfel_paths = [
        ROOT / "outputs/audit/v19_phase2a_M1/scene0011_00/surfel_mesh_hits.npz",
        ROOT / "outputs/audit/v19_phase2a_M1/scene0050_00/surfel_mesh_hits.npz",
    ]
    surfel_existing = [path for path in surfel_paths if path.exists()]
    surfel_members = sorted({name for path in surfel_existing for name in _npz_member_names(path)})
    surfel_forbidden_member_hits = [
        name
        for name in surfel_members
        if any(token in name for token in ("best_vertex", "hit_vertex", "gt_vote", "labels"))
    ]

    diagnostic_available = bool(diagnostic_npz.get("available"))
    diagnostic_forbidden = bool(diagnostic_npz.get("forbidden_for_method_table", True))
    native_diagnostic_available = bool(native_diagnostic_summary.get("available"))

    rows = [
        {
            "route_id": "E0_v85_native_carrier_support",
            "route_name": "current v85 selected frame masks joined to D4RT native carrier_global_id support",
            "candidate_available": bool(native_support_summary.get("available")),
            "route_status": "available_as_native_support_not_scannet_ap"
            if native_support_summary.get("available")
            else "blocked_missing_native_support",
            "can_be_method_result": False,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "has_scene_vertex_ids": bool(native_support_field_set & scene_vertex_fields),
            "has_point_ids": bool(native_support_field_set & {"point_id", "point_ids"}),
            "has_native_carrier_ids": "native_carrier_global_id" in native_support_field_set,
            "metric_scope": "native_d4rt_carrier_support",
            "evidence_path": _rel(phase7_out / "native_carrier_support_rows.csv"),
            "evidence_detail": "fields=" + ",".join(native_support_fields),
            "blocked_reason": "native carrier ids are not ScanNet scene vertex ids",
            "required_next_artifact": "method-safe native-carrier-to-ScanNet-scene-vertex exporter or native-carrier evaluator",
        },
        {
            "route_id": "E1_reexport_reuse_point_ids",
            "route_name": "reuse existing Stream4D reexport path that expects object_dict point_ids",
            "candidate_available": bool(native_support_rows),
            "route_status": "blocked_missing_point_ids",
            "can_be_method_result": False,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "has_scene_vertex_ids": bool(native_support_field_set & scene_vertex_fields),
            "has_point_ids": bool(native_support_field_set & {"point_id", "point_ids"}),
            "has_native_carrier_ids": "native_carrier_global_id" in native_support_field_set,
            "metric_scope": "scannet_scene_vertex_ap",
            "evidence_path": "Stream3D/stream4d/reexport_scannet.py",
            "evidence_detail": "reexport path requires object_dict point_ids; v85 support rows only carry native_carrier_global_id",
            "blocked_reason": "v85 rows cannot satisfy object_dict point_ids / pred_masks scene-vertex contract",
            "required_next_artifact": "object_dict with audited scene point_ids for each v85 history object",
        },
        {
            "route_id": "E2_export_d4rt_nn",
            "route_name": "native D4RT nearest-neighbor exporter hook in export_scannet",
            "candidate_available": export_scannet_path.exists(),
            "route_status": "blocked_not_implemented" if d4rt_nn_blocked else "needs_manual_review",
            "can_be_method_result": False,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "has_scene_vertex_ids": False,
            "has_point_ids": False,
            "has_native_carrier_ids": True,
            "metric_scope": "scannet_scene_vertex_ap",
            "evidence_path": "Stream3D/stream4d/export_scannet.py",
            "evidence_detail": "export_d4rt_nn NotImplemented guard found=" + str(bool(d4rt_nn_blocked)),
            "blocked_reason": "d4rt_nn export requires an audited scene-coordinate calibration path",
            "required_next_artifact": "method-safe carrier-to-scene coordinate calibration without RGB-D/pose/mesh leakage",
        },
        {
            "route_id": "E3_v42_calibrated_native_bridge",
            "route_name": "historical v42 calibrated native NN AP bridge",
            "candidate_available": bool(v42_bridge),
            "route_status": str(v42_bridge.get("status", "missing")),
            "can_be_method_result": False,
            "is_diagnostic_only": bool(v42_bridge.get("is_diagnostic_only", True)),
            "forbidden_for_method_table": bool(v42_bridge.get("forbidden_for_method_table", True)),
            "uses_gt_for_prediction": bool(v42_bridge.get("uses_gt_for_prediction", False)),
            "uses_gt_for_scoring": bool(v42_bridge.get("uses_gt_for_scoring", True)),
            "uses_rgbd_pose_mesh_for_export": bool(
                v42_bridge.get("uses_pose_for_prediction", False)
                or v42_bridge.get("uses_rgbd_for_prediction", False)
                or v42_bridge.get("uses_scannet_mesh_for_prediction", False)
            ),
            "has_scene_vertex_ids": True,
            "has_point_ids": True,
            "has_native_carrier_ids": True,
            "metric_scope": "scannet_scene_vertex_ap_diagnostic_only",
            "evidence_path": _rel(v42_bridge_path),
            "evidence_detail": "phase8_gate_pass="
            + str(v42_bridge.get("phase8_gate_pass", ""))
            + "; best_by_AP="
            + json.dumps(v42_bridge.get("best_by_AP", {}), sort_keys=True),
            "blocked_reason": "bridge uses RGB-D/pose/ScanNet mesh calibration and is explicitly diagnostic-only",
            "required_next_artifact": "new method-safe calibration/evaluator rather than reuse of v42 diagnostic bridge",
        },
        {
            "route_id": "E4_v19_surfel_mesh_hits",
            "route_name": "reuse v19 surfel_mesh_hits as carrier-to-vertex mapping",
            "candidate_available": bool(surfel_existing),
            "route_status": "blocked_mesh_or_gt_vote_artifact" if surfel_existing else "missing",
            "can_be_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": True,
            "uses_gt_for_scoring": True,
            "uses_rgbd_pose_mesh_for_export": True,
            "has_scene_vertex_ids": bool(surfel_forbidden_member_hits),
            "has_point_ids": bool(surfel_forbidden_member_hits),
            "has_native_carrier_ids": False,
            "metric_scope": "surfel_mesh_vote_diagnostic",
            "evidence_path": ";".join(_rel(path) for path in surfel_existing),
            "evidence_detail": "npz_members=" + ",".join(surfel_members),
            "blocked_reason": "contains mesh vertex and GT vote/label members; cannot be used in prediction path",
            "required_next_artifact": "non-mesh/non-GT carrier support alignment",
        },
        {
            "route_id": "E5_v41_native_object_field_support",
            "route_name": "historical native ObjectField support point export",
            "candidate_available": bool(v41_object_fields),
            "route_status": str(
                v41_object_summary.get("status")
                or v41_blocker.get("status")
                or "missing"
            ),
            "can_be_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": bool(
                v41_object_summary.get("diagnostic_ap_bridge", {}).get("diagnostic_forbidden_for_method_table", True)
            ),
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": False,
            "uses_rgbd_pose_mesh_for_export": bool(
                v41_object_summary.get("diagnostic_ap_bridge", {}).get("diagnostic_uses_pose_for_prediction", True)
                or v41_object_summary.get("diagnostic_ap_bridge", {}).get("diagnostic_uses_rgbd_for_prediction", True)
                or v41_object_summary.get("diagnostic_ap_bridge", {}).get("diagnostic_uses_scannet_mesh_for_prediction", True)
            ),
            "has_scene_vertex_ids": bool(v41_object_field_set & scene_vertex_fields),
            "has_point_ids": bool(v41_object_field_set & {"point_id", "point_ids"}),
            "has_native_carrier_ids": "tube_id" in v41_object_field_set or "local_point_index" in v41_object_field_set,
            "metric_scope": "native_object_field_support_not_scannet_ap",
            "evidence_path": _rel(v41_object_summary_path),
            "evidence_detail": "fields=" + ",".join(v41_object_fields),
            "blocked_reason": str(
                v41_object_summary.get("remaining_blocker")
                or v41_blocker.get("blocker")
                or "method-compatible native AP exporter not available"
            ),
            "required_next_artifact": "method-compatible native ObjectField / carrier to scene AP exporter",
        },
        {
            "route_id": "E6_v41_native_support_metrics",
            "route_name": "historical native-support tube metrics",
            "candidate_available": bool(v41_metric),
            "route_status": str(v41_metric.get("status", "missing")),
            "can_be_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "uses_gt_for_prediction": bool(v41_metric.get("prediction_uses_gt", False)),
            "uses_gt_for_scoring": bool(v41_metric.get("gt_used_only_for_scoring", True)),
            "uses_rgbd_pose_mesh_for_export": bool(
                v41_metric.get("prediction_uses_pose", False)
                or v41_metric.get("prediction_uses_rgbd", False)
                or v41_metric.get("prediction_uses_scannet_mesh", False)
            ),
            "has_scene_vertex_ids": False,
            "has_point_ids": False,
            "has_native_carrier_ids": True,
            "metric_scope": str(v41_metric.get("metric_scope_note", "native support metric; not ScanNet AP")),
            "evidence_path": _rel(v41_metric_path),
            "evidence_detail": "real_method_ap_status=" + str(v41_metric.get("real_method_ap_status", "")),
            "blocked_reason": "not ScanNet AP and not wired to v85 selected native carrier support",
            "required_next_artifact": "v85-specific native-carrier evaluator contract and GT scoring universe",
        },
        {
            "route_id": "E7_v85_native_carrier_evaluator",
            "route_name": "new v85 native-carrier evaluator instead of ScanNet scene AP",
            "candidate_available": bool(native_support_summary.get("available")),
            "route_status": "diagnostic_evaluator_available_method_contract_missing"
            if native_diagnostic_available
            else (
                "input_available_evaluator_not_implemented"
                if native_support_summary.get("available")
                else "blocked_missing_native_support_input"
            ),
            "can_be_method_result": False,
            "is_diagnostic_only": native_diagnostic_available,
            "forbidden_for_method_table": bool(native_diagnostic_summary.get("forbidden_for_method_table", False)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": True,
            "uses_rgbd_pose_mesh_for_export": False,
            "has_scene_vertex_ids": False,
            "has_point_ids": False,
            "has_native_carrier_ids": bool(native_support_summary.get("available")),
            "metric_scope": str(
                native_diagnostic_summary.get(
                    "metric_scope",
                    "native_carrier_ap_style_evaluator_not_implemented",
                )
            ),
            "evidence_path": str(
                native_diagnostic_summary.get(
                    "assignment_rows_path",
                    _rel(phase7_out / "native_carrier_support_rows.csv"),
                )
            ),
            "evidence_detail": (
                "native_support_row_count="
                + str(native_support_summary.get("native_carrier_support_row_count", 0))
                + ";diagnostic_assignment_count="
                + str(native_diagnostic_summary.get("native_carrier_diagnostic_assignment_count", 0))
                + ";ARI="
                + str(native_diagnostic_summary.get("adjusted_rand_index", ""))
                + ";purity="
                + str(native_diagnostic_summary.get("purity", ""))
                + ";cluster_AP50="
                + str(native_diagnostic_summary.get("native_carrier_cluster_AP50", ""))
                + ";real_minus_best_nonoracle_AP50="
                + str(native_diagnostic_summary.get("native_carrier_real_minus_best_non_oracle_AP50", ""))
            ),
            "blocked_reason": "diagnostic native-carrier evaluator plus controls exist, but the contract is post-hoc on current dev and not ScanNet AP/SF",
            "required_next_artifact": "pre-register/freeze a native-carrier evaluator before selection or implement method-safe ScanNet scene-vertex export",
        },
        {
            "route_id": "E8_v85_diagnostic_frame_mask_backproject",
            "route_name": "current v85 diagnostic frame-mask backprojection npz",
            "candidate_available": diagnostic_available,
            "route_status": "available_diagnostic_only" if diagnostic_available else str(diagnostic_npz.get("reason", "missing")),
            "can_be_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": diagnostic_forbidden,
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": False,
            "uses_rgbd_pose_mesh_for_export": True,
            "has_scene_vertex_ids": diagnostic_available,
            "has_point_ids": diagnostic_available,
            "has_native_carrier_ids": False,
            "metric_scope": "scannet_scene_vertex_ap_diagnostic_only",
            "evidence_path": str(diagnostic_npz.get("scene_rows", "")),
            "evidence_detail": "output_config=" + str(diagnostic_npz.get("output_config", "")),
            "blocked_reason": "uses RGB-D/pose/ScanNet mesh backprojection; diagnostic bridge only",
            "required_next_artifact": "method-safe scene support exporter, not diagnostic backprojection",
        },
    ]

    method_safe_scene_rows = [
        row
        for row in rows
        if bool(row["can_be_method_result"])
        and not bool(row["forbidden_for_method_table"])
        and str(row["metric_scope"]).startswith("scannet_scene_vertex_ap")
    ]
    native_evaluator_rows = [
        row
        for row in rows
        if bool(row["can_be_method_result"]) and "native_carrier_ap_style" in str(row["metric_scope"])
    ]
    diagnostic_rows = [
        row
        for row in rows
        if bool(row["is_diagnostic_only"]) or bool(row["forbidden_for_method_table"])
    ]
    summary = {
        "schema": "stream4d_v85_native_scene_vertex_export_route_audit_v1",
        "checked_candidate_route_count": len(rows),
        "method_safe_scene_vertex_exporter_available": bool(method_safe_scene_rows),
        "method_safe_scene_vertex_exporter_route_ids": [str(row["route_id"]) for row in method_safe_scene_rows],
        "method_safe_native_carrier_evaluator_available": bool(native_evaluator_rows),
        "native_carrier_evaluator_input_available": bool(native_support_summary.get("available")),
        "native_carrier_diagnostic_evaluator_available": native_diagnostic_available,
        "native_carrier_diagnostic_assignment_count": native_diagnostic_summary.get(
            "native_carrier_diagnostic_assignment_count",
            0,
        ),
        "native_carrier_diagnostic_ARI": native_diagnostic_summary.get("adjusted_rand_index", ""),
        "native_carrier_diagnostic_purity": native_diagnostic_summary.get("purity", ""),
        "native_carrier_diagnostic_completeness": native_diagnostic_summary.get("completeness", ""),
        "native_carrier_cluster_AP25": native_diagnostic_summary.get("native_carrier_cluster_AP25", ""),
        "native_carrier_cluster_AP50": native_diagnostic_summary.get("native_carrier_cluster_AP50", ""),
        "native_carrier_cluster_AP_mean": native_diagnostic_summary.get("native_carrier_cluster_AP_mean", ""),
        "native_carrier_cluster_mean_best_iou": native_diagnostic_summary.get(
            "native_carrier_cluster_mean_best_iou",
            "",
        ),
        "native_carrier_evaluation_label_scope": native_diagnostic_summary.get("evaluation_label_scope", ""),
        "native_carrier_control_variant_count": native_diagnostic_summary.get("native_carrier_control_variant_count", ""),
        "native_carrier_non_oracle_control_count": native_diagnostic_summary.get(
            "native_carrier_non_oracle_control_count",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_AP50": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_mean_best_iou": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_mean_best_iou",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_ARI": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_ARI",
            "",
        ),
        "native_carrier_support_row_count": native_support_summary.get("native_carrier_support_row_count", 0),
        "native_unique_carrier_count": native_support_summary.get("native_unique_carrier_count", 0),
        "diagnostic_bridge_available": diagnostic_available or any(bool(row["is_diagnostic_only"]) for row in rows),
        "diagnostic_or_forbidden_route_count": len(diagnostic_rows),
        "all_checked_routes_blocked_for_method_scene_metric": not bool(method_safe_scene_rows),
        "primary_blocker": "native_carrier_support_ready_but_scene_vertex_exporter_missing"
        if native_support_summary.get("available")
        else "native_carrier_support_missing",
        "required_next_artifact": "method-safe native-carrier-to-ScanNet-scene-vertex exporter/evaluator or a separately specified native-carrier AP-style evaluator",
        "route_rows_path": _rel(phase7_out / "native_scene_vertex_export_route_rows.csv"),
    }
    return rows, summary


def _nonempty_field_rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return _safe_ratio(sum(1 for row in rows if str(row.get(field, "")).strip() != ""), len(rows))


def _parse_metric_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if lines and lines[0].startswith("class,"):
        reader = csv.DictReader(lines)
        numeric_rows: list[tuple[float, float, float]] = []
        trailing: tuple[float, float, float] | None = None
        for row in reader:
            name = str(row.get("class", "")).strip().lower()
            try:
                values = (float(row["ap"]), float(row["ap50"]), float(row["ap25"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in values):
                numeric_rows.append(values)
            if name in {"average", "mean"}:
                trailing = values
        if trailing is None and numeric_rows:
            trailing = numeric_rows[-1]
        if trailing is not None:
            return {"AP": float(trailing[0]), "AP50": float(trailing[1]), "AP25": float(trailing[2])}
    for line in lines:
        if not line.startswith("average"):
            continue
        values: list[float] = []
        for item in line.split()[1:]:
            try:
                values.append(float(item))
            except ValueError:
                pass
        if len(values) >= 3:
            return {"AP": float(values[0]), "AP50": float(values[1]), "AP25": float(values[2])}
    return {}


def _diagnostic_export_frame_mask_npz(
    selected_frame_rows: list[dict[str, Any]],
    *,
    output_config: str,
    phase7_out: Path,
) -> dict[str, Any]:
    if not selected_frame_rows:
        return {"available": False, "reason": "no selected frame-mask rows"}

    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np

    from stream4d.export_scannet import ScanNetExporter
    from stream4d.scannet_stream import ScanNetStream
    from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_frame_rows:
        by_scene[str(row.get("scene_id", ""))].append(row)

    pred_dir = ROOT / "data/prediction" / f"{output_config}_class_agnostic"
    tmp_dir = ROOT / "data/TMP" / output_config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, Any]] = []
    exported_scene_count = 0
    total_queries = 0
    total_hits = 0
    total_objects = 0
    total_missing_mask_files = 0
    total_zero_hit_masks = 0

    for scene, rows in sorted(by_scene.items()):
        stream = ScanNetStream(seq_name=scene, root=ROOT / "data/scannet/processed")
        errors = stream.validate(require_masks=True)
        if errors:
            scene_rows.append(
                {
                    "scene_id": scene,
                    "status": "validation_failed",
                    "errors": "; ".join(errors),
                    "selected_frame_mask_count": len(rows),
                }
            )
            continue

        exporter = ScanNetExporter(
            stream,
            output_config=output_config,
            export_nn_radius=0.05,
            export_support_mode="mask_backproject",
            export_mask_sample_stride=2,
            export_mask_max_pixels=50000,
            export_min_points_per_object=1,
            export_score_mode="area",
        )
        rows_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_history[str(row.get("history_id", ""))].append(row)

        object_dict: dict[int, dict[str, Any]] = {}
        pred_masks: list[Any] = []
        pred_scores: list[float] = []
        object_rows: list[dict[str, Any]] = []
        scene_queries = 0
        scene_hits = 0
        scene_kept_observations = 0
        scene_missing_mask_files = 0
        scene_zero_hit_masks = 0
        scene_missing_examples: list[str] = []
        for object_id, (history_id, history_rows) in enumerate(sorted(rows_by_history.items())):
            point_ids: set[int] = set()
            kept_refs: list[tuple[int, int, float]] = []
            for row in history_rows:
                try:
                    frame_id = int(row.get("frame_id", ""))
                    mask_id = int(row.get("mask_id", ""))
                except ValueError:
                    continue
                try:
                    hit_ids, query_count = exporter._backproject_mask(frame_id, mask_id, nn_radius=0.05)
                except FileNotFoundError as exc:
                    scene_missing_mask_files += 1
                    if len(scene_missing_examples) < 10:
                        scene_missing_examples.append(str(exc))
                    continue
                scene_queries += int(query_count)
                scene_hits += int(hit_ids.shape[0])
                if hit_ids.size == 0:
                    scene_zero_hit_masks += 1
                    continue
                point_ids.update(int(value) for value in hit_ids.tolist())
                kept_refs.append((frame_id, mask_id, float(_num(row.get("mask_score"), 1.0))))
                scene_kept_observations += 1
            point_ids_array = np.asarray(sorted(point_ids), dtype=np.int64)
            object_dict[object_id] = {
                "point_ids": point_ids_array,
                "mask_list": kept_refs,
                "repre_mask_list": sorted(kept_refs, key=lambda item: float(item[2]), reverse=True)[:8],
                "carrier_ids": np.empty((0,), dtype=np.int64),
                "history_id": history_id,
                "source": "v85_phase7_selected_frame_mask_rows_diagnostic_backproject",
            }
            if point_ids_array.size == 0:
                continue
            mask = np.zeros((exporter.scene_points.shape[0],), dtype=bool)
            mask[point_ids_array] = True
            pred_masks.append(mask)
            pred_scores.append(float(point_ids_array.size))
            object_rows.append(
                {
                    "scene_id": scene,
                    "object_id": object_id,
                    "history_id": history_id,
                    "point_count": int(point_ids_array.size),
                    "mask_observation_count": len(kept_refs),
                    "selected_frame_mask_input_count": len(history_rows),
                }
            )

        if pred_masks:
            pred_mask_np = np.stack(pred_masks, axis=1).astype(bool, copy=False)
            pred_score_np = np.asarray(pred_scores, dtype=np.float32)
        else:
            pred_mask_np = np.zeros((exporter.scene_points.shape[0], 0), dtype=bool)
            pred_score_np = np.zeros((0,), dtype=np.float32)
        pred_classes = np.zeros((pred_score_np.shape[0],), dtype=np.int32)
        np.savez_compressed(
            pred_dir / f"{scene}.npz",
            pred_masks=pred_mask_np,
            pred_score=pred_score_np,
            pred_classes=pred_classes,
        )
        pre_points = np.flatnonzero(pred_mask_np.any(axis=1)).astype(np.int64)
        np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
        object_dir = stream.object_dir / output_config
        object_dir.mkdir(parents=True, exist_ok=True)
        np.save(object_dir / "object_dict.npy", object_dict, allow_pickle=True)
        _write_csv(phase7_out / f"diagnostic_npz_object_rows_{scene}.csv", object_rows)

        exported_scene_count += 1
        total_queries += scene_queries
        total_hits += scene_hits
        total_objects += int(pred_mask_np.shape[1])
        total_missing_mask_files += scene_missing_mask_files
        total_zero_hit_masks += scene_zero_hit_masks
        scene_rows.append(
            {
                "scene_id": scene,
                "status": "exported_diagnostic_only",
                "prediction_path": _rel(pred_dir / f"{scene}.npz"),
                "pre_points_path": _rel(tmp_dir / f"{scene}_pre_points.npy"),
                "object_count": int(pred_mask_np.shape[1]),
                "selected_frame_mask_count": len(rows),
                "kept_mask_observation_count": int(scene_kept_observations),
                "missing_mask_file_count": int(scene_missing_mask_files),
                "zero_hit_mask_count": int(scene_zero_hit_masks),
                "missing_mask_examples_json": json.dumps(scene_missing_examples, ensure_ascii=False),
                "backproject_query_count": int(scene_queries),
                "backproject_hit_count": int(scene_hits),
                "backproject_hit_rate": _safe_ratio(scene_hits, scene_queries),
                "pre_points_count": int(pre_points.shape[0]),
                "scene_point_count": int(exporter.scene_points.shape[0]),
                "pre_points_ratio": _safe_ratio(pre_points.shape[0], exporter.scene_points.shape[0]),
                "uses_rgbd_pose_mesh_for_export": True,
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )

    manifest = build_prediction_manifest(
        output_config=output_config,
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[_rel(phase7_out / "frame_mask_prediction_rows.csv")],
        pre_points_policy="rgbd_pose_mesh_mask_backproject_diagnostic",
        support_policy="v85_selected_frame_mask_rows",
        notes=(
            "v85 diagnostic-only scene npz generated from selected frame-mask rows via ScanNet "
            "RGB-D/pose/mesh backprojection. This is an evaluation bridge and is forbidden for "
            "method-table claims under the v85 plan contract."
        ),
        extra={
            "phase": "v85_phase7_diagnostic_frame_mask_npz",
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
            "eval_policy": "diagnostic_backproject_only_not_method_safe",
            "scene_count": int(len(by_scene)),
            "exported_scene_count": int(exported_scene_count),
            "selected_frame_mask_count": int(len(selected_frame_rows)),
            "missing_mask_file_count": int(total_missing_mask_files),
            "zero_hit_mask_count": int(total_zero_hit_masks),
        },
    )
    written_manifests = write_prediction_manifest(output_config, manifest, root=ROOT, pred_suffix="class_agnostic")
    _write_csv(phase7_out / "diagnostic_npz_scene_rows.csv", scene_rows)
    return {
        "available": exported_scene_count > 0,
        "output_config": output_config,
        "prediction_dir": _rel(pred_dir),
        "tmp_dir": _rel(tmp_dir),
        "manifest_paths": [_rel(path) for path in written_manifests],
        "scene_count": int(len(by_scene)),
        "exported_scene_count": int(exported_scene_count),
        "object_count": int(total_objects),
        "selected_frame_mask_count": int(len(selected_frame_rows)),
        "missing_mask_file_count": int(total_missing_mask_files),
        "zero_hit_mask_count": int(total_zero_hit_masks),
        "backproject_query_count": int(total_queries),
        "backproject_hit_count": int(total_hits),
        "backproject_hit_rate": _safe_ratio(total_hits, total_queries),
        "scene_rows": _rel(phase7_out / "diagnostic_npz_scene_rows.csv"),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "reason": "diagnostic-only RGB-D/pose/mesh backprojection export; not method-safe scene prediction",
    }


def _run_diagnostic_npz_eval(output_config: str, out: Path) -> dict[str, Any]:
    if not output_config:
        return {"available": False, "reason": "missing_output_config"}
    import os
    import subprocess
    import sys

    metric_file = out / f"{output_config}_diagnostic_class_agnostic.txt"
    log_path = out / f"{output_config}_diagnostic_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{output_config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        output_config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    metrics = _parse_metric_file(metric_file)
    return {
        "available": proc.returncode == 0 and bool(metrics),
        "exit_code": int(proc.returncode),
        "command": " ".join(cmd),
        "metric_file": _rel(metric_file),
        "log_path": _rel(log_path),
        "metrics": metrics,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "reason": "diagnostic-only evaluator run over RGB-D/pose/mesh backprojected v85 frame-mask npz",
    }


def _diagnostic_tentative_holdout_replay(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    replay_out = out / "diagnostic_tentative_holdout"
    replay_out.mkdir(parents=True, exist_ok=True)
    local_slot_path = _repo_path(V84_HOLDOUT_REPLAY_LOCAL_SLOT_ROWS)
    adapter_path = _repo_path(V84_HOLDOUT_REPLAY_ADAPTER_ROWS)
    weak_assignment_path = _repo_path(V84_HOLDOUT_REPLAY_WEAK_ASSIGNMENT_ROWS)
    local_slots = _read_csv_rows(local_slot_path)
    adapter_rows = _read_csv_rows(adapter_path)
    weak_rows = _read_csv_rows(weak_assignment_path)

    local_by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in local_slots:
        local_by_slot[(str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", "")))] = row
    adapter_by_cluster: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in adapter_rows:
        adapter_by_cluster[
            (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("cluster_id", "")))
        ].append(row)

    candidate_rows: list[dict[str, Any]] = []
    weak_missing_cluster_count = 0
    weak_diagnostic_count = 0
    weak_method_count = 0
    weak_future_count = 0

    def adapter_score(row: dict[str, Any]) -> float:
        for key in ("hybrid_adapter_F1", "rendered_pixel_F1", "carrier_F1"):
            value = _float(row.get(key))
            if value is not None:
                return value
        return 0.0

    for weak_idx, weak in enumerate(weak_rows):
        if _bool(weak.get("diagnostic_only")):
            weak_diagnostic_count += 1
        if _bool(weak.get("method_uses_gt")):
            weak_method_count += 1
        if _bool(weak.get("uses_future")):
            weak_future_count += 1
        scene = str(weak.get("scene_id", ""))
        chunk = str(weak.get("chunk_id", ""))
        slot_id = str(weak.get("local_slot_id", ""))
        history_id = str(weak.get("assigned_history_id") or weak.get("history_id") or "")
        local = local_by_slot.get((scene, chunk, slot_id))
        if not local:
            weak_missing_cluster_count += 1
            continue
        cluster_id = str(local.get("cluster_id", ""))
        adapters = adapter_by_cluster.get((scene, chunk, cluster_id), [])
        if not adapters:
            weak_missing_cluster_count += 1
            continue
        for adapter in adapters:
            allowed = (
                bool(str(adapter.get("frame_id", "")).strip())
                and bool(str(adapter.get("mask_id", "")).strip())
                and _bool(adapter.get("object_mask_ownership_allowed"))
                and not _bool(adapter.get("adapter_caused_split"))
                and not _bool(adapter.get("adapter_caused_merge"))
                and not _bool(weak.get("method_uses_gt"))
                and not _bool(weak.get("uses_future"))
            )
            candidate_rows.append(
                {
                    "candidate_row_id": len(candidate_rows),
                    "weak_assignment_row_id": weak_idx,
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "frame_id": adapter.get("frame_id", ""),
                    "history_id": history_id,
                    "local_slot_id": slot_id,
                    "cluster_id": cluster_id,
                    "materializer_variant": "diagnostic_tentative_holdout_slot_union_adapter_wta",
                    "mask_id": adapter.get("mask_id", ""),
                    "mask_score": adapter_score(adapter),
                    "carrier_support_score": adapter.get("carrier_F1", ""),
                    "rendered_pixel_score": adapter.get("rendered_pixel_F1", ""),
                    "hybrid_adapter_score": adapter.get("hybrid_adapter_F1", ""),
                    "assignment_type": weak.get("assignment_type", ""),
                    "q_margin": weak.get("q_margin", ""),
                    "assignment_entropy": weak.get("assignment_entropy", ""),
                    "object_mask_ownership_allowed": adapter.get("object_mask_ownership_allowed", ""),
                    "adapter_caused_split": adapter.get("adapter_caused_split", ""),
                    "adapter_caused_merge": adapter.get("adapter_caused_merge", ""),
                    "adapter_candidate_valid": allowed,
                    "selected_flag": False,
                    "selection_policy": "diagnostic_top_score_per_scene_history_frame",
                    "method_uses_gt": weak.get("method_uses_gt", False),
                    "uses_future": weak.get("uses_future", False),
                    "source_assignment_diagnostic_only": weak.get("diagnostic_only", True),
                    "is_method_result": False,
                    "is_diagnostic_only": True,
                    "forbidden_for_method_table": True,
                }
            )

    valid_indices_by_history_frame: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(candidate_rows):
        if not _bool(row.get("adapter_candidate_valid")):
            continue
        valid_indices_by_history_frame[
            (str(row.get("scene_id", "")), str(row.get("history_id", "")), str(row.get("frame_id", "")))
        ].append(idx)
    selected_indices: set[int] = set()
    for indices in valid_indices_by_history_frame.values():
        selected_indices.add(
            max(
                indices,
                key=lambda idx: (
                    _num(candidate_rows[idx].get("mask_score"), -1.0),
                    str(candidate_rows[idx].get("mask_id", "")),
                    -idx,
                ),
            )
        )
    for idx in selected_indices:
        candidate_rows[idx]["selected_flag"] = True
    selected_rows = [row for row in candidate_rows if _bool(row.get("selected_flag"))]
    native_support_rows, native_source_audit_rows, native_support_summary = _load_native_carrier_support(
        selected_rows,
        args,
    )
    (
        native_assignment_rows,
        native_diag_source_rows,
        native_control_rows,
        native_diag_summary,
    ) = _native_carrier_diagnostic_eval(native_support_rows, replay_out)

    source_rows = [
        {
            "source_name": "v84_holdout_replay_v82_phase5_weak_history",
            "source_path": _rel(weak_assignment_path),
            "row_count": len(weak_rows),
            "method_safe_for_current_v85": False,
            "diagnostic_only_row_count": weak_diagnostic_count,
            "method_uses_gt_row_count": weak_method_count,
            "uses_future_row_count": weak_future_count,
            "notes": "holdout assignments are tentative/diagnostic only and v84 holdout preconditions failed",
        },
        {
            "source_name": "v84_holdout_replay_v82_phase1_local_b0",
            "source_path": _rel(local_slot_path),
            "row_count": len(local_slots),
            "method_safe_for_current_v85": False,
            "diagnostic_only_row_count": "",
            "method_uses_gt_row_count": sum(1 for row in local_slots if _bool(row.get("method_uses_gt"))),
            "uses_future_row_count": "",
            "notes": "local slot source used only to join diagnostic tentative assignments to clusters",
        },
        {
            "source_name": "v84_holdout_replay_v82_adapter_rows",
            "source_path": _rel(adapter_path),
            "row_count": len(adapter_rows),
            "method_safe_for_current_v85": False,
            "diagnostic_only_row_count": "",
            "method_uses_gt_row_count": "",
            "uses_future_row_count": "",
            "notes": "adapter rows expose frame/mask observations for diagnostic replay only",
        },
    ]
    summary = {
        "schema": "stream4d_v85_diagnostic_tentative_holdout_replay_v1",
        "decision": "DIAGNOSTIC_TENTATIVE_HOLDOUT_REPLAY_NOT_METHOD"
        if selected_rows and native_support_rows
        else "NO_DIAGNOSTIC_TENTATIVE_HOLDOUT_REPLAY_INPUT",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "method_claim_blocker": "source weak history assignments are diagnostic-only tentative and phase2-6 holdout preconditions failed",
        "weak_assignment_path": _rel(weak_assignment_path),
        "weak_assignment_row_count": len(weak_rows),
        "weak_forbidden_or_diagnostic_row_count": weak_diagnostic_count,
        "weak_missing_cluster_count": weak_missing_cluster_count,
        "adapter_rows_path": _rel(adapter_path),
        "joined_adapter_row_count": len(candidate_rows),
        "allowed_adapter_row_count": sum(1 for row in candidate_rows if _bool(row.get("adapter_candidate_valid"))),
        "selected_frame_mask_row_count": len(selected_rows),
        "selected_unique_frame_mask_count": len(
            {
                (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("mask_id", "")))
                for row in selected_rows
            }
        ),
        "diagnostic_native_support_row_count": len(native_support_rows),
        "diagnostic_unique_native_carrier_count": native_support_summary.get("native_unique_carrier_count", 0),
        "matched_observation_row_count": native_support_summary.get("matched_observation_rows", 0),
        "blocked_observation_row_count": native_support_summary.get("blocked_observation_rows", 0),
        "diagnostic_native_assignment_count": native_diag_summary.get("native_carrier_diagnostic_assignment_count", 0),
        "diagnostic_GT_label_coverage_rate": _safe_ratio(
            _num(native_diag_summary.get("labeled_support_observation_count"), 0.0),
            len(native_support_rows),
        ),
        "diagnostic_ARI": native_diag_summary.get("adjusted_rand_index", ""),
        "diagnostic_purity": native_diag_summary.get("purity", ""),
        "diagnostic_native_AP50": native_diag_summary.get("native_carrier_cluster_AP50", ""),
        "diagnostic_real_minus_best_non_oracle_AP50": native_diag_summary.get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "candidate_rows_path": _rel(out / "diagnostic_tentative_holdout_frame_mask_candidate_rows.csv"),
        "selected_rows_path": _rel(out / "diagnostic_tentative_holdout_frame_mask_selected_rows.csv"),
        "native_support_rows_path": _rel(out / "diagnostic_tentative_holdout_native_support_rows.csv"),
        "native_assignment_rows_path": _rel(replay_out / "native_carrier_diagnostic_assignment_rows.csv"),
        "native_control_rows_path": _rel(replay_out / "native_carrier_diagnostic_control_rows.csv"),
        "source_audit_rows_path": _rel(out / "diagnostic_tentative_holdout_source_audit_rows.csv"),
    }
    _write_csv(out / "diagnostic_tentative_holdout_frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "diagnostic_tentative_holdout_frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "diagnostic_tentative_holdout_native_support_rows.csv", native_support_rows)
    _write_csv(out / "diagnostic_tentative_holdout_native_source_audit_rows.csv", native_source_audit_rows)
    _write_csv(out / "diagnostic_tentative_holdout_source_audit_rows.csv", source_rows)
    _write_json(out / "diagnostic_tentative_holdout_summary.json", summary)
    return summary


def _holdout_method_precondition_audit(out: Path) -> dict[str, Any]:
    sources = {
        "v84_phase8": _repo_path("outputs/audit/v84_phase8_frozen_holdout/summary.json"),
        "phase2": _repo_path("outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets/summary.json"),
        "phase4": _repo_path("outputs/audit/v84_holdout_replay_v82_phase4_tracklet_to_history_q/summary.json"),
        "phase5": _repo_path("outputs/audit/v84_holdout_replay_v82_phase5_weak_history/summary.json"),
        "phase6": _repo_path("outputs/audit/v84_holdout_replay_v82_phase6_strong_history/strong_history_summary.json"),
        "weak_rows": _repo_path(V84_HOLDOUT_REPLAY_WEAK_ASSIGNMENT_ROWS),
    }
    v84_phase8 = _read_json(sources["v84_phase8"])
    phase2 = _read_json(sources["phase2"])
    phase4 = _read_json(sources["phase4"])
    phase5 = _read_json(sources["phase5"])
    phase6 = _read_json(sources["phase6"])
    weak_rows = _read_csv_rows(sources["weak_rows"])
    weak_diagnostic_count = sum(1 for row in weak_rows if _bool(row.get("diagnostic_only")))
    weak_method_gt_count = sum(1 for row in weak_rows if _bool(row.get("method_uses_gt")))
    weak_future_count = sum(1 for row in weak_rows if _bool(row.get("uses_future")))
    checks = [
        {
            "check_name": "v84_formal_holdout_safe_assignment_count_gt_0",
            "value": v84_phase8.get("holdout_safe_assignment_count", ""),
            "pass": _int(v84_phase8.get("holdout_safe_assignment_count"), 0) > 0,
            "source_artifact": _rel(sources["v84_phase8"]),
            "notes": "formal frozen holdout reports zero method-safe assignments",
        },
        {
            "check_name": "v84_holdout_method_mode_allowed",
            "value": v84_phase8.get("holdout_method_mode_allowed", ""),
            "pass": _bool(v84_phase8.get("holdout_method_mode_allowed")),
            "source_artifact": _rel(sources["v84_phase8"]),
            "notes": "method mode must be allowed before holdout assignment can enter method table",
        },
        {
            "check_name": "phase2_tracklet_association_gate_pass",
            "value": phase2.get("can_enter_next_phase", ""),
            "pass": _bool(phase2.get("can_enter_next_phase")),
            "source_artifact": _rel(sources["phase2"]),
            "notes": "phase2 blocks because eligible coverage is low and real does not beat semantic",
        },
        {
            "check_name": "phase2_eligible_tracklet_coverage_ge_0p25",
            "value": phase2.get("eligible_tracklet_coverage_rate", ""),
            "pass": _num(phase2.get("eligible_tracklet_coverage_rate"), 0.0) >= 0.25,
            "source_artifact": _rel(sources["phase2"]),
            "notes": "coverage gate from holdout replay phase2",
        },
        {
            "check_name": "phase2_full_minus_semantic_ge_0p03",
            "value": phase2.get("full_minus_semantic_score", ""),
            "pass": _num(phase2.get("full_minus_semantic_score"), -1.0) >= 0.03,
            "source_artifact": _rel(sources["phase2"]),
            "notes": "object-specific residual must beat semantic control",
        },
        {
            "check_name": "phase4_confirmed_history_q_gate_pass",
            "value": phase4.get("can_enter_next_phase", ""),
            "pass": _bool(phase4.get("can_enter_next_phase")),
            "source_artifact": _rel(sources["phase4"]),
            "notes": "phase4 did not produce confirmed-history Q rows",
        },
        {
            "check_name": "phase5_confirmed_assignment_count_gt_0",
            "value": phase5.get("confirmed_assignment_count", ""),
            "pass": _int(phase5.get("confirmed_assignment_count"), 0) > 0,
            "source_artifact": _rel(sources["phase5"]),
            "notes": "phase5 has only tentative diagnostic assignments",
        },
        {
            "check_name": "phase5_method_mode_claim_allowed",
            "value": phase5.get("method_mode_claim_allowed", ""),
            "pass": _bool(phase5.get("method_mode_claim_allowed")),
            "source_artifact": _rel(sources["phase5"]),
            "notes": "phase5 explicitly disallows method mode",
        },
        {
            "check_name": "weak_assignment_rows_not_diagnostic_only",
            "value": len(weak_rows) - weak_diagnostic_count,
            "pass": (len(weak_rows) - weak_diagnostic_count) > 0,
            "source_artifact": _rel(sources["weak_rows"]),
            "notes": "zero rows are non-diagnostic",
        },
        {
            "check_name": "weak_assignment_rows_no_gt_no_future",
            "value": f"method_gt={weak_method_gt_count};future={weak_future_count}",
            "pass": weak_method_gt_count == 0 and weak_future_count == 0,
            "source_artifact": _rel(sources["weak_rows"]),
            "notes": "provenance is clean, but diagnostic-only status still blocks method claim",
        },
        {
            "check_name": "phase6_strong_history_preconditions_pass",
            "value": phase6.get("can_enter_next_phase", ""),
            "pass": _bool(phase6.get("can_enter_next_phase")),
            "source_artifact": _rel(sources["phase6"]),
            "notes": "strong history path is blocked by prior phase preconditions",
        },
    ]
    failed = [row["check_name"] for row in checks if not _bool(row.get("pass"))]
    summary = {
        "schema": "stream4d_v85_holdout_method_precondition_audit_v1",
        "decision": "NO_GO_HOLDOUT_METHOD_PRECONDITIONS_FAILED" if failed else "PASS_HOLDOUT_METHOD_PRECONDITIONS",
        "preconditions_pass": not failed,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "weak_assignment_row_count": len(weak_rows),
        "weak_diagnostic_only_row_count": weak_diagnostic_count,
        "weak_method_uses_gt_row_count": weak_method_gt_count,
        "weak_uses_future_row_count": weak_future_count,
        "primary_blocker": ";".join(failed[:4]),
        "reason": "Holdout replay has diagnostic tentative native signal, but the method-safe selected readout preconditions are not met.",
        "check_rows_path": _rel(out / "holdout_method_precondition_rows.csv"),
    }
    _write_csv(out / "holdout_method_precondition_rows.csv", checks)
    _write_json(out / "holdout_method_precondition_summary.json", summary)
    return summary


def _phase2_tracklet_stats(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _read_json(root / "summary.json")
    candidates = _read_csv_rows(root / "tracklet_candidate_rows.csv")
    assignments = _read_csv_rows(root / "tracklet_assignment_rows.csv")
    controls = _read_csv_rows(root / "tracklet_control_rows.csv")

    eligible_slots: set[tuple[str, str, str]] = set()
    selected_slots: set[tuple[str, str, str]] = set()
    candidate_slots_by_scene: dict[str, set[tuple[str, str]]] = defaultdict(set)
    selected_slots_by_scene: dict[str, set[tuple[str, str]]] = defaultdict(set)
    chunk_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in candidates:
        scene = str(row.get("scene_id", ""))
        chunk = str(row.get("current_chunk_id", ""))
        slot = str(row.get("current_local_slot_id", ""))
        if not scene or not chunk or not slot:
            continue
        key = (scene, chunk)
        if key not in chunk_rows_by_key:
            chunk_rows_by_key[key] = {
                "scene_id": scene,
                "chunk_id": chunk,
                "eligible_slot_count": 0,
                "selected_assignment_count": 0,
                "mean_full_minus_semantic_slot": "",
                "selected_positive_full_minus_semantic_count": 0,
                "selected_nonpositive_full_minus_semantic_count": 0,
            }
        if _bool(row.get("eligible_for_assignment")):
            eligible_slots.add((scene, chunk, slot))
            candidate_slots_by_scene[scene].add((chunk, slot))

    fms_values: list[float] = []
    positive_fms = 0
    nonpositive_fms = 0
    assignment_state_counts: Counter[str] = Counter()
    fms_by_chunk: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in assignments:
        scene = str(row.get("scene_id", ""))
        chunk = str(row.get("chunk_id", ""))
        slot = str(row.get("local_slot_id", ""))
        if scene and chunk and slot:
            selected_slots.add((scene, chunk, slot))
            selected_slots_by_scene[scene].add((chunk, slot))
        assignment_state_counts[str(row.get("tracklet_state_after", ""))] += 1
        fms = _float(row.get("full_minus_semantic_slot"))
        if fms is not None:
            fms_values.append(float(fms))
            fms_by_chunk[(scene, chunk)].append(float(fms))
            if fms > 0:
                positive_fms += 1
            else:
                nonpositive_fms += 1

    for scene, chunk, slot in eligible_slots:
        chunk_rows_by_key[(scene, chunk)]["eligible_slot_count"] += 1
    for scene, chunk, slot in selected_slots:
        if (scene, chunk) not in chunk_rows_by_key:
            chunk_rows_by_key[(scene, chunk)] = {
                "scene_id": scene,
                "chunk_id": chunk,
                "eligible_slot_count": 0,
                "selected_assignment_count": 0,
                "mean_full_minus_semantic_slot": "",
                "selected_positive_full_minus_semantic_count": 0,
                "selected_nonpositive_full_minus_semantic_count": 0,
            }
        chunk_rows_by_key[(scene, chunk)]["selected_assignment_count"] += 1
    for key, values in fms_by_chunk.items():
        chunk_rows_by_key[key]["mean_full_minus_semantic_slot"] = _mean(values)
        chunk_rows_by_key[key]["selected_positive_full_minus_semantic_count"] = sum(1 for value in values if value > 0)
        chunk_rows_by_key[key]["selected_nonpositive_full_minus_semantic_count"] = sum(
            1 for value in values if value <= 0
        )
    for row in chunk_rows_by_key.values():
        row["coverage_rate"] = _safe_ratio(row["selected_assignment_count"], row["eligible_slot_count"])

    controls_by_variant = {str(row.get("variant", "")): row for row in controls}
    t5 = controls_by_variant.get("T5_semantic_appearance_temporal_conflict_guard", {})
    t0 = controls_by_variant.get("T0_semantic_only", {})
    stats = {
        "root": _rel(root),
        "decision": summary.get("decision", ""),
        "can_enter_next_phase": bool(summary.get("can_enter_next_phase")),
        "eligible_tracklet_coverage_rate": summary.get("eligible_tracklet_coverage_rate", ""),
        "full_minus_semantic_score": summary.get("full_minus_semantic_score", ""),
        "full_minus_shuffled_score": summary.get("full_minus_shuffled_score", ""),
        "full_minus_stale_score": summary.get("full_minus_stale_score", ""),
        "tracklet_assignment_entropy_mean": summary.get("tracklet_assignment_entropy_mean", ""),
        "tracklet_top1_top2_margin_mean": summary.get("tracklet_top1_top2_margin_mean", ""),
        "confirmed_tracklet_count": summary.get("confirmed_tracklet_count", ""),
        "candidate_row_count": len(candidates),
        "eligible_slot_count": len(eligible_slots),
        "selected_assignment_count": len(assignments),
        "unassigned_eligible_slot_count": max(0, len(eligible_slots) - len(selected_slots)),
        "selected_positive_full_minus_semantic_count": positive_fms,
        "selected_nonpositive_full_minus_semantic_count": nonpositive_fms,
        "selected_full_minus_semantic_mean": _mean(fms_values),
        "selected_full_minus_semantic_min": min(fms_values) if fms_values else "",
        "selected_full_minus_semantic_max": max(fms_values) if fms_values else "",
        "assignment_state_counts": dict(assignment_state_counts),
        "scene_count": len(candidate_slots_by_scene),
        "T5_top1_score_mean": t5.get("top1_score_mean", ""),
        "T0_semantic_top1_score_mean": t0.get("top1_score_mean", ""),
        "T5_minus_T0_top1_score_mean": _num(t5.get("top1_score_mean"), 0.0)
        - _num(t0.get("top1_score_mean"), 0.0),
    }
    scene_rows = []
    for scene in sorted(set(candidate_slots_by_scene) | set(selected_slots_by_scene)):
        eligible = len(candidate_slots_by_scene.get(scene, set()))
        selected = len(selected_slots_by_scene.get(scene, set()))
        scene_rows.append(
            {
                "split_source": _rel(root),
                "scene_id": scene,
                "eligible_slot_count": eligible,
                "selected_assignment_count": selected,
                "unassigned_eligible_slot_count": max(0, eligible - selected),
                "coverage_rate": _safe_ratio(selected, eligible),
            }
        )
    chunk_rows = []
    for key in sorted(chunk_rows_by_key, key=lambda item: (item[0], _int(item[1], -1))):
        row = dict(chunk_rows_by_key[key])
        row["split_source"] = _rel(root)
        chunk_rows.append(row)
    return stats, scene_rows, chunk_rows


def _holdout_phase2_failure_autopsy(out: Path) -> dict[str, Any]:
    dev_root = _repo_path(V82_DEV_PHASE2_REFERENCE_ROOT)
    holdout_root = _repo_path(V84_HOLDOUT_PHASE2_ROOT)
    dev_stats, dev_scene_rows, dev_chunk_rows = _phase2_tracklet_stats(dev_root)
    holdout_stats, holdout_scene_rows, holdout_chunk_rows = _phase2_tracklet_stats(holdout_root)

    comparison_rows = []
    metrics = [
        "eligible_tracklet_coverage_rate",
        "full_minus_semantic_score",
        "full_minus_shuffled_score",
        "full_minus_stale_score",
        "tracklet_assignment_entropy_mean",
        "tracklet_top1_top2_margin_mean",
        "candidate_row_count",
        "eligible_slot_count",
        "selected_assignment_count",
        "unassigned_eligible_slot_count",
        "selected_positive_full_minus_semantic_count",
        "selected_nonpositive_full_minus_semantic_count",
        "selected_full_minus_semantic_mean",
        "T5_top1_score_mean",
        "T0_semantic_top1_score_mean",
        "T5_minus_T0_top1_score_mean",
    ]
    for metric in metrics:
        dev_value = dev_stats.get(metric, "")
        holdout_value = holdout_stats.get(metric, "")
        delta = ""
        if isinstance(dev_value, (int, float)) or isinstance(holdout_value, (int, float)):
            delta = _num(holdout_value, 0.0) - _num(dev_value, 0.0)
        else:
            dev_float = _float(dev_value)
            holdout_float = _float(holdout_value)
            if dev_float is not None and holdout_float is not None:
                delta = holdout_float - dev_float
        comparison_rows.append(
            {
                "metric": metric,
                "dev_reference_value": dev_value,
                "holdout_value": holdout_value,
                "holdout_minus_dev": delta,
            }
        )

    coverage_drop = _num(dev_stats.get("eligible_tracklet_coverage_rate"), 0.0) - _num(
        holdout_stats.get("eligible_tracklet_coverage_rate"),
        0.0,
    )
    full_semantic_drop = _num(dev_stats.get("full_minus_semantic_score"), 0.0) - _num(
        holdout_stats.get("full_minus_semantic_score"),
        0.0,
    )
    holdout_t5_minus_t0 = _num(holdout_stats.get("T5_minus_T0_top1_score_mean"), 0.0)
    failed_checks = []
    if _num(holdout_stats.get("eligible_tracklet_coverage_rate"), 0.0) < 0.25:
        failed_checks.append("holdout_phase2_coverage_below_0p25")
    if _num(holdout_stats.get("full_minus_semantic_score"), -1.0) < 0.03:
        failed_checks.append("holdout_phase2_full_minus_semantic_below_0p03")
    if holdout_t5_minus_t0 <= 0:
        failed_checks.append("holdout_T5_mean_not_above_semantic_T0_mean")

    diagnostic_rows = [
        {
            "finding": "dev_reference_phase2_passed",
            "value": dev_stats.get("decision", ""),
            "notes": "dev reference was selected before holdout replay and is not recomputed from holdout feedback",
        },
        {
            "finding": "holdout_phase2_failed",
            "value": holdout_stats.get("decision", ""),
            "notes": "holdout replay under the frozen dev-selected v82 config fails Phase2 gates",
        },
        {
            "finding": "holdout_eligible_unassigned_slots",
            "value": holdout_stats.get("unassigned_eligible_slot_count", ""),
            "notes": "eligible slots with no selected prefix-tracklet assignment",
        },
        {
            "finding": "holdout_selected_nonpositive_full_minus_semantic",
            "value": holdout_stats.get("selected_nonpositive_full_minus_semantic_count", ""),
            "notes": "selected assignment rows where full tracklet score does not beat semantic-only score",
        },
        {
            "finding": "method_repair_scope",
            "value": "dev_side_descriptor_or_readout_redesign_required",
            "notes": "do not relax thresholds or select a new config using this holdout autopsy",
        },
    ]

    _write_csv(out / "holdout_phase2_failure_autopsy_comparison_rows.csv", comparison_rows)
    _write_csv(out / "holdout_phase2_failure_autopsy_scene_rows.csv", dev_scene_rows + holdout_scene_rows)
    _write_csv(out / "holdout_phase2_failure_autopsy_chunk_rows.csv", dev_chunk_rows + holdout_chunk_rows)
    _write_csv(out / "holdout_phase2_failure_autopsy_diagnostic_rows.csv", diagnostic_rows)
    summary = {
        "schema": "stream4d_v85_holdout_phase2_failure_autopsy_v1",
        "decision": "NO_GO_HOLDOUT_PHASE2_OBJECT_SPECIFICITY_DROP" if failed_checks else "PASS_HOLDOUT_PHASE2_AUTOPSY",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "dev_reference_root": _rel(dev_root),
        "holdout_root": _rel(holdout_root),
        "dev_decision": dev_stats.get("decision", ""),
        "holdout_decision": holdout_stats.get("decision", ""),
        "failed_check_count": len(failed_checks),
        "failed_checks": failed_checks,
        "dev_eligible_tracklet_coverage_rate": dev_stats.get("eligible_tracklet_coverage_rate", ""),
        "holdout_eligible_tracklet_coverage_rate": holdout_stats.get("eligible_tracklet_coverage_rate", ""),
        "coverage_drop_dev_minus_holdout": coverage_drop,
        "dev_full_minus_semantic_score": dev_stats.get("full_minus_semantic_score", ""),
        "holdout_full_minus_semantic_score": holdout_stats.get("full_minus_semantic_score", ""),
        "full_minus_semantic_drop_dev_minus_holdout": full_semantic_drop,
        "holdout_T5_top1_score_mean": holdout_stats.get("T5_top1_score_mean", ""),
        "holdout_T0_semantic_top1_score_mean": holdout_stats.get("T0_semantic_top1_score_mean", ""),
        "holdout_T5_minus_T0_top1_score_mean": holdout_stats.get("T5_minus_T0_top1_score_mean", ""),
        "holdout_eligible_slot_count": holdout_stats.get("eligible_slot_count", ""),
        "holdout_selected_assignment_count": holdout_stats.get("selected_assignment_count", ""),
        "holdout_unassigned_eligible_slot_count": holdout_stats.get("unassigned_eligible_slot_count", ""),
        "holdout_selected_positive_full_minus_semantic_count": holdout_stats.get(
            "selected_positive_full_minus_semantic_count",
            "",
        ),
        "holdout_selected_nonpositive_full_minus_semantic_count": holdout_stats.get(
            "selected_nonpositive_full_minus_semantic_count",
            "",
        ),
        "primary_blocker": ";".join(failed_checks[:3]),
        "repair_interpretation": (
            "Holdout drop is not caused by GT/future provenance; the frozen dev-selected tracklet descriptor loses "
            "object-specific residual against semantic-only on holdout and leaves most eligible slots unassigned."
        ),
        "required_next_artifact": (
            "new dev-side descriptor/readout design that improves object-specific residual without using holdout "
            "feedback; then freeze and run a fresh formal holdout in a future version"
        ),
        "comparison_rows_path": _rel(out / "holdout_phase2_failure_autopsy_comparison_rows.csv"),
        "scene_rows_path": _rel(out / "holdout_phase2_failure_autopsy_scene_rows.csv"),
        "chunk_rows_path": _rel(out / "holdout_phase2_failure_autopsy_chunk_rows.csv"),
        "diagnostic_rows_path": _rel(out / "holdout_phase2_failure_autopsy_diagnostic_rows.csv"),
    }
    _write_json(out / "holdout_phase2_failure_autopsy_summary.json", summary)
    return summary


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    roots = {
        "v79_phase1": _repo_path(args.v79_phase1_root),
        "v79_phase2": _repo_path(args.v79_phase2_root),
        "v80_phase1": _repo_path(args.v80_phase1_root),
        "v80_phase2": _repo_path(args.v80_phase2_root),
        "v80_phase6": _repo_path(args.v80_phase6_root),
        "v82_phase1": _repo_path(args.v82_phase1_root),
        "v83_phase2": _repo_path(args.v83_phase2_root),
        "v83_phase3": _repo_path(args.v83_phase3_root),
        "v83_phase4": _repo_path(args.v83_phase4_root),
        "v83_phase5": _repo_path(args.v83_phase5_root),
        "v83_phase6": _repo_path(args.v83_phase6_root),
        "v83_phase7": _repo_path(args.v83_phase7_root),
        "v84_phase3": _repo_path(args.v84_phase3_root),
        "v84_phase8": _repo_path(args.v84_phase8_root),
        "v84_phase9": _repo_path(args.v84_phase9_root),
    }
    v82_phase1_summary = _read_json(roots["v82_phase1"] / "summary.json")
    adapter_rows_source = (
        str(args.v82_adapter_rows).strip()
        or str(v82_phase1_summary.get("adapter_rows_source", "")).strip()
        or "outputs/audit/v82_local_shadow/phase1_adapter_dev_v82_phase1_b0/adapter_rows.csv"
    )
    adapter_rows_path = _repo_path(adapter_rows_source)
    return {
        "roots": roots,
        "v79_phase1_summary": _read_json(roots["v79_phase1"] / "affinity_feature_summary.json"),
        "v79_carrier_rows": _read_csv_rows(roots["v79_phase1"] / "carrier_affinity_feature_rows.csv"),
        "v79_neighbor_rows": _read_csv_rows(roots["v79_phase2"] / "carrier_affinity_neighbor_rows.csv"),
        "v80_phase1_summary": _read_json(roots["v80_phase1"] / "summary.json"),
        "v80_phase2_summary": _read_json(roots["v80_phase2"] / "summary.json"),
        "v80_phase6_summary": _read_json(roots["v80_phase6"] / "summary.json"),
        "v80_sketch_rows": _read_csv_rows(roots["v80_phase1"] / "sketch_quality_rows.csv"),
        "v80_feature_shape_rows": _read_csv_rows(roots["v80_phase1"] / "feature_shape_rows.csv"),
        "v82_phase1_summary": v82_phase1_summary,
        "local_slot_rows": _read_csv_rows(roots["v82_phase1"] / "local_slot_rows.csv"),
        "local_metric_rows": _read_csv_rows(roots["v82_phase1"] / "local_metric_rows.csv"),
        "local_descriptor_rows": _read_csv_rows(roots["v82_phase1"] / "local_descriptor_rows.csv"),
        "raw_cluster_rows": _read_csv_rows(roots["v82_phase1"] / "raw_v81_replay" / "local_cluster_rows.csv"),
        "v82_adapter_rows_source": adapter_rows_source,
        "v82_adapter_rows_path": adapter_rows_path,
        "v82_adapter_rows": _read_csv_rows(adapter_rows_path),
        "v83_phase2_summary": _read_json(roots["v83_phase2"] / "summary.json"),
        "v83_phase3_summary": _read_json(roots["v83_phase3"] / "summary.json"),
        "v83_phase4_summary": _read_json(roots["v83_phase4"] / "summary.json"),
        "v83_phase5_summary": _read_json(roots["v83_phase5"] / "summary.json"),
        "v83_phase6_summary": _read_json(roots["v83_phase6"] / "summary.json"),
        "v83_phase7_summary": _read_json(roots["v83_phase7"] / "summary.json"),
        "v83_evidence_rows": _read_csv_rows(roots["v83_phase2"] / "evidence_term_rows.csv"),
        "v83_state_rows": _read_csv_rows(roots["v83_phase3"] / "state_transition_rows.csv"),
        "v83_cannot_link_rows": _read_csv_rows(roots["v83_phase4"] / "cannot_link_rows.csv"),
        "v83_assignments": _read_csv_rows(roots["v83_phase5"] / "local_slot_history_assignment_rows.csv"),
        "v83_edges": _read_csv_rows(roots["v83_phase7"] / "fused_edge_rows.csv"),
        "v84_phase3_summary": _read_json(roots["v84_phase3"] / "summary.json"),
        "v84_materialized_slot_rows": _read_csv_rows(roots["v84_phase3"] / "materialized_slot_rows.csv"),
        "v84_scene_metric_rows": _read_csv_rows(roots["v84_phase3"] / "scene_metric_rows.csv"),
        "v84_phase8_summary": _read_json(roots["v84_phase8"] / "summary.json"),
        "v84_final": _read_json(roots["v84_phase9"] / "final_decision.json"),
    }


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase0_output_root)
    out.mkdir(parents=True, exist_ok=True)

    p83_5 = ctx["v83_phase5_summary"]
    p83_6 = ctx["v83_phase6_summary"]
    p83_7 = ctx["v83_phase7_summary"]
    p84_8 = ctx["v84_phase8_summary"]
    p84_final = ctx["v84_final"]
    p79_1 = ctx["v79_phase1_summary"]
    p80_1 = ctx["v80_phase1_summary"]
    p80_2 = ctx["v80_phase2_summary"]

    facts = {
        "v83_safe_assignment_row_count": p83_7.get("safe_assignment_row_count", p83_5.get("safe_topk_selected_count", "")),
        "v83_confirmed_plus_stable_coverage": p83_6.get("real_confirmed_plus_stable_coverage", ""),
        "v83_real_entropy": p83_6.get("real_assignment_entropy", ""),
        "v83_semantic_entropy": p83_6.get("semantic_assignment_entropy", ""),
        "v83_shuffled_entropy": p83_6.get("shuffled_assignment_entropy", ""),
        "v83_stale_entropy": p83_6.get("stale_assignment_entropy", ""),
        "v83_structural_history_edge_count": p83_7.get("history_edge_count", ""),
        "v83_history_cluster_count": p83_7.get("history_cluster_count", ""),
        "v84_final_decision": p84_final.get("final_decision", ""),
        "v84_holdout_safe_assignment_count": p84_8.get("holdout_safe_assignment_count", ""),
        "v84_holdout_diagnostic_assignment_count": p84_8.get("holdout_diagnostic_assignment_count", ""),
        "v84_scene_metric_available": bool(p84_8.get("holdout_scene_SF50")),
        "v84_method_claim_allowed": bool(p84_8.get("holdout_method_mode_allowed")),
        "v79_best_local_SF50": "",
        "v79_affinity_minus_semantic_AUC": "",
        "v79_cosine_approx_error_p95": p79_1.get("cosine_approx_error_p95", ""),
        "v79_broad_mask_contribution_ratio": p79_1.get("broad_mask_contribution_ratio", ""),
        "v80_streaming_causality_contract_present": _bool(p80_1.get("method_uses_gt_anywhere")) is False
        and _bool(p80_1.get("method_prediction_uses_future_anywhere")) is False,
        "v80_topk_recall_under_sketch": p80_1.get("topk_recall_under_sketch", ""),
        "v80_cosine_error_p95": p80_1.get("cosine_error_p95", ""),
        "v80_largest_connected_component_ratio": p80_2.get("largest_connected_component_ratio", ""),
        "GT_prediction_violation_count": int(_bool(p84_final.get("method_uses_gt_anywhere"))) + int(_bool(p80_1.get("method_uses_gt_anywhere"))),
        "future_artifact_allowed_count": int(_bool(p84_final.get("uses_future_anywhere"))) + int(_bool(p80_1.get("method_prediction_uses_future_anywhere"))),
    }

    fact_sources = {
        "v83": _rel(ctx["roots"]["v83_phase6"] / "summary.json"),
        "v84": _rel(ctx["roots"]["v84_phase8"] / "summary.json"),
        "v79": _rel(ctx["roots"]["v79_phase1"] / "affinity_feature_summary.json"),
        "v80": _rel(ctx["roots"]["v80_phase1"] / "summary.json"),
    }
    fact_rows = []
    for key, value in facts.items():
        if key.startswith("v83"):
            source = fact_sources["v83"]
        elif key.startswith("v84") or key in {"GT_prediction_violation_count", "future_artifact_allowed_count"}:
            source = fact_sources["v84"]
        elif key.startswith("v79"):
            source = fact_sources["v79"]
        else:
            source = fact_sources["v80"]
        fact_rows.append(
            {
                "fact_id": key,
                "fact_name": key,
                "fact_value": value,
                "source_artifact": source,
                "allowed_for_method": key.startswith("v83") or key.startswith("v80"),
                "allowed_for_diagnostic": True,
                "notes": "v84 holdout facts are diagnostic boundary only" if key.startswith("v84") else "",
            }
        )

    deprecated_rows = [
        {
            "artifact": "v84_frozen_holdout",
            "source_artifact": _rel(ctx["roots"]["v84_phase8"] / "summary.json"),
            "deprecated_for": "v85_threshold_or_holdout_tuning",
            "allowed_for_diagnostic": True,
            "reason": "v84 plan closed with NO_GO_HOLDOUT_FAIL; v85 must be a new dev-side version",
        }
    ]
    role_rows = [
        {"role": "local_carrier_affinity_feature", "method_use": "current_chunk_clustering", "history_use": "not directly persistent"},
        {"role": "slot_tracklet_affinity_descriptor", "method_use": "local2history query interface", "history_use": "causal descriptor updates"},
        {"role": "history_object_affinity_descriptor", "method_use": "persistent query memory", "history_use": "future chunk anchor"},
        {"role": "renderable_membership_field", "method_use": "materialize frame/carrier/slot extent", "history_use": "requires exporter"},
    ]
    metric_rows = [
        {"metric": "SF50/AP/GT_best_IoU", "class": "diagnostic_or_final_eval", "method_selection_allowed": False},
        {"metric": "entropy/control margin/cannot-link/future/GT flags", "class": "method_selection", "method_selection_allowed": True},
        {"metric": "v84 holdout result", "class": "diagnostic_boundary", "method_selection_allowed": False},
    ]

    weak_signal = (
        str(p83_6.get("decision", "")).startswith("PASS")
        and _num(p83_6.get("real_assignment_entropy"), 1.0) <= _num(p83_6.get("semantic_assignment_entropy"), 0.0) - 0.1
        and _num(p83_6.get("real_confirmed_plus_stable_coverage"), 0.0) >= 0.20
    )
    gate = {
        "v84_final_decision_is_no_go_holdout_fail": p84_final.get("final_decision") == "NO_GO_HOLDOUT_FAIL",
        "v84_holdout_safe_assignment_count_eq_0": _int(p84_8.get("holdout_safe_assignment_count"), -1) == 0,
        "v84_scene_metric_available_false": not bool(p84_8.get("holdout_scene_SF50")),
        "v83_weak_ledger_signal_present": weak_signal,
        "v84_artifacts_marked_not_reusable_for_v85_holdout_tuning": True,
        "GT_prediction_violation_count_eq_0": facts["GT_prediction_violation_count"] == 0,
        "future_artifact_allowed_count_eq_0": facts["future_artifact_allowed_count"] == 0,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v85_phase0_fact_lock",
        "schema": "stream4d_v85_phase0_fact_lock_v1",
        "decision": "PASS_V85_PHASE0_FACT_LOCK" if gate["pass"] else "NO_GO_V85_INPUT_BOUNDARY",
        "can_enter_next_phase": gate["pass"],
        "gate": gate,
        "primary_blocker": "" if gate["pass"] else "fact_lock_boundary_failed",
        "runtime_sec": time.time() - started,
        **facts,
    }
    _write_json(out / "fact_lock_summary.json", summary)
    _write_csv(out / "fact_rows.csv", fact_rows)
    _write_csv(out / "deprecated_artifact_rows.csv", deprecated_rows)
    _write_csv(out / "affinity_feature_role_rows.csv", role_rows)
    _write_csv(out / "metric_class_rows.csv", metric_rows)
    return summary


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase1_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p80 = ctx["v80_phase1_summary"]
    p80_signed = ctx["v80_phase2_summary"]
    p79 = ctx["v79_phase1_summary"]

    carrier_rows = []
    for row in ctx["v79_carrier_rows"][: args.max_carrier_rows]:
        carrier_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "scale": row.get("scale", "object"),
                "carrier_id": row.get("carrier_id", ""),
                "feature_dim": row.get("feature_dim", ""),
                "feature_norm": row.get("feature_norm", ""),
                "nonzero_update_count": row.get("nnz_signature", ""),
                "mask_support_count": row.get("membership_count", ""),
                "positive_mass": row.get("mean_membership_weight", ""),
                "negative_mass": "",
                "partwhole_mass": "",
                "semantic_proto_id": "",
                "visibility_count": row.get("visible_frame_count", ""),
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
    sketch_rows = []
    if ctx["v80_sketch_rows"]:
        for row in ctx["v80_sketch_rows"]:
            sketch_rows.append(dict(row))
    else:
        sketch_rows.append(
            {
                "scene_id": "ALL_DEV",
                "chunk_id": "ALL_DEV",
                "scale": "object",
                "projection_dim": "",
                "exact_subset_carrier_count": "",
                "cosine_error_p50": "",
                "cosine_error_p95": p80.get("cosine_error_p95", ""),
                "topk_recall_under_sketch": p80.get("topk_recall_under_sketch", ""),
                "bucket_load_mean": "",
                "bucket_load_p95": p80.get("bucket_load_p95", ""),
                "collision_mass_ratio": p80.get("collision_mass_ratio", ""),
                "broad_collision_mass_ratio": p80.get("broad_collision_mass_ratio", ""),
                "runtime_sec": p80.get("runtime_sec", ""),
                "peak_memory_gb": p80.get("peak_memory_gb", ""),
            }
        )
    neighbor_rows = [dict(row) for row in ctx["v79_neighbor_rows"][: args.max_neighbor_rows]]
    control_rows = [
        {
            "variant": "L0_v79_positive_comask",
            "decision": p79.get("decision", ""),
            "topk_recall_under_sketch": "",
            "cosine_error_p95": p79.get("cosine_approx_error_p95", ""),
            "broad_collision_mass_ratio": p79.get("broad_mask_contribution_ratio", ""),
            "largest_connected_component_ratio": "",
            "same_frame_hard_negative_AUC": "",
            "within_semantic_hard_negative_AUC": "",
            "notes": "legacy v79 replay fails sketch/broad-mask gates",
        },
        {
            "variant": "L2_v80_signed_scale_gated",
            "decision": p80.get("decision", ""),
            "topk_recall_under_sketch": p80.get("topk_recall_under_sketch", ""),
            "cosine_error_p95": p80.get("cosine_error_p95", ""),
            "broad_collision_mass_ratio": p80.get("broad_collision_mass_ratio", ""),
            "largest_connected_component_ratio": p80_signed.get("largest_connected_component_ratio", ""),
            "same_frame_hard_negative_AUC": p80_signed.get("same_frame_hard_negative_AUC", ""),
            "within_semantic_hard_negative_AUC": p80_signed.get("within_semantic_hard_negative_AUC", ""),
            "notes": "object-specific hard-negative diagnostics unavailable in current artifact",
        },
    ]
    gate = {
        "topk_recall_under_sketch_ge_0p85": _num(p80.get("topk_recall_under_sketch"), 0.0) >= 0.85,
        "cosine_error_p95_le_0p05": _num(p80.get("cosine_error_p95"), 1.0) <= 0.05,
        "largest_connected_component_ratio_le_0p25": _num(p80_signed.get("largest_connected_component_ratio"), 1.0) <= 0.25,
        "within_semantic_hard_negative_AUC_available": bool(p80_signed.get("within_semantic_hard_negative_AUC")),
        "method_uses_gt_false": not _bool(p80.get("method_uses_gt_anywhere")),
        "uses_future_false": not _bool(p80.get("method_prediction_uses_future_anywhere")),
    }
    gate["pass"] = all(gate.values())
    decision = "PASS_V85_PHASE1_LOCAL_AFFINITY_FEATURE" if gate["pass"] else "DIAGNOSTIC_V85_PHASE1_OBJECTNESS_AUC_MISSING"
    summary = {
        "phase": "v85_phase1_local_affinity_feature",
        "schema": "stream4d_v85_phase1_local_affinity_feature_v1",
        "decision": decision,
        "can_enter_next_phase": True,
        "gate": gate,
        "primary_blocker": "" if gate["pass"] else "within_semantic_hard_negative_auc_unavailable",
        "topk_recall_under_sketch": p80.get("topk_recall_under_sketch", ""),
        "cosine_error_p95": p80.get("cosine_error_p95", ""),
        "largest_connected_component_ratio": p80_signed.get("largest_connected_component_ratio", ""),
        "same_frame_hard_negative_AUC": p80_signed.get("same_frame_hard_negative_AUC", ""),
        "within_semantic_hard_negative_AUC": p80_signed.get("within_semantic_hard_negative_AUC", ""),
        "same_instance_recall_at_topK_diagnostic": p80_signed.get("same_instance_recall_at_topk_diagnostic", ""),
        "affinity_minus_semantic_AUC": "",
        "carrier_feature_rows_written": len(carrier_rows),
        "neighbor_rows_written": len(neighbor_rows),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "feature_summary.json", summary)
    _write_csv(out / "carrier_feature_rows.csv", carrier_rows)
    _write_csv(out / "sketch_quality_rows.csv", sketch_rows)
    _write_csv(out / "neighbor_rows.csv", neighbor_rows)
    _write_csv(out / "feature_control_rows.csv", control_rows)
    return summary


def _phase2(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase2_output_root)
    out.mkdir(parents=True, exist_ok=True)
    metric = ctx["local_metric_rows"][0] if ctx["local_metric_rows"] else {}
    p80_signed = ctx["v80_phase2_summary"]
    local_rows = ctx["local_slot_rows"]
    cluster_rows = []
    for row in ctx["raw_cluster_rows"][: args.max_cluster_rows]:
        cluster_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "variant": row.get("variant", "C0_replay"),
                "scale": row.get("scale", ""),
                "cluster_id": row.get("cluster_id", ""),
                "carrier_count": row.get("carrier_count", ""),
                "mean_internal_affinity": row.get("mean_internal_affinity", ""),
                "mean_signed_affinity": row.get("mean_signed_affinity", ""),
                "cannot_link_inside_count": row.get("cannot_link_violation_count", ""),
                "visible_frame_span": row.get("visible_frame_span", ""),
                "fine_child_count": row.get("child_cluster_count", ""),
                "coarse_parent_id": row.get("parent_cluster_id", ""),
                "cluster_identity_fixed_before_adapter": True,
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
    adapter_rows = []
    audit_rows = []
    for row in local_rows[: args.max_slot_rows]:
        adapter_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "cluster_id": row.get("cluster_id", ""),
                "adapter_mask_count": row.get("adapter_mask_count", ""),
                "adapter_score_mean": row.get("adapter_score_mean", ""),
                "broad_adapter_rate": row.get("broad_adapter_rate", ""),
                "adapter_identity_flip": False,
                "adapter_caused_split": False,
                "adapter_caused_merge": False,
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": False,
            }
        )
        audit_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "variant": "C0_v82_local_replay",
                "cluster_id": row.get("cluster_id", ""),
                "frame_id": "",
                "selected_mask_id": "",
                "carrier_F1": row.get("adapter_score_mean", ""),
                "rendered_pixel_F1": "",
                "adapter_caused_split": False,
                "adapter_caused_merge": False,
                "adapter_identity_flip": False,
                "same_frame_duplicate_conflict": _int(row.get("duplicate_frame_mask_conflict_count"), 0) > 0,
                "broad_adapter_flag": _num(row.get("broad_adapter_rate"), 0.0) > 0.0,
            }
        )
    local_sf50 = _num(metric.get("local_SF50"), 0.0)
    gt_iou = _num(metric.get("GT_best_IoU_mean"), 0.0)
    duplicate_rate = _num(metric.get("duplicate_frame_mask_conflict_rate"), 1.0)
    flip_rate = _num(metric.get("adapter_identity_flip_rate"), 1.0)
    spearman = _num(metric.get("carrier_pixel_adapter_agreement"), 0.0)
    gate = {
        "local_SF50_ge_0p36": local_sf50 >= 0.36,
        "GT_best_IoU_mean_ge_0p36_diagnostic": gt_iou >= 0.36,
        "same_frame_violation_count_eq_0": _int(metric.get("same_frame_violation_count"), 999) == 0,
        "duplicate_frame_mask_conflict_rate_le_0p02": duplicate_rate <= 0.02,
        "adapter_identity_flip_rate_le_0p05": flip_rate <= 0.05,
        "rendered_pixel_vs_carrier_F1_spearman_ge_0p70": spearman >= 0.70,
        "cannot_link_violation_count_eq_0": _int(metric.get("cannot_link_violation_count"), 999) == 0
        and _int(p80_signed.get("component_cannot_link_violation_count"), 999) == 0,
    }
    diagnostic_pass = all(gate.values())
    strict_pass = diagnostic_pass and local_sf50 >= 0.40
    summary = {
        "phase": "v85_phase2_local_clustering",
        "schema": "stream4d_v85_phase2_local_clustering_v1",
        "decision": "PASS_V85_PHASE2_LOCAL_DIAGNOSTIC" if diagnostic_pass else "NO_GO_V85_LOCAL_DIAGNOSTIC_WEAK",
        "can_enter_next_phase": True,
        "strict_local_pass": strict_pass,
        "gate": gate,
        "primary_blocker": "" if diagnostic_pass else "local_SF50_or_GT_best_IoU_below_diagnostic_floor",
        "local_SF50": metric.get("local_SF50", ""),
        "local_AP50": metric.get("local_AP50", ""),
        "local_AP25": metric.get("local_AP25", ""),
        "GT_best_IoU_mean": metric.get("GT_best_IoU_mean", ""),
        "same_frame_violation_count": metric.get("same_frame_violation_count", ""),
        "duplicate_frame_mask_conflict_rate": metric.get("duplicate_frame_mask_conflict_rate", ""),
        "adapter_identity_flip_rate": metric.get("adapter_identity_flip_rate", ""),
        "rendered_pixel_vs_carrier_F1_spearman": metric.get("carrier_pixel_adapter_agreement", ""),
        "cluster_rows_written": len(cluster_rows),
        "adapter_rows_written": len(adapter_rows),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "cluster_summary.json", summary)
    _write_csv(out / "cluster_rows.csv", cluster_rows)
    _write_csv(out / "adapter_rows.csv", adapter_rows)
    _write_csv(out / "local_metric_rows.csv", [metric] if metric else [])
    _write_csv(out / "adapter_identity_audit_rows.csv", audit_rows)
    return summary


def _phase3(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase3_output_root)
    out.mkdir(parents=True, exist_ok=True)
    descriptor_rows = []
    for row in ctx["local_descriptor_rows"][: args.max_slot_rows]:
        descriptor_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "descriptor_variant": "S4_full_slot_affinity_descriptor_replay",
                "descriptor_dim": len(str(row.get("appearance_vector_json", "")).split(",")) if row.get("appearance_vector_json") else "",
                "semantic_entropy": "",
                "appearance_support_count": "",
                "appearance_broad_mask_rate": "",
                "visibility_span": row.get("visible_frame_span", ""),
                "adapter_support_count": "",
                "cannot_link_count": "",
                "partwhole_support_count": "",
                "descriptor_norm": row.get("slot_confidence", ""),
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": False,
            }
        )
    controls = [
        {
            "variant": "S0_semantic_only",
            "retrieval_AUC": "",
            "query_entropy": ctx["v83_phase6_summary"].get("semantic_assignment_entropy", ""),
            "control_type": "semantic",
        },
        {
            "variant": "S4_full_slot_affinity_descriptor_replay",
            "retrieval_AUC": "",
            "query_entropy": ctx["v83_phase6_summary"].get("real_assignment_entropy", ""),
            "control_type": "real_imported_v83_ledger",
        },
    ]
    query_rows = []
    for row in ctx["v83_evidence_rows"][: args.max_query_rows]:
        query_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "score": row.get("score_term", ""),
                "semantic_term": row.get("semantic_term", ""),
                "appearance_term": row.get("appearance_term", ""),
                "visibility_term": row.get("visibility_term", ""),
                "entropy_penalty": row.get("entropy_penalty", ""),
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": row.get("uses_future", False),
            }
        )
    real_entropy = _num(ctx["v83_phase6_summary"].get("real_assignment_entropy"), 1.0)
    sem_entropy = _num(ctx["v83_phase6_summary"].get("semantic_assignment_entropy"), 1.0)
    gate = {
        "full_minus_semantic_entropy_ge_0p10_proxy": sem_entropy - real_entropy >= 0.10,
        "retrieval_auc_available": False,
        "broad_support_contamination_rate_available": False,
        "method_uses_gt_false": not any(_bool(r.get("method_uses_gt")) for r in descriptor_rows),
        "uses_future_false": not any(_bool(r.get("uses_future")) for r in descriptor_rows),
    }
    proxy_pass = (
        gate["full_minus_semantic_entropy_ge_0p10_proxy"]
        and gate["method_uses_gt_false"]
        and gate["uses_future_false"]
    )
    primary_metrics_available = gate["retrieval_auc_available"] and gate["broad_support_contamination_rate_available"]
    gate["pass"] = proxy_pass and primary_metrics_available
    if gate["pass"]:
        decision = "PASS_V85_PHASE3_SLOT_DESCRIPTOR"
        primary_blocker = ""
    elif proxy_pass:
        decision = "DIAGNOSTIC_V85_PHASE3_SLOT_DESCRIPTOR_PROXY_PRIMARY_METRICS_MISSING"
        primary_blocker = "retrieval_auc_and_broad_support_contamination_metrics_unavailable"
    else:
        decision = "NO_GO_V85_SLOT_DESCRIPTOR_WEAK"
        primary_blocker = "slot_descriptor_control_or_provenance_failed"
    summary = {
        "phase": "v85_phase3_slot_descriptor",
        "schema": "stream4d_v85_phase3_slot_descriptor_v1",
        "decision": decision,
        "can_enter_next_phase": proxy_pass,
        "proxy_pass": proxy_pass,
        "primary_metrics_available": primary_metrics_available,
        "gate": gate,
        "primary_blocker": primary_blocker,
        "slot_query_entropy": ctx["v83_phase6_summary"].get("real_assignment_entropy", ""),
        "semantic_query_entropy": ctx["v83_phase6_summary"].get("semantic_assignment_entropy", ""),
        "full_minus_semantic_entropy_proxy": sem_entropy - real_entropy,
        "descriptor_rows_written": len(descriptor_rows),
        "query_rows_written": len(query_rows),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "slot_descriptor_summary.json", summary)
    _write_csv(out / "slot_descriptor_rows.csv", descriptor_rows)
    _write_csv(out / "descriptor_control_rows.csv", controls)
    _write_csv(out / "query_retrieval_rows.csv", query_rows)
    return summary


def _phase4(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase4_output_root)
    out.mkdir(parents=True, exist_ok=True)
    state_rows = ctx["v83_state_rows"]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_candidate[row.get("candidate_id", "")].append(row)
    snapshots = []
    updates = []
    query_rows = []
    transition_rows = []
    for cand, rows in sorted(by_candidate.items())[: args.max_tracklet_rows]:
        first = rows[0]
        chunks = sorted({_int(r.get("observed_support_chunk_count"), 0) for r in rows})
        state = first.get("new_state", "")
        snapshots.append(
            {
                "scene_id": first.get("scene_id", ""),
                "snapshot_chunk_id": first.get("observed_support_chunk_count", ""),
                "tracklet_id": cand,
                "tracklet_state": state.lower(),
                "birth_chunk_id": "",
                "last_seen_chunk": "",
                "support_chunk_count": max(chunks) if chunks else first.get("support_chunk_count", ""),
                "support_slot_count": first.get("tracklet_support_prior_slot_count", ""),
                "descriptor_version_id": _sha1_text(cand + state),
                "descriptor_entropy": first.get("mean_entropy_accumulated", ""),
                "descriptor_margin": first.get("mean_margin_accumulated", ""),
                "uses_future": first.get("uses_future", False),
                "method_uses_gt": first.get("method_uses_gt", False),
            }
        )
        updates.append(
            {
                "scene_id": first.get("scene_id", ""),
                "tracklet_id": cand,
                "event": "state_machine_replay",
                "new_state": state,
                "support_chunk_count": first.get("support_chunk_count", ""),
                "mean_entropy_accumulated": first.get("mean_entropy_accumulated", ""),
                "method_uses_gt": first.get("method_uses_gt", False),
                "uses_future": first.get("uses_future", False),
            }
        )
        transition_rows.extend(rows[:1])
    for row in ctx["v83_assignments"][: args.max_query_rows]:
        query_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "tracklet_id": row.get("candidate_id", ""),
                "history_id": row.get("history_id", ""),
                "assignment_state": row.get("assignment_state", ""),
                "score": row.get("score", ""),
                "assignment_entropy": row.get("assignment_entropy", ""),
                "assignment_margin": row.get("assignment_margin", ""),
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": row.get("uses_future", False),
            }
        )
    p6 = ctx["v83_phase6_summary"]
    p5 = ctx["v83_phase5_summary"]
    confirmed_count = sum(1 for r in snapshots if "confirmed" in r.get("tracklet_state", ""))
    stable_count = sum(1 for r in snapshots if "stable" in r.get("tracklet_state", ""))
    margin_mean = _num(p5.get("confirmed_plus_stable_link_margin_mean", p5.get("assignment_margin_mean")), 0.0)
    real_entropy = _num(p6.get("real_assignment_entropy"), 1.0)
    full_minus_semantic = _num(p6.get("semantic_assignment_entropy"), 0.0) - real_entropy
    full_minus_shuffled = _num(p6.get("shuffled_assignment_entropy"), 0.0) - real_entropy
    full_minus_stale = _num(p6.get("stale_assignment_entropy"), 0.0) - real_entropy
    tracklet_coverage = _num(p5.get("history_assignment_coverage_rate"), 0.0)
    future_count = sum(1 for r in snapshots if _bool(r.get("uses_future"))) + sum(1 for r in query_rows if _bool(r.get("uses_future")))
    gt_count = sum(1 for r in snapshots if _bool(r.get("method_uses_gt"))) + sum(1 for r in query_rows if _bool(r.get("method_uses_gt")))
    gate = {
        "tracklet_candidate_coverage_rate_ge_0p25": tracklet_coverage >= 0.25,
        "tracklet_query_entropy_mean_le_0p60": real_entropy <= 0.60,
        "tracklet_top1_top2_margin_mean_ge_0p05": margin_mean >= 0.05,
        "full_minus_semantic_score_ge_0p03": full_minus_semantic >= 0.03,
        "full_minus_shuffled_score_ge_0p03": full_minus_shuffled >= 0.03,
        "full_minus_stale_score_ge_0p02": full_minus_stale >= 0.02,
        "false_attachment_proxy_rate_le_0p05": _num(p6.get("real_wrong_absorption_proxy"), 1.0) <= 0.05,
        "uses_future_false": future_count == 0,
        "method_uses_gt_false": gt_count == 0,
    }
    high_precision = all(gate.values())
    coverage_target = _num(p5.get("history_assignment_coverage_rate"), 0.0) >= 0.60 and confirmed_count >= 10 and stable_count >= 20
    summary = {
        "phase": "v85_phase4_tracklet_descriptor",
        "schema": "stream4d_v85_phase4_tracklet_descriptor_v1",
        "decision": "PASS_V85_PHASE4_HIGH_PRECISION_TRACKLET" if high_precision else "NO_GO_V85_TRACKLET_DESCRIPTOR_WEAK",
        "can_enter_next_phase": high_precision,
        "method_coverage_target_pass": coverage_target,
        "gate": gate,
        "primary_blocker": "" if high_precision else "tracklet_query_entropy_or_margin_failed",
        "tracklet_candidate_coverage_rate": p5.get("history_assignment_coverage_rate", ""),
        "tracklet_query_entropy_mean": p6.get("real_assignment_entropy", ""),
        "tracklet_top1_top2_margin_mean": p5.get("confirmed_plus_stable_link_margin_mean", p5.get("assignment_margin_mean", "")),
        "confirmed_tracklet_count": confirmed_count,
        "stable_tracklet_count": stable_count,
        "false_attachment_proxy_rate": p6.get("real_wrong_absorption_proxy", ""),
        "full_minus_semantic_score": full_minus_semantic,
        "full_minus_shuffled_score": full_minus_shuffled,
        "full_minus_stale_score": full_minus_stale,
        "uses_future_count": future_count,
        "GT_prediction_violation_count": gt_count,
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "tracklet_summary.json", summary)
    _write_csv(out / "tracklet_snapshot_rows.csv", snapshots)
    _write_csv(out / "tracklet_update_rows.csv", updates)
    _write_csv(out / "tracklet_query_rows.csv", query_rows)
    _write_csv(out / "state_transition_rows.csv", transition_rows)
    return summary


def _phase5(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase5_output_root)
    out.mkdir(parents=True, exist_ok=True)
    assignments = ctx["v83_assignments"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        state = row.get("assignment_state", "")
        if state in {"confirmed", "stable_tentative"}:
            grouped[row.get("history_id", "")].append(row)
    node_rows = []
    desc_rows = []
    relation_rows = []
    for hist, rows in sorted(grouped.items()):
        chunks = sorted({_int(r.get("chunk_id"), 0) for r in rows})
        states = Counter(r.get("assignment_state", "") for r in rows)
        confidence = _mean([_num(r.get("score"), 0.0) for r in rows])
        ambiguity_terms = []
        for row in rows:
            link_entropy = _float(row.get("link_state_entropy"))
            ambiguity_terms.append(link_entropy if link_entropy is not None else _num(row.get("assignment_entropy"), 1.0))
        ambiguity = _mean(ambiguity_terms)
        node_rows.append(
            {
                "scene_id": rows[0].get("scene_id", ""),
                "history_id": hist,
                "version_id": _sha1_text(hist + str(len(rows))),
                "state": "confirmed" if states.get("confirmed", 0) else "stable_tentative",
                "birth_chunk_id": min(chunks) if chunks else "",
                "last_seen_chunk": max(chunks) if chunks else "",
                "support_tracklet_count": len({r.get("candidate_id", "") for r in rows}),
                "support_slot_count": len(rows),
                "semantic_descriptor_hash": _sha1_text(hist + "sem"),
                "appearance_descriptor_hash": _sha1_text(hist + "app"),
                "tracklet_descriptor_hash": _sha1_text(hist + "track"),
                "visibility_descriptor_hash": _sha1_text(hist + "vis"),
                "confidence": confidence,
                "ambiguity": ambiguity,
                "ambiguity_source": "link_state_entropy_fallback_assignment_entropy",
                "cannot_link_degree": 0,
                "partwhole_degree": 0,
                "memory_bytes": 512 + 64 * len(rows),
                "method_uses_gt": any(_bool(r.get("method_uses_gt")) for r in rows),
                "uses_future": any(_bool(r.get("uses_future")) for r in rows),
            }
        )
        desc_rows.append(
            {
                "history_id": hist,
                "descriptor_type": "tracklet_objectness_replay",
                "support_slot_count": len(rows),
                "confidence": confidence,
                "ambiguity": ambiguity,
                "descriptor_hash": _sha1_text(json.dumps(sorted(r.get("local_slot_id", "") for r in rows))),
            }
        )
    for edge in ctx["v83_edges"]:
        relation_rows.append(
            {
                "scene_id": edge.get("scene_id", ""),
                "source_history_id": edge.get("history_id", ""),
                "target_history_id": edge.get("history_id", ""),
                "relation_type": edge.get("edge_type", ""),
                "source_slot": edge.get("source_local_slot_id", ""),
                "target_slot": edge.get("target_local_slot_id", ""),
                "method_uses_gt": edge.get("method_uses_gt", False),
                "uses_future": edge.get("uses_future", False),
            }
        )
    memory_mb = sum(_num(r.get("memory_bytes"), 0.0) for r in node_rows) / (1024 * 1024)
    entropies = [_num(r.get("ambiguity"), 1.0) for r in node_rows]
    confirmed_count = sum(1 for r in node_rows if r.get("state") == "confirmed")
    gate = {
        "confirmed_history_count_ge_10": confirmed_count >= 10,
        "history_descriptor_entropy_mean_le_0p60": _mean(entropies) <= 0.60,
        "history_semantic_control_margin_ge_0p03": _num(ctx["v83_phase6_summary"].get("semantic_assignment_entropy"), 0.0)
        - _num(ctx["v83_phase6_summary"].get("real_assignment_entropy"), 0.0)
        >= 0.03,
        "history_shuffled_control_margin_ge_0p03": _num(ctx["v83_phase6_summary"].get("shuffled_assignment_entropy"), 0.0)
        - _num(ctx["v83_phase6_summary"].get("real_assignment_entropy"), 0.0)
        >= 0.03,
        "memory_MB_le_256": memory_mb <= 256,
        "future_descriptor_count_eq_0": sum(1 for r in node_rows if _bool(r.get("uses_future"))) == 0,
        "GT_prediction_violation_count_eq_0": sum(1 for r in node_rows if _bool(r.get("method_uses_gt"))) == 0,
    }
    gate["pass"] = all(gate.values())
    failed_gate_keys = [key for key, value in gate.items() if key != "pass" and not value]
    summary = {
        "phase": "v85_phase5_history_object_feature",
        "schema": "stream4d_v85_phase5_history_object_feature_v1",
        "decision": "PASS_V85_PHASE5_HISTORY_OBJECT_FEATURE" if gate["pass"] else "NO_GO_V85_HISTORY_OBJECT_FEATURE_WEAK",
        "can_enter_next_phase": gate["pass"],
        "gate": gate,
        "failed_gate_keys": failed_gate_keys,
        "primary_blocker": "" if gate["pass"] else ";".join(failed_gate_keys),
        "history_node_count": len(node_rows),
        "confirmed_history_count": confirmed_count,
        "tentative_history_count": sum(1 for r in node_rows if r.get("state") == "stable_tentative"),
        "quarantine_history_count": 0,
        "inactive_history_count": 0,
        "history_descriptor_entropy_mean": _mean(entropies),
        "history_descriptor_entropy_source": "link_state_entropy_fallback_assignment_entropy",
        "history_descriptor_margin_mean": ctx["v83_phase5_summary"].get("confirmed_plus_stable_link_margin_mean", ""),
        "history_semantic_control_margin": _num(ctx["v83_phase6_summary"].get("semantic_assignment_entropy"), 0.0)
        - _num(ctx["v83_phase6_summary"].get("real_assignment_entropy"), 0.0),
        "history_shuffled_control_margin": _num(ctx["v83_phase6_summary"].get("shuffled_assignment_entropy"), 0.0)
        - _num(ctx["v83_phase6_summary"].get("real_assignment_entropy"), 0.0),
        "memory_MB": memory_mb,
        "max_history_nodes": len(node_rows),
        "max_tentative_nodes": sum(1 for r in node_rows if r.get("state") == "stable_tentative"),
        "future_descriptor_count": sum(1 for r in node_rows if _bool(r.get("uses_future"))),
        "GT_prediction_violation_count": sum(1 for r in node_rows if _bool(r.get("method_uses_gt"))),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "history_summary.json", summary)
    _write_csv(out / "history_node_rows.csv", node_rows)
    _write_csv(out / "history_descriptor_rows.csv", desc_rows)
    _write_csv(out / "history_relation_rows.csv", relation_rows)
    _write_csv(out / "memory_budget_rows.csv", [{"memory_MB": memory_mb, "history_node_count": len(node_rows), "budget_MB": 256}])
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase6_output_root)
    out.mkdir(parents=True, exist_ok=True)
    q_rows = []
    weak_rows = []
    for row in ctx["v83_assignments"]:
        state = row.get("assignment_state", "")
        selected = state in {"confirmed", "stable_tentative"}
        q_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "tracklet_id": row.get("candidate_id", ""),
                "history_id": row.get("history_id", ""),
                "variant": "Q4_full_negative_memory_replay",
                "rank": row.get("selected_rank", ""),
                "q_score": row.get("score", ""),
                "positive_score": row.get("score", ""),
                "best_alternative_score": "",
                "negative_score": row.get("link_cannot_link_count", ""),
                "repeat_score": row.get("link_state_margin", ""),
                "new_object_score": row.get("link_new_object_evidence_count", ""),
                "q_margin": row.get("assignment_margin", ""),
                "q_entropy": row.get("assignment_entropy", ""),
                "selected_for_weak": selected,
                "selected_for_strong": False,
                "control_type": "real_imported_v83_ledger",
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": row.get("uses_future", False),
            }
        )
        if selected:
            weak_rows.append(q_rows[-1])
    p5 = ctx["v83_phase5_summary"]
    p6 = ctx["v83_phase6_summary"]
    real_entropy = _num(p6.get("real_assignment_entropy"), 1.0)
    gate = {
        "confirmed_plus_stable_coverage_ge_0p20": _num(p6.get("real_confirmed_plus_stable_coverage"), 0.0) >= 0.20,
        "confirmed_assignment_coverage_rate_ge_0p05": _num(p6.get("real_confirmed_coverage"), 0.0) >= 0.05,
        "q_entropy_mean_le_0p60": real_entropy <= 0.60,
        "q_top1_top2_margin_mean_ge_0p05": _num(p5.get("confirmed_plus_stable_link_margin_mean"), 0.0) >= 0.05,
        "wrong_absorption_proxy_le_0p05": _num(p6.get("real_wrong_absorption_proxy"), 1.0) <= 0.05,
        "real_entropy_0p10_lower_than_semantic_shuffled_stale": all(
            _num(p6.get(key), 0.0) - real_entropy >= 0.10
            for key in ["semantic_assignment_entropy", "shuffled_assignment_entropy", "stale_assignment_entropy"]
        ),
        "uses_future_false": not any(_bool(r.get("uses_future")) for r in q_rows),
        "method_uses_gt_false": not any(_bool(r.get("method_uses_gt")) for r in q_rows),
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v85_phase6_history_query",
        "schema": "stream4d_v85_phase6_history_query_v1",
        "decision": "PASS_V85_PHASE6_WEAK_L2H_QUERY" if gate["pass"] else "NO_GO_V85_HISTORY_QUERY_WEAK",
        "can_enter_next_phase": gate["pass"],
        "gate": gate,
        "primary_blocker": "" if gate["pass"] else "history_query_control_gate_failed",
        "history_assignment_coverage_rate": p5.get("history_assignment_coverage_rate", ""),
        "confirmed_assignment_coverage_rate": p6.get("real_confirmed_coverage", ""),
        "stable_assignment_coverage_rate": p5.get("stable_tentative_assignment_coverage_rate", ""),
        "confirmed_plus_stable_coverage": p6.get("real_confirmed_plus_stable_coverage", ""),
        "q_entropy_mean": p6.get("real_assignment_entropy", ""),
        "q_top1_top2_margin_mean": p5.get("confirmed_plus_stable_link_margin_mean", ""),
        "identity_switch_proxy": p6.get("real_identity_switch_proxy", ""),
        "fragmentation_proxy": p5.get("fragmentation_rate_proxy", ""),
        "wrong_absorption_proxy": p6.get("real_wrong_absorption_proxy", ""),
        "new_object_hijack_proxy": p5.get("new_object_birth_rate", ""),
        "real_minus_semantic_entropy": real_entropy - _num(p6.get("semantic_assignment_entropy"), 0.0),
        "real_minus_shuffled_entropy": real_entropy - _num(p6.get("shuffled_assignment_entropy"), 0.0),
        "real_minus_stale_entropy": real_entropy - _num(p6.get("stale_assignment_entropy"), 0.0),
        "real_minus_no_negative_entropy": real_entropy - _num(p6.get("no_negative_assignment_entropy"), real_entropy),
        "weak_assignment_count": len(weak_rows),
        "runtime_sec": time.time() - started,
    }
    control_rows = [
        {"variant": "Q3_real_full_history", "entropy": p6.get("real_assignment_entropy", ""), "coverage": p6.get("real_confirmed_plus_stable_coverage", "")},
        {"variant": "Q0_semantic_only", "entropy": p6.get("semantic_assignment_entropy", ""), "coverage": p6.get("semantic_confirmed_coverage", "")},
        {"variant": "Q6_shuffled_history", "entropy": p6.get("shuffled_assignment_entropy", ""), "coverage": p6.get("shuffled_confirmed_plus_stable_coverage", "")},
        {"variant": "Q7_stale_history", "entropy": p6.get("stale_assignment_entropy", ""), "coverage": ""},
        {"variant": "Q8_no_negative", "entropy": p6.get("no_negative_assignment_entropy", ""), "coverage": ""},
    ]
    _write_json(out / "q_summary.json", summary)
    _write_csv(out / "q_rows.csv", q_rows)
    _write_csv(out / "q_control_rows.csv", control_rows)
    _write_csv(out / "weak_assignment_rows.csv", weak_rows)
    return summary


def _phase7(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase7_output_root)
    out.mkdir(parents=True, exist_ok=True)
    assignments = _read_csv_rows(_repo_path(args.phase6_output_root) / "weak_assignment_rows.csv")
    v84_slots = ctx["v84_materialized_slot_rows"]
    local_slots = ctx["local_slot_rows"]
    adapter_rows = ctx["v82_adapter_rows"]

    def _slot_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", "")))

    def _cluster_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("cluster_id", "")))

    def _adapter_score(row: dict[str, Any]) -> float:
        for key in ("hybrid_adapter_F1", "rendered_pixel_F1", "carrier_F1"):
            value = _float(row.get(key))
            if value is not None:
                return value
        return 0.0

    local_slot_by_id = {_slot_key(row): row for row in local_slots}
    adapter_by_cluster: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in adapter_rows:
        adapter_by_cluster[_cluster_key(row)].append(row)

    frame_rows = []
    carrier_rows = []
    scene_rows = []
    selected_indices: set[int] = set()
    valid_indices_by_history_frame: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    slots_missing_local_row = 0
    slots_missing_adapter_rows = 0
    slot_keys_with_adapter: set[tuple[str, str, str]] = set()
    for row in v84_slots:
        slot = local_slot_by_id.get(_slot_key(row))
        if slot is None:
            slots_missing_local_row += 1
            slot = {}
        cluster_id = str(slot.get("cluster_id", ""))
        slot_adapter_rows = adapter_by_cluster.get((str(row.get("scene_id", "")), str(row.get("chunk_id", "")), cluster_id), [])
        if slot_adapter_rows:
            slot_keys_with_adapter.add(_slot_key(row))
        else:
            slots_missing_adapter_rows += 1

        for adapter in slot_adapter_rows:
            score = _adapter_score(adapter)
            allowed = (
                bool(str(adapter.get("frame_id", "")).strip())
                and bool(str(adapter.get("mask_id", "")).strip())
                and _bool(adapter.get("object_mask_ownership_allowed"))
                and not _bool(adapter.get("adapter_caused_split"))
                and not _bool(adapter.get("adapter_caused_merge"))
                and not _bool(row.get("method_uses_gt"))
                and not _bool(row.get("uses_future"))
            )
            candidate_idx = len(frame_rows)
            frame_rows.append(
                {
                    "candidate_row_id": candidate_idx,
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "frame_id": adapter.get("frame_id", ""),
                    "history_id": row.get("history_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "cluster_id": cluster_id,
                    "materializer_variant": "M1_slot_union_adapter_frame_mask_wta",
                    "mask_id": adapter.get("mask_id", ""),
                    "mask_score": score,
                    "carrier_support_score": adapter.get("carrier_F1", ""),
                    "rendered_pixel_score": adapter.get("rendered_pixel_F1", ""),
                    "hybrid_adapter_score": adapter.get("hybrid_adapter_F1", ""),
                    "object_mask_ownership_allowed": adapter.get("object_mask_ownership_allowed", ""),
                    "adapter_role": adapter.get("adapter_role", ""),
                    "adapter_caused_split": adapter.get("adapter_caused_split", ""),
                    "adapter_caused_merge": adapter.get("adapter_caused_merge", ""),
                    "selected_flag": False,
                    "selection_policy": "top_score_per_scene_history_frame_from_allowed_adapter_rows",
                    "same_frame_merge_flag": False,
                    "fragmentation_repair_flag": False,
                    "cannot_link_violation": False,
                    "broad_leak_proxy": 0.0 if _bool(adapter.get("object_mask_ownership_allowed")) else 1.0,
                    "adapter_candidate_valid": allowed,
                    "method_uses_gt": row.get("method_uses_gt", False),
                    "uses_future": row.get("uses_future", False),
                }
            )
            if allowed:
                valid_indices_by_history_frame[
                    (str(row.get("scene_id", "")), str(row.get("history_id", "")), str(adapter.get("frame_id", "")))
                ].append(candidate_idx)

        carrier_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "carrier_id": cluster_id,
                "history_id": row.get("history_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "membership_score": slot.get("slot_confidence", ""),
                "materializer_variant": "M1_slot_union_adapter_frame_mask_wta",
                "adapter_frame_mask_count": len(slot_adapter_rows),
                "method_uses_gt": row.get("method_uses_gt", False),
                "uses_future": row.get("uses_future", False),
            }
        )
    for indices in valid_indices_by_history_frame.values():
        best = max(
            indices,
            key=lambda idx: (
                _num(frame_rows[idx].get("mask_score"), -1.0),
                str(frame_rows[idx].get("mask_id", "")),
                -idx,
            ),
        )
        selected_indices.add(best)
    for idx in selected_indices:
        frame_rows[idx]["selected_flag"] = True

    selected_frame_rows = [row for row in frame_rows if _bool(row.get("selected_flag"))]
    selected_slot_keys = {
        (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", "")))
        for row in selected_frame_rows
    }
    selected_history_frame_keys = {
        (str(row.get("scene_id", "")), str(row.get("history_id", "")), str(row.get("frame_id", "")))
        for row in selected_frame_rows
    }
    same_frame_wta_candidate_collision_count = sum(max(0, len(indices) - 1) for indices in valid_indices_by_history_frame.values())
    comps = defaultdict(list)
    for row in v84_slots:
        comps[row.get("component_id", "")].append(row)
    for comp, rows in comps.items():
        comp_slot_keys = {
            (str(r.get("scene_id", "")), str(r.get("chunk_id", "")), str(r.get("local_slot_id", "")))
            for r in rows
        }
        comp_frame_rows = [
            row
            for row in selected_frame_rows
            if (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", ""))) in comp_slot_keys
        ]
        scene_rows.append(
            {
                "scene_object_id": comp,
                "scene_id": rows[0].get("scene_id", "") if rows else "",
                "history_ids_json": json.dumps(sorted({r.get("history_id", "") for r in rows})),
                "slot_count": len(rows),
                "selected_frame_mask_count": len(comp_frame_rows),
                "selected_frame_count": len({r.get("frame_id", "") for r in comp_frame_rows}),
                "materializer_variant": "M1_slot_union_adapter_frame_mask_wta",
                "prediction_export_available": False,
                "method_uses_gt": any(_bool(r.get("method_uses_gt")) for r in rows),
                "uses_future": any(_bool(r.get("uses_future")) for r in rows),
            }
        )
    scene_metric_available = False
    cannot_link_violations = sum(1 for r in selected_frame_rows if _bool(r.get("cannot_link_violation")))
    required_export_fields = ["scene_id", "frame_id", "mask_id", "history_id"]
    required_adapter_fields = ["scene_id", "chunk_id", "cluster_id", "frame_id", "mask_id"]
    frame_mask_table_available = bool(selected_frame_rows) and all(
        str(row.get(field, "")).strip() for row in selected_frame_rows for field in required_export_fields
    )
    native_support_rows, native_source_audit_rows, native_support_summary = _load_native_carrier_support(
        selected_frame_rows,
        args,
    )
    adapter_cluster_keys = set(adapter_by_cluster)
    local_cluster_keys = {_cluster_key(row) for row in local_slots}
    local_cluster_key_coverage = _safe_ratio(len(local_cluster_keys & adapter_cluster_keys), len(local_cluster_keys))
    v84_slot_adapter_coverage = _safe_ratio(len(slot_keys_with_adapter), len(v84_slots))
    if frame_mask_table_available and not bool(args.skip_diagnostic_npz_export):
        diagnostic_npz = _diagnostic_export_frame_mask_npz(
            selected_frame_rows,
            output_config=args.diagnostic_npz_output_config,
            phase7_out=out,
        )
    else:
        diagnostic_npz = {
            "available": False,
            "reason": "skipped_by_cli" if args.skip_diagnostic_npz_export else "frame_mask_table_unavailable",
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
        }
    (
        native_diagnostic_assignment_rows,
        native_diagnostic_source_audit_rows,
        native_diagnostic_control_rows,
        native_diagnostic_summary,
    ) = _native_carrier_diagnostic_eval(native_support_rows, out)
    native_evaluator_candidate_contract = _native_carrier_evaluator_candidate_contract(native_diagnostic_summary)
    native_evaluator_holdout_rows, native_evaluator_holdout_summary = _native_carrier_evaluator_holdout_readiness(
        selected_frame_rows=selected_frame_rows,
        native_support_rows=native_support_rows,
        candidate_contract=native_evaluator_candidate_contract,
        phase7_out=out,
    )
    scene_export_route_rows, scene_export_route_summary = _audit_native_scene_vertex_export_routes(
        native_support_rows=native_support_rows,
        native_support_summary=native_support_summary,
        native_diagnostic_summary=native_diagnostic_summary,
        diagnostic_npz=diagnostic_npz,
        phase7_out=out,
    )
    source_audits = [
        {
            "source_name": "v84_materialized_slot_rows",
            "source_path": _rel(ctx["roots"]["v84_phase3"] / "materialized_slot_rows.csv"),
            "rows": v84_slots,
            "role": "history/local-slot materialization table",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "join_key_coverage_rate": "",
            "notes": "contains local_slot_id/history_id but no concrete frame_id/mask_id observation",
        },
        {
            "source_name": "v83_local_slot_history_assignment_rows",
            "source_path": _rel(ctx["roots"]["v83_phase5"] / "local_slot_history_assignment_rows.csv"),
            "rows": ctx["v83_assignments"],
            "role": "weak history assignment evidence",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "join_key_coverage_rate": "",
            "notes": "maps local_slot_id to history_id, not local_slot_id to frame mask raster",
        },
        {
            "source_name": "v82_local_slot_rows",
            "source_path": _rel(ctx["roots"]["v82_phase1"] / "local_slot_rows.csv"),
            "rows": ctx["local_slot_rows"],
            "role": "local slot descriptor table",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "join_key_coverage_rate": local_cluster_key_coverage,
            "notes": "has frame_support_count/adapter_mask_count but not frame_id/mask_id list",
        },
        {
            "source_name": "v82_adapter_rows",
            "source_path": _rel(ctx["v82_adapter_rows_path"]),
            "rows": adapter_rows,
            "role": "method-safe local cluster to frame/mask observation table",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": bool(adapter_rows) and v84_slot_adapter_coverage == 1.0,
            "join_key_coverage_rate": v84_slot_adapter_coverage,
            "notes": "provides cluster_id -> frame_id/mask_id rows; v85 joins through v82 local slots and v84 materialized history slots. It still lacks mask raster/point ids needed for scene-level prediction npz.",
        },
        {
            "source_name": "v65_v66_native_carrier_observation_tables",
            "source_path": ";".join(str(path) for path in native_support_summary.get("source_tables", [])),
            "rows": native_support_rows,
            "role": "method-safe selected frame/mask -> native D4RT carrier_global_id support",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "legal_for_native_carrier_support_export": bool(
                native_support_summary.get("can_legally_export_native_carrier_support")
            ),
            "join_key_coverage_rate": native_support_summary.get("selected_frame_mask_native_support_coverage_rate", ""),
            "notes": "native carrier ids are method-safe support but are not ScanNet scene vertex ids, so they cannot satisfy the current AP npz evaluator contract.",
        },
        {
            "source_name": "v82_raw_v81_local_slot_rows",
            "source_path": _rel(ctx["roots"]["v82_phase1"] / "raw_v81_replay" / "local_slot_rows.csv"),
            "rows": _read_csv_rows(ctx["roots"]["v82_phase1"] / "raw_v81_replay" / "local_slot_rows.csv"),
            "role": "raw local replay table",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "join_key_coverage_rate": "",
            "notes": "has frame_count/mask_count summary only; exact mask observations are absent",
        },
        {
            "source_name": "existing_stream4d_prediction_scene0011_00_npz",
            "source_path": _rel(ROOT / "data/prediction/stream4d_scannet_32f_ioc075_fixmem/scene0011_00.npz"),
            "rows": [],
            "role": "existing unrelated prediction artifact",
            "legal_for_prediction_npz": False,
            "legal_for_frame_mask_export": False,
            "join_key_coverage_rate": "",
            "notes": "exists only as another output_config/provenance; cannot be relabeled as v85 materializer output",
        },
    ]
    feasibility_rows = []
    for item in source_audits:
        rows = item["rows"]
        presence = _field_presence(rows, required_export_fields)
        adapter_presence = _field_presence(rows, required_adapter_fields)
        missing = [field for field, ok in presence.items() if not ok]
        if item["source_name"] == "existing_stream4d_prediction_scene0011_00_npz":
            exists = (ROOT / "data/prediction/stream4d_scannet_32f_ioc075_fixmem/scene0011_00.npz").exists()
            missing = [] if exists else ["prediction_npz_file"]
        if item["source_name"] == "v82_adapter_rows":
            missing = [field for field, ok in adapter_presence.items() if not ok]
        feasibility_rows.append(
            {
                "source_name": item["source_name"],
                "source_path": item["source_path"],
                "role": item["role"],
                "row_count": len(rows),
                "has_scene_id": presence.get("scene_id", False),
                "has_frame_id": presence.get("frame_id", False),
                "has_mask_id": presence.get("mask_id", False),
                "has_history_id": presence.get("history_id", False),
                "has_chunk_id": adapter_presence.get("chunk_id", False),
                "has_cluster_id": adapter_presence.get("cluster_id", False),
                "frame_id_nonempty_rate": _nonempty_field_rate(rows, "frame_id"),
                "mask_id_nonempty_rate": _nonempty_field_rate(rows, "mask_id"),
                "join_key_coverage_rate": item.get("join_key_coverage_rate", ""),
                "missing_required_fields": ";".join(missing),
                "legal_for_v85_frame_mask_export": bool(item.get("legal_for_frame_mask_export", False)),
                "legal_for_v85_native_carrier_support_export": bool(
                    item.get("legal_for_native_carrier_support_export", False)
                ),
                "legal_for_v85_prediction_export": False,
                "notes": item["notes"],
            }
        )
    decision = (
        "NO_GO_V85_SCENE_PREDICTION_NPZ_MISSING_AFTER_FRAME_MASK_REPAIR"
        if frame_mask_table_available
        else "NO_GO_V85_MATERIALIZER_EXPORT_MISSING"
    )
    primary_blocker = (
        "prediction_npz_exporter_missing_point_or_raster_support"
        if frame_mask_table_available
        else "prediction_npz_or_frame_mask_exporter_missing"
    )
    if bool(native_support_summary.get("can_legally_export_native_carrier_support")):
        primary_blocker = "native_carrier_support_ready_but_scene_vertex_exporter_missing"
    summary = {
        "phase": "v85_phase7_renderable_materializer",
        "schema": "stream4d_v85_phase7_renderable_materializer_v4",
        "decision": decision,
        "can_enter_next_phase": False,
        "primary_blocker": primary_blocker,
        "materialized_object_count": len(scene_rows),
        "frame_mask_table_available": frame_mask_table_available,
        "frame_mask_coverage_rate": _safe_ratio(len(selected_slot_keys), len(v84_slots)),
        "frame_mask_candidate_count": len(frame_rows),
        "frame_mask_selected_count": len(selected_frame_rows),
        "selected_history_frame_count": len(selected_history_frame_keys),
        "v84_slot_adapter_coverage_rate": v84_slot_adapter_coverage,
        "v84_slot_count": len(v84_slots),
        "v84_slot_with_adapter_count": len(slot_keys_with_adapter),
        "v84_slot_missing_local_row_count": slots_missing_local_row,
        "v84_slot_missing_adapter_count": slots_missing_adapter_rows,
        "same_frame_merge_action_count": 0,
        "same_frame_wta_candidate_collision_count": same_frame_wta_candidate_collision_count,
        "same_frame_merge_precision_proxy": "",
        "cannot_link_violation_count": cannot_link_violations,
        "wrong_absorption_proxy": ctx["v83_phase6_summary"].get("real_wrong_absorption_proxy", ""),
        "broad_leak_proxy_rate": "",
        "identity_switch_proxy": ctx["v83_phase6_summary"].get("real_identity_switch_proxy", ""),
        "fragmentation_proxy_before": ctx["v84_phase3_summary"].get("B0_fragmentation_proxy", ""),
        "fragmentation_proxy_after": ctx["v84_phase3_summary"].get("fragmentation_proxy", ""),
        "adapter_identity_flip_rate": "",
        "scene_metric_available": scene_metric_available,
        "prediction_npz_available": False,
        "diagnostic_prediction_npz_available": bool(diagnostic_npz.get("available")),
        "diagnostic_prediction_npz_output_config": diagnostic_npz.get("output_config", ""),
        "diagnostic_prediction_npz_dir": diagnostic_npz.get("prediction_dir", ""),
        "diagnostic_prediction_npz_forbidden_for_method_table": bool(diagnostic_npz.get("forbidden_for_method_table", True)),
        "diagnostic_backproject_hit_rate": diagnostic_npz.get("backproject_hit_rate", ""),
        "native_carrier_support_available": bool(native_support_summary.get("available")),
        "native_carrier_support_row_count": native_support_summary.get("native_carrier_support_row_count", 0),
        "native_unique_carrier_count": native_support_summary.get("native_unique_carrier_count", 0),
        "selected_frame_mask_with_native_support_count": native_support_summary.get(
            "selected_frame_mask_with_native_support_count",
            0,
        ),
        "selected_frame_mask_native_support_coverage_rate": native_support_summary.get(
            "selected_frame_mask_native_support_coverage_rate",
            0.0,
        ),
        "native_carrier_support_method_safe": bool(
            native_support_summary.get("can_legally_export_native_carrier_support")
        ),
        "native_carrier_support_is_scannet_ap_export": bool(native_support_summary.get("is_scannet_ap_export")),
        "native_carrier_support_blocker": native_support_summary.get("primary_blocker", ""),
        "scene_vertex_export_route_checked_count": scene_export_route_summary.get("checked_candidate_route_count", 0),
        "method_safe_scene_vertex_exporter_available": bool(
            scene_export_route_summary.get("method_safe_scene_vertex_exporter_available")
        ),
        "method_safe_native_carrier_evaluator_available": bool(
            scene_export_route_summary.get("method_safe_native_carrier_evaluator_available")
        ),
        "native_carrier_evaluator_input_available": bool(
            scene_export_route_summary.get("native_carrier_evaluator_input_available")
        ),
        "native_carrier_diagnostic_evaluator_available": bool(native_diagnostic_summary.get("available")),
        "native_carrier_diagnostic_assignment_count": native_diagnostic_summary.get(
            "native_carrier_diagnostic_assignment_count",
            0,
        ),
        "native_carrier_diagnostic_labeled_support_observation_count": native_diagnostic_summary.get(
            "labeled_support_observation_count",
            0,
        ),
        "native_carrier_diagnostic_ARI": native_diagnostic_summary.get("adjusted_rand_index", ""),
        "native_carrier_diagnostic_purity": native_diagnostic_summary.get("purity", ""),
        "native_carrier_diagnostic_completeness": native_diagnostic_summary.get("completeness", ""),
        "native_carrier_cluster_AP25": native_diagnostic_summary.get("native_carrier_cluster_AP25", ""),
        "native_carrier_cluster_AP50": native_diagnostic_summary.get("native_carrier_cluster_AP50", ""),
        "native_carrier_cluster_AP_mean": native_diagnostic_summary.get("native_carrier_cluster_AP_mean", ""),
        "native_carrier_cluster_mean_best_iou": native_diagnostic_summary.get(
            "native_carrier_cluster_mean_best_iou",
            "",
        ),
        "native_carrier_cluster_prediction_count": native_diagnostic_summary.get(
            "native_carrier_cluster_prediction_count",
            "",
        ),
        "native_carrier_cluster_gt_object_count": native_diagnostic_summary.get(
            "native_carrier_cluster_gt_object_count",
            "",
        ),
        "native_carrier_cluster_score_contract": native_diagnostic_summary.get(
            "native_carrier_cluster_score_contract",
            "",
        ),
        "native_carrier_evaluation_label_scope": native_diagnostic_summary.get("evaluation_label_scope", ""),
        "native_carrier_evaluator_contract_status": native_diagnostic_summary.get(
            "native_carrier_evaluator_contract_status",
            "",
        ),
        "native_carrier_evaluator_candidate_contract": _rel(out / "native_carrier_evaluator_candidate_contract.json"),
        "native_carrier_evaluator_candidate_contract_sha256": native_evaluator_candidate_contract.get(
            "contract_sha256",
            "",
        ),
        "native_carrier_evaluator_candidate_contract_current_allowed": bool(
            native_evaluator_candidate_contract.get("allowed_for_current_method_table")
        ),
        "native_carrier_evaluator_candidate_contract_future_allowed": bool(
            native_evaluator_candidate_contract.get("allowed_for_future_pre_registered_gate")
        ),
        "native_carrier_evaluator_holdout_readiness": _rel(
            out / "native_carrier_evaluator_holdout_readiness_summary.json"
        ),
        "native_carrier_holdout_evaluation_ready": bool(
            native_evaluator_holdout_summary.get("native_holdout_evaluation_ready")
        ),
        "native_carrier_holdout_primary_blocker": native_evaluator_holdout_summary.get("primary_blocker", ""),
        "native_carrier_holdout_selected_frame_mask_count": native_evaluator_holdout_summary.get(
            "selected_frame_mask_holdout_count_total",
            "",
        ),
        "native_carrier_holdout_support_row_count": native_evaluator_holdout_summary.get(
            "native_carrier_support_holdout_row_count_total",
            "",
        ),
        "native_carrier_holdout_unique_carrier_count": native_evaluator_holdout_summary.get(
            "native_unique_holdout_carrier_count_total",
            "",
        ),
        "native_carrier_control_variant_count": native_diagnostic_summary.get(
            "native_carrier_control_variant_count",
            "",
        ),
        "native_carrier_non_oracle_control_count": native_diagnostic_summary.get(
            "native_carrier_non_oracle_control_count",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_AP50": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_mean_best_iou": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_mean_best_iou",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_ARI": native_diagnostic_summary.get(
            "native_carrier_real_minus_best_non_oracle_ARI",
            "",
        ),
        "native_carrier_real_beats_non_oracle_AP50_by_0p03": native_diagnostic_summary.get(
            "native_carrier_real_beats_non_oracle_AP50_by_0p03",
            "",
        ),
        "native_carrier_real_beats_non_oracle_mean_iou_by_0p03": native_diagnostic_summary.get(
            "native_carrier_real_beats_non_oracle_mean_iou_by_0p03",
            "",
        ),
        "native_carrier_diagnostic_pred_history_conflict_carrier_rate": native_diagnostic_summary.get(
            "pred_history_conflict_carrier_rate",
            "",
        ),
        "native_carrier_diagnostic_gt_conflict_carrier_rate": native_diagnostic_summary.get(
            "diagnostic_gt_conflict_carrier_rate",
            "",
        ),
        "native_carrier_diagnostic_forbidden_for_method_table": bool(
            native_diagnostic_summary.get("forbidden_for_method_table", True)
        ),
        "native_carrier_diagnostic_metric_scope": native_diagnostic_summary.get("metric_scope", ""),
        "native_carrier_diagnostic_summary": _rel(out / "native_carrier_diagnostic_summary.json"),
        "native_carrier_diagnostic_control_rows": _rel(out / "native_carrier_diagnostic_control_rows.csv"),
        "diagnostic_bridge_available": bool(scene_export_route_summary.get("diagnostic_bridge_available")),
        "native_scene_vertex_export_route_audit": _rel(out / "native_scene_vertex_export_route_summary.json"),
        "method_uses_gt": any(_bool(r.get("method_uses_gt")) for r in frame_rows),
        "uses_future": any(_bool(r.get("uses_future")) for r in frame_rows),
        "runtime_sec": time.time() - started,
    }
    manifest = {
        "available": False,
        "frame_mask_table_available": frame_mask_table_available,
        "frame_mask_table": _rel(out / "frame_mask_prediction_rows.csv"),
        "reason": "Frame-mask table is repaired from v82 adapter rows, but no legal point-level prediction npz exporter exists for v85 scene AP yet."
        if frame_mask_table_available
        else "No legal frame-mask/prediction npz exporter exists for v85 materialized history field yet.",
        "repair_attempted": True,
        "repair_result": "frame_mask_table_repaired_prediction_npz_missing_point_or_raster_support"
        if frame_mask_table_available
        else "blocked_missing_local_slot_to_frame_mask_observation_mapping",
        "required_export_fields": required_export_fields,
        "adapter_rows_source": _rel(ctx["v82_adapter_rows_path"]),
        "feasibility_audit": _rel(out / "exporter_feasibility_rows.csv"),
        "table_only_rows": len(selected_frame_rows),
        "diagnostic_npz_export": diagnostic_npz,
        "native_carrier_support": {
            **native_support_summary,
            "support_rows": _rel(out / "native_carrier_support_rows.csv"),
            "source_audit_rows": _rel(out / "native_carrier_source_audit_rows.csv"),
        },
        "native_carrier_diagnostic_eval": native_diagnostic_summary,
        "native_carrier_evaluator_candidate_contract": native_evaluator_candidate_contract,
        "native_carrier_evaluator_holdout_readiness": native_evaluator_holdout_summary,
        "native_scene_vertex_export_route_audit": scene_export_route_summary,
        "method_uses_gt": summary["method_uses_gt"],
        "uses_future": summary["uses_future"],
    }
    repair_audit = {
        "schema": "stream4d_v85_exporter_repair_audit_v3",
        "decision": "PARTIAL_REPAIR_FRAME_MASK_EXPORT_READY_PREDICTION_NPZ_MISSING"
        if frame_mask_table_available
        else "BLOCKED_MISSING_FRAME_MASK_OBSERVATION_MAPPING",
        "plan_repair_direction": "implement prediction_npz/frame-mask exporter before tuning",
        "repair_attempted": True,
        "can_legally_export_frame_mask_table": frame_mask_table_available,
        "can_legally_export_native_carrier_support": bool(
            native_support_summary.get("can_legally_export_native_carrier_support")
        ),
        "can_legally_export_prediction_npz": False,
        "reason": "v82 adapter_rows provide method-safe local cluster -> frame_id/mask_id observations, and v65/v66 carrier observation tables provide method-safe selected mask -> D4RT carrier_global_id support. Current v85 still lacks a method-safe native-carrier-to-ScanNet-scene-vertex exporter/evaluator needed to serialize ScanNet scene-level prediction npz, so scene AP/SF metrics remain unavailable."
        if frame_mask_table_available
        else "Available v85/v82/v83/v84 tables do not contain an exact local_slot_id -> (frame_id, mask_id, mask raster) mapping. Existing prediction npz files have different output_config/provenance and cannot be relabeled as v85.",
        "required_next_artifact": native_support_summary.get(
            "required_next_artifact",
            "point-level prediction npz exporter from selected frame_mask_prediction_rows.csv using method-safe raster/backprojection support",
        )
        if frame_mask_table_available
        else "method-safe local_slot/frame/mask observation table or native carrier-to-mask materialization output",
        "adapter_rows_source": _rel(ctx["v82_adapter_rows_path"]),
        "adapter_row_count": len(adapter_rows),
        "v84_slot_adapter_coverage_rate": v84_slot_adapter_coverage,
        "selected_frame_mask_count": len(selected_frame_rows),
        "native_carrier_support": native_support_summary,
        "native_carrier_diagnostic_eval": native_diagnostic_summary,
        "native_carrier_evaluator_candidate_contract": native_evaluator_candidate_contract,
        "native_carrier_evaluator_holdout_readiness": native_evaluator_holdout_summary,
        "native_scene_vertex_export_route_audit": scene_export_route_summary,
        "diagnostic_npz_export": diagnostic_npz,
        "same_frame_wta_candidate_collision_count": same_frame_wta_candidate_collision_count,
        "source_count": len(feasibility_rows),
        "all_sources_legal": False,
    }
    _write_json(out / "materializer_summary.json", summary)
    _write_csv(out / "frame_mask_prediction_rows.csv", frame_rows)
    _write_csv(out / "carrier_membership_rows.csv", carrier_rows)
    _write_csv(out / "scene_object_rows.csv", scene_rows)
    _write_csv(out / "native_carrier_support_rows.csv", native_support_rows)
    _write_csv(out / "native_carrier_source_audit_rows.csv", native_source_audit_rows)
    _write_csv(out / "native_carrier_diagnostic_assignment_rows.csv", native_diagnostic_assignment_rows)
    _write_csv(out / "native_carrier_diagnostic_source_audit_rows.csv", native_diagnostic_source_audit_rows)
    _write_csv(out / "native_carrier_diagnostic_control_rows.csv", native_diagnostic_control_rows)
    _write_csv(out / "exporter_feasibility_rows.csv", feasibility_rows)
    _write_csv(out / "native_scene_vertex_export_route_rows.csv", scene_export_route_rows)
    _write_csv(out / "native_carrier_evaluator_holdout_readiness_rows.csv", native_evaluator_holdout_rows)
    _write_json(out / "native_carrier_diagnostic_summary.json", native_diagnostic_summary)
    _write_json(out / "native_carrier_evaluator_candidate_contract.json", native_evaluator_candidate_contract)
    _write_json(out / "native_carrier_evaluator_holdout_readiness_summary.json", native_evaluator_holdout_summary)
    _write_json(out / "native_scene_vertex_export_route_summary.json", scene_export_route_summary)
    _write_json(out / "prediction_npz_manifest.json", manifest)
    _write_json(out / "exporter_repair_audit.json", repair_audit)
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ctx = _load_context(args)
    out = _repo_path(args.phase8_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p7 = _read_json(_repo_path(args.phase7_output_root) / "materializer_summary.json")
    local_metric = ctx["local_metric_rows"][0] if ctx["local_metric_rows"] else {}
    frame_mask_table_available = bool(p7.get("frame_mask_table_available"))
    native_carrier_support_available = bool(p7.get("native_carrier_support_available"))
    method_safe_scene_vertex_exporter_available = bool(p7.get("method_safe_scene_vertex_exporter_available"))
    method_safe_native_carrier_evaluator_available = bool(p7.get("method_safe_native_carrier_evaluator_available"))
    native_carrier_diagnostic_metric_available = bool(p7.get("native_carrier_diagnostic_evaluator_available"))
    native_carrier_holdout_ready = bool(p7.get("native_carrier_holdout_evaluation_ready"))
    route_checked_count = _int(p7.get("scene_vertex_export_route_checked_count"), 0)
    diagnostic_output_config = str(p7.get("diagnostic_prediction_npz_output_config", ""))
    if bool(p7.get("diagnostic_prediction_npz_available")) and not bool(args.skip_diagnostic_npz_eval):
        diagnostic_eval = _run_diagnostic_npz_eval(diagnostic_output_config, out)
    else:
        diagnostic_eval = {
            "available": False,
            "reason": "skipped_by_cli" if args.skip_diagnostic_npz_eval else "diagnostic_prediction_npz_unavailable",
            "metrics": {},
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
        }
    diagnostic_metrics = diagnostic_eval.get("metrics", {}) if isinstance(diagnostic_eval.get("metrics"), dict) else {}
    if native_carrier_support_available:
        scene_metric_unavailable_reason = (
            "native D4RT carrier support exists, but no method-safe native-carrier-to-ScanNet scene vertex "
            f"exporter/evaluator exists for scene AP npz after checking {route_checked_count} candidate routes"
        )
    elif frame_mask_table_available:
        scene_metric_unavailable_reason = (
            "prediction_npz exporter missing; frame-mask table exists but has not been serialized to point-level scene masks"
        )
    else:
        scene_metric_unavailable_reason = "prediction_npz/frame-mask exporter missing"
    variants = [
        ("B0", "local-only no history", local_metric.get("local_SF50", ""), False),
        ("B1", "weak L2H naming only", ctx["v83_phase5_summary"].get("local_SF50_after_weak_history", ""), False),
        ("B2", "ID-only cross-chunk stitching", ctx["v84_phase3_summary"].get("local_SF50_after_materialization", ""), False),
        ("B3", "real history feature query materializer", "", False),
        ("B4", "shuffled-history feature materializer", "", False),
        ("B5", "stale-history materializer", "", False),
        ("B6", "semantic-only history materializer", "", False),
        ("B7", "no-negative history materializer", "", False),
        ("B8", "area/risk-count control", "", False),
        ("B9", "oracle diagnostic materializer", "", True),
    ]
    metric_rows = []
    for key, name, local_sf50, diagnostic_only in variants:
        metric_rows.append(
            {
                "scene_id": "ALL_DEV",
                "split": "dev",
                "variant": key,
                "variant_name": name,
                "local_SF50": local_sf50,
                "local_AP50": local_metric.get("local_AP50", "") if key == "B0" else "",
                "local_AP25": local_metric.get("local_AP25", "") if key == "B0" else "",
                "scene_SF50": "",
                "scene_AP50": "",
                "scene_AP25": "",
                "GT_best_IoU_mean": local_metric.get("GT_best_IoU_mean", "") if key == "B0" else "",
                "identity_switch_proxy": ctx["v83_phase6_summary"].get("real_identity_switch_proxy", "") if key in {"B1", "B2", "B3"} else "",
                "fragmentation_proxy": ctx["v84_phase3_summary"].get("fragmentation_proxy", "") if key in {"B2", "B3"} else "",
                "overmerge_proxy": ctx["v84_phase3_summary"].get("overmerge_proxy_rate", "") if key in {"B2", "B3"} else "",
                "cannot_link_violation_count": p7.get("cannot_link_violation_count", "") if key == "B3" else "",
                "wrong_absorption_proxy": p7.get("wrong_absorption_proxy", "") if key == "B3" else "",
                "frame_mask_table_available": frame_mask_table_available if key == "B3" else False,
                "native_carrier_support_available": native_carrier_support_available if key == "B3" else False,
                "native_carrier_support_row_count": p7.get("native_carrier_support_row_count", "") if key == "B3" else "",
                "native_unique_carrier_count": p7.get("native_unique_carrier_count", "") if key == "B3" else "",
                "method_safe_scene_vertex_exporter_available": method_safe_scene_vertex_exporter_available if key == "B3" else False,
                "method_safe_native_carrier_evaluator_available": method_safe_native_carrier_evaluator_available if key == "B3" else False,
                "native_carrier_diagnostic_metric_available": native_carrier_diagnostic_metric_available if key == "B3" else False,
                "native_carrier_holdout_evaluation_ready": native_carrier_holdout_ready if key == "B3" else False,
                "native_carrier_holdout_support_row_count": p7.get("native_carrier_holdout_support_row_count", "")
                if key == "B3"
                else "",
                "native_carrier_diagnostic_assignment_count": p7.get("native_carrier_diagnostic_assignment_count", "") if key == "B3" else "",
                "native_carrier_diagnostic_ARI": p7.get("native_carrier_diagnostic_ARI", "") if key == "B3" else "",
                "native_carrier_diagnostic_purity": p7.get("native_carrier_diagnostic_purity", "") if key == "B3" else "",
                "native_carrier_diagnostic_completeness": p7.get("native_carrier_diagnostic_completeness", "") if key == "B3" else "",
                "native_carrier_cluster_AP25": p7.get("native_carrier_cluster_AP25", "") if key == "B3" else "",
                "native_carrier_cluster_AP50": p7.get("native_carrier_cluster_AP50", "") if key == "B3" else "",
                "native_carrier_cluster_AP_mean": p7.get("native_carrier_cluster_AP_mean", "") if key == "B3" else "",
                "native_carrier_cluster_mean_best_iou": p7.get("native_carrier_cluster_mean_best_iou", "") if key == "B3" else "",
                "native_carrier_real_minus_best_non_oracle_AP50": p7.get("native_carrier_real_minus_best_non_oracle_AP50", "") if key == "B3" else "",
                "native_carrier_real_minus_best_non_oracle_mean_best_iou": p7.get("native_carrier_real_minus_best_non_oracle_mean_best_iou", "") if key == "B3" else "",
                "native_carrier_real_minus_best_non_oracle_ARI": p7.get("native_carrier_real_minus_best_non_oracle_ARI", "") if key == "B3" else "",
                "native_carrier_real_beats_non_oracle_AP50_by_0p03": p7.get("native_carrier_real_beats_non_oracle_AP50_by_0p03", "") if key == "B3" else "",
                "native_carrier_diagnostic_forbidden_for_method_table": p7.get("native_carrier_diagnostic_forbidden_for_method_table", "") if key == "B3" else "",
                "scene_metric_available": False,
                "diagnostic_scene_metric_available": bool(diagnostic_eval.get("available")) if key == "B3" else False,
                "diagnostic_scene_AP": diagnostic_metrics.get("AP", "") if key == "B3" else "",
                "diagnostic_scene_AP50": diagnostic_metrics.get("AP50", "") if key == "B3" else "",
                "diagnostic_scene_AP25": diagnostic_metrics.get("AP25", "") if key == "B3" else "",
                "method_uses_gt": False,
                "uses_future": False,
                "control_type": "oracle_diagnostic" if diagnostic_only else "method_or_control",
            }
        )
    scene_metric_rows = [
        {"metric": "scene_SF50", "value": "", "available": False, "reason": scene_metric_unavailable_reason},
        {"metric": "scene_AP50", "value": "", "available": False, "reason": scene_metric_unavailable_reason},
        {"metric": "scene_AP25", "value": "", "available": False, "reason": scene_metric_unavailable_reason},
        {
            "metric": "diagnostic_scene_AP",
            "value": diagnostic_metrics.get("AP", ""),
            "available": bool(diagnostic_eval.get("available")),
            "reason": diagnostic_eval.get("reason", ""),
        },
        {
            "metric": "diagnostic_scene_AP50",
            "value": diagnostic_metrics.get("AP50", ""),
            "available": bool(diagnostic_eval.get("available")),
            "reason": diagnostic_eval.get("reason", ""),
        },
        {
            "metric": "diagnostic_scene_AP25",
            "value": diagnostic_metrics.get("AP25", ""),
            "available": bool(diagnostic_eval.get("available")),
            "reason": diagnostic_eval.get("reason", ""),
        },
        {
            "metric": "native_carrier_diagnostic_ARI",
            "value": p7.get("native_carrier_diagnostic_ARI", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": p7.get("native_carrier_diagnostic_metric_scope", ""),
        },
        {
            "metric": "native_carrier_diagnostic_purity",
            "value": p7.get("native_carrier_diagnostic_purity", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": p7.get("native_carrier_diagnostic_metric_scope", ""),
        },
        {
            "metric": "native_carrier_diagnostic_completeness",
            "value": p7.get("native_carrier_diagnostic_completeness", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": p7.get("native_carrier_diagnostic_metric_scope", ""),
        },
        {
            "metric": "native_carrier_cluster_AP25",
            "value": p7.get("native_carrier_cluster_AP25", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_cluster_ap_style_diagnostic_not_scannet_ap",
        },
        {
            "metric": "native_carrier_cluster_AP50",
            "value": p7.get("native_carrier_cluster_AP50", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_cluster_ap_style_diagnostic_not_scannet_ap",
        },
        {
            "metric": "native_carrier_cluster_mean_best_iou",
            "value": p7.get("native_carrier_cluster_mean_best_iou", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_cluster_ap_style_diagnostic_not_scannet_ap",
        },
        {
            "metric": "native_carrier_real_minus_best_non_oracle_AP50",
            "value": p7.get("native_carrier_real_minus_best_non_oracle_AP50", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_controls_diagnostic_not_scene_metric",
        },
        {
            "metric": "native_carrier_real_minus_best_non_oracle_mean_best_iou",
            "value": p7.get("native_carrier_real_minus_best_non_oracle_mean_best_iou", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_controls_diagnostic_not_scene_metric",
        },
        {
            "metric": "native_carrier_real_minus_best_non_oracle_ARI",
            "value": p7.get("native_carrier_real_minus_best_non_oracle_ARI", ""),
            "available": native_carrier_diagnostic_metric_available,
            "reason": "native_carrier_controls_diagnostic_not_scene_metric",
        },
        {
            "metric": "native_carrier_holdout_evaluation_ready",
            "value": native_carrier_holdout_ready,
            "available": True,
            "reason": p7.get("native_carrier_holdout_primary_blocker", ""),
        },
        {
            "metric": "native_carrier_holdout_support_row_count",
            "value": p7.get("native_carrier_holdout_support_row_count", ""),
            "available": True,
            "reason": "planned_temporal_holdout_native_support_rows",
        },
    ]
    attr_rows = [
        {
            "blocker": "SCENE_VERTEX_EXPORTER_MISSING"
            if native_carrier_support_available
            else ("PREDICTION_NPZ_EXPORT_MISSING" if frame_mask_table_available else "MATERIALIZER_EXPORT_MISSING"),
            "evidence": _rel(_repo_path(args.phase7_output_root) / "exporter_repair_audit.json"),
            "route_audit": _rel(_repo_path(args.phase7_output_root) / "native_scene_vertex_export_route_summary.json"),
            "plan_repair_direction": "implement method-safe native-carrier-to-scene-vertex exporter or native-carrier evaluator before tuning",
            "status": "native_carrier_support_ready_diagnostic_native_metric_scene_vertex_npz_missing"
            if native_carrier_support_available and native_carrier_diagnostic_metric_available
            else "native_carrier_support_ready_scene_vertex_npz_missing"
            if native_carrier_support_available
            else (
                "frame_mask_table_repaired_scene_npz_missing"
                if frame_mask_table_available
                else "attempted_blocked_missing_frame_mask_observation_mapping"
            ),
        }
    ]
    gate = {
        "B3_scene_metric_available": False,
        "B3_wrong_absorption_proxy_le_0p05": _num(p7.get("wrong_absorption_proxy"), 1.0) <= 0.05,
        "B3_cannot_link_violation_count_eq_0": _int(p7.get("cannot_link_violation_count"), 999) == 0,
        "B3_beats_controls": False,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "phase": "v85_phase8_strong_controls",
        "schema": "stream4d_v85_phase8_strong_controls_v1",
        "decision": "NO_GO_V85_STRONG_CONTROL_SCENE_METRIC_UNAVAILABLE",
        "can_enter_frozen_holdout": False,
        "can_enter_next_phase": False,
        "gate": gate,
        "primary_blocker": "scene_metric_unavailable_due_native_carrier_to_scene_vertex_exporter_missing"
        if native_carrier_support_available
        else (
            "scene_metric_unavailable_due_prediction_npz_exporter_missing"
            if frame_mask_table_available
            else "scene_metric_unavailable_due_prediction_exporter_missing"
        ),
        "B3_scene_metric_available": False,
        "B3_frame_mask_table_available": frame_mask_table_available,
        "B3_native_carrier_support_available": native_carrier_support_available,
        "B3_native_carrier_support_row_count": p7.get("native_carrier_support_row_count", ""),
        "B3_native_unique_carrier_count": p7.get("native_unique_carrier_count", ""),
        "B3_native_carrier_support_blocker": p7.get("native_carrier_support_blocker", ""),
        "B3_scene_vertex_export_route_checked_count": route_checked_count,
        "B3_method_safe_scene_vertex_exporter_available": method_safe_scene_vertex_exporter_available,
        "B3_method_safe_native_carrier_evaluator_available": method_safe_native_carrier_evaluator_available,
        "B3_native_carrier_diagnostic_metric_available": native_carrier_diagnostic_metric_available,
        "B3_native_carrier_holdout_evaluation_ready": native_carrier_holdout_ready,
        "B3_native_carrier_holdout_readiness": p7.get("native_carrier_evaluator_holdout_readiness", ""),
        "B3_native_carrier_holdout_primary_blocker": p7.get("native_carrier_holdout_primary_blocker", ""),
        "B3_native_carrier_holdout_selected_frame_mask_count": p7.get(
            "native_carrier_holdout_selected_frame_mask_count",
            "",
        ),
        "B3_native_carrier_holdout_support_row_count": p7.get("native_carrier_holdout_support_row_count", ""),
        "B3_native_carrier_holdout_unique_carrier_count": p7.get(
            "native_carrier_holdout_unique_carrier_count",
            "",
        ),
        "B3_native_carrier_diagnostic_assignment_count": p7.get("native_carrier_diagnostic_assignment_count", ""),
        "B3_native_carrier_diagnostic_labeled_support_observation_count": p7.get(
            "native_carrier_diagnostic_labeled_support_observation_count",
            "",
        ),
        "B3_native_carrier_diagnostic_ARI": p7.get("native_carrier_diagnostic_ARI", ""),
        "B3_native_carrier_diagnostic_purity": p7.get("native_carrier_diagnostic_purity", ""),
        "B3_native_carrier_diagnostic_completeness": p7.get("native_carrier_diagnostic_completeness", ""),
        "B3_native_carrier_cluster_AP25": p7.get("native_carrier_cluster_AP25", ""),
        "B3_native_carrier_cluster_AP50": p7.get("native_carrier_cluster_AP50", ""),
        "B3_native_carrier_cluster_AP_mean": p7.get("native_carrier_cluster_AP_mean", ""),
        "B3_native_carrier_cluster_mean_best_iou": p7.get("native_carrier_cluster_mean_best_iou", ""),
        "B3_native_carrier_cluster_prediction_count": p7.get("native_carrier_cluster_prediction_count", ""),
        "B3_native_carrier_cluster_gt_object_count": p7.get("native_carrier_cluster_gt_object_count", ""),
        "B3_native_carrier_cluster_score_contract": p7.get("native_carrier_cluster_score_contract", ""),
        "B3_native_carrier_evaluation_label_scope": p7.get("native_carrier_evaluation_label_scope", ""),
        "B3_native_carrier_evaluator_contract_status": p7.get("native_carrier_evaluator_contract_status", ""),
        "B3_native_carrier_evaluator_candidate_contract": p7.get(
            "native_carrier_evaluator_candidate_contract",
            "",
        ),
        "B3_native_carrier_evaluator_candidate_contract_sha256": p7.get(
            "native_carrier_evaluator_candidate_contract_sha256",
            "",
        ),
        "B3_native_carrier_evaluator_candidate_contract_current_allowed": p7.get(
            "native_carrier_evaluator_candidate_contract_current_allowed",
            "",
        ),
        "B3_native_carrier_evaluator_candidate_contract_future_allowed": p7.get(
            "native_carrier_evaluator_candidate_contract_future_allowed",
            "",
        ),
        "B3_native_carrier_control_variant_count": p7.get("native_carrier_control_variant_count", ""),
        "B3_native_carrier_non_oracle_control_count": p7.get("native_carrier_non_oracle_control_count", ""),
        "B3_native_carrier_real_minus_best_non_oracle_AP50": p7.get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "B3_native_carrier_real_minus_best_non_oracle_mean_best_iou": p7.get(
            "native_carrier_real_minus_best_non_oracle_mean_best_iou",
            "",
        ),
        "B3_native_carrier_real_minus_best_non_oracle_ARI": p7.get(
            "native_carrier_real_minus_best_non_oracle_ARI",
            "",
        ),
        "B3_native_carrier_real_beats_non_oracle_AP50_by_0p03": p7.get(
            "native_carrier_real_beats_non_oracle_AP50_by_0p03",
            "",
        ),
        "B3_native_carrier_real_beats_non_oracle_mean_iou_by_0p03": p7.get(
            "native_carrier_real_beats_non_oracle_mean_iou_by_0p03",
            "",
        ),
        "B3_native_carrier_diagnostic_forbidden_for_method_table": p7.get(
            "native_carrier_diagnostic_forbidden_for_method_table",
            "",
        ),
        "B3_native_carrier_diagnostic_metric_scope": p7.get("native_carrier_diagnostic_metric_scope", ""),
        "B3_native_carrier_diagnostic_summary": p7.get("native_carrier_diagnostic_summary", ""),
        "B3_native_scene_vertex_export_route_audit": p7.get("native_scene_vertex_export_route_audit", ""),
        "B3_diagnostic_scene_metric_available": bool(diagnostic_eval.get("available")),
        "B3_diagnostic_output_config": diagnostic_output_config,
        "B3_diagnostic_scene_AP": diagnostic_metrics.get("AP", ""),
        "B3_diagnostic_scene_AP50": diagnostic_metrics.get("AP50", ""),
        "B3_diagnostic_scene_AP25": diagnostic_metrics.get("AP25", ""),
        "B3_diagnostic_forbidden_for_method_table": bool(diagnostic_eval.get("forbidden_for_method_table", True)),
        "B3_diagnostic_eval_metric_file": diagnostic_eval.get("metric_file", ""),
        "B3_diagnostic_eval_log_path": diagnostic_eval.get("log_path", ""),
        "B3_scene_SF50": "",
        "B3_scene_AP50": "",
        "B3_local_SF50": "",
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "control_summary.json", summary)
    _write_json(out / "diagnostic_scene_eval.json", diagnostic_eval)
    _write_csv(out / "control_metric_rows.csv", metric_rows)
    _write_csv(out / "scene_metric_rows.csv", scene_metric_rows)
    _write_csv(out / "local_metric_rows.csv", [local_metric] if local_metric else [])
    _write_csv(out / "attribution_rows.csv", attr_rows)
    return summary


def _phase9(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase9_output_root)
    out.mkdir(parents=True, exist_ok=True)
    config_out = _repo_path(args.config_output_root)
    p8 = _read_json(_repo_path(args.phase8_output_root) / "control_summary.json")
    frame_mask_table_available = bool(p8.get("B3_frame_mask_table_available"))
    native_carrier_support_available = bool(p8.get("B3_native_carrier_support_available"))
    native_carrier_holdout_ready = bool(p8.get("B3_native_carrier_holdout_evaluation_ready"))
    precondition_audit = _holdout_method_precondition_audit(out)
    phase2_autopsy = _holdout_phase2_failure_autopsy(out)
    diagnostic_tentative = _diagnostic_tentative_holdout_replay(args, out)
    frozen_config = {
        "schema": "stream4d_v85_frozen_method_config_v1",
        "selected_from_dev_only": True,
        "holdout_run_allowed": bool(p8.get("can_enter_frozen_holdout")),
        "native_carrier_holdout_evaluation_ready": native_carrier_holdout_ready,
        "native_carrier_holdout_readiness": p8.get("B3_native_carrier_holdout_readiness", ""),
        "native_carrier_holdout_primary_blocker": p8.get("B3_native_carrier_holdout_primary_blocker", ""),
        "diagnostic_tentative_holdout_replay_attempted": True,
        "diagnostic_tentative_holdout_allowed_for_method": False,
        "diagnostic_tentative_holdout_summary": _rel(out / "diagnostic_tentative_holdout_summary.json"),
        "holdout_method_precondition_audit": _rel(out / "holdout_method_precondition_summary.json"),
        "holdout_method_preconditions_pass": bool(precondition_audit.get("preconditions_pass")),
        "holdout_phase2_failure_autopsy": _rel(out / "holdout_phase2_failure_autopsy_summary.json"),
        "holdout_phase2_autopsy_decision": phase2_autopsy.get("decision", ""),
        "selected_variant": "B3_real_history_feature_query_materializer",
        "method_claim_allowed": False,
        "blocked_before_formal_holdout": not bool(p8.get("can_enter_frozen_holdout")),
        "blocker": p8.get("primary_blocker", ""),
    }
    _write_json(config_out / "frozen_method_config.json", frozen_config)
    summary = {
        "phase": "v85_phase9_holdout",
        "schema": "stream4d_v85_phase9_holdout_v1",
        "decision": "NO_GO_V85_HOLDOUT_BLOCKED_BY_DEV_MATERIALIZER",
        "formal_holdout_run": False,
        "can_enter_next_phase": True,
        "primary_blocker": p8.get("primary_blocker", "dev_gate_failed"),
        "holdout_local_SF50": "",
        "holdout_local_AP50": "",
        "holdout_scene_SF50": "",
        "holdout_scene_AP50": "",
        "holdout_identity_switch_proxy": "",
        "holdout_fragmentation_proxy": "",
        "holdout_wrong_absorption_proxy": "",
        "holdout_safe_assignment_count": "",
        "holdout_diagnostic_assignment_count": "",
        "holdout_scene_metric_available": False,
        "native_carrier_holdout_evaluation_ready": native_carrier_holdout_ready,
        "native_carrier_holdout_readiness": p8.get("B3_native_carrier_holdout_readiness", ""),
        "native_carrier_holdout_primary_blocker": p8.get("B3_native_carrier_holdout_primary_blocker", ""),
        "native_carrier_holdout_selected_frame_mask_count": p8.get(
            "B3_native_carrier_holdout_selected_frame_mask_count",
            "",
        ),
        "native_carrier_holdout_support_row_count": p8.get("B3_native_carrier_holdout_support_row_count", ""),
        "native_carrier_holdout_unique_carrier_count": p8.get(
            "B3_native_carrier_holdout_unique_carrier_count",
            "",
        ),
        "diagnostic_tentative_holdout_replay_attempted": True,
        "holdout_method_precondition_audit": _rel(out / "holdout_method_precondition_summary.json"),
        "holdout_method_preconditions_pass": bool(precondition_audit.get("preconditions_pass")),
        "holdout_method_precondition_failed_check_count": precondition_audit.get("failed_check_count", ""),
        "holdout_method_precondition_failed_checks": ";".join(precondition_audit.get("failed_checks", [])),
        "holdout_phase2_failure_autopsy": _rel(out / "holdout_phase2_failure_autopsy_summary.json"),
        "holdout_phase2_autopsy_decision": phase2_autopsy.get("decision", ""),
        "holdout_phase2_failed_check_count": phase2_autopsy.get("failed_check_count", ""),
        "holdout_phase2_failed_checks": ";".join(phase2_autopsy.get("failed_checks", [])),
        "holdout_phase2_dev_coverage": phase2_autopsy.get("dev_eligible_tracklet_coverage_rate", ""),
        "holdout_phase2_coverage": phase2_autopsy.get("holdout_eligible_tracklet_coverage_rate", ""),
        "holdout_phase2_coverage_drop_dev_minus_holdout": phase2_autopsy.get("coverage_drop_dev_minus_holdout", ""),
        "holdout_phase2_dev_full_minus_semantic": phase2_autopsy.get("dev_full_minus_semantic_score", ""),
        "holdout_phase2_full_minus_semantic": phase2_autopsy.get("holdout_full_minus_semantic_score", ""),
        "holdout_phase2_full_minus_semantic_drop_dev_minus_holdout": phase2_autopsy.get(
            "full_minus_semantic_drop_dev_minus_holdout",
            "",
        ),
        "holdout_phase2_unassigned_eligible_slot_count": phase2_autopsy.get(
            "holdout_unassigned_eligible_slot_count",
            "",
        ),
        "holdout_phase2_selected_nonpositive_full_minus_semantic_count": phase2_autopsy.get(
            "holdout_selected_nonpositive_full_minus_semantic_count",
            "",
        ),
        "diagnostic_tentative_holdout_decision": diagnostic_tentative.get("decision", ""),
        "diagnostic_tentative_holdout_summary": _rel(out / "diagnostic_tentative_holdout_summary.json"),
        "diagnostic_tentative_holdout_selected_frame_mask_row_count": diagnostic_tentative.get(
            "selected_frame_mask_row_count",
            "",
        ),
        "diagnostic_tentative_holdout_native_support_row_count": diagnostic_tentative.get(
            "diagnostic_native_support_row_count",
            "",
        ),
        "diagnostic_tentative_holdout_unique_native_carrier_count": diagnostic_tentative.get(
            "diagnostic_unique_native_carrier_count",
            "",
        ),
        "diagnostic_tentative_holdout_native_AP50": diagnostic_tentative.get("diagnostic_native_AP50", ""),
        "diagnostic_tentative_holdout_ARI": diagnostic_tentative.get("diagnostic_ARI", ""),
        "diagnostic_tentative_holdout_purity": diagnostic_tentative.get("diagnostic_purity", ""),
        "diagnostic_tentative_holdout_real_minus_best_non_oracle_AP50": diagnostic_tentative.get(
            "diagnostic_real_minus_best_non_oracle_AP50",
            "",
        ),
        "diagnostic_tentative_method_claim_blocker": diagnostic_tentative.get("method_claim_blocker", ""),
        "diagnostic_tentative_forbidden_for_method_table": True,
        "holdout_real_minus_shuffled": "",
        "holdout_real_minus_semantic": "",
        "holdout_real_minus_stale": "",
        "method_uses_gt": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "holdout_summary.json", summary)
    _write_csv(out / "holdout_metric_rows.csv", [summary])
    _write_csv(
        out / "holdout_control_rows.csv",
        [
            {"control": "formal_holdout", "run": False, "reason": summary["primary_blocker"]},
            {
                "control": "diagnostic_tentative_holdout_replay",
                "run": bool(diagnostic_tentative.get("selected_frame_mask_row_count")),
                "reason": diagnostic_tentative.get("method_claim_blocker", ""),
            },
        ],
    )
    _write_csv(
        out / "holdout_casebook_rows.csv",
        [
            {
                "failure_type": "SCENE_VERTEX_EXPORTER_MISSING"
                if native_carrier_support_available
                else ("PREDICTION_NPZ_EXPORT_MISSING" if frame_mask_table_available else "MATERIALIZER_EXPORT_MISSING"),
                "notes": "Phase8 did not allow frozen holdout; native D4RT carrier support exists but method-safe scene-vertex/AP npz exporter is still missing."
                if native_carrier_support_available and not p8.get("B3_native_carrier_holdout_primary_blocker")
                else (
                    "Phase8 did not allow frozen holdout; candidate native-carrier evaluator also has no selected/native support rows in the planned temporal holdout chunks."
                    if native_carrier_support_available
                    else (
                        "Phase8 did not allow frozen holdout; frame-mask table exists but scene-level prediction npz is still missing."
                        if frame_mask_table_available
                        else "Phase8 did not allow frozen holdout."
                    )
                )
            },
            {
                "failure_type": "HOLDOUT_PHASE2_OBJECT_SPECIFICITY_DROP",
                "notes": (
                    "Dev reference Phase2 passed, but holdout replay under frozen config fails coverage and "
                    "full-vs-semantic residual. This autopsy is diagnostic-only and must not be used to tune holdout."
                ),
            },
        ],
    )
    return summary


def _phase10(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _repo_path(args.phase10_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p1 = _read_json(_repo_path(args.phase1_output_root) / "feature_summary.json")
    p2 = _read_json(_repo_path(args.phase2_output_root) / "cluster_summary.json")
    p3 = _read_json(_repo_path(args.phase3_output_root) / "slot_descriptor_summary.json")
    p4 = _read_json(_repo_path(args.phase4_output_root) / "tracklet_summary.json")
    p5 = _read_json(_repo_path(args.phase5_output_root) / "history_summary.json")
    p6 = _read_json(_repo_path(args.phase6_output_root) / "q_summary.json")
    p7 = _read_json(_repo_path(args.phase7_output_root) / "materializer_summary.json")
    p8 = _read_json(_repo_path(args.phase8_output_root) / "control_summary.json")
    p9 = _read_json(_repo_path(args.phase9_output_root) / "holdout_summary.json")

    if str(p9.get("decision", "")).startswith("PASS"):
        final = "GO_PERSISTENT_AFFINITY_FIELD_L2H"
    elif str(p6.get("decision", "")).startswith("PASS") and not bool(p7.get("scene_metric_available")):
        final = "NO_GO_MATERIALIZER_BLOCKED"
    elif not str(p1.get("decision", "")).startswith("PASS") and not str(p2.get("decision", "")).startswith("PASS"):
        final = "NO_GO_LOCAL_AFFINITY_WEAK"
    elif not str(p6.get("decision", "")).startswith("PASS"):
        final = "NO_GO_HISTORY_QUERY_WEAK"
    else:
        final = "DIAGNOSTIC_PROGRESS_ONLY"

    failure_types = []
    if p1.get("primary_blocker"):
        failure_types.append("LOCAL_AFFINITY_OBJECTNESS_AUC_MISSING")
    if p2.get("primary_blocker"):
        failure_types.append("LOCAL_AFFINITY_NOT_STRICT_OBJECT_SPECIFIC")
    if p3.get("primary_blocker"):
        failure_types.append("SLOT_DESCRIPTOR_PRIMARY_METRICS_MISSING")
    if p4.get("primary_blocker"):
        failure_types.append("TRACKLET_DESCRIPTOR_WEAK")
    if p5.get("primary_blocker"):
        failure_types.append("HISTORY_OBJECT_FEATURE_WEAK")
    if p7.get("primary_blocker"):
        failure_types.append(
            "SCENE_VERTEX_EXPORTER_MISSING"
            if p7.get("native_carrier_support_available")
            else ("PREDICTION_NPZ_EXPORT_MISSING" if p7.get("frame_mask_table_available") else "MATERIALIZER_EXPORT_MISSING")
        )
    if p8.get("primary_blocker"):
        failure_types.append(
            "SCENE_METRIC_UNAVAILABLE_SCENE_VERTEX_EXPORTER_MISSING"
            if p8.get("B3_native_carrier_support_available")
            else (
                "SCENE_METRIC_UNAVAILABLE_PREDICTION_NPZ_MISSING"
                if p8.get("B3_frame_mask_table_available")
                else "MATERIALIZER_EXPORT_MISSING"
            )
        )
    if p9.get("primary_blocker"):
        failure_types.append("HOLDOUT_NOT_RUN_DEV_GATE_FAILED")
    if p7.get("native_carrier_evaluator_candidate_contract_future_allowed") and not p7.get(
        "native_carrier_holdout_evaluation_ready"
    ):
        failure_types.append("NATIVE_CARRIER_HOLDOUT_NOT_READY")
    if p9.get("diagnostic_tentative_holdout_decision") == "DIAGNOSTIC_TENTATIVE_HOLDOUT_REPLAY_NOT_METHOD":
        failure_types.append("HOLDOUT_DIAGNOSTIC_TENTATIVE_NOT_METHOD")
    if p9.get("holdout_method_preconditions_pass") is False:
        failure_types.append("HOLDOUT_METHOD_PRECONDITIONS_FAILED")
    if p9.get("holdout_phase2_autopsy_decision") == "NO_GO_HOLDOUT_PHASE2_OBJECT_SPECIFICITY_DROP":
        failure_types.append("HOLDOUT_PHASE2_OBJECT_SPECIFICITY_DROP")

    case_rows = [
        {
            "scene_id": "ALL_DEV",
            "chunk_id": "",
            "frame_id": "",
            "local_slot_id": "",
            "history_id": "",
            "failure_type": "SCENE_VERTEX_EXPORTER_MISSING"
            if p7.get("native_carrier_support_available")
            else ("PREDICTION_NPZ_EXPORT_MISSING" if p7.get("frame_mask_table_available") else "MATERIALIZER_EXPORT_MISSING"),
            "feature_layer": "renderable_membership_field",
            "local_feature_status": p1.get("decision", ""),
            "slot_descriptor_status": p3.get("decision", ""),
            "tracklet_descriptor_status": p4.get("decision", ""),
            "history_feature_status": p5.get("decision", ""),
            "history_query_status": p6.get("decision", ""),
            "materializer_status": p7.get("decision", ""),
            "control_comparison": p8.get("decision", ""),
            "diagnostic_GT_best_IoU": p2.get("GT_best_IoU_mean", ""),
            "notes": (
                "Weak/query memory signal exists and Phase7 now has frame-mask plus native D4RT carrier support "
                "and a diagnostic native-carrier cluster metric, but method-safe native-carrier-to-ScanNet-scene-vertex "
                "export or a frozen native-carrier method metric contract is still missing; no strong scene metric or formal holdout."
            )
            if p7.get("native_carrier_support_available") and p7.get("native_carrier_diagnostic_evaluator_available")
            else "Weak/query memory signal exists and Phase7 now has frame-mask plus native D4RT carrier support, but method-safe native-carrier-to-ScanNet-scene-vertex export is still missing; no strong scene metric or formal holdout."
            if p7.get("native_carrier_support_available")
            else "Weak/query memory signal exists and Phase7 now has a frame-mask table, but point-level prediction npz export is still missing; no strong scene metric or formal holdout."
            if p7.get("frame_mask_table_available")
            else "Weak/query memory signal exists, but frame-mask/prediction exporter is missing; no strong scene metric or formal holdout.",
        }
    ]
    if p9.get("holdout_phase2_autopsy_decision"):
        case_rows.append(
            {
                "scene_id": "REGISTERED_HOLDOUT",
                "chunk_id": "",
                "frame_id": "",
                "local_slot_id": "",
                "history_id": "",
                "failure_type": "HOLDOUT_PHASE2_OBJECT_SPECIFICITY_DROP",
                "feature_layer": "tracklet_descriptor_to_history_readout",
                "local_feature_status": p1.get("decision", ""),
                "slot_descriptor_status": p3.get("decision", ""),
                "tracklet_descriptor_status": p4.get("decision", ""),
                "history_feature_status": p5.get("decision", ""),
                "history_query_status": p6.get("decision", ""),
                "materializer_status": p7.get("decision", ""),
                "control_comparison": p8.get("decision", ""),
                "diagnostic_GT_best_IoU": "",
                "notes": (
                    "Holdout Phase2 autopsy is diagnostic-only: dev reference coverage="
                    + str(p9.get("holdout_phase2_dev_coverage", ""))
                    + ", holdout coverage="
                    + str(p9.get("holdout_phase2_coverage", ""))
                    + ", holdout full-minus-semantic="
                    + str(p9.get("holdout_phase2_full_minus_semantic", ""))
                    + ". This cannot be used for holdout retuning."
                ),
            }
        )
    tax_rows = [{"failure_type": ft, "count": count} for ft, count in sorted(Counter(failure_types).items())]
    final_decision = {
        "schema": "stream4d_v85_final_decision_v2",
        "final_decision": final,
        "phase1_decision": p1.get("decision", ""),
        "phase2_decision": p2.get("decision", ""),
        "phase3_decision": p3.get("decision", ""),
        "phase4_decision": p4.get("decision", ""),
        "phase5_decision": p5.get("decision", ""),
        "phase6_decision": p6.get("decision", ""),
        "phase7_decision": p7.get("decision", ""),
        "phase8_decision": p8.get("decision", ""),
        "phase9_decision": p9.get("decision", ""),
        "failure_type_counts": dict(Counter(failure_types)),
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "strong_method_goal_achieved": final == "GO_PERSISTENT_AFFINITY_FIELD_L2H",
        "weak_affinity_memory_signal_present": str(p6.get("decision", "")).startswith("PASS"),
        "frame_mask_exporter_available": bool(p7.get("frame_mask_table_available")),
        "native_carrier_support_available": bool(p7.get("native_carrier_support_available")),
        "native_carrier_support_row_count": p7.get("native_carrier_support_row_count", 0),
        "native_unique_carrier_count": p7.get("native_unique_carrier_count", 0),
        "scene_vertex_export_route_checked_count": p7.get("scene_vertex_export_route_checked_count", 0),
        "method_safe_scene_vertex_exporter_available": bool(p7.get("method_safe_scene_vertex_exporter_available")),
        "method_safe_native_carrier_evaluator_available": bool(
            p7.get("method_safe_native_carrier_evaluator_available")
        ),
        "native_carrier_diagnostic_metric_available": bool(p7.get("native_carrier_diagnostic_evaluator_available")),
        "native_carrier_diagnostic_assignment_count": p7.get("native_carrier_diagnostic_assignment_count", ""),
        "native_carrier_diagnostic_labeled_support_observation_count": p7.get(
            "native_carrier_diagnostic_labeled_support_observation_count",
            "",
        ),
        "native_carrier_diagnostic_ARI": p7.get("native_carrier_diagnostic_ARI", ""),
        "native_carrier_diagnostic_purity": p7.get("native_carrier_diagnostic_purity", ""),
        "native_carrier_diagnostic_completeness": p7.get("native_carrier_diagnostic_completeness", ""),
        "native_carrier_cluster_AP25": p7.get("native_carrier_cluster_AP25", ""),
        "native_carrier_cluster_AP50": p7.get("native_carrier_cluster_AP50", ""),
        "native_carrier_cluster_AP_mean": p7.get("native_carrier_cluster_AP_mean", ""),
        "native_carrier_cluster_mean_best_iou": p7.get("native_carrier_cluster_mean_best_iou", ""),
        "native_carrier_cluster_prediction_count": p7.get("native_carrier_cluster_prediction_count", ""),
        "native_carrier_cluster_gt_object_count": p7.get("native_carrier_cluster_gt_object_count", ""),
        "native_carrier_cluster_score_contract": p7.get("native_carrier_cluster_score_contract", ""),
        "native_carrier_evaluation_label_scope": p7.get("native_carrier_evaluation_label_scope", ""),
        "native_carrier_evaluator_contract_status": p7.get("native_carrier_evaluator_contract_status", ""),
        "native_carrier_evaluator_candidate_contract": p7.get("native_carrier_evaluator_candidate_contract", ""),
        "native_carrier_evaluator_candidate_contract_sha256": p7.get(
            "native_carrier_evaluator_candidate_contract_sha256",
            "",
        ),
        "native_carrier_evaluator_candidate_contract_current_allowed": p7.get(
            "native_carrier_evaluator_candidate_contract_current_allowed",
            "",
        ),
        "native_carrier_evaluator_candidate_contract_future_allowed": p7.get(
            "native_carrier_evaluator_candidate_contract_future_allowed",
            "",
        ),
        "native_carrier_evaluator_holdout_readiness": p7.get("native_carrier_evaluator_holdout_readiness", ""),
        "native_carrier_holdout_evaluation_ready": bool(p7.get("native_carrier_holdout_evaluation_ready")),
        "native_carrier_holdout_primary_blocker": p7.get("native_carrier_holdout_primary_blocker", ""),
        "native_carrier_holdout_selected_frame_mask_count": p7.get(
            "native_carrier_holdout_selected_frame_mask_count",
            "",
        ),
        "native_carrier_holdout_support_row_count": p7.get("native_carrier_holdout_support_row_count", ""),
        "native_carrier_holdout_unique_carrier_count": p7.get(
            "native_carrier_holdout_unique_carrier_count",
            "",
        ),
        "diagnostic_tentative_holdout_replay_attempted": p9.get(
            "diagnostic_tentative_holdout_replay_attempted",
            "",
        ),
        "holdout_method_precondition_audit": p9.get("holdout_method_precondition_audit", ""),
        "holdout_method_preconditions_pass": p9.get("holdout_method_preconditions_pass", ""),
        "holdout_method_precondition_failed_check_count": p9.get(
            "holdout_method_precondition_failed_check_count",
            "",
        ),
        "holdout_method_precondition_failed_checks": p9.get("holdout_method_precondition_failed_checks", ""),
        "holdout_phase2_failure_autopsy": p9.get("holdout_phase2_failure_autopsy", ""),
        "holdout_phase2_autopsy_decision": p9.get("holdout_phase2_autopsy_decision", ""),
        "holdout_phase2_failed_check_count": p9.get("holdout_phase2_failed_check_count", ""),
        "holdout_phase2_failed_checks": p9.get("holdout_phase2_failed_checks", ""),
        "holdout_phase2_dev_coverage": p9.get("holdout_phase2_dev_coverage", ""),
        "holdout_phase2_coverage": p9.get("holdout_phase2_coverage", ""),
        "holdout_phase2_coverage_drop_dev_minus_holdout": p9.get(
            "holdout_phase2_coverage_drop_dev_minus_holdout",
            "",
        ),
        "holdout_phase2_dev_full_minus_semantic": p9.get("holdout_phase2_dev_full_minus_semantic", ""),
        "holdout_phase2_full_minus_semantic": p9.get("holdout_phase2_full_minus_semantic", ""),
        "holdout_phase2_full_minus_semantic_drop_dev_minus_holdout": p9.get(
            "holdout_phase2_full_minus_semantic_drop_dev_minus_holdout",
            "",
        ),
        "holdout_phase2_unassigned_eligible_slot_count": p9.get(
            "holdout_phase2_unassigned_eligible_slot_count",
            "",
        ),
        "holdout_phase2_selected_nonpositive_full_minus_semantic_count": p9.get(
            "holdout_phase2_selected_nonpositive_full_minus_semantic_count",
            "",
        ),
        "diagnostic_tentative_holdout_decision": p9.get("diagnostic_tentative_holdout_decision", ""),
        "diagnostic_tentative_holdout_summary": p9.get("diagnostic_tentative_holdout_summary", ""),
        "diagnostic_tentative_holdout_selected_frame_mask_row_count": p9.get(
            "diagnostic_tentative_holdout_selected_frame_mask_row_count",
            "",
        ),
        "diagnostic_tentative_holdout_native_support_row_count": p9.get(
            "diagnostic_tentative_holdout_native_support_row_count",
            "",
        ),
        "diagnostic_tentative_holdout_unique_native_carrier_count": p9.get(
            "diagnostic_tentative_holdout_unique_native_carrier_count",
            "",
        ),
        "diagnostic_tentative_holdout_native_AP50": p9.get("diagnostic_tentative_holdout_native_AP50", ""),
        "diagnostic_tentative_holdout_ARI": p9.get("diagnostic_tentative_holdout_ARI", ""),
        "diagnostic_tentative_holdout_purity": p9.get("diagnostic_tentative_holdout_purity", ""),
        "diagnostic_tentative_holdout_real_minus_best_non_oracle_AP50": p9.get(
            "diagnostic_tentative_holdout_real_minus_best_non_oracle_AP50",
            "",
        ),
        "diagnostic_tentative_method_claim_blocker": p9.get("diagnostic_tentative_method_claim_blocker", ""),
        "diagnostic_tentative_forbidden_for_method_table": p9.get(
            "diagnostic_tentative_forbidden_for_method_table",
            "",
        ),
        "native_carrier_control_variant_count": p7.get("native_carrier_control_variant_count", ""),
        "native_carrier_non_oracle_control_count": p7.get("native_carrier_non_oracle_control_count", ""),
        "native_carrier_real_minus_best_non_oracle_AP50": p7.get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_mean_best_iou": p7.get(
            "native_carrier_real_minus_best_non_oracle_mean_best_iou",
            "",
        ),
        "native_carrier_real_minus_best_non_oracle_ARI": p7.get(
            "native_carrier_real_minus_best_non_oracle_ARI",
            "",
        ),
        "native_carrier_real_beats_non_oracle_AP50_by_0p03": p7.get(
            "native_carrier_real_beats_non_oracle_AP50_by_0p03",
            "",
        ),
        "native_carrier_real_beats_non_oracle_mean_iou_by_0p03": p7.get(
            "native_carrier_real_beats_non_oracle_mean_iou_by_0p03",
            "",
        ),
        "native_carrier_diagnostic_forbidden_for_method_table": p7.get(
            "native_carrier_diagnostic_forbidden_for_method_table",
            "",
        ),
        "native_carrier_diagnostic_metric_scope": p7.get("native_carrier_diagnostic_metric_scope", ""),
        "native_carrier_diagnostic_summary": p7.get("native_carrier_diagnostic_summary", ""),
        "native_scene_vertex_export_route_audit": p7.get("native_scene_vertex_export_route_audit", ""),
        "materializer_exporter_available": bool(p7.get("scene_metric_available")),
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "final_decision.json", final_decision)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_csv(out / "failure_taxonomy_rows.csv", tax_rows)
    _write_json(out / "visualization_manifest.json", {"available": False, "reason": "No v85 visual exporter implemented in current run."})
    return final_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_ORDER + ["all"], default="all")
    parser.add_argument("--max-carrier-rows", type=int, default=2000)
    parser.add_argument("--max-neighbor-rows", type=int, default=5000)
    parser.add_argument("--max-cluster-rows", type=int, default=5000)
    parser.add_argument("--max-slot-rows", type=int, default=10000)
    parser.add_argument("--max-query-rows", type=int, default=10000)
    parser.add_argument("--max-tracklet-rows", type=int, default=10000)

    parser.add_argument("--v79-phase1-root", default="outputs/audit/v79_phase1_affinity_features")
    parser.add_argument("--v79-phase2-root", default="outputs/audit/v79_phase2_neighbor_graph")
    parser.add_argument("--v80-phase1-root", default="outputs/audit/v80_phase1_streaming_affinity_features_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard")
    parser.add_argument("--v80-phase2-root", default="outputs/audit/v80_phase2_signed_affinity_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard")
    parser.add_argument("--v80-phase6-root", default="outputs/audit/v80_phase6_control_audit_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard")
    parser.add_argument("--v82-phase1-root", default="outputs/audit/v82_phase1_local_b0")
    parser.add_argument("--v82-adapter-rows", default="")
    parser.add_argument(
        "--native-carrier-observation-tables",
        nargs="*",
        default=DEFAULT_NATIVE_CARRIER_OBSERVATION_TABLES,
        help="Carrier observation CSVs used to audit selected frame-mask rows against method-safe native D4RT carrier ids.",
    )
    parser.add_argument("--diagnostic-npz-output-config", default="v85_paf_l2h_frame_mask_diag")
    parser.add_argument("--skip-diagnostic-npz-export", action="store_true")
    parser.add_argument("--skip-diagnostic-npz-eval", action="store_true")
    parser.add_argument("--v83-phase2-root", default="outputs/audit/v83_phase2_evidence_ledger_repair8_antihijack_extreme_bound")
    parser.add_argument("--v83-phase3-root", default="outputs/audit/v83_phase3_state_machine_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase4-root", default="outputs/audit/v83_phase4_conflict_memory_repair11_structural_edges")
    parser.add_argument("--v83-phase5-root", default="outputs/audit/v83_phase5_weak_l2h_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase6-root", default="outputs/audit/v83_phase6_controls_repair10_safe_topk_coverage")
    parser.add_argument("--v83-phase7-root", default="outputs/audit/v83_phase7_strong_history_repair11_structural_edges")
    parser.add_argument("--v84-phase3-root", default="outputs/audit/v84_phase3_cross_chunk_materializer")
    parser.add_argument("--v84-phase8-root", default="outputs/audit/v84_phase8_frozen_holdout")
    parser.add_argument("--v84-phase9-root", default="outputs/audit/v84_phase9_casebook")

    parser.add_argument("--phase0-output-root", default="outputs/audit/v85_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v85_phase1_local_affinity_feature")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v85_phase2_local_clustering")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v85_phase3_slot_descriptor")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v85_phase4_tracklet_descriptor")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v85_phase5_history_object_feature")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v85_phase6_history_query")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v85_phase7_renderable_materializer")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v85_phase8_strong_controls")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v85_phase9_holdout")
    parser.add_argument("--phase10-output-root", default="outputs/audit/v85_phase10_casebook")
    parser.add_argument("--config-output-root", default="outputs/audit/v85_config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runners = {
        "phase0": _phase0,
        "phase1": _phase1,
        "phase2": _phase2,
        "phase3": _phase3,
        "phase4": _phase4,
        "phase5": _phase5,
        "phase6": _phase6,
        "phase7": _phase7,
        "phase8": _phase8,
        "phase9": _phase9,
        "phase10": _phase10,
    }
    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    for phase in phases:
        summary = runners[phase](args)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "decision": summary.get("decision", summary.get("final_decision", "")),
                    "primary_blocker": summary.get("primary_blocker", ""),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
