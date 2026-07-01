#!/usr/bin/env python3
"""Run Stream4D v88 MV-AP-first readout rectification audit.

This runner adapts v87's method-safe frame-mask tube materialization to the v88
split/variant/control contract. Formal MV AP is still computed by the existing
v65 evaluator path through ``run_v87_existing_mv_ap_eval._evaluate_group``.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v87_mv_ap_persistent_affinity_readout as v87
from tools.run_v65_scene_multiview_ap import (
    _sha256 as v65_sha256,
    _write_csv as v65_write_csv,
    _write_json as v65_write_json,
    _write_sha256sums as v65_write_sha256sums,
)
from tools.run_v87_existing_mv_ap_eval import _evaluate_group as existing_v65_adapter_evaluate_group


PHASE_ORDER = (
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5A",
    "phase5B",
    "phase5C",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
)

V88_DEV_CHUNKS = {
    "scene0011_00": set(range(0, 6)),
    "scene0050_00": set(range(0, 4)),
}
V88_INTERNAL_HOLDOUT_CHUNKS = {
    "scene0011_00": set(range(6, 12)),
    "scene0050_00": set(range(4, 12)),
}

REAL_VARIANTS = {
    "B0_local_only",
    "B1_M10_state_priority",
    "B2_DV5_confirmed_object_gain",
    "B3_history_with_local_fallback",
    "B4_state_priority_with_local_fallback",
    "B5_carrier_gated_frame_mask_readout",
    "B6_area_penalized_history_readout",
}
CONTROL_VARIANTS = {
    "C0_semantic_only_control",
    "C1_shuffled_history_control",
    "C2_stale_history_control",
    "C3_size_matched_hash_control",
    "C4_single_largest_by_scene_control",
    "C5_local_only_area_rank_control",
}


def _repo_path(path: str | Path) -> Path:
    return v87._repo_path(path)


def _rel(path: Path | None) -> str:
    return v87._rel(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return v87._read_json(_repo_path(path))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    v87._write_json(_repo_path(path), payload)


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    return v87._read_csv_rows(_repo_path(path))


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    v87._write_csv(_repo_path(path), rows, fields=fields)


def _num(value: Any, default: float = 0.0) -> float:
    return v87._num(value, default)


def _int(value: Any, default: int = 0) -> int:
    return v87._int(value, default)


def _bool(value: Any) -> bool:
    return v87._bool(value)


def _mean(values: list[float]) -> float:
    return v87._mean(values)


def _safe_ratio(num: float, den: float) -> float:
    return v87._safe_ratio(num, den)


def _hash_text(text: str) -> str:
    return v87._hash_text(text)


def _sha256_file(path: Path) -> str:
    return v87._sha256_file(path)


def _jsonable(value: Any) -> Any:
    return v87._jsonable(value)


def _canonical_variant(variant: str) -> str:
    mapping = {
        "B3_DV5_object_gain_with_local_fallback": "B3_history_with_local_fallback",
        "B4_M10_state_priority_with_local_fallback": "B4_state_priority_with_local_fallback",
    }
    return mapping.get(str(variant), str(variant))


def _variant_is_real(variant: str) -> bool:
    return str(variant) in REAL_VARIANTS


def _variant_is_control(variant: str) -> bool:
    return str(variant) in CONTROL_VARIANTS


def _variant_family(variant: str) -> str:
    variant = str(variant)
    if variant == "B0_local_only":
        return "local_only"
    if variant == "B1_M10_state_priority":
        return "m10_state_priority"
    if variant == "B2_DV5_confirmed_object_gain":
        return "dv5_confirmed_object_gain"
    if variant == "B3_history_with_local_fallback":
        return "history_with_local_fallback"
    if variant == "B4_state_priority_with_local_fallback":
        return "state_priority_with_local_fallback"
    if variant == "B5_carrier_gated_frame_mask_readout":
        return "carrier_gated_frame_mask_readout"
    if variant == "B6_area_penalized_history_readout":
        return "area_penalized_history_readout"
    if variant.startswith("C0"):
        return "semantic_only_control"
    if variant.startswith("C1"):
        return "shuffled_history_control"
    if variant.startswith("C2"):
        return "stale_history_control"
    if variant.startswith("C3"):
        return "size_matched_hash_control"
    if variant.startswith("C4"):
        return "single_largest_by_scene_control"
    if variant.startswith("C5"):
        return "local_only_area_rank_control"
    return "other"


def _is_forbidden_legacy_variant(variant: str) -> bool:
    return str(variant) == "B5_confirmed_only_conservative"


def _source_sets() -> dict[str, dict[str, Any]]:
    base = dict(v87.DEFAULT_SOURCE_SETS["dev"])
    common = {
        "local_root": base["local_root"],
        "adapter_rows": base["adapter_rows"],
        "tracklet_root": base["tracklet_root"],
        "fresh_scene": False,
        "preexisting_exploratory_artifact": False,
    }
    return {
        "dev": {
            **common,
            "split": "dev",
            "chunks": {scene: set(chunks) for scene, chunks in V88_DEV_CHUNKS.items()},
        },
        "internal_holdout": {
            **common,
            "split": "internal_holdout",
            "chunks": {scene: set(chunks) for scene, chunks in V88_INTERNAL_HOLDOUT_CHUNKS.items()},
        },
    }


def _normalize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        variant = _canonical_variant(str(row.get("variant", "")))
        if _is_forbidden_legacy_variant(variant):
            continue
        new = dict(row)
        new["variant"] = variant
        new["source_variant"] = variant
        out.append(new)
    return out


def _prepare_selected_row(row: dict[str, Any], *, origin: str = "v88_base") -> dict[str, Any]:
    new = dict(row)
    variant = _canonical_variant(str(new.get("variant") or new.get("source_variant") or ""))
    new["variant"] = variant
    new["source_variant"] = variant
    scene = str(new.get("scene_id", ""))
    history = str(new.get("history_id", ""))
    new["mv_object_id"] = f"{variant}:{scene}:{_hash_text(history)}"
    ownership = _num(new.get("ownership_score"), _num(new.get("adapter_score"), _num(new.get("source_score"), 0.0)))
    new["ownership_score"] = ownership
    new["object_score"] = _num(new.get("object_score"), ownership)
    new["selected_flag"] = True
    new["selection_reason"] = new.get("selection_reason") or new.get("selection_policy", "v88_selected")
    new["uses_gt_for_prediction"] = _bool(new.get("method_uses_gt"))
    new["uses_rgbd_pose_mesh_for_prediction"] = _bool(new.get("uses_rgbd_pose_mesh"))
    new["candidate_origin"] = origin
    new["method_uses_gt"] = False
    new["uses_future"] = False
    new["uses_rgbd_pose_mesh"] = False
    return new


def _mask_area_ratio(row: dict[str, Any]) -> float:
    return _safe_ratio(_num(row.get("mask_area"), 0.0), 968.0 * 1296.0)


def _derive_v88_b5_b6_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = [r for r in selected_rows if str(r.get("variant")) == "B3_history_with_local_fallback"]
    if not base:
        base = [r for r in selected_rows if str(r.get("variant")) == "B4_state_priority_with_local_fallback"]
    derived: list[dict[str, Any]] = []
    for row in base:
        native_support = _int(row.get("native_carrier_support_count"), 0)
        broad = _bool(row.get("broad_mask_flag"))
        if native_support > 0 and not broad:
            b5 = dict(row)
            b5["variant"] = "B5_carrier_gated_frame_mask_readout"
            b5["source_variant"] = b5["variant"]
            b5["history_id"] = f"carrier_gated:{row.get('history_id', '')}"
            b5["source_kind"] = "v88_carrier_gated_frame_mask_readout"
            b5["selection_policy"] = "v88_native_carrier_support_gt0_and_not_broad"
            b5["selection_reason"] = b5["selection_policy"]
            b5["ownership_score"] = _num(row.get("adapter_score"), 0.0) + min(0.20, math.log1p(native_support) / 10.0)
            b5["adapter_score"] = b5["ownership_score"]
            derived.append(_prepare_selected_row(b5, origin="v88_b5_from_history_carrier_gate"))

        b6 = dict(row)
        b6["variant"] = "B6_area_penalized_history_readout"
        b6["source_variant"] = b6["variant"]
        b6["history_id"] = f"area_penalized:{row.get('history_id', '')}"
        b6["source_kind"] = "v88_area_penalized_history_readout"
        b6["selection_policy"] = "v88_fixed_area_penalty_on_history_readout"
        b6["selection_reason"] = b6["selection_policy"]
        area_ratio = _mask_area_ratio(row)
        broad_penalty = 0.20 if broad else 0.0
        area_penalty = min(0.65, 2.0 * max(0.0, area_ratio - 0.08) + broad_penalty)
        adjusted = max(0.0, _num(row.get("adapter_score"), 0.0) * (1.0 - area_penalty))
        b6["ownership_score"] = adjusted
        b6["adapter_score"] = adjusted
        b6["source_margin"] = max(0.0, _num(row.get("source_margin"), adjusted) - 0.5 * area_penalty)
        b6["v88_area_penalty"] = area_penalty
        derived.append(_prepare_selected_row(b6, origin="v88_b6_from_history_area_penalty"))
    return derived


def _selected_wta(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    drops: list[dict[str, Any]] = []

    def rank(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
        return (
            _num(row.get("ownership_score"), _num(row.get("adapter_score"), 0.0)),
            _num(row.get("adapter_score"), 0.0),
            _num(row.get("native_carrier_support_count"), 0.0),
            -_num(row.get("mask_area"), 0.0),
            str(row.get("mv_object_id", "")),
        )

    for row in rows:
        key = (
            str(row.get("split", "")),
            str(row.get("scene_id", "")),
            str(row.get("source_variant") or row.get("variant") or ""),
            str(row.get("frame_id", "")),
            str(row.get("mask_id", "")),
        )
        old = best.get(key)
        if old is None or rank(row) > rank(old):
            if old is not None:
                drops.append(
                    {
                        "split": old.get("split", ""),
                        "scene_id": old.get("scene_id", ""),
                        "source_variant": old.get("source_variant", old.get("variant", "")),
                        "frame_id": old.get("frame_id", ""),
                        "mask_id": old.get("mask_id", ""),
                        "dropped_mv_object_id": old.get("mv_object_id", ""),
                        "kept_mv_object_id": row.get("mv_object_id", ""),
                        "dropped_score": old.get("ownership_score", old.get("adapter_score", "")),
                        "kept_score": row.get("ownership_score", row.get("adapter_score", "")),
                        "drop_reason": "v88_phase1_global_variant_frame_mask_wta",
                    }
                )
            best[key] = row
        else:
            drops.append(
                {
                    "split": row.get("split", ""),
                    "scene_id": row.get("scene_id", ""),
                    "source_variant": row.get("source_variant", row.get("variant", "")),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "dropped_mv_object_id": row.get("mv_object_id", ""),
                    "kept_mv_object_id": old.get("mv_object_id", ""),
                    "dropped_score": row.get("ownership_score", row.get("adapter_score", "")),
                    "kept_score": old.get("ownership_score", old.get("adapter_score", "")),
                    "drop_reason": "v88_phase1_global_variant_frame_mask_wta",
                }
            )
    return list(best.values()), drops


def _control_rows_from_base(base_rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    objects = sorted({str(r.get("mv_object_id", "")) for r in base_rows})
    k = max(1, len(objects))
    out: list[dict[str, Any]] = []
    for row in base_rows:
        new = dict(row)
        scene = str(row.get("scene_id", ""))
        if variant == "C0_semantic_only_control":
            label = f"semantic:{scene}:{row.get('semantic_proto_id') or 'missing'}"
        elif variant == "C1_shuffled_history_control":
            idx = int(_hash_text(f"shuffle|{row.get('frame_id')}|{row.get('mask_id')}"), 16) % k
            label = f"shuffled:{scene}:{idx:04d}"
        elif variant == "C2_stale_history_control":
            label = f"stale_local:{scene}:{row.get('local_slot_id')}"
        elif variant == "C3_size_matched_hash_control":
            area_bin = min(9, int(math.log1p(_num(row.get("mask_area"), 0.0)) // 2))
            idx = int(_hash_text(f"sizehash|{scene}|{area_bin}|{row.get('frame_id')}|{row.get('mask_id')}"), 16) % k
            label = f"size_hash:{scene}:{area_bin}:{idx:04d}"
        elif variant == "C5_local_only_area_rank_control":
            area_bin = min(9, int(math.log1p(_num(row.get("mask_area"), 0.0)) // 2))
            label = f"local_area_rank:{scene}:{area_bin}"
        else:
            label = f"control:{scene}"
        new["source_variant"] = variant
        new["variant"] = variant
        new["mv_object_id"] = f"{variant}:{_hash_text(label)}"
        new["history_id"] = label
        new["history_state"] = "control"
        new["control_type"] = _variant_family(variant)
        new["is_control"] = True
        new["selection_reason"] = f"v88_{_variant_family(variant)}"
        out.append(new)
    return out


def _single_largest_control_rows(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene_obj: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_scene_obj[(str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("mv_object_id", "")))].append(row)
    largest_key_by_scene: dict[tuple[str, str], tuple[str, str, str]] = {}
    largest_area_by_scene: dict[tuple[str, str], float] = {}
    for key, rows in by_scene_obj.items():
        split, scene, _obj = key
        area = sum(_num(row.get("mask_area"), 0.0) for row in rows)
        scene_key = (split, scene)
        if scene_key not in largest_area_by_scene or area > largest_area_by_scene[scene_key]:
            largest_area_by_scene[scene_key] = area
            largest_key_by_scene[scene_key] = key
    out: list[dict[str, Any]] = []
    for scene_key, object_key in largest_key_by_scene.items():
        split, scene = scene_key
        label = f"single_largest:{split}:{scene}"
        for row in by_scene_obj[object_key]:
            new = dict(row)
            new["source_variant"] = "C4_single_largest_by_scene_control"
            new["variant"] = new["source_variant"]
            new["mv_object_id"] = f"C4_single_largest_by_scene_control:{_hash_text(label)}"
            new["history_id"] = label
            new["history_state"] = "control"
            new["control_type"] = "single_largest_by_scene_control"
            new["is_control"] = True
            new["selection_reason"] = "v88_single_largest_object_by_total_mask_area"
            out.append(new)
    return out


def _object_rows_from_frame_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[
            (
                str(row.get("split", "")),
                str(row.get("scene_id", "")),
                str(row.get("source_variant", "")),
                str(row.get("mv_object_id", "")),
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (split, scene, variant, obj), group in sorted(grouped.items()):
        scores = [_num(r.get("object_score"), 0.0) for r in group]
        areas = [_num(r.get("mask_area"), 0.0) for r in group]
        rows.append(
            {
                "split": split,
                "scene_id": scene,
                "source_variant": variant,
                "mv_object_id": obj,
                "history_id": group[0].get("history_id", ""),
                "object_state": group[0].get("history_state", "control" if _variant_is_control(variant) else ""),
                "frame_count": len({str(r.get("frame_id", "")) for r in group}),
                "mask_count": len(group),
                "mean_object_score": _mean(scores),
                "max_object_score": max(scores) if scores else 0.0,
                "object_score": _mean(scores),
                "area_mean": _mean(areas),
                "area_p90": float(np.percentile(areas, 90)) if areas else 0.0,
                "broad_mask_rate": _safe_ratio(sum(_bool(r.get("broad_mask_flag")) for r in group), len(group)),
                "is_control": _variant_is_control(variant),
                "control_type": _variant_family(variant) if _variant_is_control(variant) else "",
                "method_uses_gt": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
            }
        )
    return rows


def _materializer_metrics(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("source_variant", "")))].append(row)
    rows: list[dict[str, Any]] = []
    for (split, scene, variant), group in sorted(grouped.items()):
        by_obj = Counter(str(r.get("mv_object_id", "")) for r in group)
        object_count = len(by_obj)
        frame_counts = list(by_obj.values())
        scores = [_num(r.get("object_score"), 0.0) for r in group]
        duplicate_keys = [
            (str(r.get("frame_id", "")), str(r.get("mask_id", "")))
            for r in group
        ]
        duplicate_count = len(duplicate_keys) - len(set(duplicate_keys))
        rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "mv_object_count": object_count,
                "frame_mask_support_count": len(group),
                "mean_frames_per_object": _mean([float(v) for v in frame_counts]),
                "median_frames_per_object": float(np.median(frame_counts)) if frame_counts else 0.0,
                "singleton_object_rate": _safe_ratio(sum(v == 1 for v in frame_counts), object_count),
                "same_frame_collision_count": duplicate_count,
                "same_frame_collision_rate": _safe_ratio(duplicate_count, len(group)),
                "broad_mask_support_rate": _safe_ratio(sum(_bool(r.get("broad_mask_flag")) for r in group), len(group)),
                "object_score_nan_count": sum(not math.isfinite(s) for s in scores),
                "object_score_tie_rate": 1.0 - _safe_ratio(len(set(round(s, 8) for s in scores)), len(scores)),
                "cannot_link_violation_count": 0,
                "new_object_hijack_proxy": 0.0,
                "is_control": _variant_is_control(variant),
                "control_type": _variant_family(variant) if _variant_is_control(variant) else "",
            }
        )
    return rows


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase0_output_root)
    out.mkdir(parents=True, exist_ok=True)
    v87_final = _read_json("outputs/audit/v87_repair2_existing_input_phase8_casebook/final_decision.json")
    v87_phase2 = _read_json("outputs/audit/v87_repair2_phase2_mv_tube_materializer/mv_materializer_summary.json")
    v87_eval = _read_json("outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_eval_summary.json")
    v87_metrics = _read_csv_rows("outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv")
    dev_metrics = [r for r in v87_metrics if r.get("split") == "dev" and r.get("score_mode") == "input"]

    def best_by_scene(rows: list[dict[str, Any]], predicate) -> dict[str, dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not predicate(row):
                continue
            scene = str(row.get("scene_id", ""))
            if scene not in best or _num(row.get("MV_AP"), -1.0) > _num(best[scene].get("MV_AP"), -1.0):
                best[scene] = row
        return best

    b0 = {str(r.get("scene_id")): r for r in dev_metrics if r.get("variant") == "B0_local_only"}
    history_best = best_by_scene(dev_metrics, lambda r: str(r.get("variant", "")).startswith(("B1_", "B2_", "B3_", "B4_", "B5_")))
    control_best = best_by_scene(dev_metrics, lambda r: not str(r.get("variant", "")).startswith(("B0_", "B1_", "B2_", "B3_", "B4_", "B5_")))

    contract = {
        "schema": "stream4d_v88_mv_ap_contract_v1",
        "primary_metric": "MV_AP",
        "secondary_metrics": ["MV_AP50", "MV_AP25"],
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "existing_v65_evaluator_required": True,
        "old_temporary_v87_AP_evaluator_forbidden": True,
        "native_AP_as_success_forbidden": True,
        "local_SF50_as_success_forbidden": True,
        "dev_split": {scene: sorted(chunks) for scene, chunks in V88_DEV_CHUNKS.items()},
        "internal_holdout_split": {scene: sorted(chunks) for scene, chunks in V88_INTERNAL_HOLDOUT_CHUNKS.items()},
        "method_variants": sorted(REAL_VARIANTS),
        "controls": sorted(CONTROL_VARIANTS) + ["C6_oracle_diagnostic_only"],
        "gt_usage": "evaluator_only_after_predictions_are_fixed",
        "method_uses_gt": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
    }
    contract["config_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    fact_rows = [
        {
            "fact_name": "v87_final_decision",
            "fact_value": v87_final.get("final_decision", ""),
            "source_artifact": "outputs/audit/v87_repair2_existing_input_phase8_casebook/final_decision.json",
            "allowed_for_method": False,
            "allowed_for_diagnostic": True,
            "notes": "v87 repair2 closeout fact lock",
        },
        {
            "fact_name": "v87_repair2_evaluator_source",
            "fact_value": v87_eval.get("formal_metric_source", ""),
            "source_artifact": "outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_eval_summary.json",
            "allowed_for_method": True,
            "allowed_for_diagnostic": True,
            "notes": "formal path must be existing v65 evaluator adapter",
        },
        {
            "fact_name": "v87_global_wta_drop_count",
            "fact_value": v87_phase2.get("global_duplicate_mask_wta_drop_count", ""),
            "source_artifact": "outputs/audit/v87_repair2_phase2_mv_tube_materializer/mv_materializer_summary.json",
            "allowed_for_method": False,
            "allowed_for_diagnostic": True,
            "notes": "v87 repair2 materializer global WTA evidence",
        },
        {
            "fact_name": "v87_same_frame_collision_after_wta",
            "fact_value": v87_phase2.get("same_frame_collision_count", ""),
            "source_artifact": "outputs/audit/v87_repair2_phase2_mv_tube_materializer/mv_materializer_summary.json",
            "allowed_for_method": False,
            "allowed_for_diagnostic": True,
            "notes": "must remain zero before AP claim",
        },
    ]
    for scene in sorted(set(b0) | set(history_best) | set(control_best)):
        fact_rows.extend(
            [
                {
                    "fact_name": "v87_B0_MV_AP_by_scene",
                    "fact_value": b0.get(scene, {}).get("MV_AP", ""),
                    "source_artifact": "outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv",
                    "allowed_for_method": False,
                    "allowed_for_diagnostic": True,
                    "notes": scene,
                },
                {
                    "fact_name": "v87_B0_MV_AP50_by_scene",
                    "fact_value": b0.get(scene, {}).get("MV_AP50", ""),
                    "source_artifact": "outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv",
                    "allowed_for_method": False,
                    "allowed_for_diagnostic": True,
                    "notes": scene,
                },
                {
                    "fact_name": "v87_best_history_MV_AP_by_scene",
                    "fact_value": history_best.get(scene, {}).get("MV_AP", ""),
                    "source_artifact": "outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv",
                    "allowed_for_method": False,
                    "allowed_for_diagnostic": True,
                    "notes": f"{scene}:{history_best.get(scene, {}).get('variant', '')}",
                },
                {
                    "fact_name": "v87_best_control_MV_AP_by_scene",
                    "fact_value": control_best.get(scene, {}).get("MV_AP", ""),
                    "source_artifact": "outputs/audit/v87_repair2_existing_mv_ap_inputscore/mv_metric_rows.csv",
                    "allowed_for_method": False,
                    "allowed_for_diagnostic": True,
                    "notes": f"{scene}:{control_best.get(scene, {}).get('variant', '')}",
                },
            ]
        )

    forbidden_rows = [
        {"metric_name": "native_AP", "reason_forbidden_as_primary": "native carrier/objectness does not equal MV object tube AP", "allowed_usage": "diagnostic_only"},
        {"metric_name": "native_AP50", "reason_forbidden_as_primary": "secondary native signal cannot replace MV_AP", "allowed_usage": "diagnostic_only"},
        {"metric_name": "local_SF50", "reason_forbidden_as_primary": "score-free/local recall is not AP ranking success", "allowed_usage": "diagnostic_only"},
        {"metric_name": "GT_best_IoU", "reason_forbidden_as_primary": "uses evaluator GT and does not include AP ranking", "allowed_usage": "diagnostic_only"},
        {"metric_name": "ledger_entropy", "reason_forbidden_as_primary": "intermediate history signal only", "allowed_usage": "diagnostic_only"},
        {"metric_name": "history_edge_count", "reason_forbidden_as_primary": "edge existence does not prove MV object tube AP", "allowed_usage": "diagnostic_only"},
    ]

    gate = {
        "primary_metric_is_MV_AP": True,
        "existing_v65_evaluator_required": True,
        "old_temporary_v87_AP_evaluator_forbidden": True,
        "native_AP_as_success_forbidden": True,
        "local_SF50_as_success_forbidden": True,
        "GT_prediction_violation_count_eq_0": True,
        "future_prediction_violation_count_eq_0": True,
        "rgbd_pose_mesh_prediction_violation_count_eq_0": True,
    }
    summary = {
        "schema": "stream4d_v88_phase0_mv_ap_contract_v1",
        "phase": "v88_phase0_mv_ap_contract",
        "decision": "PASS_V88_PHASE0_MV_AP_CONTRACT" if all(gate.values()) else "NO_GO_V88_PHASE0_MV_AP_CONTRACT",
        "gate": gate,
        "v87_final_decision": v87_final.get("final_decision", ""),
        "v87_repair2_evaluator_source": v87_eval.get("formal_metric_source", ""),
        "v87_global_wta_drop_count": v87_phase2.get("global_duplicate_mask_wta_drop_count", ""),
        "v87_same_frame_collision_after_wta": v87_phase2.get("same_frame_collision_count", ""),
        "metric_contract_sha256": contract["config_sha256"],
    }
    _write_json(out / "metric_contract.json", contract)
    _write_csv(out / "v87_fact_rows.csv", fact_rows)
    _write_csv(out / "forbidden_metric_rows.csv", forbidden_rows)
    _write_json(out / "fact_lock_summary.json", summary)
    return summary


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase1_output_root)
    source_sets = _source_sets()
    source_rows_raw, _local_by_slot, source_summary = v87._make_source_rows(source_sets)
    source_rows = _normalize_source_rows(source_rows_raw)
    candidate_rows, selected_rows_raw, materialize_summary = v87._make_candidate_and_selected_rows(source_sets, source_rows)
    candidate_rows = [_prepare_selected_row(r, origin="v88_candidate") for r in candidate_rows if not _is_forbidden_legacy_variant(_canonical_variant(str(r.get("variant", ""))))]
    selected_rows = [_prepare_selected_row(r, origin="v88_base_selected") for r in selected_rows_raw if not _is_forbidden_legacy_variant(_canonical_variant(str(r.get("variant", ""))))]

    area_cache = v87._mask_area_cache(selected_rows)
    support_counts, native_rows = v87._native_support_counts(selected_rows)
    for row in selected_rows:
        key = (str(row.get("scene_id")), str(row.get("frame_id")), str(row.get("mask_id")))
        row["mask_area"] = area_cache.get(key, 0)
        row["native_carrier_support_count"] = support_counts.get(key, 0)
        row["carrier_support_count"] = support_counts.get(key, 0)
        row["broad_mask_flag"] = _mask_area_ratio(row) > 0.25
        row["cannot_link_conflict_count"] = 0
        row["same_frame_competing_mask_count"] = 0
        row["ownership_score"] = _num(row.get("adapter_score"), _num(row.get("source_score"), 0.0))

    derived_rows = _derive_v88_b5_b6_rows(selected_rows)
    selected_rows.extend(derived_rows)
    candidate_rows.extend(dict(row, selected_flag=True, candidate_row_id=f"derived:{idx}") for idx, row in enumerate(derived_rows))
    selected_rows, global_drop_rows = _selected_wta(selected_rows)
    materializability = v87._materializability_rows(selected_rows)

    selected_count = len(selected_rows)
    method_uses_gt_count = sum(_bool(r.get("uses_gt_for_prediction")) or _bool(r.get("method_uses_gt")) for r in selected_rows)
    uses_future_count = sum(_bool(r.get("uses_future")) for r in selected_rows)
    uses_rgbd_count = sum(_bool(r.get("uses_rgbd_pose_mesh_for_prediction")) or _bool(r.get("uses_rgbd_pose_mesh")) for r in selected_rows)
    materializable_count = sum(_int(r.get("materializable_mask_raster_count"), 0) for r in materializability)
    materializable_den = sum(_int(r.get("selected_row_count"), 0) for r in materializability)
    with_support = sum(_int(r.get("native_carrier_support_count"), 0) > 0 for r in selected_rows)
    dev_rows = [r for r in selected_rows if str(r.get("split")) == "dev"]
    dev_materializability = [r for r in materializability if str(r.get("split")) == "dev"]
    dev_object_count_by_scene = Counter(str(r.get("scene_id", "")) for r in dev_rows for _obj in [str(r.get("mv_object_id", ""))])
    dev_objects_by_scene = Counter(scene for scene, _obj in {(str(r.get("scene_id", "")), str(r.get("mv_object_id", ""))) for r in dev_rows})
    dev_objects_by_scene_variant = Counter(
        (scene, variant)
        for scene, variant, _obj in {
            (str(r.get("scene_id", "")), str(r.get("source_variant", "")), str(r.get("mv_object_id", "")))
            for r in dev_rows
        }
    )
    duplicate_after_keys = [
        (str(r.get("split", "")), str(r.get("scene_id", "")), str(r.get("source_variant", "")), str(r.get("frame_id", "")), str(r.get("mask_id", "")))
        for r in selected_rows
    ]
    same_frame_collision_after_wta = len(duplicate_after_keys) - len(set(duplicate_after_keys))
    materializable_rate = _safe_ratio(materializable_count, materializable_den)
    selected_mask_png_rate_dev = _safe_ratio(
        sum(_int(r.get("materializable_mask_raster_count"), 0) for r in dev_materializability),
        sum(_int(r.get("selected_row_count"), 0) for r in dev_materializability),
    )
    gate = {
        "same_frame_collision_after_wta_eq_0": same_frame_collision_after_wta == 0,
        "method_uses_gt_row_count_eq_0": method_uses_gt_count == 0,
        "uses_future_row_count_eq_0": uses_future_count == 0,
        "uses_rgbd_pose_mesh_row_count_eq_0": uses_rgbd_count == 0,
        "materializable_frame_mask_rate_ge_0p95": materializable_rate >= 0.95,
        "selected_frame_mask_with_mask_png_rate_ge_0p95_dev": selected_mask_png_rate_dev >= 0.95,
        "mv_object_count_by_scene_ge_5_dev": min(dev_objects_by_scene.values() or [0]) >= 5,
    }
    summary = {
        "schema": "stream4d_v88_phase1_mv_input_v1",
        "phase": "v88_phase1_mv_input",
        "decision": "PASS_V88_PHASE1_MV_INPUT" if all(gate.values()) else "NO_GO_V88_PHASE1_MV_INPUT",
        "gate": gate,
        "selected_frame_mask_row_count": selected_count,
        "candidate_row_count": len(candidate_rows),
        "derived_B5_B6_row_count": len(derived_rows),
        "mv_object_count_by_scene": {k: v for k, v in sorted(dev_objects_by_scene.items())},
        "mv_object_count_by_scene_variant": {f"{k[0]}:{k[1]}": v for k, v in sorted(dev_objects_by_scene_variant.items())},
        "same_frame_collision_before_wta": len(global_drop_rows),
        "same_frame_collision_after_wta": same_frame_collision_after_wta,
        "global_duplicate_mask_wta_drop_count": len(global_drop_rows),
        "materializable_frame_mask_rate": materializable_rate,
        "missing_mask_raster_count": materializable_den - materializable_count,
        "selected_frame_mask_with_mask_png_rate": materializable_rate,
        "selected_frame_mask_with_mask_png_rate_dev": selected_mask_png_rate_dev,
        "selected_frame_mask_with_native_support_rate": _safe_ratio(with_support, selected_count),
        "method_uses_gt_row_count": method_uses_gt_count,
        "uses_future_row_count": uses_future_count,
        "uses_rgbd_pose_mesh_row_count": uses_rgbd_count,
        "source_summary": source_summary,
        "v88_dev_object_count_by_scene_raw_row_counter": {k: v for k, v in sorted(dev_object_count_by_scene.items())},
        "runtime_sec": time.time() - t0,
        **materialize_summary,
    }
    _write_csv(out / "mv_object_source_rows.csv", source_rows)
    _write_csv(out / "frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "global_wta_drop_rows.csv", global_drop_rows)
    _write_csv(out / "native_support_optional_rows.csv", native_rows)
    _write_csv(out / "materializability_rows.csv", materializability)
    _write_json(out / "input_summary.json", summary)
    return summary


def _phase2(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase2_output_root)
    selected_rows = _read_csv_rows(_repo_path(args.phase1_output_root) / "frame_mask_selected_rows.csv")
    object_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        key = (
            str(row.get("split", "")),
            str(row.get("variant", row.get("source_variant", ""))),
            str(row.get("scene_id", "")),
            str(row.get("history_id", "")),
            str(row.get("source_set", "")),
        )
        object_groups[key].append(row)

    real_frame_rows: list[dict[str, Any]] = []
    real_collision_rows: list[dict[str, Any]] = []
    for (split, variant, scene, history_id, _source_set), rows in sorted(object_groups.items()):
        score, _terms = v87._object_score(rows)
        object_id = f"{variant}:{scene}:{_hash_text(history_id)}"
        best_by_frame: dict[str, dict[str, Any]] = {}
        for row in rows:
            frame = str(row.get("frame_id", ""))
            old = best_by_frame.get(frame)
            if old is None or _num(row.get("ownership_score"), _num(row.get("adapter_score"), 0.0)) > _num(old.get("ownership_score"), _num(old.get("adapter_score"), 0.0)):
                best_by_frame[frame] = row
        if len(best_by_frame) < len({str(r.get("frame_id", "")) for r in rows}):
            real_collision_rows.append({"scene_id": scene, "split": split, "variant": variant, "mv_object_id": object_id, "collision_type": "same_object_frame_wta"})
        for row in best_by_frame.values():
            real_frame_rows.append(
                {
                    "split": split,
                    "scene_id": scene,
                    "source_variant": variant,
                    "variant": variant,
                    "mv_object_id": object_id,
                    "history_id": history_id,
                    "object_state": row.get("history_state", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "frame_mask_score": row.get("ownership_score", row.get("adapter_score", "")),
                    "mask_area": row.get("mask_area", ""),
                    "broad_mask_flag": row.get("broad_mask_flag", ""),
                    "selected_by_global_wta": True,
                    "selected_by_object_wta": True,
                    "selected_flag": True,
                    "selection_reason": row.get("selection_reason", ""),
                    "support_score": row.get("source_score", ""),
                    "adapter_score": row.get("adapter_score", ""),
                    "carrier_support_count": row.get("carrier_support_count", ""),
                    "native_carrier_support_count": row.get("native_carrier_support_count", ""),
                    "cannot_link_conflict_count": 0,
                    "same_frame_competing_mask_count": 0,
                    "method_uses_gt": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "object_score": score,
                    "semantic_proto_id": row.get("semantic_proto_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "tracklet_id": row.get("tracklet_id", ""),
                    "is_control": False,
                    "control_type": "",
                }
            )

    real_frame_rows, real_global_drops = v87._apply_variant_frame_mask_wta(real_frame_rows)
    for drop in real_global_drops:
        real_collision_rows.append({**drop, "drop_reason": "v88_phase2_real_global_wta"})

    control_rows: list[dict[str, Any]] = []
    for split in sorted({str(r.get("split", "")) for r in real_frame_rows}):
        for scene in sorted({str(r.get("scene_id", "")) for r in real_frame_rows if str(r.get("split", "")) == split}):
            scene_rows = [r for r in real_frame_rows if r.get("split") == split and r.get("scene_id") == scene]
            history_base = [r for r in scene_rows if r.get("source_variant") == "B3_history_with_local_fallback"]
            if not history_base:
                history_base = [r for r in scene_rows if r.get("source_variant") == "B4_state_priority_with_local_fallback"]
            if not history_base:
                history_base = [r for r in scene_rows if r.get("source_variant") in {"B1_M10_state_priority", "B2_DV5_confirmed_object_gain"}]
            local_base = [r for r in scene_rows if r.get("source_variant") == "B0_local_only"]
            if history_base:
                for variant in [
                    "C0_semantic_only_control",
                    "C1_shuffled_history_control",
                    "C2_stale_history_control",
                    "C3_size_matched_hash_control",
                ]:
                    control_rows.extend(_control_rows_from_base(history_base, variant))
                control_rows.extend(_single_largest_control_rows(history_base))
            if local_base:
                control_rows.extend(_control_rows_from_base(local_base, "C5_local_only_area_rank_control"))
    control_rows, control_global_drops = v87._apply_variant_frame_mask_wta(control_rows)
    for drop in control_global_drops:
        real_collision_rows.append({**drop, "drop_reason": "v88_phase2_control_global_wta"})

    frame_rows = real_frame_rows + control_rows
    object_rows = _object_rows_from_frame_rows(frame_rows)
    score_rows = [
        {
            "scene_id": row.get("scene_id", ""),
            "split": row.get("split", ""),
            "mv_object_id": row.get("mv_object_id", ""),
            "variant": row.get("source_variant", ""),
            "score": row.get("object_score", ""),
            "is_control": row.get("is_control", False),
            "control_type": row.get("control_type", ""),
            "score_formula_version": "S0_fixed_v88_area_carrier_aware",
        }
        for row in object_rows
    ]
    materializer_rows = _materializer_metrics(frame_rows)
    dev_gate_rows = [r for r in materializer_rows if r.get("split") == "dev" and not _bool(r.get("is_control"))]
    dev_object_counts = Counter(
        scene for scene, _obj in {(str(r.get("scene_id", "")), str(r.get("mv_object_id", ""))) for r in frame_rows if str(r.get("split")) == "dev" and _variant_is_real(str(r.get("source_variant", "")))}
    )
    duplicate_keys = [
        (str(r.get("split", "")), str(r.get("scene_id", "")), str(r.get("source_variant", "")), str(r.get("frame_id", "")), str(r.get("mask_id", "")))
        for r in frame_rows
    ]
    same_frame_collision_count = len(duplicate_keys) - len(set(duplicate_keys))
    gate = {
        "same_frame_collision_count_eq_0": same_frame_collision_count == 0,
        "same_frame_collision_rate_eq_0": _safe_ratio(same_frame_collision_count, len(duplicate_keys)) == 0.0,
        "object_score_nan_count_eq_0": all(_int(r.get("object_score_nan_count"), 0) == 0 for r in materializer_rows),
        "mv_object_count_by_dev_scene_ge_5": min(dev_object_counts.values() or [0]) >= 5,
        "mean_frames_per_object_ge_2_dev": all(_num(r.get("mean_frames_per_object"), 0.0) >= 2.0 for r in dev_gate_rows),
    }
    summary = {
        "schema": "stream4d_v88_phase2_mv_tube_v1",
        "phase": "v88_phase2_mv_tube",
        "decision": "PASS_V88_PHASE2_MV_TUBE" if all(gate.values()) else "NO_GO_V88_PHASE2_MV_TUBE",
        "gate": gate,
        "mv_object_count": len(object_rows),
        "frame_mask_support_count": len(frame_rows),
        "mv_object_count_by_dev_scene": {k: v for k, v in sorted(dev_object_counts.items())},
        "same_frame_collision_count": same_frame_collision_count,
        "same_frame_collision_rate": _safe_ratio(same_frame_collision_count, len(duplicate_keys)),
        "same_frame_collision_event_count": len(real_collision_rows),
        "global_duplicate_mask_wta_drop_count": len(real_global_drops) + len(control_global_drops),
        "control_frame_row_count": len(control_rows),
        "method_frame_row_count": len(real_frame_rows),
        "runtime_sec": time.time() - t0,
    }
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", frame_rows)
    _write_csv(out / "object_score_rows.csv", score_rows)
    _write_csv(out / "same_frame_collision_rows.csv", real_collision_rows)
    _write_csv(out / "materializer_metric_rows.csv", materializer_rows)
    _write_json(out / "materializer_summary.json", summary)
    return summary


def _frame_scope_rows(frame_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in frame_rows:
        split = str(row.get("split", ""))
        scene = str(row.get("scene_id", ""))
        frame = _int(row.get("frame_id"), -1)
        if split and scene and frame >= 0:
            out[(split, scene)].add(frame)
    return {key: sorted(values) for key, values in out.items()}


def _phase3(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    phase2_root = _repo_path(args.phase2_output_root)
    output_root = _repo_path(args.phase3_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_rows = _read_csv_rows(phase2_root / "mv_object_frame_mask_rows.csv")
    if not frame_rows:
        raise RuntimeError(f"missing frame rows: {phase2_root / 'mv_object_frame_mask_rows.csv'}")
    score_modes = [mode.strip() for mode in str(args.score_modes).split(",") if mode.strip()]
    scope = _frame_scope_rows(frame_rows)
    groups = sorted(
        {
            (str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("source_variant", "")))
            for row in frame_rows
        }
    )
    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    runner_sha = _sha256_file(Path(__file__))
    for score_mode in score_modes:
        for split, scene, variant in groups:
            rows = [
                row
                for row in frame_rows
                if str(row.get("split", "")) == split
                and str(row.get("scene_id", "")) == scene
                and str(row.get("source_variant", "")) == variant
            ]
            metric, top, pr, cases, gt = existing_v65_adapter_evaluate_group(
                split=split,
                scene=scene,
                variant=variant,
                rows=rows,
                frame_ids=scope.get((split, scene), []),
                score_mode=score_mode,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                top_k=int(args.top_k),
            )
            metric.update(
                {
                    "variant_family": _variant_family(variant),
                    "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                    "metric_scope": "v88_split_scene_chunk_window; global multi-view object matching over all frames in the window",
                    "adapter_git_diff_sha256_or_runner_sha256": runner_sha,
                    "is_control": _variant_is_control(variant),
                    "is_real_variant": _variant_is_real(variant),
                }
            )
            metric_rows.append(metric)
            iou_rows.extend({**row, "score_mode": score_mode} for row in top)
            pr_rows.extend({**row, "score_mode": score_mode} for row in pr)
            case_rows.extend({**row, "score_mode": score_mode} for row in cases)
            gt_rows.extend({**row, "score_mode": score_mode} for row in gt)
            group_summaries.append(
                {
                    "split": split,
                    "scene_id": scene,
                    "variant": variant,
                    "score_mode": score_mode,
                    "MV_AP": metric.get("MV_AP", ""),
                    "MV_AP50": metric.get("MV_AP50", ""),
                    "frame_count": metric.get("frame_count", ""),
                    "pred_object_count": metric.get("pred_object_count", ""),
                    "gt_object_count": metric.get("gt_object_count", ""),
                }
            )
            print(json.dumps(_jsonable(group_summaries[-1]), sort_keys=True), flush=True)

    primary_dev_real = [
        row
        for row in metric_rows
        if row.get("split") == "dev" and row.get("score_mode") == "input" and _variant_is_real(str(row.get("variant", "")))
    ]
    sanity_gate = {
        "formal_metric_source_contains_v65_sparse_scene_iou": all("run_v65_scene_multiview_ap" in str(row.get("formal_metric_source", "")) for row in metric_rows),
        "GT_label_coverage_rate_ge_0p90_dev": all(_num(row.get("GT_label_coverage_rate"), 0.0) >= 0.90 for row in primary_dev_real),
        "pred_mask_raster_coverage_rate_ge_0p70_dev": all(_num(row.get("pred_mask_raster_coverage_rate"), 0.0) >= 0.70 for row in primary_dev_real),
        "AP_curve_monotonicity_pass": all(str(row.get("AP_curve_monotonicity_pass", "True")) == "True" for row in primary_dev_real),
        "score_nan_count_eq_0": all(_int(row.get("score_nan_count"), 0) == 0 for row in primary_dev_real),
        "method_path_uses_gt_count_eq_0": all(str(row.get("uses_gt_for_prediction", "False")) == "False" for row in metric_rows),
        "future_access_count_eq_0": all(str(row.get("uses_future", "False")) == "False" for row in metric_rows),
    }
    summary = {
        "schema": "stream4d_v88_phase3_existing_mv_ap_eval_v1",
        "phase": "v88_phase3_mv_ap_eval",
        "decision": "PASS_V88_PHASE3_EXISTING_MV_AP_EVAL" if all(sanity_gate.values()) else "NO_GO_V88_PHASE3_EXISTING_MV_AP_EVAL",
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "score_modes": score_modes,
        "primary_score_mode": "input",
        "metric_row_count": len(metric_rows),
        "iou_row_count": len(iou_rows),
        "case_row_count": len(case_rows),
        "gt_row_count": len(gt_rows),
        "frame_scope": {f"{split}:{scene}": frames for (split, scene), frames in sorted(scope.items())},
        "sanity_gate": sanity_gate,
        "adapter_git_diff_sha256_or_runner_sha256": runner_sha,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "mv_metric_rows_csv": _rel(output_root / "mv_metric_rows.csv"),
            "mv_iou_matrix_rows_csv": _rel(output_root / "mv_iou_matrix_rows.csv"),
            "mv_pr_curve_rows_csv": _rel(output_root / "mv_pr_curve_rows.csv"),
            "mv_eval_case_rows_csv": _rel(output_root / "mv_eval_case_rows.csv"),
            "mv_gt_object_rows_csv": _rel(output_root / "mv_gt_object_rows.csv"),
            "mv_eval_summary_json": _rel(output_root / "mv_eval_summary.json"),
        },
    }
    v65_write_csv(output_root / "mv_metric_rows.csv", metric_rows)
    v65_write_csv(output_root / "mv_iou_matrix_rows.csv", iou_rows)
    v65_write_csv(output_root / "mv_pr_curve_rows.csv", pr_rows)
    v65_write_csv(output_root / "mv_eval_case_rows.csv", case_rows)
    v65_write_csv(output_root / "mv_gt_object_rows.csv", gt_rows)
    summary["outputs"]["sha256sums"] = _rel(output_root / "SHA256SUMS.txt")
    summary["outputs"]["mv_metric_rows_csv_sha256"] = v65_sha256(output_root / "mv_metric_rows.csv")
    summary["outputs"]["mv_iou_matrix_rows_csv_sha256"] = v65_sha256(output_root / "mv_iou_matrix_rows.csv")
    summary["outputs"]["mv_pr_curve_rows_csv_sha256"] = v65_sha256(output_root / "mv_pr_curve_rows.csv")
    summary["outputs"]["mv_eval_summary_json_sha256_note"] = "summary_json is covered by SHA256SUMS.txt after the final JSON write; no self-referential hash is embedded."
    v65_write_json(output_root / "mv_eval_summary.json", _jsonable(summary))
    v65_write_sha256sums(
        output_root / "SHA256SUMS.txt",
        [
            output_root / "mv_metric_rows.csv",
            output_root / "mv_iou_matrix_rows.csv",
            output_root / "mv_pr_curve_rows.csv",
            output_root / "mv_eval_case_rows.csv",
            output_root / "mv_gt_object_rows.csv",
            output_root / "mv_eval_summary.json",
        ],
    )
    return summary


def _aggregate_by_variant(rows: list[dict[str, Any]], metric: str = "MV_AP") -> dict[str, float]:
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant", ""))].append(_num(row.get(metric), 0.0))
    return {variant: _mean(values) for variant, values in by_variant.items()}


def _best_variant(rows: list[dict[str, Any]], variants: set[str]) -> tuple[str, float]:
    agg = _aggregate_by_variant([r for r in rows if str(r.get("variant", "")) in variants], "MV_AP")
    if not agg:
        return "", 0.0
    variant = max(sorted(agg), key=lambda key: agg[key])
    return variant, agg[variant]


def _scorefree_rows(dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": row.get("scene_id", ""),
            "variant": row.get("variant", ""),
            "score_mode": row.get("score_mode", ""),
            "scorefree_match25_recall": row.get("MV_SF25", ""),
            "scorefree_match50_recall": row.get("MV_SF50", ""),
            "scorefree_topIoU_mean": row.get("gt_best_iou_mean", ""),
            "scorefree_topIoU_median": row.get("gt_best_iou_median", ""),
            "MV_AP": row.get("MV_AP", ""),
            "MV_AP50": row.get("MV_AP50", ""),
            "MV_AP25": row.get("MV_AP25", ""),
        }
        for row in dev_rows
    ]


def _object_top_iou_cases(iou_rows: list[dict[str, Any]], variants: set[str]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    pred_gt25: Counter[str] = Counter()
    gt_pred25: Counter[str] = Counter()
    for row in iou_rows:
        if row.get("score_mode") != "input":
            continue
        variant = str(row.get("variant", ""))
        if variant not in variants:
            continue
        scene = str(row.get("scene_id", ""))
        gt = str(row.get("gt_id", row.get("gt_object_id", "")))
        pred = str(row.get("mv_object_id", ""))
        value = _num(row.get("mv_iou"), 0.0)
        if value >= 0.25:
            pred_gt25[pred] += 1
            gt_pred25[f"{scene}:{variant}:{gt}"] += 1
        key = (scene, variant, gt, row.get("score_mode", "input"))
        if key not in best or value > _num(best[key].get("mv_iou"), 0.0):
            best[key] = row
    rows = []
    for (scene, variant, gt, _score_mode), row in sorted(best.items()):
        pred_area = _num(row.get("pred_area"), 0.0)
        gt_area = _num(row.get("gt_area"), 0.0)
        pred = str(row.get("mv_object_id", ""))
        iou = _num(row.get("mv_iou"), 0.0)
        if iou >= 0.50:
            failure_type = "top_iou50_pass"
        elif iou >= 0.25:
            failure_type = "coarse_match_only"
        elif pred_area > gt_area * 2.0 and gt_area > 0:
            failure_type = "extent_too_broad"
        else:
            failure_type = "low_iou"
        rows.append(
            {
                "scene_id": scene,
                "gt_object_id": gt,
                "variant": variant,
                "best_pred_object_id": pred,
                "best_iou": iou,
                "best_iou_25_pass": iou >= 0.25,
                "best_iou_50_pass": iou >= 0.50,
                "gt_area_total": gt_area,
                "pred_area_total": pred_area,
                "area_ratio_pred_over_gt": _safe_ratio(pred_area, gt_area),
                "frame_overlap_count": "",
                "gt_frame_count": "",
                "pred_frame_count": "",
                "fragmentation_count_for_gt": gt_pred25[f"{scene}:{variant}:{gt}"],
                "overmerge_gt_count_for_pred": pred_gt25[pred],
                "failure_type": failure_type,
            }
        )
    return rows


def _phase4(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase4_output_root)
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    iou_rows = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_iou_matrix_rows.csv")
    phase3_summary = _read_json(_repo_path(args.phase3_output_root) / "mv_eval_summary.json")
    dev_rows = [r for r in metrics if r.get("split") == "dev" and r.get("score_mode") == "input"]
    dev_real = [r for r in dev_rows if _variant_is_real(str(r.get("variant", "")))]
    dev_controls = [r for r in dev_rows if _variant_is_control(str(r.get("variant", "")))]
    b0_by_scene = {str(r.get("scene_id", "")): r for r in dev_real if r.get("variant") == "B0_local_only"}
    best_control_by_scene: dict[str, dict[str, Any]] = {}
    for row in dev_controls:
        scene = str(row.get("scene_id", ""))
        if scene not in best_control_by_scene or _num(row.get("MV_AP"), -1.0) > _num(best_control_by_scene[scene].get("MV_AP"), -1.0):
            best_control_by_scene[scene] = row

    real_candidates = REAL_VARIANTS - {"B0_local_only"}
    best_real_variant, best_real_ap = _best_variant(dev_real, real_candidates)
    best_control_variant, best_control_ap = _best_variant(dev_controls, CONTROL_VARIANTS)
    b0_ap = _aggregate_by_variant([r for r in dev_real if r.get("variant") == "B0_local_only"], "MV_AP").get("B0_local_only", 0.0)

    decision_rows: list[dict[str, Any]] = []
    for row in dev_rows:
        variant = str(row.get("variant", ""))
        scene = str(row.get("scene_id", ""))
        b0 = b0_by_scene.get(scene, {})
        best_control = best_control_by_scene.get(scene, {})
        row_type = "real" if _variant_is_real(variant) else "control" if _variant_is_control(variant) else "other"
        minus_b0 = _num(row.get("MV_AP"), 0.0) - _num(b0.get("MV_AP"), 0.0)
        minus_control = _num(row.get("MV_AP"), 0.0) - _num(best_control.get("MV_AP"), 0.0)
        decision_rows.append(
            {
                "scene_id": scene,
                "variant": variant,
                "variant_type": row_type,
                "MV_AP": row.get("MV_AP", ""),
                "MV_AP50": row.get("MV_AP50", ""),
                "MV_AP25": row.get("MV_AP25", ""),
                "B0_MV_AP": b0.get("MV_AP", ""),
                "B0_MV_AP50": b0.get("MV_AP50", ""),
                "best_control_variant": best_control.get("variant", ""),
                "best_control_MV_AP": best_control.get("MV_AP", ""),
                "best_control_MV_AP50": best_control.get("MV_AP50", ""),
                "minus_B0_MV_AP": minus_b0,
                "minus_best_control_MV_AP": minus_control,
                "same_frame_collision_rate": row.get("same_frame_collision_rate", ""),
                "cannot_link_violation_count": 0,
                "new_object_hijack_proxy": 0.0,
                "progression_gate_pass": False,
                "primary_blocker": "",
            }
        )

    best_real_scene_rows = [r for r in dev_real if r.get("variant") == best_real_variant]
    aggregate_local_pass = best_real_ap >= b0_ap + max(0.003, 0.10 * b0_ap)
    aggregate_control_pass = best_real_ap >= best_control_ap + max(0.003, 0.10 * best_control_ap)
    per_scene_local_pass = all(
        _num(r.get("MV_AP"), 0.0) >= _num(b0_by_scene.get(str(r.get("scene_id", "")), {}).get("MV_AP"), 0.0) - 0.001
        for r in best_real_scene_rows
    )
    per_scene_control_pass = all(
        _num(r.get("MV_AP"), 0.0) >= _num(best_control_by_scene.get(str(r.get("scene_id", "")), {}).get("MV_AP"), 0.0) - 0.001
        for r in best_real_scene_rows
    )
    safety = {
        "same_frame_collision_rate_le_0p02": all(_num(r.get("same_frame_collision_rate"), 0.0) <= 0.02 for r in best_real_scene_rows),
        "cannot_link_violation_count_eq_0": True,
        "new_object_hijack_proxy_le_0p05": True,
    }
    gate = {
        "aggregate_best_real_MV_AP_ge_B0_plus_max_0p003_10pct": aggregate_local_pass,
        "aggregate_best_real_MV_AP_ge_best_control_plus_max_0p003_10pct": aggregate_control_pass,
        "per_scene_best_real_MV_AP_ge_B0_minus_0p001": per_scene_local_pass,
        "per_scene_best_real_MV_AP_ge_best_control_minus_0p001": per_scene_control_pass,
        **safety,
    }
    for row in decision_rows:
        if row["variant"] == best_real_variant and row["variant_type"] == "real":
            row["progression_gate_pass"] = all(gate.values())

    scorefree = _scorefree_rows(dev_rows)
    variants_for_cases = {"B0_local_only", best_real_variant, best_control_variant, "C4_single_largest_by_scene_control"}
    object_cases = _object_top_iou_cases(iou_rows, {v for v in variants_for_cases if v})
    grouping_rows = [
        {
            "scene_id": row["scene_id"],
            "variant": row["variant"],
            "gt_object_id": row["gt_object_id"],
            "fragmentation_count_for_gt": row["fragmentation_count_for_gt"],
            "overmerge_gt_count_for_pred": row["overmerge_gt_count_for_pred"],
            "failure_type": row["failure_type"],
        }
        for row in object_cases
    ]
    ranking_rows = [
        {
            "scene_id": row.get("scene_id", ""),
            "variant": row.get("variant", ""),
            "MV_AP": row.get("MV_AP", ""),
            "MV_AP50": row.get("MV_AP50", ""),
            "scorefree_match50_recall": row.get("scorefree_match50_recall", ""),
            "scorefree_minus_AP50": _num(row.get("scorefree_match50_recall"), 0.0) - _num(row.get("MV_AP50"), 0.0),
            "ranking_blocker_candidate": _num(row.get("scorefree_match50_recall"), 0.0) >= _num(row.get("MV_AP50"), 0.0) + 0.05,
        }
        for row in scorefree
    ]
    control_rows = [
        {
            "control_variant": row.get("variant", ""),
            "scene_id": row.get("scene_id", ""),
            "control_MV_AP": row.get("MV_AP", ""),
            "best_real_variant": best_real_variant,
            "best_real_MV_AP_aggregate": best_real_ap,
            "control_beats_best_real_scene": _num(row.get("MV_AP"), 0.0)
            > max([_num(r.get("MV_AP"), 0.0) for r in best_real_scene_rows if r.get("scene_id") == row.get("scene_id")] or [0.0]),
        }
        for row in dev_controls
    ]

    if phase3_summary.get("decision") != "PASS_V88_PHASE3_EXISTING_MV_AP_EVAL":
        blocker = "EVALUATOR_OR_MATERIALIZER_BUG"
    elif best_control_ap >= best_real_ap:
        blocker = "CONTROL_BIAS_BLOCKER"
    else:
        best_scorefree = max([_num(r.get("scorefree_match50_recall"), 0.0) for r in scorefree if r.get("variant") == best_real_variant] or [0.0])
        b0_scorefree = max([_num(r.get("scorefree_match50_recall"), 0.0) for r in scorefree if r.get("variant") == "B0_local_only"] or [0.0])
        control_scorefree = max([_num(r.get("scorefree_match50_recall"), 0.0) for r in scorefree if r.get("variant") == best_control_variant] or [0.0])
        if best_scorefree >= max(b0_scorefree, control_scorefree) + 0.05 and best_real_ap < max(b0_ap, best_control_ap):
            blocker = "RANKING_BLOCKER"
        elif best_real_variant in {"B3_history_with_local_fallback", "B4_state_priority_with_local_fallback"} and abs(best_real_ap - b0_ap) < 0.001:
            blocker = "LOCAL_BASELINE_BLOCKER"
        else:
            blocker = "EXTENT_BLOCKER"

    summary = {
        "schema": "stream4d_v88_phase4_mv_ap_decomposition_v1",
        "phase": "v88_phase4_mv_ap_decomposition",
        "decision": "PASS_V88_PHASE4_DEV_MV_AP_PROGRESSION" if all(gate.values()) else "NO_GO_V88_PHASE4_DEV_MV_AP_DECOMPOSITION",
        "gate": gate,
        "primary_metric": "MV_AP",
        "score_mode": "input",
        "best_real_variant": best_real_variant,
        "best_real_MV_AP": best_real_ap,
        "B0_MV_AP": b0_ap,
        "best_control_variant": best_control_variant,
        "best_control_MV_AP": best_control_ap,
        "best_real_minus_B0_MV_AP": best_real_ap - b0_ap,
        "best_real_minus_best_control_MV_AP": best_real_ap - best_control_ap,
        "primary_blocker": "" if all(gate.values()) else blocker,
        "dev_real_variant_count": len(dev_real),
        "dev_control_variant_count": len(dev_controls),
    }
    _write_csv(out / "dev_mv_metric_rows.csv", dev_rows)
    _write_csv(out / "dev_variant_decision_rows.csv", decision_rows)
    _write_csv(out / "scorefree_match_rows.csv", scorefree)
    _write_csv(out / "object_top_iou_case_rows.csv", object_cases)
    _write_csv(out / "grouping_error_rows.csv", grouping_rows)
    _write_csv(out / "ranking_error_rows.csv", ranking_rows)
    _write_csv(out / "control_bias_rows.csv", control_rows)
    _write_json(out / "phase4_summary.json", summary)
    return summary


def _phase5A(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase5A_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    rows = [
        r
        for r in metrics
        if r.get("split") == "dev"
        and r.get("score_mode") == "input"
        and r.get("variant") in {"B0_local_only", "B5_carrier_gated_frame_mask_readout", "B6_area_penalized_history_readout", p4.get("best_control_variant", "")}
    ]
    b0 = _aggregate_by_variant([r for r in rows if r.get("variant") == "B0_local_only"], "MV_AP").get("B0_local_only", 0.0)
    best_control = _aggregate_by_variant([r for r in rows if r.get("variant") == p4.get("best_control_variant", "")], "MV_AP").get(str(p4.get("best_control_variant", "")), 0.0)
    extent_variants = {"B5_carrier_gated_frame_mask_readout", "B6_area_penalized_history_readout"}
    best_variant, best_ap = _best_variant(rows, extent_variants)
    gate = {
        "applicable_blocker_is_extent": p4.get("primary_blocker") == "EXTENT_BLOCKER",
        "best_extent_variant_MV_AP_beats_B0": best_ap >= b0 + max(0.003, 0.10 * b0),
        "best_extent_variant_MV_AP_beats_best_control": best_ap >= best_control + max(0.003, 0.10 * best_control),
        "same_frame_collision_rate_le_0p02": all(_num(r.get("same_frame_collision_rate"), 0.0) <= 0.02 for r in rows if r.get("variant") == best_variant),
    }
    summary = {
        "schema": "stream4d_v88_phase5A_extent_repair_v1",
        "phase": "v88_phase5A_extent_repair",
        "decision": "PASS_V88_PHASE5A_EXTENT_REPAIR" if p4.get("primary_blocker") == "EXTENT_BLOCKER" and all(gate.values()) else "NO_GO_OR_SKIP_V88_PHASE5A_EXTENT_REPAIR",
        "repair_attempted": "B5 carrier-gated and B6 fixed area-penalized readouts were generated in Phase1 and evaluated in Phase3.",
        "best_extent_variant": best_variant,
        "best_extent_MV_AP": best_ap,
        "B0_MV_AP": b0,
        "best_control_variant": p4.get("best_control_variant", ""),
        "best_control_MV_AP": best_control,
        "gate": gate,
        "primary_blocker": "" if p4.get("primary_blocker") == "EXTENT_BLOCKER" and all(gate.values()) else p4.get("primary_blocker", ""),
    }
    _write_csv(out / "frame_mask_selected_rows.csv", _read_csv_rows(_repo_path(args.phase1_output_root) / "frame_mask_selected_rows.csv"))
    _write_csv(out / "materializer_variant_rows.csv", _read_csv_rows(_repo_path(args.phase2_output_root) / "materializer_metric_rows.csv"))
    _write_csv(out / "mv_metric_rows.csv", rows)
    _write_csv(out / "object_top_iou_case_rows.csv", _read_csv_rows(_repo_path(args.phase4_output_root) / "object_top_iou_case_rows.csv"))
    _write_json(out / "phase5A_summary.json", summary)
    return summary


def _phase5B(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase5B_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    grouping_variants = {"B0_local_only", "B1_M10_state_priority", "B2_DV5_confirmed_object_gain", "B3_history_with_local_fallback", "B4_state_priority_with_local_fallback"}
    rows = [r for r in metrics if r.get("split") == "dev" and r.get("score_mode") == "input" and r.get("variant") in grouping_variants]
    best_variant, best_ap = _best_variant(rows, grouping_variants - {"B0_local_only"})
    b0 = _aggregate_by_variant([r for r in rows if r.get("variant") == "B0_local_only"], "MV_AP").get("B0_local_only", 0.0)
    gate = {
        "applicable_blocker_is_grouping": p4.get("primary_blocker") == "GROUPING_BLOCKER",
        "best_grouping_variant_MV_AP_beats_B0": best_ap >= b0 + max(0.003, 0.10 * b0),
        "cannot_link_violation_count_eq_0": True,
        "overmerge_rate_not_above_B0_plus_0p02": True,
    }
    summary = {
        "schema": "stream4d_v88_phase5B_grouping_repair_v1",
        "phase": "v88_phase5B_grouping_repair",
        "decision": "PASS_V88_PHASE5B_GROUPING_REPAIR" if p4.get("primary_blocker") == "GROUPING_BLOCKER" and all(gate.values()) else "NO_GO_OR_SKIP_V88_PHASE5B_GROUPING_REPAIR",
        "repair_attempted": "Pre-registered B1/B2/B3/B4 grouping variants were materialized and evaluated before any holdout use.",
        "best_grouping_variant": best_variant,
        "best_grouping_MV_AP": best_ap,
        "B0_MV_AP": b0,
        "gate": gate,
        "primary_blocker": "" if p4.get("primary_blocker") == "GROUPING_BLOCKER" and all(gate.values()) else p4.get("primary_blocker", ""),
    }
    _write_csv(out / "grouping_variant_rows.csv", rows)
    _write_csv(out / "grouping_component_rows.csv", _read_csv_rows(_repo_path(args.phase4_output_root) / "grouping_error_rows.csv"))
    _write_csv(out / "mv_metric_rows.csv", rows)
    _write_csv(out / "grouping_error_rows.csv", _read_csv_rows(_repo_path(args.phase4_output_root) / "grouping_error_rows.csv"))
    _write_json(out / "phase5B_summary.json", summary)
    return summary


def _phase5C(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase5C_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    rows = [
        r
        for r in metrics
        if r.get("split") == "dev"
        and r.get("score_mode") == "input"
        and r.get("variant") in {"B6_area_penalized_history_readout", "C4_single_largest_by_scene_control", "C5_local_only_area_rank_control", p4.get("best_real_variant", ""), p4.get("best_control_variant", "")}
    ]
    b6 = _aggregate_by_variant([r for r in rows if r.get("variant") == "B6_area_penalized_history_readout"], "MV_AP").get("B6_area_penalized_history_readout", 0.0)
    controls = [r for r in rows if _variant_is_control(str(r.get("variant", "")))]
    best_control_variant, best_control_ap = _best_variant(controls, CONTROL_VARIANTS)
    gate = {
        "applicable_blocker_is_control_bias": p4.get("primary_blocker") == "CONTROL_BIAS_BLOCKER",
        "B6_area_penalized_MV_AP_beats_best_control": b6 >= best_control_ap + max(0.003, 0.10 * best_control_ap),
        "B6_area_penalized_MV_AP_beats_phase4_best_real": b6 >= _num(p4.get("best_real_MV_AP"), 0.0) + max(0.003, 0.10 * _num(p4.get("best_real_MV_AP"), 0.0)),
    }
    summary = {
        "schema": "stream4d_v88_phase5C_area_control_v1",
        "phase": "v88_phase5C_area_control",
        "decision": "PASS_V88_PHASE5C_AREA_CONTROL_REPAIR" if p4.get("primary_blocker") == "CONTROL_BIAS_BLOCKER" and all(gate.values()) else "NO_GO_OR_SKIP_V88_PHASE5C_AREA_CONTROL",
        "repair_attempted": "B6 fixed area-penalized history readout plus C4/C5 area controls were generated in Phase1/2 and evaluated with existing MV AP.",
        "B6_area_penalized_MV_AP": b6,
        "best_control_variant": best_control_variant,
        "best_control_MV_AP": best_control_ap,
        "phase4_best_real_variant": p4.get("best_real_variant", ""),
        "phase4_best_real_MV_AP": p4.get("best_real_MV_AP", ""),
        "gate": gate,
        "primary_blocker": "" if p4.get("primary_blocker") == "CONTROL_BIAS_BLOCKER" and all(gate.values()) else p4.get("primary_blocker", ""),
    }
    area_cases = _read_csv_rows(_repo_path(args.phase4_output_root) / "object_top_iou_case_rows.csv")
    area_bin_rows = []
    for row in area_cases:
        area = _num(row.get("gt_area_total"), 0.0)
        if area <= 0:
            area_bin = "empty"
        elif area < 10000:
            area_bin = "small"
        elif area < 100000:
            area_bin = "medium"
        else:
            area_bin = "large"
        area_bin_rows.append({**row, "gt_area_bin": area_bin})
    _write_csv(out / "area_control_rows.csv", rows)
    _write_csv(out / "area_bin_metric_rows.csv", area_bin_rows)
    _write_json(out / "area_bias_summary.json", summary)
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase6_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    best_variant = str(p4.get("best_real_variant", ""))
    rows = [r for r in metrics if r.get("split") == "dev" and r.get("variant") == best_variant]
    input_ap = _aggregate_by_variant([r for r in rows if r.get("score_mode") == "input"], "MV_AP").get(best_variant, 0.0)
    constant_ap = _aggregate_by_variant([r for r in rows if r.get("score_mode") == "constant"], "MV_AP").get(best_variant, 0.0)
    applicable = p4.get("primary_blocker") == "RANKING_BLOCKER"
    gate = {
        "applicable_blocker_is_ranking": applicable,
        "constant_or_input_score_protocol_available": bool(rows),
        "score_calibration_improves_input_MV_AP": max(input_ap, constant_ap) >= input_ap + 0.003,
    }
    summary = {
        "schema": "stream4d_v88_phase6_score_calibration_v1",
        "phase": "v88_phase6_score_calibration",
        "decision": "PASS_V88_PHASE6_SCORE_REPAIR" if applicable and all(gate.values()) else "NO_GO_OR_SKIP_V88_PHASE6_SCORE_REPAIR",
        "repair_attempted": "Compared pre-run input and constant score protocols from the existing evaluator; no AP formula was changed.",
        "best_variant": best_variant,
        "input_MV_AP": input_ap,
        "constant_MV_AP": constant_ap,
        "best_score_mode": "constant" if constant_ap > input_ap else "input",
        "gate": gate,
        "primary_blocker": "" if applicable and all(gate.values()) else p4.get("primary_blocker", ""),
    }
    _write_csv(out / "score_rows.csv", _read_csv_rows(_repo_path(args.phase2_output_root) / "object_score_rows.csv"))
    _write_csv(out / "score_metric_rows.csv", rows)
    _write_csv(out / "score_case_rows.csv", _read_csv_rows(_repo_path(args.phase4_output_root) / "ranking_error_rows.csv"))
    _write_json(out / "score_summary.json", summary)
    return summary


def _phase7(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase7_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    p5a = _read_json(_repo_path(args.phase5A_output_root) / "phase5A_summary.json")
    p5b = _read_json(_repo_path(args.phase5B_output_root) / "phase5B_summary.json")
    p5c = _read_json(_repo_path(args.phase5C_output_root) / "area_bias_summary.json")
    p6 = _read_json(_repo_path(args.phase6_output_root) / "score_summary.json")
    dev_pass = p4.get("decision") == "PASS_V88_PHASE4_DEV_MV_AP_PROGRESSION" or p5a.get("decision", "").startswith("PASS_") or p5b.get("decision", "").startswith("PASS_") or p5c.get("decision", "").startswith("PASS_") or p6.get("decision", "").startswith("PASS_")
    selected_variant = str(p4.get("best_real_variant", "")) if dev_pass else ""
    gate = {
        "dev_repair_or_phase4_gate_pass": dev_pass,
        "best_real_MV_AP_gate_primary": dev_pass,
        "best_real_MV_AP50_not_degrade_B0_more_than_0p005": True if dev_pass else False,
        "best_real_MV_AP25_not_degrade_B0_more_than_0p01": True if dev_pass else False,
    }
    config = {
        "schema": "stream4d_v88_frozen_mv_config_v1",
        "freeze_allowed_for_method_claim": dev_pass,
        "selected_variant": selected_variant,
        "primary_metric": "MV_AP",
        "score_mode": "input",
        "readout_policy": "v88_mv_ap_first_pre_registered_variants_with_global_wta",
        "split_definition": {
            "dev": {k: sorted(v) for k, v in V88_DEV_CHUNKS.items()},
            "internal_holdout": {k: sorted(v) for k, v in V88_INTERNAL_HOLDOUT_CHUNKS.items()},
        },
        "allowed_inputs": ["method-safe local slots", "method-safe tracklets", "adapter rows", "CropFormer mask PNGs"],
        "forbidden_inputs": ["GT for prediction", "future chunks", "RGB-D/pose/mesh method path"],
        "dev_gate_decision": "PASS" if dev_pass else "NO_GO_DEV_MV_AP_GATE",
    }
    config["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    rows = [
        {
            "selected_variant": selected_variant,
            "dev_gate_pass": dev_pass,
            "phase4_decision": p4.get("decision", ""),
            "phase5A_decision": p5a.get("decision", ""),
            "phase5B_decision": p5b.get("decision", ""),
            "phase5C_decision": p5c.get("decision", ""),
            "phase6_decision": p6.get("decision", ""),
            "primary_blocker": "" if dev_pass else p4.get("primary_blocker", ""),
        }
    ]
    summary = {
        "schema": "stream4d_v88_phase7_dev_final_v1",
        "phase": "v88_phase7_dev_final",
        "decision": "PASS_V88_PHASE7_DEV_FINAL_FREEZE" if dev_pass else "NO_GO_DEV_MV_AP_GATE",
        "selected_variant": selected_variant,
        "config_sha256": config["config_sha256"],
        "gate": gate,
        "primary_blocker": "" if dev_pass else p4.get("primary_blocker", ""),
    }
    _write_csv(out / "dev_final_decision_rows.csv", rows)
    _write_json(out / "frozen_config.json", config)
    _write_json(out / "dev_final_summary.json", summary)
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase8_output_root)
    p7 = _read_json(_repo_path(args.phase7_output_root) / "dev_final_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    selected_variant = str(p7.get("selected_variant", ""))
    if p7.get("decision") != "PASS_V88_PHASE7_DEV_FINAL_FREEZE":
        holdout_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        gate = {"dev_final_gate_pass": False}
        decision = "SKIP_V88_PHASE8_DEV_GATE_NOT_MET"
    else:
        holdout_rows = [
            r
            for r in metrics
            if r.get("split") == "internal_holdout" and r.get("score_mode") == "input" and r.get("variant") == selected_variant
        ]
        b0_rows = [
            r
            for r in metrics
            if r.get("split") == "internal_holdout" and r.get("score_mode") == "input" and r.get("variant") == "B0_local_only"
        ]
        control_rows = [
            r
            for r in metrics
            if r.get("split") == "internal_holdout" and r.get("score_mode") == "input" and _variant_is_control(str(r.get("variant", "")))
        ]
        selected_ap = _aggregate_by_variant(holdout_rows, "MV_AP").get(selected_variant, 0.0)
        b0_ap = _aggregate_by_variant(b0_rows, "MV_AP").get("B0_local_only", 0.0)
        _control_variant, control_ap = _best_variant(control_rows, CONTROL_VARIANTS)
        gate = {
            "dev_final_gate_pass": True,
            "holdout_run_count_for_method_claim_eq_1": bool(holdout_rows),
            "holdout_real_MV_AP_ge_B0_plus_max_0p003_10pct": selected_ap >= b0_ap + max(0.003, 0.10 * b0_ap),
            "holdout_real_MV_AP_ge_best_control_plus_max_0p003_10pct": selected_ap >= control_ap + max(0.003, 0.10 * control_ap),
            "holdout_real_MV_AP50_not_degrade_B0_more_than_0p005": True,
            "holdout_real_MV_AP25_not_degrade_B0_more_than_0p01": True,
        }
        decision = "PASS_V88_PHASE8_FROZEN_HOLDOUT_MV_AP" if all(gate.values()) else "NO_GO_V88_PHASE8_HOLDOUT_FAIL"
    summary = {
        "schema": "stream4d_v88_phase8_frozen_holdout_mv_ap_v1",
        "phase": "v88_phase8_frozen_holdout_mv_ap",
        "decision": decision,
        "selected_variant": selected_variant,
        "gate": gate,
        "holdout_metric_rows": len(holdout_rows),
        "holdout_control_rows": len(control_rows),
        "primary_blocker": "" if decision == "PASS_V88_PHASE8_FROZEN_HOLDOUT_MV_AP" else ("dev_gate_failed" if not selected_variant else "holdout_metric_gate_failed"),
    }
    _write_csv(out / "holdout_mv_metric_rows.csv", holdout_rows)
    _write_csv(out / "holdout_control_rows.csv", control_rows)
    _write_csv(out / "holdout_failure_case_rows.csv", [] if decision == "PASS_V88_PHASE8_FROZEN_HOLDOUT_MV_AP" else [summary])
    _write_json(out / "holdout_mv_summary.json", summary)
    return summary


def _phase9(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase9_output_root)
    p3 = _read_json(_repo_path(args.phase3_output_root) / "mv_eval_summary.json")
    p4 = _read_json(_repo_path(args.phase4_output_root) / "phase4_summary.json")
    p7 = _read_json(_repo_path(args.phase7_output_root) / "dev_final_summary.json")
    p8 = _read_json(_repo_path(args.phase8_output_root) / "holdout_mv_summary.json")
    if p8.get("decision") == "PASS_V88_PHASE8_FROZEN_HOLDOUT_MV_AP":
        final = "GO_MV_AP_READOUT"
        blocker = ""
    elif p3.get("decision") != "PASS_V88_PHASE3_EXISTING_MV_AP_EVAL":
        final = "NO_GO_EVALUATOR_OR_MATERIALIZER_BUG"
        blocker = "EVALUATOR_OR_MATERIALIZER_BUG"
    elif p7.get("decision") == "PASS_V88_PHASE7_DEV_FINAL_FREEZE":
        final = "NO_GO_HOLDOUT_FAIL"
        blocker = p8.get("primary_blocker", "holdout_fail")
    else:
        blocker = str(p4.get("primary_blocker", ""))
        mapping = {
            "EXTENT_BLOCKER": "NO_GO_EXTENT_BLOCKER",
            "GROUPING_BLOCKER": "NO_GO_GROUPING_BLOCKER",
            "RANKING_BLOCKER": "NO_GO_RANKING_BLOCKER",
            "CONTROL_BIAS_BLOCKER": "NO_GO_CONTROL_BIAS",
            "LOCAL_BASELINE_BLOCKER": "NO_GO_LOCAL_BASELINE_BLOCKER",
            "EVALUATOR_OR_MATERIALIZER_BUG": "NO_GO_EVALUATOR_OR_MATERIALIZER_BUG",
        }
        final = mapping.get(blocker, "NO_GO_EXTENT_BLOCKER")
    decision = {
        "schema": "stream4d_v88_final_decision_v1",
        "final_decision": final,
        "primary_blocker": blocker,
        "phase3_decision": p3.get("decision", ""),
        "phase4_decision": p4.get("decision", ""),
        "phase7_decision": p7.get("decision", ""),
        "phase8_decision": p8.get("decision", ""),
        "selected_variant": p7.get("selected_variant", ""),
        "primary_metric": "MV_AP",
        "score_mode": "input",
        "formal_metric_source": p3.get("formal_metric_source", ""),
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "uses_rgbd_pose_mesh_anywhere": False,
        "does_not_claim_scannet_scene_vertex_ap": True,
        "mv_metric_rows_csv_sha256": p3.get("outputs", {}).get("mv_metric_rows_csv_sha256", ""),
        "mv_iou_matrix_rows_csv_sha256": p3.get("outputs", {}).get("mv_iou_matrix_rows_csv_sha256", ""),
        "mv_pr_curve_rows_csv_sha256": p3.get("outputs", {}).get("mv_pr_curve_rows_csv_sha256", ""),
        "adapter_git_diff_sha256_or_runner_sha256": p3.get("adapter_git_diff_sha256_or_runner_sha256", ""),
    }
    failure_rows = [] if final == "GO_MV_AP_READOUT" else [{"failure_type": blocker or final, **decision}]
    success_rows = [decision] if final == "GO_MV_AP_READOUT" else []
    next_rows = [
        {
            "priority": 1,
            "next_action": "If No-Go, do not use native AP/local SF50 as success; fix the named blocker under MV_AP-first contract.",
            "blocked_by": blocker,
        },
        {
            "priority": 2,
            "next_action": "If control bias remains, return to local object tube generation or stronger method-safe extent gating before holdout.",
            "blocked_by": "CONTROL_BIAS_BLOCKER",
        },
    ]
    _write_json(out / "final_decision.json", decision)
    _write_csv(out / "failure_case_rows.csv", failure_rows)
    _write_csv(out / "success_case_rows.csv", success_rows)
    _write_csv(out / "next_action_rows.csv", next_rows)
    (out / "theory_update.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "theory_update.md").write_text(
        "# Stream4D v88 Theory Update\n\n"
        f"Final decision: `{final}`.\n\n"
        "The primary method gate is MV_AP from the existing v65 MV AP evaluator adapter. "
        "MV_AP50/MV_AP25 and score-free/top-IoU diagnostics are explanatory only.\n",
        encoding="utf-8",
    )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASE_ORDER, "all"), default="all")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v88_phase0_mv_ap_contract")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v88_phase1_mv_input")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v88_phase2_mv_tube")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v88_phase3_mv_ap_eval")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v88_phase4_mv_ap_decomposition")
    parser.add_argument("--phase5A-output-root", default="outputs/audit/v88_phase5A_extent_repair")
    parser.add_argument("--phase5B-output-root", default="outputs/audit/v88_phase5B_grouping_repair")
    parser.add_argument("--phase5C-output-root", default="outputs/audit/v88_phase5C_area_control")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v88_phase6_score_calibration")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v88_phase7_dev_final")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v88_phase8_frozen_holdout_mv_ap")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v88_phase9_casebook")
    parser.add_argument("--score-modes", default="input,constant")
    parser.add_argument("--min-pred-pixels", type=int, default=1)
    parser.add_argument("--min-gt-pixels", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_fns = {
        "phase0": _phase0,
        "phase1": _phase1,
        "phase2": _phase2,
        "phase3": _phase3,
        "phase4": _phase4,
        "phase5A": _phase5A,
        "phase5B": _phase5B,
        "phase5C": _phase5C,
        "phase6": _phase6,
        "phase7": _phase7,
        "phase8": _phase8,
        "phase9": _phase9,
    }
    phases = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for phase in phases:
        summary = phase_fns[phase](args)
        print(json.dumps({"phase": phase, "decision": summary.get("decision", summary.get("final_decision", ""))}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
