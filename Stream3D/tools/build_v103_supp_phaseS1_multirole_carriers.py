#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    SCENE_INPUTS,
    _apply_support_balanced_backfill,
    _compute_scene_arrays,
    _support_metrics,
)


PHASE_ID = "v103_supp_phaseS1_multirole_carriers"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_PHASES0_ROOT = AUDIT_ROOT / "v103_supp_phaseS0_fact_lock"
SUPPLEMENT_PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_supplement_multirole_carrier_affinity_field_plan.md"

DEFAULT_PHASE2_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}

ROLE_VARIANTS = [
    {
        "variant_id": "S1R0_anchor_strict_vis025_obj030_jit006",
        "anchor_min_visibility": 0.25,
        "anchor_min_in_image": 0.25,
        "anchor_min_object_like_rate": 0.30,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.006,
        "anchor_top_rate": 0.06,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.05,
        "support_min_in_image": 0.10,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R1_anchor_relax_vis015_obj020_jit006",
        "anchor_min_visibility": 0.15,
        "anchor_min_in_image": 0.20,
        "anchor_min_object_like_rate": 0.20,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.006,
        "anchor_top_rate": 0.08,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.03,
        "support_min_in_image": 0.08,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R2_anchor_relax_vis010_obj015_jit008",
        "anchor_min_visibility": 0.10,
        "anchor_min_in_image": 0.15,
        "anchor_min_object_like_rate": 0.15,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.008,
        "anchor_top_rate": 0.10,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.02,
        "support_min_in_image": 0.06,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R3_anchor_relax_visibility_only_vis008_obj010_jit010",
        "anchor_min_visibility": 0.08,
        "anchor_min_in_image": 0.12,
        "anchor_min_object_like_rate": 0.10,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.010,
        "anchor_top_rate": 0.12,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.01,
        "support_min_in_image": 0.05,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R4_anchor_last_resort_score_top15_no_broad_relax",
        "anchor_min_visibility": 0.05,
        "anchor_min_in_image": 0.10,
        "anchor_min_object_like_rate": 0.05,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.012,
        "anchor_top_rate": 0.15,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.005,
        "support_min_in_image": 0.04,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R5_support_visibility_gate_vis010",
        "anchor_min_visibility": 0.08,
        "anchor_min_in_image": 0.12,
        "anchor_min_object_like_rate": 0.10,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.010,
        "anchor_top_rate": 0.12,
        "support_min_visibility": 0.10,
        "support_min_confidence": 0.0,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.01,
        "support_min_in_image": 0.05,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R6_support_visibility_gate_vis015",
        "anchor_min_visibility": 0.08,
        "anchor_min_in_image": 0.12,
        "anchor_min_object_like_rate": 0.10,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.010,
        "anchor_top_rate": 0.12,
        "support_min_visibility": 0.15,
        "support_min_confidence": 0.0,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.01,
        "support_min_in_image": 0.05,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
    {
        "variant_id": "S1R7_support_visibility_core_vis040_diagnostic",
        "anchor_min_visibility": 0.08,
        "anchor_min_in_image": 0.12,
        "anchor_min_object_like_rate": 0.10,
        "anchor_max_broad": 0.15,
        "anchor_max_semantic_contradiction": 0.20,
        "anchor_max_competing": 0.15,
        "anchor_max_jitter": 0.010,
        "anchor_top_rate": 0.12,
        "support_min_visibility": 0.40,
        "support_min_confidence": 0.0,
        "support_max_broad": 0.50,
        "support_backfill_max_broad": 0.85,
        "support_min_object_like_rate": 0.01,
        "support_min_in_image": 0.05,
        "support_min_object_like_support_per_mask": 1,
        "support_min_boundary_support_per_mask": 1,
        "veto_max_safe_broad": 0.50,
        "veto_min_competing": 0.10,
        "veto_min_semantic_contradiction": 0.20,
        "veto_min_jitter": 0.012,
    },
]

ROLE_CODE = {
    "A_anchor": 1,
    "S_support": 2,
    "V_veto": 3,
    "U_uncertain": 4,
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(values: np.ndarray, mask: np.ndarray, default: float = 0.0) -> float:
    if not np.any(mask):
        return default
    return float(np.mean(np.asarray(values, dtype=np.float64)[mask]))


def _safe_percentile(values: np.ndarray, mask: np.ndarray, q: float, default: float = 0.0) -> float:
    if not np.any(mask):
        return default
    return float(np.percentile(np.asarray(values, dtype=np.float64)[mask], q))


def _weighted_semantic_rate(arrays: dict[str, np.ndarray], mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    pair_count = np.asarray(arrays["semantic_pair_count"], dtype=np.float64)[mask]
    contradiction = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)[mask]
    denom = float(np.sum(pair_count))
    if denom <= 0:
        return float(np.mean(contradiction))
    return float(np.sum(pair_count * contradiction) / denom)


def _source_distribution(arrays: dict[str, np.ndarray], mask: np.ndarray) -> str:
    if not np.any(mask):
        return "{}"
    source = np.asarray(arrays["query_source_code"], dtype=np.int16)[mask]
    labels, counts = np.unique(source, return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(labels.tolist(), counts.tolist())}, sort_keys=True)


def _per_frame_stats(arrays: dict[str, np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return 0.0, 0.0
    frame = np.asarray(arrays["src_frame"], dtype=np.int16)[mask]
    labels, counts = np.unique(frame, return_counts=True)
    if labels.size == 0:
        return 0.0, 0.0
    return float(np.percentile(counts.astype(np.float64), 10)), float(np.percentile(counts.astype(np.float64), 50))


def _anchor_score(arrays: dict[str, np.ndarray]) -> np.ndarray:
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    sem = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
    competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    source_risk = np.asarray(arrays["source_risk_score"], dtype=np.float64)
    score = np.asarray(arrays["reliability_s2"], dtype=np.float64).copy()
    score *= np.power(np.clip(1.0 - broad, 0.0, 1.0), 4.0)
    score *= np.square(np.clip(1.0 - sem, 0.0, 1.0))
    score *= np.power(np.clip(1.0 - competing, 0.0, 1.0), 4.0)
    score *= np.square(np.clip(1.0 - source_risk, 0.0, 1.0))
    score *= np.exp(-jitter / 0.0025)
    return score


def _support_score(arrays: dict[str, np.ndarray]) -> np.ndarray:
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    object_rate = np.asarray(arrays["object_like_mask_rate"], dtype=np.float64)
    source_risk = np.asarray(arrays["source_risk_score"], dtype=np.float64)
    score = np.asarray(arrays["reliability_s0"], dtype=np.float64).copy()
    score *= np.square(np.clip(1.0 - broad, 0.0, 1.0))
    score *= np.clip(object_rate, 0.0, 1.0)
    score *= np.clip(1.0 - source_risk, 0.05, 1.0)
    score *= np.exp(-jitter / 0.006)
    return score


def _top_subset(predicate: np.ndarray, score: np.ndarray, top_rate: float, min_count: int = 256) -> np.ndarray:
    predicate = np.asarray(predicate, dtype=bool)
    out = np.zeros(predicate.shape, dtype=bool)
    candidate_idx = np.flatnonzero(predicate)
    if candidate_idx.size == 0:
        return out
    target = max(int(min_count), int(round(float(top_rate) * int(predicate.shape[0]))))
    target = min(target, int(candidate_idx.size))
    if target <= 0:
        return out
    if candidate_idx.size <= target:
        out[candidate_idx] = True
        return out
    local_score = np.asarray(score[candidate_idx], dtype=np.float64)
    picked_local = np.argpartition(local_score, candidate_idx.size - target)[candidate_idx.size - target :]
    out[candidate_idx[picked_local]] = True
    return out


def _role_masks(variant: dict[str, Any], arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> dict[str, np.ndarray]:
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    sem = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
    competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    visibility = np.asarray(arrays["visibility_rate"], dtype=np.float64)
    in_image = np.asarray(arrays["in_image_rate"], dtype=np.float64)
    confidence = np.asarray(arrays["confidence_mean_in_image"], dtype=np.float64)
    object_rate = np.asarray(arrays["object_like_mask_rate"], dtype=np.float64)
    source = np.asarray(arrays["query_source_code"], dtype=np.int16)
    source_risk = np.asarray(arrays["source_risk_score"], dtype=np.float64)
    obs = np.asarray(arrays["obs_in_image_count"], dtype=np.int32)

    allowed_anchor_sources = np.isin(source, np.asarray([1, 2, 3, 7, 8], dtype=np.int16))
    anchor_pred = (
        allowed_anchor_sources
        & (obs > 0)
        & (visibility >= float(variant["anchor_min_visibility"]))
        & (in_image >= float(variant["anchor_min_in_image"]))
        & (object_rate >= float(variant["anchor_min_object_like_rate"]))
        & (broad <= float(variant["anchor_max_broad"]))
        & (sem <= float(variant["anchor_max_semantic_contradiction"]))
        & (competing <= float(variant["anchor_max_competing"]))
        & (jitter <= float(variant["anchor_max_jitter"]))
    )
    anchor_raw = _top_subset(anchor_pred, _anchor_score(arrays), float(variant["anchor_top_rate"]), min_count=256)

    support_raw = (
        (obs > 0)
        & (object_rate >= float(variant["support_min_object_like_rate"]))
        & (visibility >= float(variant.get("support_min_visibility", 0.0)))
        & (confidence >= float(variant.get("support_min_confidence", 0.0)))
        & (in_image >= float(variant["support_min_in_image"]))
        & (broad <= float(variant["support_max_broad"]))
        & (source_risk < 0.90)
    )
    support_backfill_candidate = (
        (obs > 0)
        & (object_rate >= float(variant["support_min_object_like_rate"]))
        & (visibility >= float(variant.get("support_min_visibility", 0.0)))
        & (confidence >= float(variant.get("support_min_confidence", 0.0)))
        & (in_image >= float(variant["support_min_in_image"]))
        & (broad <= float(variant.get("support_backfill_max_broad", variant["support_max_broad"])))
        & (source_risk < 0.90)
    )
    veto_raw = (
        (obs > 0)
        & (
            (broad > float(variant["veto_max_safe_broad"]))
            | (competing >= float(variant["veto_min_competing"]))
            | (sem > float(variant["veto_min_semantic_contradiction"]))
            | (jitter >= float(variant["veto_min_jitter"]))
            | (source_risk >= 0.75)
        )
    )

    final_anchor = anchor_raw & ~veto_raw
    final_veto = veto_raw & ~final_anchor
    support_seed = support_raw & ~final_anchor
    support_candidate = support_backfill_candidate & ~final_anchor
    support_filled, _, _ = _apply_support_balanced_backfill(
        diag=diag,
        scores=_support_score(arrays),
        candidate=support_candidate,
        retained=support_seed,
        min_object_like_support_per_mask=int(variant.get("support_min_object_like_support_per_mask", 0)),
        min_boundary_support_per_mask=int(variant.get("support_min_boundary_support_per_mask", 0)),
    )
    final_support = support_filled & ~final_anchor
    final_uncertain = ~(final_anchor | final_support | final_veto)
    return {
        "anchor_raw": anchor_raw,
        "support_raw": support_raw,
        "veto_raw": veto_raw,
        "A_anchor": final_anchor,
        "S_support": final_support,
        "V_veto": final_veto,
        "U_uncertain": final_uncertain,
    }


def _role_metric(scene: str, variant_id: str, role_name: str, mask: np.ndarray, arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    total = int(np.asarray(arrays["carrier_id"]).shape[0])
    support = _support_metrics(diag, mask)
    p10_frame, p50_frame = _per_frame_stats(arrays, mask)
    return {
        "schema_version": "stream4d_v103_supp_phaseS1_role_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant_id,
        "role_name": role_name,
        "carrier_count": count,
        "retained_rate": float(count / max(total, 1)),
        "source_stratum_distribution": _source_distribution(arrays, mask),
        "visibility_rate_mean": _safe_mean(arrays["visibility_rate"], mask),
        "visibility_rate_p10": _safe_percentile(arrays["visibility_rate"], mask, 10),
        "in_image_rate_mean": _safe_mean(arrays["in_image_rate"], mask),
        "in_image_rate_p10": _safe_percentile(arrays["in_image_rate"], mask, 10),
        "confidence_median": _safe_percentile(arrays["confidence_mean_in_image"], mask, 50),
        "confidence_p01": _safe_percentile(arrays["confidence_mean_in_image"], mask, 1),
        "jitter_norm_p50": _safe_percentile(arrays["normalized_jitter"], mask, 50),
        "jitter_norm_p90": _safe_percentile(arrays["normalized_jitter"], mask, 90),
        "short_range_semantic_stability_mean": _safe_mean(arrays["semantic_short_range_stability"], mask),
        "short_range_semantic_contradiction_rate": _weighted_semantic_rate(arrays, mask),
        "broad_mask_participation_rate": _safe_mean(arrays["broad_mask_participation_rate"], mask),
        "object_like_mask_participation_rate": _safe_mean(arrays["object_like_mask_rate"], mask),
        "competing_mask_conflict_rate": _safe_mean(arrays["competing_mask_conflict_rate"], mask),
        "object_like_mask_support_p10": support["object_like_mask_support_p10"],
        "object_like_mask_support_median": support["object_like_mask_support_p50"],
        "boundary_band_support_p10": support["boundary_band_support_p10"],
        "boundary_band_support_median": support["boundary_band_support_p50"],
        "mask_support_coverage_rate": support["mask_support_coverage_after_filter"],
        "per_frame_carrier_count_p10": p10_frame,
        "per_frame_carrier_count_median": p50_frame,
        "uses_gt_for_threshold": False,
        "uses_future": False,
    }


def _overlap_rows(scene: str, variant_id: str, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    total = int(masks["anchor_raw"].shape[0])
    anchor = masks["anchor_raw"]
    support = masks["support_raw"]
    veto = masks["veto_raw"]
    specs = [
        ("A_overlap_S_rate", anchor & support),
        ("A_overlap_V_rate", anchor & veto),
        ("S_overlap_V_rate", support & veto),
        ("A_exclusive_rate", anchor & ~support & ~veto),
        ("S_exclusive_rate", support & ~anchor & ~veto),
        ("V_exclusive_rate", veto & ~anchor & ~support),
    ]
    return [
        {
            "schema_version": "stream4d_v103_supp_phaseS1_role_overlap_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "variant_id": variant_id,
            "metric_name": name,
            "rate": float(np.count_nonzero(mask) / max(total, 1)),
            "count": int(np.count_nonzero(mask)),
            "total_carrier_count": total,
            "uses_gt": False,
        }
        for name, mask in specs
    ]


def _hard_gate_rows(scene: str, variant_id: str, role_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_role = {str(row["role_name"]): row for row in role_rows}
    overlap = {str(row["metric_name"]): row for row in overlap_rows}
    anchor = by_role["A_anchor"]
    support = by_role["S_support"]
    veto = by_role["V_veto"]
    gate_specs = [
        ("A_anchor_carrier_count_ge_256", int(anchor["carrier_count"]) >= 256, anchor["carrier_count"], 256),
        ("A_anchor_per_frame_carrier_count_p10_ge_4", float(anchor["per_frame_carrier_count_p10"]) >= 4.0, anchor["per_frame_carrier_count_p10"], 4.0),
        ("A_anchor_broad_mask_participation_rate_le_0p15", float(anchor["broad_mask_participation_rate"]) <= 0.15, anchor["broad_mask_participation_rate"], "<=0.15"),
        (
            "A_anchor_short_range_semantic_contradiction_rate_le_0p20",
            float(anchor["short_range_semantic_contradiction_rate"]) <= 0.20,
            anchor["short_range_semantic_contradiction_rate"],
            "<=0.20",
        ),
        ("A_anchor_competing_mask_conflict_rate_le_0p15", float(anchor["competing_mask_conflict_rate"]) <= 0.15, anchor["competing_mask_conflict_rate"], "<=0.15"),
        ("S_support_object_like_mask_support_p10_gt_0", float(support["object_like_mask_support_p10"]) > 0.0, support["object_like_mask_support_p10"], ">0"),
        ("S_support_boundary_band_support_p10_gt_0", float(support["boundary_band_support_p10"]) > 0.0, support["boundary_band_support_p10"], ">0"),
        ("S_support_mask_support_coverage_rate_ge_0p70", float(support["mask_support_coverage_rate"]) >= 0.70, support["mask_support_coverage_rate"], ">=0.70"),
        ("S_support_broad_mask_participation_rate_le_0p50", float(support["broad_mask_participation_rate"]) <= 0.50, support["broad_mask_participation_rate"], "<=0.50"),
        ("V_veto_boundary_band_support_p10_gt_0", float(veto["boundary_band_support_p10"]) > 0.0, veto["boundary_band_support_p10"], ">0"),
        (
            "V_veto_competing_mask_conflict_rate_ge_A_anchor",
            float(veto["competing_mask_conflict_rate"]) >= float(anchor["competing_mask_conflict_rate"]),
            veto["competing_mask_conflict_rate"],
            f">=A_anchor {anchor['competing_mask_conflict_rate']}",
        ),
        (
            "V_veto_broad_mask_participation_rate_ge_A_anchor",
            float(veto["broad_mask_participation_rate"]) >= float(anchor["broad_mask_participation_rate"]),
            veto["broad_mask_participation_rate"],
            f">=A_anchor {anchor['broad_mask_participation_rate']}",
        ),
        ("A_overlap_V_rate_le_0p05", float(overlap["A_overlap_V_rate"]["rate"]) <= 0.05, overlap["A_overlap_V_rate"]["rate"], "<=0.05"),
    ]
    rows: list[dict[str, Any]] = []
    for gate_name, ok, observed, required in gate_specs:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS1_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_variant_id": variant_id,
                "gate_name": gate_name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
                "uses_gt": False,
            }
        )
    return rows


def _gate_score(gate_rows: list[dict[str, Any]]) -> tuple[int, float]:
    ok_count = sum(1 for row in gate_rows if bool(row["pass"]))
    margin = 0.0
    for row in gate_rows:
        if bool(row["pass"]):
            margin += 1.0
    return ok_count, margin


def _role_row_by_name(role_rows: list[dict[str, Any]], role_name: str) -> dict[str, Any]:
    for row in role_rows:
        if str(row.get("role_name", "")) == role_name:
            return row
    return {}


def _selection_score(role_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> tuple[float, ...]:
    """Choose A_anchor as a precision anchor; let S_support carry coverage.

    A previous coverage-first tie-break selected broad last-resort anchor variants
    that increased A coverage but later lost clean DA3 induction. This score keeps
    the plan's role semantics sharper: A_anchor should be low-risk first, while
    mask coverage remains a tie-breaker after risk and per-frame viability.
    """
    anchor = _role_row_by_name(role_rows, "A_anchor")
    support = _role_row_by_name(role_rows, "S_support")
    ok_count, margin = _gate_score(gate_rows)
    a_overlap_v = 1.0
    for row in gate_rows:
        if str(row.get("gate_name", "")) == "A_overlap_V_rate_le_0p05":
            try:
                a_overlap_v = float(row.get("observed", 1.0))
            except Exception:
                a_overlap_v = 1.0
            break
    return (
        float(ok_count),
        float(margin),
        -float(a_overlap_v),
        -float(anchor.get("broad_mask_participation_rate", 1.0)),
        -float(anchor.get("short_range_semantic_contradiction_rate", 1.0)),
        -float(anchor.get("competing_mask_conflict_rate", 1.0)),
        -float(anchor.get("jitter_norm_p90", 1.0)),
        float(anchor.get("short_range_semantic_stability_mean", 0.0)),
        float(anchor.get("object_like_mask_participation_rate", 0.0)),
        float(anchor.get("per_frame_carrier_count_p10", 0.0)),
        float(anchor.get("mask_support_coverage_rate", 0.0)),
        float(anchor.get("object_like_mask_support_median", 0.0)),
        float(anchor.get("carrier_count", 0.0)),
        float(support.get("object_like_mask_support_p10", 0.0)),
    )


def _carrier_role_frame(scene: str, variant_id: str, arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    n = int(np.asarray(arrays["carrier_id"]).shape[0])
    role_name = np.full((n,), "U_uncertain", dtype=object)
    role_code = np.full((n,), ROLE_CODE["U_uncertain"], dtype=np.int8)
    for name in ["A_anchor", "S_support", "V_veto"]:
        role_name[masks[name]] = name
        role_code[masks[name]] = ROLE_CODE[name]
    return pd.DataFrame(
        {
            "schema_version": "stream4d_v103_supp_phaseS1_carrier_role_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "selected_variant_id": variant_id,
            "carrier_id": np.asarray(arrays["carrier_id"], dtype=np.int64),
            "query_source_code": np.asarray(arrays["query_source_code"], dtype=np.int16),
            "src_frame": np.asarray(arrays["src_frame"], dtype=np.int16),
            "src_frame_global": np.asarray(arrays["src_frame_global"], dtype=np.int32),
            "role_name": role_name,
            "role_code": role_code,
            "anchor_raw": masks["anchor_raw"],
            "support_raw": masks["support_raw"],
            "veto_raw": masks["veto_raw"],
            "is_A_anchor": masks["A_anchor"],
            "is_S_support": masks["S_support"],
            "is_V_veto": masks["V_veto"],
            "is_U_uncertain": masks["U_uncertain"],
            "r_geo": np.asarray(arrays["r_geo"], dtype=np.float32),
            "r_mask": np.asarray(arrays["r_mask"], dtype=np.float32),
            "r_sem": np.asarray(arrays["r_sem"], dtype=np.float32),
            "reliability_s0": np.asarray(arrays["reliability_s0"], dtype=np.float32),
            "reliability_s2": np.asarray(arrays["reliability_s2"], dtype=np.float32),
            "visibility_rate": np.asarray(arrays["visibility_rate"], dtype=np.float32),
            "in_image_rate": np.asarray(arrays["in_image_rate"], dtype=np.float32),
            "confidence_mean_in_image": np.asarray(arrays["confidence_mean_in_image"], dtype=np.float32),
            "normalized_jitter": np.asarray(arrays["normalized_jitter"], dtype=np.float32),
            "broad_mask_participation_rate": np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float32),
            "object_like_mask_rate": np.asarray(arrays["object_like_mask_rate"], dtype=np.float32),
            "competing_mask_conflict_rate": np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float32),
            "semantic_short_range_stability": np.asarray(arrays["semantic_short_range_stability"], dtype=np.float32),
            "semantic_contradiction_rate": np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float32),
            "source_risk_score": np.asarray(arrays["source_risk_score"], dtype=np.float32),
            "uses_gt_for_threshold": False,
            "uses_future": False,
            "support_role_allows_veto_overlap": True,
        }
    )


def _gt_diagnostic_rows(selected_by_scene: dict[str, str]) -> list[dict[str, Any]]:
    metrics = [
        "carrier_GT_purity_mean",
        "carrier_multi_GT_rate",
        "same_object_bridge_recall_at_fixed_tau",
        "same_semantic_different_object_false_bridge_rate",
        "carrier_filter_precision_proxy",
        "carrier_filter_recall_proxy",
    ]
    rows: list[dict[str, Any]] = []
    for scene, variant_id in selected_by_scene.items():
        for metric in metrics:
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS1_gt_diagnostic_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "selected_variant_id": variant_id,
                    "metric_name": metric,
                    "status": "not_run",
                    "value": "",
                    "uses_gt_for_threshold": False,
                    "note": "GT diagnostic intentionally not computed in this first S1 role-separation artifact; method gates above are GT-free.",
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    phaseS0_root = _project(args.phaseS0_root)
    phaseS0_summary = _read_json(phaseS0_root / "summary.json")
    if phaseS0_summary.get("decision") != "PASS_ENTER_PHASES1_MULTIROLE_CARRIER_SETS":
        raise RuntimeError(f"Phase S0 has not passed: {phaseS0_root / 'summary.json'}")

    output_root.mkdir(parents=True, exist_ok=True)
    all_role_metric_rows: list[dict[str, Any]] = []
    all_variant_metric_rows: list[dict[str, Any]] = []
    all_overlap_rows: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    selected_by_scene: dict[str, str] = {}
    carrier_frames: list[pd.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []

    for scene in ["scene0011_00", "scene0050_00"]:
        scene_t0 = time.time()
        spec = dict(SCENE_INPUTS[scene])
        spec["phase2_root"] = DEFAULT_PHASE2_ROOT_BY_SCENE[scene]
        diag, _, _, arrays = _compute_scene_arrays(scene, spec, output_root, int(args.cupy_device_id))

        candidate_records: list[tuple[tuple[int, float], str, dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = []
        for variant in ROLE_VARIANTS:
            masks = _role_masks(variant, arrays, diag)
            role_rows = [
                _role_metric(scene, str(variant["variant_id"]), role_name, masks[role_name], arrays, diag)
                for role_name in ["A_anchor", "S_support", "V_veto", "U_uncertain"]
            ]
            overlap_rows = _overlap_rows(scene, str(variant["variant_id"]), masks)
            gate_rows = _hard_gate_rows(scene, str(variant["variant_id"]), role_rows, overlap_rows)
            candidate_records.append((_gate_score(gate_rows), str(variant["variant_id"]), masks, role_rows, overlap_rows, gate_rows))
            all_variant_metric_rows.extend(role_rows)

        force_variant = str(getattr(args, "force_selected_variant", "")).strip()
        if force_variant:
            forced = [record for record in candidate_records if record[1] == force_variant]
            if not forced:
                known = ", ".join(record[1] for record in candidate_records)
                raise RuntimeError(f"--force-selected-variant={force_variant} not found; known variants: {known}")
            selected = forced[0]
        else:
            passing = [record for record in candidate_records if all(bool(row["pass"]) for row in record[5])]
            if passing:
                selected = max(passing, key=lambda r: _selection_score(r[3], r[5]))
            else:
                selected = max(candidate_records, key=lambda r: _selection_score(r[3], r[5]))
        _, selected_variant_id, selected_masks, selected_role_rows, selected_overlap_rows, selected_gate_rows = selected
        selected_by_scene[scene] = selected_variant_id
        all_role_metric_rows.extend(selected_role_rows)
        all_overlap_rows.extend(selected_overlap_rows)
        all_gate_rows.extend(selected_gate_rows)
        for gate in selected_gate_rows:
            if not bool(gate["pass"]):
                all_failures.append(
                    {
                        "schema_version": "stream4d_v103_supp_phaseS1_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": gate["gate_name"],
                        "severity": "blocking",
                        "evidence": f"selected_variant={selected_variant_id} observed={gate['observed']} required={gate['required']}",
                        "repair_direction": (
                            "Adjust carrier role separation, not single reliable thresholds: if A_anchor is sparse, relax visibility/semantic-stability; "
                            "if S_support is broad dominated, strengthen object-like guard; if V_veto lacks coverage, add boundary/competing/high-risk veto construction."
                        ),
                    }
                )
        carrier_frames.append(_carrier_role_frame(scene, selected_variant_id, arrays, selected_masks))
        performance_rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS1_performance_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_variant_id": selected_variant_id,
                "runtime_sec": time.time() - scene_t0,
                "projection_backend": diag["performance"].get("projection_backend", ""),
                "semantic_backend": diag["performance"].get("semantic_backend", ""),
                "cache_dir": diag["performance"].get("cache_dir", ""),
                "uses_gt_for_threshold": False,
            }
        )

    carrier_roles = pd.concat(carrier_frames, ignore_index=True)
    carrier_role_path = output_root / "carrier_role_rows.parquet"
    carrier_roles.to_parquet(carrier_role_path, index=False)

    role_metric_path = output_root / "carrier_role_metric_rows.csv"
    variant_metric_path = output_root / "variant_role_metric_rows.csv"
    overlap_path = output_root / "role_overlap_rows.csv"
    gt_diag_path = output_root / "gt_diagnostic_rows.csv"
    gate_path = output_root / "gate_rows.csv"
    failure_path = output_root / "failure_rows.csv"
    performance_path = output_root / "performance_rows.csv"
    summary_path = output_root / "summary.json"

    gt_rows = _gt_diagnostic_rows(selected_by_scene)
    _write_csv(role_metric_path, all_role_metric_rows)
    _write_csv(variant_metric_path, all_variant_metric_rows)
    _write_csv(overlap_path, all_overlap_rows)
    _write_csv(gt_diag_path, gt_rows)
    _write_csv(gate_path, all_gate_rows)
    _write_csv(failure_path, all_failures)
    _write_csv(performance_path, performance_rows)

    phaseS1_pass = not all_failures
    decision = "PASS_ENTER_PHASES2_ROLE_AWARE_AFFINITY" if phaseS1_pass else "NO_GO_REPAIR_PHASES1_MULTIROLE_CARRIER_SETS"
    role_counts = [
        {
            "scene_id": row["scene_id"],
            "role_name": row["role_name"],
            "carrier_count": row["carrier_count"],
            "count_semantics": "role_mask_multihot_count",
        }
        for row in all_role_metric_rows
    ]
    summary = {
        "schema_version": "stream4d_v103_supp_phaseS1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": decision,
        "phaseS1_pass": bool(phaseS1_pass),
        "failure_count": len(all_failures),
        "selected_variant_by_scene": selected_by_scene,
        "role_counts": role_counts,
        "variant_count": len(ROLE_VARIANTS),
        "force_selected_variant": str(getattr(args, "force_selected_variant", "")).strip(),
        "phaseS0_root": _rel(phaseS0_root),
        "provider": "D4RT48Mix maskbalanced8 qchunk16384 chunk_index1 cap24576",
        "uses_gt_for_threshold": False,
        "uses_future": False,
        "gt_diagnostic_status": "not_run_for_threshold_free_S1",
        "outputs": {
            "summary": _rel(summary_path),
            "carrier_role_rows": _rel(carrier_role_path),
            "carrier_role_metric_rows": _rel(role_metric_path),
            "variant_role_metric_rows": _rel(variant_metric_path),
            "role_overlap_rows": _rel(overlap_path),
            "gt_diagnostic_rows": _rel(gt_diag_path),
            "gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
            "performance_rows": _rel(performance_path),
        },
        "truthfulness_note": (
            "Phase S1 separates D4RT carriers into anchor/support/veto/uncertain roles using GT-free Phase3 reliability arrays. "
            "S_support is intentionally multi-hot and can overlap V_veto; A_anchor is kept low-overlap with V_veto. "
            "It does not run AP and does not compute GT diagnostics in this first artifact, so no purity or false-bridge success is claimed."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS0-root", default=str(DEFAULT_PHASES0_ROOT))
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--force-selected-variant", default="")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    raise SystemExit(0 if summary["phaseS1_pass"] else 2)


if __name__ == "__main__":
    main()
