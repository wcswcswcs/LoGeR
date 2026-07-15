#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r6_phaseR6_1_edge_attribution_casebook"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase1_edge_attribution_casebook"
DEFAULT_R6_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_ANCHOR_ONLY_ROOT = AUDIT_ROOT / "v103_supp_r5_anchor_only_local_ap_diag"
DEFAULT_CURRENT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"
DEFAULT_R5_EDGE_ATTR_ROOT = AUDIT_ROOT / "v103_supp_r5_support_edge_attribution"
DEFAULT_R5_GT_COVERAGE_ROOT = AUDIT_ROOT / "v103_supp_r5_gt_coverage"

D9 = "D9_affinity_merge_tau065_top1_broad_support_veto"
D0 = "D0_f2_original_replay"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: "" if v is None else str(v))
    df.to_parquet(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _maybe_num(value: Any) -> float | str:
    out = _num(value, np.nan)
    if np.isfinite(out):
        return out
    return ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _run_roots(local_ap_root: Path, anchor_only_root: Path, current_root: Path) -> list[dict[str, Any]]:
    roots = [
        {
            "source_root_role": "current_phase6d_lock",
            "r5_feature_variant_id": "current_phase6d_locked",
            "root": current_root,
            "prediction_family": "current_locked",
        }
    ]
    for root, role, family in [
        (anchor_only_root, "anchor_only_r5_4_diag", "anchor_only"),
        (local_ap_root, "support_weighted_r5_4_diag", "support_weighted"),
    ]:
        run_parent = root / "phase6d_runs"
        if not run_parent.exists():
            continue
        for child in sorted(run_parent.iterdir()):
            if child.is_dir():
                roots.append(
                    {
                        "source_root_role": role,
                        "r5_feature_variant_id": child.name,
                        "root": child,
                        "prediction_family": family,
                    }
                )
    return roots


def _feature_summary_map(feature_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    df = _read_csv(feature_root / "role_feature_summary_rows.csv")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, rec in df.iterrows():
        out[(str(rec.get("variant_id", "")), str(rec.get("scene_id", "")))] = rec.to_dict()
    return out


def _gt_diag_map(gt_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    df = _read_csv(gt_root / "three_d_inconsistency_summary_rows.csv")
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if df.empty:
        return out
    tau_df = df[np.isclose(df["tau"].astype(float), 0.05)] if "tau" in df.columns else df
    for (r5_variant, phase6d_variant, scene), g in tau_df.groupby(
        ["r5_feature_variant_id", "phase6d_variant_id", "scene_id"], dropna=False
    ):
        out[(str(r5_variant), str(phase6d_variant), str(scene))] = {
            "same_GT_connection_tau005": float(g["same_GT_mask_pair_connection_rate"].mean())
            if "same_GT_mask_pair_connection_rate" in g.columns
            else "",
            "false_connection_tau005": float(g["same_semantic_diff_GT_false_connection_rate"].max())
            if "same_semantic_diff_GT_false_connection_rate" in g.columns
            else "",
            "fragment_ge2_tau005": float(g["GT_fragment_count_ge2_rate"].mean())
            if "GT_fragment_count_ge2_rate" in g.columns
            else "",
            "GT_fragment_count_ge2_rate": float(g["GT_fragment_count_ge2_rate"].mean())
            if "GT_fragment_count_ge2_rate" in g.columns
            else "",
            "best_pred_IoU_mean": float(g["best_pred_IoU_mean"].mean()) if "best_pred_IoU_mean" in g.columns else "",
            "union_pred_IoU_mean": float(g["union_pred_IoU_mean"].mean()) if "union_pred_IoU_mean" in g.columns else "",
            "union_minus_best_IoU_mean": float(g["union_minus_best_IoU_mean"].mean())
            if "union_minus_best_IoU_mean" in g.columns
            else "",
        }
    return out


def _gt_key_for_metric(r5_variant: str, phase6d_variant: str) -> str:
    if r5_variant == "current_phase6d_locked":
        return "current_phase6d_locked"
    return r5_variant


def _edge_family(data: dict[str, Any], r5_variant: str) -> str:
    phase6d_variant = str(data.get("variant_id", ""))
    if _bool(data.get("shuffle_affinity", False)) or phase6d_variant.startswith("R"):
        return "shuffled_affinity_control"
    if int(_num(data.get("direct_pair_support_min_count"), 0)) > 0:
        return "DA3_direct_pair_guard"
    if _bool(data.get("broad_support_veto", False)):
        return "broad_support_veto_guarded"
    if r5_variant == "F0_anchor_only":
        return "anchor_only_affinity"
    if r5_variant.startswith("F"):
        return "support_weighted_affinity"
    return "current_primitive_affinity"


def _edge_rows(
    roots: list[dict[str, Any]],
    feature_summary: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in roots:
        root = Path(info["root"])
        df = _read_csv(root / "merge_edge_rows.csv")
        if df.empty:
            continue
        r5_variant = str(info["r5_feature_variant_id"])
        for _, rec in df.iterrows():
            data = rec.to_dict()
            scene = str(data.get("scene_id", ""))
            fs = feature_summary.get((r5_variant, scene), {})
            phase6d_variant = str(data.get("variant_id", ""))
            edge_rank = int(_num(data.get("edge_rank"), -1))
            support_count = int(_num(data.get("direct_pair_support_count"), 0))
            same_gt_count = int(_num(data.get("direct_pair_diagnostic_same_gt_count"), 0))
            diff_gt_count = int(_num(data.get("direct_pair_diagnostic_different_gt_count"), 0))
            support_feature_enabled = r5_variant.startswith("F") and r5_variant != "F0_anchor_only"
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_1_edge_attribution_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "chunk_id": "c0001",
                    "edge_id": f"{r5_variant}:{phase6d_variant}:{scene}:{edge_rank}",
                    "mask_or_group_a": str(data.get("object_a", "")),
                    "mask_or_group_b": str(data.get("object_b", "")),
                    "variant_id": f"{r5_variant}/{phase6d_variant}",
                    "r5_feature_variant_id": r5_variant,
                    "phase6d_variant_id": phase6d_variant,
                    "prediction_family": info["prediction_family"],
                    "accepted": _bool(data.get("accepted_union", False)),
                    "edge_score": _num(data.get("affinity"), 0.0),
                    "edge_family": _edge_family(data, r5_variant),
                    "has_A_anchor": r5_variant in {"current_phase6d_locked", "F0_anchor_only"} or r5_variant.startswith("F"),
                    "A_anchor_count": "not_available_edge_level",
                    "A_anchor_reliability_mean": "not_available_edge_level",
                    "S_support_count": support_count,
                    "S_support_weighted_mass": _maybe_num(fs.get("support_contribution_ratio", "")),
                    "S_support_cosine": _num(data.get("affinity"), 0.0) if support_feature_enabled else "",
                    "S_support_veto_overlap_rate": _maybe_num(fs.get("veto_overlap_contribution_ratio", "")),
                    "S_support_broad_rate": _maybe_num(fs.get("broad_contribution_ratio", "")),
                    "support_specificity_mean": _maybe_num(fs.get("mask_weight_specificity_alpha", "")),
                    "support_semantic_calibrated_mean": _maybe_num(fs.get("support_semantic_weight_mean", "")),
                    "has_F2_skeleton_edge": True,
                    "has_DA3_direct_pair": support_count > 0,
                    "V_veto_score": _num(data.get("direct_pair_broad_risk_max"), 0.0),
                    "broad_support_veto_score": 1.0 if _bool(data.get("broad_support_veto", False)) else 0.0,
                    "semantic_calibrated_similarity": _maybe_num(fs.get("support_semantic_weight_mean", ""))
                    if "semantic" in r5_variant
                    else "",
                    "same_frame_competing_conflict": _bool(data.get("specific_conflict", False)),
                    "would_create_pixel_collision": "not_available_edge_level",
                    "diagnostic_same_GT": same_gt_count > 0 and same_gt_count >= diff_gt_count,
                    "diagnostic_diff_GT": diff_gt_count > 0,
                    "diagnostic_same_semantic_diff_GT": "not_available_edge_level",
                    "diagnostic_same_GT_count": same_gt_count,
                    "diagnostic_diff_GT_count": diff_gt_count,
                    "diagnostic_same_GT_rate": _num(data.get("direct_pair_diagnostic_same_gt_rate"), 0.0),
                    "diagnostic_diff_GT_rate": _num(data.get("direct_pair_diagnostic_different_gt_rate"), 0.0),
                    "reject_reason": str(data.get("reject_reason", "")),
                    "source_phase6d_root": _rel(root),
                    "source_field_note": "A_anchor_count and pixel-collision edge counter are not present in Phase6d edge rows; support counts are direct-pair diagnostic fields.",
                    "uses_gt_for_prediction": _bool(data.get("uses_gt_for_prediction", False)),
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
    return rows


def _metric_rows(
    roots: list[dict[str, Any]],
    gt_diag: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in roots:
        root = Path(info["root"])
        df = _read_csv(root / "merge_metric_rows.csv")
        if df.empty:
            continue
        r5_variant = str(info["r5_feature_variant_id"])
        for _, rec in df.iterrows():
            data = rec.to_dict()
            phase6d_variant = str(data.get("variant_id", ""))
            scene_diags = [
                v
                for (rv, pv, _scene), v in gt_diag.items()
                if rv == _gt_key_for_metric(r5_variant, phase6d_variant) and pv == phase6d_variant
            ]
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_1_leave_one_family_out_metric_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": f"{r5_variant}/{phase6d_variant}",
                    "r5_feature_variant_id": r5_variant,
                    "phase6d_variant_id": phase6d_variant,
                    "leave_one_family_variant": _map_available_family_variant(r5_variant, phase6d_variant),
                    "evidence_status": "observed_existing_phase6d_metric",
                    "MV_AP_window": _num(data.get("MV_AP_window"), 0.0),
                    "MV_AP50_window": _num(data.get("MV_AP50_window"), 0.0),
                    "MV_AP25_window": _num(data.get("MV_AP25_window"), 0.0),
                    "ScoreFreeMatch50_window": _num(data.get("ScoreFreeMatch50_window"), 0.0),
                    "accepted_edge_count": int(_num(data.get("accepted_merge_count"), 0)),
                    "accepted_diff_gt_edge_count_diagnostic": "",
                    "same_GT_connection_tau005": _mean_diag(scene_diags, "same_GT_connection_tau005"),
                    "false_connection_tau005": _max_diag(scene_diags, "false_connection_tau005"),
                    "fragment_ge2_tau005": _mean_diag(scene_diags, "fragment_ge2_tau005"),
                    "same_frame_collision_count": int(_num(data.get("same_frame_collision_count"), 0)),
                    "pixel_collision_rate": _num(data.get("pixel_collision_rate"), 0.0),
                    "missing_mask_raster_count": int(_num(data.get("missing_mask_raster_count"), 0)),
                    "real_minus_shuffled_MV_AP_window": "",
                    "dataset_split": str(data.get("dataset_split", "")),
                    "chunk_id": str(data.get("chunk_id", "")),
                    "source_phase6d_root": _rel(root),
                    "uses_gt_for_prediction": _bool(data.get("uses_gt_for_prediction", False)),
                    "uses_gt_for_eval": True,
                    "uses_future": _bool(data.get("uses_future", False)),
                }
            )
    return rows


def _mean_diag(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [_num(row.get(key), np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else ""


def _max_diag(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [_num(row.get(key), np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.max(vals)) if vals else ""


def _map_available_family_variant(r5_variant: str, phase6d_variant: str) -> str:
    if r5_variant == "current_phase6d_locked" and phase6d_variant == D9:
        return "D9_current_locked_no_removal"
    if phase6d_variant == D0:
        return "D9_minus_all_affinity_edges_proxy"
    if r5_variant == "F0_anchor_only" and phase6d_variant == D9:
        return "D9_minus_S_support_compatibility_proxy"
    if r5_variant == "F2_anchor_plus_support_020" and phase6d_variant == D9:
        return "R5_F2_support_channel_present"
    if r5_variant == "F4_anchor_plus_semantic_filtered_support_020" and phase6d_variant == D9:
        return "R5_F4_semantic_support_filter_present"
    if phase6d_variant.startswith("R"):
        return "shuffled_affinity_control"
    if "direct_pair" in phase6d_variant:
        return "DA3_direct_pair_guard_present"
    if "broad_support_veto" in phase6d_variant:
        return "broad_support_veto_present"
    return "available_metric_variant"


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r["r5_feature_variant_id"]), str(r["phase6d_variant_id"])): r for r in rows}


def _accepted_diff_by_variant(edge_rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in edge_rows:
        if bool(row["accepted"]) and bool(row["diagnostic_diff_GT"]):
            key = (str(row["r5_feature_variant_id"]), str(row["phase6d_variant_id"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _copy_metric_for_removal(
    row: dict[str, Any] | None,
    removal_id: str,
    evidence_status: str,
    source_interpretation: str,
    accepted_diff_counts: dict[tuple[str, str], int],
    real_minus_shuffled: Any = "",
) -> dict[str, Any]:
    rms_num = _num(real_minus_shuffled, np.nan)
    if isinstance(real_minus_shuffled, float) and not np.isfinite(real_minus_shuffled):
        real_minus_shuffled = ""
    elif str(real_minus_shuffled).lower() == "nan":
        real_minus_shuffled = ""
    elif isinstance(real_minus_shuffled, (int, float, np.number)) and np.isfinite(rms_num):
        real_minus_shuffled = float(rms_num)
    base = {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_1_leave_one_family_out_metric_row_v1",
        "phase_id": PHASE_ID,
        "leave_one_family_variant": removal_id,
        "evidence_status": evidence_status,
        "source_interpretation": source_interpretation,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }
    if not row:
        base.update(
            {
                "variant_id": "",
                "r5_feature_variant_id": "",
                "phase6d_variant_id": "",
                "MV_AP_window": "",
                "MV_AP50_window": "",
                "MV_AP25_window": "",
                "ScoreFreeMatch50_window": "",
                "accepted_edge_count": "",
                "accepted_diff_gt_edge_count_diagnostic": "",
                "same_GT_connection_tau005": "",
                "false_connection_tau005": "",
                "fragment_ge2_tau005": "",
                "same_frame_collision_count": "",
                "pixel_collision_rate": "",
                "missing_mask_raster_count": "",
                "real_minus_shuffled_MV_AP_window": "",
            }
        )
        return base
    key = (str(row.get("r5_feature_variant_id", "")), str(row.get("phase6d_variant_id", "")))
    base.update(row)
    base["schema_version"] = "stream4d_v103_supp_r6_phaseR6_1_leave_one_family_out_metric_row_v1"
    base["phase_id"] = PHASE_ID
    base["leave_one_family_variant"] = removal_id
    base["evidence_status"] = evidence_status
    base["source_interpretation"] = source_interpretation
    base["accepted_diff_gt_edge_count_diagnostic"] = accepted_diff_counts.get(key, "")
    base["real_minus_shuffled_MV_AP_window"] = real_minus_shuffled
    return base


def _canonical_removal_rows(
    metric_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    r5_compare_root: Path,
) -> list[dict[str, Any]]:
    lookup = _metric_lookup(metric_rows)
    diff_counts = _accepted_diff_by_variant(edge_rows)
    compare_df = _read_csv(r5_compare_root / "support_family_compare_rows.csv")
    real_minus_shuffled: dict[tuple[str, str], Any] = {}
    if not compare_df.empty:
        for _, rec in compare_df.iterrows():
            key = (str(rec.get("r5_feature_variant_id", "")), str(rec.get("phase6d_variant_id", "")))
            if rec.get("real_minus_shuffled_MV_AP_window", "") != "":
                real_minus_shuffled[key] = rec.get("real_minus_shuffled_MV_AP_window", "")

    rows = [
        _copy_metric_for_removal(
            lookup.get(("current_phase6d_locked", D9)),
            "D9_current_locked_no_removal",
            "observed_existing_phase6d_metric",
            "Current locked D9 reference.",
            diff_counts,
            real_minus_shuffled.get(("current_phase6d_locked_D9", D9), real_minus_shuffled.get(("current_phase6d_locked", D9), "")),
        ),
        _copy_metric_for_removal(
            None,
            "D9_minus_A_anchor_edges",
            "not_available_exact_intervention",
            "No support-only local AP intervention is available in current artifacts; do not infer this metric.",
            diff_counts,
        ),
        _copy_metric_for_removal(
            lookup.get(("F0_anchor_only", D9)),
            "D9_minus_S_support_compatibility",
            "proxy_from_F0_anchor_only_D9",
            "F0 anchor-only D9 is the closest existing proxy for removing support feature compatibility.",
            diff_counts,
            real_minus_shuffled.get(("F0_anchor_only", D9), ""),
        ),
        _copy_metric_for_removal(
            lookup.get(("current_phase6d_locked", "D5_affinity_merge_tau075_top1_specific_veto")),
            "D9_minus_broad_support_veto",
            "proxy_threshold_changed",
            "D5 specific-veto is available but changes threshold from tau065 to tau075; treat as proxy only.",
            diff_counts,
        ),
        _copy_metric_for_removal(
            lookup.get(("F2_anchor_plus_support_020", D9)),
            "D9_minus_semantic_calibration",
            "proxy_from_F2_nonsemantic_support",
            "F2 is nonsemantic support-weighted; compare against F4 semantic-filtered support externally.",
            diff_counts,
            real_minus_shuffled.get(("F2_anchor_plus_support_020", D9), ""),
        ),
        _copy_metric_for_removal(
            None,
            "D9_minus_DA3_direct_pair",
            "not_available_exact_intervention",
            "Current D9 has direct_pair_support_min_count=0; exact minus-DA3 direct-pair intervention not present.",
            diff_counts,
        ),
        _copy_metric_for_removal(
            lookup.get(("current_phase6d_locked", D0)),
            "D9_minus_skeleton_edges",
            "not_equivalent_proxy_D0_replay",
            "D0 replay removes affinity merges, not the F2 skeleton itself; use only as lower-bound reference.",
            diff_counts,
        ),
        _copy_metric_for_removal(
            lookup.get(("F0_anchor_only", D9)),
            "R5_F2_minus_support_channel",
            "proxy_from_F0_anchor_only_D9",
            "F0 is the existing anchor-only proxy used to estimate support channel contribution against F2.",
            diff_counts,
            real_minus_shuffled.get(("F0_anchor_only", D9), ""),
        ),
        _copy_metric_for_removal(
            lookup.get(("F2_anchor_plus_support_020", D9)),
            "R5_F4_minus_semantic_support_filter",
            "proxy_from_F2_nonsemantic_support",
            "F2 is the existing nonsemantic support proxy for removing F4 semantic support filter.",
            diff_counts,
            real_minus_shuffled.get(("F2_anchor_plus_support_020", D9), ""),
        ),
    ]
    return rows


def _family_summary_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not edge_rows:
        return []
    df = pd.DataFrame(edge_rows)
    rows: list[dict[str, Any]] = []
    group_cols = ["r5_feature_variant_id", "phase6d_variant_id", "edge_family", "scene_id"]
    for keys, g in df.groupby(group_cols, dropna=False):
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r6_phaseR6_1_edge_family_summary_row_v1",
                "phase_id": PHASE_ID,
                "r5_feature_variant_id": keys[0],
                "phase6d_variant_id": keys[1],
                "edge_family": keys[2],
                "scene_id": keys[3],
                "edge_count": int(len(g)),
                "accepted_edge_count": int(g["accepted"].astype(bool).sum()),
                "diagnostic_diff_GT_edge_count": int(g["diagnostic_diff_GT"].astype(bool).sum()),
                "accepted_diagnostic_diff_GT_edge_count": int((g["accepted"].astype(bool) & g["diagnostic_diff_GT"].astype(bool)).sum()),
                "diagnostic_same_GT_edge_count": int(g["diagnostic_same_GT"].astype(bool).sum()),
                "S_support_count_mean": float(pd.to_numeric(g["S_support_count"], errors="coerce").fillna(0).mean()),
                "edge_score_mean": float(pd.to_numeric(g["edge_score"], errors="coerce").fillna(0).mean()),
                "V_veto_score_mean": float(pd.to_numeric(g["V_veto_score"], errors="coerce").fillna(0).mean()),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    return rows


def _casebook_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in edge_rows:
        reason = ""
        if bool(row.get("accepted")) and bool(row.get("diagnostic_diff_GT")):
            reason = "accepted_diff_GT_false_bridge_diagnostic"
        elif bool(row.get("accepted")) and float(_num(row.get("V_veto_score"), 0.0)) >= 0.7:
            reason = "accepted_high_broad_or_veto_risk_edge"
        elif bool(row.get("diagnostic_diff_GT")):
            reason = "candidate_diff_GT_diagnostic"
        if not reason:
            continue
        out = {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_1_edge_failure_casebook_row_v1",
            "phase_id": PHASE_ID,
            "case_reason": reason,
        }
        out.update(row)
        rows.append(out)
    rows.sort(
        key=lambda r: (
            0 if r["case_reason"] == "accepted_diff_GT_false_bridge_diagnostic" else 1,
            str(r.get("scene_id", "")),
            str(r.get("variant_id", "")),
            -float(_num(r.get("edge_score"), 0.0)),
        )
    )
    return rows[:500]


def _classify_support_mode(compare_root: Path, edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compare = _read_csv(compare_root / "support_family_compare_rows.csv")
    result = {
        "support_contribution_mode": "insufficient_evidence",
        "support_has_attributable_signal": False,
        "support_feature_hurts_replay": False,
        "support_improves_over_anchor_only": False,
        "false_bridge_risk_nonzero": False,
        "recommended_next_phase": "R6_1_casebook_review",
    }
    if compare.empty:
        return result
    classifications = set(compare.get("classification", pd.Series(dtype=str)).astype(str).tolist())
    result["support_feature_hurts_replay"] = "support_weighted_feature_hurts_replay" in classifications
    support_minus_anchor = pd.to_numeric(compare.get("support_minus_anchor_only_MV_AP_window", pd.Series(dtype=float)), errors="coerce")
    result["support_improves_over_anchor_only"] = bool((support_minus_anchor.dropna() > 0.002).any())
    real_minus_shuffled = pd.to_numeric(compare.get("real_minus_shuffled_MV_AP_window", pd.Series(dtype=float)), errors="coerce")
    positive_control_signal = bool((real_minus_shuffled.dropna() >= 0.003).any())
    result["false_bridge_risk_nonzero"] = any(bool(r.get("accepted")) and bool(r.get("diagnostic_diff_GT")) for r in edge_rows)
    result["support_has_attributable_signal"] = bool(result["support_improves_over_anchor_only"] or positive_control_signal)
    if result["support_feature_hurts_replay"] and result["support_has_attributable_signal"] and result["false_bridge_risk_nonzero"]:
        result["support_contribution_mode"] = "partial_support_signal_but_membership_gate_failed_false_bridge_risk"
        result["recommended_next_phase"] = "R6_2_low_weight_veto_attenuated_feature_then_R6_3_or_R6_4_if_gate_fails"
    elif not result["support_has_attributable_signal"]:
        result["support_contribution_mode"] = "no_attributable_support_contribution_in_current_membership_path"
        result["recommended_next_phase"] = "skip_R6_2_enter_R6_5_ranking_or_casebook"
    elif result["support_feature_hurts_replay"]:
        result["support_contribution_mode"] = "support_signal_exists_but_current_feature_path_hurts_replay"
        result["recommended_next_phase"] = "R6_3_R6_4_confirmation_or_R6_5_ranking"
    else:
        result["support_contribution_mode"] = "support_signal_present_needs_r6_subset_gate_test"
        result["recommended_next_phase"] = "R6_2"
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)

    fact_lock_root = _project(args.r6_fact_lock_root)
    feature_root = _project(args.feature_root)
    local_ap_root = _project(args.local_ap_root)
    anchor_root = _project(args.anchor_only_root)
    current_root = _project(args.current_phase6d_root)
    r5_edge_attr_root = _project(args.r5_edge_attr_root)
    gt_root = _project(args.r5_gt_coverage_root)

    fact_summary = _read_json(fact_lock_root / "summary.json")
    if not fact_summary.get("phase_r6_0_pass", False):
        raise SystemExit("R6-0 fact lock is missing or not passing; do not enter R6-1.")

    roots = _run_roots(local_ap_root, anchor_root, current_root)
    feature_summary = _feature_summary_map(feature_root)
    gt_diag = _gt_diag_map(gt_root)
    edge_rows = _edge_rows(roots, feature_summary)
    metric_rows = _metric_rows(roots, gt_diag)
    canonical_removal_rows = _canonical_removal_rows(metric_rows, edge_rows, r5_edge_attr_root)
    family_rows = _family_summary_rows(edge_rows)
    casebook_rows = _casebook_rows(edge_rows)
    support_mode = _classify_support_mode(r5_edge_attr_root, edge_rows)

    accepted_edges = [r for r in edge_rows if bool(r.get("accepted"))]
    accepted_diff_gt = [r for r in accepted_edges if bool(r.get("diagnostic_diff_GT"))]
    exact_missing = [
        r
        for r in canonical_removal_rows
        if str(r.get("evidence_status", "")).startswith("not_available")
        or str(r.get("evidence_status", "")).startswith("not_equivalent")
    ]
    failures = []
    if accepted_diff_gt:
        failures.append(
            {
                "schema_version": "stream4d_v103_supp_r6_phaseR6_1_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_EDGE_FALSE_BRIDGE_DIAGNOSTIC_NONZERO",
                "detail": f"accepted_diff_gt_edge_count={len(accepted_diff_gt)}",
                "repair_direction": "Keep support guarded; test low-weight/veto-attenuated or confirmation paths before any support-only merge.",
            }
        )
    if exact_missing:
        failures.append(
            {
                "schema_version": "stream4d_v103_supp_r6_phaseR6_1_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "EXACT_LEAVE_ONE_FAMILY_INTERVENTIONS_INCOMPLETE",
                "detail": ",".join(str(r.get("leave_one_family_variant", "")) for r in exact_missing),
                "repair_direction": "Treat R6-1 as attribution/casebook with proxy evidence only; do not claim exact ablation success.",
            }
        )

    _write_parquet(out / "edge_attribution_rows.parquet", edge_rows)
    _write_csv(out / "edge_family_summary_rows.csv", family_rows)
    _write_csv(out / "leave_one_family_out_metric_rows.csv", canonical_removal_rows)
    _write_csv(out / "all_available_metric_rows.csv", metric_rows)
    _write_csv(out / "edge_failure_casebook_rows.csv", casebook_rows)
    _write_csv(out / "failure_rows.csv", failures)

    summary = {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "CLASSIFIED_SUPPORT_SIGNAL_NOT_MEMBERSHIP_READY"
        if support_mode["support_has_attributable_signal"]
        else "CLASSIFIED_NO_ATTRIBUTABLE_SUPPORT_CONTRIBUTION",
        "phase_r6_1_complete": True,
        "phase_r6_1_method_pass": False,
        "edge_row_count": len(edge_rows),
        "accepted_edge_row_count": len(accepted_edges),
        "accepted_diff_gt_edge_count": len(accepted_diff_gt),
        "edge_family_summary_row_count": len(family_rows),
        "leave_one_family_out_metric_row_count": len(canonical_removal_rows),
        "casebook_row_count": len(casebook_rows),
        "exact_leave_one_family_missing_count": len(exact_missing),
        "failure_count": len(failures),
        "support_attribution": support_mode,
        "current_replay_MV_AP_window": fact_summary.get("current_replay_MV_AP_window", ""),
        "current_locked_D9_MV_AP_window": fact_summary.get("current_locked_D9_MV_AP_window", ""),
        "current_locked_D9_MV_AP50_window": fact_summary.get("current_locked_D9_MV_AP50_window", ""),
        "current_A_anchor_hit_rate": fact_summary.get("current_A_anchor_hit_rate", ""),
        "current_S_support_hit_rate": fact_summary.get("current_S_support_hit_rate", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runs_AP": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "edge_attribution_rows": _rel(out / "edge_attribution_rows.parquet"),
            "edge_family_summary_rows": _rel(out / "edge_family_summary_rows.csv"),
            "leave_one_family_out_metric_rows": _rel(out / "leave_one_family_out_metric_rows.csv"),
            "all_available_metric_rows": _rel(out / "all_available_metric_rows.csv"),
            "edge_failure_casebook_rows": _rel(out / "edge_failure_casebook_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
        },
        "truthfulness_note": "R6-1 reads existing D9/F0/F1-F4 Phase6d edge and metric artifacts. It does not run AP or create new predictions. GT fields are diagnostic-only. Several leave-one-family rows are explicitly marked proxy/not-available rather than fabricated.",
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6 Phase R6-1 support edge attribution casebook.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r6-fact-lock-root", default=str(DEFAULT_R6_FACT_LOCK_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--local-ap-root", default=str(DEFAULT_LOCAL_AP_ROOT))
    parser.add_argument("--anchor-only-root", default=str(DEFAULT_ANCHOR_ONLY_ROOT))
    parser.add_argument("--current-phase6d-root", default=str(DEFAULT_CURRENT_PHASE6D_ROOT))
    parser.add_argument("--r5-edge-attr-root", default=str(DEFAULT_R5_EDGE_ATTR_ROOT))
    parser.add_argument("--r5-gt-coverage-root", default=str(DEFAULT_R5_GT_COVERAGE_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
