#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import _compute_scene_arrays, _project  # noqa: E402


PHASE_ID = "v103_phase9e_d4rt_anchor_da3_induced_carriers"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

DEFAULT_OUT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_r1"
DEFAULT_DUAL_ROLE_ROOT = AUDIT_ROOT / "v103_phase3_dual_role_carrier_sets_r2_repair5"
DEFAULT_PHASE9B_ROOT = AUDIT_ROOT / "v103_phase9b_da3_provider_readiness"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"

RECALL_MIN = 0.35
DIFFERENT_GT_FALSE_MAX = 0.20
HARD_NEGATIVE_FALSE_ACCEPT_MAX = 0.20

SCENE_SPECS = {
    "scene0011_00": {
        "phase2_root": DEFAULT_SCENE0011_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    },
    "scene0050_00": {
        "phase2_root": DEFAULT_SCENE0050_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
    },
}

VARIANTS = [
    {
        "variant_id": "i1_seed1_sem06_bridge040_gap4",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i2_seed3_sem06_bridge035_gap4",
        "seed_min": 3,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i3_seed5_sem05_bridge030_gap4_broad020",
        "seed_min": 5,
        "semantic_cosine_min": 0.50,
        "score_column": "final_bridge_score",
        "score_min": 0.30,
        "max_gap": 4,
        "broad_limit": 0.020,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i4_seed3_sem06_ratiounion008_gap4",
        "seed_min": 3,
        "semantic_cosine_min": 0.60,
        "score_column": "gs_bridge_ratio_union",
        "score_min": 0.08,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i5_bothseed1_sem05_bridge020_gap4",
        "seed_min": 1,
        "semantic_cosine_min": 0.50,
        "score_column": "final_bridge_score",
        "score_min": 0.20,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": True,
    },
    {
        "variant_id": "i6_seed1_sem07_bridge035_gap4_clean",
        "seed_min": 1,
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i7_seed1_sem07_bridge040_gap4_clean",
        "seed_min": 1,
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i8_seed1_sem06_bridge040_union002_gap4_clean",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i9_seed1_sem08_bridge040_gap4_clean",
        "seed_min": 1,
        "semantic_cosine_min": 0.80,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i10_seed3_sem07_bridge035_union002_gap4_clean",
        "seed_min": 3,
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
    },
    {
        "variant_id": "i11_seed1_sem07_bridge035_target_object_nonbroad",
        "seed_min": 1,
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "i12_seed1_sem06_bridge040_union002_target_object_nonbroad",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "i13_seed1_sem06_bridge040_vetoratio050_gap4",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "positive_to_veto_ratio_min": 0.50,
    },
    {
        "variant_id": "i14_seed1_sem06_bridge040_vetoratio100_gap4",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "positive_to_veto_ratio_min": 1.00,
    },
    {
        "variant_id": "i15_seed1_sem06_bridge035_vetoratio050_target_object_nonbroad",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "positive_to_veto_ratio_min": 0.50,
        "target_object_like_non_broad": True,
    },
]

S5_COVERAGE_REPAIR_VARIANTS = [
    {
        "variant_id": "s5r1_anchor_target_objnonbroad_sem06_bridge040_union002",
        "seed_role": "positive",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "s5r2_support_target_objnonbroad_sem06_bridge040_union002",
        "seed_role": "support",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "s5r3_support_target_objnonbroad_sem05_bridge035_union001",
        "seed_role": "support",
        "seed_min": 1,
        "semantic_cosine_min": 0.50,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "gs_bridge_ratio_union_min": 0.01,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "s5r4_support_target_objnonbroad_sem07_bridge035_union002",
        "seed_role": "support",
        "seed_min": 1,
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "gs_bridge_ratio_union_min": 0.02,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
    {
        "variant_id": "s5r5_anchor_or_support_target_objnonbroad_sem06_bridge040_union002_vetoratio050",
        "seed_role": "positive_or_support",
        "seed_min": 1,
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "gs_bridge_ratio_union_min": 0.02,
        "positive_to_veto_ratio_min": 0.50,
        "max_gap": 4,
        "broad_limit": None,
        "require_both_seeded": False,
        "target_object_like_non_broad": True,
    },
]


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


def _project_phase9(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_role_masks(dual_role_root: Path, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    npz_path = dual_role_root / scene_id / "dual_role_carrier_sets.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing dual-role carrier set: {npz_path}")
    pack = np.load(npz_path)
    positive = np.asarray(pack["positive_mask"], dtype=bool)
    veto = np.asarray(pack["veto_mask"], dtype=bool)
    support = np.asarray(pack["support_mask"], dtype=bool) if "support_mask" in pack.files else np.zeros_like(positive, dtype=bool)
    summary = _read_json(dual_role_root / "summary.json")
    roles = dict(dict(summary.get("selected_roles_by_scene", {})).get(scene_id, {}))
    return positive, veto, support, {str(k): str(v) for k, v in roles.items()}


def _support_by_observation(scene_id: str, retained: np.ndarray, diag: dict[str, Any]) -> tuple[dict[str, int], pd.DataFrame]:
    labels = np.asarray(diag["labels"], dtype=np.int32)
    in_image = np.asarray(diag["in_image"], dtype=bool)
    frame_ids = [int(v) for v in diag["frame_ids"]]
    retained = np.asarray(retained, dtype=bool)
    support: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for fi, frame_id in enumerate(frame_ids):
        ok = retained & in_image[fi] & (labels[fi] > 0)
        if not np.any(ok):
            continue
        mask_ids, counts = np.unique(labels[fi][ok], return_counts=True)
        for mask_id, count in zip(mask_ids.tolist(), counts.tolist()):
            obs = f"{scene_id}:{frame_id}:{int(mask_id)}"
            support[obs] = int(count)
            rows.append(
                {
                    "schema_version": "stream4d_v103_phase9e_d4rt_anchor_support_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "mask_id": int(mask_id),
                    "mask_observation_id": obs,
                    "d4rt_anchor_support_count": int(count),
                    "uses_gt_for_prediction": False,
                }
            )
    return support, pd.DataFrame(rows)


def _augment_bridge_rows(
    bridge: pd.DataFrame,
    positive_support: dict[str, int],
    veto_support: dict[str, int],
    support_support: dict[str, int],
    obs_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    out = bridge.copy()
    out["d4rt_positive_support_a"] = out["mask_a_observation_id"].map(positive_support).fillna(0).astype(np.int32)
    out["d4rt_positive_support_b"] = out["mask_b_observation_id"].map(positive_support).fillna(0).astype(np.int32)
    out["d4rt_veto_support_a"] = out["mask_a_observation_id"].map(veto_support).fillna(0).astype(np.int32)
    out["d4rt_veto_support_b"] = out["mask_b_observation_id"].map(veto_support).fillna(0).astype(np.int32)
    out["d4rt_support_support_a"] = out["mask_a_observation_id"].map(support_support).fillna(0).astype(np.int32)
    out["d4rt_support_support_b"] = out["mask_b_observation_id"].map(support_support).fillna(0).astype(np.int32)
    out["d4rt_positive_anchor_support_max"] = np.maximum(
        out["d4rt_positive_support_a"].to_numpy(dtype=np.int32),
        out["d4rt_positive_support_b"].to_numpy(dtype=np.int32),
    )
    out["d4rt_positive_anchor_support_min"] = np.minimum(
        out["d4rt_positive_support_a"].to_numpy(dtype=np.int32),
        out["d4rt_positive_support_b"].to_numpy(dtype=np.int32),
    )
    out["d4rt_positive_anchor_one_side"] = out["d4rt_positive_anchor_support_max"].to_numpy(dtype=np.int32) > 0
    out["d4rt_positive_anchor_both_sides"] = out["d4rt_positive_anchor_support_min"].to_numpy(dtype=np.int32) > 0
    out["d4rt_anchor_to_unanchored"] = (
        out["d4rt_positive_anchor_one_side"].to_numpy(dtype=bool)
        & ~out["d4rt_positive_anchor_both_sides"].to_numpy(dtype=bool)
    )
    for side in ("a", "b"):
        obs_col = f"mask_{side}_observation_id"
        out[f"mask_{side}_is_object_like"] = out[obs_col].map(lambda obs: bool(obs_meta.get(str(obs), {}).get("is_object_like", False)))
        out[f"mask_{side}_is_broad"] = out[obs_col].map(lambda obs: bool(obs_meta.get(str(obs), {}).get("is_broad", True)))
    return out


def _obs_meta(diag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene = str(diag["scene_id"])
    frame_ids = [int(v) for v in diag["frame_ids"]]
    object_like_by_frame = {int(k): np.asarray(v, dtype=np.int32) for k, v in dict(diag["object_like_by_frame"]).items()}
    broad_map = np.asarray(diag["broad_map"], dtype=bool)
    object_map = np.asarray(diag["object_map"], dtype=bool)
    meta: dict[str, dict[str, Any]] = {}
    for fi, frame_id in enumerate(frame_ids):
        object_like_labels = set(int(v) for v in object_like_by_frame.get(fi, np.asarray([], dtype=np.int32)).tolist())
        for label in np.unique(diag["masks"][fi]).astype(int).tolist():
            if label <= 0:
                continue
            safe = min(int(label), broad_map.shape[1] - 1)
            meta[f"{scene}:{frame_id}:{int(label)}"] = {
                "is_object_like": bool(int(label) in object_like_labels or object_map[fi, safe]),
                "is_broad": bool(broad_map[fi, safe]),
            }
    return meta


def _seed_support(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    seed_role = str(spec.get("seed_role", "positive"))
    pos_a = df["d4rt_positive_support_a"].to_numpy(dtype=np.int32)
    pos_b = df["d4rt_positive_support_b"].to_numpy(dtype=np.int32)
    if seed_role == "positive":
        return pos_a, pos_b
    sup_a = df["d4rt_support_support_a"].to_numpy(dtype=np.int32)
    sup_b = df["d4rt_support_support_b"].to_numpy(dtype=np.int32)
    if seed_role == "support":
        return sup_a, sup_b
    if seed_role == "positive_or_support":
        return np.maximum(pos_a, sup_a), np.maximum(pos_b, sup_b)
    raise ValueError(f"unsupported seed_role={seed_role}")


def _accept_mask(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    seed_min = int(spec["seed_min"])
    seed_a, seed_b = _seed_support(df, spec)
    if bool(spec.get("require_both_seeded", False)):
        seed_ok = np.minimum(seed_a, seed_b) >= seed_min
    else:
        seed_ok = np.maximum(seed_a, seed_b) >= seed_min
    acc = (
        seed_ok
        & (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
        & df["semantic_residual_available"].to_numpy(dtype=bool)
        & (df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"]))
        & (df[str(spec["score_column"])].to_numpy(dtype=np.float64) >= float(spec["score_min"]))
    )
    if spec["broad_limit"] is not None:
        acc &= df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
    if spec.get("gs_bridge_ratio_union_min") is not None:
        acc &= df["gs_bridge_ratio_union"].to_numpy(dtype=np.float64) >= float(spec["gs_bridge_ratio_union_min"])
    if spec.get("positive_to_veto_ratio_min") is not None:
        pos_a = seed_a.astype(np.float64) + 1.0
        pos_b = seed_b.astype(np.float64) + 1.0
        veto_a = df["d4rt_veto_support_a"].to_numpy(dtype=np.float64) + 1.0
        veto_b = df["d4rt_veto_support_b"].to_numpy(dtype=np.float64) + 1.0
        ratio_min = np.minimum(pos_a / veto_a, pos_b / veto_b)
        acc &= ratio_min >= float(spec["positive_to_veto_ratio_min"])
    if bool(spec.get("target_object_like_non_broad", False)):
        a_target = (seed_a == 0) & (seed_b > 0)
        b_target = (seed_b == 0) & (seed_a > 0)
        both_seeded = (seed_a > 0) & (seed_b > 0)
        a_ok = df["mask_a_is_object_like"].to_numpy(dtype=bool) & ~df["mask_a_is_broad"].to_numpy(dtype=bool)
        b_ok = df["mask_b_is_object_like"].to_numpy(dtype=bool) & ~df["mask_b_is_broad"].to_numpy(dtype=bool)
        target_ok = (a_target & a_ok) | (b_target & b_ok) | both_seeded
        acc &= target_ok
    return acc


def _score_variant(scene_id: str, df: pd.DataFrame, spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    acc = _accept_mask(df, spec)
    seed_a, seed_b = _seed_support(df, spec)
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    diff_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    same_sem_diff_gt = df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)
    labeled = same_gt | diff_gt
    seed_one_side = np.maximum(seed_a, seed_b) > 0
    seed_both_sides = np.minimum(seed_a, seed_b) > 0
    accepted_labeled = acc & labeled
    tp = int(np.count_nonzero(acc & same_gt))
    fp = int(np.count_nonzero(acc & diff_gt))
    fp_same_sem = int(np.count_nonzero(acc & same_sem_diff_gt))
    positive_total = int(np.count_nonzero(same_gt))
    positive_anchor_reachable = int(np.count_nonzero(same_gt & seed_one_side))
    negative_total = int(np.count_nonzero(diff_gt))
    accepted_labeled_count = int(np.count_nonzero(accepted_labeled))
    recall = float(tp / max(positive_total, 1))
    reachable_recall = float(tp / max(positive_anchor_reachable, 1))
    diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    hard_false = float(fp / max(negative_total, 1))
    seed_to_unanchored = acc & seed_one_side & ~seed_both_sides
    accepted_obs = pd.concat(
        [
            df.loc[acc, ["mask_a_observation_id"]].rename(columns={"mask_a_observation_id": "obs"}),
            df.loc[acc, ["mask_b_observation_id"]].rename(columns={"mask_b_observation_id": "obs"}),
        ],
        ignore_index=True,
    )
    induced_obs = pd.concat(
        [
            df.loc[seed_to_unanchored & (seed_a == 0), ["mask_a_observation_id"]].rename(
                columns={"mask_a_observation_id": "obs"}
            ),
            df.loc[seed_to_unanchored & (seed_b == 0), ["mask_b_observation_id"]].rename(
                columns={"mask_b_observation_id": "obs"}
            ),
        ],
        ignore_index=True,
    )
    formal = bool(recall >= RECALL_MIN and diff_false <= DIFFERENT_GT_FALSE_MAX and hard_false <= HARD_NEGATIVE_FALSE_ACCEPT_MAX)
    clean_induction = bool(
        diff_false <= DIFFERENT_GT_FALSE_MAX
        and hard_false <= HARD_NEGATIVE_FALSE_ACCEPT_MAX
        and int(induced_obs["obs"].nunique()) >= 40
        and tp > 0
    )
    row = {
        "schema_version": "stream4d_v103_phase9e_induced_carrier_variant_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "variant_id": spec["variant_id"],
        "seed_role": str(spec.get("seed_role", "positive")),
        "seed_min": spec["seed_min"],
        "require_both_seeded": bool(spec.get("require_both_seeded", False)),
        "semantic_cosine_min": spec["semantic_cosine_min"],
        "score_column": spec["score_column"],
        "score_min": spec["score_min"],
        "gs_bridge_ratio_union_min": "" if spec.get("gs_bridge_ratio_union_min") is None else spec["gs_bridge_ratio_union_min"],
        "positive_to_veto_ratio_min": "" if spec.get("positive_to_veto_ratio_min") is None else spec["positive_to_veto_ratio_min"],
        "target_object_like_non_broad": bool(spec.get("target_object_like_non_broad", False)),
        "max_gap": spec["max_gap"],
        "broad_limit": "" if spec["broad_limit"] is None else spec["broad_limit"],
        "accepted_count": int(np.count_nonzero(acc)),
        "accepted_labeled_count": accepted_labeled_count,
        "accepted_distinct_mask_observation_count": int(accepted_obs["obs"].nunique()) if len(accepted_obs) else 0,
        "anchor_to_unanchored_pair_count": int(np.count_nonzero(seed_to_unanchored)),
        "induced_unanchored_mask_observation_count": int(induced_obs["obs"].nunique()) if len(induced_obs) else 0,
        "true_positive_same_gt_count": tp,
        "false_positive_different_gt_count": fp,
        "false_positive_same_semantic_different_gt_count": fp_same_sem,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_positive_pair_with_anchor_endpoint_count": positive_anchor_reachable,
        "diagnostic_negative_pair_count": negative_total,
        "same_object_bridge_recall_global": recall,
        "same_object_bridge_recall_anchor_reachable": reachable_recall,
        "different_gt_false_bridge_among_accepted": diff_false,
        "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
        "hard_negative_false_accept_rate": hard_false,
        "phase9e_bridge_gate_pass": formal,
        "phase9e_clean_induction_gate_pass": clean_induction,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    accepted = df.loc[acc].copy()
    accepted["phase9e_variant_id"] = str(spec["variant_id"])
    accepted["phase9e_seed_role"] = str(spec.get("seed_role", "positive"))
    accepted["d4rt_selected_seed_support_a"] = seed_a[acc].astype(np.int32)
    accepted["d4rt_selected_seed_support_b"] = seed_b[acc].astype(np.int32)
    return row, accepted


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (
            bool(r.get("phase9e_bridge_gate_pass", False)),
            float(r.get("same_object_bridge_recall_global", -1.0)),
            -float(r.get("different_gt_false_bridge_among_accepted", 1.0)),
            int(r.get("induced_unanchored_mask_observation_count", 0)),
        ),
    )


def _best_clean(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    clean = [r for r in rows if bool(r.get("phase9e_clean_induction_gate_pass", False))]
    if not clean:
        return None
    return max(
        clean,
        key=lambda r: (
            int(r.get("induced_unanchored_mask_observation_count", 0)),
            float(r.get("same_object_bridge_recall_global", -1.0)),
            -float(r.get("different_gt_false_bridge_among_accepted", 1.0)),
        ),
    )


def _process_scene(
    scene_id: str,
    *,
    dual_role_root: Path,
    phase9b_root: Path,
    out: Path,
    device_id: int,
    variant_family: str,
    phase2_root_by_scene: dict[str, Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame | None]:
    bridge_path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    if not bridge_path.exists():
        failure = {
            "schema_version": "stream4d_v103_phase9e_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "blocker": "phase9b_bridge_rows_missing",
            "path": _rel(bridge_path),
            "uses_gt_for_prediction": False,
        }
        return failure, [failure], None
    positive_mask, veto_mask, support_mask, roles = _load_role_masks(dual_role_root, scene_id)
    spec = dict(SCENE_SPECS[scene_id])
    if phase2_root_by_scene and scene_id in phase2_root_by_scene:
        spec["phase2_root"] = phase2_root_by_scene[scene_id]
    spec["phase2_root"] = _project(spec["phase2_root"])
    diag, _unused_a, _unused_b, _arrays = _compute_scene_arrays(scene_id, spec, scene_out, int(device_id))
    obs_meta = _obs_meta(diag)
    positive_support, positive_rows = _support_by_observation(scene_id, positive_mask, diag)
    veto_support, _veto_rows = _support_by_observation(scene_id, veto_mask, diag)
    support_support, support_rows = _support_by_observation(scene_id, support_mask, diag)
    positive_rows.to_csv(scene_out / "d4rt_positive_anchor_support_rows.csv", index=False)
    _veto_rows.to_csv(scene_out / "d4rt_veto_support_rows.csv", index=False)
    support_rows.to_csv(scene_out / "d4rt_support_coverage_rows.csv", index=False)

    bridge = pd.read_parquet(bridge_path)
    augmented = _augment_bridge_rows(bridge, positive_support, veto_support, support_support, obs_meta)
    anchor_obs_count = int((positive_rows["d4rt_anchor_support_count"] > 0).sum()) if len(positive_rows) else 0
    anchor_support_values = positive_rows["d4rt_anchor_support_count"].to_numpy(dtype=np.float64) if len(positive_rows) else np.asarray([])
    support_obs_count = int((support_rows["d4rt_anchor_support_count"] > 0).sum()) if len(support_rows) else 0
    variants = S5_COVERAGE_REPAIR_VARIANTS if str(variant_family) == "s5_coverage_repair" else VARIANTS
    variant_rows: list[dict[str, Any]] = []
    accepted_by_variant: dict[str, pd.DataFrame] = {}
    for variant in variants:
        row, accepted = _score_variant(scene_id, augmented, variant)
        variant_rows.append(row)
        accepted_by_variant[str(variant["variant_id"])] = accepted
    best = _best(variant_rows)
    best_clean = _best_clean(variant_rows)
    variant_path = scene_out / "induced_variant_rows.csv"
    accepted_path = scene_out / "best_variant_accepted_pair_rows.parquet"
    clean_accepted_path = scene_out / "best_clean_variant_accepted_pair_rows.parquet"
    accepted_variant_dir = scene_out / "accepted_pair_rows_by_variant"
    accepted_variant_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(variant_path, variant_rows)
    for variant_id, accepted in accepted_by_variant.items():
        accepted.to_parquet(accepted_variant_dir / f"{variant_id}_accepted_pair_rows.parquet", index=False)
    accepted_by_variant[str(best["variant_id"])].to_parquet(accepted_path, index=False)
    if best_clean is not None:
        accepted_by_variant[str(best_clean["variant_id"])].to_parquet(clean_accepted_path, index=False)
    formal = any(bool(r["phase9e_bridge_gate_pass"]) for r in variant_rows)
    clean_formal = any(bool(r["phase9e_clean_induction_gate_pass"]) for r in variant_rows)
    summary = {
        "schema_version": "stream4d_v103_phase9e_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "input_bridge_rows": _rel(bridge_path),
        "dual_role_root": _rel(dual_role_root),
        "phase2_root": _rel(spec["phase2_root"]),
        "positive_core_variant_id": roles.get("positive_core", ""),
        "veto_support_variant_id": roles.get("veto_support", ""),
        "variant_family": str(variant_family),
        "support_seed_policy_note": (
            "support seed variants are DA3 coverage diagnostics only; S_support is not promoted to a positive merge witness"
        ),
        "d4rt_positive_anchor_observation_count": anchor_obs_count,
        "d4rt_support_coverage_observation_count": support_obs_count,
        "d4rt_positive_anchor_support_p50": float(np.percentile(anchor_support_values, 50)) if anchor_support_values.size else 0.0,
        "d4rt_positive_anchor_support_p90": float(np.percentile(anchor_support_values, 90)) if anchor_support_values.size else 0.0,
        "candidate_pair_count": int(len(augmented)),
        "candidate_pair_with_anchor_endpoint_count": int(np.count_nonzero(augmented["d4rt_positive_anchor_one_side"].to_numpy(dtype=bool))),
        "candidate_pair_with_both_anchor_endpoints_count": int(np.count_nonzero(augmented["d4rt_positive_anchor_both_sides"].to_numpy(dtype=bool))),
        "repair_variant_count": len(variants),
        "formal_bridge_gate_pass": formal,
        "clean_induction_gate_pass": clean_formal,
        "best_variant_id": best["variant_id"],
        "best_same_object_bridge_recall_global": best["same_object_bridge_recall_global"],
        "best_same_object_bridge_recall_anchor_reachable": best["same_object_bridge_recall_anchor_reachable"],
        "best_different_gt_false_bridge_among_accepted": best["different_gt_false_bridge_among_accepted"],
        "best_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "best_hard_negative_false_accept_rate": best["hard_negative_false_accept_rate"],
        "best_anchor_to_unanchored_pair_count": best["anchor_to_unanchored_pair_count"],
        "best_induced_unanchored_mask_observation_count": best["induced_unanchored_mask_observation_count"],
        "best_clean_variant_id": "" if best_clean is None else best_clean["variant_id"],
        "best_clean_same_object_bridge_recall_global": "" if best_clean is None else best_clean["same_object_bridge_recall_global"],
        "best_clean_same_object_bridge_recall_anchor_reachable": "" if best_clean is None else best_clean["same_object_bridge_recall_anchor_reachable"],
        "best_clean_different_gt_false_bridge_among_accepted": "" if best_clean is None else best_clean["different_gt_false_bridge_among_accepted"],
        "best_clean_induced_unanchored_mask_observation_count": "" if best_clean is None else best_clean["induced_unanchored_mask_observation_count"],
        "blocker": "" if formal else "d4rt_anchor_da3_induction_recall_or_false_gate_fail",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "outputs": {
            "d4rt_positive_anchor_support_rows": _rel(scene_out / "d4rt_positive_anchor_support_rows.csv"),
            "d4rt_veto_support_rows": _rel(scene_out / "d4rt_veto_support_rows.csv"),
            "d4rt_support_coverage_rows": _rel(scene_out / "d4rt_support_coverage_rows.csv"),
            "induced_variant_rows": _rel(variant_path),
            "accepted_pair_rows_by_variant": _rel(accepted_variant_dir),
            "best_variant_accepted_pair_rows": _rel(accepted_path),
            "best_clean_variant_accepted_pair_rows": "" if best_clean is None else _rel(clean_accepted_path),
        },
    }
    return summary, [], positive_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase9e: induce DA3 primitive bridges from sparse reliable D4RT carrier anchors.")
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="scene0011_00")
    parser.add_argument("--dual-role-root", default=str(DEFAULT_DUAL_ROLE_ROOT))
    parser.add_argument("--phase9b-root", default=str(DEFAULT_PHASE9B_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variant-family", choices=["legacy", "s5_coverage_repair"], default="legacy")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project_phase9(args.output_root)
    dual_role_root = _project_phase9(args.dual_role_root)
    phase9b_root = _project_phase9(args.phase9b_root)
    phase2_root_by_scene = {
        "scene0011_00": _project_phase9(args.scene0011_phase2_root),
        "scene0050_00": _project_phase9(args.scene0050_phase2_root),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene_id in scenes:
        try:
            scene_row, scene_failures, _support_rows = _process_scene(
                scene_id,
                dual_role_root=dual_role_root,
                phase9b_root=phase9b_root,
                out=out,
                device_id=int(args.cupy_device_id),
                variant_family=str(args.variant_family),
                phase2_root_by_scene=phase2_root_by_scene,
            )
            scene_rows.append(scene_row)
            failure_rows.extend(scene_failures)
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9e_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "blocker": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            scene_rows.append(failure)
            failure_rows.append(failure)

    pass_count = sum(bool(row.get("formal_bridge_gate_pass", False)) for row in scene_rows)
    clean_count = sum(bool(row.get("clean_induction_gate_pass", False)) for row in scene_rows)
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase9e_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "dual_role_anchor_rows_available",
            "pass": dual_role_root.exists(),
            "observed": _rel(dual_role_root),
            "required": "existing v103 Phase3 dual-role carrier sets",
            "scope": "phase9e",
        },
        {
            "schema_version": "stream4d_v103_phase9e_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "all_requested_scenes_phase9e_bridge_gate_pass",
            "pass": pass_count == len(scenes) and not failure_rows,
            "observed": f"{pass_count}/{len(scenes)}",
            "required": f"{len(scenes)}/{len(scenes)}",
            "scope": "phase9e",
        },
    ]
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    decision = (
        "PASS_PHASE9E_D4RT_ANCHORED_DA3_INDUCED_CARRIERS"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9E_CLEAN_D4RT_ANCHORED_DA3_INDUCTION_DIAGNOSTIC"
        if clean_count > 0 and not failure_rows
        else "NO_GO_PHASE9E_D4RT_ANCHORED_DA3_INDUCED_CARRIERS"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9e_induced_carrier_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "scene_count": len(scenes),
        "pass_scene_count": pass_count,
        "clean_induction_scene_count": clean_count,
        "failure_count": len(failure_rows),
        "repair_variant_count": len(S5_COVERAGE_REPAIR_VARIANTS if str(args.variant_family) == "s5_coverage_repair" else VARIANTS),
        "variant_family": str(args.variant_family),
        "scene0011_phase2_root": _rel(phase2_root_by_scene["scene0011_00"]),
        "scene0050_phase2_root": _rel(phase2_root_by_scene["scene0050_00"]),
        "plan_doc": _rel(PLAN_DOC),
        "truthfulness_note": (
            "Acceptance uses GT-free sparse D4RT positive-core anchor support, DA3 shared-Gaussian bridge scores, "
            "RADIO cosine, frame gap, and optional broad-risk limits. GT labels are used only for diagnostic recall "
            "and false-bridge scoring; this diagnostic does not emit final object predictions."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
