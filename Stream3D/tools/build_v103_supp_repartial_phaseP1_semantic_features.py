#!/usr/bin/env python3
"""Build v103 supplement R2 Phase P1 semantic feature artifacts.

P1 constructs compact semantic feature stores and GT-free similarity baselines
for current c0001 mask observations. It does not cut carriers, build segments,
intervene in mask graph edges, or compute AP.
"""

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

PHASE_ID = "v103_supp_r2_phaseP1_semantic_features"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_P0_ROOT = AUDIT_ROOT / "v103_supp_r2_phaseP0_fact_lock"

RADIO_ROOTS = {
    "scene0011_00": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011",
    "scene0050_00": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050",
}
DINO_ROOT = AUDIT_ROOT / "v81_dino_feature_json_scene0011_scene0050"

PHASE2_ROOTS = {
    "scene0011_00": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            out = {}
            for key in fields:
                value = _jsonable(row.get(key, ""))
                out[key] = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            writer.writerow(out)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_ids_for_scope() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for scene_id, root in PHASE2_ROOTS.items():
        summary = _read_json(root / "summary.json")
        values = summary.get("frame_ids", [])
        out[scene_id] = [int(v) for v in values] if isinstance(values, list) else []
    return out


def _normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1).astype(np.float32)
    safe = np.maximum(norms, np.float32(1e-12))
    return (features / safe[:, None]).astype(np.float32), norms


def _countsketch(features: np.ndarray, compact_dim: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw_dim = int(features.shape[1])
    buckets = rng.integers(0, compact_dim, size=raw_dim, endpoint=False, dtype=np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=raw_dim)
    compact = np.zeros((features.shape[0], compact_dim), dtype=np.float32)
    for raw_idx in range(raw_dim):
        compact[:, buckets[raw_idx]] += features[:, raw_idx] * signs[raw_idx]
    compact_norm, norms = _normalize(compact)
    meta = {
        "projection": "deterministic_countsketch",
        "seed": int(seed),
        "raw_dim": raw_dim,
        "compact_dim": int(compact_dim),
        "bucket_sha1_like_checksum": int(np.sum((np.arange(raw_dim) + 1) * (buckets.astype(np.int64) + 1)) % 1000000007),
        "compact_norm_min": float(norms.min()) if len(norms) else "",
        "compact_norm_max": float(norms.max()) if len(norms) else "",
    }
    return compact_norm.astype(np.float32), meta


def _load_radio(scene_frames: dict[str, list[int]]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    features_by_scene: dict[str, np.ndarray] = {}
    source_meta: dict[str, Any] = {}
    for scene_id, root in RADIO_ROOTS.items():
        npz_path = root / "mask_features.npz"
        rows_path = root / "mask_feature_rows.csv"
        z = np.load(npz_path, allow_pickle=True)
        target = np.array(scene_frames[scene_id], dtype=np.int32)
        keep = np.isin(z["frame_id"], target)
        selected = np.where(keep)[0]
        features_by_scene[scene_id] = np.asarray(z["features"][selected], dtype=np.float32)
        meta_cols = [
            "scene_id",
            "frame_id",
            "mask_id",
            "mask_observation_id",
            "semantic_backend",
            "feature_layer",
            "feature_resolution",
            "feature_pooling_method",
            "feature_available",
            "feature_norm",
            "feature_nan_count",
            "feature_dim",
            "semantic_prototype_id",
            "semantic_prototype_margin",
            "semantic_entropy",
            "semantic_intra_variance",
            "used_token_count",
            "used_pixel_count",
            "broad_background_risk",
            "uses_gt_for_prediction",
            "feature_sha256",
        ]
        meta = pd.read_csv(rows_path, usecols=meta_cols)
        meta = meta[meta["frame_id"].isin(set(scene_frames[scene_id]))].copy()
        meta["semantic_source_id"] = "E_pool_radio"
        meta["source_artifact_root"] = _rel(root)
        meta["source_npz_path"] = _rel(npz_path)
        meta["source_row_index"] = selected
        meta["raw_feature_dim"] = 768
        meta["compact_feature_dim"] = 64
        frames.append(meta)
        source_meta[scene_id] = {
            "source_artifact_root": _rel(root),
            "input_row_count": int(len(z["frame_id"])),
            "target_row_count": int(len(meta)),
            "target_frame_count": int(meta["frame_id"].nunique()),
        }
    return pd.concat(frames, ignore_index=True), features_by_scene, source_meta


def _load_dino(scene_frames: dict[str, list[int]]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    path = DINO_ROOT / "mask_feature_rows.csv"
    usecols = [
        "scene_id",
        "frame_id",
        "mask_id",
        "mask_observation_id",
        "semantic_backend",
        "feature_layer",
        "feature_resolution",
        "feature_pooling_method",
        "feature_available",
        "feature_norm",
        "feature_nan_count",
        "feature_dim",
        "semantic_prototype_id",
        "semantic_prototype_margin",
        "semantic_entropy",
        "semantic_intra_variance",
        "used_token_count",
        "used_pixel_count",
        "broad_background_risk",
        "uses_gt_for_prediction",
        "feature_sha256",
        "feature_json",
    ]
    df = pd.read_csv(path, usecols=usecols)
    keep = np.zeros(len(df), dtype=bool)
    for scene_id, frames in scene_frames.items():
        keep |= (df["scene_id"].to_numpy() == scene_id) & df["frame_id"].isin(set(frames)).to_numpy()
    df = df[keep].copy()
    parsed: list[list[float]] = []
    for value in df.pop("feature_json").tolist():
        parsed.append(json.loads(value))
    features = np.asarray(parsed, dtype=np.float32)
    df["semantic_source_id"] = "E_pool_dino"
    df["source_artifact_root"] = _rel(DINO_ROOT)
    df["source_npz_path"] = ""
    df["source_row_index"] = np.arange(len(df), dtype=np.int32)
    df["raw_feature_dim"] = 384
    df["compact_feature_dim"] = 64
    features_by_scene = {
        scene_id: features[df["scene_id"].to_numpy() == scene_id] for scene_id in scene_frames
    }
    source_meta = {
        scene_id: {
            "source_artifact_root": _rel(DINO_ROOT),
            "target_row_count": int((df["scene_id"] == scene_id).sum()),
            "target_frame_count": int(df.loc[df["scene_id"] == scene_id, "frame_id"].nunique()),
        }
        for scene_id in scene_frames
    }
    return df.reset_index(drop=True), features_by_scene, source_meta


def _pack_features(
    radio_df: pd.DataFrame,
    radio_features: dict[str, np.ndarray],
    dino_df: pd.DataFrame,
    dino_features: dict[str, np.ndarray],
    compact_dim: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    all_rows: list[pd.DataFrame] = []
    raw_norm_by_source: dict[str, np.ndarray] = {}
    compact_blocks: list[np.ndarray] = []
    projection_rows: list[dict[str, Any]] = []
    row_offset = 0
    for source_id, df, feature_by_scene, seed_base in [
        ("E_pool_radio", radio_df, radio_features, 103101),
        ("E_pool_dino", dino_df, dino_features, 103201),
    ]:
        if df.empty:
            continue
        scene_features: list[np.ndarray] = []
        for scene_id in df["scene_id"].drop_duplicates().tolist():
            scene_features.append(feature_by_scene[str(scene_id)])
        raw = np.concatenate(scene_features, axis=0).astype(np.float32)
        raw_norm, raw_norm_values = _normalize(raw)
        compact, projection_meta = _countsketch(raw_norm, compact_dim, seed_base)
        block = df.copy().reset_index(drop=True)
        block["semantic_feature_row_index"] = np.arange(row_offset, row_offset + len(block), dtype=np.int32)
        block["compact_feature_row_index"] = block["semantic_feature_row_index"]
        block["source_local_feature_index"] = np.arange(len(block), dtype=np.int32)
        block["raw_feature_norm_observed"] = raw_norm_values
        block["feature_available"] = block["feature_available"].astype(str).str.lower().isin(["true", "1", "yes"])
        block["feature_nan_count"] = pd.to_numeric(block["feature_nan_count"], errors="coerce").fillna(0).astype(int)
        block["broad_background_risk"] = block["broad_background_risk"].astype(str).str.lower().isin(["true", "1", "yes"])
        block["uses_gt_for_prediction"] = block["uses_gt_for_prediction"].astype(str).str.lower().isin(["true", "1", "yes"])
        block["uses_future"] = False
        all_rows.append(block)
        raw_norm_by_source[source_id] = raw_norm
        compact_blocks.append(compact)
        projection_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP1_projection_row_v1",
                "phase_id": PHASE_ID,
                "semantic_source_id": source_id,
                **projection_meta,
            }
        )
        row_offset += len(block)
    feature_rows = pd.concat(all_rows, ignore_index=True)
    compact_all = np.concatenate(compact_blocks, axis=0).astype(np.float32)
    return feature_rows, raw_norm_by_source, compact_all, projection_rows


def _sample_pairs_for_class(
    df: pd.DataFrame,
    pair_class: str,
    rng: np.random.Generator,
    max_pairs: int,
) -> np.ndarray:
    idx = df.index.to_numpy(dtype=np.int64)
    if len(idx) < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    if pair_class == "random_pair":
        a = rng.choice(idx, size=max_pairs, replace=True)
        b = rng.choice(idx, size=max_pairs, replace=True)
        keep = a != b
        return np.stack([a[keep], b[keep]], axis=1).astype(np.int64)
    if pair_class == "same_frame_competing":
        for _, group in df.groupby("frame_id"):
            ids = group.index.to_numpy(dtype=np.int64)
            if len(ids) < 2:
                continue
            local_pairs = np.array(np.triu_indices(len(ids), k=1)).T
            for a_i, b_i in local_pairs:
                pairs.append((int(ids[a_i]), int(ids[b_i])))
    elif pair_class == "short_range_same_prototype":
        valid = df[df["semantic_prototype_id"].astype(str) != ""]
        for _, group in valid.groupby("semantic_prototype_id"):
            if len(group) < 2:
                continue
            vals = group[["frame_id"]].copy()
            vals["idx"] = group.index.to_numpy(dtype=np.int64)
            arr = vals[["idx", "frame_id"]].to_numpy(dtype=np.int64)
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if 0 < abs(int(arr[i, 1]) - int(arr[j, 1])) <= 15:
                        pairs.append((int(arr[i, 0]), int(arr[j, 0])))
    elif pair_class == "broad_pair":
        ids = df[df["broad_background_risk"]].index.to_numpy(dtype=np.int64)
        if len(ids) >= 2:
            a = rng.choice(ids, size=max_pairs, replace=True)
            b = rng.choice(ids, size=max_pairs, replace=True)
            keep = a != b
            return np.stack([a[keep], b[keep]], axis=1).astype(np.int64)
    elif pair_class == "object_like_pair":
        ids = df[~df["broad_background_risk"]].index.to_numpy(dtype=np.int64)
        if len(ids) >= 2:
            a = rng.choice(ids, size=max_pairs, replace=True)
            b = rng.choice(ids, size=max_pairs, replace=True)
            keep = a != b
            return np.stack([a[keep], b[keep]], axis=1).astype(np.int64)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(pairs, dtype=np.int64)
    if len(arr) > max_pairs:
        arr = arr[rng.choice(len(arr), size=max_pairs, replace=False)]
    return arr


def _pair_rows(
    feature_rows: pd.DataFrame,
    raw_norm_by_source: dict[str, np.ndarray],
    compact: np.ndarray,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    pair_classes = [
        "random_pair",
        "same_frame_competing",
        "short_range_same_prototype",
        "broad_pair",
        "object_like_pair",
    ]
    for source_id in feature_rows["semantic_source_id"].drop_duplicates().tolist():
        source_df = feature_rows[feature_rows["semantic_source_id"] == source_id]
        for scene_id in source_df["scene_id"].drop_duplicates().tolist():
            scene_df = source_df[source_df["scene_id"] == scene_id]
            for pair_class in pair_classes:
                pairs = _sample_pairs_for_class(scene_df, pair_class, rng, max_pairs)
                if len(pairs) == 0:
                    continue
                raw_norm = raw_norm_by_source[str(source_id)]
                left_local = feature_rows.loc[pairs[:, 0], "source_local_feature_index"].to_numpy(dtype=np.int64)
                right_local = feature_rows.loc[pairs[:, 1], "source_local_feature_index"].to_numpy(dtype=np.int64)
                sim_raw = np.einsum("ij,ij->i", raw_norm[left_local], raw_norm[right_local])
                sim_compact = np.einsum("ij,ij->i", compact[pairs[:, 0]], compact[pairs[:, 1]])
                left = feature_rows.loc[pairs[:, 0]]
                right = feature_rows.loc[pairs[:, 1]]
                rows.append(
                    pd.DataFrame(
                        {
                            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_pair_distribution_row_v1",
                            "phase_id": PHASE_ID,
                            "semantic_source_id": source_id,
                            "scene_id": scene_id,
                            "pair_class": pair_class,
                            "left_row_index": pairs[:, 0],
                            "right_row_index": pairs[:, 1],
                            "left_frame_id": left["frame_id"].to_numpy(),
                            "right_frame_id": right["frame_id"].to_numpy(),
                            "left_mask_id": left["mask_id"].to_numpy(),
                            "right_mask_id": right["mask_id"].to_numpy(),
                            "similarity_raw": sim_raw.astype(np.float32),
                            "similarity_compact": sim_compact.astype(np.float32),
                            "similarity_used": sim_compact.astype(np.float32),
                            "uses_gt_for_pair_label": False,
                            "pair_sampling_policy": "GT-free random/competing/same-prototype/broad-risk/object-like diagnostics",
                        }
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _stats(values: pd.Series) -> dict[str, Any]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _baseline_rows(pair_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pair_df.empty:
        return rows
    for (source_id, scene_id), group in pair_df.groupby(["semantic_source_id", "scene_id"]):
        row: dict[str, Any] = {
            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_baseline_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": source_id,
            "scene_id": scene_id,
            "feature_space": "compact64",
            "uses_gt_for_pair_label": False,
        }
        for pair_class, class_group in group.groupby("pair_class"):
            st = _stats(class_group["similarity_used"])
            prefix = pair_class
            for key, value in st.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    for source_id, group in pair_df.groupby("semantic_source_id"):
        row = {
            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_baseline_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": source_id,
            "scene_id": "all",
            "feature_space": "compact64",
            "uses_gt_for_pair_label": False,
        }
        for pair_class, class_group in group.groupby("pair_class"):
            st = _stats(class_group["similarity_used"])
            for key, value in st.items():
                row[f"{pair_class}_{key}"] = value
        rows.append(row)
    return rows


def _summary_rows(feature_rows: pd.DataFrame, pair_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id, group in feature_rows.groupby("semantic_source_id"):
        pair_group = pair_df[pair_df["semantic_source_id"] == source_id] if not pair_df.empty else pd.DataFrame()
        row = {
            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_source_summary_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": source_id,
            "scene_ids": sorted(group["scene_id"].drop_duplicates().tolist()),
            "feature_row_count": int(len(group)),
            "frame_count": int(group[["scene_id", "frame_id"]].drop_duplicates().shape[0]),
            "feature_available_rate": float(group["feature_available"].mean()) if len(group) else 0.0,
            "semantic_nan_rate": float((group["feature_nan_count"] > 0).mean()) if len(group) else 1.0,
            "raw_feature_dim": int(group["raw_feature_dim"].iloc[0]) if len(group) else "",
            "compact_feature_dim": int(group["compact_feature_dim"].iloc[0]) if len(group) else "",
            "broad_background_risk_rate": float(group["broad_background_risk"].mean()) if len(group) else "",
            "random_pair_count": int((pair_group["pair_class"] == "random_pair").sum()) if not pair_group.empty else 0,
            "hard_negative_pair_count": int((pair_group["pair_class"] == "same_frame_competing").sum()) if not pair_group.empty else 0,
            "short_range_pseudo_positive_pair_count": int((pair_group["pair_class"] == "short_range_same_prototype").sum()) if not pair_group.empty else 0,
            "uses_gt_for_prediction": bool(group["uses_gt_for_prediction"].any()),
            "uses_future": bool(group["uses_future"].any()),
            "enabled_for_phaseP2": True,
        }
        rows.append(row)
    rows.append(
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_source_summary_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": "E_clip_crop_b16",
            "scene_ids": ["scene0011_00", "scene0050_00"],
            "feature_row_count": 0,
            "frame_count": 0,
            "feature_available_rate": 0.0,
            "semantic_nan_rate": "",
            "raw_feature_dim": 512,
            "compact_feature_dim": 64,
            "random_pair_count": 0,
            "hard_negative_pair_count": 0,
            "short_range_pseudo_positive_pair_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "enabled_for_phaseP2": False,
            "note": "Not built in this P1 pass. P0 policy allows sparse mask-level or low-resolution compact map, not high-dimensional dense pixel map.",
        }
    )
    rows.append(
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP1_semantic_source_summary_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": "E_consensus",
            "scene_ids": ["scene0011_00", "scene0050_00"],
            "feature_row_count": 0,
            "frame_count": 0,
            "feature_available_rate": 0.0,
            "semantic_nan_rate": "",
            "raw_feature_dim": "",
            "compact_feature_dim": "",
            "random_pair_count": 0,
            "hard_negative_pair_count": 0,
            "short_range_pseudo_positive_pair_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "enabled_for_phaseP2": False,
            "note": "Not enabled until at least one object-level crop source exists; no fake consensus is fabricated.",
        }
    )
    return rows


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP1_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def _build_gates(source_rows: list[dict[str, Any]], pair_df: pd.DataFrame, clip_enabled: bool) -> list[dict[str, Any]]:
    enabled = [row for row in source_rows if row.get("enabled_for_phaseP2") is True]
    non_clip = [row for row in enabled if str(row.get("semantic_source_id")) != "E_clip_crop_b16"]
    min_avail = min(float(row.get("feature_available_rate", 0.0)) for row in non_clip) if non_clip else 0.0
    max_nan = max(float(row.get("semantic_nan_rate", 1.0)) for row in non_clip) if non_clip else 1.0
    max_compact = max(int(row.get("compact_feature_dim", 9999)) for row in non_clip) if non_clip else 9999
    random_counts = {row["semantic_source_id"]: row.get("random_pair_count", 0) for row in non_clip}
    hard_counts = {row["semantic_source_id"]: row.get("hard_negative_pair_count", 0) for row in non_clip}
    return [
        _gate("non_clip_semantic_source_count_ge_1", len(non_clip) >= 1, [row["semantic_source_id"] for row in non_clip], ">=1", "Bind/build at least one non-CLIP source before P2."),
        _gate("non_clip_available_rate_ge_0p95", min_avail >= 0.95, min_avail, ">=0.95", "Repair feature extraction or target-frame join before P2."),
        _gate("semantic_nan_rate_eq_0", max_nan == 0.0, max_nan, "0.0", "Drop/repair NaN semantic rows before P2."),
        _gate("compact_dim_le_128", max_compact <= 128, max_compact, "<=128", "Use compact projection before P2; do not pass high-dimensional dense features."),
        _gate("random_pair_distribution_nonempty", all(int(v) > 0 for v in random_counts.values()), random_counts, ">0 for each enabled source", "Increase pair sample cap or inspect source row count."),
        _gate("hard_negative_distribution_nonempty", all(int(v) > 0 for v in hard_counts.values()), hard_counts, ">0 for each enabled source", "Build same-frame competing distribution before P2."),
        _gate("clip_path_gate_skipped_when_disabled", not clip_enabled, clip_enabled, "clip disabled or feature_available_rate>=0.95", "If enabling CLIP, build sparse/compact CLIP crop features and rerun P1."),
        _gate("uses_gt_for_prediction_false", not any(bool(row.get("uses_gt_for_prediction")) for row in enabled), [row["semantic_source_id"] for row in enabled if row.get("uses_gt_for_prediction")], "[]", "Remove any prediction-time GT dependency."),
        _gate("uses_future_false", not any(bool(row.get("uses_future")) for row in enabled), [row["semantic_source_id"] for row in enabled if row.get("uses_future")], "[]", "Remove future-frame dependency."),
        _gate("pair_distribution_rows_nonempty", not pair_df.empty, int(len(pair_df)), ">0", "Repair pair sampling before P2."),
    ]


def _existing_pass(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return {}
    summary = _read_json(summary_path)
    required = [
        output_root / "semantic_feature_rows.parquet",
        output_root / "semantic_baseline_rows.csv",
        output_root / "semantic_pair_distribution_rows.parquet",
        output_root / "semantic_source_summary_rows.csv",
        output_root / "semantic_features_compact_fp16.npz",
    ]
    if summary.get("phaseP1_pass") and all(path.exists() for path in required):
        return summary
    return {}


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    if args.reuse_existing and not args.force:
        existing = _existing_pass(output_root)
        if existing:
            existing["cache_reused"] = True
            return existing

    start = time.time()
    compact_dim = int(args.compact_dim)
    scene_frames = _frame_ids_for_scope()
    radio_df, radio_features, radio_meta = _load_radio(scene_frames)
    dino_df, dino_features, dino_meta = _load_dino(scene_frames) if args.enable_dino else (pd.DataFrame(), {}, {})
    feature_rows, raw_norm_by_source, compact, projection_rows = _pack_features(
        radio_df,
        radio_features,
        dino_df,
        dino_features,
        compact_dim,
    )
    pair_df = _pair_rows(feature_rows, raw_norm_by_source, compact, args.max_pairs_per_class, args.seed)
    baseline_rows = _baseline_rows(pair_df)
    source_summary_rows = _summary_rows(feature_rows, pair_df)
    gates = _build_gates(source_summary_rows, pair_df, clip_enabled=False)
    failure_rows = [row for row in gates if not row["pass"]]
    phaseP1_pass = len(failure_rows) == 0

    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows_path = output_root / "semantic_feature_rows.parquet"
    pair_rows_path = output_root / "semantic_pair_distribution_rows.parquet"
    baseline_csv = output_root / "semantic_baseline_rows.csv"
    source_summary_csv = output_root / "semantic_source_summary_rows.csv"
    gate_csv = output_root / "gate_rows.csv"
    failure_csv = output_root / "failure_rows.csv"
    projection_csv = output_root / "semantic_projection_rows.csv"
    compact_npz = output_root / "semantic_features_compact_fp16.npz"
    summary_path = output_root / "summary.json"

    feature_rows.to_parquet(feature_rows_path, index=False)
    pair_df.to_parquet(pair_rows_path, index=False)
    _write_csv(baseline_csv, baseline_rows)
    _write_csv(source_summary_csv, source_summary_rows)
    _write_csv(gate_csv, gates)
    _write_csv(failure_csv, failure_rows)
    _write_csv(projection_csv, projection_rows)
    np.savez_compressed(
        compact_npz,
        compact_features=compact.astype(np.float16),
        semantic_feature_row_index=feature_rows["semantic_feature_row_index"].to_numpy(dtype=np.int32),
        semantic_source_id=feature_rows["semantic_source_id"].astype(str).to_numpy(),
        scene_id=feature_rows["scene_id"].astype(str).to_numpy(),
        frame_id=feature_rows["frame_id"].to_numpy(dtype=np.int32),
        mask_id=feature_rows["mask_id"].to_numpy(dtype=np.int32),
        mask_observation_id=feature_rows["mask_observation_id"].astype(str).to_numpy(),
    )

    summary = {
        "schema_version": "stream4d_v103_supp_r2_phaseP1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - start,
        "decision": "PASS_ENTER_PHASEP2_OBSERVATION_RELIABILITY" if phaseP1_pass else "NO_GO_REPAIR_PHASEP1_SEMANTIC_FEATURES",
        "phaseP1_pass": bool(phaseP1_pass),
        "failure_count": len(failure_rows),
        "p0_root": _rel(_project(args.p0_root)),
        "scope_frame_ids": scene_frames,
        "semantic_sources_enabled_for_P2": [
            row["semantic_source_id"] for row in source_summary_rows if row.get("enabled_for_phaseP2") is True
        ],
        "semantic_sources_disabled": [
            row["semantic_source_id"] for row in source_summary_rows if row.get("enabled_for_phaseP2") is not True
        ],
        "radio_meta": radio_meta,
        "dino_meta": dino_meta,
        "feature_row_count": int(len(feature_rows)),
        "pair_distribution_row_count": int(len(pair_df)),
        "compact_dim": compact_dim,
        "max_pairs_per_class": int(args.max_pairs_per_class),
        "clip_path_enabled": False,
        "clip_policy_note": (
            "CLIP crop is not built in this pass. P0 permits sparse mask-level CLIP or low-resolution compact maps, "
            "but high-dimensional dense pixel-level CLIP maps remain disallowed."
        ),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(summary_path),
            "semantic_feature_rows": _rel(feature_rows_path),
            "semantic_baseline_rows": _rel(baseline_csv),
            "semantic_pair_distribution_rows": _rel(pair_rows_path),
            "semantic_source_summary_rows": _rel(source_summary_csv),
            "semantic_features_compact_fp16": _rel(compact_npz),
            "semantic_projection_rows": _rel(projection_csv),
            "gate_rows": _rel(gate_csv),
            "failure_rows": _rel(failure_csv),
        },
        "truthfulness_note": (
            "P1 constructs GT-free semantic feature/calibration artifacts only. It does not cut carriers, "
            "build segment primitives, intervene in graph edges, or compute AP."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--p0-root", default=str(DEFAULT_P0_ROOT))
    parser.add_argument("--compact-dim", type=int, default=64)
    parser.add_argument("--max-pairs-per-class", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=10301)
    parser.add_argument("--enable-dino", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.compact_dim <= 0 or args.compact_dim > 128:
        raise SystemExit("--compact-dim must be in 1..128 for P1")
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if summary["phaseP1_pass"] else 2)


if __name__ == "__main__":
    main()
