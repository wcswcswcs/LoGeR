#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
OUT_DIR = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q4_dense16384_fast"

PHASE_ID = "v103_phase3_fast_carrier_reliability_filter"
VISIBLE_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.0
SEMANTIC_DELTA_LOCAL = 3
SEMANTIC_CONTRADICTION_THRESHOLD = 0.20
SELF_ERROR_SIGMA_NORM = 0.015
BROAD_AREA_RATIO = 0.12
OBJECT_LIKE_AREA_MIN = 0.001
OBJECT_LIKE_AREA_MAX = 0.20

NEEDED_CACHE_KEYS = [
    "carrier_id",
    "src_frame",
    "src_uv",
    "xyz_ref",
    "uv_pred",
    "visibility_prob",
    "confidence_prob",
    "valid",
    "src_frame_global",
    "src_xy",
    "src_mask_id",
    "query_source_code",
]

SCENE_INPUTS = {
    "scene0011_00": {
        "phase2_root": AUDIT_ROOT / "v103_phase2_stratified_q4_dense16384_scene0011_first32",
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    },
    "scene0050_00": {
        "phase2_root": AUDIT_ROOT / "v103_phase2_stratified_q4_dense16384_scene0050_first32",
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
    },
}

VARIANTS = [
    {"variant_id": "S0_no_semantic_top40", "semantic": False, "top_rate": 0.40, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top40", "semantic": True, "top_rate": 0.40, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top30", "semantic": True, "top_rate": 0.30, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top40_hardveto", "semantic": True, "top_rate": 0.40, "hard_veto": True, "score_mode": "base"},
    {"variant_id": "S2_clean_broad_jitter_top40", "semantic": True, "top_rate": 0.40, "hard_veto": False, "score_mode": "clean_broad_jitter"},
]

REPAIR_BROAD_JITTER_VARIANTS = [
    {
        "variant_id": "R1_broad080_jitter006_semhard_top20",
        "semantic": True,
        "top_rate": 0.20,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.006,
    },
    {
        "variant_id": "R2_broad080_jitter006_semhard_top10",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.006,
    },
    {
        "variant_id": "R3_broad070_jitter006_sem015_top20",
        "semantic": True,
        "top_rate": 0.20,
        "hard_veto": False,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.70,
        "max_jitter": 0.006,
        "max_semantic_contradiction": 0.15,
    },
    {
        "variant_id": "R4_broad090_jitter004_semhard_top15",
        "semantic": True,
        "top_rate": 0.15,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
    },
    {
        "variant_id": "R5_broad080_semhard_top30",
        "semantic": True,
        "top_rate": 0.30,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
    },
]

SUPPORT_BALANCED_VARIANTS = [
    {
        "variant_id": "B1_support_balanced_broad090_jitter006_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.90,
        "max_jitter": 0.006,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "B2_support_balanced_broad085_jitter008_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.85,
        "max_jitter": 0.008,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "B3_support_balanced_broad080_jitter006_sem015_top15_floor80b16",
        "semantic": True,
        "top_rate": 0.15,
        "hard_veto": False,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.006,
        "max_semantic_contradiction": 0.15,
        "min_object_like_support_per_mask": 80,
        "min_boundary_support_per_mask": 16,
    },
    {
        "variant_id": "B4_support_balanced_broad090_jitter004_semhard_top15_floor60b12",
        "semantic": True,
        "top_rate": 0.15,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
]

SUPPORT_BALANCED_REPAIR2_VARIANTS = [
    {
        "variant_id": "B5_repair2_broad090_jitter004_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "B6_repair2_broad085_jitter004_semhard_top08_floor50b10",
        "semantic": True,
        "top_rate": 0.08,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.85,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
    {
        "variant_id": "B7_repair2_broad080_jitter005_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.005,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "B8_repair2_broad080_jitter004_sem010_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": False,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.004,
        "max_semantic_contradiction": 0.10,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "B9_repair2_broad070_jitter004_sem010_top12_floor50b10",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": False,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.70,
        "max_jitter": 0.004,
        "max_semantic_contradiction": 0.10,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
]

SOURCE_BALANCED_REPAIR3_VARIANTS = [
    {
        "variant_id": "C1_sourcepenalty_broad090_jitter004_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.45, 4: 0.15, 5: 0.25, 6: 0.15, 7: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "C2_block_comp_sem_broad_broad090_jitter004_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "allowed_query_sources": [1, 2, 3, 7],
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.50, 7: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "C3_interior_uniform_overlap_broad090_jitter004_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "allowed_query_sources": [1, 2, 7],
        "source_penalty": {1: 1.20, 2: 1.0, 7: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "C4_sourcepenalty_broad085_jitter004_semhard_top08_floor50b10",
        "semantic": True,
        "top_rate": 0.08,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "source_penalty": {1: 1.25, 2: 1.0, 3: 0.35, 4: 0.10, 5: 0.20, 6: 0.10, 7: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
    {
        "variant_id": "C5_block_comp_sem_broad_broad085_jitter005_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "allowed_query_sources": [1, 2, 3, 7],
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.45, 7: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.005,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
]

FALSE_BRIDGE_REPAIR4_VARIANTS = [
    {
        "variant_id": "D1_broad085_jitter006_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "D2_broad085_jitter005_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.85,
        "max_jitter": 0.005,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "D3_broad080_jitter006_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.80,
        "max_jitter": 0.006,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "D4_broad085_jitter006_sem010_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": False,
        "score_mode": "broad_jitter_semantic_strong",
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "max_semantic_contradiction": 0.10,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "D5_sourceveto_broad085_jitter006_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "allowed_query_sources": [1, 2, 3, 7],
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.50, 7: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "D6_interior_only_broad085_jitter006_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_strong",
        "allowed_query_sources": [1, 2, 7],
        "source_penalty": {1: 1.10, 2: 1.0, 7: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
]

COMPETING_REPAIR5_VARIANTS = [
    {
        "variant_id": "E1_compveto020_srcpen_broad085_jitter004_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_competing_strong",
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.35, 4: 0.05, 5: 0.15, 6: 0.05, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.004,
        "max_competing_conflict_rate": 0.20,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "E2_compveto015_srcpen_broad085_jitter005_semhard_top12_floor60b12",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_competing_strong",
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.30, 4: 0.05, 5: 0.12, 6: 0.05, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.005,
        "max_competing_conflict_rate": 0.15,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "E3_compsoft_srcpen_broad090_jitter004_semhard_top10_floor60b12",
        "semantic": True,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_competing_strong",
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.40, 4: 0.08, 5: 0.18, 6: 0.08, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "E4_compveto025_boundary_keep_broad090_jitter005_semhard_top12_floor50b10",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_competing_strong",
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.55, 4: 0.12, 5: 0.25, 6: 0.08, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.005,
        "max_competing_conflict_rate": 0.25,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
    {
        "variant_id": "E5_compveto010_interior_overlap_boundary_broad085_jitter006_semhard_top12_floor50b10",
        "semantic": True,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_semantic_competing_strong",
        "allowed_query_sources": [1, 2, 3, 7, 8],
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.45, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "max_competing_conflict_rate": 0.10,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
]

SEMANTIC_VETO_REPAIR6_VARIANTS = [
    {
        "variant_id": "V1_s0_compveto010_srcpen_broad085_jitter006_semhard_top12_floor50b10",
        "semantic": False,
        "top_rate": 0.12,
        "hard_veto": True,
        "score_mode": "broad_jitter_competing_no_sempos",
        "allowed_query_sources": [1, 2, 3, 7, 8],
        "source_penalty": {1: 1.10, 2: 1.0, 3: 0.45, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.85,
        "max_jitter": 0.006,
        "max_competing_conflict_rate": 0.10,
        "min_object_like_support_per_mask": 50,
        "min_boundary_support_per_mask": 10,
    },
    {
        "variant_id": "V2_s0_compveto020_srcpen_broad090_jitter006_semhard_top15_floor60b12",
        "semantic": False,
        "top_rate": 0.15,
        "hard_veto": True,
        "score_mode": "broad_jitter_competing_no_sempos",
        "allowed_query_sources": [1, 2, 3, 7, 8],
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.45, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.006,
        "max_competing_conflict_rate": 0.20,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
    {
        "variant_id": "V3_s0_compsoft_srcpen_broad090_jitter004_semhard_top10_floor60b12",
        "semantic": False,
        "top_rate": 0.10,
        "hard_veto": True,
        "score_mode": "broad_jitter_competing_no_sempos",
        "source_penalty": {1: 1.15, 2: 1.0, 3: 0.40, 4: 0.08, 5: 0.18, 6: 0.08, 7: 1.0, 8: 1.0},
        "max_broad_rate": 0.90,
        "max_jitter": 0.004,
        "min_object_like_support_per_mask": 60,
        "min_boundary_support_per_mask": 12,
    },
]

ALL_SUPPORT_BALANCED_VARIANTS = (
    SUPPORT_BALANCED_VARIANTS
    + SUPPORT_BALANCED_REPAIR2_VARIANTS
    + SOURCE_BALANCED_REPAIR3_VARIANTS
    + FALSE_BRIDGE_REPAIR4_VARIANTS
    + COMPETING_REPAIR5_VARIANTS
    + SEMANTIC_VETO_REPAIR6_VARIANTS
)


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


def _sha256_short(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while read < max_bytes:
            chunk = f.read(min(1024 * 1024, max_bytes - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _ensure_mmap_cache(phase2_root: Path) -> tuple[Path, dict[str, Any]]:
    phase2_root = _project(phase2_root)
    batch_path = phase2_root / "carrier_batch.npz"
    cache_dir = phase2_root / "carrier_batch_mmap_cache"
    manifest_path = cache_dir / "manifest.json"
    if not batch_path.exists():
        raise FileNotFoundError(batch_path)
    batch_stat = batch_path.stat()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            manifest.get("source_path") == _rel(batch_path)
            and int(manifest.get("source_size_bytes", -1)) == int(batch_stat.st_size)
            and all((cache_dir / f"{key}.npy").exists() for key in NEEDED_CACHE_KEYS)
        )
        if valid:
            manifest["cache_reused"] = True
            return cache_dir, manifest

    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_rows: list[dict[str, Any]] = []
    with np.load(batch_path, allow_pickle=False) as data:
        missing = [key for key in NEEDED_CACHE_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"carrier_batch missing keys: {missing}")
        for key in NEEDED_CACHE_KEYS:
            arr = np.asarray(data[key])
            out_path = cache_dir / f"{key}.npy"
            np.save(out_path, arr)
            array_rows.append(
                {
                    "key": key,
                    "path": _rel(out_path),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size_bytes": out_path.stat().st_size,
                    "sha256_first64m": _sha256_short(out_path),
                }
            )
    manifest = {
        "schema_version": "stream4d_v103_phase3_carrier_mmap_cache_manifest_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "source_path": _rel(batch_path),
        "source_size_bytes": batch_stat.st_size,
        "cache_dir": _rel(cache_dir),
        "cache_reused": False,
        "array_rows": array_rows,
        "truthfulness_note": "Cache is an uncompressed .npy expansion of D4RT carrier_batch.npz for mmap-friendly Phase3 reads; it does not change D4RT outputs.",
    }
    _write_json(manifest_path, manifest)
    return cache_dir, manifest


def _load_cached(cache_dir: Path) -> dict[str, np.ndarray]:
    return {key: np.load(cache_dir / f"{key}.npy", mmap_mode="r") for key in NEEDED_CACHE_KEYS}


def _load_semantic(scene: str, spec: dict[str, Path]) -> tuple[dict[tuple[int, int], int], np.ndarray, dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    pack = np.load(spec["semantic_npz"], allow_pickle=False)
    features = _normalize_rows(pack["features"].astype(np.float32))
    frame_ids = pack["frame_id"].astype(np.int64)
    mask_ids = pack["mask_id"].astype(np.int64)
    feature_index = {(int(f), int(m)): int(i) for i, (f, m) in enumerate(zip(frame_ids.tolist(), mask_ids.tolist()))}
    rows = pd.read_csv(spec["semantic_rows"])
    meta: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        if str(row.get("scene_id")) != scene:
            continue
        frame_id = int(row["frame_id"])
        mask_id = int(row["mask_id"])
        meta[(frame_id, mask_id)] = {
            "broad_background_risk": _parse_bool(row.get("broad_background_risk")),
            "semantic_background_score_proxy": _parse_bool(row.get("semantic_background_score_proxy")),
            "used_pixel_count": int(float(row.get("used_pixel_count") or 0)),
            "semantic_boundary_variance": float(row.get("semantic_boundary_variance") or 0.0),
            "semantic_entropy": float(row.get("semantic_entropy") or 0.0),
        }
    rng = np.random.default_rng(10303)
    pair_count = min(8192, max(0, features.shape[0] * 2))
    if features.shape[0] >= 2 and pair_count > 0:
        a = rng.integers(0, features.shape[0], size=pair_count)
        b = rng.integers(0, features.shape[0], size=pair_count)
        neq = a != b
        sims = np.sum(features[a[neq]] * features[b[neq]], axis=1)
    else:
        sims = np.asarray([], dtype=np.float32)
    constants = {
        "scene_id": scene,
        "semantic_source": _rel(spec["semantic_npz"]),
        "semantic_pair_sample_count": int(sims.shape[0]),
        "mu_sem_random_mask_pair_mean": float(np.mean(sims)) if sims.size else 0.0,
        "mu_sem_random_mask_pair_p50": float(np.percentile(sims, 50)) if sims.size else 0.0,
        "mu_sem_used": float(np.mean(sims)) if sims.size else 0.0,
    }
    return feature_index, features, meta, constants


def _load_scene_summary_and_masks(scene: str, phase2_root: Path) -> tuple[dict[str, Any], list[int], np.ndarray]:
    summary = json.loads((_project(phase2_root) / "summary.json").read_text(encoding="utf-8"))
    frame_ids = [int(v) for v in summary["frame_ids"]]
    mask_root = _project(summary["mask_root"])
    masks = np.stack([_load_mask(mask_root / f"{frame_id}.png") for frame_id in frame_ids], axis=0)
    return summary, frame_ids, masks


def _mask_meta_maps(
    scene: str,
    frame_ids: list[int],
    masks: np.ndarray,
    semantic_meta: dict[tuple[int, int], dict[str, Any]],
    feature_index: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    max_label = int(np.max(masks)) if masks.size else 0
    broad_map = np.zeros((len(frame_ids), max_label + 1), dtype=bool)
    object_map = np.zeros((len(frame_ids), max_label + 1), dtype=bool)
    feature_map = np.full((len(frame_ids), max_label + 1), -1, dtype=np.int32)
    object_like_by_frame: dict[int, np.ndarray] = {}
    h, w = masks.shape[1:]
    denom = float(max(h * w, 1))
    for fi, frame_id in enumerate(frame_ids):
        labels, counts = np.unique(masks[fi], return_counts=True)
        object_labels: list[int] = []
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label <= 0:
                continue
            area_ratio = float(count) / denom
            row = semantic_meta.get((int(frame_id), label), {})
            broad = (
                bool(row.get("broad_background_risk"))
                or bool(row.get("semantic_background_score_proxy"))
                or area_ratio >= BROAD_AREA_RATIO
            )
            object_like = (OBJECT_LIKE_AREA_MIN <= area_ratio <= OBJECT_LIKE_AREA_MAX) and not broad
            broad_map[fi, label] = broad
            object_map[fi, label] = object_like
            feature_map[fi, label] = int(feature_index.get((int(frame_id), label), -1))
            if object_like:
                object_labels.append(label)
        object_like_by_frame[fi] = np.asarray(sorted(object_labels), dtype=np.int32)
    meta = {
        "max_label": max_label,
        "object_like_mask_count": int(sum(len(v) for v in object_like_by_frame.values())),
        "object_like_by_frame": object_like_by_frame,
        "height": h,
        "width": w,
    }
    return broad_map, object_map, feature_map, meta


def _project_labels_cupy(batch: dict[str, np.ndarray], masks: np.ndarray, device_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    try:
        import cupy as cp
    except Exception:
        cp = None  # type: ignore[assignment]
    t0 = time.time()
    uv_pred = batch["uv_pred"]
    xyz_ref = batch["xyz_ref"]
    valid = batch["valid"]
    frame_count, carrier_count = valid.shape
    height, width = masks.shape[1:]
    labels = np.full((frame_count, carrier_count), -1, dtype=np.int32)
    in_image = np.zeros((frame_count, carrier_count), dtype=bool)
    finite = np.zeros((frame_count, carrier_count), dtype=bool)
    xs_all = np.zeros((frame_count, carrier_count), dtype=np.int16)
    ys_all = np.zeros((frame_count, carrier_count), dtype=np.int16)
    if cp is None:
        backend = "numpy_vectorized_fallback"
        for fi in range(frame_count):
            uv = np.asarray(uv_pred[fi], dtype=np.float32)
            xyz = np.asarray(xyz_ref[fi], dtype=np.float32)
            ok_finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
            ok = (
                np.asarray(valid[fi], dtype=bool)
                & ok_finite
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            xs = np.rint(np.clip(uv[:, 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(np.int32)
            ys = np.rint(np.clip(uv[:, 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(np.int32)
            labels[fi, ok] = masks[fi, ys[ok], xs[ok]]
            in_image[fi] = ok
            finite[fi] = ok_finite
            xs_all[fi] = xs.astype(np.int16)
            ys_all[fi] = ys.astype(np.int16)
        return labels, in_image, finite, xs_all, ys_all, time.time() - t0, backend

    backend = "cupy_framewise_vectorized_projection"
    with cp.cuda.Device(int(device_id)):
        for fi in range(frame_count):
            uv_g = cp.asarray(np.asarray(uv_pred[fi], dtype=np.float32))
            xyz_g = cp.asarray(np.asarray(xyz_ref[fi], dtype=np.float32))
            valid_g = cp.asarray(np.asarray(valid[fi], dtype=np.bool_))
            mask_g = cp.asarray(masks[fi], dtype=cp.int32)
            finite_g = cp.isfinite(uv_g).all(axis=1) & cp.isfinite(xyz_g).all(axis=1)
            ok = valid_g & finite_g & (uv_g[:, 0] >= 0.0) & (uv_g[:, 0] <= 1.0) & (uv_g[:, 1] >= 0.0) & (uv_g[:, 1] <= 1.0)
            xs = cp.rint(cp.clip(uv_g[:, 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(cp.int32)
            ys = cp.rint(cp.clip(uv_g[:, 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(cp.int32)
            out = cp.full((carrier_count,), -1, dtype=cp.int32)
            out[ok] = mask_g[ys[ok], xs[ok]]
            labels[fi] = cp.asnumpy(out)
            in_image[fi] = cp.asnumpy(ok)
            finite[fi] = cp.asnumpy(finite_g)
            xs_all[fi] = cp.asnumpy(xs).astype(np.int16, copy=False)
            ys_all[fi] = cp.asnumpy(ys).astype(np.int16, copy=False)
        cp.cuda.Stream.null.synchronize()
    return labels, in_image, finite, xs_all, ys_all, time.time() - t0, backend


def _semantic_similarity_matrix(features: np.ndarray, device_id: int) -> tuple[np.ndarray, float, str]:
    try:
        import cupy as cp
    except Exception:
        cp = None  # type: ignore[assignment]
    t0 = time.time()
    if cp is None:
        return features @ features.T, time.time() - t0, "numpy_matmul_fallback"
    with cp.cuda.Device(int(device_id)):
        feat_g = cp.asarray(features.astype(np.float32, copy=False))
        sim_g = feat_g @ feat_g.T
        cp.cuda.Stream.null.synchronize()
        sim = cp.asnumpy(sim_g)
    return sim.astype(np.float32, copy=False), time.time() - t0, "cupy_matmul_full_semantic_matrix"


def _boundary_any_maps(masks: np.ndarray) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=bool)
    kernel = np.ones((5, 5), dtype=np.uint8)
    for fi in range(masks.shape[0]):
        frame = masks[fi]
        for label in np.unique(frame).tolist():
            label = int(label)
            if label <= 0:
                continue
            binary = (frame == label).astype(np.uint8)
            grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel).astype(bool)
            out[fi] |= grad
    return out


def _same_frame_competing_boundary_maps(masks: np.ndarray, object_map: np.ndarray, broad_map: np.ndarray) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=bool)
    for fi in range(masks.shape[0]):
        frame = np.asarray(masks[fi], dtype=np.int32)
        max_label = int(object_map.shape[1] - 1)
        safe = np.clip(frame, 0, max_label)
        object_pix = np.asarray(object_map[fi], dtype=bool)[safe]
        broad_pix = np.asarray(broad_map[fi], dtype=bool)[safe]
        foreground = frame > 0

        def mark(a_slice: tuple[slice, slice], b_slice: tuple[slice, slice]) -> None:
            a = frame[a_slice]
            b = frame[b_slice]
            fg = foreground[a_slice] & foreground[b_slice]
            diff = fg & (a != b)
            if not np.any(diff):
                return
            obj_pair = object_pix[a_slice] | object_pix[b_slice]
            broad_object_pair = (object_pix[a_slice] & broad_pix[b_slice]) | (object_pix[b_slice] & broad_pix[a_slice])
            conflict = diff & (obj_pair | broad_object_pair)
            out[fi][a_slice] |= conflict
            out[fi][b_slice] |= conflict

        mark((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
        mark((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    return out


def _carrier_hit_rate(hit_maps: np.ndarray, labels: np.ndarray, in_image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(labels.shape[1])
    hit_count = np.zeros((n,), dtype=np.int32)
    obs_count = np.zeros((n,), dtype=np.int32)
    for fi in range(labels.shape[0]):
        ok = np.asarray(in_image[fi], dtype=bool) & (np.asarray(labels[fi], dtype=np.int32) > 0)
        obs_count += ok.astype(np.int32)
        if np.any(ok):
            hit = np.asarray(hit_maps[fi], dtype=bool)[ys[fi], xs[fi]]
            hit_count += (ok & hit).astype(np.int32)
    rate = np.divide(
        hit_count.astype(np.float32),
        np.maximum(obs_count, 1).astype(np.float32),
        out=np.zeros((n,), dtype=np.float32),
        where=obs_count > 0,
    )
    return rate.astype(np.float32, copy=False), hit_count.astype(np.int32, copy=False)


def _source_risk_score(query_source_code: np.ndarray) -> np.ndarray:
    source = np.asarray(query_source_code, dtype=np.int16)
    risk = np.zeros(source.shape, dtype=np.float32)
    risk[source == 3] = 0.45
    risk[source == 4] = 1.00
    risk[source == 5] = 0.75
    risk[source == 6] = 1.00
    risk[source == 8] = 0.0
    return risk


def _variant_scores_and_candidate(variant: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    score_key = "reliability_s2" if bool(variant["semantic"]) else "reliability_s0"
    scores = np.asarray(arrays[score_key], dtype=np.float64).copy()
    mode = str(variant.get("score_mode", "base"))
    if mode == "clean_broad_jitter":
        broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
        contradiction = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
        jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
        clean = np.square(np.clip(1.0 - broad, 0.0, 1.0))
        clean *= np.clip(1.0 - contradiction, 0.0, 1.0)
        clean *= np.exp(-jitter / 0.004)
        scores *= clean
    if mode in {"broad_jitter_semantic_strong", "broad_jitter_semantic_competing_strong"}:
        broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
        contradiction = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
        jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
        clean = np.power(np.clip(1.0 - broad, 0.0, 1.0), 4.0)
        clean *= np.square(np.clip(1.0 - contradiction, 0.0, 1.0))
        clean *= np.exp(-jitter / 0.0025)
        if mode == "broad_jitter_semantic_competing_strong":
            competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
            source_risk = np.asarray(arrays.get("source_risk_score", np.zeros_like(scores)), dtype=np.float64)
            clean *= np.power(np.clip(1.0 - competing, 0.0, 1.0), 4.0)
            clean *= np.square(np.clip(1.0 - source_risk, 0.0, 1.0))
        scores *= clean
    if mode == "broad_jitter_competing_no_sempos":
        broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
        jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
        competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
        source_risk = np.asarray(arrays.get("source_risk_score", np.zeros_like(scores)), dtype=np.float64)
        clean = np.power(np.clip(1.0 - broad, 0.0, 1.0), 4.0)
        clean *= np.exp(-jitter / 0.0025)
        clean *= np.power(np.clip(1.0 - competing, 0.0, 1.0), 4.0)
        clean *= np.square(np.clip(1.0 - source_risk, 0.0, 1.0))
        scores *= clean
    if variant.get("source_penalty"):
        source = np.asarray(arrays["query_source_code"], dtype=np.int16)
        penalty = np.ones_like(scores, dtype=np.float64)
        for code, value in dict(variant["source_penalty"]).items():
            penalty[source == int(code)] = float(value)
        scores *= penalty
    n = int(scores.shape[0])
    candidate = np.ones((n,), dtype=bool)
    if "max_broad_rate" in variant:
        candidate &= np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64) <= float(variant["max_broad_rate"])
    if "max_jitter" in variant:
        candidate &= np.asarray(arrays["normalized_jitter"], dtype=np.float64) <= float(variant["max_jitter"])
    if "max_semantic_contradiction" in variant:
        candidate &= np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64) <= float(variant["max_semantic_contradiction"])
    if "max_competing_conflict_rate" in variant:
        candidate &= np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64) <= float(variant["max_competing_conflict_rate"])
    if "allowed_query_sources" in variant:
        candidate &= np.isin(np.asarray(arrays["query_source_code"], dtype=np.int16), np.asarray(variant["allowed_query_sources"], dtype=np.int16))
    scores = np.where(candidate, scores, -np.inf)
    return scores, candidate


def _variant_hard_ok(variant: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    n = int(np.asarray(arrays["carrier_id"]).shape[0])
    hard_ok = np.ones((n,), dtype=bool)
    if bool(variant.get("hard_veto")):
        hard_ok &= np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64) <= SEMANTIC_CONTRADICTION_THRESHOLD
    if bool(variant.get("competing_hard_veto")) and "max_competing_conflict_rate" in variant:
        hard_ok &= np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64) <= float(variant["max_competing_conflict_rate"])
    return hard_ok


def _compute_scene_arrays(
    scene: str,
    spec: dict[str, Path],
    output_root: Path,
    device_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_t0 = time.time()
    phase2_root = _project(spec["phase2_root"])
    cache_dir, cache_manifest = _ensure_mmap_cache(phase2_root)
    batch = _load_cached(cache_dir)
    summary, frame_ids, masks = _load_scene_summary_and_masks(scene, phase2_root)
    feature_index, features, semantic_meta, constants = _load_semantic(scene, spec)
    broad_map, object_map, feature_map, meta = _mask_meta_maps(scene, frame_ids, masks, semantic_meta, feature_index)
    labels, in_image, finite, xs, ys, projection_runtime, projection_backend = _project_labels_cupy(batch, masks, device_id)

    n = int(batch["carrier_id"].shape[0])
    obs_positive = in_image & (labels > 0)
    obs_count = np.sum(obs_positive, axis=0).astype(np.int32)
    broad_count = np.zeros((n,), dtype=np.int32)
    object_count = np.zeros((n,), dtype=np.int32)
    feat_idx_by_frame = np.full(labels.shape, -1, dtype=np.int32)
    for fi in range(labels.shape[0]):
        lab = np.clip(labels[fi], 0, broad_map.shape[1] - 1)
        broad_count += (obs_positive[fi] & broad_map[fi, lab]).astype(np.int32)
        object_count += (obs_positive[fi] & object_map[fi, lab]).astype(np.int32)
        feat = feature_map[fi, lab]
        feat_idx_by_frame[fi] = np.where(obs_positive[fi], feat, -1).astype(np.int32)

    broad_rate = np.where(obs_count > 0, broad_count / np.maximum(obs_count, 1), 1.0).astype(np.float32)
    object_like_rate = np.where(obs_count > 0, object_count / np.maximum(obs_count, 1), 0.0).astype(np.float32)
    visibility = np.asarray(batch["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(batch["confidence_prob"], dtype=np.float32)
    valid = np.asarray(batch["valid"], dtype=bool)
    visible = valid & finite & (visibility >= VISIBLE_THRESHOLD) & (confidence >= CONFIDENCE_THRESHOLD)
    visibility_rate = np.mean(visible, axis=0).astype(np.float32)
    in_image_rate = np.mean(in_image, axis=0).astype(np.float32)
    conf_sum = np.sum(np.where(in_image, confidence, 0.0), axis=0, dtype=np.float64)
    conf_count = np.sum(in_image, axis=0)
    conf_mean = np.where(conf_count > 0, conf_sum / np.maximum(conf_count, 1), 0.0).astype(np.float32)

    height, width = masks.shape[1:]
    diag = float(math.sqrt(float(width * width + height * height)))
    src_frame = np.asarray(batch["src_frame"], dtype=np.int64)
    src_frame_clipped = np.clip(src_frame, 0, labels.shape[0] - 1)
    carrier_idx = np.arange(n, dtype=np.int64)
    self_uv = np.asarray(batch["uv_pred"][src_frame_clipped, carrier_idx], dtype=np.float32)
    src_uv = np.asarray(batch["src_uv"], dtype=np.float32)
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    self_error_px = np.linalg.norm((self_uv - src_uv) * scale, axis=1).astype(np.float32)
    self_error_px[~np.isfinite(self_error_px)] = diag
    self_error_norm = np.clip(self_error_px / max(diag, 1.0), 0.0, 1.0).astype(np.float32)

    sem_t0 = time.time()
    sem_matrix, sem_matrix_runtime, sem_backend = _semantic_similarity_matrix(features, device_id)
    mu_sem = float(constants["mu_sem_used"])
    sem_cal = np.clip((sem_matrix - mu_sem) / max(1.0 - mu_sem, 1e-6), 0.0, 1.0).astype(np.float32)
    sem_pair_count = np.zeros((n,), dtype=np.int16)
    sem_bad_count = np.zeros((n,), dtype=np.int16)
    sim_rows: list[np.ndarray] = []
    for fi in range(labels.shape[0]):
        idx_a = feat_idx_by_frame[fi]
        for fj in range(fi + 1, min(labels.shape[0], fi + SEMANTIC_DELTA_LOCAL + 1)):
            idx_b = feat_idx_by_frame[fj]
            ok = (idx_a >= 0) & (idx_b >= 0)
            row = np.full((n,), np.nan, dtype=np.float32)
            if np.any(ok):
                vals = sem_cal[idx_a[ok], idx_b[ok]]
                row[ok] = vals
                sem_pair_count += ok.astype(np.int16)
                sem_bad_count += (ok & (row < SEMANTIC_CONTRADICTION_THRESHOLD)).astype(np.int16)
            sim_rows.append(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sem_stability = np.nanmedian(np.stack(sim_rows, axis=0), axis=0).astype(np.float32)
    sem_stability[~np.isfinite(sem_stability)] = 1.0
    sem_contradiction = np.where(sem_pair_count > 0, sem_bad_count / np.maximum(sem_pair_count, 1), 0.0).astype(np.float32)
    total_sem_pairs = int(np.sum(sem_pair_count))
    total_sem_bad = int(np.sum(sem_bad_count))
    unfiltered_sem = float(total_sem_bad / total_sem_pairs) if total_sem_pairs else 0.0
    semantic_runtime = time.time() - sem_t0

    boundary_any = _boundary_any_maps(masks)
    competing_boundary = _same_frame_competing_boundary_maps(masks, object_map, broad_map)
    boundary_hit_rate, boundary_hit_count = _carrier_hit_rate(boundary_any, labels, in_image, xs, ys)
    same_frame_competing_conflict_rate, same_frame_competing_conflict_count = _carrier_hit_rate(competing_boundary, labels, in_image, xs, ys)
    source_risk = _source_risk_score(np.asarray(batch["query_source_code"], dtype=np.int16))
    conflict_rate = same_frame_competing_conflict_rate
    combined_conflict_rate = np.maximum(sem_contradiction, same_frame_competing_conflict_rate).astype(np.float32)
    geo = (conf_mean * visibility_rate * in_image_rate * np.exp(-self_error_norm / SELF_ERROR_SIGMA_NORM)).astype(np.float32)
    mask_rel = np.maximum(0.0, (1.0 - broad_rate) * (1.0 - combined_conflict_rate) * object_like_rate).astype(np.float32)
    reliability_s0 = (geo * mask_rel).astype(np.float32)
    reliability_s2 = (geo * mask_rel * sem_stability).astype(np.float32)

    scene_arrays = {
        "carrier_id": np.asarray(batch["carrier_id"], dtype=np.int64),
        "query_source_code": np.asarray(batch["query_source_code"], dtype=np.int16),
        "src_frame": np.asarray(batch["src_frame"], dtype=np.int16),
        "src_frame_global": np.asarray(batch["src_frame_global"], dtype=np.int32),
        "obs_in_image_count": obs_count,
        "in_image_rate": in_image_rate,
        "visibility_rate": visibility_rate,
        "confidence_mean_in_image": conf_mean,
        "self_uv_error_px": self_error_px,
        "normalized_jitter": self_error_norm,
        "broad_mask_participation_rate": broad_rate,
        "object_like_mask_rate": object_like_rate,
        "mask_boundary_hit_rate": boundary_hit_rate,
        "mask_boundary_hit_count": boundary_hit_count,
        "competing_mask_conflict_rate": conflict_rate,
        "same_frame_competing_mask_conflict_count": same_frame_competing_conflict_count,
        "combined_semantic_competing_conflict_rate": combined_conflict_rate,
        "source_risk_score": source_risk,
        "semantic_short_range_stability": sem_stability,
        "semantic_contradiction_rate": sem_contradiction,
        "semantic_pair_count": sem_pair_count.astype(np.int16),
        "r_geo": geo,
        "r_mask": mask_rel,
        "r_sem": sem_stability,
        "reliability_s0": reliability_s0,
        "reliability_s2": reliability_s2,
    }

    perf = {
        "cache_manifest": cache_manifest,
        "projection_runtime_sec": projection_runtime,
        "projection_backend": projection_backend,
        "semantic_matrix_runtime_sec": sem_matrix_runtime,
        "semantic_runtime_sec": semantic_runtime,
        "semantic_backend": sem_backend,
        "scene_runtime_sec": time.time() - scene_t0,
        "cache_dir": _rel(cache_dir),
    }
    diag = {
        "scene_id": scene,
        "frame_ids": frame_ids,
        "masks": masks,
        "labels": labels,
        "in_image": in_image,
        "xs": xs,
        "ys": ys,
        "boundary_any": boundary_any,
        "broad_map": broad_map,
        "object_map": object_map,
        "object_like_by_frame": meta["object_like_by_frame"],
        "object_like_mask_count": meta["object_like_mask_count"],
        "unfiltered_semantic_contradiction_rate": unfiltered_sem,
        "projection_label_shape": list(labels.shape),
        "phase2_summary": summary,
        "semantic_constants": constants,
        "performance": perf,
    }
    carrier_df = pd.DataFrame(scene_arrays)
    carrier_df.insert(0, "scene_id", scene)
    carrier_df.insert(0, "phase_id", PHASE_ID)
    carrier_df.insert(0, "schema_version", "stream4d_v103_phase3_fast_carrier_reliability_row_v1")
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    carrier_path = scene_dir / "carrier_reliability_rows.parquet"
    carrier_df.to_parquet(carrier_path, index=False)
    preview_rows = carrier_df.sort_values("reliability_s2", ascending=True).head(64).to_dict("records")
    _write_csv(scene_dir / "carrier_filter_casebook_rows.csv", preview_rows)
    diag["carrier_reliability_rows"] = _rel(carrier_path)
    diag["carrier_filter_casebook_rows"] = _rel(scene_dir / "carrier_filter_casebook_rows.csv")
    return diag, [], [], scene_arrays


def _support_metrics(diag: dict[str, Any], retained: np.ndarray) -> dict[str, Any]:
    labels = diag["labels"]
    in_image = diag["in_image"]
    xs = diag["xs"]
    ys = diag["ys"]
    boundary_any = diag["boundary_any"]
    object_like_by_frame = diag["object_like_by_frame"]
    support_values: list[int] = []
    boundary_values: list[int] = []
    retained_idx = np.flatnonzero(retained)
    max_label = int(np.max(labels)) if labels.size else 0
    for fi in range(labels.shape[0]):
        obj_labels = np.asarray(object_like_by_frame.get(fi, np.asarray([], dtype=np.int32)), dtype=np.int32)
        if obj_labels.size == 0:
            continue
        if retained_idx.size == 0:
            support_values.extend([0] * int(obj_labels.size))
            boundary_values.extend([0] * int(obj_labels.size))
            continue
        lab = labels[fi, retained_idx]
        ok = in_image[fi, retained_idx] & (lab > 0)
        counts = np.bincount(lab[ok].astype(np.int32), minlength=max_label + 1) if np.any(ok) else np.zeros(max_label + 1, dtype=np.int64)
        bnd_ok = np.zeros_like(ok)
        if np.any(ok):
            ok_pos = np.flatnonzero(ok)
            bnd_ok[ok_pos] = boundary_any[fi, ys[fi, retained_idx[ok_pos]], xs[fi, retained_idx[ok_pos]]]
        bnd_counts = (
            np.bincount(lab[bnd_ok].astype(np.int32), minlength=max_label + 1) if np.any(bnd_ok) else np.zeros(max_label + 1, dtype=np.int64)
        )
        support_values.extend(counts[obj_labels].astype(int).tolist())
        boundary_values.extend(bnd_counts[obj_labels].astype(int).tolist())
    support_arr = np.asarray(support_values, dtype=np.float64)
    boundary_arr = np.asarray(boundary_values, dtype=np.float64)
    return {
        "object_like_mask_count": int(diag["object_like_mask_count"]),
        "object_like_mask_support_p10": float(np.percentile(support_arr, 10)) if support_arr.size else 0.0,
        "object_like_mask_support_p50": float(np.percentile(support_arr, 50)) if support_arr.size else 0.0,
        "boundary_band_support_p10": float(np.percentile(boundary_arr, 10)) if boundary_arr.size else 0.0,
        "boundary_band_support_p50": float(np.percentile(boundary_arr, 50)) if boundary_arr.size else 0.0,
        "mask_support_coverage_after_filter": float(np.mean(support_arr > 0)) if support_arr.size else 0.0,
    }


def _top_score_indices(scores: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    if int(count) <= 0 or indices.size == 0:
        return np.asarray([], dtype=np.int64)
    if indices.size <= int(count):
        return indices.astype(np.int64, copy=False)
    local_scores = np.asarray(scores[indices], dtype=np.float64)
    kth = int(indices.size - int(count))
    picked_local = np.argpartition(local_scores, kth)[kth:]
    return indices[picked_local].astype(np.int64, copy=False)


def _apply_support_balanced_backfill(
    *,
    diag: dict[str, Any],
    scores: np.ndarray,
    candidate: np.ndarray,
    retained: np.ndarray,
    min_object_like_support_per_mask: int,
    min_boundary_support_per_mask: int,
) -> tuple[np.ndarray, int, int]:
    labels = diag["labels"]
    in_image = diag["in_image"]
    xs = diag["xs"]
    ys = diag["ys"]
    boundary_any = diag["boundary_any"]
    object_like_by_frame = diag["object_like_by_frame"]
    retained = np.asarray(retained, dtype=bool).copy()
    candidate = np.asarray(candidate, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    added_object = 0
    added_boundary = 0
    min_obj = int(min_object_like_support_per_mask)
    min_bnd = int(min_boundary_support_per_mask)
    if min_obj <= 0 and min_bnd <= 0:
        return retained, added_object, added_boundary

    for fi in range(labels.shape[0]):
        obj_labels = np.asarray(object_like_by_frame.get(fi, np.asarray([], dtype=np.int32)), dtype=np.int32)
        if obj_labels.size == 0:
            continue
        frame_ok = candidate & in_image[fi]
        if not np.any(frame_ok):
            continue
        frame_labels = labels[fi]
        frame_boundary = boundary_any[fi, ys[fi], xs[fi]]
        for label in obj_labels.tolist():
            label = int(label)
            label_ok = frame_ok & (frame_labels == label)
            if min_obj > 0:
                current = int(np.count_nonzero(retained & label_ok))
                need = max(0, min_obj - current)
                if need:
                    add_idx = _top_score_indices(scores, np.flatnonzero(label_ok & ~retained), need)
                    before = int(np.count_nonzero(retained))
                    retained[add_idx] = True
                    added_object += int(np.count_nonzero(retained) - before)
            if min_bnd > 0:
                boundary_ok = label_ok & frame_boundary
                current_bnd = int(np.count_nonzero(retained & boundary_ok))
                need_bnd = max(0, min_bnd - current_bnd)
                if need_bnd:
                    add_idx = _top_score_indices(scores, np.flatnonzero(boundary_ok & ~retained), need_bnd)
                    before = int(np.count_nonzero(retained))
                    retained[add_idx] = True
                    added_boundary += int(np.count_nonzero(retained) - before)
    return retained, added_object, added_boundary


def _evaluate_variant(scene: str, variant: dict[str, Any], arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> dict[str, Any]:
    score_key = "reliability_s2" if bool(variant["semantic"]) else "reliability_s0"
    scores, candidate = _variant_scores_and_candidate(variant, arrays)
    n = int(scores.shape[0])
    keep_n = max(1, int(round(float(variant["top_rate"]) * n)))
    candidate_count = int(np.count_nonzero(candidate))
    if keep_n >= n:
        threshold = float(np.min(scores))
        retained = np.ones((n,), dtype=bool)
    elif candidate_count <= keep_n:
        retained = candidate.copy()
        threshold = float(np.min(scores[retained])) if np.any(retained) else -float("inf")
    else:
        order = np.argpartition(scores, n - keep_n)
        keep = order[n - keep_n :]
        threshold = float(np.min(scores[keep]))
        retained = np.zeros((n,), dtype=bool)
        retained[keep] = True
    if bool(variant.get("hard_veto")):
        hard_ok = _variant_hard_ok(variant, arrays)
        retained &= hard_ok
        candidate &= hard_ok
    support_backfill_added_object = 0
    support_backfill_added_boundary = 0
    if "min_object_like_support_per_mask" in variant or "min_boundary_support_per_mask" in variant:
        retained, support_backfill_added_object, support_backfill_added_boundary = _apply_support_balanced_backfill(
            diag=diag,
            scores=scores,
            candidate=candidate,
            retained=retained,
            min_object_like_support_per_mask=int(variant.get("min_object_like_support_per_mask", 0)),
            min_boundary_support_per_mask=int(variant.get("min_boundary_support_per_mask", 0)),
        )
    retained_count = int(np.count_nonzero(retained))
    support = _support_metrics(diag, retained)
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
    source_risk = np.asarray(arrays["source_risk_score"], dtype=np.float64)
    sem_pair_count = np.asarray(arrays["semantic_pair_count"], dtype=np.float64)
    sem_bad_rate = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    unfiltered_broad = float(np.mean(broad))
    filtered_broad = float(np.mean(broad[retained])) if retained_count else 1.0
    unfiltered_competing = float(np.mean(competing))
    filtered_competing = float(np.mean(competing[retained])) if retained_count else 1.0
    unfiltered_source_risk = float(np.mean(source_risk))
    filtered_source_risk = float(np.mean(source_risk[retained])) if retained_count else 1.0
    unfiltered_sem = float(diag["unfiltered_semantic_contradiction_rate"])
    filtered_sem = (
        float(np.sum(sem_pair_count[retained] * sem_bad_rate[retained]) / max(np.sum(sem_pair_count[retained]), 1.0))
        if retained_count
        else 1.0
    )
    unfiltered_jitter_p90 = float(np.percentile(jitter, 90))
    filtered_jitter_p90 = float(np.percentile(jitter[retained], 90)) if retained_count else 1.0
    return {
        "schema_version": "stream4d_v103_phase3_fast_filter_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant["variant_id"],
        "score_key": score_key,
        "threshold": threshold,
        "candidate_count_after_gtfree_prefilter": candidate_count,
        "retained_carrier_count": retained_count,
        "total_carrier_count": n,
        "retained_carrier_rate": float(retained_count / max(n, 1)),
        "object_like_mask_support_p10": support["object_like_mask_support_p10"],
        "object_like_mask_support_p50": support["object_like_mask_support_p50"],
        "boundary_band_support_p10": support["boundary_band_support_p10"],
        "boundary_band_support_p50": support["boundary_band_support_p50"],
        "mask_support_coverage_after_filter": support["mask_support_coverage_after_filter"],
        "broad_mask_participation_rate": filtered_broad,
        "unfiltered_broad_mask_participation_rate": unfiltered_broad,
        "broad_relative_reduction": float((unfiltered_broad - filtered_broad) / max(unfiltered_broad, 1e-9)),
        "competing_mask_conflict_rate": filtered_competing,
        "unfiltered_competing_mask_conflict_rate": unfiltered_competing,
        "competing_mask_conflict_relative_reduction": float((unfiltered_competing - filtered_competing) / max(unfiltered_competing, 1e-9)) if unfiltered_competing > 0 else 0.0,
        "source_risk_score_mean": filtered_source_risk,
        "unfiltered_source_risk_score_mean": unfiltered_source_risk,
        "source_risk_relative_reduction": float((unfiltered_source_risk - filtered_source_risk) / max(unfiltered_source_risk, 1e-9)) if unfiltered_source_risk > 0 else 0.0,
        "semantic_contradiction_rate": filtered_sem,
        "unfiltered_semantic_contradiction_rate": unfiltered_sem,
        "semantic_relative_reduction": float((unfiltered_sem - filtered_sem) / max(unfiltered_sem, 1e-9)) if unfiltered_sem > 0 else 0.0,
        "normalized_jitter_p90": filtered_jitter_p90,
        "unfiltered_normalized_jitter_p90": unfiltered_jitter_p90,
        "jitter_relative_reduction": float((unfiltered_jitter_p90 - filtered_jitter_p90) / max(unfiltered_jitter_p90, 1e-9)),
        "in_image_rate_mean": float(np.mean(np.asarray(arrays["in_image_rate"], dtype=np.float64)[retained])) if retained_count else 0.0,
        "visibility_rate_mean": float(np.mean(np.asarray(arrays["visibility_rate"], dtype=np.float64)[retained])) if retained_count else 0.0,
        "object_like_mask_count": support["object_like_mask_count"],
        "support_backfill_object_floor": int(variant.get("min_object_like_support_per_mask", 0)),
        "support_backfill_boundary_floor": int(variant.get("min_boundary_support_per_mask", 0)),
        "support_backfill_added_object": int(support_backfill_added_object),
        "support_backfill_added_boundary": int(support_backfill_added_boundary),
        "uses_gt_for_threshold": False,
        "uses_future": False,
    }


def _select_and_gate(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    selected_by_scene: dict[str, str] = {}
    scenes = sorted({str(row["scene_id"]) for row in metric_rows})
    for scene in scenes:
        scene_rows = [row for row in metric_rows if row["scene_id"] == scene]

        def pass_key(row: dict[str, Any]) -> tuple[int, float]:
            checks = [
                0.05 <= float(row["retained_carrier_rate"]) <= 0.60,
                float(row["object_like_mask_support_p10"]) >= 50.0,
                float(row["boundary_band_support_p10"]) >= 10.0,
                float(row["broad_relative_reduction"]) >= 0.20,
                float(row["semantic_relative_reduction"]) >= 0.20 if float(row["unfiltered_semantic_contradiction_rate"]) > 0 else True,
                float(row["jitter_relative_reduction"]) >= 0.20,
            ]
            margin = (
                float(row["broad_relative_reduction"])
                + float(row["semantic_relative_reduction"])
                + float(row["jitter_relative_reduction"])
                + float(row.get("competing_mask_conflict_relative_reduction", 0.0))
                + float(row.get("source_risk_relative_reduction", 0.0))
            )
            return (sum(bool(v) for v in checks), margin)

        selected = max(scene_rows, key=pass_key)
        selected_by_scene[scene] = str(selected["variant_id"])
        gate_specs = [
            ("retained_carrier_rate_between_0p05_0p60", 0.05 <= float(selected["retained_carrier_rate"]) <= 0.60, selected["retained_carrier_rate"], "0.05..0.60"),
            ("object_like_mask_support_p10_ge_50", float(selected["object_like_mask_support_p10"]) >= 50.0, selected["object_like_mask_support_p10"], 50.0),
            ("boundary_band_support_p10_ge_10", float(selected["boundary_band_support_p10"]) >= 10.0, selected["boundary_band_support_p10"], 10.0),
            ("broad_mask_participation_relative_reduction_ge_0p20", float(selected["broad_relative_reduction"]) >= 0.20, selected["broad_relative_reduction"], 0.20),
            ("semantic_contradiction_relative_reduction_ge_0p20", (float(selected["semantic_relative_reduction"]) >= 0.20 if float(selected["unfiltered_semantic_contradiction_rate"]) > 0 else True), selected["semantic_relative_reduction"], 0.20),
            ("normalized_jitter_p90_relative_reduction_ge_0p20", float(selected["jitter_relative_reduction"]) >= 0.20, selected["jitter_relative_reduction"], 0.20),
        ]
        for name, ok, observed, required in gate_specs:
            gate = {
                "schema_version": "stream4d_v103_phase3_fast_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_variant_id": selected["variant_id"],
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
            }
            gate_rows.append(gate)
            if not ok:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase3_fast_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"selected_variant={selected['variant_id']} observed={observed} required={required}",
                        "repair_direction": "Follow Phase3 repair ladder: query strata/density, semantic hard veto, broad downweight, jitter normalization; do not enter clustering or DA3 without a real D4RT blocker.",
                    }
                )
    return gate_rows, failure_rows, selected_by_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast v103 Phase3 carrier reliability filtering with mmap cache and GPU vectorized lookups.")
    parser.add_argument("--output-root", default=str(OUT_DIR))
    parser.add_argument("--scene0011-phase2-root", default=str(SCENE_INPUTS["scene0011_00"]["phase2_root"]))
    parser.add_argument("--scene0050-phase2-root", default=str(SCENE_INPUTS["scene0050_00"]["phase2_root"]))
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument(
        "--variant-family",
        choices=[
            "base",
            "repair_broad_jitter",
            "support_balanced",
            "support_balanced_repair2",
            "source_balanced_repair3",
            "false_bridge_repair4",
            "competing_repair5",
            "semantic_veto_repair6",
        ],
        default="base",
    )
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = {scene: dict(spec) for scene, spec in SCENE_INPUTS.items()}
    specs["scene0011_00"]["phase2_root"] = _project(args.scene0011_phase2_root)
    specs["scene0050_00"]["phase2_root"] = _project(args.scene0050_phase2_root)
    scene_ids = list(specs) if args.scene == "all" else [args.scene]

    metric_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    if args.variant_family == "base":
        variants = VARIANTS
    elif args.variant_family == "repair_broad_jitter":
        variants = REPAIR_BROAD_JITTER_VARIANTS
    elif args.variant_family == "support_balanced":
        variants = SUPPORT_BALANCED_VARIANTS
    elif args.variant_family == "support_balanced_repair2":
        variants = SUPPORT_BALANCED_REPAIR2_VARIANTS
    elif args.variant_family == "source_balanced_repair3":
        variants = SOURCE_BALANCED_REPAIR3_VARIANTS
    elif args.variant_family == "false_bridge_repair4":
        variants = FALSE_BRIDGE_REPAIR4_VARIANTS
    elif args.variant_family == "competing_repair5":
        variants = COMPETING_REPAIR5_VARIANTS
    else:
        variants = SEMANTIC_VETO_REPAIR6_VARIANTS
    for scene in scene_ids:
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, specs[scene], output_root, int(args.cupy_device_id))
        semantic_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_fast_semantic_distribution_row_v1",
                "phase_id": PHASE_ID,
                **diag["semantic_constants"],
                "unfiltered_semantic_contradiction_rate": diag["unfiltered_semantic_contradiction_rate"],
            }
        )
        for variant in variants:
            metric_rows.append(_evaluate_variant(scene, variant, arrays, diag))
        perf = diag["performance"]
        performance_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_fast_performance_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "carrier_count": int(len(arrays["carrier_id"])),
                "projection_backend": perf["projection_backend"],
                "projection_runtime_sec": perf["projection_runtime_sec"],
                "semantic_backend": perf["semantic_backend"],
                "semantic_matrix_runtime_sec": perf["semantic_matrix_runtime_sec"],
                "semantic_runtime_sec": perf["semantic_runtime_sec"],
                "scene_runtime_sec": perf["scene_runtime_sec"],
                "cache_reused": bool(perf["cache_manifest"].get("cache_reused")),
                "cache_runtime_sec": perf["cache_manifest"].get("runtime_sec", 0.0),
                "cache_dir": perf["cache_dir"],
            }
        )
        artifact_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_fast_artifact_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "role": "carrier_reliability_rows",
                "path": diag["carrier_reliability_rows"],
                "exists": _project(diag["carrier_reliability_rows"]).exists(),
                "size_bytes": _project(diag["carrier_reliability_rows"]).stat().st_size,
            }
        )
        artifact_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_fast_artifact_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "role": "carrier_filter_casebook_rows",
                "path": diag["carrier_filter_casebook_rows"],
                "exists": _project(diag["carrier_filter_casebook_rows"]).exists(),
                "size_bytes": _project(diag["carrier_filter_casebook_rows"]).stat().st_size,
            }
        )

    metric_path = output_root / "carrier_filter_metric_rows.csv"
    semantic_path = output_root / "semantic_distribution_rows.csv"
    performance_path = output_root / "performance_rows.csv"
    artifact_path = output_root / "artifact_rows.csv"
    _write_csv(metric_path, metric_rows)
    _write_csv(semantic_path, semantic_rows)
    _write_csv(performance_path, performance_rows)
    _write_csv(artifact_path, artifact_rows)
    gate_rows, failure_rows, selected_by_scene = _select_and_gate(metric_rows)
    gate_path = output_root / "gate_rows.csv"
    failure_path = output_root / "failure_rows.csv"
    _write_csv(gate_path, gate_rows)
    _write_csv(failure_path, failure_rows)
    phase3_pass = len(failure_rows) == 0
    summary = {
        "schema_version": "stream4d_v103_phase3_fast_carrier_reliability_filter_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_ENTER_PHASE4_PRIMITIVE_AFFINITY" if phase3_pass else "NO_GO_REPAIR_PHASE3_CARRIER_FILTERING",
        "phase3_pass": phase3_pass,
        "failure_count": len(failure_rows),
        "selected_variant_by_scene": selected_by_scene,
        "variant_ids": [v["variant_id"] for v in variants],
        "variant_family": args.variant_family,
        "evaluated_variant_ids": [v["variant_id"] for v in variants],
        "visible_threshold": VISIBLE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "semantic_delta_local": SEMANTIC_DELTA_LOCAL,
        "semantic_contradiction_threshold": SEMANTIC_CONTRADICTION_THRESHOLD,
        "self_error_sigma_norm": SELF_ERROR_SIGMA_NORM,
        "plan_doc": _rel(PLAN_DOC),
        "truthfulness_note": (
            "Fast Phase3 uses mmap cache and GPU/vectorized projection/semantic operations where available. "
            "Gates remain GT-free; AP is not computed here."
        ),
        "outputs": {
            "summary": _rel(output_root / "summary.json"),
            "carrier_filter_metric_rows": _rel(metric_path),
            "semantic_distribution_rows": _rel(semantic_path),
            "performance_rows": _rel(performance_path),
            "artifact_rows": _rel(artifact_path),
            "gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
        },
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
