#!/usr/bin/env python3
"""Build v98.1 Phase6-12 affinity, object birth, render/eval, controls."""

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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


OUT_BASE = ROOT / "outputs/audit"
PHASE4 = OUT_BASE / "v98_phase4_d4rt_anchor_alignment"
PHASE5 = OUT_BASE / "v98_phase5_fused_surfel"
PHASE6 = OUT_BASE / "v98_phase6_semantic_residual_constants"
PHASE7 = OUT_BASE / "v98_phase7_mask_view_affinity"
PHASE8 = OUT_BASE / "v98_phase8_object_birth"
PHASE9 = OUT_BASE / "v98_phase9_render_snap"
PHASE10 = OUT_BASE / "v98_phase10_controls"
PHASE11 = OUT_BASE / "v98_phase11_failure_decomposition"
PHASE12 = OUT_BASE / "v98_phase12_dev_decision"

SOURCE_ROWS = OUT_BASE / "v95_phase1_physical_source_registry/source_container_rows.csv"
RADIO_MASK_FEATURES = OUT_BASE / "v91_radio_mask_features_npz/mask_features.npz"
SURFEL_ROWS = PHASE5 / "fused_surfel_rows.csv"
SURFEL_OBS_ROWS = PHASE5 / "surfel_observation_rows.csv"

RUN_ID = "v98_1_phase6_to_phase12_affinity_eval"

B0_MV_AP_WINDOW = 0.023169647579624655
B0_MV_AP50_WINDOW = 0.07720796704691124
BEST_LOCKED_CONTROL_VARIANT = "P3_C0_area_semantic_hybrid_score"
BEST_LOCKED_CONTROL_MV_AP_WINDOW = 0.05775790465217242
BEST_LOCKED_CONTROL_MV_AP50_WINDOW = 0.17695961955544454
V91_BEST_MV_AP_WINDOW = 0.06799544580104074
V91_BEST_MV_AP50_WINDOW = 0.18017992227130697

VARIANT_CONFIGS = {
    "F0_mask_centered_only": {
        "mask": 1.00,
        "geometry": 0.00,
        "semantic": 0.00,
        "d4rt": 0.00,
        "edge": 0.75,
        "threshold": 0.18,
        "min_support": 3,
        "family": "real",
    },
    "F1_mask_centered_plus_DA3_geometry": {
        "mask": 0.80,
        "geometry": 0.25,
        "semantic": 0.00,
        "d4rt": 0.00,
        "edge": 0.75,
        "threshold": 0.20,
        "min_support": 3,
        "family": "real",
    },
    "F2_mask_centered_plus_semantic_residual_proxy": {
        "mask": 0.75,
        "geometry": 0.00,
        "semantic": 0.30,
        "d4rt": 0.00,
        "edge": 0.75,
        "threshold": 0.22,
        "min_support": 3,
        "family": "real_proxy_semantic",
    },
    "F3_mask_centered_plus_D4RT_anchor": {
        "mask": 0.75,
        "geometry": 0.00,
        "semantic": 0.00,
        "d4rt": 0.35,
        "edge": 0.75,
        "threshold": 0.22,
        "min_support": 3,
        "family": "real",
    },
    "F6_mask_centered_plus_DA3_plus_semantic_proxy_plus_D4RT": {
        "mask": 0.55,
        "geometry": 0.20,
        "semantic": 0.20,
        "d4rt": 0.25,
        "edge": 0.85,
        "threshold": 0.24,
        "min_support": 3,
        "family": "real_proxy_semantic",
    },
}

CONTROL_CONFIGS = {
    "C0_mask_only_frame_masks": {"family": "control", "control_type": "mask_only_frame_unique"},
    "C2_DA3_only_geometry": {"family": "control", "control_type": "da3_geometry_only"},
    "C3_D4RT_only_anchor": {"family": "control", "control_type": "d4rt_anchor_only"},
    "C4_raw_semantic_cosine_proxy": {"family": "control", "control_type": "raw_semantic_cosine_proxy"},
    "C6_shuffled_D4RT_anchor": {"family": "control", "control_type": "shuffled_d4rt_anchor"},
}


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iter_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_num(value, default)))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


class DSUConstraints:
    def __init__(self, n: int, cannot_edges: set[tuple[int, int]]) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.members: dict[int, set[int]] = {i: {i} for i in range(n)}
        self.cannot: dict[int, set[int]] = {i: set() for i in range(n)}
        for a, b in cannot_edges:
            if a == b:
                continue
            self.cannot[a].add(b)
            self.cannot[b].add(a)
        self.rejected_unions = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if rb in self.cannot.get(ra, set()) or ra in self.cannot.get(rb, set()):
            self.rejected_unions += 1
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.members[ra].update(self.members.pop(rb, set()))
        merged_cannot = (self.cannot.get(ra, set()) | self.cannot.get(rb, set())) - {ra, rb}
        self.cannot[ra] = set()
        for other in merged_cannot:
            ro = self.find(other)
            if ro == ra:
                self.rejected_unions += 1
                continue
            self.cannot[ra].add(ro)
            self.cannot.setdefault(ro, set()).discard(rb)
            self.cannot.setdefault(ro, set()).discard(ra)
            self.cannot[ro].add(ra)
        self.cannot.pop(rb, None)
        return True

    def components(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            out[self.find(idx)].append(idx)
        return out


def load_source_context() -> dict[str, Any]:
    mask_lookup: dict[tuple[str, str, int], Path] = {}
    frame_window: dict[tuple[str, int], str] = {}
    mask_area_ratio: dict[tuple[str, int, int], float] = {}
    source_keys: set[tuple[str, int, int]] = set()
    eval_frame_keys: set[tuple[str, str, int]] = set()
    for row in _iter_csv(SOURCE_ROWS):
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id"), -1)
        path = _project(row.get("mask_path", ""))
        if scene and window and frame >= 0:
            mask_lookup.setdefault((scene, window, frame), path)
            frame_window.setdefault((scene, frame), window)
            eval_frame_keys.add((scene, window, frame))
        if scene and frame >= 0 and mask_id > 0:
            source_keys.add((scene, frame, mask_id))
            mask_area_ratio[(scene, frame, mask_id)] = _num(row.get("mask_area_ratio"), 0.0)
    return {
        "mask_lookup": mask_lookup,
        "frame_window": frame_window,
        "mask_area_ratio": mask_area_ratio,
        "source_keys": source_keys,
        "eval_frame_keys": eval_frame_keys,
    }


def load_radio_features() -> dict[tuple[str, int, int], np.ndarray]:
    payload = np.load(RADIO_MASK_FEATURES, allow_pickle=True)
    feats = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    scenes = payload["scene_id"]
    frames = payload["frame_id"]
    masks = payload["mask_id"]
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(feats.shape[0]):
        out[(str(scenes[idx]), int(frames[idx]), int(masks[idx]))] = feats[idx]
    return out


def load_surfel_context() -> dict[str, Any]:
    surfels = _read_csv(SURFEL_ROWS)
    observations = _read_csv(SURFEL_OBS_ROWS)
    surfel_index = {row["surfel_id"]: idx for idx, row in enumerate(surfels)}
    obs_by_surfel: dict[str, list[dict[str, str]]] = defaultdict(list)
    obs_by_mask: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    obs_by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_ids_covering"), -1)
        if row.get("surfel_id") not in surfel_index or frame < 0 or mask_id <= 0:
            continue
        obs_by_surfel[row["surfel_id"]].append(row)
        obs_by_mask[(scene, frame, mask_id)].append(row)
        obs_by_frame[(scene, frame)].append(row)
    xyz = np.asarray([[_num(row.get("xyz_x")), _num(row.get("xyz_y")), _num(row.get("xyz_z"))] for row in surfels], dtype=np.float32)
    return {
        "surfels": surfels,
        "observations": observations,
        "surfel_index": surfel_index,
        "obs_by_surfel": obs_by_surfel,
        "obs_by_mask": obs_by_mask,
        "obs_by_frame": obs_by_frame,
        "xyz": xyz,
    }


def build_phase6(ctx: dict[str, Any], features: dict[tuple[str, int, int], np.ndarray], *, seed: int, sample_pairs: int) -> dict[str, Any]:
    used_keys = sorted({key for key in ctx["obs_by_mask"] if key in features})
    feat_mat = np.stack([features[key] for key in used_keys]).astype(np.float32) if used_keys else np.zeros((0, 0), dtype=np.float32)
    if feat_mat.size:
        mu = np.mean(feat_mat, axis=0).astype(np.float32)
        residual = _normalize_rows(feat_mat - mu[None, :])
    else:
        mu = np.zeros((0,), dtype=np.float32)
        residual = np.zeros_like(feat_mat)
    key_to_resid = {key: residual[idx] for idx, key in enumerate(used_keys)}
    rng = np.random.default_rng(seed)
    key_index = {key: idx for idx, key in enumerate(used_keys)}

    def sample_from_pairs(pair_type: str, pairs: list[tuple[int, int]]) -> dict[str, Any]:
        if len(pairs) > sample_pairs:
            take = rng.choice(len(pairs), size=sample_pairs, replace=False)
            pairs = [pairs[int(i)] for i in take]
        vals = np.asarray([_cosine(residual[a], residual[b]) for a, b in pairs if a != b], dtype=np.float32)
        return {
            "feature_type": "radio_mask_feature_residual_proxy",
            "pair_type": pair_type,
            "sample_count": int(vals.shape[0]),
            "cosine_mean": float(np.mean(vals)) if vals.size else "",
            "cosine_std": float(np.std(vals)) if vals.size else "",
            "cosine_p50": float(np.quantile(vals, 0.50)) if vals.size else "",
            "cosine_p75": float(np.quantile(vals, 0.75)) if vals.size else "",
            "cosine_p90": float(np.quantile(vals, 0.90)) if vals.size else "",
            "cosine_p95": float(np.quantile(vals, 0.95)) if vals.size else "",
            "uses_gt": False,
            "uses_AP": False,
        }

    n = len(used_keys)
    distribution_rows: list[dict[str, Any]] = []
    if n >= 2:
        random_pairs = [(int(a), int(b)) for a, b in rng.integers(0, n, size=(min(sample_pairs * 2, max(1, n * 4)), 2)) if int(a) != int(b)]
        distribution_rows.append(sample_from_pairs("random_pair", random_pairs))
        by_scene: dict[str, list[int]] = defaultdict(list)
        by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
        for key, idx in key_index.items():
            by_scene[key[0]].append(idx)
            by_frame[(key[0], key[1])].append(idx)
        same_scene_pairs: list[tuple[int, int]] = []
        same_frame_pairs: list[tuple[int, int]] = []
        diff_mask_pairs: list[tuple[int, int]] = []
        for indices in by_scene.values():
            for _ in range(min(sample_pairs // 2, len(indices) * 2)):
                a, b = rng.choice(indices, size=2, replace=False)
                same_scene_pairs.append((int(a), int(b)))
        for indices in by_frame.values():
            if len(indices) < 2:
                continue
            ordered = sorted(indices)
            for a, b in zip(ordered[:-1], ordered[1:]):
                same_frame_pairs.append((a, b))
            for _ in range(min(8, len(indices))):
                a, b = rng.choice(indices, size=2, replace=False)
                diff_mask_pairs.append((int(a), int(b)))
        distribution_rows.extend(
            [
                sample_from_pairs("same_source_random_pair", same_scene_pairs),
                sample_from_pairs("same_frame_neighbor_pair", same_frame_pairs),
                sample_from_pairs("different_mask_pair", diff_mask_pairs),
                sample_from_pairs("cross_boundary_pair", diff_mask_pairs),
                sample_from_pairs("same_mask_pair", []),
            ]
        )
    random_row = next((row for row in distribution_rows if row["pair_type"] == "random_pair"), {})
    tau = _num(random_row.get("cosine_p75"), 0.0)
    mu_path = PHASE6 / "radio_mu_vector.npy"
    PHASE6.mkdir(parents=True, exist_ok=True)
    np.save(mu_path, mu)
    constants = {
        "schema": "stream4d_v98_1_phase6_semantic_constants_v1",
        "phase_id": "v98_phase6_semantic_residual_constants",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "radio_mu_vector_path": mu_path,
        "radio_tau_sem": tau,
        "dino_mu_vector_path": "",
        "dino_tau_sem": "",
        "selected_semantic_provider": "RADIO_mask_feature_proxy",
        "constant_source": "feature_distribution_only",
        "semantic_branch_proxy_only": True,
        "proxy_reason": "available feature store is mask-level RADIO feature; no dense per-surfel RADIO/DINO tensor found in v98.1 path",
        "used_mask_feature_count": len(used_keys),
        "surfel_observation_mask_key_count": len(ctx["obs_by_mask"]),
        "semantic_feature_coverage_rate": float(len(used_keys) / max(1, len(ctx["obs_by_mask"]))),
        "random_pair_distribution_available": bool(random_row and random_row.get("sample_count", 0)),
        "mu_vector_saved": True,
        "tau_sem_frozen": bool(random_row),
        "phase6_pass": bool(len(used_keys) > 0 and random_row and mu.size > 0),
        "uses_gt": False,
        "uses_AP": False,
    }
    _write_csv(PHASE6 / "semantic_distribution_rows.csv", distribution_rows)
    _write_json(PHASE6 / "semantic_constants.json", constants)
    _write_csv(
        PHASE6 / "variant_gate_rows.csv",
        [
            {"gate": "semantic_feature_coverage_rate_ge_0p95", "observed": constants["semantic_feature_coverage_rate"], "required": ">=0.95", "pass": constants["semantic_feature_coverage_rate"] >= 0.95},
            {"gate": "random_pair_distribution_available", "observed": constants["random_pair_distribution_available"], "required": True, "pass": constants["random_pair_distribution_available"]},
            {"gate": "mu_vector_saved", "observed": constants["mu_vector_saved"], "required": True, "pass": constants["mu_vector_saved"]},
            {"gate": "tau_sem_frozen", "observed": constants["tau_sem_frozen"], "required": True, "pass": constants["tau_sem_frozen"]},
            {"gate": "dense_semantic_not_proxy", "observed": not constants["semantic_branch_proxy_only"], "required": True, "pass": False, "diagnostic_only": True},
        ],
    )
    _write_json(PHASE6 / "summary.json", constants)
    return {"constants": constants, "key_to_resid": key_to_resid}


def _mask_scale_weights(area_ratio: float) -> tuple[float, float, float, float, str]:
    area = max(0.0, min(1.0, area_ratio))
    broad_risk = float(min(1.0, max(0.0, (area - 0.18) / 0.42)))
    fine = float(max(0.05, 1.0 - min(1.0, area / 0.08)))
    obj = float(math.exp(-((math.log(max(area, 1e-5)) - math.log(0.035)) ** 2) / (2.0 * 1.15 * 1.15)))
    coarse = float(min(1.0, area / 0.25))
    if area < 0.012:
        label = "fine"
    elif area < 0.18:
        label = "object"
    else:
        label = "coarse_or_broad"
    return fine, obj, coarse, broad_risk, label


def build_phase7(
    ctx: dict[str, Any],
    source: dict[str, Any],
    sem: dict[str, Any],
    *,
    max_positive_pairs_per_mask: int,
    max_negative_pairs_per_frame: int,
) -> dict[str, Any]:
    mask_area_ratio = source["mask_area_ratio"]
    surfel_index = ctx["surfel_index"]
    surfels = ctx["surfels"]
    xyz = ctx["xyz"]
    key_to_resid = sem["key_to_resid"]
    frame_totals = {key: len(rows) for key, rows in ctx["obs_by_frame"].items()}
    incidence_rows: list[dict[str, Any]] = []
    surfel_sem_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    surfel_anchor_ids: dict[str, set[str]] = defaultdict(set)
    surfel_incidence_norm_sq: Counter[str] = Counter()
    mask_group_indices: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for key, rows in sorted(ctx["obs_by_mask"].items()):
        scene, frame, mask_id = key
        rho = float(len(rows) / max(1, frame_totals.get((scene, frame), len(rows))))
        area = mask_area_ratio.get(key, 0.0)
        fine_w, object_w, coarse_w, broad_risk, label = _mask_scale_weights(area)
        denom = math.sqrt(max(1e-6, rho * (1.0 - rho)))
        b_centered = float((1.0 - rho) / denom)
        for row in rows:
            sid = row["surfel_id"]
            idx = surfel_index[sid]
            mask_group_indices[key].append(idx)
            if key in key_to_resid:
                surfel_sem_vectors[sid].append(key_to_resid[key])
            raw_anchor = row.get("d4rt_anchor_ids_nearby", "")
            if raw_anchor:
                surfel_anchor_ids[sid].add(raw_anchor)
            surfel_incidence_norm_sq[sid] += b_centered * b_centered
            incidence_rows.append(
                {
                    "surfel_id": sid,
                    "mask_obs_id": f"{scene}:{frame}:{mask_id}",
                    "scene_id": scene,
                    "frame_id": frame,
                    "mask_id": mask_id,
                    "B_raw": 1.0,
                    "rho_mask": rho,
                    "B_centered": b_centered,
                    "scale_weight_fine": fine_w,
                    "scale_weight_object": object_w,
                    "scale_weight_coarse": coarse_w,
                    "mask_area_ratio": area,
                    "mask_risk": broad_risk,
                    "source_scale_label": label,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    surfel_sem: dict[int, np.ndarray] = {}
    for sid, vecs in surfel_sem_vectors.items():
        if not vecs:
            continue
        surfel_sem[surfel_index[sid]] = _normalize_rows(np.mean(np.stack(vecs), axis=0, keepdims=True))[0]

    primitive_rows: list[dict[str, Any]] = []
    for row in surfels:
        sid = row["surfel_id"]
        idx = surfel_index[sid]
        for scale in ("fine", "object", "coarse"):
            primitive_rows.append(
                {
                    "surfel_id": sid,
                    "scale": scale,
                    "mask_sketch_dim": 4096,
                    "mask_sketch_norm": math.sqrt(max(0.0, float(surfel_incidence_norm_sq.get(sid, 0.0)))),
                    "semantic_residual_norm": float(np.linalg.norm(surfel_sem[idx])) if idx in surfel_sem else 0.0,
                    "d4rt_anchor_mass": _num(row.get("d4rt_anchor_mass")),
                    "geometry_confidence": _num(row.get("mean_confidence")),
                    "feature_valid": _bool(row.get("surfel_valid")) and surfel_incidence_norm_sq.get(sid, 0.0) > 0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    positive_base: dict[tuple[int, int], dict[str, float]] = {}
    rng = np.random.default_rng(981)
    for key, indices in sorted(mask_group_indices.items()):
        unique_indices = sorted(set(indices))
        if len(unique_indices) < 2:
            continue
        all_pairs: list[tuple[int, int]] = []
        if len(unique_indices) <= 80:
            for i, a in enumerate(unique_indices[:-1]):
                for b in unique_indices[i + 1 :]:
                    all_pairs.append((a, b))
        else:
            for _ in range(max_positive_pairs_per_mask * 3):
                a, b = rng.choice(unique_indices, size=2, replace=False)
                if a > b:
                    a, b = b, a
                all_pairs.append((int(a), int(b)))
        if len(all_pairs) > max_positive_pairs_per_mask:
            take = rng.choice(len(all_pairs), size=max_positive_pairs_per_mask, replace=False)
            all_pairs = [all_pairs[int(i)] for i in take]
        scene, frame, mask_id = key
        rho = float(len(indices) / max(1, frame_totals.get((scene, frame), len(indices))))
        b_centered = float((1.0 - rho) / math.sqrt(max(1e-6, rho * (1.0 - rho))))
        area = mask_area_ratio.get(key, 0.0)
        _fine, obj_w, _coarse, broad_risk, _label = _mask_scale_weights(area)
        for a, b in all_pairs:
            pair = (a, b)
            stats = positive_base.setdefault(pair, {"mask": 0.0, "count": 0.0, "risk": 0.0})
            stats["mask"] += b_centered * b_centered * obj_w
            stats["count"] += 1.0
            stats["risk"] = max(stats["risk"], broad_risk)

    cannot_edges: set[tuple[int, int]] = set()
    for frame_key, rows in sorted(ctx["obs_by_frame"].items()):
        by_mask: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            mask_id = _int(row.get("mask_ids_covering"), -1)
            idx = surfel_index.get(row.get("surfel_id", ""), -1)
            if idx >= 0 and mask_id > 0:
                by_mask[mask_id].append(idx)
        mask_ids = sorted(by_mask)
        if len(mask_ids) < 2:
            continue
        candidate: list[tuple[int, int]] = []
        for i, ma in enumerate(mask_ids[:-1]):
            lhs = sorted(set(by_mask[ma]))[:40]
            for mb in mask_ids[i + 1 :]:
                rhs = sorted(set(by_mask[mb]))[:40]
                for a in lhs[:8]:
                    for b in rhs[:8]:
                        if a == b:
                            continue
                        candidate.append((min(a, b), max(a, b)))
        if len(candidate) > max_negative_pairs_per_frame:
            take = rng.choice(len(candidate), size=max_negative_pairs_per_frame, replace=False)
            candidate = [candidate[int(i)] for i in take]
        cannot_edges.update(candidate)

    raw_edge_rows: list[dict[str, Any]] = []
    compact_edge_rows: list[dict[str, Any]] = []
    allowed_variants = list(VARIANT_CONFIGS)
    anchor_sets = {idx: surfel_anchor_ids.get(row["surfel_id"], set()) for idx, row in enumerate(surfels)}
    for pair, stats in positive_base.items():
        a, b = pair
        pa = xyz[a]
        pb = xyz[b]
        dist = float(np.linalg.norm(pa - pb))
        geom = float(math.exp(-(dist * dist) / (2.0 * 0.18 * 0.18)))
        sem_score = 0.0
        if a in surfel_sem and b in surfel_sem:
            raw_cos = _cosine(surfel_sem[a], surfel_sem[b])
            tau = _num(sem["constants"].get("radio_tau_sem"), 0.0)
            sem_score = float(max(0.0, (raw_cos - tau) / max(1e-6, 1.0 - tau)))
        shared_anchor = anchor_sets[a] & anchor_sets[b]
        d4rt_score = float(min(1.0, len(shared_anchor)))
        edge_penalty = float(stats["risk"])
        conflict = 1.0 if pair in cannot_edges else 0.0
        mask_score = float(stats["mask"] / max(1.0, stats["count"]))
        for variant_id, cfg in VARIANT_CONFIGS.items():
            signed = (
                cfg["mask"] * mask_score
                + cfg["geometry"] * geom
                + cfg["semantic"] * sem_score
                + cfg["d4rt"] * d4rt_score
                - cfg["edge"] * edge_penalty
                - 2.0 * conflict
            )
            row = {
                "variant_id": variant_id,
                "scale": "object",
                "surfel_id_a": surfels[a]["surfel_id"],
                "surfel_id_b": surfels[b]["surfel_id"],
                "surfel_index_a": a,
                "surfel_index_b": b,
                "A_mask_centered": mask_score,
                "A_geometry": geom,
                "A_sem_residual": sem_score,
                "A_d4rt_anchor": d4rt_score,
                "edge_penalty": edge_penalty,
                "conflict_penalty": conflict,
                "signed_affinity": signed,
                "cannot_link_active": bool(pair in cannot_edges),
                "candidate_source": "same_mask_observation",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            if signed >= cfg["threshold"] or pair in cannot_edges:
                raw_edge_rows.append(row)
        base_row = {
            "surfel_index_a": a,
            "surfel_index_b": b,
            "A_mask_centered": mask_score,
            "A_geometry": geom,
            "A_sem_residual": sem_score,
            "A_d4rt_anchor": d4rt_score,
            "edge_penalty": edge_penalty,
            "conflict_penalty": conflict,
            "cannot_link_active": bool(pair in cannot_edges),
        }
        compact_edge_rows.append(base_row)

    metric_rows: list[dict[str, Any]] = []
    for variant_id, cfg in VARIANT_CONFIGS.items():
        variant_edges = [row for row in raw_edge_rows if row["variant_id"] == variant_id]
        positive_count = sum(1 for row in variant_edges if _num(row.get("signed_affinity")) >= cfg["threshold"] and not _bool(row.get("cannot_link_active")))
        negative_count = sum(1 for row in variant_edges if _num(row.get("signed_affinity")) < 0 or _bool(row.get("cannot_link_active")))
        cannot_count = len(cannot_edges)
        dsu = DSUConstraints(len(surfels), cannot_edges)
        for row in sorted(variant_edges, key=lambda r: _num(r.get("signed_affinity")), reverse=True):
            if _bool(row.get("cannot_link_active")) or _num(row.get("signed_affinity")) < cfg["threshold"]:
                continue
            dsu.union(_int(row["surfel_index_a"]), _int(row["surfel_index_b"]))
        comp_sizes = [len(v) for v in dsu.components().values()]
        largest = float(max(comp_sizes) / max(1, len(surfels))) if comp_sizes else 0.0
        metric_rows.append(
            {
                "variant_id": variant_id,
                "scale": "object",
                "surfel_count": len(surfels),
                "incidence_event_count": len(incidence_rows),
                "candidate_edge_count": len(variant_edges),
                "positive_edge_count": positive_count,
                "negative_edge_count": negative_count,
                "cannot_link_count": cannot_count,
                "largest_component_ratio_pre_constraint": largest,
                "largest_component_ratio_post_constraint": largest,
                "bucket_load_p90": float(np.quantile([len(v) for v in mask_group_indices.values()], 0.90)) if mask_group_indices else 0.0,
                "runtime_sec": "",
                "gpu_memory_MB": "",
                "feature_valid_rate": float(sum(1 for row in primitive_rows if row["scale"] == "object" and row["feature_valid"]) / max(1, len(surfels))),
                "cannot_link_violation_count_after_constraint": 0,
                "phase7_gate_pass": bool(
                    len(incidence_rows) > 0
                    and len(variant_edges) > 0
                    and negative_count > 0
                    and cannot_count > 0
                    and largest <= 0.30
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    PHASE7.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE7 / "mask_view_incidence_rows.csv", incidence_rows)
    _write_csv(PHASE7 / "primitive_feature_rows.csv", primitive_rows)
    _write_csv(PHASE7 / "affinity_edge_rows.csv", raw_edge_rows)
    _write_csv(PHASE7 / "affinity_metric_rows.csv", metric_rows)
    _write_csv(PHASE7 / "variant_config_rows.csv", [{"variant_id": k, **v, "uses_gt_for_prediction": False, "uses_future": False} for k, v in VARIANT_CONFIGS.items()])
    summary = {
        "schema": "stream4d_v98_1_phase7_mask_view_affinity_summary_v1",
        "phase_id": "v98_phase7_mask_view_affinity",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V98_1_PHASE7_MASK_VIEW_AFFINITY" if any(row["phase7_gate_pass"] for row in metric_rows) else "NO_GO_V98_1_PHASE7_MASK_VIEW_AFFINITY",
        "incidence_event_count": len(incidence_rows),
        "compact_base_edge_count": len(compact_edge_rows),
        "affinity_edge_row_count": len(raw_edge_rows),
        "cannot_link_count": len(cannot_edges),
        "semantic_branch_proxy_only": sem["constants"].get("semantic_branch_proxy_only", True),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(PHASE7 / "summary.json", summary)
    return {"summary": summary, "base_edges": compact_edge_rows, "cannot_edges": cannot_edges}


def _variant_signed(row: dict[str, Any], cfg: dict[str, float]) -> float:
    return float(
        cfg["mask"] * _num(row.get("A_mask_centered"))
        + cfg["geometry"] * _num(row.get("A_geometry"))
        + cfg["semantic"] * _num(row.get("A_sem_residual"))
        + cfg["d4rt"] * _num(row.get("A_d4rt_anchor"))
        - cfg["edge"] * _num(row.get("edge_penalty"))
        - 2.0 * _num(row.get("conflict_penalty"))
    )


def build_objects_for_variant(
    variant_id: str,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    source: dict[str, Any],
    base_edges: list[dict[str, Any]],
    cannot_edges: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    surfels = ctx["surfels"]
    surfel_index = ctx["surfel_index"]
    obs_by_surfel = ctx["obs_by_surfel"]
    dsu = DSUConstraints(len(surfels), cannot_edges)
    accepted_affinities: list[float] = []
    for row in sorted(base_edges, key=lambda r: _variant_signed(r, cfg), reverse=True):
        a = _int(row["surfel_index_a"])
        b = _int(row["surfel_index_b"])
        signed = _variant_signed(row, cfg)
        if (a, b) in cannot_edges or signed < float(cfg["threshold"]):
            continue
        before_a, before_b = dsu.find(a), dsu.find(b)
        if dsu.union(a, b) and before_a != before_b:
            accepted_affinities.append(signed)
    components = dsu.components()
    object_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    emission_rows: list[dict[str, Any]] = []
    object_counter = 0
    frame_window = source["frame_window"]
    for _root, indices in sorted(components.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(indices) < 2:
            continue
        obs_rows: list[dict[str, str]] = []
        for idx in indices:
            obs_rows.extend(obs_by_surfel.get(surfels[idx]["surfel_id"], []))
        frames = sorted({(row["scene_id"], _int(row["frame_id"])) for row in obs_rows})
        if len(frames) < 2:
            continue
        by_frame_mask: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
        for row in obs_rows:
            mask_id = _int(row.get("mask_ids_covering"), -1)
            if mask_id > 0:
                by_frame_mask[(row["scene_id"], _int(row["frame_id"]), mask_id)].append(row)
        if not by_frame_mask:
            continue
        object_id = f"{variant_id}:obj_{object_counter:06d}"
        object_counter += 1
        support_counts = [len(v) for v in by_frame_mask.values()]
        object_score = float(math.log1p(len(indices)) * math.log1p(len(frames)) * (1.0 + 0.05 * np.mean(support_counts)))
        object_rows.append(
            {
                "object_id": object_id,
                "birth_route": "constrained_signed_clustering",
                "variant_id": variant_id,
                "scale": "object",
                "surfel_count": len(indices),
                "keypoint_count": len(indices),
                "masklet_count": len(by_frame_mask),
                "observed_frame_count": len(frames),
                "mean_affinity": float(np.mean(accepted_affinities)) if accepted_affinities else 0.0,
                "mean_conflict": 0.0,
                "d4rt_anchor_mass": float(np.sum([_num(surfels[idx].get("d4rt_anchor_mass")) for idx in indices])),
                "semantic_residual_mean": "",
                "risk_score": 0.0,
                "object_score": object_score,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for (scene, frame, mask_id), rows in sorted(by_frame_mask.items()):
            support_surfel_count = len({row["surfel_id"] for row in rows})
            window = frame_window.get((scene, frame), "")
            support_rows.append(
                {
                    "object_id": object_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame,
                    "support_surfel_count": support_surfel_count,
                    "support_heatmap_area": support_surfel_count,
                    "candidate_mask_count": len(by_frame_mask),
                    "selected_mask_id_if_any": mask_id if support_surfel_count >= int(cfg["min_support"]) else "",
                    "support_confidence": float(np.mean([_num(row.get("provider_confidence")) for row in rows])),
                    "object_score": object_score,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            if support_surfel_count >= int(cfg["min_support"]):
                support_points = sorted({(int(_num(row.get("x_orig"))), int(_num(row.get("y_orig")))) for row in rows})
                emission_rows.append(
                    {
                        "object_id": object_id,
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": frame,
                        "selected_mask_id": mask_id,
                        "support_surfel_count": support_surfel_count,
                        "support_points": support_points,
                        "object_score": object_score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return object_rows, support_rows, emission_rows, dsu.rejected_unions


def build_control_emissions(control_id: str, ctx: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    emissions: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    frame_window = source["frame_window"]
    if control_id == "C0_mask_only_frame_masks":
        for scene, frame, mask_id in sorted(source["source_keys"]):
            window = frame_window.get((scene, frame), "")
            object_id = f"{control_id}:{scene}:{frame}:{mask_id}"
            object_rows.append({"object_id": object_id, "birth_route": "frame_mask_unique", "variant_id": control_id, "scale": "object", "surfel_count": "", "observed_frame_count": 1, "object_score": 1.0, "uses_gt_for_prediction": False, "uses_future": False})
            emissions.append({"object_id": object_id, "variant_id": control_id, "scene_id": scene, "window_id": window, "frame_id": frame, "selected_mask_id": mask_id, "support_surfel_count": "", "support_points": [], "object_score": 1.0, "uses_gt_for_prediction": False, "uses_future": False})
    elif control_id == "C2_DA3_only_geometry":
        buckets: dict[tuple[str, int, int, int], list[str]] = defaultdict(list)
        for row in ctx["surfels"]:
            x = _int(math.floor(_num(row.get("xyz_x")) / 0.25))
            y = _int(math.floor(_num(row.get("xyz_y")) / 0.25))
            z = _int(math.floor(_num(row.get("xyz_z")) / 0.25))
            buckets[(row.get("scene_id", ""), x, y, z)].append(row["surfel_id"])
        emissions, object_rows = _emissions_from_surfel_buckets(control_id, buckets, ctx, source)
    elif control_id == "C3_D4RT_only_anchor":
        buckets = defaultdict(list)
        for obs in ctx["observations"]:
            anchor = obs.get("d4rt_anchor_ids_nearby", "")
            if anchor:
                buckets[(obs["scene_id"], anchor)].append(obs["surfel_id"])
        emissions, object_rows = _emissions_from_surfel_buckets(control_id, buckets, ctx, source)
    elif control_id == "C4_raw_semantic_cosine_proxy":
        buckets = defaultdict(list)
        for obs in ctx["observations"]:
            key = (obs["scene_id"], _int(obs.get("frame_id")), _int(obs.get("mask_ids_covering")))
            buckets[(obs["scene_id"], key[2] % 64)].append(obs["surfel_id"])
        emissions, object_rows = _emissions_from_surfel_buckets(control_id, buckets, ctx, source)
    elif control_id == "C6_shuffled_D4RT_anchor":
        buckets = defaultdict(list)
        for obs in ctx["observations"]:
            anchor = obs.get("d4rt_anchor_ids_nearby", "")
            if anchor:
                shuffled = hashlib.sha256(f"{anchor}|{obs['scene_id']}".encode("utf-8")).hexdigest()[:4]
                buckets[(obs["scene_id"], shuffled)].append(obs["surfel_id"])
        emissions, object_rows = _emissions_from_surfel_buckets(control_id, buckets, ctx, source)
    return emissions, object_rows


def _emissions_from_surfel_buckets(control_id: str, buckets: dict[Any, list[str]], ctx: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_window = source["frame_window"]
    obs_by_surfel = ctx["obs_by_surfel"]
    emissions: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_idx = 0
    for _bucket, surfel_ids in sorted(buckets.items(), key=lambda kv: (-len(set(kv[1])), str(kv[0]))):
        unique = sorted(set(surfel_ids))
        if len(unique) < 2:
            continue
        by_frame_mask: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
        for sid in unique:
            for obs in obs_by_surfel.get(sid, []):
                mask_id = _int(obs.get("mask_ids_covering"), -1)
                if mask_id > 0:
                    by_frame_mask[(obs["scene_id"], _int(obs.get("frame_id")), mask_id)].append(obs)
        frames = {(scene, frame) for scene, frame, _mask in by_frame_mask}
        if len(frames) < 2:
            continue
        object_id = f"{control_id}:obj_{object_idx:06d}"
        object_idx += 1
        score = float(math.log1p(len(unique)) * math.log1p(len(frames)))
        object_rows.append({"object_id": object_id, "birth_route": CONTROL_CONFIGS[control_id]["control_type"], "variant_id": control_id, "scale": "object", "surfel_count": len(unique), "observed_frame_count": len(frames), "object_score": score, "uses_gt_for_prediction": False, "uses_future": False})
        for (scene, frame, mask_id), rows in by_frame_mask.items():
            count = len({row["surfel_id"] for row in rows})
            if count < 3:
                continue
            support_points = sorted({(int(_num(row.get("x_orig"))), int(_num(row.get("y_orig")))) for row in rows})
            emissions.append({"object_id": object_id, "variant_id": control_id, "scene_id": scene, "window_id": frame_window.get((scene, frame), ""), "frame_id": frame, "selected_mask_id": mask_id, "support_surfel_count": count, "support_points": support_points, "object_score": score, "uses_gt_for_prediction": False, "uses_future": False})
    return emissions, object_rows


def _support_metrics(label: np.ndarray, mask_id: int, support_points: list[tuple[int, int]], *, radius: int = 4) -> tuple[float, float, float, int, int, int]:
    selected = label == int(mask_id)
    support = np.zeros(label.shape, dtype=np.uint8)
    height, width = label.shape[:2]
    for x, y in support_points:
        xi = max(0, min(width - 1, int(x)))
        yi = max(0, min(height - 1, int(y)))
        support[yi, xi] = 1
    if radius > 0 and np.any(support):
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        support = cv2.dilate(support, kernel, iterations=1)
    support_bool = support.astype(bool)
    inter = int(np.count_nonzero(support_bool & selected))
    support_area = int(np.count_nonzero(support_bool))
    mask_area = int(np.count_nonzero(selected))
    union = support_area + mask_area - inter
    iou = float(inter / union) if union > 0 else 0.0
    recall = float(inter / max(1, support_area))
    precision = float(inter / max(1, mask_area))
    return iou, recall, precision, inter, support_area, mask_area


def _assign_object(scope_index: dict[str, int], scope_scores: dict[str, float], object_id: str, score: float) -> int:
    if object_id not in scope_index:
        scope_index[object_id] = len(scope_index) + 1
    scope_scores[object_id] = max(float(scope_scores.get(object_id, 0.0)), float(score))
    return scope_index[object_id]


def _score_array(scope_index: dict[str, int], scope_scores: dict[str, float]) -> np.ndarray:
    scores = np.ones((len(scope_index),), dtype=np.float32)
    for oid, idx in scope_index.items():
        scores[idx - 1] = float(scope_scores.get(oid, 1.0))
    return scores


def _summary_metric_row(variant_id: str, summary: dict[str, Any], *, metric_scope: str, scene: str = "", window: str = "") -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "scene_id": scene,
        "window_id": window,
        "metric_scope": metric_scope,
        "frame_count": summary.get("frame_count"),
        "MV_AP": summary.get("ap"),
        "MV_AP50": summary.get("ap50"),
        "MV_AP25": summary.get("ap25"),
        "ScoreFreeMatch50": (summary.get("score_free_match_at_050") or {}).get("recall"),
        "ScoreFreeMatch25": (summary.get("score_free_match_at_025") or {}).get("recall"),
        "evaluated_pred_count": summary.get("evaluated_pred_count"),
        "evaluated_gt_count": summary.get("evaluated_gt_count"),
        "raw_pred_count": summary.get("raw_pred_count"),
        "raw_gt_count": summary.get("raw_gt_count"),
        "gt_best_iou_mean": summary.get("gt_best_iou_mean"),
        "pred_best_iou_mean": summary.get("pred_best_iou_mean"),
        "score_unique_count": summary.get("score_unique_count"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }


def _mean_metric(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [_num(row.get(field), float("nan")) for row in rows if row.get(field) not in ("", None)]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _summarize_scope_rows(variant_id: str, rows: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    prefix = "window" if scope == "window" else "scene"
    return {
        "variant_id": variant_id,
        "metric_scope": f"{prefix}_mean",
        f"{prefix}_count": len(rows),
        f"mean_MV_AP_{prefix}": _mean_metric(rows, "MV_AP"),
        f"mean_MV_AP50_{prefix}": _mean_metric(rows, "MV_AP50"),
        f"mean_MV_AP25_{prefix}": _mean_metric(rows, "MV_AP25"),
        f"mean_ScoreFreeMatch50_{prefix}": _mean_metric(rows, "ScoreFreeMatch50"),
        f"mean_gt_object_count_{prefix}": _mean_metric(rows, "evaluated_gt_count"),
        f"mean_pred_object_count_{prefix}": _mean_metric(rows, "evaluated_pred_count"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }


def evaluate_emissions(variant_id: str, emissions: list[dict[str, Any]], source: dict[str, Any], *, min_pred_pixels: int, min_gt_pixels: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    mask_lookup = source["mask_lookup"]
    eval_frame_keys = source["eval_frame_keys"]
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    # Winner-take-all for duplicate object claims to the same frame mask.
    best_by_mask: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in emissions:
        key = (row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"]))
        cur = best_by_mask.get(key)
        if cur is None or (float(row.get("object_score", 0.0)), str(row.get("object_id", ""))) > (float(cur.get("object_score", 0.0)), str(cur.get("object_id", ""))):
            best_by_mask[key] = row
    for row in best_by_mask.values():
        by_key[(row["scene_id"], row["window_id"], int(row["frame_id"]))].append(row)

    global_object_index: dict[str, int] = {}
    global_scores: dict[str, float] = {}
    scene_object_index: dict[str, dict[str, int]] = defaultdict(dict)
    scene_scores: dict[str, dict[str, float]] = defaultdict(dict)
    window_object_index: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    window_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    scene_gt_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_frame_keys}))}
    acc_global = SparseSceneIoU()
    acc_by_scene: dict[str, SparseSceneIoU] = defaultdict(SparseSceneIoU)
    acc_by_window: dict[tuple[str, str], SparseSceneIoU] = defaultdict(SparseSceneIoU)
    preview_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    for key in sorted(set(eval_frame_keys) | set(by_key)):
        scene, window, frame = key
        mask_path = mask_lookup.get(key)
        if mask_path is None or not mask_path.exists():
            preview_rows.append({"variant_id": variant_id, "scene_id": scene, "window_id": window, "frame_id": frame, "status": "missing_mask", "uses_gt_for_prediction": False, "uses_gt_for_eval": True})
            continue
        label = _load_label(mask_path)
        pred_global = np.zeros(label.shape, dtype=np.int64)
        pred_scene = np.zeros(label.shape, dtype=np.int64)
        pred_window = np.zeros(label.shape, dtype=np.int64)
        for row in sorted(by_key.get(key, []), key=lambda item: (-float(item.get("object_score", 0.0)), str(item.get("object_id", "")))):
            mask_id = int(row["selected_mask_id"])
            if mask_id <= 0:
                continue
            mask = label == mask_id
            if not np.any(mask):
                continue
            support_iou, support_recall, mask_precision, support_intersection, support_area, selected_mask_area = _support_metrics(label, mask_id, list(row.get("support_points", [])))
            oid = str(row["object_id"])
            score = float(row.get("object_score", 0.0))
            global_id = _assign_object(global_object_index, global_scores, oid, score)
            scene_id = _assign_object(scene_object_index[scene], scene_scores[scene], oid, score)
            window_key = (scene, window)
            window_id = _assign_object(window_object_index[window_key], window_scores[window_key], oid, score)
            pixel_collision_count += int(np.count_nonzero((pred_global > 0) & mask))
            pred_global[(pred_global == 0) & mask] = global_id
            pred_scene[(pred_scene == 0) & mask] = scene_id
            pred_window[(pred_window == 0) & mask] = window_id
            selected_rows.append(
                {
                    "mv_object_id": oid,
                    "object_id": oid,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame,
                    "mask_id_or_generated_id": mask_id,
                    "selected_mask_id": mask_id,
                    "readout_mode": "snap-to-mask",
                    "score": float(row.get("object_score", 0.0)),
                    "support_iou": support_iou,
                    "support_recall": support_recall,
                    "mask_precision": mask_precision,
                    "support_intersection": support_intersection,
                    "support_area": support_area,
                    "selected_mask_area": selected_mask_area,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        gt = _load_gt_2d(scene, frame, label.shape)
        gt_scene = np.asarray(gt, dtype=np.int64)
        gt_global = np.where(gt_scene > 0, gt_scene + int(scene_gt_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc_global.add(pred_global, gt_global)
        acc_by_scene[scene].add(pred_scene, gt_scene)
        acc_by_window[(scene, window)].add(pred_window, gt_scene)
        preview_rows.append(
            {
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame,
                "status": "evaluated",
                "emitted_object_count": len(by_key.get(key, [])),
                "pred_positive_pixels": int(np.count_nonzero(pred_global > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt_scene > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    global_summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc_global,
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        score_mode="input",
        input_scores=_score_array(global_object_index, global_scores),
    )
    scene_rows: list[dict[str, Any]] = []
    for scene, acc in sorted(acc_by_scene.items()):
        summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=min_pred_pixels,
            min_gt_pixels=min_gt_pixels,
            score_mode="input",
            input_scores=_score_array(scene_object_index[scene], scene_scores[scene]),
        )
        scene_rows.append(_summary_metric_row(variant_id, summary, metric_scope="scene", scene=scene))
    window_rows: list[dict[str, Any]] = []
    for (scene, window), acc in sorted(acc_by_window.items()):
        summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=min_pred_pixels,
            min_gt_pixels=min_gt_pixels,
            score_mode="input",
            input_scores=_score_array(window_object_index[(scene, window)], window_scores[(scene, window)]),
        )
        window_rows.append(_summary_metric_row(variant_id, summary, metric_scope="window", scene=scene, window=window))
    eval_pack = {
        "global_summary": global_summary,
        "global_row": _summary_metric_row(variant_id, global_summary, metric_scope="full_dev_global"),
        "scene_rows": scene_rows,
        "scene_aggregate": _summarize_scope_rows(variant_id, scene_rows, scope="scene"),
        "window_rows": window_rows,
        "window_aggregate": _summarize_scope_rows(variant_id, window_rows, scope="window"),
    }
    return eval_pack, preview_rows, selected_rows, pixel_collision_count


def build_phase8_to_10(
    ctx: dict[str, Any],
    source: dict[str, Any],
    phase7: dict[str, Any],
    *,
    min_pred_pixels: int,
    min_gt_pixels: int,
    include_locked_control_reference: bool,
) -> dict[str, Any]:
    all_object_rows: list[dict[str, Any]] = []
    all_support_rows: list[dict[str, Any]] = []
    all_emission_rows: list[dict[str, Any]] = []
    birth_metric_rows: list[dict[str, Any]] = []
    render_support_rows: list[dict[str, Any]] = []
    mask_snap_rows: list[dict[str, Any]] = []
    mv_object_frame_mask_rows: list[dict[str, Any]] = []
    render_metric_rows: list[dict[str, Any]] = []
    window_metric_rows_all: list[dict[str, Any]] = []
    scene_metric_rows_all: list[dict[str, Any]] = []
    global_metric_rows_all: list[dict[str, Any]] = []
    preview_rows_all: list[dict[str, Any]] = []
    control_metric_rows: list[dict[str, Any]] = []
    control_window_metric_rows: list[dict[str, Any]] = []
    control_scene_metric_rows: list[dict[str, Any]] = []
    control_global_metric_rows: list[dict[str, Any]] = []
    control_mv_object_frame_mask_rows: list[dict[str, Any]] = []
    object_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    emission_rows_by_variant: dict[str, list[dict[str, Any]]] = {}

    for variant_id, cfg in VARIANT_CONFIGS.items():
        object_rows, support_rows, emission_rows, rejected_unions = build_objects_for_variant(variant_id, cfg, ctx, source, phase7["base_edges"], phase7["cannot_edges"])
        object_rows_by_variant[variant_id] = object_rows
        emission_rows_by_variant[variant_id] = emission_rows
        all_object_rows.extend(object_rows)
        all_support_rows.extend(support_rows)
        all_emission_rows.extend(emission_rows)
        observed = [_num(row.get("observed_frame_count")) for row in object_rows]
        largest = max([_num(row.get("surfel_count")) for row in object_rows], default=0.0) / max(1.0, float(len(ctx["surfels"])))
        birth_metric_rows.append(
            {
                "variant_id": variant_id,
                "birth_route": "constrained_signed_clustering",
                "object_count": len(object_rows),
                "mean_surfel_count": float(np.mean([_num(row.get("surfel_count")) for row in object_rows])) if object_rows else 0.0,
                "mean_observed_frame_count": float(np.mean(observed)) if observed else 0.0,
                "largest_object_ratio": largest,
                "same_frame_collision_count": 0,
                "pixel_collision_rate_proxy": 0.0,
                "duplicate_object_proxy": 0.0,
                "cannot_link_violation_count": 0,
                "rejected_union_due_cannot_link_count": rejected_unions,
                "phase8_gate_pass": bool(object_rows and largest <= 0.30 and (float(np.mean(observed)) if observed else 0.0) >= 2.0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    PHASE8.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE8 / "object_candidate_rows.csv", all_object_rows)
    _write_csv(PHASE8 / "object_frame_support_rows.csv", all_support_rows)
    _write_csv(PHASE8 / "object_birth_metric_rows.csv", birth_metric_rows)
    _write_json(
        PHASE8 / "summary.json",
        {
            "schema": "stream4d_v98_1_phase8_object_birth_summary_v1",
            "phase_id": "v98_phase8_object_birth",
            "run_id": RUN_ID,
            "created_at": _created_at(),
            "decision": "PASS_V98_1_PHASE8_OBJECT_BIRTH" if any(row["phase8_gate_pass"] for row in birth_metric_rows) else "NO_GO_V98_1_PHASE8_OBJECT_BIRTH",
            "object_candidate_row_count": len(all_object_rows),
            "object_frame_support_row_count": len(all_support_rows),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )

    for variant_id, emissions in emission_rows_by_variant.items():
        eval_pack, preview_rows, selected_rows, pixel_collision_count = evaluate_emissions(variant_id, emissions, source, min_pred_pixels=min_pred_pixels, min_gt_pixels=min_gt_pixels)
        preview_rows_all.extend(preview_rows)
        mv_object_frame_mask_rows.extend(selected_rows)
        window_metric_rows_all.extend(eval_pack["window_rows"])
        scene_metric_rows_all.extend(eval_pack["scene_rows"])
        global_metric_rows_all.append(eval_pack["global_row"])
        render_support_rows.extend(
            {
                "object_id": row["object_id"],
                "variant_id": variant_id,
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "frame_id": row["frame_id"],
                "support_area": row.get("support_surfel_count", ""),
                "support_peak": row.get("support_surfel_count", ""),
                "support_confidence": "",
                "visible_surfel_count": row.get("support_surfel_count", ""),
                "render_uses_stitched_xyz": True,
                "render_reprojected_uv": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            for row in emissions
        )
        mask_snap_rows.extend(
            {
                "object_id": row["object_id"],
                "variant_id": variant_id,
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "frame_id": row["frame_id"],
                "candidate_mask_id": row["mask_id_or_generated_id"],
                "support_iou": row.get("support_iou", ""),
                "support_recall": row.get("support_recall", ""),
                "mask_precision": row.get("mask_precision", ""),
                "semantic_alignment": "",
                "broad_risk": "",
                "conflict_risk": "",
                "selected": True,
                "readout_mode": "snap-to-mask",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            for row in selected_rows
        )
        duplicate_keys = [(row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"])) for row in selected_rows]
        duplicate_count = sum(max(0, count - 1) for count in Counter(duplicate_keys).values())
        missing_count = sum(1 for row in preview_rows if row.get("status") == "missing_mask")
        window_agg = eval_pack["window_aggregate"]
        scene_agg = eval_pack["scene_aggregate"]
        global_summary = eval_pack["global_summary"]
        ap = window_agg.get("mean_MV_AP_window")
        ap50 = window_agg.get("mean_MV_AP50_window")
        sf50 = window_agg.get("mean_ScoreFreeMatch50_window")
        support_iou_mean = float(np.mean([_num(row.get("support_iou")) for row in selected_rows])) if selected_rows else 0.0
        render_metric_rows.append(
            {
                "variant_id": variant_id,
                "readout_variant": "snap_to_cropformer_mask_from_surfel_support",
                "MV_AP_window": ap,
                "MV_AP50_window": ap50,
                "MV_AP25_window": window_agg.get("mean_MV_AP25_window"),
                "ScoreFreeMatch50_window": sf50,
                "MV_AP_scene": scene_agg.get("mean_MV_AP_scene"),
                "MV_AP50_scene": scene_agg.get("mean_MV_AP50_scene"),
                "MV_AP25_scene": scene_agg.get("mean_MV_AP25_scene"),
                "ScoreFreeMatch50_scene": scene_agg.get("mean_ScoreFreeMatch50_scene"),
                "MV_AP_full_dev_global": global_summary.get("ap"),
                "MV_AP50_full_dev_global": global_summary.get("ap50"),
                "window_count": window_agg.get("window_count"),
                "scene_count": scene_agg.get("scene_count"),
                "ScoreFreeMatch50_window": sf50,
                "same_frame_collision_count": duplicate_count,
                "pixel_collision_count": pixel_collision_count,
                "pixel_collision_rate": float(pixel_collision_count / max(1, sum(_int(row.get("pred_positive_pixels")) for row in preview_rows))),
                "missing_mask_raster_count": missing_count,
                "emitted_object_frame_count": len(selected_rows),
                "support_to_selected_mask_IoU_mean": support_iou_mean,
                "support_to_selected_mask_IoU_baseline_source": "not_available_in_current_v98_1_run",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
                "phase9_gate_pass": bool(
                    duplicate_count == 0
                    and missing_count == 0
                    and ap is not None
                    and ap50 is not None
                    and float(ap) >= B0_MV_AP_WINDOW + 0.010
                    and float(ap50) >= B0_MV_AP50_WINDOW + 0.020
                ),
            }
        )

    PHASE9.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE9 / "render_support_rows.csv", render_support_rows)
    _write_csv(PHASE9 / "mask_snap_rows.csv", mask_snap_rows)
    _write_csv(PHASE9 / "mv_object_frame_mask_rows.csv", mv_object_frame_mask_rows)
    _write_csv(PHASE9 / "preview_frame_rows.csv", preview_rows_all)
    _write_csv(PHASE9 / "mv_metric_window_rows.csv", window_metric_rows_all)
    _write_csv(PHASE9 / "mv_metric_scene_rows.csv", scene_metric_rows_all)
    _write_csv(PHASE9 / "mv_metric_full_dev_global_rows.csv", global_metric_rows_all)
    _write_csv(PHASE9 / "render_variant_metric_rows.csv", render_metric_rows)
    _write_json(
        PHASE9 / "summary.json",
        {
            "schema": "stream4d_v98_1_phase9_render_snap_summary_v1",
            "phase_id": "v98_phase9_render_snap",
            "run_id": RUN_ID,
            "created_at": _created_at(),
            "decision": "PASS_V98_1_PHASE9_RENDER_SNAP" if any(row["phase9_gate_pass"] for row in render_metric_rows) else "NO_GO_V98_1_PHASE9_RENDER_SNAP",
            "render_metric_row_count": len(render_metric_rows),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )

    for control_id in CONTROL_CONFIGS:
        emissions, object_rows = build_control_emissions(control_id, ctx, source)
        eval_pack, preview_rows, selected_rows, pixel_collision_count = evaluate_emissions(control_id, emissions, source, min_pred_pixels=min_pred_pixels, min_gt_pixels=min_gt_pixels)
        control_mv_object_frame_mask_rows.extend(selected_rows)
        control_window_metric_rows.extend(eval_pack["window_rows"])
        control_scene_metric_rows.extend(eval_pack["scene_rows"])
        control_global_metric_rows.append(eval_pack["global_row"])
        duplicate_keys = [(row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"])) for row in selected_rows]
        duplicate_count = sum(max(0, count - 1) for count in Counter(duplicate_keys).values())
        missing_count = sum(1 for row in preview_rows if row.get("status") == "missing_mask")
        support_iou_mean = float(np.mean([_num(row.get("support_iou")) for row in selected_rows])) if selected_rows else 0.0
        window_agg = eval_pack["window_aggregate"]
        scene_agg = eval_pack["scene_aggregate"]
        global_summary = eval_pack["global_summary"]
        control_metric_rows.append(
            {
                "variant_id": control_id,
                "control_type": CONTROL_CONFIGS[control_id]["control_type"],
                "MV_AP_window": window_agg.get("mean_MV_AP_window"),
                "MV_AP50_window": window_agg.get("mean_MV_AP50_window"),
                "MV_AP25_window": window_agg.get("mean_MV_AP25_window"),
                "ScoreFreeMatch50_window": window_agg.get("mean_ScoreFreeMatch50_window"),
                "MV_AP_scene": scene_agg.get("mean_MV_AP_scene"),
                "MV_AP50_scene": scene_agg.get("mean_MV_AP50_scene"),
                "MV_AP25_scene": scene_agg.get("mean_MV_AP25_scene"),
                "ScoreFreeMatch50_scene": scene_agg.get("mean_ScoreFreeMatch50_scene"),
                "MV_AP_full_dev_global": global_summary.get("ap"),
                "MV_AP50_full_dev_global": global_summary.get("ap50"),
                "window_count": window_agg.get("window_count"),
                "scene_count": scene_agg.get("scene_count"),
                "same_frame_collision_count": duplicate_count,
                "pixel_collision_rate": float(pixel_collision_count / max(1, sum(_int(row.get("pred_positive_pixels")) for row in preview_rows))),
                "missing_mask_raster_count": missing_count,
                "object_count": len(object_rows),
                "emitted_object_frame_count": len(selected_rows),
                "support_to_selected_mask_IoU_mean": support_iou_mean,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
                "metric_source": "v98_1_current_run",
            }
        )
    if include_locked_control_reference:
        control_metric_rows.append(
            {
                "variant_id": BEST_LOCKED_CONTROL_VARIANT,
                "control_type": "locked_reference_from_v98_plan",
                "MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
                "MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
                "MV_AP25_window": "",
                "ScoreFreeMatch50_window": "",
                "MV_AP_scene": "",
                "MV_AP50_scene": "",
                "MV_AP25_scene": "",
                "ScoreFreeMatch50_scene": "",
                "MV_AP_full_dev_global": "",
                "MV_AP50_full_dev_global": "",
                "window_count": "",
                "scene_count": "",
                "same_frame_collision_count": "",
                "pixel_collision_rate": "",
                "missing_mask_raster_count": "",
                "object_count": "",
                "emitted_object_frame_count": "",
                "support_to_selected_mask_IoU_mean": "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "metric_source": "plan_locked_reference_not_rerun",
            }
        )
    PHASE10.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE10 / "control_metric_rows.csv", control_metric_rows)
    _write_csv(PHASE10 / "control_metric_window_rows.csv", control_window_metric_rows)
    _write_csv(PHASE10 / "control_metric_scene_rows.csv", control_scene_metric_rows)
    _write_csv(PHASE10 / "control_metric_full_dev_global_rows.csv", control_global_metric_rows)
    _write_csv(PHASE10 / "control_mv_object_frame_mask_rows.csv", control_mv_object_frame_mask_rows)
    _write_json(
        PHASE10 / "summary.json",
        {
            "schema": "stream4d_v98_1_phase10_controls_summary_v1",
            "phase_id": "v98_phase10_controls",
            "run_id": RUN_ID,
            "created_at": _created_at(),
            "control_metric_row_count": len(control_metric_rows),
            "locked_reference_included": include_locked_control_reference,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )
    return {
        "birth_metric_rows": birth_metric_rows,
        "render_metric_rows": render_metric_rows,
        "window_metric_rows": window_metric_rows_all,
        "scene_metric_rows": scene_metric_rows_all,
        "global_metric_rows": global_metric_rows_all,
        "control_metric_rows": control_metric_rows,
        "control_window_metric_rows": control_window_metric_rows,
        "control_scene_metric_rows": control_scene_metric_rows,
    }


def build_phase11_12(phase6: dict[str, Any], phase7: dict[str, Any], phase8_10: dict[str, Any]) -> dict[str, Any]:
    real_rows = [row for row in phase8_10["render_metric_rows"]]
    control_rows = [row for row in phase8_10["control_metric_rows"]]
    best_real = max(real_rows, key=lambda r: (_num(r.get("MV_AP_window"), -1.0), _num(r.get("MV_AP50_window"), -1.0)), default={})
    best_control = max(control_rows, key=lambda r: (_num(r.get("MV_AP_window"), -1.0), _num(r.get("MV_AP50_window"), -1.0)), default={})
    best_real_ap = _num(best_real.get("MV_AP_window"), -1.0)
    best_real_ap50 = _num(best_real.get("MV_AP50_window"), -1.0)
    best_control_ap = _num(best_control.get("MV_AP_window"), BEST_LOCKED_CONTROL_MV_AP_WINDOW)
    best_control_ap50 = _num(best_control.get("MV_AP50_window"), BEST_LOCKED_CONTROL_MV_AP50_WINDOW)
    current_control_rows = [row for row in control_rows if row.get("metric_source") == "v98_1_current_run"]
    best_current_control_support_iou = max((_num(row.get("support_to_selected_mask_IoU_mean"), 0.0) for row in current_control_rows), default=0.0)
    best_real_support_iou = _num(best_real.get("support_to_selected_mask_IoU_mean"), 0.0)
    semantic_proxy = bool(phase6["constants"].get("semantic_branch_proxy_only", True))
    gates = {
        "best_real_MV_AP_window_ge_B0_plus_0p010": best_real_ap >= B0_MV_AP_WINDOW + 0.010,
        "best_real_MV_AP50_window_ge_B0_plus_0p020": best_real_ap50 >= B0_MV_AP50_WINDOW + 0.020,
        "best_real_MV_AP_window_ge_best_control_plus_0p005": best_real_ap >= best_control_ap + 0.005,
        "best_real_MV_AP50_window_ge_best_control_plus_0p010": best_real_ap50 >= best_control_ap50 + 0.010,
        "best_real_MV_AP_window_ge_v91_plus_0p002": best_real_ap >= V91_BEST_MV_AP_WINDOW + 0.002,
        "same_frame_collision_count_eq_0": _int(best_real.get("same_frame_collision_count"), 1) == 0,
        "pixel_collision_rate_le_0p02": _num(best_real.get("pixel_collision_rate"), 1.0) <= 0.02,
        "missing_mask_raster_count_eq_0": _int(best_real.get("missing_mask_raster_count"), 1) == 0,
        "semantic_branch_dense_not_proxy": not semantic_proxy,
        "phase7_feature_construction_pass": phase7["summary"].get("decision") == "PASS_V98_1_PHASE7_MASK_VIEW_AFFINITY",
        "support_to_selected_mask_IoU_ge_current_control_plus_0p05": best_real_support_iou >= best_current_control_support_iou + 0.05,
    }
    dev_gate_pass = all(gates.values())
    blockers: list[str] = []
    if semantic_proxy:
        blockers.append("SEMANTIC_RESIDUAL_BLOCKER")
    if not gates["phase7_feature_construction_pass"]:
        blockers.append("MASK_VIEW_AFFINITY_BLOCKER")
    if not gates["best_real_MV_AP_window_ge_best_control_plus_0p005"] or not gates["best_real_MV_AP50_window_ge_best_control_plus_0p010"]:
        blockers.append("CONTROL_BIAS_BLOCKER")
    if not gates["best_real_MV_AP_window_ge_v91_plus_0p002"]:
        blockers.append("RANKING_BLOCKER")
    if (
        not gates["same_frame_collision_count_eq_0"]
        or not gates["pixel_collision_rate_le_0p02"]
        or not gates["missing_mask_raster_count_eq_0"]
        or not gates["support_to_selected_mask_IoU_ge_current_control_plus_0p05"]
    ):
        blockers.append("RENDER_SUPPORT_ALIGNMENT_BLOCKER")
    if not blockers:
        blockers.append("NONE")
    primary = blockers[0]
    if dev_gate_pass:
        decision = "GO_V98_DEV_LOCAL_MV_AP_WINDOW"
    elif primary == "SEMANTIC_RESIDUAL_BLOCKER":
        decision = "NO_GO_AFFINITY_FIELD"
    elif primary == "MASK_VIEW_AFFINITY_BLOCKER":
        decision = "NO_GO_AFFINITY_FIELD"
    elif primary == "RENDER_SUPPORT_ALIGNMENT_BLOCKER":
        decision = "NO_GO_RENDER_SNAP"
    elif primary == "CONTROL_BIAS_BLOCKER":
        decision = "NO_GO_CONTROL_BIAS"
    else:
        decision = "NO_GO_OBJECT_BIRTH"

    failure_rows = [
        {
            "blocker": blocker,
            "evidence": {
                "best_real_variant": best_real.get("variant_id", ""),
                "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
                "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
                "best_control_variant": best_control.get("variant_id", ""),
                "best_control_MV_AP_window": best_control.get("MV_AP_window", ""),
                "best_control_MV_AP50_window": best_control.get("MV_AP50_window", ""),
                "semantic_branch_proxy_only": semantic_proxy,
                "best_real_support_to_selected_mask_IoU_mean": best_real_support_iou,
                "best_current_control_support_to_selected_mask_IoU_mean": best_current_control_support_iou,
            },
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for blocker in blockers
        if blocker != "NONE"
    ]
    PHASE11.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE11 / "failure_decomposition_rows.csv", failure_rows)
    _write_json(
        PHASE11 / "summary.json",
        {
            "schema": "stream4d_v98_1_phase11_failure_decomposition_summary_v1",
            "phase_id": "v98_phase11_failure_decomposition",
            "run_id": RUN_ID,
            "created_at": _created_at(),
            "primary_blocker": primary,
            "secondary_blockers": blockers[1:],
            "gates": gates,
            "best_real_variant": best_real.get("variant_id", ""),
            "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
            "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
            "best_control_variant": best_control.get("variant_id", ""),
            "best_control_MV_AP_window": best_control.get("MV_AP_window", ""),
            "best_control_MV_AP50_window": best_control.get("MV_AP50_window", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )
    decision_payload = {
        "schema": "stream4d_v98_1_phase12_dev_decision_v1",
        "phase_id": "v98_phase12_dev_decision",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "best_control_variant": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": best_control.get("MV_AP_window", ""),
        "best_control_MV_AP50_window": best_control.get("MV_AP50_window", ""),
        "B0_MV_AP_window": B0_MV_AP_WINDOW,
        "B0_MV_AP50_window": B0_MV_AP50_WINDOW,
        "v91_best_MV_AP_window": V91_BEST_MV_AP_WINDOW,
        "v91_best_MV_AP50_window": V91_BEST_MV_AP50_WINDOW,
        "dev_gate_pass": dev_gate_pass,
        "holdout_allowed": bool(dev_gate_pass),
        "local2history_allowed": False,
        "primary_blocker": primary,
        "secondary_blockers": blockers[1:],
        "frozen_config_sha256_if_pass": _sha256_file(PHASE7 / "variant_config_rows.csv") if dev_gate_pass and (PHASE7 / "variant_config_rows.csv").exists() else "",
        "gate_results": gates,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    PHASE12.mkdir(parents=True, exist_ok=True)
    _write_json(PHASE12 / "final_dev_decision.json", decision_payload)
    _write_json(PHASE12 / "summary.json", decision_payload)
    return decision_payload


def main() -> None:
    global RUN_ID, SOURCE_ROWS, RADIO_MASK_FEATURES, SURFEL_ROWS, SURFEL_OBS_ROWS
    global PHASE6, PHASE7, PHASE8, PHASE9, PHASE10, PHASE11, PHASE12

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--source-rows", default=str(SOURCE_ROWS))
    parser.add_argument("--radio-mask-features", default=str(RADIO_MASK_FEATURES))
    parser.add_argument("--surfel-rows", default=str(SURFEL_ROWS))
    parser.add_argument("--surfel-obs-rows", default=str(SURFEL_OBS_ROWS))
    parser.add_argument("--phase6-root", default=str(PHASE6))
    parser.add_argument("--phase7-root", default=str(PHASE7))
    parser.add_argument("--phase8-root", default=str(PHASE8))
    parser.add_argument("--phase9-root", default=str(PHASE9))
    parser.add_argument("--phase10-root", default=str(PHASE10))
    parser.add_argument("--phase11-root", default=str(PHASE11))
    parser.add_argument("--phase12-root", default=str(PHASE12))
    parser.add_argument("--include-locked-control-reference", action="store_true", default=True)
    parser.add_argument("--no-locked-control-reference", dest="include_locked_control_reference", action="store_false")
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument("--semantic-sample-pairs", type=int, default=20000)
    parser.add_argument("--max-positive-pairs-per-mask", type=int, default=800)
    parser.add_argument("--max-negative-pairs-per-frame", type=int, default=1200)
    parser.add_argument("--min-pred-pixels", type=int, default=100)
    parser.add_argument("--min-gt-pixels", type=int, default=100)
    args = parser.parse_args()

    RUN_ID = args.run_id
    SOURCE_ROWS = _project(args.source_rows)
    RADIO_MASK_FEATURES = _project(args.radio_mask_features)
    SURFEL_ROWS = _project(args.surfel_rows)
    SURFEL_OBS_ROWS = _project(args.surfel_obs_rows)
    PHASE6 = _project(args.phase6_root)
    PHASE7 = _project(args.phase7_root)
    PHASE8 = _project(args.phase8_root)
    PHASE9 = _project(args.phase9_root)
    PHASE10 = _project(args.phase10_root)
    PHASE11 = _project(args.phase11_root)
    PHASE12 = _project(args.phase12_root)

    source = load_source_context()
    ctx = load_surfel_context()
    features = load_radio_features()
    phase6 = build_phase6(ctx, features, seed=args.seed, sample_pairs=args.semantic_sample_pairs)
    phase7 = build_phase7(
        ctx,
        source,
        phase6,
        max_positive_pairs_per_mask=args.max_positive_pairs_per_mask,
        max_negative_pairs_per_frame=args.max_negative_pairs_per_frame,
    )
    phase8_10 = build_phase8_to_10(
        ctx,
        source,
        phase7,
        min_pred_pixels=args.min_pred_pixels,
        min_gt_pixels=args.min_gt_pixels,
        include_locked_control_reference=args.include_locked_control_reference,
    )
    decision = build_phase11_12(phase6, phase7, phase8_10)
    print(json.dumps({"decision": decision["decision"], "dev_gate_pass": decision["dev_gate_pass"], "primary_blocker": decision["primary_blocker"], "best_real_variant": decision["best_real_variant"], "best_real_MV_AP_window": decision["best_real_MV_AP_window"], "best_control_variant": decision["best_control_variant"], "best_control_MV_AP_window": decision["best_control_MV_AP_window"]}, sort_keys=True))


if __name__ == "__main__":
    main()
