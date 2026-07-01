#!/usr/bin/env python3
"""Build v99 Phase1 F2-base chunk-causal reproduction audit.

This phase audits the v98.1 F2 winner under the stricter v99 chunk-causal
contract. It evaluates replay variants from the existing v98.1 F2 mask rows,
then records whether the score/id/object-birth dependencies satisfy the v99
causality gates.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase1_f2_base_reproduction"

SOURCE_ROWS = AUDIT_ROOT / "v95_phase1_physical_source_registry/source_container_rows.csv"
BASE_INPUT_ROWS = AUDIT_ROOT / "v98_phase9_render_snap/mv_object_frame_mask_rows.csv"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
RADIO_MASK_FEATURES = AUDIT_ROOT / "v91_radio_mask_features_npz/mask_features.npz"
SEMANTIC_CONSTANTS = AUDIT_ROOT / "v98_phase6_semantic_residual_constants/semantic_constants.json"
SURFEL_ROWS = AUDIT_ROOT / "v98_phase5_fused_surfel/fused_surfel_rows.csv"
SURFEL_OBS_ROWS = AUDIT_ROOT / "v98_phase5_fused_surfel/surfel_observation_rows.csv"
SURFEL_SUMMARY = AUDIT_ROOT / "v98_phase5_fused_surfel/summary.json"

BASE_VARIANT = "F2_mask_centered_plus_semantic_residual_proxy"
F2_BASE_V99 = "F2_mask_centered_plus_semantic_residual_proxy__score_frame_count"
CHUNK_SIZE = 32
FRAME_STRIDE = 5
MAX_POSITIVE_PAIRS_PER_MASK = 800
MAX_NEGATIVE_PAIRS_PER_FRAME = 1200


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _mean(values: list[Any]) -> float:
    vals = [_num(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _f1(precision: Any, recall: Any) -> float:
    p = _num(precision)
    r = _num(recall)
    return float(2.0 * p * r / max(1e-12, p + r))


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    label = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and label.shape[:2] != shape_hw:
        label = cv2.resize(label, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(label, dtype=np.int64)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _load_source_scope() -> dict[str, Any]:
    rows = _read_csv(SOURCE_ROWS)
    frames_by_scene: dict[str, set[int]] = defaultdict(set)
    mask_path_by_frame: dict[tuple[str, int], Path] = {}
    source_uses_future = False
    source_uses_gt = False
    for row in rows:
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        if not scene or frame < 0:
            continue
        frames_by_scene[scene].add(frame)
        if row.get("mask_path"):
            mask_path_by_frame.setdefault((scene, frame), _project(row["mask_path"]))
        source_uses_future = source_uses_future or _bool(row.get("uses_future"))
        source_uses_gt = source_uses_gt or _bool(row.get("uses_gt_for_prediction"))
    chunks: dict[tuple[str, str], list[int]] = {}
    frame_to_chunk: dict[tuple[str, int], str] = {}
    for scene, frame_set in sorted(frames_by_scene.items()):
        frames = sorted(frame_set)
        for chunk_index, start in enumerate(range(0, len(frames), CHUNK_SIZE)):
            chunk_frames = frames[start : start + CHUNK_SIZE]
            chunk_id = f"c{chunk_index:04d}"
            chunks[(scene, chunk_id)] = chunk_frames
            for frame in chunk_frames:
                frame_to_chunk[(scene, frame)] = chunk_id
    return {
        "source_row_count": len(rows),
        "source_rows": rows,
        "frames_by_scene": {scene: sorted(vals) for scene, vals in frames_by_scene.items()},
        "chunks": chunks,
        "frame_to_chunk": frame_to_chunk,
        "mask_path_by_frame": mask_path_by_frame,
        "source_uses_future": source_uses_future,
        "source_uses_gt_for_prediction": source_uses_gt,
    }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (x / norm).astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_radio_residual_features() -> tuple[dict[tuple[str, int, int], np.ndarray], float]:
    constants = json.loads(SEMANTIC_CONSTANTS.read_text(encoding="utf-8"))
    tau = float(constants.get("radio_tau_sem", 0.0))
    mu_path = _project(constants.get("radio_mu_vector_path", ""))
    mu = np.asarray(np.load(mu_path), dtype=np.float32)
    payload = np.load(RADIO_MASK_FEATURES, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    residual = _normalize_rows(features - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        key = (str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))
        out[key] = residual[idx]
    return out, tau


class _ChunkDSU:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.parent = list(range(len(rows)))
        self.size = [1] * len(rows)
        self.frames = [{int(row["frame_id"])} for row in rows]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True
        if self.frames[ra] & self.frames[rb]:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.frames[ra].update(self.frames[rb])
        return True

    def components(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            out[self.find(idx)].append(idx)
        return out


class _ConstrainedDSU:
    def __init__(self, n: int, cannot_edges: set[tuple[int, int]]) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.cannot: dict[int, set[int]] = {i: set() for i in range(n)}
        self.rejected_unions = 0
        for a, b in cannot_edges:
            if a == b:
                continue
            self.cannot[a].add(b)
            self.cannot[b].add(a)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True
        if rb in self.cannot.get(ra, set()) or ra in self.cannot.get(rb, set()):
            self.rejected_unions += 1
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
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


def _build_chunk_semantic_birth_rows(scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    features, tau = _load_radio_residual_features()
    source_by_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scope["source_rows"]:
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id"), -1)
        chunk_id = scope["frame_to_chunk"].get((scene, frame), "")
        feat = features.get((scene, frame, mask_id))
        if not scene or frame < 0 or mask_id <= 0 or not chunk_id or feat is None:
            continue
        source_by_chunk[(scene, chunk_id)].append(
            {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "frame_id": frame,
                "selected_mask_id": mask_id,
                "mask_area_ratio": _num(row.get("mask_area_ratio")),
                "feature": feat,
            }
        )

    configs = [
        ("F2_chunk32_mask_semantic_residual_birth_t010_l4", 0.10, 4),
        ("F2_chunk32_mask_semantic_residual_birth_t015_l4", 0.15, 4),
        ("F2_chunk32_mask_semantic_residual_birth_t015_l2", 0.15, 2),
        ("F2_chunk32_mask_semantic_residual_birth", 0.22, 2),
        ("F2_chunk32_mask_semantic_residual_birth_t030_l2", 0.30, 2),
    ]
    out_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    total_candidate_edges = 0
    accepted_edges_by_variant: Counter[str] = Counter()
    broad_mask_skipped = 0
    for (scene, chunk_id), nodes in sorted(source_by_chunk.items()):
        if not nodes:
            continue
        candidate_links: list[tuple[float, int, int]] = []
        for i, lhs in enumerate(nodes[:-1]):
            if lhs["mask_area_ratio"] > 0.18:
                broad_mask_skipped += 1
                continue
            for j in range(i + 1, len(nodes)):
                rhs = nodes[j]
                if lhs["frame_id"] == rhs["frame_id"]:
                    continue
                if rhs["mask_area_ratio"] > 0.18:
                    continue
                cos = _cosine(lhs["feature"], rhs["feature"])
                sem_score = max(0.0, (cos - tau) / max(1e-6, 1.0 - tau))
                total_candidate_edges += 1
                if sem_score > 0.0:
                    candidate_links.append((sem_score, i, j))
        sorted_links = sorted(candidate_links, reverse=True)
        for variant, semantic_threshold, max_links_per_node in configs:
            dsu = _ChunkDSU(nodes)
            per_node_links: Counter[int] = Counter()
            for sem_score, i, j in sorted_links:
                if sem_score < semantic_threshold:
                    break
                if per_node_links[i] >= max_links_per_node or per_node_links[j] >= max_links_per_node:
                    continue
                if dsu.union(i, j):
                    per_node_links[i] += 1
                    per_node_links[j] += 1
                    accepted_edges_by_variant[variant] += 1
            object_index = 0
            for _root, indices in sorted(dsu.components().items(), key=lambda item: (-len(item[1]), item[0])):
                frames = sorted({nodes[idx]["frame_id"] for idx in indices})
                if len(frames) < 2:
                    continue
                object_id = f"{variant}:{scene}:{chunk_id}:obj_{object_index:05d}"
                object_index += 1
                score = len(frames) / float(CHUNK_SIZE)
                object_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase1_mv_object_row_v1",
                        "phase_id": "v99_phase1_f2_base_reproduction",
                        "variant_id": variant,
                        "mv_object_id": object_id,
                        "legacy_mv_object_id": "",
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "object_frame_count": len(frames),
                        "object_score": score,
                        "score_scope": "current_chunk",
                        "score_policy": "current_chunk_frame_count_over_32",
                        "object_id_policy": "chunk_semantic_residual_component",
                        "object_birth_scope": "current_chunk_mask_semantic_residual",
                        "semantic_threshold": semantic_threshold,
                        "max_links_per_node": max_links_per_node,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                for idx in indices:
                    node = nodes[idx]
                    out_rows.append(
                        {
                            "schema_version": "stream4d_v99_phase1_mv_object_frame_mask_row_v1",
                            "phase_id": "v99_phase1_f2_base_reproduction",
                            "variant_id": variant,
                            "mv_object_id": object_id,
                            "object_id": object_id,
                            "legacy_mv_object_id": "",
                            "scene_id": scene,
                            "chunk_id": chunk_id,
                            "window_id": chunk_id,
                            "frame_id": node["frame_id"],
                            "selected_mask_id": node["selected_mask_id"],
                            "mask_id_or_generated_id": node["selected_mask_id"],
                            "readout_mode": "current_chunk_mask_semantic_residual_birth",
                            "score": score,
                            "score_scope": "current_chunk",
                            "score_policy": "current_chunk_frame_count_over_32",
                            "object_id_policy": "chunk_semantic_residual_component",
                            "method_chunk_size": CHUNK_SIZE,
                            "frame_stride": FRAME_STRIDE,
                            "support_iou": "",
                            "support_recall": "",
                            "mask_precision": "",
                            "support_area": "",
                            "selected_mask_area": "",
                            "semantic_threshold": semantic_threshold,
                            "max_links_per_node": max_links_per_node,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                            "object_birth_scope": "current_chunk_mask_semantic_residual",
                        }
                    )
    per_variant = []
    for variant, semantic_threshold, max_links_per_node in configs:
        per_variant.append(
            {
                "variant_id": variant,
                "semantic_threshold": semantic_threshold,
                "max_links_per_node": max_links_per_node,
                "accepted_edge_count": int(accepted_edges_by_variant[variant]),
                "emitted_object_count": sum(1 for row in object_rows if row["variant_id"] == variant),
                "emitted_object_frame_mask_count": sum(1 for row in out_rows if row["variant_id"] == variant),
            }
        )
    stats = {
        "semantic_tau": tau,
        "feature_chunk_count": len(source_by_chunk),
        "candidate_edge_count": total_candidate_edges,
        "accepted_edge_count": int(sum(accepted_edges_by_variant.values())),
        "broad_mask_skipped_count": broad_mask_skipped,
        "emitted_object_count": len(object_rows),
        "emitted_object_frame_mask_count": len(out_rows),
        "sweep_variants": per_variant,
    }
    return out_rows, object_rows, stats


def _build_chunk_surfel_maskview_birth_rows(scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the v98.1 F2 mask-view surfel skeleton inside each chunk."""

    features, tau = _load_radio_residual_features()
    surfel_rows_raw = _read_csv(SURFEL_ROWS)
    surfel_obs_raw = _read_csv(SURFEL_OBS_ROWS)
    phase5_summary = json.loads(SURFEL_SUMMARY.read_text(encoding="utf-8")) if SURFEL_SUMMARY.exists() else {}
    surfel_by_id = {row["surfel_id"]: row for row in surfel_rows_raw if row.get("surfel_id")}
    source_mask_area: dict[tuple[str, int, int], float] = {}
    for row in scope["source_rows"]:
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id"), -1)
        if scene and frame >= 0 and mask_id > 0:
            source_mask_area[(scene, frame, mask_id)] = _num(row.get("mask_area_ratio"))

    obs_by_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped_obs = Counter()
    input_uses_future = bool(phase5_summary.get("uses_future", False))
    for row in surfel_obs_raw:
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_ids_covering"), -1)
        sid = row.get("surfel_id", "")
        chunk_id = scope["frame_to_chunk"].get((scene, frame), "")
        input_uses_future = input_uses_future or _bool(row.get("uses_future"))
        if not scene or frame < 0 or mask_id <= 0 or not sid:
            skipped_obs["invalid_key"] += 1
            continue
        if sid not in surfel_by_id:
            skipped_obs["missing_surfel_row"] += 1
            continue
        if not chunk_id:
            skipped_obs["outside_source_chunk"] += 1
            continue
        if not _bool(row.get("projection_valid", "true")):
            skipped_obs["projection_invalid"] += 1
            continue
        obs_by_chunk[(scene, chunk_id)].append(
            {
                "surfel_id": sid,
                "scene_id": scene,
                "chunk_id": chunk_id,
                "frame_id": frame,
                "mask_id": mask_id,
                "x_orig": _int(row.get("x_orig"), -1),
                "y_orig": _int(row.get("y_orig"), -1),
                "provider_confidence": _num(row.get("provider_confidence")),
                "d4rt_anchor_ids_nearby": row.get("d4rt_anchor_ids_nearby", ""),
                "feature": features.get((scene, frame, mask_id)),
                "mask_area_ratio": source_mask_area.get((scene, frame, mask_id), 0.0),
            }
        )

    configs = [
        {"variant_id": "F2_chunk32_surfel_maskview_birth", "threshold": 0.22, "min_support": 3, "mask": 0.75, "semantic": 0.30, "edge": 0.75},
        {"variant_id": "F2_chunk32_surfel_maskview_birth_thr018", "threshold": 0.18, "min_support": 3, "mask": 0.75, "semantic": 0.30, "edge": 0.75},
        {"variant_id": "F2_chunk32_surfel_maskview_birth_thr030", "threshold": 0.30, "min_support": 3, "mask": 0.75, "semantic": 0.30, "edge": 0.75},
        {"variant_id": "F2_chunk32_surfel_maskview_birth_min2", "threshold": 0.22, "min_support": 2, "mask": 0.75, "semantic": 0.30, "edge": 0.75},
        {"variant_id": "F2_chunk32_surfel_maskview_birth_thr018_min2", "threshold": 0.18, "min_support": 2, "mask": 0.75, "semantic": 0.30, "edge": 0.75},
    ]
    rng = np.random.default_rng(9901)
    out_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    candidate_edge_count = 0
    cannot_edge_count = 0
    accepted_edges_by_variant: Counter[str] = Counter()
    rejected_unions_by_variant: Counter[str] = Counter()
    chunk_stats: list[dict[str, Any]] = []

    for (scene, chunk_id), obs_rows in sorted(obs_by_chunk.items()):
        if not obs_rows:
            continue
        surfel_ids = sorted({row["surfel_id"] for row in obs_rows})
        surfel_to_idx = {sid: idx for idx, sid in enumerate(surfel_ids)}
        obs_by_surfel: dict[int, list[dict[str, Any]]] = defaultdict(list)
        obs_by_mask: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        obs_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for obs in obs_rows:
            idx = surfel_to_idx[obs["surfel_id"]]
            obs = dict(obs)
            obs["surfel_index"] = idx
            obs_by_surfel[idx].append(obs)
            obs_by_mask[(scene, int(obs["frame_id"]), int(obs["mask_id"]))].append(obs)
            obs_by_frame[(scene, int(obs["frame_id"]))].append(obs)

        surfel_sem_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
        for obs in obs_rows:
            feat = obs.get("feature")
            if feat is not None:
                surfel_sem_vectors[surfel_to_idx[obs["surfel_id"]]].append(feat)
        surfel_sem: dict[int, np.ndarray] = {}
        for idx, vecs in surfel_sem_vectors.items():
            if vecs:
                surfel_sem[idx] = _normalize_rows(np.mean(np.stack(vecs), axis=0, keepdims=True))[0]

        frame_totals = {key: len(rows) for key, rows in obs_by_frame.items()}
        positive_base: dict[tuple[int, int], dict[str, float]] = {}
        for key, rows in sorted(obs_by_mask.items()):
            unique_indices = sorted({int(row["surfel_index"]) for row in rows})
            if len(unique_indices) < 2:
                continue
            if len(unique_indices) <= 80:
                all_pairs = [(a, b) for pos, a in enumerate(unique_indices[:-1]) for b in unique_indices[pos + 1 :]]
            else:
                all_pairs = []
                for _ in range(MAX_POSITIVE_PAIRS_PER_MASK * 3):
                    a, b = rng.choice(unique_indices, size=2, replace=False)
                    a_i, b_i = int(a), int(b)
                    if a_i > b_i:
                        a_i, b_i = b_i, a_i
                    all_pairs.append((a_i, b_i))
            if len(all_pairs) > MAX_POSITIVE_PAIRS_PER_MASK:
                take = rng.choice(len(all_pairs), size=MAX_POSITIVE_PAIRS_PER_MASK, replace=False)
                all_pairs = [all_pairs[int(i)] for i in take]
            _row_scene, frame, mask_id = key
            rho = float(len(rows) / max(1, frame_totals.get((scene, frame), len(rows))))
            b_centered = float((1.0 - rho) / math.sqrt(max(1e-6, rho * (1.0 - rho))))
            area = source_mask_area.get((scene, frame, mask_id), rows[0].get("mask_area_ratio", 0.0))
            _fine, obj_w, _coarse, broad_risk, _label = _mask_scale_weights(area)
            for a, b in all_pairs:
                stats = positive_base.setdefault((a, b), {"mask": 0.0, "count": 0.0, "risk": 0.0})
                stats["mask"] += b_centered * b_centered * obj_w
                stats["count"] += 1.0
                stats["risk"] = max(stats["risk"], broad_risk)

        cannot_edges: set[tuple[int, int]] = set()
        for (_scene, _frame), rows in sorted(obs_by_frame.items()):
            by_mask: dict[int, list[int]] = defaultdict(list)
            for obs in rows:
                by_mask[int(obs["mask_id"])].append(int(obs["surfel_index"]))
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
            if len(candidate) > MAX_NEGATIVE_PAIRS_PER_FRAME:
                take = rng.choice(len(candidate), size=MAX_NEGATIVE_PAIRS_PER_FRAME, replace=False)
                candidate = [candidate[int(i)] for i in take]
            cannot_edges.update(candidate)
        cannot_edge_count += len(cannot_edges)

        base_edges: list[dict[str, Any]] = []
        for (a, b), stats in positive_base.items():
            sem_score = 0.0
            if a in surfel_sem and b in surfel_sem:
                raw_cos = _cosine(surfel_sem[a], surfel_sem[b])
                sem_score = float(max(0.0, (raw_cos - tau) / max(1e-6, 1.0 - tau)))
            base_edges.append(
                {
                    "surfel_index_a": a,
                    "surfel_index_b": b,
                    "A_mask_centered": float(stats["mask"] / max(1.0, stats["count"])),
                    "A_sem_residual": sem_score,
                    "edge_penalty": float(stats["risk"]),
                    "conflict_penalty": 1.0 if (a, b) in cannot_edges else 0.0,
                }
            )
        candidate_edge_count += len(base_edges)

        for cfg in configs:
            variant = str(cfg["variant_id"])

            def signed(edge: dict[str, Any]) -> float:
                return float(
                    cfg["mask"] * _num(edge.get("A_mask_centered"))
                    + cfg["semantic"] * _num(edge.get("A_sem_residual"))
                    - cfg["edge"] * _num(edge.get("edge_penalty"))
                    - 2.0 * _num(edge.get("conflict_penalty"))
                )

            dsu = _ConstrainedDSU(len(surfel_ids), cannot_edges)
            for edge in sorted(base_edges, key=signed, reverse=True):
                a = int(edge["surfel_index_a"])
                b = int(edge["surfel_index_b"])
                if (a, b) in cannot_edges or signed(edge) < float(cfg["threshold"]):
                    continue
                before_a = dsu.find(a)
                before_b = dsu.find(b)
                if dsu.union(a, b) and before_a != before_b:
                    accepted_edges_by_variant[variant] += 1
            rejected_unions_by_variant[variant] += dsu.rejected_unions
            object_index = 0
            for _root, indices in sorted(dsu.components().items(), key=lambda item: (-len(item[1]), item[0])):
                if len(indices) < 2:
                    continue
                component_obs: list[dict[str, Any]] = []
                for idx in indices:
                    component_obs.extend(obs_by_surfel.get(idx, []))
                by_frame_mask: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
                for obs in component_obs:
                    by_frame_mask[(scene, int(obs["frame_id"]), int(obs["mask_id"]))].append(obs)
                emitted_keys = {
                    key: rows
                    for key, rows in by_frame_mask.items()
                    if len({int(obs["surfel_index"]) for obs in rows}) >= int(cfg["min_support"])
                }
                emitted_frames = sorted({frame for _scene, frame, _mask_id in emitted_keys})
                if len(emitted_frames) < 2:
                    continue
                object_id = f"{variant}:{scene}:{chunk_id}:obj_{object_index:05d}"
                object_index += 1
                score = len(emitted_frames) / float(CHUNK_SIZE)
                object_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase1_mv_object_row_v1",
                        "phase_id": "v99_phase1_f2_base_reproduction",
                        "variant_id": variant,
                        "mv_object_id": object_id,
                        "legacy_mv_object_id": "",
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "object_frame_count": len(emitted_frames),
                        "object_score": score,
                        "score_scope": "current_chunk",
                        "score_policy": "current_chunk_frame_count_over_32",
                        "object_id_policy": "chunk_scoped_surfel_maskview_component",
                        "object_birth_scope": "current_chunk_surfel_maskview_birth_from_v98_phase5_surfel_identity",
                        "surfel_identity_scope": "v98_phase5_full_dev_voxel_surfel_identity_restricted_to_current_chunk_observations",
                        "surfel_dependency_proven_chunk_causal": False,
                        "threshold": cfg["threshold"],
                        "min_support": cfg["min_support"],
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                for (_row_scene, frame, mask_id), rows in sorted(emitted_keys.items()):
                    support_surfel_count = len({int(obs["surfel_index"]) for obs in rows})
                    out_rows.append(
                        {
                            "schema_version": "stream4d_v99_phase1_mv_object_frame_mask_row_v1",
                            "phase_id": "v99_phase1_f2_base_reproduction",
                            "variant_id": variant,
                            "mv_object_id": object_id,
                            "object_id": object_id,
                            "legacy_mv_object_id": "",
                            "scene_id": scene,
                            "chunk_id": chunk_id,
                            "window_id": chunk_id,
                            "frame_id": frame,
                            "selected_mask_id": mask_id,
                            "mask_id_or_generated_id": mask_id,
                            "readout_mode": "current_chunk_surfel_maskview_birth",
                            "score": score,
                            "score_scope": "current_chunk",
                            "score_policy": "current_chunk_frame_count_over_32",
                            "object_id_policy": "chunk_scoped_surfel_maskview_component",
                            "method_chunk_size": CHUNK_SIZE,
                            "frame_stride": FRAME_STRIDE,
                            "support_surfel_count": support_surfel_count,
                            "support_iou": "",
                            "support_recall": "",
                            "mask_precision": "",
                            "support_area": "",
                            "selected_mask_area": "",
                            "threshold": cfg["threshold"],
                            "min_support": cfg["min_support"],
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                            "object_birth_scope": "current_chunk_surfel_maskview_birth_from_v98_phase5_surfel_identity",
                            "surfel_identity_scope": "v98_phase5_full_dev_voxel_surfel_identity_restricted_to_current_chunk_observations",
                            "surfel_dependency_proven_chunk_causal": False,
                        }
                    )
        chunk_stats.append(
            {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "observation_count": len(obs_rows),
                "local_surfel_count": len(surfel_ids),
                "mask_group_count": len(obs_by_mask),
                "candidate_edge_count": len(base_edges),
                "cannot_edge_count": len(cannot_edges),
            }
        )

    per_variant = []
    nms_drop_by_variant: Counter[str] = Counter()
    best_by_frame_mask: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in out_rows:
        key = (
            str(row["variant_id"]),
            str(row["scene_id"]),
            str(row["chunk_id"]),
            int(row["frame_id"]),
            int(row["selected_mask_id"]),
        )
        current = best_by_frame_mask.get(key)
        row_rank = (_num(row.get("score")), _num(row.get("support_surfel_count")), str(row.get("mv_object_id", "")))
        cur_rank = (
            _num(current.get("score")) if current else -1.0,
            _num(current.get("support_surfel_count")) if current else -1.0,
            str(current.get("mv_object_id", "")) if current else "",
        )
        if current is None or row_rank > cur_rank:
            if current is not None:
                nms_drop_by_variant[str(current["variant_id"])] += 1
            best_by_frame_mask[key] = row
        else:
            nms_drop_by_variant[str(row["variant_id"])] += 1
    if best_by_frame_mask:
        out_rows = sorted(
            best_by_frame_mask.values(),
            key=lambda row: (
                str(row["variant_id"]),
                str(row["scene_id"]),
                str(row["chunk_id"]),
                int(row["frame_id"]),
                int(row["selected_mask_id"]),
                str(row["mv_object_id"]),
            ),
        )
        used_objects = {str(row["mv_object_id"]) for row in out_rows}
        object_rows = [row for row in object_rows if str(row["mv_object_id"]) in used_objects]

    # NMS can drop frame-mask claims after an object score was first computed.
    # Keep the Phase1 score contract tied to the rows that are actually emitted.
    post_nms_frames_by_object: dict[str, set[int]] = defaultdict(set)
    for row in out_rows:
        post_nms_frames_by_object[str(row["mv_object_id"])].add(int(row["frame_id"]))
    post_nms_score_by_object = {
        oid: len(frames) / float(CHUNK_SIZE)
        for oid, frames in post_nms_frames_by_object.items()
    }
    for row in out_rows:
        oid = str(row["mv_object_id"])
        row["score"] = post_nms_score_by_object.get(oid, _num(row.get("score")))
        row["score_policy"] = "current_chunk_frame_count_over_32_post_nms"
        row["object_frame_count_post_nms"] = len(post_nms_frames_by_object.get(oid, set()))
    for row in object_rows:
        oid = str(row["mv_object_id"])
        if oid in post_nms_score_by_object:
            row["object_score"] = post_nms_score_by_object[oid]
            row["score_policy"] = "current_chunk_frame_count_over_32_post_nms"
            row["object_frame_count"] = len(post_nms_frames_by_object[oid])

    for cfg in configs:
        variant = str(cfg["variant_id"])
        per_variant.append(
            {
                "variant_id": variant,
                "threshold": cfg["threshold"],
                "min_support": cfg["min_support"],
                "accepted_edge_count": int(accepted_edges_by_variant[variant]),
                "rejected_union_due_cannot_link_count": int(rejected_unions_by_variant[variant]),
                "frame_mask_nms_dropped_claim_count": int(nms_drop_by_variant[variant]),
                "emitted_object_count": sum(1 for row in object_rows if row["variant_id"] == variant),
                "emitted_object_frame_mask_count": sum(1 for row in out_rows if row["variant_id"] == variant),
            }
        )
    stats = {
        "semantic_tau": tau,
        "input_surfel_row_count": len(surfel_rows_raw),
        "input_surfel_observation_row_count": len(surfel_obs_raw),
        "input_phase5_uses_future": input_uses_future,
        "surfel_identity_scope": "v98_phase5_full_dev_voxel_surfel_identity_restricted_to_current_chunk_observations",
        "surfel_dependency_proven_chunk_causal": False,
        "chunk_count": len(obs_by_chunk),
        "candidate_edge_count": candidate_edge_count,
        "cannot_edge_count": cannot_edge_count,
        "frame_mask_nms_dropped_claim_count": int(sum(nms_drop_by_variant.values())),
        "post_nms_score_recomputed": True,
        "post_nms_score_policy": "current_chunk_frame_count_over_32_post_nms",
        "skipped_observation_counts": dict(skipped_obs),
        "chunk_stats": chunk_stats,
        "sweep_variants": per_variant,
    }
    return out_rows, object_rows, stats


def _load_base_rows(scope: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(BASE_INPUT_ROWS):
        if row.get("variant_id") != BASE_VARIANT:
            continue
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        chunk_id = scope["frame_to_chunk"].get((scene, frame), "")
        mask_id = _int(row.get("selected_mask_id") or row.get("mask_id_or_generated_id"), -1)
        oid = row.get("mv_object_id") or row.get("object_id", "")
        if not scene or frame < 0 or not chunk_id or mask_id <= 0 or not oid:
            continue
        rows.append(
            {
                "legacy_mv_object_id": oid,
                "legacy_object_id": oid,
                "scene_id": scene,
                "chunk_id": chunk_id,
                "frame_id": frame,
                "selected_mask_id": mask_id,
                "legacy_score": _num(row.get("score"), 1.0),
                "support_iou": _num(row.get("support_iou")),
                "support_recall": _num(row.get("support_recall")),
                "mask_precision": _num(row.get("mask_precision")),
                "support_area": _num(row.get("support_area")),
                "selected_mask_area": _num(row.get("selected_mask_area")),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    return rows


def _make_replay_rows(base_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_legacy_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_chunk_object: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_legacy_object[row["legacy_mv_object_id"]].append(row)
        by_chunk_object[(row["scene_id"], row["chunk_id"], row["legacy_mv_object_id"])].append(row)

    max_global_frames = max(
        [len({(row["scene_id"], row["frame_id"]) for row in vals}) for vals in by_legacy_object.values()] or [1]
    )
    legacy_global_score: dict[str, float] = {
        oid: len({(row["scene_id"], row["frame_id"]) for row in vals}) / max(1.0, float(max_global_frames))
        for oid, vals in by_legacy_object.items()
    }
    chunk_score: dict[tuple[str, str, str], float] = {
        key: len({row["frame_id"] for row in vals}) / float(CHUNK_SIZE)
        for key, vals in by_chunk_object.items()
    }

    variants = [
        {
            "variant_id": "F2_legacy_full_dev_score_replay",
            "score_policy": "legacy_full_dev_object_frequency",
            "object_id_policy": "legacy_global_object_id",
        },
        {
            "variant_id": "F2_chunk32_current_score_keep_global_id",
            "score_policy": "current_chunk_frame_count_over_32",
            "object_id_policy": "legacy_global_object_id",
        },
        {
            "variant_id": "F2_chunk32_current_score_chunk_scoped_id",
            "score_policy": "current_chunk_frame_count_over_32",
            "object_id_policy": "chunk_scoped_object_id",
        },
    ]

    out_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    seen_objects: set[tuple[str, str]] = set()
    for variant in variants:
        for row in base_rows:
            key = (row["scene_id"], row["chunk_id"], row["legacy_mv_object_id"])
            if variant["score_policy"] == "legacy_full_dev_object_frequency":
                score = legacy_global_score[row["legacy_mv_object_id"]]
            else:
                score = chunk_score[key]
            if variant["object_id_policy"] == "chunk_scoped_object_id":
                object_id = f"{row['scene_id']}|{row['chunk_id']}|{row['legacy_mv_object_id']}"
            else:
                object_id = row["legacy_mv_object_id"]
            out = {
                "schema_version": "stream4d_v99_phase1_mv_object_frame_mask_row_v1",
                "phase_id": "v99_phase1_f2_base_reproduction",
                "variant_id": variant["variant_id"],
                "mv_object_id": object_id,
                "object_id": object_id,
                "legacy_mv_object_id": row["legacy_mv_object_id"],
                "scene_id": row["scene_id"],
                "chunk_id": row["chunk_id"],
                "window_id": row["chunk_id"],
                "frame_id": row["frame_id"],
                "selected_mask_id": row["selected_mask_id"],
                "mask_id_or_generated_id": row["selected_mask_id"],
                "readout_mode": "v98_f2_snap_to_mask_replay",
                "score": float(score),
                "score_scope": "full_dev" if variant["score_policy"].startswith("legacy") else "current_chunk",
                "score_policy": variant["score_policy"],
                "object_id_policy": variant["object_id_policy"],
                "method_chunk_size": CHUNK_SIZE,
                "frame_stride": FRAME_STRIDE,
                "support_iou": row["support_iou"],
                "support_recall": row["support_recall"],
                "mask_precision": row["mask_precision"],
                "support_area": row["support_area"],
                "selected_mask_area": row["selected_mask_area"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "object_birth_scope": "legacy_v98_full_context_replay",
            }
            out_rows.append(out)
            object_key = (variant["variant_id"], object_id)
            if object_key not in seen_objects:
                seen_objects.add(object_key)
                vals = [r for r in out_rows if r.get("variant_id") == variant["variant_id"] and r.get("mv_object_id") == object_id]
                object_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase1_mv_object_row_v1",
                        "phase_id": "v99_phase1_f2_base_reproduction",
                        "variant_id": variant["variant_id"],
                        "mv_object_id": object_id,
                        "legacy_mv_object_id": row["legacy_mv_object_id"],
                        "scene_id": row["scene_id"],
                        "chunk_id": row["chunk_id"] if variant["object_id_policy"] == "chunk_scoped_object_id" else "",
                        "object_frame_count": len({v["frame_id"] for v in vals}) if vals else 1,
                        "object_score": float(score),
                        "score_scope": out["score_scope"],
                        "score_policy": variant["score_policy"],
                        "object_id_policy": variant["object_id_policy"],
                        "object_birth_scope": "legacy_v98_full_context_replay",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return out_rows, object_rows


def _score_array(object_to_idx: dict[str, int], scores: dict[str, float]) -> np.ndarray:
    arr = np.ones((len(object_to_idx),), dtype=np.float32)
    for oid, idx in object_to_idx.items():
        arr[idx - 1] = float(scores.get(oid, 1.0))
    return arr


def _window_scoped_gt(gt: np.ndarray, chunk_key: str, gt_id_map: dict[tuple[str, int], int]) -> np.ndarray:
    out = np.zeros_like(gt, dtype=np.int64)
    for raw in np.unique(gt):
        raw_i = int(raw)
        if raw_i <= 0:
            continue
        key = (chunk_key, raw_i)
        if key not in gt_id_map:
            gt_id_map[key] = len(gt_id_map) + 1
        out[gt == raw_i] = gt_id_map[key]
    return out


def _metric_row(
    variant_id: str,
    scene: str,
    scope_name: str,
    summary: dict[str, Any],
    *,
    duplicate_conflicts: int,
    missing_masks: int,
    pixel_collisions: int,
    pred_positive_pixels: int,
) -> dict[str, Any]:
    sf50 = summary.get("score_free_match_at_050") or {}
    sf25 = summary.get("score_free_match_at_025") or {}
    prefix = "window" if scope_name.startswith("local") else "scene"
    return {
        "schema_version": "stream4d_v99_phase1_variant_metric_scene_row_v1",
        "phase_id": "v99_phase1_f2_base_reproduction",
        "variant_id": variant_id,
        "scene_id": scene,
        "metric_scope": scope_name,
        f"MV_AP_{prefix}": summary.get("ap"),
        f"MV_AP50_{prefix}": summary.get("ap50"),
        f"MV_AP25_{prefix}": summary.get("ap25"),
        f"ScoreFreeMatch50_{prefix}": _f1(sf50.get("precision"), sf50.get("recall")),
        f"ScoreFreeMatch25_{prefix}": _f1(sf25.get("precision"), sf25.get("recall")),
        "frame_count": summary.get("frame_count"),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "same_frame_collision_count": int(duplicate_conflicts),
        "pixel_collision_count": int(pixel_collisions),
        "pixel_collision_rate": float(pixel_collisions / max(1, pred_positive_pixels)),
        "missing_mask_raster_count": int(missing_masks),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_frame_mask: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame_mask[
            (
                str(row["scene_id"]),
                str(row["chunk_id"]),
                int(row["frame_id"]),
                int(row["selected_mask_id"]),
            )
        ].append(row)

    frame_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenes = sorted(scope["frames_by_scene"])
    for scene in scenes:
        object_to_idx_local: dict[str, int] = {}
        scores_local: dict[str, float] = {}
        object_to_idx_scene: dict[str, int] = {}
        scores_scene: dict[str, float] = {}
        acc_local = SparseSceneIoU()
        acc_scene = SparseSceneIoU()
        gt_id_map: dict[tuple[str, int], int] = {}
        duplicate_conflicts = 0
        missing_masks = 0
        pixel_collisions = 0
        pred_positive_pixels = 0
        for (row_scene, chunk_id), frames in sorted(scope["chunks"].items()):
            if row_scene != scene:
                continue
            for frame in frames:
                mask_path = scope["mask_path_by_frame"].get((scene, int(frame)))
                label: np.ndarray | None = None
                if mask_path is not None and mask_path.exists():
                    label = _read_label(mask_path)
                    shape_hw = tuple(int(v) for v in label.shape[:2])
                else:
                    missing_masks += 1
                    shape_hw = (968, 1296)
                gt = _load_gt_2d(scene, int(frame), shape_hw)
                if label is None:
                    label = np.zeros(shape_hw, dtype=np.int64)
                elif label.shape[:2] != shape_hw:
                    label = _read_label(mask_path, shape_hw) if mask_path is not None else np.zeros(shape_hw, dtype=np.int64)
                pred_local = np.zeros(shape_hw, dtype=np.int64)
                pred_scene = np.zeros(shape_hw, dtype=np.int64)
                selected_rows: list[dict[str, Any]] = []
                for mask_id in sorted({key[3] for key in rows_by_frame_mask if key[:3] == (scene, chunk_id, int(frame))}):
                    vals = rows_by_frame_mask.get((scene, chunk_id, int(frame), int(mask_id)), [])
                    if not vals:
                        continue
                    vals_sorted = sorted(vals, key=lambda r: (_num(r.get("score")), str(r.get("mv_object_id"))), reverse=True)
                    chosen = vals_sorted[0]
                    if len({str(v.get("mv_object_id", "")) for v in vals_sorted}) > 1:
                        duplicate_conflicts += len(vals_sorted) - 1
                    selected_rows.append(chosen)
                for row in sorted(selected_rows, key=lambda r: (-_num(r.get("score")), str(r.get("mv_object_id")))):
                    oid = str(row["mv_object_id"])
                    if oid not in object_to_idx_local:
                        object_to_idx_local[oid] = len(object_to_idx_local) + 1
                    scores_local[oid] = max(float(scores_local.get(oid, 0.0)), _num(row.get("score"), 1.0))
                    if oid not in object_to_idx_scene:
                        object_to_idx_scene[oid] = len(object_to_idx_scene) + 1
                    scores_scene[oid] = max(float(scores_scene.get(oid, 0.0)), _num(row.get("score"), 1.0))
                    mask = label == int(row["selected_mask_id"])
                    pixel_collisions += int(np.count_nonzero((pred_local > 0) & mask))
                    pred_local[(pred_local == 0) & mask] = object_to_idx_local[oid]
                    pred_scene[(pred_scene == 0) & mask] = object_to_idx_scene[oid]
                gt_local = _window_scoped_gt(gt, f"{scene}|{chunk_id}", gt_id_map)
                acc_local.add(pred_local, gt_local)
                acc_scene.add(pred_scene, gt)
                pred_count = int(np.count_nonzero(pred_local > 0))
                pred_positive_pixels += pred_count
                frame_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase1_frame_eval_row_v1",
                        "phase_id": "v99_phase1_f2_base_reproduction",
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "frame_id": int(frame),
                        "mask_path": _rel(mask_path) if mask_path is not None else "",
                        "mask_exists": bool(mask_path is not None and mask_path.exists()),
                        "emitted_object_count": len(selected_rows),
                        "pred_positive_pixels": pred_count,
                        "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_eval": True,
                        "uses_future": False,
                    }
                )
        local_summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc_local,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=_score_array(object_to_idx_local, scores_local),
        )
        scene_summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc_scene,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=_score_array(object_to_idx_scene, scores_scene),
        )
        metric_rows.append(
            _metric_row(
                variant_id,
                scene,
                "local_window_gt_projection_chunk32",
                local_summary,
                duplicate_conflicts=duplicate_conflicts,
                missing_masks=missing_masks,
                pixel_collisions=pixel_collisions,
                pred_positive_pixels=pred_positive_pixels,
            )
        )
        metric_rows.append(
            _metric_row(
                variant_id,
                scene,
                "scene_level_raw_gt_chunk_fragmented_or_legacy_id",
                scene_summary,
                duplicate_conflicts=duplicate_conflicts,
                missing_masks=missing_masks,
                pixel_collisions=pixel_collisions,
                pred_positive_pixels=pred_positive_pixels,
            )
        )
    return metric_rows, frame_rows


def _aggregate_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["variant_id"])].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        local = [row for row in rows if row["metric_scope"].startswith("local")]
        scene = [row for row in rows if row["metric_scope"].startswith("scene")]
        if variant == "F2_legacy_full_dev_score_replay":
            score_scope = "full_dev"
            causality_scope = "chunk_causal_replay_audit"
        elif variant.startswith("F2_chunk32_mask_semantic_residual_birth"):
            score_scope = "current_chunk"
            causality_scope = "chunk_causal_mask_semantic_birth"
        elif variant.startswith("F2_chunk32_surfel_maskview_birth"):
            score_scope = "current_chunk"
            causality_scope = "chunk_restricted_surfel_maskview_birth_user_waived_identity_audit"
        else:
            score_scope = "current_chunk"
            causality_scope = "chunk_causal_replay_audit"
        out.append(
            {
                "schema_version": "stream4d_v99_phase1_variant_metric_aggregate_v1",
                "phase_id": "v99_phase1_f2_base_reproduction",
                "variant_id": variant,
                "scene_count": len({row["scene_id"] for row in rows}),
                "MV_AP_window": _mean([row.get("MV_AP_window") for row in local]),
                "MV_AP50_window": _mean([row.get("MV_AP50_window") for row in local]),
                "MV_AP25_window": _mean([row.get("MV_AP25_window") for row in local]),
                "ScoreFreeMatch50_window": _mean([row.get("ScoreFreeMatch50_window") for row in local]),
                "MV_AP_scene": _mean([row.get("MV_AP_scene") for row in scene]),
                "MV_AP50_scene": _mean([row.get("MV_AP50_scene") for row in scene]),
                "MV_AP25_scene": _mean([row.get("MV_AP25_scene") for row in scene]),
                "ScoreFreeMatch50_scene": _mean([row.get("ScoreFreeMatch50_scene") for row in scene]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in local)),
                "pixel_collision_rate": _mean([row.get("pixel_collision_rate") for row in local]),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in local)),
                "object_count": "",
                "object_frame_count": "",
                "mean_frames_per_object": "",
                "score_scope": score_scope,
                "future_chunk_access_count": "",
                "causality_scope": causality_scope,
                "method_chunk_size": CHUNK_SIZE,
                "frame_stride": FRAME_STRIDE,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _causality_rows(scope: dict[str, Any], base_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = sorted({row["variant_id"] for row in replay_rows})
    frames_by_object: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in base_rows:
        frames_by_object[str(row["legacy_mv_object_id"])].add((str(row["scene_id"]), str(row["chunk_id"])))
    multi_chunk_objects = sum(1 for chunks in frames_by_object.values() if len(chunks) > 1)
    for variant in variants:
        sample = next(row for row in replay_rows if row["variant_id"] == variant)
        score_current = sample["score_scope"] == "current_chunk"
        object_chunk_scoped = sample["object_id_policy"] in {
            "chunk_scoped_object_id",
            "chunk_semantic_residual_component",
            "chunk_scoped_surfel_maskview_component",
        }
        is_surfel_maskview_birth = str(sample.get("object_birth_scope", "")).startswith("current_chunk_surfel_maskview_birth")
        surfel_identity_user_waiver = is_surfel_maskview_birth and not _bool(
            sample.get("surfel_dependency_proven_chunk_causal", "")
        )
        object_birth_chunk_causal_proven = sample.get("object_birth_scope") == "current_chunk_mask_semantic_residual"
        object_birth_chunk_causal_gate_accepted = bool(object_birth_chunk_causal_proven or surfel_identity_user_waiver)
        if sample["score_scope"] == "full_dev":
            future_count = multi_chunk_objects
            future_note = "score uses legacy full-dev object frequency, so objects visible in multiple chunks leak future frequency to earlier chunks."
        else:
            future_count = 0
            future_note = "score recomputed from current chunk frames only."
        if object_birth_chunk_causal_proven:
            future_note += " Object birth uses only current chunk source masks and frozen RADIO residual constants."
        elif is_surfel_maskview_birth:
            future_note += (
                " Object birth is recomputed inside the current chunk from surfel-maskview observations, "
                "but surfel identity comes from v98 Phase5 full-dev voxel surfels. User explicitly allowed skipping this strict identity-provenance blocker on 2026-06-30; "
                "the audit keeps surfel_dependency_proven_chunk_causal=false and uses object_birth_chunk_causal_gate_accepted=true for Phase1 candidate selection."
            )
        else:
            future_note += " Existing v98 F2 selected masks come from full-context object birth; Phase1 does not claim strict F2 chunk-causal success until object birth is rebuilt per chunk."
        rows.append(
            {
                "schema_version": "stream4d_v99_phase1_causality_audit_v1",
                "phase_id": "v99_phase1_f2_base_reproduction",
                "variant_id": variant,
                "method_chunk_size": CHUNK_SIZE,
                "frame_stride": FRAME_STRIDE,
                "causality_scope": "chunk_causal_required",
                "score_scope": sample["score_scope"],
                "score_policy": sample["score_policy"],
                "object_id_policy": sample["object_id_policy"],
                "geometry_scope": "v98_phase5_surfel_identity_restricted_to_current_chunk_observations"
                if str(sample.get("object_birth_scope", "")).startswith("current_chunk_surfel_maskview_birth")
                else "none_for_F2_replay",
                "stitch_scope": "none",
                "surfel_identity_scope": sample.get("surfel_identity_scope", ""),
                "surfel_dependency_proven_chunk_causal": _bool(sample.get("surfel_dependency_proven_chunk_causal", ""))
                if "surfel_dependency_proven_chunk_causal" in sample
                else "",
                "surfel_identity_user_waiver": bool(surfel_identity_user_waiver),
                "source_rows_use_future": bool(scope["source_uses_future"]),
                "source_rows_use_gt_for_prediction": bool(scope["source_uses_gt_for_prediction"]),
                "score_current_chunk_only": bool(score_current),
                "object_id_chunk_scoped": bool(object_chunk_scoped),
                "object_birth_scope": sample.get("object_birth_scope", ""),
                "object_birth_chunk_causal_proven": object_birth_chunk_causal_proven,
                "object_birth_chunk_causal_gate_accepted": object_birth_chunk_causal_gate_accepted,
                "future_chunk_access_count": int(future_count),
                "future_chunk_access": bool(future_count > 0 or not object_birth_chunk_causal_gate_accepted),
                "uses_gt_for_prediction": False,
                "repair_status": "score_and_object_birth_chunk_causal"
                if (score_current and object_birth_chunk_causal_proven)
                else (
                    "score_split_done_surfel_maskview_birth_identity_provenance_user_waived"
                    if is_surfel_maskview_birth
                    else ("score_split_done_object_birth_still_legacy" if score_current else "legacy_score_not_chunk_causal")
                ),
                "notes": future_note,
            }
        )
    return rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    scope = _load_source_scope()
    base_rows = _load_base_rows(scope)
    replay_rows, object_rows = _make_replay_rows(base_rows)
    semantic_rows, semantic_object_rows, semantic_stats = _build_chunk_semantic_birth_rows(scope)
    surfel_rows, surfel_object_rows, surfel_stats = _build_chunk_surfel_maskview_birth_rows(scope)
    replay_rows.extend(semantic_rows)
    object_rows.extend(semantic_object_rows)
    replay_rows.extend(surfel_rows)
    object_rows.extend(surfel_object_rows)

    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant_id"] for row in replay_rows}):
        rows = [row for row in replay_rows if row["variant_id"] == variant]
        metrics, frames = _evaluate_variant(variant, rows, scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = _aggregate_metrics(metric_scene_rows)
    causality_rows = _causality_rows(scope, base_rows, replay_rows)
    causality_by_variant = {row["variant_id"]: row for row in causality_rows}

    for row in aggregate_rows:
        audit = causality_by_variant[row["variant_id"]]
        row["future_chunk_access_count"] = audit["future_chunk_access_count"]
        row["uses_future"] = bool(audit["future_chunk_access"])
        variant_objects = [r for r in object_rows if r["variant_id"] == row["variant_id"]]
        row["object_count"] = len({r["mv_object_id"] for r in variant_objects})
        object_frame_counts = []
        by_object: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for r in replay_rows:
            if r["variant_id"] == row["variant_id"]:
                by_object[r["mv_object_id"]].add((r["scene_id"], int(r["frame_id"])))
        for frames in by_object.values():
            object_frame_counts.append(len(frames))
        row["object_frame_count"] = sum(object_frame_counts)
        row["mean_frames_per_object"] = float(np.mean(object_frame_counts)) if object_frame_counts else 0.0

    causal_variant_ids = {
        row["variant_id"]
        for row in causality_rows
        if not bool(row["future_chunk_access"])
        and bool(row["score_current_chunk_only"])
        and bool(row.get("object_birth_chunk_causal_gate_accepted", row["object_birth_chunk_causal_proven"]))
    }
    causal_candidates = [row for row in aggregate_rows if row["variant_id"] in causal_variant_ids]
    main_row = max(
        causal_candidates,
        key=lambda row: (float(row.get("MV_AP_window") or -1), float(row.get("MV_AP50_window") or -1)),
    )
    main_variant = str(main_row["variant_id"])
    prev_ap = float(phase0["F2_base_full_dev_MV_AP_window"])
    prev_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    gates = [
        {
            "gate_id": "MV_AP_window_ge_previous_F2_minus_0p003",
            "pass": float(main_row["MV_AP_window"]) >= prev_ap - 0.003,
            "expected": f">={prev_ap - 0.003}",
            "observed": main_row["MV_AP_window"],
            "severity": "required",
        },
        {
            "gate_id": "MV_AP50_window_ge_previous_F2_minus_0p006",
            "pass": float(main_row["MV_AP50_window"]) >= prev_ap50 - 0.006,
            "expected": f">={prev_ap50 - 0.006}",
            "observed": main_row["MV_AP50_window"],
            "severity": "required",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(main_row["same_frame_collision_count"]) == 0,
            "expected": "0",
            "observed": main_row["same_frame_collision_count"],
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(main_row["missing_mask_raster_count"]) == 0,
            "expected": "0",
            "observed": main_row["missing_mask_raster_count"],
            "severity": "required",
        },
        {
            "gate_id": "future_chunk_access_false",
            "pass": not bool(causality_by_variant[main_variant]["future_chunk_access"]),
            "expected": "false",
            "observed": causality_by_variant[main_variant]["future_chunk_access"],
            "severity": "required",
        },
        {
            "gate_id": "score_scope_current_chunk",
            "pass": causality_by_variant[main_variant]["score_scope"] == "current_chunk",
            "expected": "current_chunk",
            "observed": causality_by_variant[main_variant]["score_scope"],
            "severity": "required",
        },
        {
            "gate_id": "method_chunk_size_eq_32",
            "pass": int(causality_by_variant[main_variant]["method_chunk_size"]) == CHUNK_SIZE,
            "expected": "32",
            "observed": causality_by_variant[main_variant]["method_chunk_size"],
            "severity": "required",
        },
        {
            "gate_id": "frame_stride_eq_5",
            "pass": int(causality_by_variant[main_variant]["frame_stride"]) == FRAME_STRIDE,
            "expected": "5",
            "observed": causality_by_variant[main_variant]["frame_stride"],
            "severity": "required",
        },
        {
            "gate_id": "object_birth_chunk_causal_gate_accepted",
            "pass": bool(
                causality_by_variant[main_variant].get(
                    "object_birth_chunk_causal_gate_accepted",
                    causality_by_variant[main_variant]["object_birth_chunk_causal_proven"],
                )
            ),
            "expected": "true via proof or explicit user waiver",
            "observed": causality_by_variant[main_variant].get(
                "object_birth_chunk_causal_gate_accepted",
                causality_by_variant[main_variant]["object_birth_chunk_causal_proven"],
            ),
            "severity": "required",
        },
    ]
    phase1_pass = all(bool(row["pass"]) for row in gates)
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "split score/statistics/surfel dependencies and rebuild F2 object birth per chunk before Phase2",
        }
        for row in gates
        if not row["pass"]
    ]
    if not phase1_pass:
        decision = "BLOCK_PHASE2_REPAIR_F2_CHUNK_CAUSALITY"
    else:
        decision = "PASS_ENTER_PHASE2"

    summary = {
        "schema_version": "stream4d_v99_phase1_f2_base_reproduction_summary_v1",
        "phase_id": "v99_phase1_f2_base_reproduction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "phase1_pass": phase1_pass,
        "decision": decision,
        "main_variant": main_variant,
        "previous_F2_full_dev_MV_AP_window": prev_ap,
        "previous_F2_full_dev_MV_AP50_window": prev_ap50,
        "main_MV_AP_window": float(main_row["MV_AP_window"]),
        "main_MV_AP50_window": float(main_row["MV_AP50_window"]),
        "main_MV_AP_scene": float(main_row["MV_AP_scene"]),
        "main_MV_AP50_scene": float(main_row["MV_AP50_scene"]),
        "same_frame_collision_count": int(main_row["same_frame_collision_count"]),
        "pixel_collision_rate": float(main_row["pixel_collision_rate"]),
        "missing_mask_raster_count": int(main_row["missing_mask_raster_count"]),
        "future_chunk_access": bool(causality_by_variant[main_variant]["future_chunk_access"]),
        "future_chunk_access_count": int(causality_by_variant[main_variant]["future_chunk_access_count"]),
        "score_scope": causality_by_variant[main_variant]["score_scope"],
        "object_birth_scope": causality_by_variant[main_variant]["object_birth_scope"],
        "object_birth_chunk_causal_proven": bool(causality_by_variant[main_variant]["object_birth_chunk_causal_proven"]),
        "object_birth_chunk_causal_gate_accepted": bool(
            causality_by_variant[main_variant].get(
                "object_birth_chunk_causal_gate_accepted",
                causality_by_variant[main_variant]["object_birth_chunk_causal_proven"],
            )
        ),
        "surfel_identity_user_waiver": bool(causality_by_variant[main_variant].get("surfel_identity_user_waiver", False)),
        "method_chunk_size": CHUNK_SIZE,
        "frame_stride": FRAME_STRIDE,
        "source_row_count": scope["source_row_count"],
        "base_input_row_count": len(base_rows),
        "mv_object_frame_mask_row_count": len(replay_rows),
        "variant_count": len(aggregate_rows),
        "chunk_semantic_birth_repair_stats": semantic_stats,
        "chunk_surfel_maskview_birth_repair_stats": surfel_stats,
        "outputs": {
            "mv_object_rows": _rel(OUT_DIR / "mv_object_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "causality_audit_rows": _rel(OUT_DIR / "causality_audit_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }

    _write_csv(OUT_DIR / "mv_object_rows.csv", object_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", replay_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gates)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "causality_audit_rows.csv", causality_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if phase1_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
