from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase6OcclusionReappearanceConfig
from .phase3_handoff import _real_label_metrics


@dataclass
class MaskDescriptor:
    key: str
    kind: str
    global_id: int | None
    obj_id: int | None
    frame_id: int
    chunk_index: int
    mask_area: int
    bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy_norm: Tuple[float, float]
    bbox_wh_norm: Tuple[float, float]
    aspect_ratio: float
    extent: float
    sam2_vector: List[float] | None
    sam2_available: bool
    provenance: str


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def _load_mask(path: Path, shape: Tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    mask = img > 0
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask.astype(bool)


def _summary_label_map(repo_root: Path, summary_path: Path) -> Dict[int, Path]:
    payload = _read_json(summary_path)
    out: Dict[int, Path] = {}
    for row in payload.get("records", []):
        out[int(row["frame_id"])] = _resolve(repo_root, row["label_path"])
    return out


def _visible_global_ids(label: np.ndarray) -> List[int]:
    return [int(v) for v in np.unique(label.astype(np.int64)).tolist() if int(v) > 0]


def _mask_for_global_id(label: np.ndarray, global_id: int) -> np.ndarray:
    return (label == int(global_id)).astype(bool)


def _mask_for_obj_id(label: np.ndarray, obj_id: int) -> np.ndarray:
    return (label == int(obj_id) + 1).astype(bool)


def _relabeled_label(label: np.ndarray, assignments: Dict[int, int]) -> np.ndarray:
    out = label.copy()
    for candidate_obj_id, history_global_id in assignments.items():
        out[label == int(candidate_obj_id) + 1] = int(history_global_id)
    return out


def _write_label(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), label.astype(np.uint16, copy=False))
    if not ok:
        raise IOError(f"failed to write label: {path}")


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _descriptor_stats(mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape[:2]
    area = int(np.count_nonzero(mask))
    x0, y0, x1, y1 = _bbox(mask)
    bw = max(1, int(x1 - x0))
    bh = max(1, int(y1 - y0))
    if area > 0:
        ys, xs = np.where(mask)
        cx = float(xs.mean() / max(1, w - 1))
        cy = float(ys.mean() / max(1, h - 1))
    else:
        cx = 0.0
        cy = 0.0
    return {
        "mask_area": int(area),
        "bbox_xyxy": (int(x0), int(y0), int(x1), int(y1)),
        "centroid_xy_norm": (float(cx), float(cy)),
        "bbox_wh_norm": (float(bw / max(1, w)), float(bh / max(1, h))),
        "aspect_ratio": float(bw / max(1, bh)),
        "extent": float(area / max(1, bw * bh)),
    }


def _descriptor_from_mask(
    *,
    key: str,
    kind: str,
    global_id: int | None,
    obj_id: int | None,
    frame_id: int,
    chunk_index: int,
    mask: np.ndarray,
    sam2_vector: List[float] | None,
    provenance: str,
) -> MaskDescriptor:
    stats = _descriptor_stats(mask)
    return MaskDescriptor(
        key=key,
        kind=kind,
        global_id=global_id,
        obj_id=obj_id,
        frame_id=int(frame_id),
        chunk_index=int(chunk_index),
        mask_area=int(stats["mask_area"]),
        bbox_xyxy=tuple(int(v) for v in stats["bbox_xyxy"]),
        centroid_xy_norm=tuple(float(v) for v in stats["centroid_xy_norm"]),
        bbox_wh_norm=tuple(float(v) for v in stats["bbox_wh_norm"]),
        aspect_ratio=float(stats["aspect_ratio"]),
        extent=float(stats["extent"]),
        sam2_vector=sam2_vector,
        sam2_available=sam2_vector is not None,
        provenance=provenance,
    )


def _as_json_descriptor(desc: MaskDescriptor) -> Dict[str, Any]:
    return {
        "key": desc.key,
        "kind": desc.kind,
        "global_id": desc.global_id,
        "obj_id": desc.obj_id,
        "frame_id": int(desc.frame_id),
        "chunk_index": int(desc.chunk_index),
        "mask_area": int(desc.mask_area),
        "bbox_xyxy": [int(v) for v in desc.bbox_xyxy],
        "centroid_xy_norm": [float(v) for v in desc.centroid_xy_norm],
        "bbox_wh_norm": [float(v) for v in desc.bbox_wh_norm],
        "aspect_ratio": float(desc.aspect_ratio),
        "extent": float(desc.extent),
        "sam2_available": bool(desc.sam2_available),
        "sam2_vector_l2_norm": _vector_norm(desc.sam2_vector) if desc.sam2_vector is not None else None,
        "provenance": desc.provenance,
    }


def _vector_norm(vec: List[float] | None) -> float | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    return float(np.linalg.norm(arr))


def _cosine(a: List[float] | None, b: List[float] | None) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0.0:
        return None
    return float(np.dot(av, bv) / denom)


def _ratio_score(a: float, b: float) -> float:
    a = float(max(a, 1.0e-9))
    b = float(max(b, 1.0e-9))
    return float(min(a, b) / max(a, b))


def _shape_similarity(a: MaskDescriptor, b: MaskDescriptor) -> float:
    aw, ah = a.bbox_wh_norm
    bw, bh = b.bbox_wh_norm
    dim_score = 0.5 * (_ratio_score(aw, bw) + _ratio_score(ah, bh))
    aspect_score = _ratio_score(a.aspect_ratio, b.aspect_ratio)
    extent_score = _ratio_score(a.extent, b.extent)
    dx = float(a.centroid_xy_norm[0] - b.centroid_xy_norm[0])
    dy = float(a.centroid_xy_norm[1] - b.centroid_xy_norm[1])
    centroid_score = max(0.0, 1.0 - float((dx * dx + dy * dy) ** 0.5) / (2.0**0.5))
    return float(0.35 * dim_score + 0.25 * aspect_score + 0.20 * extent_score + 0.20 * centroid_score)


def _candidate_by_scheduled_obj(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        role = str(row.get("phase5_role", ""))
        if role != "birth_new":
            continue
        obj_id = int(row["obj_id"])
        out[obj_id] = row
    return out


def _load_phase5_classification(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = _read_json(path)
    rows = payload if isinstance(payload, list) else payload.get("records", [])
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        scheduled = row.get("scheduled_obj_id")
        if scheduled is not None and str(row.get("action")) == "birth_new":
            out[int(scheduled)] = row
    return out


def _history_records(
    *,
    repo_root: Path,
    config: Phase6OcclusionReappearanceConfig,
    scene_state: Dict[str, Any],
    c0_labels: Dict[int, Path],
    c1_labels: Dict[int, Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_records = scene_state.get("chunk_records", [])
    previous = None
    for row in chunk_records:
        if int(row.get("chunk_index")) == int(config.previous_chunk_index):
            previous = row
            break
    if previous is None:
        raise ValueError(f"missing previous chunk state for chunk {config.previous_chunk_index}")
    occluded = [int(v) for v in previous.get("occluded_global_ids", [])]
    histories = []
    stale_histories = []
    for global_id in sorted(occluded):
        if int(config.current_chunk_index) - int(config.previous_chunk_index) > int(config.max_occlusion_age_chunks):
            continue
        real = _last_visible_record(global_id, [(1, c1_labels), (0, c0_labels)])
        stale = _last_visible_record(global_id, [(0, c0_labels)])
        if real is not None:
            histories.append(real)
        if stale is not None:
            stale_histories.append(stale)
    return histories, stale_histories


def _last_visible_record(global_id: int, chunks: List[Tuple[int, Dict[int, Path]]]) -> Dict[str, Any] | None:
    for chunk_index, label_map in chunks:
        for frame_id in sorted(label_map.keys(), reverse=True):
            label = _load_label(label_map[frame_id])
            mask = _mask_for_global_id(label, global_id)
            area = int(np.count_nonzero(mask))
            if area > 0:
                return {
                    "global_id": int(global_id),
                    "frame_id": int(frame_id),
                    "chunk_index": int(chunk_index),
                    "mask": mask,
                    "mask_area": int(area),
                    "label_path": str(label_map[frame_id]),
                }
    return None


def _candidate_records(
    *,
    repo_root: Path,
    config: Phase6OcclusionReappearanceConfig,
    phase5_birth_records: Dict[str, Any],
    classification_by_obj: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_obj = _candidate_by_scheduled_obj(phase5_birth_records.get("rows", []))
    out = []
    for obj_id, row in sorted(rows_by_obj.items()):
        mask_path = _resolve(repo_root, row["mask_path"])
        mask = _load_mask(mask_path)
        cls = classification_by_obj.get(int(obj_id), {})
        out.append(
            {
                "obj_id": int(obj_id),
                "scheduled_global_id": int(obj_id) + 1,
                "frame_id": int(row["frame_id"]),
                "chunk_index": int(config.current_chunk_index),
                "chunk_frame_index": int(row["chunk_frame_index"]),
                "mask": mask,
                "mask_area": int(np.count_nonzero(mask)),
                "mask_path": str(mask_path),
                "persistence_frames": int(cls.get("persistence_frames_from_anchor", 0) or 0),
                "best_inherited_obj_id": cls.get("best_inherited_obj_id"),
                "best_overlap_coeff": float(cls.get("best_overlap_coeff", 0.0) or 0.0),
            }
        )
    return out


def _pool_sam2_vectors(
    *,
    repo_root: Path,
    config: Phase6OcclusionReappearanceConfig,
    requests: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    cache_path = _resolve(repo_root, config.sam2_descriptor_cache) if config.sam2_descriptor_cache else output_dir / "sam2_pooled_descriptors.json"
    required_keys = sorted({str(row["key"]) for row in requests})
    if cache_path.exists():
        cached = _read_json(cache_path)
        vectors = cached.get("vectors", {})
        if all(key in vectors for key in required_keys):
            cached["cache_used"] = True
            return cached

    if not bool(config.use_sam2_descriptors):
        payload = {
            "schema_version": "stream4d_v106_phase6_sam2_descriptor_cache_v1",
            "sam2_available": False,
            "cache_used": False,
            "disabled_by_config": True,
            "vectors": {},
        }
        write_json(cache_path, payload)
        return payload

    import torch
    from types import SimpleNamespace

    from Stream3D.sgq_v105.sam2_feature_bank import Sam2FrameFeatureBank
    from tools.audit_v105_baseline_x_sam2_twostage_tracking import setup_models
    from tools.build_v105_phase5_frozen_birth_replay import make_baseline_args

    t0 = time.time()
    frame_ids = sorted({int(row["frame_id"]) for row in requests})
    rgb_root = _resolve(repo_root, config.rgb_root) / config.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:8])

    cli = SimpleNamespace(
        scene_id=config.scene_id,
        rgb_root=config.rgb_root,
        frame_start=frame_ids[0],
        frame_stride=1,
        frame_count=len(frame_ids),
        frame_ids=",".join(str(v) for v in frame_ids),
        output_root=str(output_dir / "sam2_descriptor_model_tmp"),
        seed=0,
    )
    args = make_baseline_args(_resolve(repo_root, config.sam2_baseline_config), cli)
    args.model_dtype = str(config.sam2_descriptor_model_dtype)
    models = setup_models(args)
    tracker_model = models["tracker_model"]
    bank = Sam2FrameFeatureBank(storage_device="cuda", clone_tensors=False)
    dtype_name = str(config.sam2_descriptor_model_dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    elif dtype_name in {"fp16", "float16"}:
        autocast_context = torch.autocast("cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext

        autocast_context = nullcontext()
    with autocast_context:
        bank.build_for_video_paths(tracker_model, frame_ids=frame_ids, frame_paths=frame_paths)

    requests_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in requests:
        requests_by_frame[int(row["frame_id"])].append(row)

    vectors: Dict[str, List[float]] = {}
    for frame_id in frame_ids:
        record = bank.get_image_features(frame_id)
        features = record.backbone_out.get(str(config.sam2_descriptor_layer))
        if features is None:
            raise KeyError(f"SAM2 feature layer missing: {config.sam2_descriptor_layer}")
        if features.ndim != 4:
            raise ValueError(f"unexpected feature shape for {config.sam2_descriptor_layer}: {tuple(features.shape)}")
        feat = features[0].float()
        _, fh, fw = feat.shape
        for req in requests_by_frame[frame_id]:
            mask = req["mask"]
            small = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
            if not np.any(small):
                vectors[str(req["key"])] = []
                continue
            mask_t = torch.as_tensor(small, dtype=torch.bool, device=feat.device)
            pooled = feat[:, mask_t].mean(dim=1)
            pooled = pooled / torch.clamp(torch.linalg.vector_norm(pooled), min=1.0e-12)
            vectors[str(req["key"])] = [float(v) for v in pooled.detach().cpu().tolist()]

    payload = {
        "schema_version": "stream4d_v106_phase6_sam2_descriptor_cache_v1",
        "sam2_available": True,
        "cache_used": False,
        "descriptor_layer": str(config.sam2_descriptor_layer),
        "model_dtype": str(config.sam2_descriptor_model_dtype),
        "frame_ids": frame_ids,
        "frame_count": int(len(frame_ids)),
        "request_count": int(len(requests)),
        "vector_dim": int(len(next(iter(vectors.values()), []))),
        "build_runtime_sec": float(time.time() - t0),
        "feature_bank_summary": bank.summary().to_json() if hasattr(bank.summary(), "to_json") else bank.summary(),
        "vectors": vectors,
    }
    write_json(cache_path, payload)
    del bank
    del models
    torch.cuda.empty_cache()
    return payload


def _make_descriptors(
    *,
    repo_root: Path,
    config: Phase6OcclusionReappearanceConfig,
    output_dir: Path,
    histories: List[Dict[str, Any]],
    stale_histories: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    requests = []
    for row in histories:
        requests.append({"key": f"history:{row['global_id']}", "frame_id": int(row["frame_id"]), "mask": row["mask"]})
    for row in stale_histories:
        requests.append({"key": f"history_stale:{row['global_id']}", "frame_id": int(row["frame_id"]), "mask": row["mask"]})
    for row in candidates:
        requests.append({"key": f"candidate:{row['obj_id']}", "frame_id": int(row["frame_id"]), "mask": row["mask"]})
    sam2_cache = _pool_sam2_vectors(repo_root=repo_root, config=config, requests=requests, output_dir=output_dir)
    vectors = sam2_cache.get("vectors", {})

    history_desc = []
    stale_desc = []
    candidate_desc = []
    for row in histories:
        key = f"history:{row['global_id']}"
        vec = vectors.get(key)
        history_desc.append(
            _descriptor_from_mask(
                key=key,
                kind="history",
                global_id=int(row["global_id"]),
                obj_id=None,
                frame_id=int(row["frame_id"]),
                chunk_index=int(row["chunk_index"]),
                mask=row["mask"],
                sam2_vector=vec if vec else None,
                provenance=str(row["label_path"]),
            )
        )
    for row in stale_histories:
        key = f"history_stale:{row['global_id']}"
        vec = vectors.get(key)
        stale_desc.append(
            _descriptor_from_mask(
                key=key,
                kind="history_stale",
                global_id=int(row["global_id"]),
                obj_id=None,
                frame_id=int(row["frame_id"]),
                chunk_index=int(row["chunk_index"]),
                mask=row["mask"],
                sam2_vector=vec if vec else None,
                provenance=str(row["label_path"]),
            )
        )
    for row in candidates:
        key = f"candidate:{row['obj_id']}"
        vec = vectors.get(key)
        candidate_desc.append(
            _descriptor_from_mask(
                key=key,
                kind="candidate",
                global_id=int(row["scheduled_global_id"]),
                obj_id=int(row["obj_id"]),
                frame_id=int(row["frame_id"]),
                chunk_index=int(row["chunk_index"]),
                mask=row["mask"],
                sam2_vector=vec if vec else None,
                provenance=str(row["mask_path"]),
            )
        )
    return {
        "sam2_cache": sam2_cache,
        "histories": history_desc,
        "stale_histories": stale_desc,
        "candidates": candidate_desc,
    }


def _score_pair(
    candidate: MaskDescriptor,
    history: MaskDescriptor,
    *,
    persistence_frames: int,
    occlusion_age_chunks: int,
    max_occlusion_age_chunks: int,
    conflict_score: float,
) -> Dict[str, Any]:
    cos = _cosine(candidate.sam2_vector, history.sam2_vector)
    sam2_score = 0.5 if cos is None else max(0.0, min(1.0, (float(cos) + 1.0) * 0.5))
    shape_score = _shape_similarity(candidate, history)
    area_score = _ratio_score(float(candidate.mask_area), float(history.mask_area))
    time_score = max(0.0, 1.0 - float(max(0, occlusion_age_chunks)) / max(1.0, float(max_occlusion_age_chunks)))
    persistence_score = max(0.0, min(1.0, float(persistence_frames) / 3.0))
    score = (
        0.45 * sam2_score
        + 0.20 * shape_score
        + 0.15 * area_score
        + 0.10 * time_score
        + 0.10 * persistence_score
        - 0.50 * float(conflict_score)
    )
    return {
        "score": float(score),
        "S_sam2": float(sam2_score),
        "sam2_cosine": cos,
        "S_shape": float(shape_score),
        "S_area": float(area_score),
        "S_time": float(time_score),
        "S_persistence": float(persistence_score),
        "S_conflict": float(conflict_score),
    }


def _run_matching(
    *,
    histories: List[MaskDescriptor],
    candidates: List[MaskDescriptor],
    classification_by_obj: Dict[int, Dict[str, Any]],
    config: Phase6OcclusionReappearanceConfig,
    control_name: str,
) -> Dict[str, Any]:
    histories_by_gid = {int(h.global_id): h for h in histories if h.global_id is not None}
    pair_records = []
    candidate_rankings: Dict[int, List[Dict[str, Any]]] = {}
    for cand in candidates:
        if cand.obj_id is None:
            continue
        cls = classification_by_obj.get(int(cand.obj_id), {})
        persistence_frames = int(cls.get("persistence_frames_from_anchor", 0) or 0)
        conflict_score = max(0.0, min(1.0, float(cls.get("best_overlap_coeff", 0.0) or 0.0)))
        rows = []
        for hist in histories_by_gid.values():
            age = int(config.current_chunk_index) - int(hist.chunk_index)
            score = _score_pair(
                cand,
                hist,
                persistence_frames=persistence_frames,
                occlusion_age_chunks=age,
                max_occlusion_age_chunks=int(config.max_occlusion_age_chunks),
                conflict_score=conflict_score,
            )
            rec = {
                "control": control_name,
                "candidate_obj_id": int(cand.obj_id),
                "candidate_frame_id": int(cand.frame_id),
                "history_global_id": int(hist.global_id),
                "history_frame_id": int(hist.frame_id),
                "history_chunk_index": int(hist.chunk_index),
                "occlusion_age_chunks": int(age),
                **score,
            }
            rows.append(rec)
            pair_records.append(rec)
        rows.sort(key=lambda item: item["score"], reverse=True)
        candidate_rankings[int(cand.obj_id)] = rows

    proposals = []
    for cand_obj_id, rows in candidate_rankings.items():
        if not rows:
            continue
        top = rows[0]
        second_score = float(rows[1]["score"]) if len(rows) > 1 else -1.0
        margin = float(top["score"] - second_score)
        status = "confirmed" if top["score"] >= config.tau_confirm and margin >= config.tau_margin else "new_identity"
        if status == "new_identity" and top["score"] >= config.tentative_tau_confirm:
            status = "tentative_reappearance"
        proposals.append({**top, "top2_score": second_score, "margin": margin, "status": status})

    proposals.sort(key=lambda item: item["score"], reverse=True)
    used_histories: set[int] = set()
    assignments = []
    tentative = []
    rejected = []
    for row in proposals:
        gid = int(row["history_global_id"])
        if row["status"] == "confirmed" and gid not in used_histories:
            used_histories.add(gid)
            assignments.append(row)
        elif row["status"] == "tentative_reappearance":
            tentative.append(row)
        else:
            copied = dict(row)
            if row["status"] == "confirmed" and gid in used_histories:
                copied["status"] = "rejected_one_to_one_conflict"
            rejected.append(copied)

    assigned_scores = [float(row["score"]) for row in assignments]
    return {
        "control": control_name,
        "pair_records": pair_records,
        "candidate_rankings": {str(k): v[:5] for k, v in candidate_rankings.items()},
        "assignments": assignments,
        "tentative": tentative,
        "rejected": rejected,
        "confirmed_count": int(len(assignments)),
        "tentative_count": int(len(tentative)),
        "mean_confirmed_score": float(np.mean(assigned_scores)) if assigned_scores else 0.0,
        "max_confirmed_score": float(max(assigned_scores)) if assigned_scores else 0.0,
    }


def _shuffle_histories(histories: List[MaskDescriptor], seed: int = 106) -> List[MaskDescriptor]:
    rng = random.Random(seed)
    payload = [
        {
            "mask_area": h.mask_area,
            "bbox_xyxy": h.bbox_xyxy,
            "centroid_xy_norm": h.centroid_xy_norm,
            "bbox_wh_norm": h.bbox_wh_norm,
            "aspect_ratio": h.aspect_ratio,
            "extent": h.extent,
            "sam2_vector": h.sam2_vector,
            "provenance": h.provenance,
            "frame_id": h.frame_id,
            "chunk_index": h.chunk_index,
        }
        for h in histories
    ]
    rng.shuffle(payload)
    out = []
    for h, shuffled in zip(histories, payload, strict=True):
        out.append(
            MaskDescriptor(
                key=f"{h.key}:shuffled",
                kind="history_shuffled",
                global_id=h.global_id,
                obj_id=h.obj_id,
                frame_id=int(shuffled["frame_id"]),
                chunk_index=int(shuffled["chunk_index"]),
                mask_area=int(shuffled["mask_area"]),
                bbox_xyxy=tuple(shuffled["bbox_xyxy"]),
                centroid_xy_norm=tuple(shuffled["centroid_xy_norm"]),
                bbox_wh_norm=tuple(shuffled["bbox_wh_norm"]),
                aspect_ratio=float(shuffled["aspect_ratio"]),
                extent=float(shuffled["extent"]),
                sam2_vector=shuffled["sam2_vector"],
                sam2_available=shuffled["sam2_vector"] is not None,
                provenance=str(shuffled["provenance"]),
            )
        )
    return out


def _stale_for_real(histories: List[MaskDescriptor], stale_histories: List[MaskDescriptor]) -> List[MaskDescriptor]:
    stale_by_gid = {int(h.global_id): h for h in stale_histories if h.global_id is not None}
    out = []
    for hist in histories:
        stale = stale_by_gid.get(int(hist.global_id))
        out.append(stale if stale is not None else hist)
    return out


def _random_one_to_one_control(
    *,
    histories: List[MaskDescriptor],
    candidates: List[MaskDescriptor],
    classification_by_obj: Dict[int, Dict[str, Any]],
    config: Phase6OcclusionReappearanceConfig,
) -> Dict[str, Any]:
    rng = random.Random(6106)
    shuffled = histories[:]
    rng.shuffle(shuffled)
    assignments = []
    for cand, hist in zip(candidates, shuffled, strict=False):
        if cand.obj_id is None or hist.global_id is None:
            continue
        cls = classification_by_obj.get(int(cand.obj_id), {})
        score = _score_pair(
            cand,
            hist,
            persistence_frames=int(cls.get("persistence_frames_from_anchor", 0) or 0),
            occlusion_age_chunks=int(config.current_chunk_index) - int(hist.chunk_index),
            max_occlusion_age_chunks=int(config.max_occlusion_age_chunks),
            conflict_score=float(cls.get("best_overlap_coeff", 0.0) or 0.0),
        )
        assignments.append(
            {
                "control": "random_one_to_one",
                "candidate_obj_id": int(cand.obj_id),
                "history_global_id": int(hist.global_id),
                **score,
            }
        )
    scores = [float(row["score"]) for row in assignments]
    return {
        "control": "random_one_to_one",
        "assignments": assignments,
        "confirmed_count": int(len(assignments)),
        "mean_confirmed_score": float(np.mean(scores)) if scores else 0.0,
        "max_confirmed_score": float(max(scores)) if scores else 0.0,
    }


def _score_real_pairs_under_control(
    *,
    real_assignments: List[Dict[str, Any]],
    control_name: str,
    histories: List[MaskDescriptor],
    candidates: List[MaskDescriptor],
    classification_by_obj: Dict[int, Dict[str, Any]],
    config: Phase6OcclusionReappearanceConfig,
) -> Dict[str, Any]:
    histories_by_gid = {int(h.global_id): h for h in histories if h.global_id is not None}
    candidates_by_obj = {int(c.obj_id): c for c in candidates if c.obj_id is not None}
    records = []
    for row in real_assignments:
        cand_obj_id = int(row["candidate_obj_id"])
        history_global_id = int(row["history_global_id"])
        cand = candidates_by_obj.get(cand_obj_id)
        hist = histories_by_gid.get(history_global_id)
        if cand is None or hist is None:
            continue
        cls = classification_by_obj.get(cand_obj_id, {})
        score = _score_pair(
            cand,
            hist,
            persistence_frames=int(cls.get("persistence_frames_from_anchor", 0) or 0),
            occlusion_age_chunks=int(config.current_chunk_index) - int(hist.chunk_index),
            max_occlusion_age_chunks=int(config.max_occlusion_age_chunks),
            conflict_score=float(cls.get("best_overlap_coeff", 0.0) or 0.0),
        )
        records.append(
            {
                "control": control_name,
                "candidate_obj_id": cand_obj_id,
                "history_global_id": history_global_id,
                "real_score": float(row["score"]),
                "control_same_pair_score": float(score["score"]),
                "score_delta_real_minus_control": float(row["score"]) - float(score["score"]),
                **score,
            }
        )
    deltas = [float(r["score_delta_real_minus_control"]) for r in records]
    control_scores = [float(r["control_same_pair_score"]) for r in records]
    real_scores = [float(r["real_score"]) for r in records]
    return {
        "control": control_name,
        "pair_count": int(len(records)),
        "records": records,
        "real_assignment_mean_score": float(np.mean(real_scores)) if real_scores else 0.0,
        "control_same_pair_mean_score": float(np.mean(control_scores)) if control_scores else 0.0,
        "mean_delta_real_minus_control": float(np.mean(deltas)) if deltas else 0.0,
        "positive_delta_count": int(sum(1 for v in deltas if v > 0.0)),
    }


def _metrics_for_labels(
    *,
    repo_root: Path,
    pred_labels: Dict[int, Path],
    reference_summary_path: Path,
    config: Phase6OcclusionReappearanceConfig,
) -> Dict[str, Any]:
    reference_labels = _summary_label_map(repo_root, reference_summary_path)
    frame_ids = sorted(set(pred_labels) & set(reference_labels))
    asa_num = 0
    asa_den = 0
    ttp_overlap: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ttp_den_by_pred: Dict[int, int] = defaultdict(int)
    records = []
    for frame_id in frame_ids:
        pred = _load_label(pred_labels[frame_id])
        ref = _load_label(reference_labels[frame_id])
        ref_ids = _visible_global_ids(ref)
        for pred_id in _visible_global_ids(pred):
            pred_mask = _mask_for_global_id(pred, pred_id)
            pred_area = int(np.count_nonzero(pred_mask))
            asa_den += pred_area
            ttp_den_by_pred[int(pred_id)] += pred_area
            best = 0
            for ref_id in ref_ids:
                overlap = int(np.count_nonzero(pred_mask & _mask_for_global_id(ref, ref_id)))
                best = max(best, overlap)
                ttp_overlap[int(pred_id)][int(ref_id)] += overlap
            asa_num += int(best)
        row = _real_label_metrics(
            ref,
            pred,
            fragment_overlap_fraction_threshold=float(config.fragment_overlap_fraction_threshold),
            merge_overlap_fraction_threshold=float(config.merge_overlap_fraction_threshold),
        )
        row.update({"frame_id": int(frame_id), "pred_label_path": str(pred_labels[frame_id])})
        records.append(row)
    ttp_num = sum(max(v.values()) if v else 0 for v in ttp_overlap.values())
    ttp_den = sum(ttp_den_by_pred.values())
    summary = {
        "frame_count": int(len(frame_ids)),
        "asa": float(asa_num / asa_den) if asa_den else 1.0,
        "ttp": float(ttp_num / ttp_den) if ttp_den else 1.0,
        "asa_numerator": int(asa_num),
        "asa_denominator": int(asa_den),
        "ttp_numerator": int(ttp_num),
        "ttp_denominator": int(ttp_den),
        "max_cmr": float(max([row["CMR"] for row in records], default=0.0)),
        "max_bfmr": float(max([row["BFMR"] for row in records], default=0.0)),
        "max_cfr": float(max([row["CFR"] for row in records], default=0.0)),
        "mean_foreground_union_iou": float(np.mean([row["foreground_union_iou"] for row in records]))
        if records else 1.0,
    }
    return {"summary": summary, "records": records}


def _write_relabeled_summary(
    *,
    repo_root: Path,
    output_dir: Path,
    phase5_replay_summary: Dict[str, Any],
    assignments: List[Dict[str, Any]],
) -> Tuple[Dict[int, Path], Dict[str, Any]]:
    assignment_map = {
        int(row["candidate_obj_id"]): int(row["history_global_id"])
        for row in assignments
    }
    label_dir = output_dir / "L2_reappearance_relabeled" / "labels"
    label_map = {}
    records = []
    for row in phase5_replay_summary.get("records", []):
        frame_id = int(row["frame_id"])
        src = _resolve(repo_root, row["label_path"])
        label = _load_label(src)
        relabeled = _relabeled_label(label, assignment_map)
        dst = label_dir / f"frame_{frame_id:06d}.png"
        _write_label(dst, relabeled)
        label_map[frame_id] = dst
        records.append(
            {
                "frame_id": frame_id,
                "label_path": str(dst),
                "source_label_path": str(src),
                "assignment_count": int(len(assignment_map)),
            }
        )
    summary = {
        "schema_version": "stream4d_v106_phase6_reappearance_relabeled_summary_v1",
        "frame_count": int(len(records)),
        "assignment_count": int(len(assignment_map)),
        "assignments": assignment_map,
        "records": records,
    }
    summary_path = output_dir / "L2_reappearance_relabeled" / "relabeled_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return label_map, summary


def run_phase6_occlusion_reappearance(
    repo_root: Path,
    config: Phase6OcclusionReappearanceConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scene_state": _resolve(repo_root, config.scene_state),
        "c0_summary": _resolve(repo_root, config.c0_summary),
        "c1_summary": _resolve(repo_root, config.c1_summary),
        "phase5_birth_records": _resolve(repo_root, config.phase5_birth_records),
        "phase5_classification_records": _resolve(repo_root, config.phase5_classification_records),
        "phase5_replay_summary": _resolve(repo_root, config.phase5_replay_summary),
        "reference_summary": _resolve(repo_root, config.reference_summary),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        summary = {
            "schema_version": "stream4d_v106_phase6_occlusion_reappearance_summary_v1",
            "passes": False,
            "missing_paths": missing,
        }
        write_json(output_dir / "gate_records.json", summary)
        write_json(output_dir / "failure_records.json", [{"failure": "missing_required_artifact", "paths": missing}])
        return summary

    scene_state = _read_json(paths["scene_state"])
    c0_labels = _summary_label_map(repo_root, paths["c0_summary"])
    c1_labels = _summary_label_map(repo_root, paths["c1_summary"])
    phase5_birth_records = _read_json(paths["phase5_birth_records"])
    phase5_replay_summary = _read_json(paths["phase5_replay_summary"])
    phase5_replay_labels = _summary_label_map(repo_root, paths["phase5_replay_summary"])
    classification_by_obj = _load_phase5_classification(paths["phase5_classification_records"])

    histories_raw, stale_histories_raw = _history_records(
        repo_root=repo_root,
        config=config,
        scene_state=scene_state,
        c0_labels=c0_labels,
        c1_labels=c1_labels,
    )
    candidates_raw = _candidate_records(
        repo_root=repo_root,
        config=config,
        phase5_birth_records=phase5_birth_records,
        classification_by_obj=classification_by_obj,
    )
    descriptors = _make_descriptors(
        repo_root=repo_root,
        config=config,
        output_dir=output_dir,
        histories=histories_raw,
        stale_histories=stale_histories_raw,
        candidates=candidates_raw,
    )
    histories = descriptors["histories"]
    stale_histories = descriptors["stale_histories"]
    candidates = descriptors["candidates"]

    real = _run_matching(
        histories=histories,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
        control_name="real",
    )
    shuffled_histories = _shuffle_histories(histories)
    shuffled = _run_matching(
        histories=shuffled_histories,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
        control_name="shuffled_history_descriptors",
    )
    shuffled_same_pair = _score_real_pairs_under_control(
        real_assignments=real["assignments"],
        control_name="shuffled_history_descriptors_same_real_pairs",
        histories=shuffled_histories,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
    )
    stale_histories_for_real = _stale_for_real(histories, stale_histories)
    stale = _run_matching(
        histories=stale_histories_for_real,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
        control_name="stale_descriptors",
    )
    stale_same_pair = _score_real_pairs_under_control(
        real_assignments=real["assignments"],
        control_name="stale_descriptors_same_real_pairs",
        histories=stale_histories_for_real,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
    )
    no_history = {
        "control": "no_history",
        "assignments": [],
        "confirmed_count": 0,
        "mean_confirmed_score": 0.0,
        "max_confirmed_score": 0.0,
    }
    random_control = _random_one_to_one_control(
        histories=histories,
        candidates=candidates,
        classification_by_obj=classification_by_obj,
        config=config,
    )

    baseline_metrics = _metrics_for_labels(
        repo_root=repo_root,
        pred_labels=phase5_replay_labels,
        reference_summary_path=paths["reference_summary"],
        config=config,
    )
    relabeled_labels, relabeled_summary = _write_relabeled_summary(
        repo_root=repo_root,
        output_dir=output_dir,
        phase5_replay_summary=phase5_replay_summary,
        assignments=real["assignments"],
    )
    relabeled_metrics = _metrics_for_labels(
        repo_root=repo_root,
        pred_labels=relabeled_labels,
        reference_summary_path=paths["reference_summary"],
        config=config,
    )

    controls = {
        "shuffled_history_descriptors": {
            "confirmed_count": shuffled["confirmed_count"],
            "mean_confirmed_score": shuffled["mean_confirmed_score"],
            "max_confirmed_score": shuffled["max_confirmed_score"],
            "same_real_pair_mean_score": shuffled_same_pair["control_same_pair_mean_score"],
            "same_real_pair_mean_delta": shuffled_same_pair["mean_delta_real_minus_control"],
            "same_real_pair_positive_delta_count": shuffled_same_pair["positive_delta_count"],
        },
        "stale_descriptors": {
            "confirmed_count": stale["confirmed_count"],
            "mean_confirmed_score": stale["mean_confirmed_score"],
            "max_confirmed_score": stale["max_confirmed_score"],
            "same_real_pair_mean_score": stale_same_pair["control_same_pair_mean_score"],
            "same_real_pair_mean_delta": stale_same_pair["mean_delta_real_minus_control"],
            "same_real_pair_positive_delta_count": stale_same_pair["positive_delta_count"],
        },
        "no_history": no_history,
        "random_one_to_one": {
            "confirmed_count": random_control["confirmed_count"],
            "mean_confirmed_score": random_control["mean_confirmed_score"],
            "max_confirmed_score": random_control["max_confirmed_score"],
        },
    }

    checks = [
        {
            "name": "sam2_descriptor_available_for_L2",
            "passes": bool(descriptors["sam2_cache"].get("sam2_available")),
            "actual": {
                "sam2_available": descriptors["sam2_cache"].get("sam2_available"),
                "request_count": descriptors["sam2_cache"].get("request_count"),
                "vector_dim": descriptors["sam2_cache"].get("vector_dim"),
            },
            "expected": "real SAM2 pooled descriptors used, not geometry-only fallback",
        },
        {
            "name": "reappearance_confirmed_count_gt_0",
            "passes": int(real["confirmed_count"]) > 0,
            "actual": int(real["confirmed_count"]),
            "expected_min_exclusive": 0,
        },
        {
            "name": "ttp_not_down_after_relabel",
            "passes": float(relabeled_metrics["summary"]["ttp"]) + float(config.metric_epsilon)
            >= float(baseline_metrics["summary"]["ttp"]),
            "actual": {
                "before": baseline_metrics["summary"]["ttp"],
                "after": relabeled_metrics["summary"]["ttp"],
            },
            "expected": "after >= before",
        },
        {
            "name": "false_merge_not_up_after_relabel",
            "passes": (
                float(relabeled_metrics["summary"]["max_cmr"])
                <= float(baseline_metrics["summary"]["max_cmr"]) + float(config.metric_epsilon)
                and float(relabeled_metrics["summary"]["max_bfmr"])
                <= float(baseline_metrics["summary"]["max_bfmr"]) + float(config.metric_epsilon)
            ),
            "actual": {
                "max_cmr_before": baseline_metrics["summary"]["max_cmr"],
                "max_cmr_after": relabeled_metrics["summary"]["max_cmr"],
                "max_bfmr_before": baseline_metrics["summary"]["max_bfmr"],
                "max_bfmr_after": relabeled_metrics["summary"]["max_bfmr"],
            },
            "expected": "CMR and BFMR do not increase",
        },
        {
            "name": "real_better_than_shuffled",
            "passes": (
                float(real["mean_confirmed_score"])
                > float(shuffled_same_pair["control_same_pair_mean_score"]) + float(config.metric_epsilon)
            ),
            "actual": {
                "real_confirmed": real["confirmed_count"],
                "shuffled_confirmed": shuffled["confirmed_count"],
                "real_mean_score": real["mean_confirmed_score"],
                "shuffled_mean_score": shuffled["mean_confirmed_score"],
                "shuffled_same_real_pair_mean_score": shuffled_same_pair["control_same_pair_mean_score"],
                "same_real_pair_mean_delta": shuffled_same_pair["mean_delta_real_minus_control"],
            },
            "expected": "real assignment score higher than shuffled descriptor score on the same candidate-history pairs",
        },
        {
            "name": "real_better_than_stale",
            "passes": (
                float(real["mean_confirmed_score"])
                > float(stale_same_pair["control_same_pair_mean_score"]) + float(config.metric_epsilon)
            ),
            "actual": {
                "real_confirmed": real["confirmed_count"],
                "stale_confirmed": stale["confirmed_count"],
                "real_mean_score": real["mean_confirmed_score"],
                "stale_mean_score": stale["mean_confirmed_score"],
                "stale_same_real_pair_mean_score": stale_same_pair["control_same_pair_mean_score"],
                "same_real_pair_mean_delta": stale_same_pair["mean_delta_real_minus_control"],
            },
            "expected": "real assignment score higher than stale descriptor score on the same candidate-history pairs",
        },
    ]

    write_json(output_dir / "history_descriptor_records.json", [_as_json_descriptor(row) for row in histories])
    write_json(output_dir / "stale_history_descriptor_records.json", [_as_json_descriptor(row) for row in stale_histories])
    write_json(output_dir / "candidate_descriptor_records.json", [_as_json_descriptor(row) for row in candidates])
    write_json(output_dir / "real_pair_score_records.json", real["pair_records"])
    write_json(output_dir / "real_reappearance_match_records.json", real["assignments"])
    write_json(output_dir / "tentative_reappearance_records.json", real["tentative"])
    write_json(output_dir / "control_shuffled_records.json", shuffled)
    write_json(output_dir / "control_shuffled_same_real_pair_records.json", shuffled_same_pair)
    write_json(output_dir / "control_stale_records.json", stale)
    write_json(output_dir / "control_stale_same_real_pair_records.json", stale_same_pair)
    write_json(output_dir / "control_random_records.json", random_control)
    write_json(output_dir / "baseline_metric_records_vs_reference.json", baseline_metrics["records"])
    write_json(output_dir / "baseline_metric_summary_vs_reference.json", baseline_metrics["summary"])
    write_json(output_dir / "relabeled_metric_records_vs_reference.json", relabeled_metrics["records"])
    write_json(output_dir / "relabeled_metric_summary_vs_reference.json", relabeled_metrics["summary"])

    failure_records = [check for check in checks if not bool(check["passes"])]
    summary = {
        "schema_version": "stream4d_v106_phase6_occlusion_reappearance_summary_v1",
        "passes": not failure_records,
        "failure_count": int(len(failure_records)),
        "checks": checks,
        "scope": {
            "scene_id": config.scene_id,
            "current_chunk_index": int(config.current_chunk_index),
            "previous_chunk_index": int(config.previous_chunk_index),
            "history_occluded_count": int(len(histories)),
            "stale_history_count": int(len(stale_histories)),
            "candidate_birth_count": int(len(candidates)),
            "phase5_variant_birth_records": str(paths["phase5_birth_records"]),
            "phase5_replay_summary": str(paths["phase5_replay_summary"]),
            "reference_summary": str(paths["reference_summary"]),
        },
        "descriptor_provenance": {
            "sam2_cache_path": str(
                _resolve(repo_root, config.sam2_descriptor_cache)
                if config.sam2_descriptor_cache
                else output_dir / "sam2_pooled_descriptors.json"
            ),
            "sam2_available": bool(descriptors["sam2_cache"].get("sam2_available")),
            "sam2_cache_used": bool(descriptors["sam2_cache"].get("cache_used", False)),
            "descriptor_layer": descriptors["sam2_cache"].get("descriptor_layer"),
            "model_dtype": descriptors["sam2_cache"].get("model_dtype"),
            "build_runtime_sec": descriptors["sam2_cache"].get("build_runtime_sec"),
        },
        "real": {
            "confirmed_count": real["confirmed_count"],
            "tentative_count": real["tentative_count"],
            "mean_confirmed_score": real["mean_confirmed_score"],
            "max_confirmed_score": real["max_confirmed_score"],
            "assignments": real["assignments"],
        },
        "controls": controls,
        "baseline_metrics": baseline_metrics["summary"],
        "relabeled_metrics": relabeled_metrics["summary"],
        "relabeled_summary": relabeled_summary,
        "ap_generated": False,
        "mv_ap_scene_used_as_gate": False,
    }
    write_json(output_dir / "gate_records.json", summary)
    write_json(output_dir / "failure_records.json", failure_records)
    return summary
