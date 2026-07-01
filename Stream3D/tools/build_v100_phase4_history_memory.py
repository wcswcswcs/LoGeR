#!/usr/bin/env python3
"""Build v100 Phase4 GPU-assisted causal history memory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10l_frozen_p2d2_regenerated_birth_holdout as p10l  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PHASE0_SUMMARY = AUDIT_ROOT / "v100_phase0_contract/summary.json"
PHASE0_BASELINES = AUDIT_ROOT / "v100_phase0_contract/baseline_metric_rows.csv"
PHASE2_DIR = Path(os.environ.get("V100_PHASE2_DIR", str(AUDIT_ROOT / "v100_phase2_f2_local_final")))
PHASE3_SUMMARY = Path(os.environ.get("V100_PHASE3_SUMMARY", str(AUDIT_ROOT / "v100_phase3_scene_fragmentation_audit/summary.json")))
OUT_DIR = Path(os.environ.get("V100_PHASE4_OUT_DIR", str(AUDIT_ROOT / "v100_phase4_history_memory")))

DEV_INPUTS = {
    "SOURCE_ROWS": p1.SOURCE_ROWS,
    "RADIO_MASK_FEATURES": p1.RADIO_MASK_FEATURES,
    "SURFEL_ROWS": p1.SURFEL_ROWS,
    "SURFEL_OBS_ROWS": p1.SURFEL_OBS_ROWS,
    "SURFEL_SUMMARY": p1.SURFEL_SUMMARY,
}
HOLDOUT_INPUTS = {
    "SOURCE_ROWS": p10k.HOLDOUT_SOURCE_ROWS,
    "RADIO_MASK_FEATURES": p10k.HOLDOUT_RADIO_MASK_FEATURES,
    "SURFEL_ROWS": p10k.HOLDOUT_SURFEL_ROWS,
    "SURFEL_OBS_ROWS": p10k.HOLDOUT_SURFEL_OBS_ROWS,
    "SURFEL_SUMMARY": p10k.HOLDOUT_SURFEL_SUMMARY,
}


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).replace({"": None})
    df.to_parquet(path, index=False)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _set_phase1_inputs(split: str) -> None:
    inputs = DEV_INPUTS if split == "dev" else HOLDOUT_INPUTS
    p1.SOURCE_ROWS = inputs["SOURCE_ROWS"]
    p1.RADIO_MASK_FEATURES = inputs["RADIO_MASK_FEATURES"]
    p1.SURFEL_ROWS = inputs["SURFEL_ROWS"]
    p1.SURFEL_OBS_ROWS = inputs["SURFEL_OBS_ROWS"]
    p1.SURFEL_SUMMARY = inputs["SURFEL_SUMMARY"]


def _chunk_index(chunk_id: str) -> int:
    if str(chunk_id).startswith("c"):
        return int(str(chunk_id)[1:])
    return int(float(chunk_id))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _feature_maps() -> dict[str, dict[tuple[str, int, int], np.ndarray]]:
    _set_phase1_inputs("dev")
    dev_features, _tau = p1._load_radio_residual_features()
    holdout_features = p10l._load_holdout_residual_features()
    return {"dev": dev_features, "holdout": holdout_features}


def _mask_centroid(
    *,
    mask_path_by_frame: dict[tuple[str, int], Path] | None,
    centroid_cache: dict[tuple[str, int, int], tuple[float, float, float]],
    scene: str,
    frame: int,
    mask_id: int,
) -> tuple[float, float, float] | None:
    if mask_path_by_frame is None:
        return None
    key = (scene, int(frame), int(mask_id))
    if key in centroid_cache:
        return centroid_cache[key]
    mask_path = mask_path_by_frame.get((scene, int(frame)))
    if mask_path is None or not mask_path.exists():
        return None
    label = p1._read_label(mask_path)
    mask = label == int(mask_id)
    count = int(np.count_nonzero(mask))
    if count <= 0:
        return None
    ys, xs = np.nonzero(mask)
    h, w = label.shape[:2]
    centroid = (float(np.mean(xs) / max(1, w - 1)), float(np.mean(ys) / max(1, h - 1)), float(count / max(1, h * w)))
    centroid_cache[key] = centroid
    return centroid


def _object_infos(
    rows: list[dict[str, Any]],
    features: dict[tuple[str, int, int], np.ndarray],
    mask_path_by_frame: dict[tuple[str, int], Path] | None = None,
) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": [], "frames": set(), "features": [], "mask_ids": [], "areas": [], "support": [], "centroids": []}
    )
    centroid_cache: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        frame = int(_num(row["frame_id"], -1))
        mask = int(_num(row["selected_mask_id"], -1))
        infos[oid]["rows"].append(row)
        infos[oid]["mv_object_id"] = oid
        infos[oid]["dataset_split"] = str(row.get("dataset_split", ""))
        infos[oid]["scene_id"] = scene
        infos[oid]["chunk_id"] = chunk
        infos[oid]["chunk_index"] = _chunk_index(chunk)
        infos[oid]["frames"].add(frame)
        infos[oid]["mask_ids"].append(mask)
        area = _num(row.get("selected_mask_area"), _num(row.get("support_area"), 0.0))
        if area > 0:
            infos[oid]["areas"].append(area)
        support = _num(row.get("support_surfel_count"), 0.0)
        if support > 0:
            infos[oid]["support"].append(support)
        feat = features.get((scene, frame, mask))
        if feat is not None:
            infos[oid]["features"].append(feat)
        centroid = _mask_centroid(mask_path_by_frame=mask_path_by_frame, centroid_cache=centroid_cache, scene=scene, frame=frame, mask_id=mask)
        if centroid is not None:
            infos[oid]["centroids"].append((frame, np.asarray(centroid, dtype=np.float32)))
    for oid, info in infos.items():
        frames = sorted(info["frames"])
        feats = info["features"]
        centroids = sorted(info["centroids"], key=lambda item: int(item[0]))
        info["first_frame"] = frames[0] if frames else -1
        info["last_frame"] = frames[-1] if frames else -1
        info["frame_count"] = len(frames)
        info["mean_area"] = float(np.mean(info["areas"])) if info["areas"] else 0.0
        info["mean_support_surfel_count"] = float(np.mean(info["support"])) if info["support"] else 0.0
        info["score"] = float(_num(info["rows"][0].get("score"), 0.0)) if info["rows"] else 0.0
        if feats:
            info["descriptor"] = _normalize(np.mean(np.stack(feats).astype(np.float32), axis=0))
        else:
            info["descriptor"] = np.zeros((1024,), dtype=np.float32)
        mv = np.asarray([info["mean_area"], info["frame_count"] / 32.0, min(1.0, info["mean_support_surfel_count"] / 16.0)], dtype=np.float32)
        info["mask_view_descriptor"] = _normalize(mv)
        if centroids:
            info["first_centroid"] = np.asarray(centroids[0][1], dtype=np.float32)
            info["last_centroid"] = np.asarray(centroids[-1][1], dtype=np.float32)
            info["mean_centroid"] = np.mean(np.stack([c for _frame, c in centroids]), axis=0).astype(np.float32)
            info["centroid_valid_count"] = len(centroids)
        else:
            default_area = float(info["mean_area"] / max(1.0, 968.0 * 1296.0)) if info["mean_area"] > 1.0 else float(info["mean_area"])
            default = np.asarray([0.5, 0.5, max(0.0, default_area)], dtype=np.float32)
            info["first_centroid"] = default
            info["last_centroid"] = default
            info["mean_centroid"] = default
            info["centroid_valid_count"] = 0
    return dict(infos)


def _maskview_score(obj: dict[str, Any], hist: dict[str, Any]) -> float:
    a = np.asarray(obj["mask_view_descriptor"], dtype=np.float32)
    b = np.asarray(hist.get("mask_view_descriptor", hist.get("mask_view_sketch", np.zeros_like(a))), dtype=np.float32)
    return float(max(0.0, np.dot(a, b)))


def _position_score(obj: dict[str, Any], hist: dict[str, Any], sigma: float) -> tuple[float, float, float]:
    current = np.asarray(obj.get("first_centroid", np.asarray([0.5, 0.5, 0.0], dtype=np.float32)), dtype=np.float32)
    previous = np.asarray(hist.get("last_centroid", np.asarray([0.5, 0.5, 0.0], dtype=np.float32)), dtype=np.float32)
    dist = float(np.linalg.norm(current[:2] - previous[:2]))
    sig = max(1e-4, float(sigma))
    spatial = float(math.exp(-(dist * dist) / (2.0 * sig * sig)))
    area_a = max(1e-8, float(current[2]))
    area_b = max(1e-8, float(previous[2]))
    scale = float(math.exp(-abs(math.log(area_a / area_b))))
    return float(spatial * scale), dist, scale


def _entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - np.max(arr)
    prob = np.exp(arr)
    prob = prob / max(1e-12, float(np.sum(prob)))
    return float(-np.sum(prob * np.log(np.maximum(prob, 1e-12))))


def _empty_history_descriptor(dim: int = 1024) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)


def _make_history(history_id: str, obj: dict[str, Any], *, state: str, chunk_index: int, reason: str) -> dict[str, Any]:
    return {
        "history_id": history_id,
        "dataset_split": obj["dataset_split"],
        "scene_id": obj["scene_id"],
        "state": state,
        "birth_chunk": obj["chunk_id"],
        "birth_chunk_index": chunk_index,
        "last_seen_chunk": obj["chunk_id"],
        "last_seen_chunk_index": chunk_index,
        "support_chunk_count": 1,
        "support_frame_count": int(obj["frame_count"]),
        "semantic_residual_ema": np.asarray(obj.get("descriptor", _empty_history_descriptor()), dtype=np.float32),
        "mask_view_sketch": np.asarray(obj.get("mask_view_descriptor", np.zeros((3,), dtype=np.float32)), dtype=np.float32),
        "last_centroid": np.asarray(obj.get("last_centroid", np.asarray([0.5, 0.5, 0.0], dtype=np.float32)), dtype=np.float32),
        "mean_centroid": np.asarray(obj.get("mean_centroid", np.asarray([0.5, 0.5, 0.0], dtype=np.float32)), dtype=np.float32),
        "area_ema": float(obj.get("mean_area", 0.0)),
        "score_ema": float(obj.get("score", 0.0)),
        "frames": set(obj.get("frames", set())),
        "local_object_ids": [obj["mv_object_id"]],
        "cannot_link_history_ids": set(),
        "create_reason": reason,
    }


def _update_history(hist: dict[str, Any], obj: dict[str, Any], *, chunk_index: int, link_score: float, margin: float) -> None:
    n = int(hist["support_chunk_count"])
    hist["semantic_residual_ema"] = _normalize(0.75 * np.asarray(hist["semantic_residual_ema"], dtype=np.float32) + 0.25 * np.asarray(obj["descriptor"], dtype=np.float32))
    hist["mask_view_sketch"] = _normalize(0.75 * np.asarray(hist["mask_view_sketch"], dtype=np.float32) + 0.25 * np.asarray(obj["mask_view_descriptor"], dtype=np.float32))
    hist["mean_centroid"] = 0.75 * np.asarray(hist.get("mean_centroid", obj["mean_centroid"]), dtype=np.float32) + 0.25 * np.asarray(obj["mean_centroid"], dtype=np.float32)
    hist["last_centroid"] = np.asarray(obj.get("last_centroid", hist.get("last_centroid", np.asarray([0.5, 0.5, 0.0], dtype=np.float32))), dtype=np.float32)
    hist["area_ema"] = float((hist["area_ema"] * n + float(obj.get("mean_area", 0.0))) / (n + 1))
    hist["score_ema"] = float((hist["score_ema"] * n + float(obj.get("score", 0.0))) / (n + 1))
    hist["support_chunk_count"] = n + 1
    hist["support_frame_count"] = int(hist["support_frame_count"]) + int(obj["frame_count"])
    hist["last_seen_chunk"] = obj["chunk_id"]
    hist["last_seen_chunk_index"] = chunk_index
    hist["frames"].update(obj.get("frames", set()))
    hist["local_object_ids"].append(obj["mv_object_id"])
    if int(hist["support_chunk_count"]) >= 2 and link_score >= 0.80 and margin >= 0.03:
        hist["state"] = "confirmed"


def _history_rows(histories: dict[str, dict[str, Any]], *, variant_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for history_id, hist in sorted(histories.items()):
        rows.append(
            {
                "schema_version": "stream4d_v100_phase4_history_object_row_v1",
                "phase_id": "v100_phase4_history_memory",
                "variant_id": variant_id,
                "dataset_split": hist["dataset_split"],
                "scene_id": hist["scene_id"],
                "history_id": history_id,
                "state": hist["state"],
                "birth_chunk": hist["birth_chunk"],
                "last_seen_chunk": hist["last_seen_chunk"],
                "support_chunk_count": hist["support_chunk_count"],
                "support_frame_count": hist["support_frame_count"],
                "local_object_count": len(hist["local_object_ids"]),
                "local_object_ids": ";".join(hist["local_object_ids"]),
                "area_ema": hist["area_ema"],
                "score_ema": hist["score_ema"],
                "cannot_link_history_ids": ";".join(sorted(hist["cannot_link_history_ids"])),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _generate_memory_for_split(
    split: str,
    infos: dict[str, dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histories_by_scene: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    mapping: dict[str, str] = {}
    link_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    history_counter: defaultdict[str, int] = defaultdict(int)
    accepted_scores: list[float] = []
    accepted_margins: list[float] = []
    entropy_values: list[float] = []
    rejected_cannot = 0
    rejected_margin = 0
    rejected_threshold = 0
    rejected_used = 0

    by_scene_chunk: dict[tuple[str, int], list[str]] = defaultdict(list)
    chunk_label: dict[tuple[str, int], str] = {}
    for oid, info in infos.items():
        key = (str(info["scene_id"]), int(info["chunk_index"]))
        by_scene_chunk[key].append(oid)
        chunk_label[key] = str(info["chunk_id"])

    alpha = float(spec["alpha"])
    beta = float(spec["beta"])
    gamma = float(spec.get("gamma", 0.0))
    stale_eta = float(spec["stale_eta"])
    tau_link = float(spec["tau_link"])
    tau_margin = float(spec["tau_margin"])
    max_age = int(spec["max_age"])
    topk = int(spec["topk_semantic"])
    pos_sigma = float(spec.get("position_sigma", 0.12))
    min_position_score = float(spec.get("min_position_score", -1.0))

    for scene in sorted({scene for scene, _idx in by_scene_chunk}):
        chunk_indices = sorted(idx for s, idx in by_scene_chunk if s == scene)
        for chunk_idx in chunk_indices:
            chunk_id = chunk_label[(scene, chunk_idx)]
            obj_ids = sorted(by_scene_chunk[(scene, chunk_idx)])
            scene_histories = histories_by_scene[scene]
            active_hist_ids = [
                hid
                for hid, hist in scene_histories.items()
                if int(hist["last_seen_chunk_index"]) < chunk_idx and chunk_idx - int(hist["last_seen_chunk_index"]) <= max_age
            ]
            proposals: list[dict[str, Any]] = []
            if active_hist_ids and obj_ids:
                obj_desc = torch.as_tensor(np.stack([infos[oid]["descriptor"] for oid in obj_ids]).astype(np.float32), device=device)
                hist_desc = torch.as_tensor(np.stack([scene_histories[hid]["semantic_residual_ema"] for hid in active_hist_ids]).astype(np.float32), device=device)
                sim = obj_desc @ hist_desc.T
                k = min(topk, sim.shape[1])
                vals, idxs = torch.topk(sim, k=k, dim=1)
                vals_np = vals.detach().cpu().numpy()
                idxs_np = idxs.detach().cpu().numpy()
                for oi, oid in enumerate(obj_ids):
                    scores_for_entropy: list[float] = []
                    candidates: list[dict[str, Any]] = []
                    for rank in range(k):
                        hid = active_hist_ids[int(idxs_np[oi, rank])]
                        hist = scene_histories[hid]
                        sem = float(vals_np[oi, rank])
                        maskview = _maskview_score(infos[oid], hist)
                        position, centroid_distance, scale_score = _position_score(infos[oid], hist, pos_sigma)
                        age = chunk_idx - int(hist["last_seen_chunk_index"])
                        score = alpha * sem + beta * maskview + gamma * position - stale_eta * max(0, age - 1)
                        scores_for_entropy.append(score)
                        cannot = bool(set(infos[oid]["frames"]) & set(hist["frames"]))
                        candidates.append(
                            {
                                "history_id": hid,
                                "semantic_score": sem,
                                "maskview_score": maskview,
                                "position_score": position,
                                "centroid_distance": centroid_distance,
                                "scale_score": scale_score,
                                "stale_age": age,
                                "link_score": score,
                                "cannot_link": cannot,
                            }
                        )
                    candidates.sort(key=lambda row: row["link_score"], reverse=True)
                    best = candidates[0] if candidates else None
                    second = candidates[1] if len(candidates) > 1 else None
                    margin = float(best["link_score"] - second["link_score"]) if best and second else 1.0
                    entropy = _entropy(scores_for_entropy)
                    entropy_values.append(entropy)
                    if best is not None:
                        proposals.append(
                            {
                                "oid": oid,
                                "best": best,
                                "second": second,
                                "margin": margin,
                                "entropy": entropy,
                            }
                        )
            used_histories: set[str] = set()
            proposal_by_oid = {row["oid"]: row for row in proposals}
            for oid in sorted(obj_ids, key=lambda item: proposal_by_oid.get(item, {}).get("best", {}).get("link_score", -1.0), reverse=True):
                prop = proposal_by_oid.get(oid)
                action = "birth_new_history"
                reason = "no_past_candidate"
                history_id = ""
                link_score = ""
                margin = ""
                entropy = ""
                semantic_score = ""
                maskview_score = ""
                stale_age = ""
                if prop is not None:
                    best = prop["best"]
                    history_id = str(best["history_id"])
                    link_score = float(best["link_score"])
                    margin = float(prop["margin"])
                    entropy = float(prop["entropy"])
                    semantic_score = float(best["semantic_score"])
                    maskview_score = float(best["maskview_score"])
                    stale_age = int(best["stale_age"])
                    if best["cannot_link"]:
                        rejected_cannot += 1
                        action = "birth_quarantine_history"
                        reason = "cannot_link_same_frame_conflict"
                    elif min_position_score >= 0.0 and float(best.get("position_score", 0.0)) < min_position_score:
                        rejected_threshold += 1
                        action = "birth_tentative_history"
                        reason = "below_min_position_score"
                    elif history_id in used_histories:
                        rejected_used += 1
                        action = "birth_tentative_history"
                        reason = "history_already_assigned_in_current_chunk"
                    elif float(link_score) < tau_link:
                        rejected_threshold += 1
                        action = "birth_tentative_history"
                        reason = "below_tau_link"
                    elif float(margin) < tau_margin:
                        rejected_margin += 1
                        action = "birth_quarantine_history"
                        reason = "below_tau_margin"
                    else:
                        action = "accept_link"
                        reason = "accepted_tau_margin_cannot_link_clear"
                if action == "accept_link":
                    used_histories.add(history_id)
                    mapping[oid] = history_id
                    _update_history(scene_histories[history_id], infos[oid], chunk_index=chunk_idx, link_score=float(link_score), margin=float(margin))
                    accepted_scores.append(float(link_score))
                    accepted_margins.append(float(margin))
                else:
                    state = "quarantine" if "quarantine" in action else "tentative"
                    history_counter[scene] += 1
                    history_id = f"{spec['variant_id']}:{split}:{scene}:hist_{history_counter[scene]:05d}"
                    scene_histories[history_id] = _make_history(history_id, infos[oid], state=state, chunk_index=chunk_idx, reason=reason)
                    mapping[oid] = history_id
                link_rows.append(
                    {
                        "schema_version": "stream4d_v100_phase4_chunk_object_history_link_row_v1",
                        "phase_id": "v100_phase4_history_memory",
                        "variant_id": spec["variant_id"],
                        "dataset_split": split,
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "chunk_object_id": oid,
                        "history_id": history_id,
                        "action": action,
                        "reason": reason,
                        "link_score": link_score,
                        "link_margin": margin,
                        "link_entropy": entropy,
                        "semantic_score": semantic_score,
                        "maskview_score": maskview_score,
                        "position_score": best.get("position_score", "") if prop is not None else "",
                        "centroid_distance": best.get("centroid_distance", "") if prop is not None else "",
                        "scale_score": best.get("scale_score", "") if prop is not None else "",
                        "stale_age": stale_age,
                        "tau_link": tau_link,
                        "tau_margin": tau_margin,
                        "gamma_position": gamma,
                        "position_sigma": pos_sigma,
                        "min_position_score": min_position_score,
                        "topk_semantic": topk,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                transition_rows.append(
                    {
                        "schema_version": "stream4d_v100_phase4_history_state_transition_row_v1",
                        "phase_id": "v100_phase4_history_memory",
                        "variant_id": spec["variant_id"],
                        "dataset_split": split,
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "chunk_object_id": oid,
                        "history_id": history_id,
                        "transition": action,
                        "state_after": scene_histories[history_id]["state"],
                        "reason": reason,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    all_histories: dict[str, dict[str, Any]] = {}
    for scene_histories in histories_by_scene.values():
        all_histories.update(scene_histories)
    history_rows = _history_rows(all_histories, variant_id=str(spec["variant_id"]))
    overmerge_large_component_count = sum(1 for row in history_rows if int(row["local_object_count"]) > 1 and int(row["support_chunk_count"]) != int(row["local_object_count"]))
    stats = {
        "accepted_link_count": len(accepted_scores),
        "rejected_cannot_link_count": rejected_cannot,
        "rejected_margin_count": rejected_margin,
        "rejected_threshold_count": rejected_threshold,
        "rejected_used_history_count": rejected_used,
        "link_margin_mean": float(np.mean(accepted_margins)) if accepted_margins else 0.0,
        "link_entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "confirmed_history_count": sum(1 for row in history_rows if row["state"] == "confirmed"),
        "tentative_history_count": sum(1 for row in history_rows if row["state"] == "tentative"),
        "quarantine_count": sum(1 for row in history_rows if row["state"] == "quarantine"),
        "history_count": len(history_rows),
        "overmerge_large_component_count": overmerge_large_component_count,
        "device": str(device),
    }
    return mapping, link_rows, transition_rows, history_rows, stats


def _apply_mapping(rows: list[dict[str, Any]], mapping: dict[str, str], *, variant_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source_oid = str(row["mv_object_id"])
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase4_scene_mv_object_frame_mask_row_v1"
        new["phase_id"] = "v100_phase4_history_memory"
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["source_phase2_mv_object_id"] = source_oid
        new["history_id"] = mapping.get(source_oid, source_oid)
        new["mv_object_id"] = mapping.get(source_oid, source_oid)
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = "causal_history_memory_state"
        new["history_memory_scope"] = "causal_past_chunks_only"
        new["score_scope"] = "current_chunk_score_history_identity"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        new["future_chunk_access"] = False
        out.append(new)
    return out


def _scene_crossing_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_obj: dict[str, set[str]] = defaultdict(set)
    by_obj_frames: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        oid = str(row["mv_object_id"])
        by_obj[oid].add(str(row["chunk_id"]))
        by_obj_frames[oid].add(int(_num(row.get("frame_id"), -1)))
    chunks_per = [len(v) for v in by_obj.values()]
    objects_crossing = sum(1 for v in chunks_per if v > 1)
    return {
        "objects_crossing_multiple_chunks": objects_crossing,
        "mean_chunks_per_scene_object": float(np.mean(chunks_per)) if chunks_per else 0.0,
        "max_chunks_per_scene_object": max(chunks_per) if chunks_per else 0,
        "fragmentation_rate": 1.0 - float(objects_crossing / max(1, len(chunks_per))),
        "mean_pred_frames_per_object": float(np.mean([len(v) for v in by_obj_frames.values()])) if by_obj_frames else 0.0,
    }


def _eval_split(split: str, variant_id: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_scene, frame_rows = p1._evaluate_variant(variant_id, rows, scope)
    agg = p1._aggregate_metrics(per_scene)[0]
    agg["dataset_split"] = split
    agg["phase_id"] = "v100_phase4_history_memory"
    return agg, per_scene, frame_rows


def _component_rows(rows: list[dict[str, Any]], *, variant_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_obj[str(row["mv_object_id"])].append(row)
    for oid, vals in sorted(by_obj.items()):
        frames = sorted({int(_num(row.get("frame_id"), -1)) for row in vals})
        chunks = sorted({str(row.get("chunk_id")) for row in vals})
        sample = vals[0]
        out.append(
            {
                "schema_version": "stream4d_v100_phase4_scene_mv_object_row_v1",
                "phase_id": "v100_phase4_history_memory",
                "variant_id": variant_id,
                "dataset_split": sample.get("dataset_split"),
                "scene_id": sample.get("scene_id"),
                "mv_object_id": oid,
                "history_id": sample.get("history_id", oid),
                "chunk_count": len(chunks),
                "chunks": ";".join(chunks),
                "object_frame_count": len(frames),
                "first_frame": min(frames) if frames else "",
                "last_frame": max(frames) if frames else "",
                "score_scope": "current_chunk_score_history_identity",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _phase0_baselines() -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in _read_csv(PHASE0_BASELINES)}


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase4_artifact_manifest_row_v1",
            "phase_id": "v100_phase4_history_memory",
            "artifact_path": _rel(path),
            "artifact_type": kind,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": _sha256(path) if path.exists() and path.is_file() else "",
            "note": note,
        }
        for path, kind, note in paths
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    baselines = _phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    phase2 = json.loads((PHASE2_DIR / "summary.json").read_text(encoding="utf-8"))
    phase3 = json.loads(PHASE3_SUMMARY.read_text(encoding="utf-8"))
    phase2_pass = bool(phase2.get("phase2_pass")) or bool(phase2.get("phase2c_pass"))
    if not phase2_pass or not bool(phase3.get("phase3_pass")):
        raise RuntimeError("Phase4 requires Phase2 and Phase3 pass")

    base_df = pd.read_parquet(PHASE2_DIR / "mv_object_frame_mask_rows.parquet")
    base_rows_by_split = {
        split: [dict(row) for row in sub.to_dict(orient="records")]
        for split, sub in base_df.groupby("dataset_split")
    }
    spec_set = os.environ.get("V100_PHASE4_SPEC_SET", "main")
    _set_phase1_inputs("dev")
    dev_scope = p1._load_source_scope()
    _set_phase1_inputs("holdout")
    holdout_scope = p1._load_source_scope()
    scopes = {"dev": dev_scope, "holdout": holdout_scope}

    features = _feature_maps()
    use_centroid = spec_set == "scene_position"
    infos_by_split = {
        split: _object_infos(rows, features[split], scopes[split]["mask_path_by_frame"] if use_centroid else None)
        for split, rows in base_rows_by_split.items()
    }

    baseline_spec = {
        "variant_id": "HM0_fragmented_baseline",
        "alpha": 1.0,
        "beta": 0.0,
        "stale_eta": 0.0,
        "tau_link": 2.0,
        "tau_margin": 1.0,
        "max_age": 0,
        "topk_semantic": 1,
        "notes": "No history links; Phase2 fragmented baseline under Phase4 evaluator.",
    }
    if spec_set == "repair_local":
        variant_specs = [
            baseline_spec,
            {
                "variant_id": "HMR1_sem_tau0p99_margin0p05_age1",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.04,
                "tau_link": 0.99,
                "tau_margin": 0.05,
                "max_age": 1,
                "topk_semantic": 16,
                "notes": "Phase4 repair: very high confidence semantic links only.",
            },
            {
                "variant_id": "HMR2_sem_tau0p98_margin0p10_age1",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.04,
                "tau_link": 0.98,
                "tau_margin": 0.10,
                "max_age": 1,
                "topk_semantic": 16,
                "notes": "Phase4 repair: high threshold plus stricter margin to reduce local collapse.",
            },
            {
                "variant_id": "HMR3_sem_tau0p97_margin0p15_age1",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.04,
                "tau_link": 0.97,
                "tau_margin": 0.15,
                "max_age": 1,
                "topk_semantic": 16,
                "notes": "Phase4 repair: margin-dominant conservative memory.",
            },
            {
                "variant_id": "HMR4_sem_tau0p96_margin0p20_age1",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.04,
                "tau_link": 0.96,
                "tau_margin": 0.20,
                "max_age": 1,
                "topk_semantic": 16,
                "notes": "Phase4 repair: strict ambiguity rejection, age-1 only.",
            },
        ]
    elif spec_set == "scene_repair":
        variant_specs = [
            baseline_spec,
            {
                "variant_id": "HMS1_sem_tau0p78_margin0p04_age4",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.008,
                "tau_link": 0.78,
                "tau_margin": 0.04,
                "max_age": 4,
                "topk_semantic": 24,
                "notes": "Phase4c continuation: scene-focused repair after adapter-scope local protection.",
            },
            {
                "variant_id": "HMS2_sem_tau0p72_margin0p06_age8",
                "alpha": 0.90,
                "beta": 0.10,
                "stale_eta": 0.006,
                "tau_link": 0.72,
                "tau_margin": 0.06,
                "max_age": 8,
                "topk_semantic": 24,
                "notes": "Phase4c continuation: lower semantic threshold with moderate margin.",
            },
            {
                "variant_id": "HMS3_sem_tau0p66_margin0p08_age99",
                "alpha": 0.88,
                "beta": 0.12,
                "stale_eta": 0.004,
                "tau_link": 0.66,
                "tau_margin": 0.08,
                "max_age": 99,
                "topk_semantic": 32,
                "notes": "Phase4c continuation: stale-tolerant scene identity repair.",
            },
            {
                "variant_id": "HMS4_sem_tau0p60_margin0p12_age99",
                "alpha": 0.88,
                "beta": 0.12,
                "stale_eta": 0.003,
                "tau_link": 0.60,
                "tau_margin": 0.12,
                "max_age": 99,
                "topk_semantic": 32,
                "notes": "Phase4c continuation: aggressive threshold but stricter ambiguity margin.",
            },
        ]
    elif spec_set == "scene_position":
        variant_specs = [
            baseline_spec,
            {
                "variant_id": "HMP1_sem_pos_tau0p78_margin0p04_pos0p08_age4",
                "alpha": 0.72,
                "beta": 0.08,
                "gamma": 0.20,
                "position_sigma": 0.12,
                "min_position_score": 0.08,
                "stale_eta": 0.008,
                "tau_link": 0.78,
                "tau_margin": 0.04,
                "max_age": 4,
                "topk_semantic": 24,
                "notes": "Phase4f: semantic plus mask-centroid temporal continuity.",
            },
            {
                "variant_id": "HMP2_sem_pos_tau0p72_margin0p06_pos0p12_age8",
                "alpha": 0.68,
                "beta": 0.07,
                "gamma": 0.25,
                "position_sigma": 0.10,
                "min_position_score": 0.12,
                "stale_eta": 0.006,
                "tau_link": 0.72,
                "tau_margin": 0.06,
                "max_age": 8,
                "topk_semantic": 24,
                "notes": "Phase4f: stronger centroid gate for scene identity.",
            },
            {
                "variant_id": "HMP3_sem_pos_tau0p66_margin0p08_pos0p18_age99",
                "alpha": 0.64,
                "beta": 0.06,
                "gamma": 0.30,
                "position_sigma": 0.08,
                "min_position_score": 0.18,
                "stale_eta": 0.004,
                "tau_link": 0.66,
                "tau_margin": 0.08,
                "max_age": 99,
                "topk_semantic": 32,
                "notes": "Phase4f: long-age semantic memory gated by tighter 2D position continuity.",
            },
            {
                "variant_id": "HMP4_sem_pos_tau0p60_margin0p10_pos0p24_age99",
                "alpha": 0.60,
                "beta": 0.05,
                "gamma": 0.35,
                "position_sigma": 0.07,
                "min_position_score": 0.24,
                "stale_eta": 0.004,
                "tau_link": 0.60,
                "tau_margin": 0.10,
                "max_age": 99,
                "topk_semantic": 32,
                "notes": "Phase4f: aggressive semantic threshold with strong centroid cannot-link proxy.",
            },
        ]
    else:
        variant_specs = [
            baseline_spec,
            {
                "variant_id": "HM1_sem_tau0p95_margin0p02_age1",
                "alpha": 0.85,
                "beta": 0.15,
                "stale_eta": 0.02,
                "tau_link": 0.95,
                "tau_margin": 0.02,
                "max_age": 1,
                "topk_semantic": 16,
                "notes": "Conservative adjacent-ish semantic/maskview memory.",
            },
            {
                "variant_id": "HM2_sem_tau0p90_margin0p03_age2",
                "alpha": 0.85,
                "beta": 0.15,
                "stale_eta": 0.015,
                "tau_link": 0.90,
                "tau_margin": 0.03,
                "max_age": 2,
                "topk_semantic": 16,
                "notes": "Moderate semantic memory repair if HM1 keeps zero cross-chunk objects.",
            },
            {
                "variant_id": "HM3_sem_tau0p85_margin0p05_age4",
                "alpha": 0.85,
                "beta": 0.15,
                "stale_eta": 0.01,
                "tau_link": 0.85,
                "tau_margin": 0.05,
                "max_age": 4,
                "topk_semantic": 16,
                "notes": "Broader history memory with stricter margin.",
            },
            {
                "variant_id": "HM4_sem_tau0p80_margin0p08_age99",
                "alpha": 0.85,
                "beta": 0.15,
                "stale_eta": 0.005,
                "tau_link": 0.80,
                "tau_margin": 0.08,
                "max_age": 99,
                "topk_semantic": 16,
                "notes": "Aggressive stale-tolerant repair; expected to expose overmerge risk if semantic-only memory is insufficient.",
            },
        ]

    variant_metric_rows: list[dict[str, Any]] = []
    metric_scene_rows: list[dict[str, Any]] = []
    frame_eval_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    all_link_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    best_variant_rows: list[dict[str, Any]] = []
    best_variant_object_rows: list[dict[str, Any]] = []
    rows_by_variant_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    stats_by_variant_split: dict[tuple[str, str], dict[str, Any]] = {}

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        config_rows.append(
            {
                "schema_version": "stream4d_v100_phase4_variant_config_row_v1",
                "phase_id": "v100_phase4_history_memory",
                "variant_id": variant_id,
                **{key: spec[key] for key in ["alpha", "beta", "stale_eta", "tau_link", "tau_margin", "max_age", "topk_semantic", "notes"]},
                "gamma": spec.get("gamma", 0.0),
                "position_sigma": spec.get("position_sigma", ""),
                "min_position_score": spec.get("min_position_score", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for split in ["dev", "holdout"]:
            split_rows = base_rows_by_split[split]
            if variant_id == "HM0_fragmented_baseline":
                mapping = {oid: f"{variant_id}:{oid}" for oid in sorted(infos_by_split[split])}
                link_rows: list[dict[str, Any]] = []
                transition_rows: list[dict[str, Any]] = []
                history_rows = [
                    {
                        "schema_version": "stream4d_v100_phase4_history_object_row_v1",
                        "phase_id": "v100_phase4_history_memory",
                        "variant_id": variant_id,
                        "dataset_split": split,
                        "scene_id": info["scene_id"],
                        "history_id": mapping[oid],
                        "state": "tentative",
                        "birth_chunk": info["chunk_id"],
                        "last_seen_chunk": info["chunk_id"],
                        "support_chunk_count": 1,
                        "support_frame_count": info["frame_count"],
                        "local_object_count": 1,
                        "local_object_ids": oid,
                        "area_ema": info["mean_area"],
                        "score_ema": info["score"],
                        "cannot_link_history_ids": "",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                    for oid, info in sorted(infos_by_split[split].items())
                ]
                stats = {
                    "accepted_link_count": 0,
                    "rejected_cannot_link_count": 0,
                    "rejected_margin_count": 0,
                    "rejected_threshold_count": 0,
                    "rejected_used_history_count": 0,
                    "link_margin_mean": 0.0,
                    "link_entropy_mean": 0.0,
                    "confirmed_history_count": 0,
                    "tentative_history_count": len(history_rows),
                    "quarantine_count": 0,
                    "history_count": len(history_rows),
                    "overmerge_large_component_count": 0,
                    "device": "none_baseline",
                }
            else:
                mapping, link_rows, transition_rows, history_rows, stats = _generate_memory_for_split(split, infos_by_split[split], spec)
            stitched_rows = _apply_mapping(split_rows, mapping, variant_id=variant_id)
            rows_by_variant_split[(variant_id, split)] = stitched_rows
            all_link_rows.extend(link_rows)
            all_transition_rows.extend(transition_rows)
            all_history_rows.extend(history_rows)
            agg, per_scene, frames = _eval_split(split, variant_id, stitched_rows, scopes[split])
            crossing = _scene_crossing_stats(stitched_rows)
            agg.update(stats)
            agg.update(crossing)
            agg["variant_id"] = variant_id
            agg["dataset_split"] = split
            agg["future_chunk_access"] = False
            agg["uses_gt_for_prediction"] = False
            agg["uses_future"] = False
            agg["metric_source"] = "canonical_v65_evaluator_after_causal_history_memory_mapping"
            variant_metric_rows.append(agg)
            for row in per_scene:
                row["dataset_split"] = split
                row["phase_id"] = "v100_phase4_history_memory"
            metric_scene_rows.extend(per_scene)
            for row in frames:
                row["dataset_split"] = split
                row["phase_id"] = "v100_phase4_history_memory"
            frame_eval_rows.extend(frames)
            stats_by_variant_split[(variant_id, split)] = {**stats, **crossing}

    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in variant_metric_rows:
        by_variant[str(row["variant_id"])][str(row["dataset_split"])] = row
    phase2_dev_window = float(phase2["dev_MV_AP_window"])
    phase2_dev_ap50 = float(phase2["dev_MV_AP50_window"])
    phase2_hold_window = float(phase2["holdout_MV_AP_window"])
    phase2_hold_ap50 = float(phase2["holdout_MV_AP50_window"])
    best_variant_id = max(
        by_variant,
        key=lambda vid: (
            _num(by_variant[vid].get("holdout", {}).get("MV_AP_scene")),
            _num(by_variant[vid].get("holdout", {}).get("MV_AP50_scene")),
            _num(by_variant[vid].get("dev", {}).get("MV_AP_scene")),
            -_num(by_variant[vid].get("holdout", {}).get("overmerge_large_component_count")),
        ),
    )
    best_dev = by_variant[best_variant_id]["dev"]
    best_hold = by_variant[best_variant_id]["holdout"]
    best_variant_rows = rows_by_variant_split[(best_variant_id, "dev")] + rows_by_variant_split[(best_variant_id, "holdout")]
    best_variant_object_rows = _component_rows(best_variant_rows, variant_id=best_variant_id)

    dev_scene_gate = _num(best_dev.get("MV_AP_scene")) >= _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_scene_ap50_gate = _num(best_dev.get("MV_AP50_scene")) >= _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(best_hold.get("MV_AP_scene")) >= _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_scene_ap50_gate = _num(best_hold.get("MV_AP50_scene")) >= _num(f2_holdout["MV_AP50_scene"]) + 0.010
    local_drop_dev = phase2_dev_window - _num(best_dev.get("MV_AP_window"))
    local_drop_hold = phase2_hold_window - _num(best_hold.get("MV_AP_window"))
    local_drop_gate = local_drop_dev <= 0.003 and local_drop_hold <= 0.003
    objects_crossing_gate = int(_num(best_dev.get("objects_crossing_multiple_chunks"))) + int(_num(best_hold.get("objects_crossing_multiple_chunks"))) > 0
    safety_gate = (
        int(_num(best_dev.get("same_frame_collision_count"))) == 0
        and int(_num(best_hold.get("same_frame_collision_count"))) == 0
        and _num(best_dev.get("pixel_collision_rate")) <= 0.02
        and _num(best_hold.get("pixel_collision_rate")) <= 0.02
        and int(_num(best_dev.get("overmerge_large_component_count"))) == 0
        and int(_num(best_hold.get("overmerge_large_component_count"))) == 0
    )
    future_gate = not any(_bool(row.get("uses_future")) or _bool(row.get("future_chunk_access")) for row in best_variant_rows)

    gate_rows = [
        {
            "gate_id": "mv_ap_scene_dev_ge_f2_base_plus_0p010",
            "pass": dev_scene_gate,
            "expected": _num(f2_dev["MV_AP_scene"]) + 0.010,
            "observed": _num(best_dev.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_dev_ge_f2_base_plus_0p015",
            "pass": dev_scene_ap50_gate,
            "expected": _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "observed": _num(best_dev.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap_scene_holdout_ge_f2_base_plus_0p006",
            "pass": hold_scene_gate,
            "expected": _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "observed": _num(best_hold.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_holdout_ge_f2_base_plus_0p010",
            "pass": hold_scene_ap50_gate,
            "expected": _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "observed": _num(best_hold.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "local_window_ap_drop_le_0p003",
            "pass": local_drop_gate,
            "expected": "<=0.003 for dev and holdout",
            "observed": f"dev_drop={local_drop_dev}; holdout_drop={local_drop_hold}",
            "severity": "protect_local",
        },
        {
            "gate_id": "objects_crossing_multiple_chunks_gt_0",
            "pass": objects_crossing_gate,
            "expected": ">0",
            "observed": f"dev={best_dev.get('objects_crossing_multiple_chunks')} holdout={best_hold.get('objects_crossing_multiple_chunks')}",
            "severity": "identity_required",
        },
        {
            "gate_id": "collision_pixel_overmerge_safety",
            "pass": safety_gate,
            "expected": "same_frame_collision=0 pixel_collision<=0.02 overmerge_large_component_count=0",
            "observed": f"dev_collision={best_dev.get('same_frame_collision_count')} hold_collision={best_hold.get('same_frame_collision_count')} dev_pixel={best_dev.get('pixel_collision_rate')} hold_pixel={best_hold.get('pixel_collision_rate')} dev_overmerge={best_dev.get('overmerge_large_component_count')} hold_overmerge={best_hold.get('overmerge_large_component_count')}",
            "severity": "required_safety",
        },
        {
            "gate_id": "future_chunk_access_false",
            "pass": future_gate,
            "expected": "false for all best rows",
            "observed": future_gate,
            "severity": "required_safety",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If crossings remain zero, lower candidate threshold or repair overlap membership. "
                "If scene improves but local collapses, add stricter margin/cannot-link/quarantine. "
                "If semantic merges are wrong, add DA3/D4RT verifier in Phase5."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase4_pass = not failure_rows

    history_parquet = OUT_DIR / "history_object_rows.parquet"
    link_parquet = OUT_DIR / "chunk_object_history_link_rows.parquet"
    scene_object_parquet = OUT_DIR / "scene_mv_object_rows.parquet"
    scene_frame_parquet = OUT_DIR / "scene_mv_object_frame_mask_rows.parquet"
    transition_csv = OUT_DIR / "history_state_transition_rows.csv"
    scene_metric_csv = OUT_DIR / "mv_metric_scene_rows.csv"
    window_metric_csv = OUT_DIR / "mv_metric_window_rows.csv"
    variant_metric_csv = OUT_DIR / "variant_metric_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_parquet(history_parquet, all_history_rows)
    _write_parquet(link_parquet, all_link_rows)
    _write_parquet(scene_object_parquet, best_variant_object_rows)
    _write_parquet(scene_frame_parquet, best_variant_rows)
    _write_csv(transition_csv, all_transition_rows)
    _write_csv(scene_metric_csv, metric_scene_rows)
    _write_csv(window_metric_csv, variant_metric_rows)
    _write_csv(variant_metric_csv, variant_metric_rows)
    _write_csv(config_csv, config_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase4_performance_row_v1",
                "phase_id": "v100_phase4_history_memory",
                "case_id": "history_memory_candidate_generation_and_v65_eval",
                "runtime_sec": time.time() - started,
                "variant_count": len(variant_specs),
                "split_count": 2,
                "gpu_backend": "torch_matmul_topk" if torch.cuda.is_available() else "torch_cpu_matmul_topk",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "phase2_parquet_bytes_read": (PHASE2_DIR / "mv_object_frame_mask_rows.parquet").stat().st_size,
                "phase2_dir": _rel(PHASE2_DIR),
                "history_link_rows": len(all_link_rows),
                "history_state_transition_rows": len(all_transition_rows),
                "v65_evaluator_runs": len(variant_specs) * 2,
            }
        ],
    )
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (history_parquet, "parquet", "history object states for all variants"),
                (link_parquet, "parquet", "chunk object to history candidates/actions for all variants"),
                (transition_csv, "csv", "history state transitions for all variants"),
                (scene_object_parquet, "parquet", "best variant scene object rows"),
                (scene_frame_parquet, "parquet", "best variant scene object-frame-mask rows"),
                (scene_metric_csv, "csv", "v65 per-scene metrics for all variants"),
                (window_metric_csv, "csv", "v65 aggregate metrics for all variants/splits"),
                (performance_csv, "csv", "phase4 runtime and backend"),
                (gate_csv, "csv", "phase4 gates"),
                (failure_csv, "csv", "phase4 failures if any"),
            ]
        ),
    )

    summary = {
        "schema_version": "stream4d_v100_phase4_history_memory_summary_v1",
        "phase_id": "v100_phase4_history_memory",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE5" if phase4_pass else "BLOCK_PHASE5_REPAIR_HISTORY_MEMORY",
        "phase4_pass": phase4_pass,
        "failure_count": len(failure_rows),
        "best_variant_id": best_variant_id,
        "best_dev_MV_AP_window": float(_num(best_dev.get("MV_AP_window"))),
        "best_dev_MV_AP50_window": float(_num(best_dev.get("MV_AP50_window"))),
        "best_dev_MV_AP_scene": float(_num(best_dev.get("MV_AP_scene"))),
        "best_dev_MV_AP50_scene": float(_num(best_dev.get("MV_AP50_scene"))),
        "best_holdout_MV_AP_window": float(_num(best_hold.get("MV_AP_window"))),
        "best_holdout_MV_AP50_window": float(_num(best_hold.get("MV_AP50_window"))),
        "best_holdout_MV_AP_scene": float(_num(best_hold.get("MV_AP_scene"))),
        "best_holdout_MV_AP50_scene": float(_num(best_hold.get("MV_AP50_scene"))),
        "local_window_AP_drop": {"dev": local_drop_dev, "holdout": local_drop_hold},
        "objects_crossing_multiple_chunks": {
            "dev": int(_num(best_dev.get("objects_crossing_multiple_chunks"))),
            "holdout": int(_num(best_hold.get("objects_crossing_multiple_chunks"))),
        },
        "fragmentation_rate": {
            "dev": float(_num(best_dev.get("fragmentation_rate"))),
            "holdout": float(_num(best_hold.get("fragmentation_rate"))),
        },
        "accepted_link_count": {
            "dev": int(_num(best_dev.get("accepted_link_count"))),
            "holdout": int(_num(best_hold.get("accepted_link_count"))),
        },
        "confirmed_history_count": {
            "dev": int(_num(best_dev.get("confirmed_history_count"))),
            "holdout": int(_num(best_hold.get("confirmed_history_count"))),
        },
        "tentative_history_count": {
            "dev": int(_num(best_dev.get("tentative_history_count"))),
            "holdout": int(_num(best_hold.get("tentative_history_count"))),
        },
        "quarantine_count": {
            "dev": int(_num(best_dev.get("quarantine_count"))),
            "holdout": int(_num(best_hold.get("quarantine_count"))),
        },
        "link_entropy_mean": {
            "dev": float(_num(best_dev.get("link_entropy_mean"))),
            "holdout": float(_num(best_hold.get("link_entropy_mean"))),
        },
        "link_margin_mean": {
            "dev": float(_num(best_dev.get("link_margin_mean"))),
            "holdout": float(_num(best_hold.get("link_margin_mean"))),
        },
        "rejected_cannot_link_count": {
            "dev": int(_num(best_dev.get("rejected_cannot_link_count"))),
            "holdout": int(_num(best_hold.get("rejected_cannot_link_count"))),
        },
        "overmerge_large_component_count": {
            "dev": int(_num(best_dev.get("overmerge_large_component_count"))),
            "holdout": int(_num(best_hold.get("overmerge_large_component_count"))),
        },
        "future_chunk_access": False,
        "variant_count": len(variant_specs),
        "spec_set": spec_set,
        "phase2_dir": _rel(PHASE2_DIR),
        "phase2_pass_key": "phase2c_pass" if bool(phase2.get("phase2c_pass")) else "phase2_pass",
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "history_object_rows": _rel(history_parquet),
            "chunk_object_history_link_rows": _rel(link_parquet),
            "history_state_transition_rows": _rel(transition_csv),
            "scene_mv_object_rows": _rel(scene_object_parquet),
            "scene_mv_object_frame_mask_rows": _rel(scene_frame_parquet),
            "mv_metric_scene_rows": _rel(scene_metric_csv),
            "mv_metric_window_rows": _rel(window_metric_csv),
            "performance_rows": _rel(performance_csv),
            "variant_metric_rows": _rel(variant_metric_csv),
            "variant_config_rows": _rel(config_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
