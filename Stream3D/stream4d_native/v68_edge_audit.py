from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v67_local_mask_graph import _mask_summary_by_pair  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _float_or_none, _load_csv_rows, _mean, _rel  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_json(value: Any, fallback: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _candidate_node(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["scene_id"]), int(float(row["frame_id"])), int(float(row["mask_id"])))


def _prep_candidate_row(
    row: dict[str, Any],
    row_idx: int,
    appearance_by_node: dict[tuple[str, int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    comps = {str(item) for item in _parse_json(row.get("d4rt_component_ids"), []) if str(item)}
    center = [float(v) for v in _parse_json(row.get("bbox_center"), [0.0, 0.0])[:2]]
    size = [float(v) for v in _parse_json(row.get("bbox_size"), [0.0, 0.0])[:2]]
    competitors = [int(v) for v in _parse_json(row.get("same_frame_competing_masks"), [])]
    appearance = (appearance_by_node or {}).get(_candidate_node(row), {})
    out = dict(row)
    out.update(
        {
            "_row_idx": int(row_idx),
            "_node": _candidate_node(row),
            "_frame_id": int(float(row["frame_id"])),
            "_mask_id": int(float(row["mask_id"])),
            "_area_ratio": _safe_float(row.get("area_ratio")),
            "_bbox_area_ratio": _safe_float(row.get("bbox_area_ratio")),
            "_aspect_ratio": _safe_float(row.get("aspect_ratio"), 1.0),
            "_solidity": _safe_float(row.get("mask_solidity_proxy"), 0.0),
            "_component_entropy": _safe_float(row.get("d4rt_component_entropy")),
            "_component_ids": comps,
            "_bbox_center": center if len(center) == 2 else [0.0, 0.0],
            "_bbox_size": size if len(size) == 2 else [0.0, 0.0],
            "_same_frame_competing_masks": competitors,
            "_representative": _parse_bool(row.get("representative_available")),
            "_shared_support": _parse_bool(row.get("shared_support_only")),
            "_underseg": _parse_bool(row.get("underseg_risk")),
            "_large_mask": _parse_bool(row.get("large_mask_risk")),
            "_small_mask": _parse_bool(row.get("small_mask_risk")),
            "_signature": str(row.get("repeated_signature_id") or ""),
            "_semantic_mode": str(row.get("semantic_mode_id") or ""),
            "_appearance_feature": appearance.get("feature", []),
            "_appearance_valid": bool(appearance.get("valid", False)),
            "_appearance_mode": str(appearance.get("mode_id") or ""),
            "_appearance_used_pixels": int(appearance.get("used_pixels") or 0),
        }
    )
    return out


def _read_mask_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _hist_feature(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> list[float]:
    if values.size == 0:
        return [0.0] * bins
    hist, _edges = np.histogram(values.astype(np.float32), bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    denom = float(hist.sum())
    if denom > 0.0:
        hist /= denom
    return [float(v) for v in hist.tolist()]


def _appearance_feature(rgb: np.ndarray, hsv: np.ndarray, lab: np.ndarray, binary: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(binary)
    if ys.size < 16:
        return {"valid": False, "feature": [], "mode_id": "", "used_pixels": int(ys.size)}
    if ys.size > 4096:
        order = np.linspace(0, ys.size - 1, 4096).astype(np.int64)
        ys = ys[order]
        xs = xs[order]
    rgb_pixels = rgb[ys, xs].astype(np.float32) / 255.0
    hsv_pixels = hsv[ys, xs].astype(np.float32)
    lab_pixels = lab[ys, xs].astype(np.float32) / 255.0
    rgb_mean = rgb_pixels.mean(axis=0)
    rgb_std = rgb_pixels.std(axis=0)
    lab_mean = lab_pixels.mean(axis=0)
    lab_std = lab_pixels.std(axis=0)
    hue_hist = _hist_feature(hsv_pixels[:, 0], 8, (0.0, 180.0))
    sat_hist = _hist_feature(hsv_pixels[:, 1], 4, (0.0, 256.0))
    val_hist = _hist_feature(hsv_pixels[:, 2], 4, (0.0, 256.0))
    feature = [float(v) for v in rgb_mean.tolist()]
    feature += [float(v) for v in rgb_std.tolist()]
    feature += [float(v) for v in lab_mean.tolist()]
    feature += [float(v) for v in lab_std.tolist()]
    feature += hue_hist + sat_hist + val_hist
    arr = np.asarray(feature, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    hue_bin = int(np.argmax(np.asarray(hue_hist))) if hue_hist else 0
    sat_bin = int(np.argmax(np.asarray(sat_hist))) if sat_hist else 0
    val_bin = int(np.argmax(np.asarray(val_hist))) if val_hist else 0
    rgb_bins = [int(max(0, min(3, math.floor(float(v) * 4.0)))) for v in rgb_mean.tolist()]
    mode_id = f"h{hue_bin}|s{sat_bin}|v{val_bin}|rgb{rgb_bins[0]}{rgb_bins[1]}{rgb_bins[2]}"
    return {"valid": True, "feature": [float(v) for v in arr.tolist()], "mode_id": mode_id, "used_pixels": int(ys.size)}


def _compute_appearance_by_node(
    raw_rows: list[dict[str, Any]],
    *,
    scenes: list[str],
    pipeline_roots_abs: dict[str, Path],
    backend: str,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    if backend in {"", "none", "disabled"}:
        return {}, {"backend": backend or "disabled", "enabled": False}
    if backend not in {"rgb_hist", "dinov2_timm"}:
        raise ValueError(f"unsupported appearance backend: {backend}")
    feature_adapter: FrozenFeatureAdapter | None = None
    feature_checkpoint: str | None = None
    if backend == "dinov2_timm":
        feature_checkpoint = checkpoint or locate_default_dinov2_checkpoint()
        feature_adapter = FrozenFeatureAdapter(
            backend="dinov2_timm",
            device=device,
            checkpoint=feature_checkpoint,
        )
    wanted: dict[tuple[str, int], set[int]] = defaultdict(set)
    scene_set = set(scenes)
    for row in raw_rows:
        scene = str(row.get("scene_id"))
        if scene not in scene_set:
            continue
        if not _parse_bool(row.get("representative_available")):
            continue
        wanted[(scene, int(float(row.get("frame_id") or 0)))].add(int(float(row.get("mask_id") or 0)))
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing_frames: list[dict[str, Any]] = []
    for (scene, frame_id), mask_ids in sorted(wanted.items()):
        pipeline_root = pipeline_roots_abs.get(scene)
        if pipeline_root is None:
            missing_frames.append({"scene": scene, "frame_id": int(frame_id), "missing": "pipeline_root"})
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        try:
            rgb = stream.load_rgb(int(frame_id))
        except FileNotFoundError:
            missing_frames.append({"scene": scene, "frame_id": int(frame_id), "missing": "rgb"})
            continue
        mask = _read_mask_label(mask_dir / f"{int(frame_id)}.png", rgb.shape[:2])
        if mask is None:
            missing_frames.append({"scene": scene, "frame_id": int(frame_id), "missing": "mask"})
            continue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        feature_map = None
        if feature_adapter is not None:
            try:
                feature_map = feature_adapter.extract_dense_features(rgb)
            except Exception as exc:
                missing_frames.append({"scene": scene, "frame_id": int(frame_id), "missing": f"dinov2_extract_failed:{type(exc).__name__}:{exc}"})
                continue
        for mask_id in sorted(mask_ids):
            binary = mask == int(mask_id)
            if feature_adapter is None or feature_map is None:
                out[(scene, int(frame_id), int(mask_id))] = _appearance_feature(
                    rgb,
                    hsv,
                    lab,
                    binary,
                )
            else:
                used_pixels = int(binary.sum())
                if used_pixels < 16:
                    out[(scene, int(frame_id), int(mask_id))] = {"valid": False, "feature": [], "mode_id": "", "used_pixels": used_pixels}
                    continue
                pooled = np.asarray(feature_adapter.pool_mask_feature(feature_map, binary), dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(pooled))
                if pooled.size == 0 or norm <= 1e-8:
                    out[(scene, int(frame_id), int(mask_id))] = {"valid": False, "feature": [], "mode_id": "", "used_pixels": used_pixels}
                    continue
                pooled = pooled / norm
                top = np.argsort(np.abs(pooled))[-3:]
                signs = ["p" if pooled[int(idx)] >= 0 else "n" for idx in top]
                mode_id = "dino|" + "|".join(f"{sign}{int(idx)}" for sign, idx in zip(signs, top.tolist()))
                out[(scene, int(frame_id), int(mask_id))] = {
                    "valid": True,
                    "feature": [float(v) for v in pooled.tolist()],
                    "mode_id": mode_id,
                    "used_pixels": used_pixels,
                }
    valid_count = int(sum(1 for item in out.values() if item.get("valid")))
    summary = {
        "backend": backend,
        "enabled": True,
        "device": device if backend == "dinov2_timm" else "",
        "checkpoint": feature_checkpoint or "",
        "requested_unique_mask_observation_count": int(sum(len(v) for v in wanted.values())),
        "feature_row_count": int(len(out)),
        "valid_feature_count": valid_count,
        "feature_success_rate": float(valid_count / max(1, len(out))),
        "missing_frame_count": int(len(missing_frames)),
        "missing_frame_examples": missing_frames[:20],
        "uses_gt_for_prediction": False,
        "uses_rgb_for_prediction": True,
        "uses_frozen_dense_features": bool(backend == "dinov2_timm"),
        "uses_depth_pose_mesh_for_prediction": False,
    }
    return out, summary


def _diag_label(
    a: dict[str, Any],
    b: dict[str, Any],
    diag_by_node: dict[tuple[str, int, int], dict[str, Any]],
    *,
    purity_threshold: float,
) -> dict[str, Any]:
    da = diag_by_node.get(a["_node"], {})
    db = diag_by_node.get(b["_node"], {})
    gi = str(da.get("diagnostic_gt_instance") or "").strip()
    gj = str(db.get("diagnostic_gt_instance") or "").strip()
    pi = _safe_float(da.get("diagnostic_gt_purity"), 0.0)
    pj = _safe_float(db.get("diagnostic_gt_purity"), 0.0)
    label: int | None
    if gi and gj and gi != "0" and gj != "0" and pi >= purity_threshold and pj >= purity_threshold:
        label = 1 if gi == gj else 0
    else:
        label = None
    return {
        "same_object_label": "" if label is None else int(label),
        "diagnostic_gt_i": gi,
        "diagnostic_gt_j": gj,
        "diagnostic_gt_purity_i": pi,
        "diagnostic_gt_purity_j": pj,
        "diagnostic_gt_purity_threshold": float(purity_threshold),
        "diagnostic_label_known": label is not None,
    }


def _area_similarity(a: float, b: float) -> float:
    return float(min(a, b) / max(max(a, b), 1e-9))


def _center_similarity(a: list[float], b: list[float]) -> float:
    dist = math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)
    return float(max(0.0, 1.0 - dist / 0.75))


def _size_similarity(a: list[float], b: list[float]) -> float:
    aw, ah = max(float(a[0]), 1e-9), max(float(a[1]), 1e-9)
    bw, bh = max(float(b[0]), 1e-9), max(float(b[1]), 1e-9)
    return float(math.sqrt(min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)))


def _material_overlap(a: set[str], b: set[str]) -> tuple[int, float, float]:
    if not a or not b:
        return 0, 0.0, 0.0
    inter = len(a & b)
    union = len(a | b)
    cosine = float(inter / max(1.0, math.sqrt(len(a) * len(b))))
    jaccard = float(inter / max(1, union))
    return int(inter), cosine, jaccard


def _cosine_feature(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(float(np.dot(aa, bb) / denom), -1.0, 1.0) * 0.5 + 0.5)


def _score_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    same_frame = int(a["_frame_id"]) == int(b["_frame_id"])
    frame_delta = abs(int(a["_frame_id"]) - int(b["_frame_id"]))
    inter, mat_cosine, mat_jaccard = _material_overlap(a["_component_ids"], b["_component_ids"])
    entropy_penalty = 1.0 / (1.0 + 0.35 * (a["_component_entropy"] + b["_component_entropy"]))
    solidity = math.sqrt(max(0.0, min(1.0, a["_solidity"])) * max(0.0, min(1.0, b["_solidity"])))
    material_residual = float(mat_cosine * entropy_penalty * (0.35 + 0.65 * solidity))
    center_sim = _center_similarity(a["_bbox_center"], b["_bbox_center"])
    size_sim = _size_similarity(a["_bbox_size"], b["_bbox_size"])
    area_sim = _area_similarity(a["_area_ratio"], b["_area_ratio"])
    shape_score = float(0.35 * area_sim + 0.35 * size_sim + 0.30 * center_sim)
    temporal_decay = math.exp(-frame_delta / 25.0) if frame_delta > 0 else 0.0
    temporal_score = float(temporal_decay * shape_score * (0.65 + 0.35 * mat_cosine))
    signature_match = bool(a["_signature"] and a["_signature"] == b["_signature"] and not same_frame)
    signature_score = float((1.0 if signature_match else 0.0) * (0.55 + 0.45 * temporal_score))
    semantic_match = bool(a["_semantic_mode"] and a["_semantic_mode"] == b["_semantic_mode"] and not same_frame)
    appearance_mode_match = bool(a["_appearance_mode"] and a["_appearance_mode"] == b["_appearance_mode"] and not same_frame)
    appearance_abs = _cosine_feature(a["_appearance_feature"], b["_appearance_feature"]) if a["_appearance_valid"] and b["_appearance_valid"] and not same_frame else 0.0
    semantic_score = float(1.0 if semantic_match else 0.0)
    no_temporal = float(material_residual)
    combined = float(0.50 * material_residual + 0.25 * signature_score + 0.25 * temporal_score + 0.10 * semantic_score)
    if a["_underseg"] or b["_underseg"] or a["_shared_support"] or b["_shared_support"]:
        if mat_cosine < 0.20:
            combined *= 0.55
            no_temporal *= 0.65
        else:
            combined *= 0.85
            no_temporal *= 0.90
    if same_frame:
        combined = 0.0
        no_temporal = 0.0
        temporal_score = 0.0
        signature_score = 0.0
    hard_negative = bool((not same_frame) and shape_score < 0.18 and mat_cosine == 0.0)
    return {
        "same_frame": same_frame,
        "frame_delta": int(frame_delta),
        "component_intersection_count": int(inter),
        "component_cosine": mat_cosine,
        "component_jaccard": mat_jaccard,
        "score_material_overlap": mat_cosine,
        "score_material_residual": material_residual,
        "score_signature": signature_score,
        "score_temporal_adjacent": temporal_score,
        "score_semantic_mode": semantic_score,
        "score_appearance_abs": appearance_abs,
        "appearance_mode_match": appearance_mode_match,
        "score_shape_area_bbox": shape_score,
        "score_combined_no_semantic": combined,
        "score_combined_no_temporal": no_temporal,
        "hard_negative_candidate": hard_negative,
        "signature_match": signature_match,
        "semantic_match": semantic_match,
        "appearance_valid_pair": bool(a["_appearance_valid"] and b["_appearance_valid"]),
        "area_similarity": area_sim,
        "bbox_center_similarity": center_sim,
        "bbox_size_similarity": size_sim,
    }


def _pair_key(a_idx: int, b_idx: int) -> tuple[int, int]:
    return (int(a_idx), int(b_idx)) if int(a_idx) < int(b_idx) else (int(b_idx), int(a_idx))


def _add_pair(
    pairs: dict[tuple[int, int], set[str]],
    rows: list[dict[str, Any]],
    i: int,
    j: int,
    reason: str,
) -> None:
    if i == j:
        return
    key = _pair_key(i, j)
    pairs[key].add(reason)


def _sample_chunk_pairs(
    chunk_rows: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_pairs: int,
    max_component_bucket: int,
    max_signature_bucket: int,
) -> dict[tuple[int, int], set[str]]:
    pairs: dict[tuple[int, int], set[str]] = defaultdict(set)
    by_component: dict[str, list[int]] = defaultdict(list)
    by_signature: dict[str, list[int]] = defaultdict(list)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for local_idx, row in enumerate(chunk_rows):
        if row["_representative"]:
            by_signature[row["_signature"]].append(local_idx)
            by_frame[int(row["_frame_id"])].append(local_idx)
            for comp in row["_component_ids"]:
                by_component[comp].append(local_idx)

    for bucket in by_component.values():
        if len(bucket) < 2:
            continue
        items = list(bucket)
        rng.shuffle(items)
        items = items[:max_component_bucket]
        for pos, i in enumerate(items):
            for j in items[pos + 1 :]:
                if chunk_rows[i]["_frame_id"] != chunk_rows[j]["_frame_id"]:
                    _add_pair(pairs, chunk_rows, i, j, "component_overlap")

    for signature, bucket in by_signature.items():
        if not signature or len(bucket) < 2:
            continue
        items = list(bucket)
        rng.shuffle(items)
        items = items[:max_signature_bucket]
        for pos, i in enumerate(items):
            for j in items[pos + 1 :]:
                if chunk_rows[i]["_frame_id"] != chunk_rows[j]["_frame_id"]:
                    _add_pair(pairs, chunk_rows, i, j, "repeated_signature")

    frames = sorted(by_frame)
    for frame in frames:
        for next_frame in [f for f in frames if 0 < f - frame <= 15][:3]:
            left = list(by_frame[frame])
            right = list(by_frame[next_frame])
            rng.shuffle(left)
            rng.shuffle(right)
            for i in left[:32]:
                scored: list[tuple[float, int]] = []
                for j in right[:96]:
                    score = _score_pair(chunk_rows[i], chunk_rows[j])["score_temporal_adjacent"]
                    scored.append((float(score), j))
                for _score, j in sorted(scored, reverse=True)[:8]:
                    _add_pair(pairs, chunk_rows, i, j, "temporal_neighbor")

    for frame, items in by_frame.items():
        if len(items) < 2:
            continue
        local_by_mask = {chunk_rows[i]["_mask_id"]: i for i in items}
        for i in items:
            for other_mask in chunk_rows[i]["_same_frame_competing_masks"][:8]:
                j = local_by_mask.get(int(other_mask))
                if j is not None:
                    _add_pair(pairs, chunk_rows, i, j, "same_frame_cannot_link")
        sampled = list(items)
        rng.shuffle(sampled)
        for pos, i in enumerate(sampled[:48]):
            for j in sampled[pos + 1 : pos + 9]:
                _add_pair(pairs, chunk_rows, i, j, "same_frame_random_negative")

    reps = [idx for idx, row in enumerate(chunk_rows) if row["_representative"]]
    if len(reps) >= 2:
        for _ in range(min(max_pairs, max(250, len(reps) * 3))):
            i, j = rng.sample(reps, 2)
            if chunk_rows[i]["_frame_id"] == chunk_rows[j]["_frame_id"]:
                continue
            _add_pair(pairs, chunk_rows, i, j, "deterministic_random_cross_frame")

    if len(pairs) > max_pairs:
        keys = sorted(pairs)
        rng.shuffle(keys)
        keep = set(keys[:max_pairs])
        pairs = {key: pairs[key] for key in keys if key in keep}
    return pairs


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [(score, idx) for idx, (score, label) in enumerate(zip(scores, labels)) if label == 1]
    neg_count = sum(1 for label in labels if label == 0)
    pos_count = len(pos)
    if pos_count == 0 or neg_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        avg_rank = (cursor + 1 + end) * 0.5
        for pos_idx in range(cursor, end):
            ranks[order[pos_idx]] = avg_rank
        cursor = end
    rank_sum_pos = sum(ranks[idx] for idx, label in enumerate(labels) if label == 1)
    return float((rank_sum_pos - pos_count * (pos_count + 1) * 0.5) / max(1, pos_count * neg_count))


def _metric_for_score(rows: list[dict[str, Any]], score_key: str, *, edge_type: str) -> dict[str, Any]:
    known = [row for row in rows if row.get("diagnostic_label_known")]
    scores = [float(row.get(score_key) or 0.0) for row in known]
    labels = [int(row["same_object_label"]) for row in known]
    auc = _auc(scores, labels)
    by_node: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in known:
        score = float(row.get(score_key) or 0.0)
        label = int(row["same_object_label"])
        node_i = str(row["node_i"])
        node_j = str(row["node_j"])
        by_node[node_i].append((score, label))
        by_node[node_j].append((score, label))
    top1_labels: list[int] = []
    top1_query_count = 0
    top1_query_with_positive_count = 0
    top3_hit = 0
    top3_den = 0
    for candidates in by_node.values():
        ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
        if ordered and any(label == 1 for _score, label in ordered):
            top1_query_with_positive_count += 1
            top1_labels.append(int(ordered[0][1]))
        if ordered:
            top1_query_count += 1
        if any(label == 1 for _score, label in ordered):
            top3_den += 1
            if any(label == 1 for _score, label in ordered[:3]):
                top3_hit += 1
    top1_precision = float(np.mean(top1_labels)) if top1_labels else None
    top3_recall = float(top3_hit / top3_den) if top3_den else None
    top_global = sorted(((float(row.get(score_key) or 0.0), int(row["same_object_label"])) for row in known), reverse=True)[:5000]
    top5k_precision = float(np.mean([label for _score, label in top_global])) if top_global else None
    positives = [row for row in known if int(row["same_object_label"]) == 1]
    false_negative_rate = (
        float(np.mean([float(row.get(score_key) or 0.0) < 0.50 for row in positives])) if positives else None
    )
    same_frame_edges = [row for row in known if _parse_bool(row.get("same_frame"))]
    same_frame_violation_rate = (
        float(np.mean([float(row.get(score_key) or 0.0) >= 0.50 for row in same_frame_edges])) if same_frame_edges else 0.0
    )
    return {
        "edge_type": edge_type,
        "score_key": score_key,
        "edge_count": int(len(rows)),
        "diagnostic_known_edge_count": int(len(known)),
        "positive_edge_count": int(sum(labels)),
        "negative_edge_count": int(len(labels) - sum(labels)),
        "edge_AUC": auc,
        "top1_precision": top1_precision,
        "top1_query_count": int(top1_query_count),
        "top1_query_with_positive_count": int(top1_query_with_positive_count),
        "top3_recall": top3_recall,
        "top5k_precision": top5k_precision,
        "same_frame_violation_rate": same_frame_violation_rate,
        "false_negative_rate": false_negative_rate,
    }


def _add_relative_appearance_scores(rows: list[dict[str, Any]], *, enabled: bool) -> None:
    if not rows:
        return
    if not enabled:
        for row in rows:
            row["score_appearance_relative"] = 0.0
            row["score_combined_appearance_relative"] = row.get("score_combined_no_semantic", 0.0)
            row["score_combined_frozen_appearance"] = row.get("score_combined_no_semantic", 0.0)
        return
    by_chunk: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if _parse_bool(row.get("appearance_valid_pair")) and not _parse_bool(row.get("same_frame")):
            by_chunk[str(row.get("chunk_id"))].append(float(row.get("score_appearance_abs") or 0.0))
    stats: dict[str, tuple[float, float]] = {}
    for chunk_id, values in by_chunk.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        q50 = float(np.quantile(arr, 0.50))
        q90 = float(np.quantile(arr, 0.90))
        stats[chunk_id] = (q50, max(q90 - q50, 1e-6))
    for row in rows:
        chunk_id = str(row.get("chunk_id"))
        q50, scale = stats.get(chunk_id, (1.0, 1.0))
        rel = 0.0
        if _parse_bool(row.get("appearance_valid_pair")) and not _parse_bool(row.get("same_frame")):
            rel = float(np.clip((float(row.get("score_appearance_abs") or 0.0) - q50) / scale, 0.0, 1.0))
        if _parse_bool(row.get("appearance_mode_match")):
            rel = max(rel, 0.65)
        row["score_appearance_relative"] = rel
        combined = (
            0.40 * float(row.get("score_material_residual") or 0.0)
            + 0.20 * float(row.get("score_signature") or 0.0)
            + 0.20 * float(row.get("score_temporal_adjacent") or 0.0)
            + 0.20 * rel
        )
        frozen_combined = float(rel * (0.80 + 0.20 * float(row.get("score_temporal_adjacent") or 0.0)))
        if _parse_bool(row.get("same_frame")):
            combined = 0.0
            frozen_combined = 0.0
        row["score_combined_appearance_relative"] = float(combined)
        row["score_combined_frozen_appearance"] = float(frozen_combined)


def _hard_negative_precision(rows: list[dict[str, Any]]) -> float | None:
    hard = [row for row in rows if row.get("diagnostic_label_known") and _parse_bool(row.get("hard_negative_candidate"))]
    if not hard:
        return None
    return float(np.mean([int(row["same_object_label"]) == 0 for row in hard]))


def _scene_auc(rows: list[dict[str, Any]], scene: str, score_key: str) -> float | None:
    subset = [row for row in rows if row.get("scene_id") == scene and row.get("diagnostic_label_known")]
    return _auc([float(row.get(score_key) or 0.0) for row in subset], [int(row["same_object_label"]) for row in subset])


def _write_hist_png(path: Path, rows: list[dict[str, Any]], score_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((520, 860, 3), 255, dtype=np.uint8)
    known = [row for row in rows if row.get("diagnostic_label_known")]
    pos = [float(row.get(score_key) or 0.0) for row in known if int(row["same_object_label"]) == 1]
    neg = [float(row.get(score_key) or 0.0) for row in known if int(row["same_object_label"]) == 0]
    bins = np.linspace(0.0, 1.0, 21)
    pos_hist, _ = np.histogram(pos, bins=bins)
    neg_hist, _ = np.histogram(neg, bins=bins)
    max_count = max(int(pos_hist.max(initial=0)), int(neg_hist.max(initial=0)), 1)
    cv2.putText(img, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f"positive={len(pos)} negative={len(neg)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    x0, y0 = 60, 450
    width = 36
    for idx, (p_count, n_count) in enumerate(zip(pos_hist, neg_hist)):
        x = x0 + idx * width
        ph = int(320 * p_count / max_count)
        nh = int(320 * n_count / max_count)
        cv2.rectangle(img, (x, y0 - nh), (x + 13, y0), (60, 120, 220), -1)
        cv2.rectangle(img, (x + 15, y0 - ph), (x + 28, y0), (220, 90, 70), -1)
    cv2.line(img, (x0, y0), (x0 + len(pos_hist) * width, y0), (0, 0, 0), 1)
    cv2.putText(img, "blue=negative red=positive", (30, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _write_scene_auc_png(path: Path, scene_auc: dict[str, float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((420, 860, 3), 255, dtype=np.uint8)
    cv2.putText(img, "v68 edge audit combined AUC by scene", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    x0, y0 = 80, 350
    for idx, (scene, auc) in enumerate(scene_auc.items()):
        value = 0.0 if auc is None else float(auc)
        x = x0 + idx * 145
        h = int(260 * max(0.0, min(1.0, value)))
        cv2.rectangle(img, (x, y0 - h), (x + 70, y0), (80, 150, 90), -1)
        cv2.putText(img, f"{value:.3f}" if auc is not None else "NA", (x, y0 - h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, scene.replace("scene", "s"), (x - 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(img, (60, y0 - int(260 * 0.75)), (810, y0 - int(260 * 0.75)), (40, 40, 200), 1)
    cv2.putText(img, "gate 0.75", (715, y0 - int(260 * 0.75) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    visual_root = Path(args.visual_root)
    if not visual_root.is_absolute():
        visual_root = ROOT / visual_root
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)

    scenes = _parse_csv_list(args.scenes)
    candidate_path = Path(args.candidate_rows)
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    raw_rows = _load_csv_rows(candidate_path)
    rng = random.Random(int(args.seed))

    pipeline_roots: dict[str, str] = {}
    pipeline_roots_abs: dict[str, Path] = {}
    diag_by_node: dict[tuple[str, int, int], dict[str, Any]] = {}
    for scene in scenes:
        print(f"[v68-edge-audit] load diagnostics scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        pipeline_roots_abs[scene] = pipeline_root
        for (frame_id, mask_id), row in _mask_summary_by_pair(pipeline_root, scene).items():
            diag_by_node[(scene, int(frame_id), int(mask_id))] = row

    appearance_by_node, appearance_summary = _compute_appearance_by_node(
        raw_rows,
        scenes=scenes,
        pipeline_roots_abs=pipeline_roots_abs,
        backend=str(args.appearance_backend),
        device=str(args.appearance_device),
        checkpoint=str(args.appearance_checkpoint or "") or None,
    )

    rows_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(raw_rows):
        if str(row.get("scene_id")) not in set(scenes):
            continue
        if bool(args.representative_only) and not _parse_bool(row.get("representative_available")):
            continue
        rows_by_chunk[str(row.get("chunk_id"))].append(_prep_candidate_row(row, idx, appearance_by_node))

    edge_rows: list[dict[str, Any]] = []
    shuffle_values: list[float] = []
    for chunk_id in sorted(rows_by_chunk):
        chunk_rows = rows_by_chunk[chunk_id]
        if len(chunk_rows) < 2:
            continue
        scene = str(chunk_rows[0]["scene_id"])
        chunk_rng = random.Random(f"{int(args.seed)}::{chunk_id}")
        print(f"[v68-edge-audit] chunk={chunk_id} masks={len(chunk_rows)}", file=sys.stderr, flush=True)
        pairs = _sample_chunk_pairs(
            chunk_rows,
            rng=chunk_rng,
            max_pairs=int(args.max_pairs_per_chunk),
            max_component_bucket=int(args.max_component_bucket),
            max_signature_bucket=int(args.max_signature_bucket),
        )
        for (i, j), reasons in sorted(pairs.items()):
            a = chunk_rows[i]
            b = chunk_rows[j]
            score = _score_pair(a, b)
            label = _diag_label(a, b, diag_by_node, purity_threshold=float(args.diagnostic_gt_purity_threshold))
            shuffled = rng.random()
            shuffle_values.append(shuffled)
            edge_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "node_i": f"{scene}:{a['_frame_id']}:{a['_mask_id']}",
                    "node_j": f"{scene}:{b['_frame_id']}:{b['_mask_id']}",
                    "frame_i": a["_frame_id"],
                    "mask_i": a["_mask_id"],
                    "frame_j": b["_frame_id"],
                    "mask_j": b["_mask_id"],
                    "edge_type": "+".join(sorted(reasons)),
                    "source_reason_count": int(len(reasons)),
                    "area_i": a["_area_ratio"],
                    "area_j": b["_area_ratio"],
                    "component_count_i": len(a["_component_ids"]),
                    "component_count_j": len(b["_component_ids"]),
                    "underseg_i": bool(a["_underseg"]),
                    "underseg_j": bool(b["_underseg"]),
                    "shared_support_i": bool(a["_shared_support"]),
                    "shared_support_j": bool(b["_shared_support"]),
                    "appearance_valid_i": bool(a["_appearance_valid"]),
                    "appearance_valid_j": bool(b["_appearance_valid"]),
                    "appearance_used_pixels_i": int(a["_appearance_used_pixels"]),
                    "appearance_used_pixels_j": int(b["_appearance_used_pixels"]),
                    "appearance_mode_i": str(a["_appearance_mode"]),
                    "appearance_mode_j": str(b["_appearance_mode"]),
                    **score,
                    "score_combined_shuffled_control": shuffled,
                    **label,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                    "diagnostic_only": True,
                }
            )

    appearance_enabled = bool(appearance_summary.get("enabled"))
    _add_relative_appearance_scores(edge_rows, enabled=appearance_enabled)
    metric_specs = [
        ("E_mat_overlap", "score_material_overlap"),
        ("E_mat_residual", "score_material_residual"),
        ("E_signature", "score_signature"),
        ("E_temporal_adjacent", "score_temporal_adjacent"),
        ("E_sem_mode_unavailable", "score_semantic_mode"),
        ("E_appearance_abs" if appearance_enabled else "E_appearance_abs_unavailable", "score_appearance_abs"),
        ("E_appearance_relative" if appearance_enabled else "E_appearance_relative_unavailable", "score_appearance_relative"),
        ("E_combined_no_semantic", "score_combined_no_semantic"),
        ("E_combined_appearance_relative" if appearance_enabled else "E_combined_appearance_relative_unavailable", "score_combined_appearance_relative"),
        ("E_combined_frozen_appearance" if appearance_enabled else "E_combined_frozen_appearance_unavailable", "score_combined_frozen_appearance"),
        ("E_combined_no_temporal", "score_combined_no_temporal"),
        ("E_shuffled_control", "score_combined_shuffled_control"),
    ]
    metric_rows = [_metric_for_score(edge_rows, score_key, edge_type=edge_type) for edge_type, score_key in metric_specs]
    metric_by_type = {row["edge_type"]: row for row in metric_rows}
    if str(appearance_summary.get("backend")) == "dinov2_timm":
        combined_edge_type = "E_combined_frozen_appearance"
    else:
        combined_edge_type = "E_combined_appearance_relative" if appearance_enabled else "E_combined_no_semantic"
    combined = metric_by_type.get(combined_edge_type, {})
    shuffled = metric_by_type.get("E_shuffled_control", {})
    no_temporal = metric_by_type.get("E_combined_no_temporal", {})
    combined_auc = _float_or_none(combined.get("edge_AUC"))
    shuffled_auc = _float_or_none(shuffled.get("edge_AUC"))
    no_temporal_auc = _float_or_none(no_temporal.get("edge_AUC"))
    real_minus_shuffled = None if combined_auc is None or shuffled_auc is None else float(combined_auc - shuffled_auc)
    real_minus_no_temporal = None if combined_auc is None or no_temporal_auc is None else float(combined_auc - no_temporal_auc)
    hard_neg_precision = _hard_negative_precision(edge_rows)
    combined_score_key = str(combined.get("score_key") or "score_combined_no_semantic")
    scene_auc = {scene: _scene_auc(edge_rows, scene, combined_score_key) for scene in scenes}
    combined.update(
        {
            "real_minus_shuffled_AUC": real_minus_shuffled,
            "real_minus_no_temporal_AUC": real_minus_no_temporal,
            "hard_negative_precision": hard_neg_precision,
            "scene0081_AUC": scene_auc.get("scene0081_01"),
            "scene0591_AUC": scene_auc.get("scene0591_00"),
        }
    )
    gate = {
        "combined_positive_edge_AUC_ge_0p75": combined_auc is not None and combined_auc >= 0.75,
        "combined_positive_top1_precision_ge_0p75": _float_or_none(combined.get("top1_precision")) is not None and float(combined["top1_precision"]) >= 0.75,
        "combined_positive_top3_recall_ge_0p50": _float_or_none(combined.get("top3_recall")) is not None and float(combined["top3_recall"]) >= 0.50,
        "real_minus_shuffled_AUC_ge_0p15": real_minus_shuffled is not None and real_minus_shuffled >= 0.15,
        "real_minus_no_temporal_AUC_ge_0p10": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.10,
        "hard_negative_precision_ge_0p90": hard_neg_precision is not None and hard_neg_precision >= 0.90,
        "same_frame_violation_rate_after_filter_eq_0": _float_or_none(combined.get("same_frame_violation_rate")) == 0.0,
    }
    gate["pass"] = bool(all(gate.values()))
    if gate["pass"]:
        decision = "PASS_EDGE_EVIDENCE_CALIBRATION"
    else:
        decision = "NO_GO_EDGE_EVIDENCE" if combined_auc is None or combined_auc < 0.75 else "EDGE_EVIDENCE_PARTIAL_REPAIR_NEEDED"

    _write_csv(output_root / "edge_rows.csv", edge_rows)
    _write_csv(output_root / "edge_metric_rows.csv", metric_rows)
    _write_hist_png(visual_root / "combined_score_hist.png", edge_rows, combined_score_key, "v68 combined edge score diagnostic")
    _write_hist_png(visual_root / "material_residual_score_hist.png", edge_rows, "score_material_residual", "v68 material residual edge score diagnostic")
    if appearance_enabled:
        _write_hist_png(visual_root / "appearance_relative_score_hist.png", edge_rows, "score_appearance_relative", "v68 RGB appearance relative score diagnostic")
    _write_scene_auc_png(visual_root / "combined_auc_by_scene.png", scene_auc)
    visual_rows = [
        {"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)}
        for path in sorted(visual_root.glob("*.png"))
    ]
    _write_csv(output_root / "visualization_rows.csv", visual_rows)
    summary = {
        "phase": "v68_edge_audit",
        "decision": decision,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "candidate_rows": _rel(candidate_path),
        "representative_only": bool(args.representative_only),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "appearance": appearance_summary,
        "sampling": {
            "seed": int(args.seed),
            "max_pairs_per_chunk": int(args.max_pairs_per_chunk),
            "max_component_bucket": int(args.max_component_bucket),
            "max_signature_bucket": int(args.max_signature_bucket),
            "diagnostic_gt_purity_threshold": float(args.diagnostic_gt_purity_threshold),
            "chunk_count": int(len(rows_by_chunk)),
            "candidate_mask_count": int(sum(len(rows) for rows in rows_by_chunk.values())),
            "edge_count": int(len(edge_rows)),
            "diagnostic_known_edge_count": int(sum(1 for row in edge_rows if row.get("diagnostic_label_known"))),
            "positive_edge_count": int(sum(1 for row in edge_rows if row.get("diagnostic_label_known") and int(row["same_object_label"]) == 1)),
            "negative_edge_count": int(sum(1 for row in edge_rows if row.get("diagnostic_label_known") and int(row["same_object_label"]) == 0)),
        },
        "gate": gate,
        "combined_metrics": combined,
        "combined_edge_type": combined_edge_type,
        "scene_auc": scene_auc,
        "metric_rows": metric_rows,
        "rows": {
            "edge_rows_csv": _rel(output_root / "edge_rows.csv"),
            "edge_metric_rows_csv": _rel(output_root / "edge_metric_rows.csv"),
            "visualization_rows_csv": _rel(output_root / "visualization_rows.csv"),
        },
        "visualizations": visual_rows,
        "notes": [
            "Edge scores use candidate-bank bbox/area/component/signature/temporal fields; optional appearance repair uses RGB pixels inside the non-GT 2D masks.",
            "When appearance backend is enabled, RGB-derived mask appearance features are used as a training-free non-GT fallback; rgb_hist uses color/HSV/LAB histograms and dinov2_timm uses frozen dense RGB features. No depth, pose, mesh, or GT labels are used for prediction.",
            "GT majority instance labels from mask_summary_rows.csv are used only for diagnostic labels and AUC/top-k evaluation; labels below the configured purity threshold are treated as unknown.",
            "Semantic edge score is recorded as unavailable/zero because Phase 1 did not expose a reliable semantic feature table.",
            "Same-frame pairs are hard-filtered to zero combined score before gate evaluation and tracked separately from hard-negative precision because same-frame mask fragments can share a diagnostic GT id.",
        ],
    }
    _write_json(output_root / "edge_audit_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "edge_audit_summary.json",
        output_root / "edge_rows.csv",
        output_root / "edge_metric_rows.csv",
        output_root / "visualization_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stream4D v68 cross-frame edge evidence audit.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v68_edge_audit")
    parser.add_argument("--visual-root", default="outputs/audit/v68_visualizations/edge_audit")
    parser.add_argument("--seed", type=int, default=6802)
    parser.add_argument("--max-pairs-per-chunk", type=int, default=3500)
    parser.add_argument("--max-component-bucket", type=int, default=48)
    parser.add_argument("--max-signature-bucket", type=int, default=40)
    parser.add_argument("--diagnostic-gt-purity-threshold", type=float, default=0.50)
    parser.add_argument("--appearance-backend", default="none", choices=["none", "disabled", "rgb_hist", "dinov2_timm"])
    parser.add_argument("--appearance-device", default="cpu")
    parser.add_argument("--appearance-checkpoint", default="")
    parser.add_argument("--representative-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
