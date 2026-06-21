from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WindowTrace:
    window_index: int
    path: str
    frame_ids: list[int]
    carrier_ids: np.ndarray
    visible: np.ndarray
    labels_by_frame: dict[int, np.ndarray]


@dataclass
class MaskNode:
    node_id: int
    scene: str
    frame_id: int
    mask_id: int
    area: int
    support_count: int = 0
    support_density: float = 0.0
    dominant_gt: int | None = None
    dominant_gt_purity: float | None = None
    inc_by_window: dict[int, set[int]] = field(default_factory=dict)
    carrier_keys: set[tuple[int, int]] = field(default_factory=set)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _load_label(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int32, copy=False)


def _load_mask_label(scene: str, frame_id: int) -> np.ndarray | None:
    return _load_label(ROOT / "data/scannet/processed" / scene / "output_Cropformer/mask" / f"{int(frame_id)}.png")


def _load_gt_label(scene: str, frame_id: int) -> np.ndarray | None:
    return _load_label(ROOT / "data/scannet/processed" / scene / "instance/instance" / f"{int(frame_id)}.png")


def _dominant_gt(mask: np.ndarray, gt: np.ndarray | None) -> tuple[int | None, float | None]:
    if gt is None:
        return None, None
    if gt.shape != mask.shape:
        gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    values, counts = np.unique(gt[np.asarray(mask, dtype=bool)], return_counts=True)
    pairs = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) > 0]
    if not pairs:
        return None, None
    label, count = max(pairs, key=lambda item: item[1])
    return int(label), float(count / max(int(mask.sum()), 1))


def _window_sort_key(path: Path) -> int:
    text = path.stem.replace("carriers_window", "")
    try:
        return int(text)
    except ValueError:
        return 0


def _read_manifest(scene_dir: Path, npz_path: Path) -> dict[str, Any]:
    manifest = scene_dir / f"{npz_path.stem}_manifest.json"
    if not manifest.exists():
        return {}
    return json.loads(manifest.read_text())


def _load_scene_windows(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_threshold: float,
    confidence_threshold: float,
    min_mask_area: int,
) -> tuple[list[WindowTrace], list[dict[str, Any]], dict[str, Any]]:
    scene_dir = carrier_cache_root / scene
    windows: list[WindowTrace] = []
    window_rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for window_serial, npz_path in enumerate(sorted(scene_dir.glob("carriers_window*.npz"), key=_window_sort_key)):
        manifest = _read_manifest(scene_dir, npz_path)
        frame_ids = [int(x) for x in manifest.get("frame_ids", [])]
        if not frame_ids:
            continue
        data = np.load(npz_path)
        uv = np.asarray(data["uv_pred"], dtype=np.float32)
        visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
        confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        carrier_ids = np.asarray(data["carrier_id"], dtype=np.int64)
        labels_by_frame: dict[int, np.ndarray] = {}
        visible_bool = valid & (visibility >= float(visibility_threshold)) & (confidence >= float(confidence_threshold))
        frame_count_with_masks = 0
        valid_visible_obs = 0
        uv_in01_obs = 0
        inside_mask_obs = 0
        for local_index, frame_id in enumerate(frame_ids[: uv.shape[0]]):
            label = _load_mask_label(scene, int(frame_id))
            if label is None:
                continue
            frame_count_with_masks += 1
            height, width = label.shape[:2]
            values, counts = np.unique(label, return_counts=True)
            allowed_labels = {int(value) for value, count in zip(values, counts) if int(value) > 0 and int(count) >= int(min_mask_area)}
            frame_visible = visible_bool[local_index]
            valid_visible_obs += int(frame_visible.sum())
            pts = uv[local_index]
            u = pts[:, 0]
            v = pts[:, 1]
            in01 = frame_visible & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            uv_in01_obs += int(in01.sum())
            label_at_carrier = np.zeros((uv.shape[1],), dtype=np.int32)
            if np.any(in01):
                xs = np.rint(u[in01] * float(width - 1)).astype(np.int64)
                ys = np.rint(v[in01] * float(height - 1)).astype(np.int64)
                sampled = label[ys, xs].astype(np.int32, copy=False)
                if allowed_labels:
                    sampled = np.asarray([int(value) if int(value) in allowed_labels else 0 for value in sampled], dtype=np.int32)
                else:
                    sampled = np.zeros_like(sampled, dtype=np.int32)
                label_at_carrier[np.flatnonzero(in01)] = sampled
                inside_mask_obs += int((sampled > 0).sum())
            labels_by_frame[int(frame_id)] = label_at_carrier
        windows.append(
            WindowTrace(
                window_index=int(window_serial),
                path=str(npz_path.relative_to(ROOT)),
                frame_ids=frame_ids,
                carrier_ids=carrier_ids,
                visible=visible_bool,
                labels_by_frame=labels_by_frame,
            )
        )
        window_rows.append(
            {
                "scene": scene,
                "window_index": int(window_serial),
                "carrier_npz": str(npz_path.relative_to(ROOT)),
                "manifest": str((scene_dir / f"{npz_path.stem}_manifest.json").relative_to(ROOT)),
                "variant": manifest.get("variant"),
                "frame_count": len(frame_ids),
                "frame_count_with_masks": frame_count_with_masks,
                "carrier_count": int(carrier_ids.shape[0]),
                "valid_visible_observation_count": valid_visible_obs,
                "uv_in01_observation_count": uv_in01_obs,
                "inside_mask_observation_count": inside_mask_obs,
                "uv_in01_rate": float(uv_in01_obs / max(valid_visible_obs, 1)),
                "carrier_inside_any_mask_ratio": float(inside_mask_obs / max(uv_in01_obs, 1)),
                "uses_gt_for_prediction": False,
                "uses_pose_for_prediction": bool(manifest.get("uses_pose_for_prediction", False)),
                "uses_rgbd_for_prediction": bool(manifest.get("uses_rgbd_for_prediction", False)),
                "uses_scannet_mesh_for_prediction": bool(manifest.get("uses_scannet_mesh_for_prediction", False)),
                "is_diagnostic_only": bool(manifest.get("is_diagnostic_only", True)),
            }
        )
        source_manifests.append(manifest)
    return windows, window_rows, {"source_manifests": source_manifests}


def _build_nodes(scene: str, windows: list[WindowTrace], *, min_mask_area: int) -> tuple[list[MaskNode], list[dict[str, Any]], dict[str, Any]]:
    node_by_key: dict[tuple[int, int], MaskNode] = {}
    frame_rows: list[dict[str, Any]] = []
    for window in windows:
        for local_index, frame_id in enumerate(window.frame_ids):
            if int(frame_id) not in window.labels_by_frame:
                continue
            label = _load_mask_label(scene, int(frame_id))
            if label is None:
                continue
            values, counts = np.unique(label, return_counts=True)
            labels_in_frame = [int(value) for value, count in zip(values, counts) if int(value) > 0 and int(count) >= int(min_mask_area)]
            gt = _load_gt_label(scene, int(frame_id))
            for mask_id in labels_in_frame:
                key = (int(frame_id), int(mask_id))
                node = node_by_key.get(key)
                if node is None:
                    mask = label == int(mask_id)
                    dominant_gt, dominant_purity = _dominant_gt(mask, gt)
                    node = MaskNode(
                        node_id=len(node_by_key),
                        scene=scene,
                        frame_id=int(frame_id),
                        mask_id=int(mask_id),
                        area=int(mask.sum()),
                        dominant_gt=dominant_gt,
                        dominant_gt_purity=dominant_purity,
                    )
                    node_by_key[key] = node
            labels_at_carrier = window.labels_by_frame[int(frame_id)]
            covered = labels_at_carrier > 0
            frame_rows.append(
                {
                    "scene": scene,
                    "window_index": window.window_index,
                    "frame_id": int(frame_id),
                    "mask_count": len(labels_in_frame),
                    "carrier_count": int(labels_at_carrier.shape[0]),
                    "visible_carrier_count": int(window.visible[local_index].sum()),
                    "inside_mask_carrier_count": int(covered.sum()),
                    "inside_mask_ratio": float(covered.sum() / max(int(window.visible[local_index].sum()), 1)),
                }
            )
            for local_carrier_index, mask_id in enumerate(labels_at_carrier.tolist()):
                if int(mask_id) <= 0:
                    continue
                node = node_by_key.get((int(frame_id), int(mask_id)))
                if node is None:
                    continue
                node.support_count += 1
                node.inc_by_window.setdefault(window.window_index, set()).add(int(local_carrier_index))
                node.carrier_keys.add((window.window_index, int(local_carrier_index)))
    nodes = list(node_by_key.values())
    for node in nodes:
        node.support_density = float(node.support_count / max(node.area, 1))
    diag = {
        "observed_frame_count": int(len({node.frame_id for node in nodes})),
        "node_count": int(len(nodes)),
        "gt_labeled_node_count": int(sum(1 for node in nodes if node.dominant_gt is not None)),
    }
    return nodes, frame_rows, diag


def _rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos_count = sum(1 for label, _score in pairs if label)
    neg_count = len(pairs) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    pairs.sort(key=lambda item: item[1])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum_pos += avg_rank * sum(1 for label, _score in pairs[idx:end] if label)
        idx = end
    return float((rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    labeled = [row for row in rows if row.get("diagnostic_same_gt") is not None]
    ranked = sorted(labeled, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(labeled))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if row.get("diagnostic_same_gt") is True) / len(ranked))


def _shared_jaccard(left: MaskNode, right: MaskNode) -> float:
    if not left.carrier_keys and not right.carrier_keys:
        return 0.0
    inter = len(left.carrier_keys & right.carrier_keys)
    union = len(left.carrier_keys | right.carrier_keys)
    return float(inter / max(union, 1))


def _view_consensus(
    left: MaskNode,
    right: MaskNode,
    windows_by_index: dict[int, WindowTrace],
    *,
    min_visible_carriers: int,
) -> tuple[float, int, float]:
    observer_scores: list[float] = []
    common_windows = sorted(set(left.inc_by_window) & set(right.inc_by_window))
    for window_index in common_windows:
        window = windows_by_index[window_index]
        left_idx = np.asarray(sorted(left.inc_by_window[window_index]), dtype=np.int64)
        right_idx = np.asarray(sorted(right.inc_by_window[window_index]), dtype=np.int64)
        if left_idx.size == 0 or right_idx.size == 0:
            continue
        for local_index, frame_id in enumerate(window.frame_ids):
            labels_at_carrier = window.labels_by_frame.get(int(frame_id))
            if labels_at_carrier is None:
                continue
            left_visible = left_idx[window.visible[local_index, left_idx]]
            right_visible = right_idx[window.visible[local_index, right_idx]]
            if left_visible.size < int(min_visible_carriers) or right_visible.size < int(min_visible_carriers):
                continue
            left_labels = labels_at_carrier[left_visible]
            right_labels = labels_at_carrier[right_visible]
            left_counts = _positive_label_counts(left_labels)
            right_counts = _positive_label_counts(right_labels)
            if not left_counts or not right_counts:
                observer_scores.append(0.0)
                continue
            score = 0.0
            for label in set(left_counts) & set(right_counts):
                score = max(score, min(left_counts[label] / left_visible.size, right_counts[label] / right_visible.size))
            observer_scores.append(float(score))
    if not observer_scores:
        return 0.0, 0, 0.0
    return float(np.mean(observer_scores)), int(len(observer_scores)), float(np.max(observer_scores))


def _positive_label_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _deterministic_permutation(nodes: list[MaskNode], *, seed_text: str) -> list[int]:
    keyed: list[tuple[str, int]] = []
    for idx, node in enumerate(nodes):
        key = f"{seed_text}:{node.scene}:{node.frame_id}:{node.mask_id}:{node.support_count}:{idx}"
        keyed.append((hashlib.sha1(key.encode("utf-8")).hexdigest(), idx))
    keyed.sort()
    return [idx for _key, idx in keyed]


def _edge_rows_for_scene(
    *,
    scene: str,
    nodes: list[MaskNode],
    windows: list[WindowTrace],
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        node
        for node in nodes
        if node.support_count >= int(min_node_carriers) and node.dominant_gt is not None and node.dominant_gt_purity is not None
    ]
    eligible.sort(key=lambda node: (node.support_count, node.area), reverse=True)
    capped = eligible[: int(max_edge_nodes)]
    windows_by_index = {window.window_index: window for window in windows}
    permuted_indices = _deterministic_permutation(capped, seed_text=f"v46_raw_shuffle:{scene}") if capped else []
    shuffled_by_node_id: dict[int, MaskNode] = {}
    for idx, node in enumerate(capped):
        shuffled_by_node_id[node.node_id] = capped[permuted_indices[idx % len(permuted_indices)]]
    edge_rows: list[dict[str, Any]] = []
    for i, left in enumerate(capped):
        for right in capped[i + 1 :]:
            same_gt = bool(left.dominant_gt == right.dominant_gt)
            vc, observer_count, vc_max = _view_consensus(
                left,
                right,
                windows_by_index,
                min_visible_carriers=int(min_visible_carriers),
            )
            shuffled_right = shuffled_by_node_id.get(right.node_id, right)
            shuffled_vc, shuffled_observer_count, _shuffled_vc_max = _view_consensus(
                left,
                shuffled_right,
                windows_by_index,
                min_visible_carriers=int(min_visible_carriers),
            )
            edge_rows.append(
                {
                    "scene": scene,
                    "left_node_id": left.node_id,
                    "right_node_id": right.node_id,
                    "left_frame_id": left.frame_id,
                    "right_frame_id": right.frame_id,
                    "left_mask_id": left.mask_id,
                    "right_mask_id": right.mask_id,
                    "left_support_count": left.support_count,
                    "right_support_count": right.support_count,
                    "left_gt": left.dominant_gt,
                    "right_gt": right.dominant_gt,
                    "diagnostic_same_gt": same_gt,
                    "raw_view_consensus": vc,
                    "raw_view_consensus_max_observer": vc_max,
                    "shared_carrier_jaccard": _shared_jaccard(left, right),
                    "observer_count": observer_count,
                    "shuffled_view_consensus": shuffled_vc,
                    "shuffled_observer_count": shuffled_observer_count,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    labels = [bool(row["diagnostic_same_gt"]) for row in edge_rows]
    summary_rows: list[dict[str, Any]] = []
    for score_key in ["raw_view_consensus", "shared_carrier_jaccard", "shuffled_view_consensus"]:
        scores = [float(row[score_key]) for row in edge_rows]
        summary_rows.append(
            {
                "scene": scene,
                "score_key": score_key,
                "edge_eval_node_count": len(capped),
                "edge_count": len(edge_rows),
                "mean_observer_count": _safe_mean(row["observer_count"] for row in edge_rows),
                "median_observer_count": _safe_median(row["observer_count"] for row in edge_rows),
                "score_mean": _safe_mean(scores),
                "score_p90": _safe_quantile(scores, 0.90),
                "edge_same_gt_AUC": _rank_auc(labels, scores),
                "edge_precision@top1k": _precision_at_k(edge_rows, score_key, 1000),
                "edge_precision@top5k": _precision_at_k(edge_rows, score_key, 5000),
            }
        )
    raw = next((row for row in summary_rows if row["score_key"] == "raw_view_consensus"), {})
    shared = next((row for row in summary_rows if row["score_key"] == "shared_carrier_jaccard"), {})
    shuffled = next((row for row in summary_rows if row["score_key"] == "shuffled_view_consensus"), {})
    if raw:
        raw_auc = raw.get("edge_same_gt_AUC")
        shared_auc = shared.get("edge_same_gt_AUC")
        shuffled_auc = shuffled.get("edge_same_gt_AUC")
        raw_p5 = raw.get("edge_precision@top5k")
        shared_p5 = shared.get("edge_precision@top5k")
        raw["real_minus_shared_edge_AUC"] = None if raw_auc is None or shared_auc is None else float(raw_auc - shared_auc)
        raw["real_minus_shuffled_edge_AUC"] = None if raw_auc is None or shuffled_auc is None else float(raw_auc - shuffled_auc)
        raw["precision_top5k_minus_shared"] = None if raw_p5 is None or shared_p5 is None else float(raw_p5 - shared_p5)
        raw["gate_pass"] = bool(
            raw_auc is not None
            and shared_auc is not None
            and shuffled_auc is not None
            and float(raw_auc) >= float(shared_auc) + 0.08
            and float(raw_auc) >= float(shuffled_auc) + 0.10
            and raw["precision_top5k_minus_shared"] is not None
            and float(raw["precision_top5k_minus_shared"]) >= 0.10
        )
    return edge_rows, summary_rows


def _safe_mean(values: Any) -> float | None:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(nums)) if nums else None


def _safe_median(values: Any) -> float | None:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(nums)) if nums else None


def _safe_quantile(values: Any, q: float) -> float | None:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.quantile(nums, float(q))) if nums else None


def _scene_rows_for_thresholds(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_thresholds: list[float],
    confidence_threshold: float,
    min_mask_area: int,
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
) -> dict[str, Any]:
    scene_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_edge_summary_rows: list[dict[str, Any]] = []
    first_variant_diag: dict[str, Any] = {}
    for visibility_threshold in visibility_thresholds:
        windows, window_rows, manifest_diag = _load_scene_windows(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_threshold=float(visibility_threshold),
            confidence_threshold=float(confidence_threshold),
            min_mask_area=int(min_mask_area),
        )
        nodes, frame_rows, node_diag = _build_nodes(scene, windows, min_mask_area=int(min_mask_area))
        support_counts = [node.support_count for node in nodes]
        support_density = [node.support_density for node in nodes]
        window_inside_obs = sum(int(row["inside_mask_observation_count"]) for row in window_rows)
        window_uv_in_obs = sum(int(row["uv_in01_observation_count"]) for row in window_rows)
        row = {
            "scene": scene,
            "variant": f"I_raw_visibility_{visibility_threshold:g}",
            "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
            "incidence_mode": "raw_uv_containment_prepared_cropformer_masks",
            "window_count": len(windows),
            "mask_frame_count": len({node.frame_id for node in nodes}),
            "mask_count": len(nodes),
            "carrier_count": int(sum(len(window.carrier_ids) for window in windows)),
            "visible_carrier_count": int(sum(int(window.visible.sum()) for window in windows)),
            "mask_with_ge1_carrier_ratio": float(sum(v >= 1 for v in support_counts) / max(len(support_counts), 1)),
            "mask_with_ge5_carrier_ratio": float(sum(v >= 5 for v in support_counts) / max(len(support_counts), 1)),
            "mask_with_ge16_carrier_ratio": float(sum(v >= 16 for v in support_counts) / max(len(support_counts), 1)),
            "carrier_inside_any_mask_ratio": float(window_inside_obs / max(window_uv_in_obs, 1)),
            "mean_carriers_per_mask": _safe_mean(support_counts),
            "median_carriers_per_mask": _safe_median(support_counts),
            "p10_carriers_per_mask": _safe_quantile(support_counts, 0.10),
            "support_density_mean": _safe_mean(support_density),
            "support_density_p10": _safe_quantile(support_density, 0.10),
            "scene0081_support_density": _safe_mean(support_density) if scene == "scene0081_01" else None,
            "scene0591_support_density": _safe_mean(support_density) if scene == "scene0591_00" else None,
            "uv_in01_rate": float(window_uv_in_obs / max(sum(int(x["valid_visible_observation_count"]) for x in window_rows), 1)),
            "uses_raw_uv_containment": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["gate_pass"] = bool(
            row["mask_with_ge5_carrier_ratio"] >= 0.70
            and row["mask_with_ge16_carrier_ratio"] >= 0.40
            and row["carrier_inside_any_mask_ratio"] >= 0.65
            and row["support_density_p10"] is not None
            and float(row["support_density_p10"]) > 0.0
        )
        scene_rows.append(row)
        for node in nodes:
            all_node_rows.append(
                {
                    "scene": scene,
                    "variant": row["variant"],
                    "node_id": node.node_id,
                    "frame_id": node.frame_id,
                    "mask_id": node.mask_id,
                    "area": node.area,
                    "carrier_support_count": node.support_count,
                    "support_density": node.support_density,
                    "diagnostic_gt_instance": node.dominant_gt,
                    "diagnostic_gt_purity": node.dominant_gt_purity,
                    "uses_gt_for_prediction": False,
                }
            )
        for frame_row in frame_rows:
            all_frame_rows.append({"variant": row["variant"], **frame_row})
        for window_row in window_rows:
            all_window_rows.append({"variant": row["variant"], **window_row})
        edge_rows, edge_summary_rows = _edge_rows_for_scene(
            scene=scene,
            nodes=nodes,
            windows=windows,
            max_edge_nodes=int(max_edge_nodes),
            min_node_carriers=int(min_node_carriers),
            min_visible_carriers=int(min_visible_carriers),
        )
        for edge_row in edge_rows:
            all_edge_rows.append({"variant": row["variant"], **edge_row})
        for edge_summary_row in edge_summary_rows:
            all_edge_summary_rows.append({"variant": row["variant"], **edge_summary_row})
        if not first_variant_diag:
            first_variant_diag = {**manifest_diag, **node_diag}
    return {
        "scene_rows": scene_rows,
        "node_rows": all_node_rows,
        "frame_rows": all_frame_rows,
        "window_rows": all_window_rows,
        "edge_rows": all_edge_rows,
        "edge_summary_rows": all_edge_summary_rows,
        "diag": first_variant_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair audit for v46 raw carrier-mask incidence from D4RT carrier cache.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--visibility-thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--max-edge-nodes", type=int, default=360)
    parser.add_argument("--min-node-carriers", type=int, default=5)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--output-root", default="outputs/audit/v46_raw_carrier_incidence_repair")
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    visibility_thresholds = [float(item.strip()) for item in str(args.visibility_thresholds).split(",") if item.strip()]
    all_scene_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_edge_summary_rows: list[dict[str, Any]] = []
    diags: dict[str, Any] = {}
    for scene in scenes:
        payload = _scene_rows_for_thresholds(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_thresholds=visibility_thresholds,
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            max_edge_nodes=int(args.max_edge_nodes),
            min_node_carriers=int(args.min_node_carriers),
            min_visible_carriers=int(args.min_visible_carriers),
        )
        all_scene_rows.extend(payload["scene_rows"])
        all_node_rows.extend(payload["node_rows"])
        all_frame_rows.extend(payload["frame_rows"])
        all_window_rows.extend(payload["window_rows"])
        all_edge_rows.extend(payload["edge_rows"])
        all_edge_summary_rows.extend(payload["edge_summary_rows"])
        diags[scene] = payload["diag"]
    edge_gate_rows = [
        row for row in all_edge_summary_rows if row.get("score_key") == "raw_view_consensus" and row.get("variant") == f"I_raw_visibility_{visibility_thresholds[0]:g}"
    ]
    incidence_gate_rows = [row for row in all_scene_rows if row.get("variant") == f"I_raw_visibility_{visibility_thresholds[0]:g}"]
    payload = {
        "phase": "v46_raw_carrier_incidence_repair",
        "created_at": _utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "visibility_thresholds": visibility_thresholds,
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "max_edge_nodes": int(args.max_edge_nodes),
        "min_node_carriers": int(args.min_node_carriers),
        "min_visible_carriers": int(args.min_visible_carriers),
        "scene_rows": all_scene_rows,
        "edge_summary_rows": all_edge_summary_rows,
        "diag": diags,
        "gate": {
            "uses_raw_uv_containment": True,
            "incidence_all_scene_gate_pass": bool(incidence_gate_rows and all(bool(row["gate_pass"]) for row in incidence_gate_rows)),
            "incidence_any_scene_gate_pass": any(bool(row["gate_pass"]) for row in incidence_gate_rows),
            "raw_view_consensus_any_scene_gate_pass": any(bool(row.get("gate_pass")) for row in edge_gate_rows),
            "raw_view_consensus_all_scene_gate_pass": bool(edge_gate_rows and all(bool(row.get("gate_pass")) for row in edge_gate_rows)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["incidence_all_scene_gate_pass"] and payload["gate"]["raw_view_consensus_all_scene_gate_pass"])
    out = ROOT / str(args.output_root)
    _write_json(out / "raw_carrier_incidence_repair.json", payload)
    _write_csv(out / "raw_incidence_scene_rows.csv", all_scene_rows)
    _write_csv(out / "raw_incidence_node_rows.csv", all_node_rows)
    _write_csv(out / "raw_incidence_frame_rows.csv", all_frame_rows)
    _write_csv(out / "raw_incidence_window_rows.csv", all_window_rows)
    _write_csv(out / "raw_view_consensus_edge_summary_rows.csv", all_edge_summary_rows)
    _write_csv(out / "raw_view_consensus_edge_rows.csv", all_edge_rows)
    print(json.dumps({"summary": str(out / "raw_carrier_incidence_repair.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
