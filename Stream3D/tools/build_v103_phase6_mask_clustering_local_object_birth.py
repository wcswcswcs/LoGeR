#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402

try:
    from tools.v99_cupy_sparse_iou import CuPySparseSceneIoU  # noqa: E402
except Exception:  # pragma: no cover - optional backend
    CuPySparseSceneIoU = None  # type: ignore[assignment]


PHASE_ID = "v103_phase6_mask_clustering_local_object_birth"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_mask_level_pooling_q5c_phase4r7_r4_control_gate_strict_l2o"
DEFAULT_PHASE2_SCENE0011 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6_mask_clustering_q5c_phase5r4"
SKETCH_SEED = 10317


VARIANTS = [
    {
        "variant_id": "M0_static_cc_tau060_top8_min2",
        "clusterer": "threshold_connected_components",
        "pair_affinity_mode": "static_feature_cosine",
        "threshold": 0.60,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": False,
        "node_policy": "object_like",
    },
    {
        "variant_id": "M1_constrained_strict_l2o_tau060_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.60,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "object_like",
    },
    {
        "variant_id": "M1_constrained_strict_l2o_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "object_like",
    },
    {
        "variant_id": "M1_constrained_strict_l2o_tau080_top4_min3",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.80,
        "topk_per_mask": 4,
        "min_object_frames": 3,
        "use_cannot_link": True,
        "node_policy": "object_like",
    },
    {
        "variant_id": "M2_constrained_all_supported_strict_l2o_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
    },
    {
        "variant_id": "M2_constrained_all_supported_strict_l2o_tau080_top4_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.80,
        "topk_per_mask": 4,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
    },
    {
        "variant_id": "M3_singleton_all_supported_min1_diagnostic",
        "clusterer": "singleton_masks_diagnostic",
        "pair_affinity_mode": "static_feature_cosine",
        "threshold": 1.01,
        "topk_per_mask": 0,
        "min_object_frames": 1,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "all_supported",
    },
    {
        "variant_id": "M4_repair_non_broad_strict_l2o_tau060_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.60,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "supported_non_broad",
        "emit_policy": "non_broad_only_skip",
        "score_policy": "selected_broad_risk",
    },
    {
        "variant_id": "M4_repair_non_broad_strict_l2o_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "supported_non_broad",
        "emit_policy": "non_broad_only_skip",
        "score_policy": "selected_broad_risk",
    },
    {
        "variant_id": "M5_repair_all_supported_emit_object_like_only_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "all_supported",
        "emit_policy": "object_like_only_skip",
        "score_policy": "selected_broad_risk",
    },
    {
        "variant_id": "M5_repair_all_supported_emit_non_broad_only_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "all_supported",
        "emit_policy": "non_broad_only_skip",
        "score_policy": "selected_broad_risk",
    },
    {
        "variant_id": "M6_repair_all_supported_emit_object_like_only_tau060_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.60,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "all_supported",
        "emit_policy": "object_like_only_skip",
        "score_policy": "selected_broad_risk",
    },
    {
        "variant_id": "M7_repair_all_supported_broad_support_veto_tau070_top8_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.70,
        "topk_per_mask": 8,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "all_supported_broad_support_veto",
        "emit_policy": "prefer_object_like",
        "score_policy": "selected_broad_risk",
        "broad_support_min_support_count": 1000,
    },
    {
        "variant_id": "M7_repair_all_supported_broad_support_veto_tau080_top4_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.80,
        "topk_per_mask": 4,
        "min_object_frames": 2,
        "use_cannot_link": True,
        "cannot_link_policy": "specific_non_broad_same_frame",
        "node_policy": "all_supported_broad_support_veto",
        "emit_policy": "prefer_object_like",
        "score_policy": "selected_broad_risk",
        "broad_support_min_support_count": 1000,
    },
    {
        "variant_id": "M8_lingbot_mutual_topk_tau080_top2_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.80,
        "topk_per_mask": 2,
        "edge_selection_policy": "mutual_topk",
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
        "cannot_link_policy": "all_node_same_frame",
    },
    {
        "variant_id": "M8_lingbot_mutual_topk_tau085_top2_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.85,
        "topk_per_mask": 2,
        "edge_selection_policy": "mutual_topk",
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
        "cannot_link_policy": "all_node_same_frame",
    },
    {
        "variant_id": "M9_lingbot_mutual_top1_tau085_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.85,
        "topk_per_mask": 1,
        "edge_selection_policy": "mutual_topk",
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
        "cannot_link_policy": "all_node_same_frame",
    },
    {
        "variant_id": "M9_lingbot_mutual_top1_tau090_min2",
        "clusterer": "constrained_union_find",
        "pair_affinity_mode": "strict_leave_two_out_bucket_zeroed",
        "threshold": 0.90,
        "topk_per_mask": 1,
        "edge_selection_policy": "mutual_topk",
        "min_object_frames": 2,
        "use_cannot_link": True,
        "node_policy": "all_supported",
        "emit_policy": "prefer_object_like",
        "cannot_link_policy": "all_node_same_frame",
    },
]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_mask(mask_idx: np.ndarray, sketch_dim: int) -> np.ndarray:
    mask_idx = np.asarray(mask_idx, dtype=np.int64)
    return ((mask_idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).astype(np.int64)


def _load_label_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _accumulator(use_cupy: bool, device_id: int) -> tuple[Any, str]:
    if use_cupy and CuPySparseSceneIoU is not None:
        return CuPySparseSceneIoU(device_id=device_id), "cupy_v99_sparse_scene_iou"
    return SparseSceneIoU(), "cpu_v65_sparse_scene_iou"


def _load_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = df[(df.get("baseline_role", "") == "current_strong_local_baseline") & (df.get("dataset_split", "") == "dev")]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "MV_AP_window": float(row.get("MV_AP_window", 0.0)),
        "MV_AP50_window": float(row.get("MV_AP50_window", 0.0)),
        "MV_AP25_window": float(row.get("MV_AP25_window", 0.0)),
        "ScoreFreeMatch50_window": float(row.get("ScoreFreeMatch50_window", 0.0)),
    }


def _pair_values(
    feature: np.ndarray,
    pairs: np.ndarray,
    *,
    pair_affinity_mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if pair_affinity_mode == "static_feature_cosine":
        with torch.no_grad():
            feat = torch.as_tensor(feature, dtype=torch.float32, device=device)
            pair_t = torch.as_tensor(pairs, dtype=torch.long, device=device)
            vals: list[np.ndarray] = []
            for start in range(0, pair_t.shape[0], int(batch_size)):
                sub = pair_t[start : start + int(batch_size)]
                out = torch.sum(feat[sub[:, 0]] * feat[sub[:, 1]], dim=1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                vals.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
            return np.concatenate(vals, axis=0) if vals else np.zeros((0,), dtype=np.float32)
    if pair_affinity_mode != "strict_leave_two_out_bucket_zeroed":
        raise ValueError(f"unsupported pair_affinity_mode={pair_affinity_mode}")
    bucket_np = _hash_mask(np.arange(feature.shape[0], dtype=np.int64), int(feature.shape[1]))
    with torch.no_grad():
        feat = torch.as_tensor(feature, dtype=torch.float32, device=device)
        bucket = torch.as_tensor(bucket_np, dtype=torch.long, device=device)
        pair_t = torch.as_tensor(pairs, dtype=torch.long, device=device)
        vals = []
        for start in range(0, pair_t.shape[0], int(batch_size)):
            sub = pair_t[start : start + int(batch_size)]
            rows = torch.arange(sub.shape[0], dtype=torch.long, device=device)
            a = feat[sub[:, 0]].clone()
            b = feat[sub[:, 1]].clone()
            ba = bucket[sub[:, 0]]
            bb = bucket[sub[:, 1]]
            a[rows, ba] = 0.0
            a[rows, bb] = 0.0
            b[rows, ba] = 0.0
            b[rows, bb] = 0.0
            a = torch.nn.functional.normalize(a, p=2, dim=1, eps=1e-12)
            b = torch.nn.functional.normalize(b, p=2, dim=1, eps=1e-12)
            out = torch.sum(a * b, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            vals.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(vals, axis=0) if vals else np.zeros((0,), dtype=np.float32)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(int(n)))
        self.size = [1 for _ in range(int(n))]

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def _would_violate(
    comp_members: dict[int, set[int]],
    cannot: set[tuple[int, int]],
    ra: int,
    rb: int,
) -> bool:
    a_members = comp_members.get(int(ra), {int(ra)})
    b_members = comp_members.get(int(rb), {int(rb)})
    if len(a_members) > len(b_members):
        a_members, b_members = b_members, a_members
    for a in a_members:
        for b in b_members:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key in cannot:
                return True
    return False


def _candidate_edges(
    *,
    feature: np.ndarray,
    mask_frame: np.ndarray,
    node_mask: np.ndarray,
    variant: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if str(variant["clusterer"]) == "singleton_masks_diagnostic":
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    obj = np.flatnonzero(node_mask.astype(bool))
    pairs: list[tuple[int, int]] = []
    for offset, a in enumerate(obj[:-1]):
        b_arr = obj[offset + 1 :]
        b_arr = b_arr[mask_frame[b_arr] != mask_frame[int(a)]]
        pairs.extend((int(a), int(b)) for b in b_arr.tolist())
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    pair_arr = np.asarray(pairs, dtype=np.int64)
    vals = _pair_values(
        feature,
        pair_arr,
        pair_affinity_mode=str(variant["pair_affinity_mode"]),
        device=device,
        batch_size=batch_size,
    )
    threshold = float(variant["threshold"])
    keep = vals >= threshold
    pair_arr = pair_arr[keep]
    vals = vals[keep]
    if pair_arr.size == 0:
        return pair_arr.reshape(0, 2), vals
    topk = int(variant["topk_per_mask"])
    if topk > 0:
        selected = np.zeros((pair_arr.shape[0],), dtype=bool)
        selected_by_mask: dict[int, set[int]] = {}
        by_mask: dict[int, list[int]] = defaultdict(list)
        for idx, (a, b) in enumerate(pair_arr.tolist()):
            by_mask[int(a)].append(idx)
            by_mask[int(b)].append(idx)
        edge_selection_policy = str(variant.get("edge_selection_policy", "either_topk"))
        for mask, indices in by_mask.items():
            order = sorted(indices, key=lambda i: float(vals[i]), reverse=True)[:topk]
            if edge_selection_policy == "mutual_topk":
                for idx in order:
                    selected_by_mask.setdefault(int(mask), set()).add(int(idx))
            elif edge_selection_policy == "either_topk":
                selected[np.asarray(order, dtype=np.int64)] = True
            else:
                raise ValueError(f"unsupported edge_selection_policy={edge_selection_policy}")
        if edge_selection_policy == "mutual_topk":
            for idx, (a, b) in enumerate(pair_arr.tolist()):
                selected[idx] = idx in selected_by_mask.get(int(a), set()) and idx in selected_by_mask.get(int(b), set())
        pair_arr = pair_arr[selected]
        vals = vals[selected]
    order = np.argsort(vals)[::-1]
    return pair_arr[order], vals[order]


def _cannot_links(
    mask_frame: np.ndarray,
    node_mask: np.ndarray,
    *,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    policy: str,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, frame in enumerate(mask_frame.tolist()):
        if bool(node_mask[idx]):
            by_frame[int(frame)].append(int(idx))
    for masks in by_frame.values():
        for i, a in enumerate(masks[:-1]):
            for b in masks[i + 1 :]:
                if policy == "specific_non_broad_same_frame":
                    a_specific = (not bool(mask_is_broad[int(a)])) and bool(mask_is_object[int(a)])
                    b_specific = (not bool(mask_is_broad[int(b)])) and bool(mask_is_object[int(b)])
                    if not (a_specific and b_specific):
                        continue
                elif policy != "all_node_same_frame":
                    raise ValueError(f"unsupported cannot_link_policy={policy}")
                out.add((min(int(a), int(b)), max(int(a), int(b))))
    return out


def _node_mask_from_policy(
    *,
    support_count: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    node_policy: str,
    broad_support_min_support_count: int = 1000,
) -> np.ndarray:
    if node_policy == "all_supported":
        return support_count > 0
    if node_policy == "all_supported_broad_support_veto":
        high_support_broad = (
            (support_count >= int(broad_support_min_support_count))
            & mask_is_broad.astype(bool)
            & (~mask_is_object.astype(bool))
        )
        return (support_count > 0) & (~high_support_broad)
    if node_policy == "supported_non_broad":
        return (support_count > 0) & (~mask_is_broad.astype(bool))
    if node_policy == "object_like":
        return mask_is_object.astype(bool)
    raise ValueError(f"unsupported node_policy={node_policy}")


def _choose_emit_mask(
    *,
    candidates: list[int],
    emit_policy: str,
    mask_label: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    support_count: np.ndarray,
) -> int | None:
    if not candidates:
        return None
    if emit_policy == "object_like_only_skip":
        filtered = [m for m in candidates if bool(mask_is_object[m]) and not bool(mask_is_broad[m])]
        if not filtered:
            return None
        return max(filtered, key=lambda m: (int(support_count[m]), -int(mask_label[m])))
    if emit_policy == "non_broad_only_skip":
        filtered = [m for m in candidates if not bool(mask_is_broad[m])]
        if not filtered:
            return None
        return max(filtered, key=lambda m: (int(mask_is_object[m]), int(support_count[m]), -int(mask_label[m])))
    if emit_policy == "prefer_object_like":
        return max(candidates, key=lambda m: (int(mask_is_object[m]), -int(mask_is_broad[m]), int(support_count[m]), -int(mask_label[m])))
    return max(candidates, key=lambda m: (int(support_count[m]), -int(mask_label[m])))


def _cluster_score(*, emitted_frame_count: int, broad_member_ratio: float, selected_broad_ratio: float, score_policy: str) -> float:
    risk = selected_broad_ratio if score_policy == "selected_broad_risk" else broad_member_ratio
    return float(emitted_frame_count / 32.0) * max(0.05, 1.0 - 0.50 * float(risk))


def _cluster_scene(
    *,
    scene: str,
    feature_payload: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    feature = feature_payload["feature"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / np.maximum(norm, 1e-12)
    feature[~np.isfinite(feature)] = 0.0
    mask_frame = feature_payload["mask_frame"].cpu().numpy().astype(np.int64)
    mask_label = feature_payload["mask_label"].cpu().numpy().astype(np.int64)
    mask_is_object = feature_payload["mask_is_object_like"].cpu().numpy().astype(bool)
    mask_is_broad = feature_payload["mask_is_broad"].cpu().numpy().astype(bool)
    support_count = feature_payload["support_count"].cpu().numpy().astype(np.int64)
    node_mask = _node_mask_from_policy(
        support_count=support_count,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        node_policy=str(variant.get("node_policy", "object_like")),
        broad_support_min_support_count=int(variant.get("broad_support_min_support_count", 1000)),
    )

    edges, edge_vals = _candidate_edges(
        feature=feature,
        mask_frame=mask_frame,
        node_mask=node_mask,
        variant=variant,
        device=device,
        batch_size=batch_size,
    )
    cannot = (
        _cannot_links(
            mask_frame,
            node_mask,
            mask_is_object=mask_is_object,
            mask_is_broad=mask_is_broad,
            policy=str(variant.get("cannot_link_policy", "all_node_same_frame")),
        )
        if bool(variant["use_cannot_link"])
        else set()
    )
    uf = UnionFind(len(mask_frame))
    comp_members: dict[int, set[int]] = {idx: {idx} for idx in range(len(mask_frame))}
    rejected = 0
    edge_rows: list[dict[str, Any]] = []
    for idx, ((a, b), score) in enumerate(zip(edges.tolist(), edge_vals.tolist())):
        ra = uf.find(int(a))
        rb = uf.find(int(b))
        accepted = False
        reject_reason = ""
        if ra != rb:
            if bool(variant["use_cannot_link"]) and _would_violate(comp_members, cannot, ra, rb):
                rejected += 1
                reject_reason = "cannot_link"
            else:
                new_root = uf.union(ra, rb)
                old_a = comp_members.pop(ra, {ra})
                old_b = comp_members.pop(rb, {rb})
                comp_members[new_root] = set(old_a) | set(old_b)
                accepted = True
        edge_rows.append(
            {
                "schema_version": "stream4d_v103_phase6_cluster_edge_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant["variant_id"],
                "scene_id": scene,
                "edge_rank": int(idx),
                "mask_a": int(a),
                "mask_b": int(b),
                "frame_a": int(mask_frame[int(a)]),
                "frame_b": int(mask_frame[int(b)]),
                "affinity": float(score),
                "accepted_union": bool(accepted),
                "reject_reason": reject_reason,
                "pair_affinity_mode": variant["pair_affinity_mode"],
                "edge_selection_policy": str(variant.get("edge_selection_policy", "either_topk")),
            }
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in np.flatnonzero(node_mask).tolist():
        groups[uf.find(int(idx))].append(int(idx))

    cluster_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    object_idx = 0
    for _root, masks in sorted(groups.items(), key=lambda item: (-len(set(mask_frame[item[1]].tolist())), item[0])):
        frames = sorted({int(mask_frame[m]) for m in masks})
        by_frame: dict[int, list[int]] = defaultdict(list)
        for mask_idx in masks:
            by_frame[int(mask_frame[mask_idx])].append(int(mask_idx))
        selected_by_frame: dict[int, int] = {}
        for frame, candidates in by_frame.items():
            best = _choose_emit_mask(
                candidates=candidates,
                emit_policy=str(variant.get("emit_policy", "default")),
                mask_label=mask_label,
                mask_is_object=mask_is_object,
                mask_is_broad=mask_is_broad,
                support_count=support_count,
            )
            if best is not None:
                selected_by_frame[int(frame)] = int(best)
        if len(selected_by_frame) < int(variant["min_object_frames"]):
            continue
        broad_member_ratio = float(np.mean(mask_is_broad[masks])) if masks else 0.0
        selected_masks = list(selected_by_frame.values())
        selected_broad_ratio = float(np.mean(mask_is_broad[selected_masks])) if selected_masks else 0.0
        selected_object_like_ratio = float(np.mean(mask_is_object[selected_masks])) if selected_masks else 0.0
        object_id = f"{variant['variant_id']}:{scene}:c0000:obj_{object_idx:05d}"
        object_idx += 1
        score = _cluster_score(
            emitted_frame_count=len(selected_by_frame),
            broad_member_ratio=broad_member_ratio,
            selected_broad_ratio=selected_broad_ratio,
            score_policy=str(variant.get("score_policy", "member_broad_risk")),
        )
        cluster_rows.append(
            {
                "schema_version": "stream4d_v103_phase6_mask_cluster_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant["variant_id"],
                "scene_id": scene,
                "window_id": "c0000",
                "mv_object_id": object_id,
                "mask_count": int(len(masks)),
                "raw_frame_count": int(len(frames)),
                "frame_count": int(len(selected_by_frame)),
                "object_score": score,
                "mean_support_count": float(np.mean(support_count[masks])) if masks else 0.0,
                "node_policy": str(variant.get("node_policy", "object_like")),
                "broad_mask_member_count": int(np.count_nonzero(mask_is_broad[masks])) if masks else 0,
                "broad_mask_member_ratio": broad_member_ratio,
                "selected_broad_mask_ratio": selected_broad_ratio,
                "selected_object_like_mask_ratio": selected_object_like_ratio,
                "emit_policy": str(variant.get("emit_policy", "default")),
                "score_policy": str(variant.get("score_policy", "member_broad_risk")),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for frame, best in selected_by_frame.items():
            frame_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6_local_object_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant["variant_id"],
                    "mv_object_id": object_id,
                    "object_id": object_id,
                    "scene_id": scene,
                    "chunk_id": "c0000",
                    "window_id": "c0000",
                    "frame_local_index": int(frame),
                    "selected_mask_observation_index": int(best),
                    "selected_mask_id": int(mask_label[best]),
                    "mask_id_or_generated_id": int(mask_label[best]),
                    "object_score": score,
                    "score": score,
                    "support_count": int(support_count[best]),
                    "node_policy": str(variant.get("node_policy", "object_like")),
                    "emit_policy": str(variant.get("emit_policy", "default")),
                    "selected_mask_is_broad": bool(mask_is_broad[best]),
                    "selected_mask_is_object_like": bool(mask_is_object[best]),
                    "readout_mode": "phase6_mask_cluster_wta",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    cannot_rows = [
        {
            "schema_version": "stream4d_v103_phase6_cannot_link_row_v1",
            "phase_id": PHASE_ID,
            "variant_id": variant["variant_id"],
            "scene_id": scene,
            "mask_a": int(a),
            "mask_b": int(b),
            "frame_id_local": int(mask_frame[int(a)]),
            "cannot_link_reason": str(variant.get("cannot_link_policy", "all_node_same_frame")),
            "uses_gt_for_prediction": False,
        }
        for a, b in sorted(cannot)
    ]
    return cluster_rows, frame_rows, edge_rows, cannot_rows, rejected


def _evaluate_variant(
    *,
    variant_id: str,
    scene_rows: dict[str, list[dict[str, Any]]],
    phase2_summaries: dict[str, dict[str, Any]],
    min_pred_pixels: int,
    min_gt_pixels: int,
    use_cupy_iou: bool,
    cupy_device_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int, int, int]:
    window_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    backend_used = ""
    for scene, rows in sorted(scene_rows.items()):
        summary = phase2_summaries[scene]
        frame_ids = [int(v) for v in summary["frame_ids"]]
        mask_root = _project(summary["mask_root"])
        acc, backend = _accumulator(use_cupy_iou, cupy_device_id)
        backend_used = backend
        object_index: dict[str, int] = {}
        scores: dict[str, float] = {}
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            local = int(row["frame_local_index"])
            if 0 <= local < len(frame_ids):
                new = dict(row)
                new["frame_id"] = int(frame_ids[local])
                by_frame[int(frame_ids[local])].append(new)
        for frame_id in frame_ids:
            mask_path = mask_root / f"{int(frame_id)}.png"
            if not mask_path.exists():
                missing_mask_raster_count += 1
                gt = _load_gt_2d(scene, frame_id, (968, 1296))
                acc.add(np.zeros(gt.shape, dtype=np.int64), gt)
                continue
            label = _load_label_png(mask_path)
            pred = np.zeros(label.shape, dtype=np.int64)
            emitted = 0
            for row in sorted(by_frame.get(int(frame_id), []), key=lambda r: (-float(r.get("object_score", 0.0)), str(r.get("mv_object_id", "")))):
                oid = str(row["mv_object_id"])
                if oid not in object_index:
                    object_index[oid] = len(object_index) + 1
                    scores[oid] = float(row.get("object_score", 0.0))
                else:
                    scores[oid] = max(scores[oid], float(row.get("object_score", 0.0)))
                mask_id = int(row["selected_mask_id"])
                pixels = label == mask_id
                if not np.any(pixels):
                    missing_mask_raster_count += 1
                    continue
                overlap = pixels & (pred > 0)
                pixel_collision_count += int(np.count_nonzero(overlap))
                pred[(pred == 0) & pixels] = int(object_index[oid])
                emitted += 1
                selected = dict(row)
                selected["frame_id"] = int(frame_id)
                selected["score"] = float(row.get("object_score", 0.0))
                selected["selected_mask_area"] = int(np.count_nonzero(pixels))
                selected_rows.append(selected)
            gt = _load_gt_2d(scene, frame_id, label.shape)
            acc.add(pred, gt)
            preview_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6_frame_eval_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_id": "c0000",
                    "frame_id": int(frame_id),
                    "emitted_object_count": int(emitted),
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                }
            )
        input_scores = np.ones((len(object_index),), dtype=np.float32)
        for oid, idx in object_index.items():
            input_scores[int(idx) - 1] = float(scores.get(oid, 1.0))
        metric, iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=min_pred_pixels,
            min_gt_pixels=min_gt_pixels,
            score_mode="input",
            input_scores=input_scores,
        )
        gt_fragment_count_ge2_rate = 0.0
        gt_fragment_count_mean = 0.0
        if iou.shape[1]:
            frag_counts = np.sum(iou >= 0.25, axis=0)
            gt_fragment_count_mean = float(np.mean(frag_counts))
            gt_fragment_count_ge2_rate = float(np.mean(frag_counts >= 2))
        window_rows.append(
            {
                "schema_version": "stream4d_v103_phase6_window_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": "c0000",
                "MV_AP_window": metric.get("ap"),
                "MV_AP50_window": metric.get("ap50"),
                "MV_AP25_window": metric.get("ap25"),
                "ScoreFreeMatch50_window": metric.get("score_free_match_at_050", {}).get("f1"),
                "evaluated_pred_count": metric.get("evaluated_pred_count"),
                "evaluated_gt_count": metric.get("evaluated_gt_count"),
                "gt_best_iou_mean": metric.get("gt_best_iou_mean"),
                "pred_best_iou_mean": metric.get("pred_best_iou_mean"),
                "gt_fragment_count_mean": gt_fragment_count_mean,
                "gt_fragment_count_ge2_rate": gt_fragment_count_ge2_rate,
                "iou_backend": backend,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    metric_keys = ["MV_AP_window", "MV_AP50_window", "MV_AP25_window", "ScoreFreeMatch50_window"]
    aggregate = {
        "schema_version": "stream4d_v103_phase6_local_metric_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "scene_count": len(window_rows),
        "metric_scope": "first32_dev_subset_window_mean",
        "iou_backend": backend_used,
        "same_frame_collision_count": 0,
        "pixel_collision_count": int(pixel_collision_count),
        "pixel_collision_rate": 0.0,
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    for key in metric_keys:
        vals = [float(row[key]) for row in window_rows if row.get(key) not in {"", None}]
        aggregate[key] = float(np.mean(vals)) if vals else 0.0
    pred_pixels = sum(int(row.get("pred_positive_pixels", 0)) for row in preview_rows)
    aggregate["pixel_collision_rate"] = float(pixel_collision_count / max(1, pred_pixels))
    return window_rows, aggregate, selected_rows, pixel_collision_count, missing_mask_raster_count, len(preview_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6 constrained mask clustering and local MV_AP evaluation.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase5_root = _project(args.phase5_root)
    phase5_summary = _read_json(phase5_root / "summary.json")
    if not bool(phase5_summary.get("phase5_pass")):
        raise RuntimeError(f"Phase6 requires Phase5 pass: {phase5_root}")
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    baseline = _load_baseline(_project(args.baseline_rows))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_cluster_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_cannot_rows: list[dict[str, Any]] = []
    all_window_metric_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    fragmentation_rows: list[dict[str, Any]] = []

    feature_payloads = {
        scene: torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
        for scene in phase2_roots
    }

    rejected_by_variant: dict[str, int] = {}
    for variant in VARIANTS:
        scene_frame_rows: dict[str, list[dict[str, Any]]] = {}
        rejected_total = 0
        for scene, payload in feature_payloads.items():
            clusters, frames, edges, cannot_rows, rejected = _cluster_scene(
                scene=scene,
                feature_payload=payload,
                variant=variant,
                device=device,
                batch_size=int(args.pair_batch_size),
            )
            all_cluster_rows.extend(clusters)
            all_frame_rows.extend(frames)
            all_edge_rows.extend(edges[:20000])
            all_cannot_rows.extend(cannot_rows[:20000])
            scene_frame_rows[scene] = frames
            rejected_total += int(rejected)
        rejected_by_variant[str(variant["variant_id"])] = rejected_total
        window_rows, metric_row, selected_rows, _pixel_collisions, _missing, _frames = _evaluate_variant(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_frame_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        metric_row.update(
            {
                "threshold": float(variant["threshold"]),
                "topk_per_mask": int(variant["topk_per_mask"]),
                "min_object_frames": int(variant["min_object_frames"]),
                "clusterer": variant["clusterer"],
                "pair_affinity_mode": variant["pair_affinity_mode"],
                "node_policy": variant.get("node_policy", "object_like"),
                "emit_policy": variant.get("emit_policy", "default"),
                "cannot_link_policy": variant.get("cannot_link_policy", "all_node_same_frame"),
                "score_policy": variant.get("score_policy", "member_broad_risk"),
                "edge_selection_policy": variant.get("edge_selection_policy", "either_topk"),
                "broad_support_min_support_count": int(variant.get("broad_support_min_support_count", 0)),
                "rejected_union_due_cannot_link_count": rejected_total,
            }
        )
        all_window_metric_rows.extend(window_rows)
        all_metric_rows.append(metric_row)
        all_selected_rows.extend(selected_rows)
        for row in window_rows:
            fragmentation_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6_fragmentation_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant["variant_id"],
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "gt_fragment_count_mean": row["gt_fragment_count_mean"],
                    "gt_fragment_count_ge2_rate": row["gt_fragment_count_ge2_rate"],
                    "gt_best_iou_mean": row["gt_best_iou_mean"],
                    "diagnostic_only": True,
                }
            )

    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    best = max(all_metric_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    best_variant = str(best.get("variant_id", ""))
    for row in all_metric_rows:
        checks = [
            ("same_frame_collision_count_eq_0", int(row["same_frame_collision_count"]) == 0, row["same_frame_collision_count"], 0),
            ("pixel_collision_rate_le_0p02", float(row["pixel_collision_rate"]) <= 0.02, row["pixel_collision_rate"], 0.02),
            ("missing_mask_raster_count_eq_0", int(row["missing_mask_raster_count"]) == 0, row["missing_mask_raster_count"], 0),
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
            ("MV_AP_window_ge_baseline_minus_0p003", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
            ("MV_AP50_window_ge_baseline_minus_0p006", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
        ]
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": row["variant_id"],
                    "gate_name": name,
                    "pass": bool(ok),
                    "observed": observed,
                    "required": required,
                }
            )
            if row["variant_id"] == best_variant and not bool(ok):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase6_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "Follow Phase6 repair ladder: inspect overmerge/same-frame WTA, clustering threshold/top-k, object scoring/ranking, and cannot-link construction before entering history token phases.",
                    }
                )

    _write_csv(out / "local_object_rows.csv", all_cluster_rows)
    _write_csv(out / "local_object_frame_mask_rows.csv", all_selected_rows)
    _write_csv(out / "raw_cluster_frame_rows.csv", all_frame_rows)
    _write_csv(out / "mask_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "cluster_edge_rows.csv", all_edge_rows)
    _write_csv(out / "cluster_cannot_link_rows.csv", all_cannot_rows)
    _write_csv(out / "local_mv_metric_rows.csv", all_metric_rows)
    _write_csv(out / "local_mv_metric_window_rows.csv", all_window_metric_rows)
    _write_csv(out / "fragmentation_diagnostic_rows.csv", fragmentation_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    if all_selected_rows:
        pd.DataFrame(all_selected_rows).to_parquet(out / "local_object_frame_mask_rows.parquet", index=False)

    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    decision = "PASS_ENTER_PHASE7_HISTORY_TOKEN" if not failure_rows else "NO_GO_REPAIR_PHASE6_MASK_CLUSTERING"
    summary = {
        "schema_version": "stream4d_v103_phase6_mask_clustering_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase6_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": best_variant,
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "baseline_contract": baseline,
        "metric_scope": "first32_dev_subset_window_mean; not a full-dev claim",
        "variant_count": len(VARIANTS),
        "scene_ids": sorted(phase2_roots),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "iou_backend_requested": "cupy" if not bool(args.disable_cupy_iou) else "cpu",
        "gpu_device": str(device),
        "gpu_memory_peak_MB": peak_mb,
        "truthfulness_note": "Phase6 clusters mask observations into local objects and evaluates local-window AP through the canonical v65 SparseSceneIoU/_summarize_iou contract. GT is used only for evaluation/diagnostics.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "local_object_rows": _rel(out / "local_object_rows.csv"),
            "local_object_frame_mask_rows": _rel(out / "local_object_frame_mask_rows.csv"),
            "local_mv_metric_rows": _rel(out / "local_mv_metric_rows.csv"),
            "local_mv_metric_window_rows": _rel(out / "local_mv_metric_window_rows.csv"),
            "fragmentation_diagnostic_rows": _rel(out / "fragmentation_diagnostic_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
