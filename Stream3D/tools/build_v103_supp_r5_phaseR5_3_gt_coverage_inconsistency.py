#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
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

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_supp_r5_phaseR5_3_gt_coverage_inconsistency"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_gt_coverage"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_ANCHOR_ONLY_ROOT = AUDIT_ROOT / "v103_supp_r5_anchor_only_local_ap_diag"
DEFAULT_CURRENT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_D4RT_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}
D9_VARIANT = "D9_affinity_merge_tau065_top1_broad_support_veto"
D0_VARIANT = "D0_f2_original_replay"


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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _load_label_png(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(str(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return np.asarray(label, dtype=np.int32)


def _gt_boundary(gt: np.ndarray) -> np.ndarray:
    pos = gt > 0
    if not np.any(pos):
        return pos
    boundary = np.zeros(gt.shape, dtype=bool)
    boundary[1:, :] |= gt[1:, :] != gt[:-1, :]
    boundary[:-1, :] |= gt[:-1, :] != gt[1:, :]
    boundary[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    boundary[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    return boundary & pos


def _role_flags(role_df: pd.DataFrame, scene: str, carrier_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdf = role_df[role_df["scene_id"].astype(str) == scene]
    ids = sdf["carrier_id"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    target = np.asarray(carrier_id, dtype=np.int64)
    pos = np.searchsorted(ids_sorted, target)
    found = (pos < ids_sorted.shape[0]) & (ids_sorted[np.minimum(pos, ids_sorted.shape[0] - 1)] == target)
    if not np.all(found):
        raise RuntimeError(f"S1 role rows do not cover all carriers for {scene}; missing={int(np.count_nonzero(~found))}")

    def col(name: str) -> np.ndarray:
        return sdf[name].to_numpy(dtype=bool, copy=False)[order][pos]

    return col("is_A_anchor"), col("is_S_support"), col("is_V_veto")


def _empty_gt_acc(scene: str, gt_id: int) -> dict[str, Any]:
    return {
        "scene_id": scene,
        "gt_id": int(gt_id),
        "gt_visible_frames": set(),
        "gt_pixel_area_sum": 0,
        "A_anchor_support_count": 0,
        "S_support_support_count": 0,
        "A_anchor_boundary_support_count": 0,
        "S_support_boundary_support_count": 0,
        "A_anchor_frame_hits": set(),
        "S_support_frame_hits": set(),
        "A_anchor_object_like_mask_overlap_count": 0,
        "S_support_object_like_mask_overlap_count": 0,
        "A_anchor_broad_overlap_count": 0,
        "S_support_broad_overlap_count": 0,
    }


def _add_counts(acc: dict[int, dict[str, Any]], scene: str, frame_id: int, labels: np.ndarray, role_name: str, boundary: bool = False) -> None:
    labels = labels[labels > 0]
    if labels.size == 0:
        return
    values, counts = np.unique(labels, return_counts=True)
    for gt_id, count in zip(values.tolist(), counts.tolist()):
        item = acc.setdefault(int(gt_id), _empty_gt_acc(scene, int(gt_id)))
        key = f"{role_name}_boundary_support_count" if boundary else f"{role_name}_support_count"
        item[key] += int(count)
        if not boundary:
            item[f"{role_name}_frame_hits"].add(int(frame_id))


def _select_support_overlap_variant(scene_payload: dict[str, Any]) -> dict[str, Any]:
    variants = scene_payload["variants"]
    preferred = [
        "F1_anchor_plus_support_010",
        "R6F1_support005_specificity",
        "R6F2_support010_specificity_semantic",
        "R6F3_support010_specificity_semantic_vetoatten",
        "R6F4_support020_specificity_semantic_vetoatten",
        "R6F5_support010_semantic_gate_strict",
    ]
    for key in preferred:
        if key in variants:
            return variants[key]
    for value in variants.values():
        support_count = value.get("support_support_count")
        if support_count is not None and int(support_count.sum().item()) > 0:
            return value
    return next(iter(variants.values()))


def _mask_level_gt_overlap(
    *,
    acc: dict[int, dict[str, Any]],
    scene: str,
    feature_payload: dict[str, Any],
    d4rt_summary: dict[str, Any],
) -> None:
    scene_payload = feature_payload["scenes"][scene]
    frame_ids = [int(v) for v in d4rt_summary["frame_ids"]]
    mask_root = _project(d4rt_summary["mask_root"])
    mask_frame = scene_payload["mask_frame"].cpu().numpy().astype(np.int64)
    mask_label = scene_payload["mask_label"].cpu().numpy().astype(np.int64)
    is_object = scene_payload["mask_is_object_like"].cpu().numpy().astype(bool)
    is_broad = scene_payload["mask_is_broad"].cpu().numpy().astype(bool)
    variant = _select_support_overlap_variant(scene_payload)
    anchor_count = variant["anchor_support_count"].cpu().numpy().astype(np.int64)
    support_count = variant["support_support_count"].cpu().numpy().astype(np.int64)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, fi in enumerate(mask_frame.tolist()):
        by_frame[int(fi)].append(idx)
    for fi, obs_indices in by_frame.items():
        if fi < 0 or fi >= len(frame_ids):
            continue
        frame_id = frame_ids[fi]
        mask_path = mask_root / f"{int(frame_id)}.png"
        if not mask_path.exists():
            continue
        labels = _load_label_png(mask_path)
        gt = _load_gt_2d(scene, frame_id, labels.shape)
        for obs_idx in obs_indices:
            pixels = labels == int(mask_label[obs_idx])
            if not np.any(pixels):
                continue
            vals = gt[pixels]
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            gt_ids, counts = np.unique(vals, return_counts=True)
            gt_id = int(gt_ids[int(np.argmax(counts))])
            item = acc.setdefault(gt_id, _empty_gt_acc(scene, gt_id))
            if bool(is_object[obs_idx]):
                if int(anchor_count[obs_idx]) > 0:
                    item["A_anchor_object_like_mask_overlap_count"] += 1
                if int(support_count[obs_idx]) > 0:
                    item["S_support_object_like_mask_overlap_count"] += 1
            if bool(is_broad[obs_idx]):
                if int(anchor_count[obs_idx]) > 0:
                    item["A_anchor_broad_overlap_count"] += 1
                if int(support_count[obs_idx]) > 0:
                    item["S_support_broad_overlap_count"] += 1


def _gt_coverage_rows(
    *,
    role_df: pd.DataFrame,
    feature_payload: dict[str, Any],
    d4rt_roots: dict[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, root in sorted(d4rt_roots.items()):
        summary = _read_json(root / "summary.json")
        frame_ids = [int(v) for v in summary["frame_ids"]]
        batch = np.load(root / "carrier_batch.npz")
        carrier_id = np.asarray(batch["carrier_id"], dtype=np.int64)
        a_role, s_role, _v_role = _role_flags(role_df, scene, carrier_id)
        a_idx = np.flatnonzero(a_role)
        s_idx = np.flatnonzero(s_role)
        acc: dict[int, dict[str, Any]] = {}
        for fi, frame_id in enumerate(frame_ids):
            gt = _load_gt_2d(scene, frame_id, (968, 1296))
            gt_ids, gt_counts = np.unique(gt[gt > 0], return_counts=True)
            for gt_id, count in zip(gt_ids.tolist(), gt_counts.tolist()):
                item = acc.setdefault(int(gt_id), _empty_gt_acc(scene, int(gt_id)))
                item["gt_visible_frames"].add(int(frame_id))
                item["gt_pixel_area_sum"] += int(count)
            boundary = _gt_boundary(gt)
            for role_name, indices in [("A_anchor", a_idx), ("S_support", s_idx)]:
                if indices.size == 0:
                    continue
                uv = np.asarray(batch["uv_pred"][fi, indices], dtype=np.float32)
                valid = np.asarray(batch["valid"][fi, indices], dtype=bool)
                finite_uv = uv[np.all(np.isfinite(uv), axis=1)]
                normalized_uv = bool(finite_uv.size and float(np.nanmax(np.abs(finite_uv))) <= 2.0)
                if normalized_uv:
                    x = np.rint(uv[:, 0] * float(gt.shape[1] - 1)).astype(np.int64)
                    y = np.rint(uv[:, 1] * float(gt.shape[0] - 1)).astype(np.int64)
                else:
                    x = np.rint(uv[:, 0]).astype(np.int64)
                    y = np.rint(uv[:, 1]).astype(np.int64)
                ok = valid & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & (x >= 0) & (x < gt.shape[1]) & (y >= 0) & (y < gt.shape[0])
                if not np.any(ok):
                    continue
                labels = gt[y[ok], x[ok]]
                _add_counts(acc, scene, frame_id, labels, role_name, boundary=False)
                boundary_labels = gt[y[ok][boundary[y[ok], x[ok]]], x[ok][boundary[y[ok], x[ok]]]]
                _add_counts(acc, scene, frame_id, boundary_labels, role_name, boundary=True)
        _mask_level_gt_overlap(acc=acc, scene=scene, feature_payload=feature_payload, d4rt_summary=summary)
        areas = np.asarray([item["gt_pixel_area_sum"] for item in acc.values() if item["gt_pixel_area_sum"] > 0], dtype=np.float64)
        q33 = float(np.percentile(areas, 33.3)) if areas.size else 0.0
        q66 = float(np.percentile(areas, 66.6)) if areas.size else 0.0
        for gt_id, item in sorted(acc.items()):
            area = int(item["gt_pixel_area_sum"])
            if area <= q33:
                bucket = "small"
            elif area <= q66:
                bucket = "medium"
            else:
                bucket = "large"
            visible = len(item["gt_visible_frames"])
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_3_gt_object_coverage_row_v1",
                    "phase_id": PHASE_ID,
                    "split": "dev",
                    "scene_id": scene,
                    "window_id": "c0001",
                    "gt_id": int(gt_id),
                    "gt_visible_frame_count": visible,
                    "gt_pixel_area_sum": area,
                    "gt_area_bucket": bucket,
                    "A_anchor_hit": int(item["A_anchor_support_count"] > 0),
                    "S_support_hit": int(item["S_support_support_count"] > 0),
                    "A_anchor_support_count": int(item["A_anchor_support_count"]),
                    "S_support_support_count": int(item["S_support_support_count"]),
                    "A_anchor_frame_hit_rate": float(len(item["A_anchor_frame_hits"]) / max(visible, 1)),
                    "S_support_frame_hit_rate": float(len(item["S_support_frame_hits"]) / max(visible, 1)),
                    "A_anchor_boundary_support_count": int(item["A_anchor_boundary_support_count"]),
                    "S_support_boundary_support_count": int(item["S_support_boundary_support_count"]),
                    "A_anchor_object_like_mask_overlap_count": int(item["A_anchor_object_like_mask_overlap_count"]),
                    "S_support_object_like_mask_overlap_count": int(item["S_support_object_like_mask_overlap_count"]),
                    "A_anchor_broad_overlap_count": int(item["A_anchor_broad_overlap_count"]),
                    "S_support_broad_overlap_count": int(item["S_support_broad_overlap_count"]),
                    "coverage_source": "D4RT carrier projection over window plus mask-level dominant-GT overlap diagnostics",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
    return rows


def _coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    df = pd.DataFrame(rows)
    groups: list[tuple[str, pd.DataFrame]] = [("all", df)]
    for scene, sub in df.groupby("scene_id", sort=True):
        groups.append((f"scene={scene}", sub))
    for bucket, sub in df.groupby("gt_area_bucket", sort=True):
        groups.append((f"area_bucket={bucket}", sub))
    for key, sub in groups:
        out.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_3_gt_object_coverage_summary_row_v1",
                "phase_id": PHASE_ID,
                "split": "dev",
                "group_key": key,
                "GT_object_count": int(len(sub)),
                "A_anchor_hit_rate": float(sub["A_anchor_hit"].mean()) if len(sub) else 0.0,
                "S_support_hit_rate": float(sub["S_support_hit"].mean()) if len(sub) else 0.0,
                "A_or_S_hit_rate": float(((sub["A_anchor_hit"].astype(int) + sub["S_support_hit"].astype(int)) > 0).mean()) if len(sub) else 0.0,
                "A_anchor_frame_hit_rate_mean": float(sub["A_anchor_frame_hit_rate"].mean()) if len(sub) else 0.0,
                "S_support_frame_hit_rate_mean": float(sub["S_support_frame_hit_rate"].mean()) if len(sub) else 0.0,
                "small_object_S_support_hit_rate": _bucket_rate(df, "small", "S_support_hit"),
                "medium_object_S_support_hit_rate": _bucket_rate(df, "medium", "S_support_hit"),
                "large_object_S_support_hit_rate": _bucket_rate(df, "large", "S_support_hit"),
                "GT_object_boundary_A_anchor_support_p10": _p10(sub["A_anchor_boundary_support_count"]),
                "GT_object_boundary_S_support_support_p10": _p10(sub["S_support_boundary_support_count"]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    return out


def _bucket_rate(df: pd.DataFrame, bucket: str, col: str) -> float:
    sub = df[df["gt_area_bucket"].astype(str) == bucket]
    return float(sub[col].mean()) if len(sub) else 0.0


def _p10(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 10)) if arr.size else 0.0


def _prediction_sources(local_ap_root: Path, anchor_only_root: Path, current_root: Path) -> list[dict[str, Any]]:
    sources = [
        {
            "prediction_family": "current_locked_replay",
            "r5_feature_variant_id": "current_phase6d_locked",
            "phase6d_variant_id": D0_VARIANT,
            "root": current_root,
        },
        {
            "prediction_family": "current_locked_d9",
            "r5_feature_variant_id": "current_phase6d_locked",
            "phase6d_variant_id": D9_VARIANT,
            "root": current_root,
        },
    ]
    for root, family in [(anchor_only_root, "anchor_only_d9"), (local_ap_root, "support_weighted_d9")]:
        run_parent = root / "phase6d_runs"
        if not run_parent.exists():
            continue
        for child in sorted(run_parent.iterdir()):
            if child.is_dir():
                sources.append(
                    {
                        "prediction_family": family,
                        "r5_feature_variant_id": child.name,
                        "phase6d_variant_id": D9_VARIANT,
                        "root": child,
                    }
                )
    return sources


def _pair_stats(mask_items: list[dict[str, Any]], gt_id: int) -> dict[str, float]:
    same = [m for m in mask_items if int(m.get("dominant_gt", 0)) == int(gt_id)]
    same_pair_count = len(same) * (len(same) - 1) // 2
    connected = 0
    for a, b in combinations(same, 2):
        connected += int(str(a["mv_object_id"]) == str(b["mv_object_id"]))
    diff_total = 0
    diff_connected = 0
    diff = [m for m in mask_items if int(m.get("dominant_gt", 0)) > 0 and int(m.get("dominant_gt", 0)) != int(gt_id)]
    for a in same:
        for b in diff:
            diff_total += 1
            diff_connected += int(str(a["mv_object_id"]) == str(b["mv_object_id"]))
    return {
        "same_gt_mask_pair_count": float(same_pair_count),
        "same_gt_mask_pair_connected_count": float(connected),
        "same_gt_mask_pair_connection_rate": float(connected / max(same_pair_count, 1)),
        "same_semantic_diff_gt_false_connection_count": float(diff_connected),
        "same_semantic_diff_gt_false_connection_rate": float(diff_connected / max(diff_total, 1)),
    }


def _build_prediction_iou(
    *,
    source: dict[str, Any],
    d4rt_roots: dict[str, Path],
    min_pred_pixels: int,
    min_gt_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(source["root"])
    selected = _read_csv(root / "merge_selected_rows.csv")
    if selected.empty:
        return [], []
    selected = selected[selected["variant_id"].astype(str) == str(source["phase6d_variant_id"])]
    rows: list[dict[str, Any]] = []
    mask_items_all: list[dict[str, Any]] = []
    for scene, scene_rows in selected.groupby("scene_id", sort=True):
        d4rt_summary = _read_json(d4rt_roots[str(scene)] / "summary.json")
        frame_ids = [int(v) for v in d4rt_summary["frame_ids"]]
        mask_root = _project(d4rt_summary["mask_root"])
        acc = SparseSceneIoU()
        object_index: dict[str, int] = {}
        mask_items: list[dict[str, Any]] = []
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for _, rec in scene_rows.iterrows():
            data = rec.to_dict()
            local = int(_num(data.get("frame_local_index"), -1))
            if 0 <= local < len(frame_ids):
                data["frame_id"] = frame_ids[local]
                by_frame[frame_ids[local]].append(data)
        for frame_id in frame_ids:
            mask_path = mask_root / f"{int(frame_id)}.png"
            if not mask_path.exists():
                gt = _load_gt_2d(str(scene), frame_id, (968, 1296))
                acc.add(np.zeros(gt.shape, dtype=np.int64), gt)
                continue
            labels = _load_label_png(mask_path)
            gt = _load_gt_2d(str(scene), frame_id, labels.shape)
            pred = np.zeros(labels.shape, dtype=np.int64)
            for item in sorted(by_frame.get(frame_id, []), key=lambda r: (-_num(r.get("object_score"), 0.0), str(r.get("mv_object_id", "")))):
                oid = str(item.get("mv_object_id", ""))
                if oid not in object_index:
                    object_index[oid] = len(object_index) + 1
                mid = int(_num(item.get("selected_mask_id"), -1))
                pixels = labels == mid
                if not np.any(pixels):
                    continue
                vals = gt[pixels]
                vals = vals[vals > 0]
                dominant_gt = 0
                purity = 0.0
                if vals.size:
                    gt_ids, counts = np.unique(vals, return_counts=True)
                    best_idx = int(np.argmax(counts))
                    dominant_gt = int(gt_ids[best_idx])
                    purity = float(counts[best_idx] / max(np.count_nonzero(pixels), 1))
                pred[(pred == 0) & pixels] = object_index[oid]
                mask_items.append(
                    {
                        "scene_id": str(scene),
                        "frame_id": int(frame_id),
                        "selected_mask_id": int(mid),
                        "mv_object_id": oid,
                        "pred_id": int(object_index[oid]),
                        "dominant_gt": int(dominant_gt),
                        "dominant_gt_purity": purity,
                    }
                )
            acc.add(pred, gt)
        built = acc.build(min_pred_pixels=min_pred_pixels, min_gt_pixels=min_gt_pixels)
        iou = np.asarray(built["iou"], dtype=np.float32)
        inter = np.asarray(built["intersection"], dtype=np.float32)
        pred_ids = list(built["pred_ids"])
        gt_ids = list(built["gt_ids"])
        pred_area = np.asarray(built["pred_area"], dtype=np.float64)
        gt_area = np.asarray(built["gt_area"], dtype=np.float64)
        for tau in [0.0, 0.05]:
            for gidx, gt_id in enumerate(gt_ids):
                touching = np.flatnonzero(iou[:, gidx] > float(tau)) if iou.shape[0] else np.zeros((0,), dtype=np.int64)
                best_iou = float(np.max(iou[:, gidx])) if iou.shape[0] else 0.0
                union_iou = 0.0
                pred_touch_ids: list[str] = []
                if touching.size:
                    inter_sum = float(np.sum(inter[touching, gidx]))
                    pred_sum = float(np.sum(pred_area[touching]))
                    denom = pred_sum + float(gt_area[gidx]) - inter_sum
                    union_iou = float(inter_sum / denom) if denom > 0 else 0.0
                    pred_touch_ids = [str(pred_ids[int(idx)]) for idx in touching.tolist()]
                pair = _pair_stats(mask_items, int(gt_id))
                rows.append(
                    {
                        "schema_version": "stream4d_v103_supp_r5_phaseR5_3_three_d_inconsistency_row_v1",
                        "phase_id": PHASE_ID,
                        "split": "dev",
                        "scene_id": str(scene),
                        "window_id": "c0001",
                        "gt_id": int(gt_id),
                        "tau": float(tau),
                        "prediction_family": source["prediction_family"],
                        "r5_feature_variant_id": source["r5_feature_variant_id"],
                        "phase6d_variant_id": source["phase6d_variant_id"],
                        "fragment_count": int(touching.size),
                        "fragment_count_ge2": int(touching.size >= 2),
                        "fragment_count_ge3": int(touching.size >= 3),
                        "best_pred_iou": best_iou,
                        "union_pred_iou": union_iou,
                        "union_minus_best_iou": union_iou - best_iou,
                        "pred_object_ids_touching_gt": ";".join(pred_touch_ids),
                        "same_gt_mask_pair_count": int(pair["same_gt_mask_pair_count"]),
                        "same_gt_mask_pair_connected_count": int(pair["same_gt_mask_pair_connected_count"]),
                        "same_gt_mask_pair_connection_rate": pair["same_gt_mask_pair_connection_rate"],
                        "same_semantic_diff_gt_false_connection_count": int(pair["same_semantic_diff_gt_false_connection_count"]),
                        "same_semantic_diff_gt_false_connection_rate": pair["same_semantic_diff_gt_false_connection_rate"],
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_eval": True,
                        "uses_future": False,
                    }
                )
        mask_items_all.extend(mask_items)
    return rows, mask_items_all


def _inconsistency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for (family, r5_variant, phase6d_variant, scene, tau), sub in df.groupby(
        ["prediction_family", "r5_feature_variant_id", "phase6d_variant_id", "scene_id", "tau"], sort=True
    ):
        out.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_3_three_d_inconsistency_summary_row_v1",
                "phase_id": PHASE_ID,
                "split": "dev",
                "prediction_family": family,
                "r5_feature_variant_id": r5_variant,
                "phase6d_variant_id": phase6d_variant,
                "scene_id": scene,
                "tau": float(tau),
                "GT_object_count": int(len(sub)),
                "GT_fragment_count_mean": float(sub["fragment_count"].mean()) if len(sub) else 0.0,
                "GT_fragment_count_ge2_rate": float(sub["fragment_count_ge2"].mean()) if len(sub) else 0.0,
                "GT_fragment_count_ge3_rate": float(sub["fragment_count_ge3"].mean()) if len(sub) else 0.0,
                "GT_fragment_count_p90": float(np.percentile(sub["fragment_count"].to_numpy(dtype=np.float64), 90)) if len(sub) else 0.0,
                "best_pred_IoU_mean": float(sub["best_pred_iou"].mean()) if len(sub) else 0.0,
                "union_pred_IoU_mean": float(sub["union_pred_iou"].mean()) if len(sub) else 0.0,
                "union_minus_best_IoU_mean": float(sub["union_minus_best_iou"].mean()) if len(sub) else 0.0,
                "same_GT_mask_pair_connection_rate": float(sub["same_gt_mask_pair_connection_rate"].mean()) if len(sub) else 0.0,
                "same_semantic_diff_GT_false_connection_rate": float(sub["same_semantic_diff_gt_false_connection_rate"].mean()) if len(sub) else 0.0,
                "same_frame_competing_connection_count": "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)

    feature_root = _project(args.feature_root)
    local_ap_root = _project(args.local_ap_root)
    anchor_root = _project(args.anchor_only_root)
    current_root = _project(args.current_phase6d_root)
    phases1_root = _project(args.phaseS1_root)
    d4rt_roots = {
        "scene0011_00": _project(args.scene0011_d4rt_root),
        "scene0050_00": _project(args.scene0050_d4rt_root),
    }

    role_df = pd.read_parquet(phases1_root / "carrier_role_rows.parquet")
    feature_payload = torch.load(feature_root / "role_mask_level_feature.pt", map_location="cpu", weights_only=False)
    coverage_rows = _gt_coverage_rows(role_df=role_df, feature_payload=feature_payload, d4rt_roots=d4rt_roots)
    coverage_summary_rows = _coverage_summary(coverage_rows)

    inconsistency_rows: list[dict[str, Any]] = []
    mask_pair_rows: list[dict[str, Any]] = []
    for source in _prediction_sources(local_ap_root, anchor_root, current_root):
        rows, masks = _build_prediction_iou(
            source=source,
            d4rt_roots=d4rt_roots,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
        )
        inconsistency_rows.extend(rows)
        for item in masks:
            mask_pair_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_3_mask_dominant_gt_row_v1",
                    "phase_id": PHASE_ID,
                    **source,
                    "root": _rel(source["root"]),
                    **item,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
    inconsistency_summary_rows = _inconsistency_summary(inconsistency_rows)

    failure_rows: list[dict[str, Any]] = []
    cov_all = next((r for r in coverage_summary_rows if r["group_key"] == "all"), {})
    if _num(cov_all.get("S_support_hit_rate"), 0.0) >= 0.95 and _num(cov_all.get("A_anchor_hit_rate"), 0.0) < _num(cov_all.get("S_support_hit_rate"), 0.0):
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_3_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_COVERAGE_NOT_USED",
                "detail": (
                    f"S_support_hit_rate={cov_all.get('S_support_hit_rate')} "
                    f"A_anchor_hit_rate={cov_all.get('A_anchor_hit_rate')}"
                ),
                "repair_direction": "Support covers GT objects, but R5-4 local AP gate failed; keep support as guarded compatibility/veto/ranking evidence.",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    support_false = [
        r
        for r in inconsistency_summary_rows
        if str(r.get("prediction_family")) == "support_weighted_d9"
        and float(r.get("tau", 0.0)) == 0.05
        and _num(r.get("same_semantic_diff_GT_false_connection_rate"), 0.0) > 0.0
    ]
    if support_false:
        max_false = max(_num(r.get("same_semantic_diff_GT_false_connection_rate"), 0.0) for r in support_false)
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_3_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_EDGE_FALSE_BRIDGE_DIAGNOSTIC_NONZERO",
                "detail": f"max_support_weighted_same_semantic_diff_gt_false_connection_rate_tau005={max_false}",
                "repair_direction": "Do not allow support-only unions; require stronger semantic/veto checks before any future merge family.",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )

    summary = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_3_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "DIAGNOSTIC_ONLY_R5_4_GATE_ALREADY_FAILED",
        "phase_r5_3_diag_complete": True,
        "gt_object_coverage_row_count": len(coverage_rows),
        "gt_object_coverage_summary_row_count": len(coverage_summary_rows),
        "three_d_inconsistency_row_count": len(inconsistency_rows),
        "three_d_inconsistency_summary_row_count": len(inconsistency_summary_rows),
        "failure_count": len(failure_rows),
        "coverage_all": cov_all,
        "outputs": {
            "gt_object_coverage_rows": _rel(out / "gt_object_coverage_rows.csv"),
            "gt_object_coverage_summary_rows": _rel(out / "gt_object_coverage_summary_rows.csv"),
            "three_d_inconsistency_rows": _rel(out / "three_d_inconsistency_rows.csv"),
            "three_d_inconsistency_summary_rows": _rel(out / "three_d_inconsistency_summary_rows.csv"),
            "mask_dominant_gt_rows": _rel(out / "mask_dominant_gt_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "truthfulness_note": "R5-3 is GT-only diagnostic. It is not used to pick thresholds or variants, and it does not alter method predictions.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
    }

    _write_csv(out / "gt_object_coverage_rows.csv", coverage_rows)
    _write_csv(out / "gt_object_coverage_summary_rows.csv", coverage_summary_rows)
    _write_csv(out / "three_d_inconsistency_rows.csv", inconsistency_rows)
    _write_csv(out / "three_d_inconsistency_summary_rows.csv", inconsistency_summary_rows)
    _write_csv(out / "mask_dominant_gt_rows.csv", mask_pair_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--local-ap-root", default=str(DEFAULT_LOCAL_AP_ROOT))
    parser.add_argument("--anchor-only-root", default=str(DEFAULT_ANCHOR_ONLY_ROOT))
    parser.add_argument("--current-phase6d-root", default=str(DEFAULT_CURRENT_PHASE6D_ROOT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
