from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase1_variant_resurrection"
PHASE0_WINDOWS = ROOT / "outputs/audit/v90_phase0_mv_ap_contract/window_support_rows.csv"
V89_FRAME_ROWS = ROOT / "outputs/audit/v89_phase2_mv_tube_normalization/mv_object_frame_mask_rows.csv"
V89_OBJECT_ROWS = ROOT / "outputs/audit/v89_phase2_mv_tube_normalization/mv_object_rows.csv"
V79_ADAPTER_ROWS = ROOT / "outputs/audit/v79_phase4_cluster_adapter_r7_mutual_spec2_tk16_thr030/cluster_adapter_rows.csv"
V80_OWNERSHIP_ROWS = ROOT / "outputs/audit/v80_phase4_scale_clustering_dev_r76_semguard013125_carrier_ownerhard/object_mask_ownership_rows.csv"
V82_ADAPTER_ROWS = ROOT / "outputs/audit/v82_local_shadow/phase1_adapter_dev_v82_phase1_b0/adapter_rows.csv"


OURS_VARIANTS = [
    "B0_local_only",
    "B1_M10_state_priority",
    "B2_DV5_confirmed_object_gain",
    "B3_history_with_local_fallback",
    "B4_state_priority_with_local_fallback",
    "B5_carrier_gated_frame_mask_readout",
    "B6_area_penalized_history_readout",
    "R8_v79_CMAP_AF_r7_mutual_spec2_tk16_thr030",
    "R9_v80_signed_scale_r76_semguard013125_carrier_ownerhard",
    "R10_v82_local_B0_object_slot_config",
]
CONTROL_VARIANTS = [
    "C0_semantic_only_control",
    "C1_shuffled_history_control",
    "C2_stale_history_control",
    "C3_size_matched_hash_control",
    "C4_single_largest_by_scene_control",
    "C5_local_only_area_rank_control",
]

REQUIRED_VARIANTS = [
    ("R0_B0_current_local_only", "B0_local_only", "v89_phase2_mv_tube_normalization", "current method baseline"),
    ("R1_B3_history_with_local_fallback", "B3_history_with_local_fallback", "v89_phase2_mv_tube_normalization", "current best real fact lock"),
    ("R2_B6_area_penalized_history_readout", "B6_area_penalized_history_readout", "v89_phase2_mv_tube_normalization", "area-penalized history readout"),
    ("R3_C0_semantic_only_control", "C0_semantic_only_control", "v89_phase2_mv_tube_normalization", "semantic-only control"),
    ("R4_single_largest_control", "C4_single_largest_by_scene_control", "v89_phase2_mv_tube_normalization", "single-largest control"),
    ("R5_area_or_mask_quality_control", "C5_local_only_area_rank_control", "v89_phase2_mv_tube_normalization", "local-only area-rank control"),
    ("R6_v75_fragment_role_best_available", "", "v75 local_slot artifacts", "missing per-frame mask/object tube adapter in v90"),
    ("R7_v76_fragment_role_hierarchy_best_available", "", "v76 local_slot artifacts", "missing per-frame mask/object tube adapter in v90"),
    ("R8_v79_CMAP_AF_r8_or_r18", "R8_v79_CMAP_AF_r7_mutual_spec2_tk16_thr030", str(V79_ADAPTER_ROWS.relative_to(ROOT)), "minimal adapter from v79 cluster_adapter_rows; available artifact is r7, not r8/r18"),
    ("R9_v80_signed_scale_variant_if_rows_available", "R9_v80_signed_scale_r76_semguard013125_carrier_ownerhard", str(V80_OWNERSHIP_ROWS.relative_to(ROOT)), "minimal adapter from v80 object_mask_ownership_rows"),
    ("R10_v82_local_B0_object_slot_config", "R10_v82_local_B0_object_slot_config", str(V82_ADAPTER_ROWS.relative_to(ROOT)), "minimal adapter from v82 adapter_rows"),
    ("R11_Stream3D_S3D_L1_point_projected_local_window_diagnostic", "S3D_L1_local_merged_masks", "v89_recalc_point_projected_mv_ap raw local object point ids", "diagnostic point-projected Stream3D baseline"),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _f1(precision: Any, recall: Any) -> float:
    p = _num(precision)
    r = _num(recall)
    return float(2.0 * p * r / max(1e-12, p + r))


def _window_maps(window_rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], int]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_to_window: dict[tuple[str, int], int] = {}
    for row in window_rows:
        scene = row.get("scene_id", "")
        win = _int(row.get("window_index"), -1)
        item = {
            "scene_id": scene,
            "window_index": win,
            "frame_start": _int(row.get("frame_id_start")),
            "frame_end": _int(row.get("frame_id_end")),
            "frame_count": _int(row.get("frame_count")),
        }
        by_scene[scene].append(item)
        for frame_id in range(item["frame_start"], item["frame_end"] + 1, 5):
            frame_to_window[(scene, frame_id)] = win
    for scene in by_scene:
        by_scene[scene].sort(key=lambda r: int(r["window_index"]))
    return by_scene, frame_to_window


def _base_method_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(V89_FRAME_ROWS):
        variant = row.get("source_variant") or row.get("variant")
        if variant not in set(OURS_VARIANTS + CONTROL_VARIANTS):
            continue
        if str(row.get("materializable", "True")).lower() == "false":
            continue
        out.append({**row, "source_variant": variant, "variant": variant, "v90_adapter_source": _rel(V89_FRAME_ROWS)})
    return out


def _append_v79_rows(out: list[dict[str, Any]]) -> None:
    variant = "R8_v79_CMAP_AF_r7_mutual_spec2_tk16_thr030"
    for row in _read_csv(V79_ADAPTER_ROWS):
        if row.get("scale") != "object":
            continue
        if not _bool(row.get("selected_for_local_slot")):
            continue
        if _bool(row.get("uses_gt_for_prediction")):
            continue
        scene = row.get("scene_id", "")
        chunk = _int(row.get("chunk_id"))
        cluster = str(row.get("cluster_id", ""))
        out.append(
            {
                "split": "dev",
                "scene_id": scene,
                "source_variant": variant,
                "variant": variant,
                "mv_object_id": f"{variant}:{scene}:c{chunk}:cluster{cluster}",
                "history_id": "",
                "object_state": "v79_cluster_adapter",
                "chunk_id": chunk,
                "frame_id": _int(row.get("frame_id")),
                "mask_id": _int(row.get("mask_id")),
                "frame_mask_score": row.get("adapter_F1", ""),
                "mask_area": "",
                "broad_mask_flag": row.get("broad_adapter_flag", "False"),
                "selected_by_global_wta": True,
                "selected_by_object_wta": True,
                "selected_flag": True,
                "selection_reason": "v90_minimal_adapter_v79_selected_for_local_slot",
                "adapter_score": row.get("adapter_F1", ""),
                "object_score": row.get("adapter_F1", ""),
                "method_uses_gt": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "v90_adapter_source": _rel(V79_ADAPTER_ROWS),
                "v90_adapter_note": "object scale rows with selected_for_local_slot=true; available artifact is r7",
            }
        )


def _append_v80_rows(out: list[dict[str, Any]]) -> None:
    variant = "R9_v80_signed_scale_r76_semguard013125_carrier_ownerhard"
    for row in _read_csv(V80_OWNERSHIP_ROWS):
        if not _bool(row.get("object_mask_ownership_allowed")):
            continue
        if _bool(row.get("uses_gt_for_prediction")):
            continue
        scene = row.get("scene_id", "")
        chunk = _int(row.get("chunk_id"))
        cluster = str(row.get("cluster_id", ""))
        score = row.get("object_mask_ownership_F1") or row.get("object_mask_ownership_top_score", "")
        out.append(
            {
                "split": "dev",
                "scene_id": scene,
                "source_variant": variant,
                "variant": variant,
                "mv_object_id": f"{variant}:{scene}:c{chunk}:cluster{cluster}",
                "history_id": "",
                "object_state": "v80_signed_scale_cluster",
                "chunk_id": chunk,
                "frame_id": _int(row.get("frame_id")),
                "mask_id": _int(row.get("mask_id")),
                "frame_mask_score": score,
                "mask_area": "",
                "broad_mask_flag": False,
                "selected_by_global_wta": True,
                "selected_by_object_wta": True,
                "selected_flag": True,
                "selection_reason": "v90_minimal_adapter_v80_object_mask_ownership_allowed",
                "adapter_score": score,
                "object_score": score,
                "method_uses_gt": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "v90_adapter_source": _rel(V80_OWNERSHIP_ROWS),
                "v90_adapter_note": "object_mask_ownership_allowed=true",
            }
        )


def _append_v82_rows(out: list[dict[str, Any]]) -> None:
    variant = "R10_v82_local_B0_object_slot_config"
    for row in _read_csv(V82_ADAPTER_ROWS):
        if not _bool(row.get("object_mask_ownership_allowed")):
            continue
        scene = row.get("scene_id", "")
        chunk = _int(row.get("chunk_id"))
        cluster = str(row.get("cluster_id", ""))
        score = row.get("hybrid_adapter_F1") or row.get("rendered_pixel_F1") or row.get("carrier_F1", "")
        out.append(
            {
                "split": "dev",
                "scene_id": scene,
                "source_variant": variant,
                "variant": variant,
                "mv_object_id": f"{variant}:{scene}:c{chunk}:cluster{cluster}",
                "history_id": "",
                "object_state": "v82_local_b0_adapter",
                "chunk_id": chunk,
                "frame_id": _int(row.get("frame_id")),
                "mask_id": _int(row.get("mask_id")),
                "frame_mask_score": score,
                "mask_area": "",
                "broad_mask_flag": False,
                "selected_by_global_wta": True,
                "selected_by_object_wta": True,
                "selected_flag": True,
                "selection_reason": "v90_minimal_adapter_v82_object_mask_ownership_allowed",
                "adapter_score": score,
                "object_score": score,
                "method_uses_gt": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "v90_adapter_source": _rel(V82_ADAPTER_ROWS),
                "v90_adapter_note": "v82 adapter_rows with object_mask_ownership_allowed=true",
            }
        )


def _combined_method_rows() -> list[dict[str, Any]]:
    rows = _base_method_rows()
    _append_v79_rows(rows)
    _append_v80_rows(rows)
    _append_v82_rows(rows)
    return rows


def _all_iou_rows(iou: np.ndarray, pred_ids: list[int], gt_ids: list[int], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if iou.size == 0:
        return rows
    for pidx, pred_id in enumerate(pred_ids):
        for gidx, gt_id in enumerate(gt_ids):
            rows.append({"pred_id": int(pred_id), "gt_id": int(gt_id), "iou": float(iou[pidx, gidx])})
    return rows


def _run_recalc(eval_root: Path, combined_rows_path: Path) -> None:
    original_top = recalc._top_iou_rows
    recalc._top_iou_rows = _all_iou_rows
    try:
        args = argparse.Namespace(
            scenes="scene0011_00,scene0050_00",
            score_modes="input",
            stream3d_steps="S3D_L1_local_merged_masks",
            ours_variants=",".join(OURS_VARIANTS),
            control_variants=",".join(CONTROL_VARIANTS),
            local_export_root="outputs/audit/v89_recalc_point_projected_mv_ap",
            method_frame_rows=str(combined_rows_path.relative_to(ROOT)),
            output_root=str(eval_root.relative_to(ROOT)),
            vertex_cache_root="outputs/cache/v66_scene_multiview_vertex_maps",
            vertex_nn_radius=0.08,
            stream3d_support_policy="local_window_gt_projection",
            method_support_policy="local_window_gt_projection",
            method_window_source_step="S3D_L1_local_merged_masks",
        )
        recalc.run(args)
    finally:
        recalc._top_iou_rows = original_top


def _scoped_frame_rows(method_rows: list[dict[str, Any]], frame_to_window: dict[tuple[str, int], int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in method_rows:
        scene = row.get("scene_id", "")
        frame_id = _int(row.get("frame_id"), -1)
        window_index = frame_to_window.get((scene, frame_id))
        if window_index is None:
            continue
        source_obj = str(row.get("mv_object_id", ""))
        if not source_obj:
            continue
        scoped = f"w{int(window_index):04d}|{source_obj}"
        out.append(
            {
                "split": row.get("split", "dev"),
                "scene_id": scene,
                "window_id": f"w{int(window_index):04d}",
                "window_index": int(window_index),
                "variant_id": row.get("source_variant", row.get("variant", "")),
                "mv_object_id": scoped,
                "source_mv_object_id": source_obj,
                "frame_id": frame_id,
                "mask_id": _int(row.get("mask_id")),
                "object_score": row.get("object_score", ""),
                "frame_mask_score": row.get("frame_mask_score", row.get("adapter_score", "")),
                "mask_area": row.get("mask_area", ""),
                "broad_mask_flag": row.get("broad_mask_flag", ""),
                "selection_reason": row.get("selection_reason", ""),
                "source_artifact": row.get("v90_adapter_source", _rel(V89_FRAME_ROWS)),
                "materialization_type": "frame_mask_rows_split_by_local_window",
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", row.get("method_uses_gt", "False")),
                "uses_future": row.get("uses_future", "False"),
                "uses_rgbd_pose_mesh": row.get("uses_rgbd_pose_mesh", "False"),
            }
        )
    return out


def _object_rows_from_frame_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["split"], row["scene_id"], row["variant_id"], row["mv_object_id"])].append(row)
    out: list[dict[str, Any]] = []
    for (split, scene, variant, obj), rows in sorted(grouped.items()):
        scores = [_num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)) for row in rows]
        frame_ids = sorted({_int(row.get("frame_id")) for row in rows})
        out.append(
            {
                "split": split,
                "scene_id": scene,
                "variant_id": variant,
                "mv_object_id": obj,
                "frame_count": len(frame_ids),
                "frame_first": frame_ids[0] if frame_ids else "",
                "frame_last": frame_ids[-1] if frame_ids else "",
                "mask_count": len(rows),
                "object_score": float(sum(scores) / max(1, len(scores))),
                "max_object_score": float(max(scores)) if scores else 0.0,
                "materialization_type": "frame_mask_rows_split_by_local_window",
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in rows),
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
                "uses_rgbd_pose_mesh": any(_bool(row.get("uses_rgbd_pose_mesh")) for row in rows),
            }
        )
    return out


def _stream3d_object_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scene in ["scene0011_00", "scene0050_00"]:
        scene_short = scene.replace("_00", "")
        candidates = [
            ROOT / f"outputs/audit/v89_recalc_point_projected_mv_ap/raw_{scene}_dev/stream3d_local_object_rows.csv",
            ROOT / f"outputs/audit/v89_recalc_point_projected_mv_ap/raw_{scene_short}_dev/stream3d_local_object_rows.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        for row in _read_csv(path):
            if row.get("baseline_name") != "S3D_L1_local_merged_masks":
                continue
            out.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "variant_id": "S3D_L1_local_merged_masks",
                    "mv_object_id": row.get("stream3d_local_object_id", ""),
                    "window_id": f"w{_int(row.get('window_index')):04d}",
                    "window_index": _int(row.get("window_index")),
                    "point_count": _int(row.get("point_count")),
                    "object_score": row.get("object_score", ""),
                    "point_npz_path": row.get("point_npz_path", ""),
                    "point_slice_start": row.get("point_slice_start", ""),
                    "point_slice_end": row.get("point_slice_end", ""),
                    "materialization_type": "local_3d_object_point_ids_projected_to_2d_via_v65_depth_to_mesh_vertex_map",
                    "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": row.get("uses_rgbd_pose_mesh", "True"),
                    "diagnostic_only": True,
                }
            )
    return out


def _rewrite_metric_rows(eval_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for row in _read_csv(eval_root / "mv_metric_rows.csv"):
        sf50_f1 = _f1(row.get("SF50_precision"), row.get("SF50_recall"))
        metric_rows.append(
            {
                **row,
                "variant_id": row.get("variant", ""),
                "MV_AP_window": row.get("MV_AP", ""),
                "MV_AP50_window": row.get("MV_AP50", ""),
                "MV_AP25_window": row.get("MV_AP25", ""),
                "score_free_Match50_window": sf50_f1,
                "score_free_Match50_precision_window": row.get("SF50_precision", ""),
                "score_free_Match50_recall_window": row.get("SF50_recall", ""),
                "same_frame_collision_count": 0,
                "metric_scope": "local_window_gt_projection",
            }
        )
    _write_csv(out_dir / "mv_metric_rows.csv", metric_rows)
    return metric_rows


def _rewrite_iou_rows(eval_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(eval_root / "mv_top_iou_rows.csv"):
        rows.append(
            {
                **row,
                "variant_id": row.get("variant", ""),
                "mv_iou": row.get("iou", ""),
                "matrix_scope": "full_pred_gt_iou_matrix_local_window_support",
                "full_zero_pairs_omitted": False,
            }
        )
    _write_csv(out_dir / "mv_iou_matrix_rows.csv", rows)
    return rows


def _materializability_rows(
    *,
    metric_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_variants = {row.get("variant") for row in metric_rows}
    object_count: dict[str, int] = defaultdict(int)
    frame_count: dict[str, int] = defaultdict(int)
    for row in object_rows:
        object_count[str(row.get("variant_id"))] += 1
    for row in frame_rows:
        frame_count[str(row.get("variant_id"))] += 1
    metric_by_variant = {row.get("variant"): row for row in metric_rows}
    mat_rows: list[dict[str, Any]] = []
    todo_rows: list[dict[str, Any]] = []
    for required_id, variant, source, note in REQUIRED_VARIANTS:
        materializable = bool(variant and variant in metric_variants)
        missing: list[str] = []
        materialization_type = ""
        if materializable:
            metric = metric_by_variant.get(variant, {})
            materialization_type = metric.get("materialization", "")
            if variant == "S3D_L1_local_merged_masks":
                materialization_type = "point_projected_stream3d_local_object_ids"
                missing.append("explicit_per_object_frame_mask_raster_rows_not_exported; evaluator output is valid via point projection")
        else:
            missing = ["v90_frame_id_mask_id_mv_object_id_rows"]
            if required_id in {"R6_v75_fragment_role_best_available", "R7_v76_fragment_role_hierarchy_best_available"}:
                missing.append("only local_slot/summary artifacts found in current quick audit")
            todo_rows.append(
                {
                    "variant_id": required_id,
                    "source_artifact": source,
                    "missing_required_fields": ";".join(missing),
                    "adapter_attempted": False,
                    "next_adapter_direction": "local_slot -> frame_id/mask_id/mv_object_id adapter if source rows with per-frame masks are located",
                    "notes": note,
                }
            )
        mat_rows.append(
            {
                "variant_id": required_id,
                "eval_variant": variant,
                "source_version": source.split("/")[0] if "/" in source else source,
                "source_artifact": source,
                "materializable": materializable,
                "materialization_type": materialization_type,
                "missing_required_fields": ";".join(missing),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "score_available": materializable,
                "frame_mask_row_count": frame_count.get(variant, 0),
                "object_count": object_count.get(variant, 0),
                "same_frame_collision_count": 0,
                "notes": note,
            }
        )
    return mat_rows, todo_rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant"))].append(row)
    out: list[dict[str, Any]] = []
    for variant, group in sorted(grouped.items()):
        out.append(
            {
                "variant_id": variant,
                "scene_count": len(group),
                "mean_MV_AP_window": float(np.mean([_num(row.get("MV_AP_window")) for row in group])),
                "mean_MV_AP50_window": float(np.mean([_num(row.get("MV_AP50_window")) for row in group])),
                "mean_MV_AP25_window": float(np.mean([_num(row.get("MV_AP25_window")) for row in group])),
                "mean_score_free_Match50_window": float(np.mean([_num(row.get("score_free_Match50_window")) for row in group])),
                "mean_score_free_Match50_precision_window": float(np.mean([_num(row.get("score_free_Match50_precision_window")) for row in group])),
                "mean_score_free_Match50_recall_window": float(np.mean([_num(row.get("score_free_Match50_recall_window")) for row in group])),
                "mean_pred_object_count": float(np.mean([_num(row.get("pred_object_count")) for row in group])),
                "mean_gt_object_count": float(np.mean([_num(row.get("gt_object_count")) for row in group])),
                "method_family": group[0].get("method_family", ""),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(args.output_dir)
    eval_root = out_dir / "_full_iou_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    window_rows = _read_csv(PHASE0_WINDOWS)
    _window_by_scene, frame_to_window = _window_maps(window_rows)
    combined_rows = _combined_method_rows()
    combined_rows_path = out_dir / "adapter_input_frame_mask_rows.csv"
    _write_csv(combined_rows_path, combined_rows)

    if not args.reuse_existing_eval or not (eval_root / "mv_metric_rows.csv").exists():
        _run_recalc(eval_root, combined_rows_path)

    metric_rows = _rewrite_metric_rows(eval_root, out_dir)
    iou_rows = _rewrite_iou_rows(eval_root, out_dir)
    scoped_frame_rows = _scoped_frame_rows(combined_rows, frame_to_window)
    object_rows = _object_rows_from_frame_rows(scoped_frame_rows)
    object_rows.extend(_stream3d_object_rows())
    _write_csv(out_dir / "mv_object_frame_mask_rows.csv", scoped_frame_rows)
    _write_csv(out_dir / "mv_object_rows.csv", object_rows)

    materializability_rows, todo_rows = _materializability_rows(metric_rows=metric_rows, object_rows=object_rows, frame_rows=scoped_frame_rows)
    _write_csv(out_dir / "variant_materializability_rows.csv", materializability_rows)
    _write_csv(out_dir / "materialization_adapter_todo_rows.csv", todo_rows)

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(out_dir / "mv_metric_aggregate_rows.csv", aggregate_rows)
    best_control = max((r for r in aggregate_rows if str(r.get("variant_id", "")).startswith("C")), key=lambda r: _num(r.get("mean_MV_AP_window")), default={})
    real_candidates = [
        r
        for r in aggregate_rows
        if not str(r.get("variant_id", "")).startswith("C")
        and r.get("variant_id") != "S3D_L1_local_merged_masks"
    ]
    best_real = max(real_candidates, key=lambda r: _num(r.get("mean_MV_AP_window")), default={})
    b0 = next((r for r in aggregate_rows if r.get("variant_id") == "B0_local_only"), {})
    stream3d = next((r for r in aggregate_rows if r.get("variant_id") == "S3D_L1_local_merged_masks"), {})

    summary = {
        "schema": "stream4d_v90_phase1_variant_resurrection_v1",
        "phase": "v90_phase1_variant_resurrection",
        "runtime_sec": time.time() - t0,
        "support_policy": "local_window_gt_projection",
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "score_free_Match50_window_definition": "F1 from v65 score_free_match_at_050 precision/recall; precision and recall are exported as separate columns.",
        "combined_adapter_input_rows": len(combined_rows),
        "mv_object_rows": len(object_rows),
        "mv_object_frame_mask_rows": len(scoped_frame_rows),
        "mv_metric_rows": len(metric_rows),
        "mv_iou_matrix_rows": len(iou_rows),
        "full_zero_iou_pairs_omitted": False,
        "materializable_required_variant_count": sum(1 for row in materializability_rows if _bool(row.get("materializable"))),
        "not_materializable_required_variant_count": sum(1 for row in materializability_rows if not _bool(row.get("materializable"))),
        "todo_rows": len(todo_rows),
        "B0_MV_AP_window": _num(b0.get("mean_MV_AP_window")),
        "B0_MV_AP50_window": _num(b0.get("mean_MV_AP50_window")),
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": _num(best_real.get("mean_MV_AP_window")),
        "best_real_MV_AP50_window": _num(best_real.get("mean_MV_AP50_window")),
        "best_control_variant": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": _num(best_control.get("mean_MV_AP_window")),
        "best_control_MV_AP50_window": _num(best_control.get("mean_MV_AP50_window")),
        "Stream3D_S3D_L1_MV_AP_window": _num(stream3d.get("mean_MV_AP_window")),
        "Stream3D_S3D_L1_MV_AP50_window": _num(stream3d.get("mean_MV_AP50_window")),
        "best_real_minus_B0_MV_AP_window": _num(best_real.get("mean_MV_AP_window")) - _num(b0.get("mean_MV_AP_window")),
        "best_real_minus_best_control_MV_AP_window": _num(best_real.get("mean_MV_AP_window")) - _num(best_control.get("mean_MV_AP_window")),
        "phase1_pass": bool(metric_rows)
        and "B0_local_only" in {r.get("variant") for r in metric_rows}
        and "C0_semantic_only_control" in {r.get("variant") for r in metric_rows}
        and "S3D_L1_local_merged_masks" in {r.get("variant") for r in metric_rows},
        "outputs": {
            "mv_object_rows": _rel(out_dir / "mv_object_rows.csv"),
            "mv_object_frame_mask_rows": _rel(out_dir / "mv_object_frame_mask_rows.csv"),
            "mv_metric_rows": _rel(out_dir / "mv_metric_rows.csv"),
            "mv_iou_matrix_rows": _rel(out_dir / "mv_iou_matrix_rows.csv"),
            "variant_materializability_rows": _rel(out_dir / "variant_materializability_rows.csv"),
            "materialization_adapter_todo_rows": _rel(out_dir / "materialization_adapter_todo_rows.csv"),
            "mv_metric_aggregate_rows": _rel(out_dir / "mv_metric_aggregate_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(
        out_dir / "SHA256SUMS.json",
        {
            _rel(path): _sha256(path)
            for path in [
                out_dir / "adapter_input_frame_mask_rows.csv",
                out_dir / "mv_object_rows.csv",
                out_dir / "mv_object_frame_mask_rows.csv",
                out_dir / "mv_metric_rows.csv",
                out_dir / "mv_iou_matrix_rows.csv",
                out_dir / "variant_materializability_rows.csv",
                out_dir / "materialization_adapter_todo_rows.csv",
                out_dir / "mv_metric_aggregate_rows.csv",
                out_dir / "summary.json",
            ]
            if path.exists()
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v90 Phase1 MV_AP_window resurrection under local-window support.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--reuse-existing-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
