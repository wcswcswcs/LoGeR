#!/usr/bin/env python3
"""Materialize v102 Phase7c components and run a local chunk32 AP diagnostic."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE7C_DIR = AUDIT_ROOT / "v102_phase7c_node_quality_constrained_rebirth"
PHASE7C_SUMMARY = PHASE7C_DIR / "summary.json"
PHASE7C_COMPONENT_ROWS = PHASE7C_DIR / "node_quality_component_rows.csv"
FEATURE_ROWS = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_feature_rows.csv"
PHASE1B_SUMMARY = AUDIT_ROOT / "v102_phase1b_repair_space_policy_sweep" / "summary.json"
SOURCE_ROWS = AUDIT_ROOT / "v95_phase1_physical_source_registry" / "source_container_rows.csv"

PHASE_ID = "v102_phase7d_phase7c_materialized_ap_diagnostic"
SOURCE_PHASE7C_VARIANT = "entropy045_tau05_block_centroid090"
VARIANT_ID = "P2_v102_phase7d_entropy045_node_quality_rebirth_mask_component"
SCENE_ID = "scene0050_00"
CHUNK_ID = "c0000"
FRAME_STRIDE = 5
CHUNK_SIZE = 32


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _project_stream3d(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


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


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_path_lookup() -> tuple[dict[tuple[str, int], Path], dict[str, Any]]:
    out: dict[tuple[str, int], Path] = {}
    duplicate_conflict_count = 0
    source_uses_gt = False
    source_uses_future = False
    with SOURCE_ROWS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = str(row.get("scene_id", ""))
            try:
                frame_id = int(_num(row.get("frame_id")))
            except Exception:
                continue
            raw = str(row.get("mask_path", "")).strip()
            if not raw:
                continue
            path = _project_stream3d(raw)
            key = (scene, frame_id)
            if key in out and out[key] != path:
                duplicate_conflict_count += 1
                continue
            out.setdefault(key, path)
            source_uses_gt = source_uses_gt or _bool(row.get("uses_gt_for_prediction"))
            source_uses_future = source_uses_future or _bool(row.get("uses_future"))
    return out, {
        "source_rows": _rel(SOURCE_ROWS),
        "mask_path_frame_count": int(len(out)),
        "duplicate_conflict_count": int(duplicate_conflict_count),
        "source_uses_gt_for_prediction": bool(source_uses_gt),
        "source_uses_future": bool(source_uses_future),
    }


def _parse_node_id(node_id: str) -> tuple[str, int, int]:
    parts = str(node_id).split(":")
    if len(parts) != 3:
        raise ValueError(f"unsupported node id: {node_id}")
    return parts[0], int(parts[1]), int(parts[2])


def _top_iou_rows(
    iou: np.ndarray,
    pred_ids: list[int],
    gt_ids: list[int],
    idx_to_object: dict[int, str],
    object_meta: dict[str, dict[str, Any]],
    top_k: int = 80,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if iou.size == 0:
        return rows
    flat = np.argsort(iou.reshape(-1))[::-1]
    for flat_idx in flat[: int(top_k)]:
        pidx = int(flat_idx // max(1, iou.shape[1]))
        gidx = int(flat_idx % max(1, iou.shape[1]))
        value = float(iou[pidx, gidx])
        if value <= 0:
            break
        pred_id = int(pred_ids[pidx])
        object_id = idx_to_object.get(pred_id, "")
        meta = object_meta.get(object_id, {})
        rows.append(
            {
                "schema_version": "stream4d_v102_phase7d_top_iou_pair_v1",
                "phase_id": PHASE_ID,
                "variant_id": VARIANT_ID,
                "pred_id": pred_id,
                "mv_object_id": object_id,
                "source_component_id": meta.get("source_component_id", ""),
                "gt_id": int(gt_ids[gidx]),
                "iou": value,
                "object_score": meta.get("object_score", ""),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    return rows


def _score_component(row: dict[str, Any]) -> float:
    frame_term = int(_num(row.get("frame_count"))) / float(CHUNK_SIZE)
    margin_term = _num(row.get("semantic_margin_mean")) * 1.0e-4
    entropy_term = (1.0 - _num(row.get("semantic_entropy_mean"))) * 1.0e-6
    return float(frame_term + margin_term + entropy_term)


def _frame_universe() -> list[int]:
    return list(range(0, FRAME_STRIDE * CHUNK_SIZE, FRAME_STRIDE))


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase7c_summary = _read_json(PHASE7C_SUMMARY)
    phase1b_summary = _read_json(PHASE1B_SUMMARY)
    feature_rows = pd.read_csv(FEATURE_ROWS)
    feature_by_node = feature_rows.set_index("mask_observation_id").to_dict(orient="index")
    component_df = pd.read_csv(PHASE7C_COMPONENT_ROWS)
    component_df = component_df[component_df["variant_id"].astype(str) == SOURCE_PHASE7C_VARIANT].copy()
    component_df = component_df.sort_values(["component_id"]).reset_index(drop=True)

    if component_df.empty:
        raise RuntimeError(f"no Phase7c component rows for variant {SOURCE_PHASE7C_VARIANT}")

    truncated_rows = component_df[component_df["node_ids_truncated"].map(_bool)]
    node_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    materialized_component_rows: list[dict[str, Any]] = []
    object_scores: dict[str, float] = {}
    object_index: dict[str, int] = {}
    index_to_object: dict[int, str] = {}
    object_meta: dict[str, dict[str, Any]] = {}

    for component_idx, component in component_df.iterrows():
        source_component_id = str(component["component_id"])
        local_component_id = f"component_{int(component_idx):04d}"
        mv_object_id = f"dev:{VARIANT_ID}:{SCENE_ID}:{CHUNK_ID}:{local_component_id}"
        object_index[mv_object_id] = len(object_index) + 1
        index_to_object[object_index[mv_object_id]] = mv_object_id
        score = _score_component(component.to_dict())
        object_scores[mv_object_id] = score
        node_ids = [part for part in str(component["node_ids_joined"]).split("|") if part]
        feature_missing_count = 0
        for node_id in node_ids:
            scene, frame_id, mask_id = _parse_node_id(node_id)
            if scene != SCENE_ID:
                raise ValueError(f"unexpected scene in node {node_id}; expected {SCENE_ID}")
            feature = feature_by_node.get(node_id, {})
            feature_missing_count += int(not bool(feature))
            node_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7d_mv_object_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "dataset_split": "dev",
                    "variant_id": VARIANT_ID,
                    "source_phase7c_variant_id": SOURCE_PHASE7C_VARIANT,
                    "mv_object_id": mv_object_id,
                    "object_id": mv_object_id,
                    "source_component_id": source_component_id,
                    "scene_id": scene,
                    "chunk_id": CHUNK_ID,
                    "window_id": CHUNK_ID,
                    "frame_id": int(frame_id),
                    "selected_mask_id": int(mask_id),
                    "mask_id_or_generated_id": int(mask_id),
                    "source_mask_observation_id": node_id,
                    "readout_mode": "phase7c_component_snap_to_cropformer_mask",
                    "score": score,
                    "object_score": score,
                    "score_scope": "current_chunk32_diagnostic",
                    "score_policy": "component_frame_count_over_32_plus_gtfree_semantic_quality_tiebreak",
                    "component_node_count": int(_num(component.get("node_count"))),
                    "component_frame_count": int(_num(component.get("frame_count"))),
                    "method_chunk_size": CHUNK_SIZE,
                    "method_chunk_overlap": 0,
                    "frame_stride": FRAME_STRIDE,
                    "object_id_policy": "phase7c_component_identity_chunk_scoped",
                    "object_birth_scope": "phase7c_node_quality_constrained_primitive_rebirth",
                    "semantic_entropy": feature.get("semantic_entropy", ""),
                    "semantic_prototype_margin": feature.get("semantic_prototype_margin", ""),
                    "broad_background_risk": feature.get("broad_background_risk", ""),
                    "used_pixel_count": feature.get("used_pixel_count", ""),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": False,
                    "uses_future": False,
                }
            )
        materialized_component_rows.append(
            {
                "schema_version": "stream4d_v102_phase7d_component_materialization_v1",
                "phase_id": PHASE_ID,
                "variant_id": VARIANT_ID,
                "source_phase7c_variant_id": SOURCE_PHASE7C_VARIANT,
                "source_component_id": source_component_id,
                "mv_object_id": mv_object_id,
                "node_count": int(_num(component.get("node_count"))),
                "frame_count": int(_num(component.get("frame_count"))),
                "object_score": score,
                "feature_missing_count": int(feature_missing_count),
                "node_ids_truncated": bool(_bool(component.get("node_ids_truncated"))),
                "diagnostic_gt_dominant": component.get("diagnostic_gt_dominant", ""),
                "diagnostic_gt_purity": _num(component.get("diagnostic_gt_purity")),
                "diagnostic_gt_count": int(_num(component.get("diagnostic_gt_count"))),
                "diagnostic_semantic_dominant": component.get("diagnostic_semantic_dominant", ""),
                "diagnostic_semantic_count": int(_num(component.get("diagnostic_semantic_count"))),
                "semantic_entropy_mean": _num(component.get("semantic_entropy_mean")),
                "semantic_margin_mean": _num(component.get("semantic_margin_mean")),
                "clean_component_proxy": bool(_bool(component.get("clean_component_proxy"))),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        object_meta[mv_object_id] = materialized_component_rows[-1]

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        by_object[str(row["mv_object_id"])].append(row)
    for mv_object_id, rows in sorted(by_object.items()):
        frames = sorted({int(row["frame_id"]) for row in rows})
        object_rows.append(
            {
                "schema_version": "stream4d_v102_phase7d_mv_object_row_v1",
                "phase_id": PHASE_ID,
                "dataset_split": "dev",
                "variant_id": VARIANT_ID,
                "mv_object_id": mv_object_id,
                "object_id": mv_object_id,
                "source_component_id": rows[0]["source_component_id"],
                "scene_id": SCENE_ID,
                "chunk_id": CHUNK_ID,
                "object_frame_count": len(frames),
                "object_score": float(object_scores[mv_object_id]),
                "score_scope": "current_chunk32_diagnostic",
                "object_id_policy": "phase7c_component_identity_chunk_scoped",
                "object_birth_scope": "phase7c_node_quality_constrained_primitive_rebirth",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    eval_frame_ids = _frame_universe()
    mask_path_by_frame, mask_source = _mask_path_lookup()
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        rows_by_frame[int(row["frame_id"])].append(row)

    duplicate_keys = Counter(
        (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"])) for row in node_rows
    )
    same_frame_duplicate_mask_count = sum(max(0, count - 1) for count in duplicate_keys.values())

    acc_window = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    total_pred_positive = 0
    total_gt_positive = 0
    for frame_id in eval_frame_ids:
        mask_path = mask_path_by_frame.get((SCENE_ID, int(frame_id)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = _read_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        selected_rows = sorted(
            rows_by_frame.get(int(frame_id), []),
            key=lambda row: (-float(row["object_score"]), str(row["mv_object_id"])),
        )
        emitted_pixels = 0
        for row in selected_rows:
            mask = label == int(row["selected_mask_id"])
            mask_pixels = int(np.count_nonzero(mask))
            if mask_pixels <= 0:
                selected_mask_missing_count += 1
                continue
            pred_id = object_index[str(row["mv_object_id"])]
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            writable = (pred == 0) & mask
            pred[writable] = int(pred_id)
            emitted_pixels += int(np.count_nonzero(writable))
        gt = _load_gt_2d(SCENE_ID, int(frame_id), label.shape)
        acc_window.add(pred, gt)
        pred_positive = int(np.count_nonzero(pred > 0))
        gt_positive = int(np.count_nonzero(gt > 0))
        total_pred_positive += pred_positive
        total_gt_positive += gt_positive
        frame_rows.append(
            {
                "schema_version": "stream4d_v102_phase7d_preview_frame_v1",
                "phase_id": PHASE_ID,
                "variant_id": VARIANT_ID,
                "metric_name": "MV_AP_window",
                "scene_id": SCENE_ID,
                "chunk_id": CHUNK_ID,
                "window_id": CHUNK_ID,
                "frame_id": int(frame_id),
                "mask_path": _rel(mask_path),
                "emitted_object_count": len(selected_rows),
                "emitted_pixels": int(emitted_pixels),
                "pred_positive_pixels": pred_positive,
                "gt_positive_pixels": gt_positive,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    input_scores = np.ones((len(object_index),), dtype=np.float32)
    for oid, pred_id in object_index.items():
        input_scores[pred_id - 1] = float(object_scores.get(oid, 1.0))
    window_summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc_window,
        min_pred_pixels=64,
        min_gt_pixels=64,
        score_mode="input",
        input_scores=input_scores,
    )
    scene_summary = dict(window_summary)
    top_iou_rows = _top_iou_rows(iou, pred_ids, gt_ids, index_to_object, object_meta)

    ap = window_summary.get("ap")
    ap50 = window_summary.get("ap50")
    ap25 = window_summary.get("ap25")
    scene_ap = scene_summary.get("ap")
    scene_ap50 = scene_summary.get("ap50")
    sf50 = (window_summary.get("score_free_match_at_050") or {}).get("recall")
    sf25 = (window_summary.get("score_free_match_at_025") or {}).get("recall")
    integrity_pass = bool(
        len(truncated_rows) == 0
        and missing_mask_frame_count == 0
        and selected_mask_missing_count == 0
        and same_frame_duplicate_mask_count == 0
        and pixel_collision_count == 0
    )
    local_diagnostic_ap_recorded = bool(ap is not None and ap50 is not None)
    formal_v102_target_achieved = False
    decision = (
        "PASS_PHASE7D_LOCAL_DIAGNOSTIC_AP_RECORDED__FORMAL_TARGET_NOT_CLAIMED"
        if integrity_pass and local_diagnostic_ap_recorded
        else "NO_GO_PHASE7D_MATERIALIZATION_INTEGRITY_OR_AP_BLOCKED"
    )

    metric_rows = [
        {
            "schema_version": "stream4d_v102_phase7d_variant_metric_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
            "MV_AP_window": ap,
            "MV_AP50_window": ap50,
            "MV_AP25_window": ap25,
            "MV_AP_scene": scene_ap,
            "MV_AP50_scene": scene_ap50,
            "MV_AP25_scene": scene_summary.get("ap25"),
            "ScoreFreeMatch50_window": sf50,
            "ScoreFreeMatch25_window": sf25,
            "object_count": len(object_rows),
            "frame_mask_count": len(node_rows),
            "eval_frame_count": len(frame_rows),
            "same_frame_duplicate_mask_count": int(same_frame_duplicate_mask_count),
            "pixel_collision_count": int(pixel_collision_count),
            "missing_mask_frame_count": int(missing_mask_frame_count),
            "selected_mask_missing_count": int(selected_mask_missing_count),
            "total_pred_positive_pixels": int(total_pred_positive),
            "total_gt_positive_pixels": int(total_gt_positive),
            "integrity_pass": integrity_pass,
            "local_diagnostic_ap_recorded": local_diagnostic_ap_recorded,
            "formal_v102_target_achieved": formal_v102_target_achieved,
            "phase6_ap_repair_allowed": bool(
                phase1b_summary.get("phase6_ap_repair_allowed", False)
            ),
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        }
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "phase7c_source_component_gate_passed",
            "pass": bool(phase7c_summary.get("any_variant_safe_for_primitive_rebirth")),
            "observed": phase7c_summary.get("decision"),
            "required": "Phase7c safe diagnostic components",
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "node_ids_not_truncated",
            "pass": len(truncated_rows) == 0,
            "observed": int(len(truncated_rows)),
            "required": 0,
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "same_frame_duplicate_mask_count_eq_0",
            "pass": same_frame_duplicate_mask_count == 0,
            "observed": int(same_frame_duplicate_mask_count),
            "required": 0,
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "pixel_collision_count_eq_0",
            "pass": pixel_collision_count == 0,
            "observed": int(pixel_collision_count),
            "required": 0,
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "missing_mask_frame_count_eq_0",
            "pass": missing_mask_frame_count == 0,
            "observed": int(missing_mask_frame_count),
            "required": 0,
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "selected_mask_missing_count_eq_0",
            "pass": selected_mask_missing_count == 0,
            "observed": int(selected_mask_missing_count),
            "required": 0,
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "MV_AP_window_recorded_local_diagnostic",
            "pass": ap is not None,
            "observed": ap,
            "required": "record diagnostic AP; not a formal full-dev gate",
        },
        {
            "schema_version": "stream4d_v102_phase7d_gate_v1",
            "phase_id": PHASE_ID,
            "variant_id": VARIANT_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "not claimed from chunk32 diagnostic; Phase6 still blocked by Phase1b repair-space",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]

    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", node_rows)
    _write_csv(OUT_DIR / "mv_object_rows.csv", object_rows)
    _write_parquet(OUT_DIR / "mv_object_frame_mask_rows.parquet", node_rows)
    _write_parquet(OUT_DIR / "mv_object_rows.parquet", object_rows)
    _write_csv(OUT_DIR / "materialized_component_rows.csv", materialized_component_rows)
    _write_csv(OUT_DIR / "preview_frame_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "top_iou_pairs.csv", top_iou_rows)

    summary = {
        "schema_version": "stream4d_v102_phase7d_materialized_ap_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "variant_id": VARIANT_ID,
        "source_phase7c_variant_id": SOURCE_PHASE7C_VARIANT,
        "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
        "local_diagnostic_ap_recorded": local_diagnostic_ap_recorded,
        "formal_v102_target_achieved": formal_v102_target_achieved,
        "formal_target_blocker": "Phase6 full AP repair remains blocked by Phase1b safe_promotable_candidate_count_max=0; this Phase7d result is local chunk32 diagnostic only.",
        "MV_AP_window": ap,
        "MV_AP50_window": ap50,
        "MV_AP25_window": ap25,
        "MV_AP_scene": scene_ap,
        "MV_AP50_scene": scene_ap50,
        "MV_AP25_scene": scene_summary.get("ap25"),
        "ScoreFreeMatch50_window": sf50,
        "ScoreFreeMatch25_window": sf25,
        "object_count": len(object_rows),
        "frame_mask_count": len(node_rows),
        "component_count": int(len(component_df)),
        "eval_frame_count": int(len(frame_rows)),
        "eval_frame_first": int(eval_frame_ids[0]) if eval_frame_ids else None,
        "eval_frame_last": int(eval_frame_ids[-1]) if eval_frame_ids else None,
        "same_frame_duplicate_mask_count": int(same_frame_duplicate_mask_count),
        "pixel_collision_count": int(pixel_collision_count),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "node_ids_truncated_component_count": int(len(truncated_rows)),
        "integrity_pass": integrity_pass,
        "mask_source": mask_source,
        "phase7c_decision": phase7c_summary.get("decision"),
        "phase1b_phase6_ap_repair_allowed": bool(phase1b_summary.get("phase6_ap_repair_allowed", False)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "This run materializes Phase7c GT-free component identities into CropFormer mask predictions and uses GT only "
            "for the v65 SparseSceneIoU diagnostic readout. It is not a full-dev/holdout AP improvement claim."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "mv_object_frame_mask_rows_csv": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "mv_object_frame_mask_rows_parquet": _rel(OUT_DIR / "mv_object_frame_mask_rows.parquet"),
            "mv_object_rows_csv": _rel(OUT_DIR / "mv_object_rows.csv"),
            "mv_object_rows_parquet": _rel(OUT_DIR / "mv_object_rows.parquet"),
            "materialized_component_rows": _rel(OUT_DIR / "materialized_component_rows.csv"),
            "preview_frame_rows": _rel(OUT_DIR / "preview_frame_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "top_iou_pairs": _rel(OUT_DIR / "top_iou_pairs.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
