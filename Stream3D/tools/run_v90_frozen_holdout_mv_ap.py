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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase9_holdout_mv_ap"
PHASE8 = ROOT / "outputs/audit/v90_phase8_dev_decision"
DEFAULT_SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase9_holdout_v82_full_support/native_carrier_support_rows.csv"
DEFAULT_ADAPTER_ROWS = ROOT / "outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"
DEV_LAST_FRAME = {"scene0011_00": 880, "scene0050_00": 590}
HOLDOUT_CHUNKS = {"scene0011_00": set(range(6, 12)), "scene0050_00": set(range(4, 12))}


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _mask_dirs() -> dict[str, Path]:
    return {scene: recalc._mask_dir(scene) for scene in DEV_LAST_FRAME}


def _semantic_features() -> dict[tuple[str, int, int], dict[str, Any]]:
    return phase4._load_semantic_features()


def _holdout_base_rows(adapter_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in adapter_rows:
        scene = row.get("scene_id", "")
        chunk = _int(row.get("chunk_id"), -1)
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        cluster = str(row.get("cluster_id", ""))
        if scene not in HOLDOUT_CHUNKS or chunk not in HOLDOUT_CHUNKS[scene]:
            continue
        if frame_id <= DEV_LAST_FRAME[scene] or mask_id <= 0 or cluster == "":
            continue
        if not _bool(row.get("object_mask_ownership_allowed", "True")):
            continue
        score = row.get("hybrid_adapter_F1") or row.get("rendered_pixel_F1") or row.get("carrier_F1") or 1.0
        local_slot = f"V80_object:c{chunk}:cluster{cluster}"
        base_obj = f"holdout_v82_local:{scene}:c{chunk}:cluster{cluster}"
        rows.append(
            {
                "split": "holdout",
                "scene_id": scene,
                "chunk_id": chunk,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "local_slot_id": local_slot,
                "source_variant": "B0_local_only",
                "variant": "B0_local_only",
                "mv_object_id": f"B0_local_only:{base_obj}",
                "object_score": score,
                "frame_mask_score": score,
                "adapter_score": score,
                "selection_reason": "v90_phase9_holdout_from_v84_v82_local_b0_adapter",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
            }
        )
    return rows


def _control_rows(base_rows: list[dict[str, Any]], features: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in base_rows:
        scene = row["scene_id"]
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        feat = features.get((scene, frame_id, mask_id), {})
        proto = str(feat.get("semantic_prototype_id", "") or "missing")
        label = f"semantic:{scene}:{proto}"
        new = dict(row)
        new["source_variant"] = "C0_semantic_only_control"
        new["variant"] = "C0_semantic_only_control"
        new["mv_object_id"] = f"C0_semantic_only_control:{_hash_text(label)}"
        new["history_id"] = label
        new["semantic_proto_id"] = proto
        new["control_type"] = "semantic_only_control"
        new["is_control"] = True
        new["selection_reason"] = "v90_phase9_semantic_only_control_from_holdout_features"
        out.append(new)
    return out


def _window_rows(base_rows: list[dict[str, Any]], mask_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    by_scene_chunk: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in base_rows:
        by_scene_chunk[(row["scene_id"], _int(row.get("chunk_id"), -1))].add(_int(row.get("frame_id"), -1))
    rows: list[dict[str, Any]] = []
    for (scene, chunk), frames_set in sorted(by_scene_chunk.items()):
        frames = sorted(frame for frame in frames_set if frame >= 0)
        if not frames:
            continue
        stream = recalc.ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        shape_hw = tuple(int(v) for v in stream.load_depth(frames[0]).shape)
        gt_ids: set[int] = set()
        for frame in frames:
            gt = recalc._load_gt_2d(scene, int(frame), shape_hw)
            gt_ids.update(int(v) for v in np.unique(gt) if int(v) > 0)
        rows.append(
            {
                "scene_id": scene,
                "split": "holdout",
                "window_id": f"h_c{chunk:02d}",
                "window_index": int(chunk),
                "chunk_id": int(chunk),
                "frame_id_start": int(frames[0]),
                "frame_id_end": int(frames[-1]),
                "frame_count": int(len(frames)),
                "window_scoped_gt_count": int(len(gt_ids)),
                "mask_source": _rel(mask_dirs[scene]),
                "support_policy": "local_window_gt_projection",
                "GT_scope": "gt ids scoped by (scene_id, holdout_chunk_id, gt_id)",
                "prediction_scope": "predicted object tubes scoped to the same holdout chunk before v65 evaluator",
            }
        )
    return rows


def _frame_window_maps(window_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], str], dict[str, list[dict[str, Any]]]]:
    frame_to_window: dict[tuple[str, int], int] = {}
    frame_to_window_id: dict[tuple[str, int], str] = {}
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        scene = row["scene_id"]
        window_index = _int(row.get("window_index"), -1)
        start = _int(row.get("frame_id_start"), -1)
        end = _int(row.get("frame_id_end"), -1)
        frames = list(range(start, end + 1, 5))
        item = {**row, "frames": frames}
        by_scene[scene].append(item)
        for frame in frames:
            frame_to_window[(scene, frame)] = window_index
            frame_to_window_id[(scene, frame)] = str(row.get("window_id", f"h_c{window_index:02d}"))
    return frame_to_window, frame_to_window_id, by_scene


def _slot_maps(base_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], dict[tuple[str, str], float]]:
    slot_to_obj: dict[tuple[str, str], str] = {}
    slot_to_area: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in base_rows:
        scene = row["scene_id"]
        slot = row["local_slot_id"]
        slot_to_obj.setdefault((scene, slot), str(row["mv_object_id"]).replace("B0_local_only:", ""))
        score = _num(row.get("object_score"), 0.0)
        if score > 0:
            slot_to_area[(scene, slot)].append(score)
    return slot_to_obj, {}, {key: _mean(vals) for key, vals in slot_to_area.items()}


def _eval_rows_from_selection(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def _object_scores(rows: list[dict[str, Any]], object_to_idx: dict[str, int], frame_to_window: dict[tuple[str, int], int]) -> np.ndarray:
    by_obj: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scene = row.get("scene_id", "")
        frame_id = _int(row.get("frame_id"), -1)
        window_index = frame_to_window.get((scene, frame_id))
        if window_index is None:
            continue
        obj = str(row.get("mv_object_id", ""))
        scoped = f"{scene}|h_c{window_index:02d}|{obj}"
        if scoped in object_to_idx:
            by_obj[scoped].append(_num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)))
    scores = np.ones((len(object_to_idx),), dtype=np.float32)
    for obj, idx in object_to_idx.items():
        vals = by_obj.get(obj, [1.0])
        scores[idx - 1] = float(sum(vals) / max(1, len(vals)))
    return scores


def _evaluate_variant(
    variant: str,
    rows: list[dict[str, Any]],
    windows_by_scene: dict[str, list[dict[str, Any]]],
    frame_to_window: dict[tuple[str, int], int],
    mask_dirs: dict[str, Path],
    generated_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[row.get("scene_id", "")].append(row)

    for scene, scene_windows in sorted(windows_by_scene.items()):
        scene_rows = rows_by_scene.get(scene, [])
        object_ids = sorted(
            {
                f"{scene}|h_c{int(frame_to_window[(scene, _int(row.get('frame_id'), -1))]):02d}|{row.get('mv_object_id', '')}"
                for row in scene_rows
                if (scene, _int(row.get("frame_id"), -1)) in frame_to_window and row.get("mv_object_id", "")
            }
        )
        object_to_idx = {obj: idx + 1 for idx, obj in enumerate(object_ids)}
        idx_to_obj = {idx: obj for obj, idx in object_to_idx.items()}
        mask_to_obj: dict[tuple[int, int, int], int] = {}
        duplicate_conflicts = 0
        for row in scene_rows:
            frame_id = _int(row.get("frame_id"), -1)
            window_index = frame_to_window.get((scene, frame_id))
            if window_index is None:
                continue
            obj = f"{scene}|h_c{window_index:02d}|{row.get('mv_object_id', '')}"
            if obj not in object_to_idx:
                continue
            key = (window_index, frame_id, _int(row.get("mask_id"), -1))
            if key[2] <= 0:
                continue
            old = mask_to_obj.get(key)
            if old is not None and old != object_to_idx[obj]:
                duplicate_conflicts += 1
                continue
            mask_to_obj[key] = object_to_idx[obj]
        scores = _object_scores(scene_rows, object_to_idx, frame_to_window)
        stream = recalc.ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        first_frame = int(scene_windows[0]["frames"][0])
        shape_hw = tuple(int(v) for v in stream.load_depth(first_frame).shape)
        acc = recalc.SparseSceneIoU()
        gt_id_map: dict[tuple[int, int], int] = {}
        missing_masks = 0
        for window in scene_windows:
            window_index = _int(window.get("window_index"), -1)
            for frame_id in window["frames"]:
                gt = recalc._load_gt_2d(scene, int(frame_id), shape_hw)
                gt_window = recalc._window_scoped_gt(gt, window_index, gt_id_map)
                pred = np.zeros(shape_hw, dtype=np.int64)
                if variant == "W9b_risk_balanced_p165_plus_carving":
                    mask_path = generated_root / variant / scene / "mask" / f"{int(frame_id)}.png"
                else:
                    mask_path = mask_dirs[scene] / f"{int(frame_id)}.png"
                if mask_path.exists():
                    mask = recalc._read_label_png(mask_path, shape_hw)
                    for mask_id in np.unique(mask):
                        mask_id = int(mask_id)
                        if mask_id <= 0:
                            continue
                        label = mask_to_obj.get((window_index, int(frame_id), mask_id), 0)
                        if label > 0:
                            pred[mask == mask_id] = label
                else:
                    missing_masks += 1
                acc.add(pred, gt_window)
                case_rows.append(
                    {
                        "split": "holdout",
                        "scene_id": scene,
                        "variant": variant,
                        "window_id": window.get("window_id", ""),
                        "window_index": window_index,
                        "frame_id": int(frame_id),
                        "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                        "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                        "mask_path": _rel(mask_path),
                        "mask_exists": bool(mask_path.exists()),
                    }
                )
        summary, iou, pred_ids, gt_ids = recalc._summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=scores,
        )
        metric = {
            "split": "holdout",
            "scene_id": scene,
            "variant": variant,
            "variant_id": variant,
            "MV_AP_window": summary.get("ap"),
            "MV_AP50_window": summary.get("ap50"),
            "MV_AP25_window": summary.get("ap25"),
            "score_free_Match50_window": phase1._f1(
                summary.get("score_free_match_at_050", {}).get("precision"),
                summary.get("score_free_match_at_050", {}).get("recall"),
            ),
            "score_free_Match50_precision_window": summary.get("score_free_match_at_050", {}).get("precision"),
            "score_free_Match50_recall_window": summary.get("score_free_match_at_050", {}).get("recall"),
            "pred_object_count": summary.get("evaluated_pred_count"),
            "gt_object_count": summary.get("evaluated_gt_count"),
            "gt_best_iou_mean": summary.get("gt_best_iou_mean"),
            "frame_first": int(min(frame for window in scene_windows for frame in window["frames"])),
            "frame_last": int(max(frame for window in scene_windows for frame in window["frames"])),
            "support_window_count": len(scene_windows),
            "duplicate_frame_mask_conflict_count": duplicate_conflicts,
            "same_frame_collision_count": duplicate_conflicts,
            "missing_mask_raster_count": missing_masks,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
            "support_policy": "local_window_gt_projection",
        }
        metric_rows.append(metric)
        for row in phase4._all_iou_rows(iou, pred_ids, gt_ids, top_k=100):
            iou_rows.append(
                {
                    "split": "holdout",
                    "scene_id": scene,
                    "variant": variant,
                    "variant_id": variant,
                    "pred_id": row["pred_id"],
                    "mv_object_id": idx_to_obj.get(int(row["pred_id"]), ""),
                    "gt_id": row["gt_id"],
                    "iou": row["iou"],
                    "mv_iou": row["iou"],
                    "matrix_scope": "phase9_full_pred_gt_iou_matrix_holdout_local_window_support",
                    "full_zero_pairs_omitted": False,
                }
            )
    return metric_rows, iou_rows, case_rows


def _aggregate(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("variant", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        out.append(
            {
                "variant_id": variant,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([_num(row.get("MV_AP_window")) for row in rows]),
                "mean_MV_AP50_window": _mean([_num(row.get("MV_AP50_window")) for row in rows]),
                "mean_MV_AP25_window": _mean([_num(row.get("MV_AP25_window")) for row in rows]),
                "mean_score_free_Match50_window": _mean([_num(row.get("score_free_Match50_window")) for row in rows]),
                "mean_gt_object_count": _mean([_num(row.get("gt_object_count")) for row in rows]),
                "mean_pred_object_count": _mean([_num(row.get("pred_object_count")) for row in rows]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in rows)),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in rows),
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    phase4.OUT = OUT
    mask_dirs = _mask_dirs()
    features = _semantic_features()
    support_path = ROOT / args.native_support_rows if not Path(args.native_support_rows).is_absolute() else Path(args.native_support_rows)
    adapter_path = ROOT / args.adapter_rows if not Path(args.adapter_rows).is_absolute() else Path(args.adapter_rows)
    frozen_config = PHASE8 / "frozen_candidate_config.json"
    phase8_summary = json.loads((PHASE8 / "summary.json").read_text(encoding="utf-8"))
    config_file_sha256 = _sha256(frozen_config)
    config_sha256_matches = config_file_sha256 == phase8_summary.get("frozen_config_file_sha256")

    base_rows = _holdout_base_rows(_read_csv(adapter_path))
    control_rows = _control_rows(base_rows, features)
    window_rows = _window_rows(base_rows, mask_dirs)
    frame_to_window, frame_to_window_id, windows_by_scene = _frame_window_maps(window_rows)
    slot_to_obj, slot_to_proto, slot_to_area = _slot_maps(base_rows)

    allowed_frame_keys = set(frame_to_window)
    candidates, support_points = phase4._load_support_candidates(support_path, set(slot_to_obj), features, mask_dirs)
    candidates = [
        row
        for row in candidates
        if (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1)) in allowed_frame_keys
    ]
    support_points = {
        key: points
        for key, points in support_points.items()
        if (str(key[0]), int(key[2])) in allowed_frame_keys
    }
    slot_to_proto, slot_to_area = phase4._fill_slot_priors_from_candidates(candidates, slot_to_proto, slot_to_area)
    selection_rows, dropped_rows = phase4._select_original_masklets(candidates, slot_to_obj, slot_to_proto, slot_to_area, frame_to_window, frame_to_window_id)
    w8b_rows = [row for row in selection_rows if row.get("variant_id") == "W8b_risk_balanced_p165_witnesses"]
    generated_rows, w9b_selection_rows, w9b_eval_rows = phase4._generate_carved_masks(
        w8b_rows,
        support_points,
        mask_dirs,
        radius=int(args.carving_radius),
        support_point_radius=int(args.support_point_radius),
        variant="W9b_risk_balanced_p165_plus_carving",
        source_variant="W8b_risk_balanced_p165_witnesses",
    )
    w9b_eval_rows = _eval_rows_from_selection(w9b_eval_rows, "holdout")
    b0_eval_rows = [dict(row) for row in base_rows]
    c0_eval_rows = [dict(row) for row in control_rows]

    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for variant, rows in [
        ("B0_local_only", b0_eval_rows),
        ("C0_semantic_only_control", c0_eval_rows),
        ("W9b_risk_balanced_p165_plus_carving", w9b_eval_rows),
    ]:
        metrics, ious, cases = _evaluate_variant(variant, rows, windows_by_scene, frame_to_window, mask_dirs, OUT / "generated_masks")
        metric_rows.extend(metrics)
        iou_rows.extend(ious)
        case_rows.extend(cases)

    aggregate_rows = _aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    b0 = by_variant.get("B0_local_only", {})
    c0 = by_variant.get("C0_semantic_only_control", {})
    w9b = by_variant.get("W9b_risk_balanced_p165_plus_carving", {})
    gate = {
        "holdout_best_real_MV_AP_window_ge_B0_plus_0p008": _num(w9b.get("mean_MV_AP_window")) >= _num(b0.get("mean_MV_AP_window")) + 0.008,
        "holdout_best_real_MV_AP50_window_ge_B0_plus_0p015": _num(w9b.get("mean_MV_AP50_window")) >= _num(b0.get("mean_MV_AP50_window")) + 0.015,
        "holdout_best_real_MV_AP_window_ge_control_plus_0p005": _num(w9b.get("mean_MV_AP_window")) >= _num(c0.get("mean_MV_AP_window")) + 0.005,
        "holdout_same_frame_collision_count_eq_0": _int(w9b.get("same_frame_collision_count")) == 0,
        "holdout_uses_gt_for_prediction_false": not _bool(w9b.get("uses_gt_for_prediction")),
        "holdout_uses_future_false": not _bool(w9b.get("uses_future")),
        "config_sha256_matches": config_sha256_matches,
    }
    phase9_pass = all(gate.values())
    failure_rows = [
        {
            "case_type": "holdout_gate",
            "gate": key,
            "pass": value,
            "best_real_variant": "W9b_risk_balanced_p165_plus_carving",
            "B0_MV_AP_window": b0.get("mean_MV_AP_window", ""),
            "C0_MV_AP_window": c0.get("mean_MV_AP_window", ""),
            "W9b_MV_AP_window": w9b.get("mean_MV_AP_window", ""),
            "next_action": "Do not retune v90 holdout config; use failure evidence for v91." if not value else "",
        }
        for key, value in gate.items()
    ]

    _write_csv(OUT / "holdout_window_support_rows.csv", window_rows)
    _write_csv(OUT / "holdout_source_frame_mask_rows.csv", b0_eval_rows + c0_eval_rows)
    _write_csv(OUT / "holdout_masklet_candidate_rows.csv", candidates)
    _write_csv(OUT / "holdout_witness_cover_selection_rows.csv", selection_rows + dropped_rows + w9b_selection_rows)
    _write_csv(OUT / "holdout_generated_mask_rows.csv", generated_rows)
    _write_csv(OUT / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "holdout_iou_matrix_rows.csv", iou_rows)
    _write_csv(OUT / "holdout_casebook_rows.csv", case_rows + failure_rows)
    _write_json(
        OUT / "config_sha256_check.json",
        {
            "config_path": _rel(frozen_config),
            "expected_config_file_sha256": phase8_summary.get("frozen_config_file_sha256", ""),
            "actual_config_file_sha256": config_file_sha256,
            "config_sha256_matches": config_sha256_matches,
        },
    )
    summary = {
        "phase": "v90_phase9_holdout_mv_ap",
        "schema": "stream4d_v90_phase9_frozen_holdout_mv_ap_v1",
        "phase9_pass": phase9_pass,
        "holdout_run_executed": True,
        "blocked": False,
        "blocked_reason": "",
        "selected_frozen_variant": "W9b_risk_balanced_p165_plus_carving",
        "config_sha256_matches": config_sha256_matches,
        "config_file_sha256": config_file_sha256,
        "holdout_B0": b0,
        "holdout_best_control_variant": "C0_semantic_only_control",
        "holdout_best_control": c0,
        "holdout_best_real_variant": "W9b_risk_balanced_p165_plus_carving",
        "holdout_best_real": w9b,
        "gate": gate,
        "inputs": {
            "native_support_rows": _rel(support_path),
            "adapter_rows": _rel(adapter_path),
            "frozen_config": _rel(frozen_config),
        },
        "row_counts": {
            "base_rows": len(base_rows),
            "control_rows": len(control_rows),
            "support_candidates": len(candidates),
            "selection_rows": len(selection_rows),
            "w9b_selection_rows": len(w9b_selection_rows),
            "generated_rows": len(generated_rows),
            "metric_rows": len(metric_rows),
            "iou_rows": len(iou_rows),
            "casebook_rows": len(case_rows) + len(failure_rows),
            "window_rows": len(window_rows),
        },
        "provenance_caveat": "Holdout B0/source rows are materialized from v84 holdout replay v82 local B0 adapter rows because v89 dev frame rows do not cover later windows. C0 is rebuilt from the same holdout frame-mask rows using semantic_prototype_id features. No holdout metric is used to change W9b thresholds.",
        "runtime_sec": time.time() - started,
    }
    _write_json(OUT / "summary.json", summary)
    sha_paths = [
        OUT / "holdout_window_support_rows.csv",
        OUT / "holdout_source_frame_mask_rows.csv",
        OUT / "holdout_masklet_candidate_rows.csv",
        OUT / "holdout_witness_cover_selection_rows.csv",
        OUT / "holdout_generated_mask_rows.csv",
        OUT / "holdout_metric_rows.csv",
        OUT / "holdout_metric_aggregate_rows.csv",
        OUT / "holdout_iou_matrix_rows.csv",
        OUT / "holdout_casebook_rows.csv",
        OUT / "config_sha256_check.json",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v90 Phase9 frozen W9b holdout MV_AP_window once.")
    parser.add_argument("--native-support-rows", default=str(DEFAULT_SUPPORT_ROWS.relative_to(ROOT)))
    parser.add_argument("--adapter-rows", default=str(DEFAULT_ADAPTER_ROWS.relative_to(ROOT)))
    parser.add_argument("--support-point-radius", type=int, default=3)
    parser.add_argument("--carving-radius", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
