from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import (  # noqa: E402
    SparseSceneIoU,
    _load_gt_2d,
    _read_label_png,
    _summarize_iou,
    _top_iou_rows,
    _write_csv,
    _write_json,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_obs(value: Any) -> tuple[str, int, int] | None:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _best_variant(pipeline_root: Path) -> str:
    summary = _read_json(pipeline_root / "local_objectlets/local_objectlet_summary.json")
    variant = str(summary.get("best_real_variant") or summary.get("best_real_row", {}).get("variant") or "").strip()
    if not variant:
        raise RuntimeError(f"missing best local objectlet variant: {pipeline_root}")
    return variant


def _selected_candidate_ids(pipeline_root: Path, scene: str, variant: str) -> tuple[set[str], dict[str, str]]:
    selected: set[str] = set()
    owners: dict[str, str] = {}
    path = pipeline_root / "local_objectlets/objectlet_rows.csv"
    for row in _read_csv(path):
        if row.get("scene") != scene or row.get("variant") != variant:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        objectlet_id = str(row.get("objectlet_id") or "").strip()
        if candidate_id:
            selected.add(candidate_id)
            owners[candidate_id] = objectlet_id or candidate_id
    return selected, owners


def _ledger_pair_sets(pipeline_root: Path, scene: str, variant: str) -> dict[str, Any]:
    selected_candidates, candidate_to_object = _selected_candidate_ids(pipeline_root, scene, variant)
    u0: set[tuple[int, int]] = set()
    u1: set[tuple[int, int]] = set()
    owners_u0: dict[tuple[int, int], set[str]] = defaultdict(set)
    owners_u1: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in _read_csv(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv"):
        if not _parse_bool(row.get("reprojection_success")):
            continue
        parsed = _parse_obs(row.get("best_mask_observation_id"))
        if parsed is None:
            continue
        row_scene, frame_id, mask_id = parsed
        if row_scene != scene or mask_id <= 0:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        key = (int(frame_id), int(mask_id))
        u1.add(key)
        owners_u1[key].add(candidate_id)
        if candidate_id in selected_candidates:
            u0.add(key)
            owners_u0[key].add(candidate_to_object.get(candidate_id, candidate_id))
    return {"U0": u0, "U1": u1, "owners_U0": owners_u0, "owners_U1": owners_u1}


def _representative_pairs(pipeline_root: Path, scene: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    path = pipeline_root / "representative_observations/representative_mask_rows.csv"
    if not path.exists():
        return out
    for row in _read_csv(path):
        if row.get("scene") != scene:
            continue
        parsed = _parse_obs(row.get("mask_observation_id"))
        if parsed is not None and parsed[0] == scene:
            out.add((int(parsed[1]), int(parsed[2])))
            continue
        try:
            frame_id = int(float(row.get("frame_id") or ""))
            mask_id = int(float(row.get("mask_id") or ""))
        except ValueError:
            continue
        if mask_id > 0:
            out.add((frame_id, mask_id))
    return out


def _duplicate_risk(owners: dict[tuple[int, int], set[str]] | None, mask_count: int) -> tuple[int, float]:
    if not owners:
        return 0, 0.0
    duplicate = sum(1 for values in owners.values() if len(values) > 1)
    return int(duplicate), float(duplicate / max(1, len(owners), mask_count))


def _load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    return _read_label_png(path, shape_hw)


def _frame_mask_stats(mask: np.ndarray, gt: np.ndarray) -> dict[int, dict[str, Any]]:
    pos = mask > 0
    if not np.any(pos):
        return {}
    mask_ids, mask_counts = np.unique(mask[pos], return_counts=True)
    gt_pos = gt > 0
    gt_ids, gt_counts = np.unique(gt[gt_pos], return_counts=True) if np.any(gt_pos) else ([], [])
    gt_area = {int(gid): int(count) for gid, count in zip(gt_ids, gt_counts)}
    overlap: dict[int, Counter[int]] = {int(mid): Counter() for mid in mask_ids}
    both = pos & gt_pos
    if np.any(both):
        max_gt = int(np.max(gt[both]))
        base = max_gt + 1
        encoded = mask[both].astype(np.int64) * base + gt[both].astype(np.int64)
        pair_ids, pair_counts = np.unique(encoded, return_counts=True)
        for encoded_value, count in zip(pair_ids, pair_counts):
            mid = int(encoded_value // base)
            gid = int(encoded_value % base)
            overlap.setdefault(mid, Counter())[gid] += int(count)
    stats: dict[int, dict[str, Any]] = {}
    total_pixels = int(mask.size)
    for mid, area in zip(mask_ids, mask_counts):
        mid_i = int(mid)
        area_i = int(area)
        votes = overlap.get(mid_i, Counter())
        if votes:
            majority_gt, majority_inter = votes.most_common(1)[0]
            union = area_i + int(gt_area.get(majority_gt, 0)) - int(majority_inter)
            majority_iou = float(majority_inter / max(1, union))
            majority_purity = float(majority_inter / max(1, area_i))
        else:
            majority_gt = 0
            majority_inter = 0
            majority_iou = 0.0
            majority_purity = 0.0
        stats[mid_i] = {
            "mask_id": mid_i,
            "mask_area": area_i,
            "mask_area_ratio": float(area_i / max(1, total_pixels)),
            "overlap_counts": dict(votes),
            "gt_area": gt_area,
            "majority_gt": int(majority_gt),
            "majority_intersection": int(majority_inter),
            "majority_iou": float(majority_iou),
            "majority_purity": float(majority_purity),
            "positive_gt_count": int(len(votes)),
            "underseg_risk": bool(len(votes) >= 2 and majority_purity < 0.75),
        }
    return stats


def _quality_pass(stats: dict[str, Any], min_area_ratio: float, max_area_ratio: float) -> bool:
    area_ratio = float(stats["mask_area_ratio"])
    if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
        return False
    if bool(stats["underseg_risk"]):
        return False
    if int(stats["majority_gt"]) <= 0:
        return False
    return True


def _oracle_best_assignments(
    stats_by_mask: dict[int, dict[str, Any]],
) -> tuple[dict[int, int], int]:
    gt_ids = sorted({int(gid) for stats in stats_by_mask.values() for gid in stats.get("overlap_counts", {})})
    choices: list[tuple[float, int, int]] = []
    for gid in gt_ids:
        best_iou = 0.0
        best_mid = 0
        for mid, stats in stats_by_mask.items():
            overlaps = stats.get("overlap_counts", {})
            inter = int(overlaps.get(int(gid), 0))
            if inter <= 0:
                continue
            area = int(stats["mask_area"])
            gt_area = int(stats.get("gt_area", {}).get(int(gid), 0))
            iou = float(inter / max(1, area + gt_area - inter))
            if iou > best_iou:
                best_iou = iou
                best_mid = int(mid)
        if best_mid > 0:
            choices.append((best_iou, best_mid, gid))
    by_mask: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for iou, mid, gid in choices:
        by_mask[mid].append((iou, gid))
    duplicate_before_tie = sum(1 for values in by_mask.values() if len(values) > 1)
    out: dict[int, int] = {}
    for mid, values in by_mask.items():
        values = sorted(values, key=lambda item: (-item[0], item[1]))
        out[int(mid)] = int(values[0][1])
    return out, int(duplicate_before_tie)


def _summary_to_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    sf25 = summary.get("score_free_match_at_025", {})
    sf50 = summary.get("score_free_match_at_050", {})
    gt_count = int(summary.get("evaluated_gt_count") or 0)
    recall25 = summary.get("gt_recall_best_iou_ge_025")
    recall50 = summary.get("gt_recall_best_iou_ge_050")
    return {
        "oracle_SF25": sf25.get("recall"),
        "oracle_SF50": sf50.get("recall"),
        "oracle_AP": summary.get("ap"),
        "oracle_AP50": summary.get("ap50"),
        "oracle_AP25": summary.get("ap25"),
        "oracle_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "oracle_pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "coverage_GT_count_at_IoU25": int(round(float(recall25) * gt_count)) if recall25 is not None else None,
        "coverage_GT_count_at_IoU50": int(round(float(recall50) * gt_count)) if recall50 is not None else None,
        "pred_count": summary.get("evaluated_pred_count"),
        "gt_count": summary.get("evaluated_gt_count"),
    }


def _evaluate_oracle_universe(
    *,
    scene: str,
    stride: int,
    mask_dir: Path,
    universe_name: str,
    mapping_mode: str,
    allowed_pairs: set[tuple[int, int]] | None,
    owners: dict[tuple[int, int], set[str]] | None,
    high_quality_only: bool,
    oracle_best_raw: bool,
    min_area_ratio: float,
    max_area_ratio: float,
    output_dir: Path,
) -> dict[str, Any]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    frame_ids = stream.frame_ids(stride=int(stride), max_frames=None)
    shape_hw = tuple(int(value) for value in stream.load_depth(frame_ids[0]).shape)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    mask_count = 0
    small_count = 0
    large_count = 0
    underseg_count = 0
    missing_mask_frames = 0
    oracle_best_duplicate_before_tie = 0
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        used_masks = 0
        raw_masks = 0
        if mask is None:
            missing_mask_frames += 1
        else:
            stats_by_mask = _frame_mask_stats(mask, gt)
            raw_masks = len(stats_by_mask)
            oracle_assignments: dict[int, int] = {}
            if oracle_best_raw:
                oracle_assignments, dup = _oracle_best_assignments(stats_by_mask)
                oracle_best_duplicate_before_tie += dup
            for mask_id, stats in stats_by_mask.items():
                key = (int(frame_id), int(mask_id))
                if allowed_pairs is not None and key not in allowed_pairs:
                    continue
                if high_quality_only and not _quality_pass(stats, min_area_ratio, max_area_ratio):
                    continue
                if stats["mask_area_ratio"] < min_area_ratio:
                    small_count += 1
                if stats["mask_area_ratio"] > max_area_ratio:
                    large_count += 1
                if stats["underseg_risk"]:
                    underseg_count += 1
                if oracle_best_raw:
                    label = int(oracle_assignments.get(mask_id, 0))
                else:
                    label = int(stats["majority_gt"])
                if label <= 0:
                    continue
                pred[mask == int(mask_id)] = label
                mask_count += 1
                used_masks += 1
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "universe_name": universe_name,
                "raw_mask_count": int(raw_masks),
                "used_mask_count": int(used_masks),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
            }
        )
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    top_rows = _top_iou_rows(iou, pred_ids, gt_ids, top_k=50)
    duplicate_count, duplicate_rate = _duplicate_risk(owners, mask_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_rows.csv", frame_rows)
    _write_csv(output_dir / "top_iou_pairs.csv", top_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "scene_id": scene,
            "universe_name": universe_name,
            "mapping_mode": mapping_mode,
            "diagnostic_only": True,
            "uses_gt_for_prediction": True,
            "forbidden_for_method_table": True,
            "summary": summary,
            "top_iou_pairs": top_rows,
        },
    )
    metrics = _summary_to_metrics(summary)
    return {
        "scene_id": scene,
        "universe_name": universe_name,
        "mapping_mode": mapping_mode,
        "source_scope": "current_run_v67_reanalysis",
        "frame_count": int(len(frame_ids)),
        "mask_count": int(mask_count),
        "mean_masks_per_frame": float(mask_count / max(1, len(frame_ids))),
        "missing_mask_frame_count": int(missing_mask_frames),
        "duplicate_risk_count": int(duplicate_count + oracle_best_duplicate_before_tie),
        "duplicate_risk": float(duplicate_rate),
        "oracle_best_duplicate_before_tie": int(oracle_best_duplicate_before_tie),
        "underseg_rate_diagnostic": float(underseg_count / max(1, mask_count)),
        "small_mask_rate": float(small_count / max(1, mask_count)),
        "large_mask_rate": float(large_count / max(1, mask_count)),
        "uses_gt_for_prediction": True,
        "forbidden_for_method_table": True,
        "diagnostic_only": True,
        "output_summary": _rel(output_dir / "summary.json"),
        **metrics,
    }


def _colorize_labels(label: np.ndarray) -> np.ndarray:
    arr = np.asarray(label, dtype=np.int64)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    ids = [int(value) for value in np.unique(arr) if int(value) > 0]
    for value in ids:
        color = np.asarray(
            [
                (37 * value + 17) % 255,
                (67 * value + 53) % 255,
                (97 * value + 101) % 255,
            ],
            dtype=np.uint8,
        )
        out[arr == value] = color
    return out


def _write_visual_samples(scene: str, stride: int, mask_dir: Path, output_root: Path, max_frames: int) -> list[dict[str, Any]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    frame_ids = stream.frame_ids(stride=stride, max_frames=None)
    if not frame_ids:
        return []
    sample_ids = [frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]][: int(max_frames)]
    rows: list[dict[str, Any]] = []
    for frame_id in sample_ids:
        rgb = stream.load_rgb(int(frame_id))
        shape_hw = tuple(int(v) for v in stream.load_depth(int(frame_id)).shape)
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        if mask is None:
            mask = np.zeros(shape_hw, dtype=np.int64)
        rgb_small = cv2.resize(rgb, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != shape_hw else rgb
        panel = np.concatenate(
            [
                rgb_small,
                _colorize_labels(gt),
                _colorize_labels(mask),
            ],
            axis=1,
        )
        path = output_root / f"{scene}_frame{int(frame_id):06d}_rgb_gt_rawmask.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        rows.append({"scene_id": scene, "frame_id": int(frame_id), "visualization_path": _rel(path)})
    return rows


def _stream3d_diagnostic_rows(v66_mv_ap_root: Path, scenes: list[str]) -> list[dict[str, Any]]:
    path = v66_mv_ap_root / "mv_ap_rows.csv"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for row in _read_csv(path):
        if row.get("scene_id") not in scenes:
            continue
        if row.get("method") != "Stream3D_constant" or str(row.get("stride")) != "5":
            continue
        out.append(
            {
                "scene_id": row.get("scene_id"),
                "universe_name": "U5_Stream3D_key_masks_diagnostic",
                "mapping_mode": "stream3d_rendered_diagnostic_not_raw_mask_oracle",
                "source_scope": "prior_v66_diagnostic",
                "frame_count": row.get("frame_count"),
                "mask_count": row.get("pred_count"),
                "mean_masks_per_frame": "",
                "missing_mask_frame_count": row.get("missing_mask_frame_count", ""),
                "duplicate_risk_count": "",
                "duplicate_risk": "",
                "oracle_best_duplicate_before_tie": "",
                "underseg_rate_diagnostic": "",
                "small_mask_rate": "",
                "large_mask_rate": "",
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": True,
                "diagnostic_only": True,
                "output_summary": row.get("output_summary", ""),
                "oracle_SF25": row.get("score_free_match25_recall"),
                "oracle_SF50": row.get("score_free_match50_recall"),
                "oracle_AP": row.get("AP"),
                "oracle_AP50": row.get("AP50"),
                "oracle_AP25": row.get("AP25"),
                "oracle_GT_best_IoU_mean": row.get("gt_best_iou_mean"),
                "oracle_pred_best_IoU_median": row.get("pred_best_iou_median"),
                "coverage_GT_count_at_IoU25": "",
                "coverage_GT_count_at_IoU50": "",
                "pred_count": row.get("pred_count"),
                "gt_count": row.get("gt_count"),
            }
        )
    return out


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_universe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_universe[str(row["universe_name"])].append(row)
    numeric_mean_fields = [
        "oracle_SF25",
        "oracle_SF50",
        "oracle_AP",
        "oracle_AP50",
        "oracle_AP25",
        "oracle_GT_best_IoU_mean",
        "oracle_pred_best_IoU_median",
        "mean_masks_per_frame",
        "duplicate_risk",
        "underseg_rate_diagnostic",
        "small_mask_rate",
        "large_mask_rate",
    ]
    sum_fields = ["mask_count", "frame_count", "coverage_GT_count_at_IoU25", "coverage_GT_count_at_IoU50"]
    out: list[dict[str, Any]] = []
    for universe, group in sorted(by_universe.items()):
        row: dict[str, Any] = {
            "universe_name": universe,
            "scene_count": int(len(group)),
            "mapping_mode": group[0].get("mapping_mode", ""),
            "source_scope": group[0].get("source_scope", ""),
            "uses_gt_for_prediction": group[0].get("uses_gt_for_prediction", ""),
            "forbidden_for_method_table": group[0].get("forbidden_for_method_table", ""),
            "diagnostic_only": group[0].get("diagnostic_only", ""),
        }
        for field in numeric_mean_fields:
            values = [value for value in (_to_float(item.get(field)) for item in group) if value is not None]
            row[f"{field}_mean"] = float(np.mean(values)) if values else None
        for field in sum_fields:
            values = [value for value in (_to_float(item.get(field)) for item in group) if value is not None]
            row[f"{field}_sum"] = float(np.sum(values)) if values else None
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    visual_root = _project(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    stride = int(args.stride)
    per_scene_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    visualization_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        variant = _best_variant(pipeline_root)
        pair_sets = _ledger_pair_sets(pipeline_root, scene, variant)
        representative_pairs = _representative_pairs(pipeline_root, scene)
        visualization_rows.extend(_write_visual_samples(scene, stride, mask_dir, visual_root, int(args.visual_frames_per_scene)))
        universe_specs = [
            ("U0_current_selected_masks", pair_sets["U0"], pair_sets["owners_U0"], False, False, "oracle_by_majority_GT"),
            ("U1_all_reprojection_candidates", pair_sets["U1"], pair_sets["owners_U1"], False, False, "oracle_by_majority_GT"),
            ("U2_all_representative_masks", representative_pairs, None, False, False, "oracle_by_majority_GT"),
            ("U3_all_CropFormer_masks_stride5", None, None, False, False, "oracle_by_majority_GT"),
            ("U4_high_quality_CropFormer_masks", None, None, True, False, "oracle_by_majority_GT_with_quality_filter"),
            ("U6_oracle_best_raw_masks", None, None, False, True, "oracle_by_best_GT_forbidden_upper_bound"),
        ]
        for universe_name, allowed_pairs, owners, high_quality_only, oracle_best_raw, mapping_mode in universe_specs:
            row = _evaluate_oracle_universe(
                scene=scene,
                stride=stride,
                mask_dir=mask_dir,
                universe_name=universe_name,
                mapping_mode=mapping_mode,
                allowed_pairs=allowed_pairs,
                owners=owners,
                high_quality_only=bool(high_quality_only),
                oracle_best_raw=bool(oracle_best_raw),
                min_area_ratio=float(args.min_area_ratio),
                max_area_ratio=float(args.max_area_ratio),
                output_dir=output_root / "runs" / f"{scene}_{universe_name}",
            )
            row["pipeline_root"] = _rel(pipeline_root)
            row["mask_dir"] = _rel(mask_dir)
            row["best_objectlet_variant"] = variant
            per_scene_rows.append(row)
    per_scene_rows.extend(_stream3d_diagnostic_rows(_project(args.v66_mv_ap_root), scenes))
    metric_rows = _aggregate(per_scene_rows)

    by_name = {row["universe_name"]: row for row in metric_rows}
    u0_sf50 = _to_float(by_name.get("U0_current_selected_masks", {}).get("oracle_SF50_mean"))
    u1_sf50 = _to_float(by_name.get("U1_all_reprojection_candidates", {}).get("oracle_SF50_mean"))
    u2_sf50 = _to_float(by_name.get("U2_all_representative_masks", {}).get("oracle_SF50_mean"))
    u3_sf50 = _to_float(by_name.get("U3_all_CropFormer_masks_stride5", {}).get("oracle_SF50_mean"))
    u4_sf50 = _to_float(by_name.get("U4_high_quality_CropFormer_masks", {}).get("oracle_SF50_mean"))
    raw_best_sf50 = max([value for value in [u3_sf50, u4_sf50] if value is not None], default=None)
    selected_gap = raw_best_sf50 - u0_sf50 if raw_best_sf50 is not None and u0_sf50 is not None else None
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "raw_u3_or_u4_oracle_sf50_ge_0p20": raw_best_sf50 is not None and raw_best_sf50 >= 0.20,
        "raw_u3_or_u4_oracle_ge_u0_plus_0p20": selected_gap is not None and selected_gap >= 0.20,
        "u2_high_u0_low": u2_sf50 is not None and u0_sf50 is not None and u2_sf50 >= u0_sf50 + 0.20,
        "u1_high_u0_low": u1_sf50 is not None and u0_sf50 is not None and u1_sf50 >= u0_sf50 + 0.20,
    }
    decision = "INCONCLUSIVE"
    if not gate["all_pipeline_roots_available"]:
        decision = "MISSING_INPUT"
    elif raw_best_sf50 is not None and raw_best_sf50 < 0.20:
        decision = "MASK_SOURCE_OR_ALIGNMENT_BLOCKER"
    elif selected_gap is not None and selected_gap >= 0.20:
        decision = "CANDIDATE_UNIVERSE_LOSS"
    elif gate["u2_high_u0_low"]:
        decision = "OBJECTLET_SELECTION_BLOCKER"
    elif gate["u1_high_u0_low"]:
        decision = "LOCAL_OBJECTLET_SELECTION_BLOCKER"
    else:
        decision = "NO_STRONG_RAW_HEADROOM"

    _write_csv(output_root / "per_scene_universe_rows.csv", per_scene_rows)
    _write_csv(output_root / "universe_metric_rows.csv", metric_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_csv(output_root / "visualization_rows.csv", visualization_rows)
    payload = {
        "phase": "v67_mask_universe",
        "diagnostic_only": True,
        "scenes": scenes,
        "stride": stride,
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "decision": decision,
        "u0_selected_oracle_sf50_mean": u0_sf50,
        "u1_reprojection_oracle_sf50_mean": u1_sf50,
        "u2_representative_oracle_sf50_mean": u2_sf50,
        "u3_raw_cropformer_oracle_sf50_mean": u3_sf50,
        "u4_high_quality_raw_oracle_sf50_mean": u4_sf50,
        "raw_best_u3_u4_oracle_sf50_mean": raw_best_sf50,
        "raw_best_minus_u0_sf50": selected_gap,
        "rows": {
            "universe_metric_rows_csv": _rel(output_root / "universe_metric_rows.csv"),
            "per_scene_universe_rows_csv": _rel(output_root / "per_scene_universe_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
            "visualization_rows_csv": _rel(output_root / "visualization_rows.csv"),
        },
        "visual_root": _rel(visual_root),
        "oracle_policy": {
            "oracle_universe_rows_use_gt_for_prediction": True,
            "oracle_universe_rows_forbidden_for_method_table": True,
            "stream3d_u5_is_prior_diagnostic_not_method": True,
        },
        "notes": [
            "U0/U1/U2/U3/U4/U6 are recomputed in this v67 run from mask PNGs and ScanNet GT instance maps.",
            "All GT-grouped oracle rows are diagnostic upper bounds and forbidden for method tables.",
            "U5 is imported as a prior v66 Stream3D diagnostic sanity row, not a v67 method result.",
        ],
    }
    _write_json(output_root / "mask_universe_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v67 Phase 1 mask-universe oracle audit.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v67_mask_universe")
    parser.add_argument("--visual-root", default="outputs/audit/v67_visualizations/mask_universe")
    parser.add_argument("--v66-mv-ap-root", default="outputs/audit/v66_scene_mv_ap_probe5_full")
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--visual-frames-per-scene", type=int, default=3)
    return parser.parse_args()
