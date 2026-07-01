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


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import (  # noqa: E402
    SparseSceneIoU,
    _load_gt_2d,
    _read_label_png,
    _sha256,
    _summarize_iou,
    _write_csv,
    _write_json,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
    _rel,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_json_list(value: Any) -> list[str]:
    if value in ("", None):
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _parse_mask_observation_id(value: Any) -> tuple[str, int, int] | None:
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
        raise RuntimeError(f"best real objectlet variant missing: {pipeline_root}")
    return variant


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _chunk_rows(pipeline_root: Path, scene: str) -> list[dict[str, Any]]:
    rows = [row for row in _load_csv_rows(pipeline_root / "chunk_universe/chunk_rows.csv") if row.get("scene") == scene]
    return sorted(rows, key=lambda row: int(float(row.get("chunk_index") or 0)))


def _local_metric_lookup(pipeline_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = pipeline_root / "local_reproduction/local_metric_rows.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_csv_rows(path):
        out[(str(row.get("variant") or ""), str(row.get("chunk_id") or ""))] = row
    return out


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return _read_label_png(path, shape_hw)


def _frame_data(scene: str, frame_ids: list[int], mask_dir: Path) -> list[dict[str, Any]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(value) for value in stream.load_depth(frame_ids[0]).shape)
    out: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, frame_id, shape_hw)
        mask = _read_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        out.append({"frame_id": int(frame_id), "gt": gt, "mask": mask})
    return out


def _selected_candidates(
    *,
    objectlet_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    variant: str,
) -> tuple[dict[str, int], dict[int, str]]:
    object_to_idx: dict[str, int] = {}
    candidate_to_object_idx: dict[str, int] = {}
    object_idx_to_id: dict[int, str] = {}
    for row in objectlet_rows:
        if row.get("scene") != scene or row.get("chunk_id") != chunk_id or row.get("variant") != variant:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        object_id = str(row.get("objectlet_id") or "").strip()
        if not candidate_id or not object_id:
            continue
        if object_id not in object_to_idx:
            object_to_idx[object_id] = len(object_to_idx) + 1
            object_idx_to_id[object_to_idx[object_id]] = object_id
        candidate_to_object_idx[candidate_id] = object_to_idx[object_id]
    return candidate_to_object_idx, object_idx_to_id


def _mapping_from_candidates(
    *,
    ledger_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    candidate_to_object_idx: dict[str, int],
    wta: bool,
    oracle_gt: bool = False,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    pair_votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    pair_objects: dict[tuple[int, int], set[int]] = defaultdict(set)
    used_rows = 0
    for row in ledger_rows:
        if row.get("scene") != scene or row.get("chunk_id") != chunk_id:
            continue
        if not _parse_bool(row.get("reprojection_success")):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        object_idx = int(candidate_to_object_idx.get(candidate_id, 0))
        if object_idx <= 0:
            continue
        parsed = _parse_mask_observation_id(row.get("best_mask_observation_id"))
        if parsed is None:
            continue
        row_scene, frame_id, mask_id = parsed
        if row_scene != scene or mask_id <= 0:
            continue
        if oracle_gt:
            gt_id = int(float(row.get("diagnostic_best_gt") or 0))
            if gt_id <= 0:
                continue
            object_idx = gt_id
        key = (int(frame_id), int(mask_id))
        weight = int(float(row.get("best_mask_related_carrier_count") or 1))
        pair_votes[key][object_idx] += max(1, weight)
        pair_objects[key].add(object_idx)
        used_rows += 1
    mapping: dict[tuple[int, int], int] = {}
    for key, objects in pair_objects.items():
        if wta:
            mapping[key] = min(objects, key=lambda object_idx: (-pair_votes[key][object_idx], object_idx))
        else:
            mapping[key] = min(objects)
    conflict_pair_count = int(sum(1 for objects in pair_objects.values() if len(objects) > 1))
    return mapping, {
        "support_pair_count": int(len(pair_objects)),
        "duplicate_frame_mask_conflict_pairs": conflict_pair_count,
        "duplicate_frame_mask_conflict_rate": float(conflict_pair_count / max(1, len(pair_objects))),
        "max_objects_per_frame_mask": max((len(objects) for objects in pair_objects.values()), default=0),
        "used_ledger_row_count": int(used_rows),
    }


def _candidate_rows_by_chunk(candidate_rows: list[dict[str, Any]], scene: str, chunk_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in candidate_rows:
        if row.get("scene") == scene and row.get("chunk_id") == chunk_id:
            rows.append(row)
    return rows


def _greedy_set_cover_candidates(
    rows: list[dict[str, Any]],
    *,
    penalty: float,
    max_selected: int,
) -> list[str]:
    components_by_candidate: dict[str, set[str]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        components = set(_parse_json_list(row.get("component_ids")))
        if candidate_id and components:
            components_by_candidate[candidate_id] = components
    uncovered = set().union(*components_by_candidate.values()) if components_by_candidate else set()
    selected: list[str] = []
    while uncovered and len(selected) < int(max_selected):
        best_id = ""
        best_gain = 0
        best_score = float("-inf")
        for candidate_id, components in components_by_candidate.items():
            if candidate_id in selected:
                continue
            gain = len(components & uncovered)
            score = float(gain) - float(penalty)
            if score > best_score or (score == best_score and gain > best_gain):
                best_id = candidate_id
                best_gain = gain
                best_score = score
        if not best_id or best_gain <= 0 or best_score <= 0:
            break
        selected.append(best_id)
        uncovered -= components_by_candidate[best_id]
    return selected


def _greedy_candidate_mapping(
    *,
    candidate_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    penalty: float,
    max_selected: int,
    wta: bool,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    chunk_candidates = _candidate_rows_by_chunk(candidate_rows, scene, chunk_id)
    selected = _greedy_set_cover_candidates(chunk_candidates, penalty=penalty, max_selected=max_selected)
    candidate_to_object_idx = {candidate_id: idx + 1 for idx, candidate_id in enumerate(selected)}
    mapping, diag = _mapping_from_candidates(
        ledger_rows=ledger_rows,
        scene=scene,
        chunk_id=chunk_id,
        candidate_to_object_idx=candidate_to_object_idx,
        wta=wta,
    )
    total_components = set()
    covered_components = set()
    component_lookup: dict[str, set[str]] = {}
    for row in chunk_candidates:
        candidate_id = str(row.get("candidate_id") or "")
        components = set(_parse_json_list(row.get("component_ids")))
        component_lookup[candidate_id] = components
        total_components |= components
    for candidate_id in selected:
        covered_components |= component_lookup.get(candidate_id, set())
    diag.update(
        {
            "selected_mask_count": int(len(selected)),
            "set_cover_penalty": float(penalty),
            "set_cover_max_selected": int(max_selected),
            "coverage_ratio": float(len(covered_components) / max(1, len(total_components))),
        }
    )
    return mapping, diag


def _evaluate_frame_data(
    *,
    frame_data: list[dict[str, Any]],
    variant: str,
    mapping: dict[tuple[int, int], int] | None,
    raw_per_frame_masks: bool,
) -> tuple[dict[str, Any], np.ndarray, list[int], list[int]]:
    acc = SparseSceneIoU()
    for item in frame_data:
        frame_id = int(item["frame_id"])
        gt = np.asarray(item["gt"], dtype=np.int64)
        mask = item["mask"]
        pred = np.zeros(gt.shape, dtype=np.int64)
        if mask is not None:
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                if raw_per_frame_masks:
                    pred_label = int(frame_id) * 100000 + mask_id
                else:
                    pred_label = int((mapping or {}).get((frame_id, mask_id), 0))
                if pred_label > 0:
                    pred[mask == mask_id] = pred_label
        acc.add(pred, gt)
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    summary["variant"] = variant
    return summary, iou, pred_ids, gt_ids


def _frag_overmerge_means(iou: np.ndarray) -> tuple[float | None, float | None]:
    if iou.size == 0:
        return None, None
    if iou.shape[1]:
        frag = [int(np.count_nonzero(iou[:, col] >= 0.25)) for col in range(iou.shape[1])]
        fragmentation_mean = float(np.mean(frag)) if frag else None
    else:
        fragmentation_mean = None
    if iou.shape[0]:
        over = [int(np.count_nonzero(iou[row, :] >= 0.25)) for row in range(iou.shape[0])]
        overmerge_mean = float(np.mean(over)) if over else None
    else:
        overmerge_mean = None
    return fragmentation_mean, overmerge_mean


def _score_free(summary: dict[str, Any]) -> float | None:
    value = (summary.get("score_free_match_at_050") or {}).get("recall")
    return None if value is None else float(value)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    return float(np.mean(real)) if real else None


def _summarize_variant(rows: list[dict[str, Any]], scene: str, variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["scene_id"] == scene and row["variant"] == variant]
    return {
        "scene_id": scene,
        "variant": variant,
        "chunk_count": len(subset),
        "local_AP50_mean": _mean([_float_or_none(row.get("local_AP50")) for row in subset]),
        "local_AP25_mean": _mean([_float_or_none(row.get("local_AP25")) for row in subset]),
        "local_score_free_match50_recall_mean": _mean([_float_or_none(row.get("local_score_free_match50_recall")) for row in subset]),
        "local_GT_best_IoU_mean_mean": _mean([_float_or_none(row.get("local_GT_best_IoU_mean")) for row in subset]),
        "local_pred_best_IoU_median_mean": _mean([_float_or_none(row.get("local_pred_best_IoU_median")) for row in subset]),
        "local_duplicate_frame_mask_conflict_rate_mean": _mean(
            [_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate")) for row in subset]
        ),
        "local_object_count_mean": _mean([_float_or_none(row.get("local_object_count")) for row in subset]),
        "local_gt_count_mean": _mean([_float_or_none(row.get("local_gt_count")) for row in subset]),
        "component_coverage_ratio_mean": _mean([_float_or_none(row.get("component_coverage_ratio")) for row in subset]),
        "same_frame_conflict_rate_mean": _mean([_float_or_none(row.get("same_frame_conflict_rate")) for row in subset]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    stride = int(args.stride)
    chunk_rows_out: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    variants_seen: set[tuple[str, str]] = set()
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        best_variant = _best_variant(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=stride, max_frames=None)
        objectlet_rows = _load_csv_rows(pipeline_root / "local_objectlets/objectlet_rows.csv")
        ledger_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv")
        candidate_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/candidate_rows.csv")
        local_metrics = _local_metric_lookup(pipeline_root)
        for chunk in _chunk_rows(pipeline_root, scene):
            t0 = time.time()
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            data = _frame_data(scene, frame_ids, mask_dir)
            eval_specs: list[dict[str, Any]] = []
            candidate_to_object_idx, _object_idx_to_id = _selected_candidates(
                objectlet_rows=objectlet_rows,
                scene=scene,
                chunk_id=chunk_id,
                variant=best_variant,
            )
            l1_mapping, l1_diag = _mapping_from_candidates(
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                candidate_to_object_idx=candidate_to_object_idx,
                wta=False,
            )
            l5_mapping, l5_diag = _mapping_from_candidates(
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                candidate_to_object_idx=candidate_to_object_idx,
                wta=True,
            )
            l7_mapping, l7_diag = _mapping_from_candidates(
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                candidate_to_object_idx=candidate_to_object_idx,
                wta=True,
                oracle_gt=True,
            )
            k1_mapping, k1_diag = _greedy_candidate_mapping(
                candidate_rows=candidate_rows,
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                penalty=0.0,
                max_selected=int(args.max_k_candidates_per_chunk),
                wta=False,
            )
            k2_mapping, k2_diag = _greedy_candidate_mapping(
                candidate_rows=candidate_rows,
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                penalty=float(args.k2_penalty),
                max_selected=int(args.max_k_candidates_per_chunk),
                wta=True,
            )
            eval_specs.extend(
                [
                    {
                        "variant": "L0_raw_CropFormer_per_frame_masks",
                        "mapping": None,
                        "raw": True,
                        "diag": {
                            "selected_mask_count": "",
                            "support_pair_count": "",
                            "duplicate_frame_mask_conflict_rate": 0.0,
                            "uses_gt_for_prediction": False,
                            "forbidden_for_method_table": True,
                        },
                    },
                    {
                        "variant": f"L1_current_best_{best_variant}",
                        "mapping": l1_mapping,
                        "raw": False,
                        "diag": {**l1_diag, "selected_mask_count": len(candidate_to_object_idx), "uses_gt_for_prediction": False},
                    },
                    {
                        "variant": f"L5_WTA_{best_variant}",
                        "mapping": l5_mapping,
                        "raw": False,
                        "diag": {**l5_diag, "selected_mask_count": len(candidate_to_object_idx), "uses_gt_for_prediction": False},
                    },
                    {
                        "variant": "K1_greedy_component_set_cover",
                        "mapping": k1_mapping,
                        "raw": False,
                        "diag": {**k1_diag, "uses_gt_for_prediction": False},
                    },
                    {
                        "variant": f"K2_greedy_component_set_cover_penalty{float(args.k2_penalty):g}_WTA",
                        "mapping": k2_mapping,
                        "raw": False,
                        "diag": {**k2_diag, "uses_gt_for_prediction": False},
                    },
                    {
                        "variant": f"L7_oracle_gt_group_selected_masks_{best_variant}",
                        "mapping": l7_mapping,
                        "raw": False,
                        "diag": {
                            **l7_diag,
                            "selected_mask_count": len(candidate_to_object_idx),
                            "uses_gt_for_prediction": True,
                            "forbidden_for_method_table": True,
                        },
                    },
                ]
            )
            for spec in eval_specs:
                summary, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                    frame_data=data,
                    variant=str(spec["variant"]),
                    mapping=spec["mapping"],
                    raw_per_frame_masks=bool(spec["raw"]),
                )
                frag_mean, over_mean = _frag_overmerge_means(iou)
                local_metric = local_metrics.get((best_variant, chunk_id), {}) if str(spec["variant"]).startswith(("L1_", "L5_", "L7_")) else {}
                if str(spec["variant"]).startswith("L0_"):
                    local_metric = local_metrics.get(("L0_raw_U32_components", chunk_id), {})
                row = {
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "variant": str(spec["variant"]),
                    "chunk_frame_count": int(len(frame_ids)),
                    "frame_min": int(frame_ids[0]),
                    "frame_max": int(frame_ids[-1]),
                    "mask_count": int(float(chunk.get("mask_count") or 0)),
                    "selected_mask_count": spec["diag"].get("selected_mask_count", ""),
                    "local_object_count": summary.get("evaluated_pred_count"),
                    "local_gt_count": summary.get("evaluated_gt_count"),
                    "local_AP": summary.get("ap"),
                    "local_AP50": summary.get("ap50"),
                    "local_AP25": summary.get("ap25"),
                    "local_score_free_match50_recall": _score_free(summary),
                    "local_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
                    "local_pred_best_IoU_median": summary.get("pred_best_iou_median"),
                    "local_duplicate_frame_mask_conflict_rate": spec["diag"].get("duplicate_frame_mask_conflict_rate", 0.0),
                    "local_fragmentation_mean": frag_mean,
                    "local_overmerge_mean": over_mean,
                    "component_coverage_ratio": local_metric.get("component_coverage_ratio", spec["diag"].get("coverage_ratio", "")),
                    "same_frame_conflict_rate": local_metric.get("conflict_rate", ""),
                    "support_pair_count": spec["diag"].get("support_pair_count", ""),
                    "duplicate_frame_mask_conflict_pairs": spec["diag"].get("duplicate_frame_mask_conflict_pairs", ""),
                    "uses_gt_for_prediction": bool(spec["diag"].get("uses_gt_for_prediction", False)),
                    "forbidden_for_method_table": bool(spec["diag"].get("forbidden_for_method_table", False)),
                    "diagnostic_only": True,
                    "pipeline_root": _rel(pipeline_root),
                    "runtime_sec_chunk_all_variants_so_far": float(time.time() - t0),
                }
                if "set_cover_penalty" in spec["diag"]:
                    row["set_cover_penalty"] = spec["diag"].get("set_cover_penalty")
                    row["set_cover_max_selected"] = spec["diag"].get("set_cover_max_selected")
                    row["set_cover_coverage_ratio"] = spec["diag"].get("coverage_ratio")
                chunk_rows_out.append(row)
                variants_seen.add((scene, str(spec["variant"])))
    scene_summary_rows = [_summarize_variant(chunk_rows_out, scene, variant) for scene, variant in sorted(variants_seen)]
    l1_rows = [row for row in scene_summary_rows if row["variant"].startswith("L1_current_best_")]
    k_rows = [row for row in scene_summary_rows if row["variant"].startswith("K")]
    oracle_rows = [row for row in scene_summary_rows if row["variant"].startswith("L7_oracle")]
    l1_mean_sf50 = _mean([_float_or_none(row.get("local_score_free_match50_recall_mean")) for row in l1_rows])
    l1_mean_gt_best = _mean([_float_or_none(row.get("local_GT_best_IoU_mean_mean")) for row in l1_rows])
    l1_mean_dup = _mean([_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate_mean")) for row in l1_rows])
    l1_mean_ap50 = _mean([_float_or_none(row.get("local_AP50_mean")) for row in l1_rows])
    k_best_sf50 = max((_float_or_none(row.get("local_score_free_match50_recall_mean")) or 0.0 for row in k_rows), default=0.0)
    k_best_gt_best = max((_float_or_none(row.get("local_GT_best_IoU_mean_mean")) or 0.0 for row in k_rows), default=0.0)
    k_best_dup = min((_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate_mean")) or 1.0 for row in k_rows), default=None)
    oracle_best_sf50 = max((_float_or_none(row.get("local_score_free_match50_recall_mean")) or 0.0 for row in oracle_rows), default=0.0)
    gate = {
        "all_pipeline_roots_available": len(missing_rows) == 0,
        "l1_local_score_free_match50_recall_mean_ge_0p40": l1_mean_sf50 is not None and l1_mean_sf50 >= 0.40,
        "l1_local_GT_best_IoU_mean_ge_0p35": l1_mean_gt_best is not None and l1_mean_gt_best >= 0.35,
        "l1_local_duplicate_frame_mask_conflict_rate_le_0p02": l1_mean_dup is not None and l1_mean_dup <= 0.02,
        "l1_local_AP50_ge_0p05": l1_mean_ap50 is not None and l1_mean_ap50 >= 0.05,
        "k_best_score_free_improves_l1_by_0p15": l1_mean_sf50 is not None and (k_best_sf50 - l1_mean_sf50) >= 0.15,
        "k_best_GT_best_improves_l1_by_0p10": l1_mean_gt_best is not None and (k_best_gt_best - l1_mean_gt_best) >= 0.10,
        "k_best_duplicate_rate_le_0p02": k_best_dup is not None and k_best_dup <= 0.02,
        "oracle_selected_masks_has_headroom": l1_mean_sf50 is not None and (oracle_best_sf50 - l1_mean_sf50) >= 0.15,
    }
    gate["l1_local_pass"] = (
        gate["all_pipeline_roots_available"]
        and gate["l1_local_score_free_match50_recall_mean_ge_0p40"]
        and gate["l1_local_GT_best_IoU_mean_ge_0p35"]
        and gate["l1_local_duplicate_frame_mask_conflict_rate_le_0p02"]
        and gate["l1_local_AP50_ge_0p05"]
    )
    gate["k_repair_pass"] = (
        gate["k_best_score_free_improves_l1_by_0p15"]
        and gate["k_best_GT_best_improves_l1_by_0p10"]
        and gate["k_best_duplicate_rate_le_0p02"]
    )
    _write_csv(output_root / "local_chunk_rows.csv", chunk_rows_out)
    _write_csv(output_root / "local_scene_summary_rows.csv", scene_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v66_local_chunk_eval",
        "diagnostic_only": True,
        "scenes": scenes,
        "stride": stride,
        "gate": gate,
        "l1_means": {
            "local_score_free_match50_recall_mean": l1_mean_sf50,
            "local_GT_best_IoU_mean": l1_mean_gt_best,
            "local_duplicate_frame_mask_conflict_rate": l1_mean_dup,
            "local_AP50": l1_mean_ap50,
        },
        "k_best": {
            "local_score_free_match50_recall_mean": k_best_sf50,
            "local_GT_best_IoU_mean": k_best_gt_best,
            "local_duplicate_frame_mask_conflict_rate": k_best_dup,
        },
        "oracle_best_score_free_match50_recall_mean": oracle_best_sf50,
        "rows": {
            "local_chunk_rows_csv": _rel(output_root / "local_chunk_rows.csv"),
            "local_scene_summary_rows_csv": _rel(output_root / "local_scene_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Local MV-AP is computed within each chunk frame window using ScanNet 2D instance maps only as diagnostic labels.",
            "K1/K2 greedy set-cover variants use candidate component_ids and reprojection ledger support only; they do not use GT labels for prediction.",
            "L7 oracle groups selected masks by diagnostic_best_gt and is forbidden for method tables.",
        ],
    }
    _write_json(output_root / "local_chunk_eval_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "local_chunk_eval_summary.json",
        output_root / "local_chunk_rows.csv",
        output_root / "local_scene_summary_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v66 chunk-local multi-view AP diagnostics and local set-cover probes.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v66_local_chunk_eval")
    parser.add_argument("--max-k-candidates-per-chunk", type=int, default=128)
    parser.add_argument("--k2-penalty", type=float, default=32.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
