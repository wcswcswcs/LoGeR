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
    _sha256,
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
    _rel,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _selected_objects(pipeline_root: Path, scene: str, variant: str) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    selected: dict[str, dict[str, Any]] = {}
    object_to_idx: dict[str, int] = {}
    path = pipeline_root / "local_objectlets/objectlet_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene or row.get("variant") != variant:
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            object_id = str(row.get("objectlet_id") or "").strip()
            if not candidate_id or not object_id:
                continue
            if object_id not in object_to_idx:
                object_to_idx[object_id] = len(object_to_idx) + 1
            selected[candidate_id] = {
                **row,
                "objectlet_id": object_id,
                "object_idx": int(object_to_idx[object_id]),
            }
    return selected, object_to_idx


def _load_support(
    *,
    pipeline_root: Path,
    scene: str,
    variant: str,
) -> dict[str, Any]:
    selected, object_to_idx = _selected_objects(pipeline_root, scene, variant)
    pair_votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    pair_objects: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    object_rows: dict[int, dict[str, Any]] = {}
    object_support_frames: dict[int, set[int]] = defaultdict(set)
    object_support_pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    object_gt_votes: dict[int, Counter[int]] = defaultdict(Counter)
    ledger_rows = 0
    used_rows = 0
    skipped_failed_rows = 0
    skipped_unselected_rows = 0
    path = pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ledger_rows += 1
            selected_row = selected.get(str(row.get("candidate_id") or ""))
            if selected_row is None:
                skipped_unselected_rows += 1
                continue
            if not _parse_bool(row.get("reprojection_success")):
                skipped_failed_rows += 1
                continue
            parsed = _parse_mask_observation_id(row.get("best_mask_observation_id"))
            if parsed is None:
                continue
            row_scene, frame_id, mask_id = parsed
            if row_scene != scene or mask_id <= 0:
                continue
            object_idx = int(selected_row["object_idx"])
            object_id = str(selected_row["objectlet_id"])
            key = (int(frame_id), int(mask_id))
            weight = int(float(row.get("best_mask_related_carrier_count") or 1))
            pair_votes[key][object_idx] += max(1, weight)
            pair_objects[key].add(object_idx)
            pair_rows[key].append(
                {
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "mask_id": int(mask_id),
                    "object_idx": object_idx,
                    "objectlet_id": object_id,
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "diagnostic_best_gt": int(float(row.get("diagnostic_best_gt") or 0)),
                    "best_mask_related_carrier_count": weight,
                    "inside_best_mask_ratio": row.get("inside_best_mask_ratio"),
                    "mask_explained_ratio": row.get("mask_explained_ratio"),
                }
            )
            object_rows.setdefault(
                object_idx,
                {
                    "scene_id": scene,
                    "object_idx": object_idx,
                    "objectlet_id": object_id,
                    "source_candidate_id": str(selected_row.get("candidate_id") or ""),
                    "chunk_id": str(selected_row.get("chunk_id") or ""),
                    "component_count": selected_row.get("component_count"),
                    "candidate_success_rate": selected_row.get("candidate_success_rate"),
                    "selection_sort_mode": selected_row.get("selection_sort_mode"),
                },
            )
            object_support_frames[object_idx].add(int(frame_id))
            object_support_pairs[object_idx].add(key)
            gt_id = int(float(row.get("diagnostic_best_gt") or 0))
            if gt_id > 0:
                object_gt_votes[object_idx][gt_id] += 1
            used_rows += 1

    current_mapping: dict[tuple[int, int], int] = {}
    wta_mapping: dict[tuple[int, int], int] = {}
    conflict_rows: list[dict[str, Any]] = []
    for key in sorted(pair_objects):
        object_ids = sorted(pair_objects[key])
        current_mapping[key] = min(object_ids)
        wta_mapping[key] = min(
            object_ids,
            key=lambda object_idx: (-pair_votes[key][object_idx], object_idx),
        )
        if len(object_ids) > 1:
            rows = pair_rows[key]
            conflict_rows.append(
                {
                    "scene_id": scene,
                    "frame_id": int(key[0]),
                    "mask_id": int(key[1]),
                    "object_count": int(len(object_ids)),
                    "object_indices": ",".join(str(value) for value in object_ids),
                    "wta_object_idx": int(wta_mapping[key]),
                    "current_min_object_idx": int(current_mapping[key]),
                    "total_vote_weight": int(sum(pair_votes[key].values())),
                    "wta_vote_weight": int(pair_votes[key][wta_mapping[key]]),
                    "candidate_ids": ",".join(sorted({str(row["candidate_id"]) for row in rows})),
                    "diagnostic_best_gts": ",".join(str(value) for value in sorted({int(row["diagnostic_best_gt"]) for row in rows if int(row["diagnostic_best_gt"]) > 0})),
                }
            )

    object_support_rows: list[dict[str, Any]] = []
    for object_idx in sorted(object_rows):
        frames = sorted(object_support_frames[object_idx])
        gt_votes = object_gt_votes.get(object_idx, Counter())
        top_gt = gt_votes.most_common(1)[0][0] if gt_votes else 0
        object_support_rows.append(
            {
                **object_rows[object_idx],
                "support_frame_count": int(len(frames)),
                "support_pair_count": int(len(object_support_pairs[object_idx])),
                "frame_min": int(frames[0]) if frames else "",
                "frame_max": int(frames[-1]) if frames else "",
                "top_diagnostic_gt": int(top_gt),
                "top_diagnostic_gt_vote_count": int(gt_votes[top_gt]) if top_gt else 0,
                "diagnostic_gt_unique_count": int(len(gt_votes)),
            }
        )
    return {
        "selected": selected,
        "object_to_idx": object_to_idx,
        "ledger_rows": int(ledger_rows),
        "used_rows": int(used_rows),
        "skipped_failed_rows": int(skipped_failed_rows),
        "skipped_unselected_rows": int(skipped_unselected_rows),
        "pair_objects": pair_objects,
        "current_mapping": current_mapping,
        "wta_mapping": wta_mapping,
        "conflict_rows": conflict_rows,
        "object_support_rows": object_support_rows,
    }


def _load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return _read_label_png(path, shape_hw)


def _evaluate_mapping(
    *,
    scene: str,
    stride: int,
    mask_dir: Path,
    mapping: dict[tuple[int, int], int],
    output_dir: Path,
    score_mode: str,
) -> dict[str, Any]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    frame_ids = stream.frame_ids(stride=int(stride), max_frames=None)
    if not frame_ids:
        raise RuntimeError(f"no frames for scene={scene} stride={stride}")
    shape_hw = tuple(int(value) for value in stream.load_depth(frame_ids[0]).shape)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    missing = 0
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        mapped_mask_ids = 0
        positive_mask_pixels = 0
        if mask is None:
            missing += 1
        else:
            positive_mask_pixels = int(np.count_nonzero(mask > 0))
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                object_idx = int(mapping.get((int(frame_id), mask_id), 0))
                if object_idx <= 0:
                    continue
                pred[mask == mask_id] = object_idx
                mapped_mask_ids += 1
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "mask_exists": mask is not None,
                "positive_mask_pixels": positive_mask_pixels,
                "mapped_mask_ids": int(mapped_mask_ids),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
            }
        )
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode=score_mode,
        input_scores=None,
    )
    summary["frame_count"] = int(len(frame_ids))
    summary["missing_mask_frame_count"] = int(missing)
    top_rows = _top_iou_rows(iou, pred_ids, gt_ids, top_k=100)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_rows.csv", frame_rows)
    _write_csv(output_dir / "top_iou_pairs.csv", top_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "phase": "v66_ledger_audit_mapping_eval",
            "scene": scene,
            "stride": int(stride),
            "score_mode": score_mode,
            "summary": summary,
            "top_iou_pairs": top_rows,
        },
    )
    return {
        "summary": summary,
        "iou": iou,
        "pred_ids": pred_ids,
        "gt_ids": gt_ids,
        "top_rows": top_rows,
        "summary_json": _rel(output_dir / "summary.json"),
    }


def _fragmentation_rows(scene: str, iou: np.ndarray, pred_ids: list[int], gt_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (0.10, 0.25):
        if iou.shape[1] == 0:
            continue
        for gt_col, gt_id in enumerate(gt_ids):
            values = iou[:, gt_col] if iou.shape[0] else np.asarray([], dtype=np.float32)
            hits = np.nonzero(values >= threshold)[0]
            if hits.shape[0] <= 1 and (values.size == 0 or float(np.max(values)) >= 0.25):
                continue
            ordered = sorted(hits.tolist(), key=lambda idx: float(values[idx]), reverse=True)
            best = float(np.max(values)) if values.size else 0.0
            rows.append(
                {
                    "scene_id": scene,
                    "threshold": float(threshold),
                    "gt_id": int(gt_id),
                    "fragment_pred_count": int(hits.shape[0]),
                    "best_pred_iou": best,
                    "top_pred_ids": ",".join(str(pred_ids[idx]) for idx in ordered[:10]),
                    "top_pred_ious": ",".join(f"{float(values[idx]):.6f}" for idx in ordered[:10]),
                }
            )
    return rows


def _overmerge_rows(scene: str, iou: np.ndarray, pred_ids: list[int], gt_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (0.10, 0.25):
        if iou.shape[0] == 0:
            continue
        for pred_row, pred_id in enumerate(pred_ids):
            values = iou[pred_row, :] if iou.shape[1] else np.asarray([], dtype=np.float32)
            hits = np.nonzero(values >= threshold)[0]
            if hits.shape[0] <= 1 and (values.size == 0 or float(np.max(values)) >= 0.25):
                continue
            ordered = sorted(hits.tolist(), key=lambda idx: float(values[idx]), reverse=True)
            best = float(np.max(values)) if values.size else 0.0
            rows.append(
                {
                    "scene_id": scene,
                    "threshold": float(threshold),
                    "pred_id": int(pred_id),
                    "matched_gt_count": int(hits.shape[0]),
                    "best_gt_iou": best,
                    "top_gt_ids": ",".join(str(gt_ids[idx]) for idx in ordered[:10]),
                    "top_gt_ious": ",".join(f"{float(values[idx]):.6f}" for idx in ordered[:10]),
                }
            )
    return rows


def _score_free(summary: dict[str, Any], threshold_key: str = "050") -> float | None:
    value = (summary.get(f"score_free_match_at_{threshold_key}") or {}).get("recall")
    return None if value is None else float(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    stride = int(args.stride)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, Any]] = []
    all_conflict_rows: list[dict[str, Any]] = []
    all_fragmentation_rows: list[dict[str, Any]] = []
    all_overmerge_rows: list[dict[str, Any]] = []
    all_object_support_rows: list[dict[str, Any]] = []
    wta_delta_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        variant = _best_variant(pipeline_root)
        support = _load_support(pipeline_root=pipeline_root, scene=scene, variant=variant)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        current_eval = _evaluate_mapping(
            scene=scene,
            stride=stride,
            mask_dir=mask_dir,
            mapping=support["current_mapping"],
            output_dir=output_root / "runs" / f"{scene}_current_min_mapping_stride{stride}",
            score_mode="constant",
        )
        wta_eval = _evaluate_mapping(
            scene=scene,
            stride=stride,
            mask_dir=mask_dir,
            mapping=support["wta_mapping"],
            output_dir=output_root / "runs" / f"{scene}_wta_mapping_stride{stride}",
            score_mode="constant",
        )
        current_summary = current_eval["summary"]
        wta_summary = wta_eval["summary"]
        pair_count = int(len(support["pair_objects"]))
        conflict_count = int(sum(1 for objects in support["pair_objects"].values() if len(objects) > 1))
        conflict_rate = float(conflict_count / max(1, pair_count))
        max_objects = max((len(objects) for objects in support["pair_objects"].values()), default=0)
        frag = _fragmentation_rows(scene, current_eval["iou"], current_eval["pred_ids"], current_eval["gt_ids"])
        over = _overmerge_rows(scene, current_eval["iou"], current_eval["pred_ids"], current_eval["gt_ids"])
        all_conflict_rows.extend(support["conflict_rows"])
        all_fragmentation_rows.extend(frag)
        all_overmerge_rows.extend(over)
        all_object_support_rows.extend(support["object_support_rows"])
        current_ap50 = current_summary.get("ap50")
        wta_ap50 = wta_summary.get("ap50")
        current_sf50 = _score_free(current_summary)
        wta_sf50 = _score_free(wta_summary)
        wta_delta_rows.append(
            {
                "scene_id": scene,
                "current_AP50": current_ap50,
                "wta_AP50": wta_ap50,
                "delta_AP50": float(wta_ap50 or 0.0) - float(current_ap50 or 0.0),
                "current_score_free_match50_recall": current_sf50,
                "wta_score_free_match50_recall": wta_sf50,
                "delta_score_free_match50_recall": float(wta_sf50 or 0.0) - float(current_sf50 or 0.0),
                "wta_uses_gt_for_prediction": False,
            }
        )
        scene_rows.append(
            {
                "scene_id": scene,
                "pipeline_root": _rel(pipeline_root),
                "variant": variant,
                "object_count": int(len(support["object_to_idx"])),
                "ledger_row_count": int(support["ledger_rows"]),
                "used_ledger_row_count": int(support["used_rows"]),
                "support_pair_count": pair_count,
                "duplicate_frame_mask_conflict_pairs": conflict_count,
                "duplicate_frame_mask_conflict_rate": conflict_rate,
                "max_objects_per_frame_mask": int(max_objects),
                "current_AP": current_summary.get("ap"),
                "current_AP50": current_ap50,
                "current_AP25": current_summary.get("ap25"),
                "current_score_free_match50_recall": current_sf50,
                "current_gt_best_iou_mean": current_summary.get("gt_best_iou_mean"),
                "wta_AP50": wta_ap50,
                "wta_score_free_match50_recall": wta_sf50,
                "fragmentation_row_count": int(len(frag)),
                "overmerge_row_count": int(len(over)),
                "current_summary_json": current_eval["summary_json"],
                "wta_summary_json": wta_eval["summary_json"],
            }
        )

    _write_csv(output_root / "scene_audit_rows.csv", scene_rows)
    _write_csv(output_root / "frame_mask_conflict_rows.csv", all_conflict_rows)
    _write_csv(output_root / "fragmentation_rows.csv", all_fragmentation_rows)
    _write_csv(output_root / "overmerge_rows.csv", all_overmerge_rows)
    _write_csv(output_root / "object_support_rows.csv", all_object_support_rows)
    _write_csv(output_root / "wta_delta_rows.csv", wta_delta_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    conflict_rates = [float(row["duplicate_frame_mask_conflict_rate"]) for row in scene_rows]
    wta_deltas = [float(row["delta_AP50"]) for row in wta_delta_rows]
    gate = {
        "all_pipeline_roots_available": len(missing_rows) == 0,
        "duplicate_conflict_rate_all_le_0p02": bool(scene_rows) and all(value <= 0.02 for value in conflict_rates),
        "wta_diagnostic_ran": bool(wta_delta_rows),
        "wta_no_scene_delta_ap50_gt_0p05": bool(wta_delta_rows) and all(value <= 0.05 for value in wta_deltas),
    }
    gate["pass"] = gate["all_pipeline_roots_available"] and gate["duplicate_conflict_rate_all_le_0p02"]
    summary = {
        "phase": "v66_ledger_audit",
        "diagnostic_only": True,
        "scenes": scenes,
        "stride": stride,
        "gate": gate,
        "scene_count": len(scene_rows),
        "max_duplicate_frame_mask_conflict_rate": max(conflict_rates) if conflict_rates else None,
        "max_wta_delta_AP50": max(wta_deltas) if wta_deltas else None,
        "rows": {
            "scene_audit_rows_csv": _rel(output_root / "scene_audit_rows.csv"),
            "frame_mask_conflict_rows_csv": _rel(output_root / "frame_mask_conflict_rows.csv"),
            "fragmentation_rows_csv": _rel(output_root / "fragmentation_rows.csv"),
            "overmerge_rows_csv": _rel(output_root / "overmerge_rows.csv"),
            "object_support_rows_csv": _rel(output_root / "object_support_rows.csv"),
            "wta_delta_rows_csv": _rel(output_root / "wta_delta_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "current_min_mapping matches the v65 conflict fallback semantics: duplicate frame/mask owners map to the lowest object index.",
            "WTA mapping is diagnostic-only and uses only support vote weights from the reprojection ledger, not GT labels.",
            "fragmentation/overmerge rows are derived from scene-level multi-view 2D IoU matrices for the current mapping.",
        ],
    }
    _write_json(output_root / "ledger_audit_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "ledger_audit_summary.json",
        output_root / "scene_audit_rows.csv",
        output_root / "frame_mask_conflict_rows.csv",
        output_root / "fragmentation_rows.csv",
        output_root / "overmerge_rows.csv",
        output_root / "object_support_rows.csv",
        output_root / "wta_delta_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v66 SOMA ledger conflict and local object audit diagnostics.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v66_ledger_audit")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
