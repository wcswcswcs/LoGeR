from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v67_mask_universe import _frame_mask_stats, _representative_pairs  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import (  # noqa: E402
    _best_variant,
    _chunk_rows,
    _evaluate_frame_data,
    _float_or_none,
    _frag_overmerge_means,
    _frame_data,
    _load_csv_rows,
    _mapping_from_candidates,
    _mean,
    _rel,
    _score_free,
    _selected_candidates,
    _summarize_variant,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)
from stream4d.scannet_stream import ScanNetStream  # noqa: E402


def _object_frame_stats(
    *,
    frame_data: list[dict[str, Any]],
    mapping: dict[tuple[int, int], int] | None,
    raw_per_frame_masks: bool,
) -> dict[str, Any]:
    object_frames: dict[int, set[int]] = defaultdict(set)
    support_pair_count = 0
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        for mask_id in np.unique(mask):
            mask_id_i = int(mask_id)
            if mask_id_i <= 0:
                continue
            if raw_per_frame_masks:
                object_id = frame_id * 100000 + mask_id_i
            else:
                object_id = int((mapping or {}).get((frame_id, mask_id_i), 0))
            if object_id <= 0:
                continue
            object_frames[object_id].add(frame_id)
            support_pair_count += 1
    object_count = len(object_frames)
    single_frame_count = sum(1 for frames in object_frames.values() if len(frames) == 1)
    return {
        "support_pair_count": int(support_pair_count),
        "mean_masks_per_object": float(support_pair_count / max(1, object_count)),
        "single_frame_object_rate": float(single_frame_count / max(1, object_count)),
    }


def _all_candidate_mapping(
    *,
    candidate_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    wta: bool,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    candidate_to_object_idx: dict[str, int] = {}
    for row in candidate_rows:
        if row.get("scene") != scene or row.get("chunk_id") != chunk_id:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in candidate_to_object_idx:
            candidate_to_object_idx[candidate_id] = len(candidate_to_object_idx) + 1
    mapping, diag = _mapping_from_candidates(
        ledger_rows=ledger_rows,
        scene=scene,
        chunk_id=chunk_id,
        candidate_to_object_idx=candidate_to_object_idx,
        wta=wta,
    )
    diag.update({"selected_mask_count": int(len(candidate_to_object_idx))})
    return mapping, diag


def _representative_pairs_by_chunk(pipeline_root: Path, scene: str) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    path = pipeline_root / "representative_observations/representative_mask_rows.csv"
    if not path.exists():
        return out
    for row in _load_csv_rows(path):
        if row.get("scene") != scene:
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        try:
            frame_id = int(float(row.get("frame_id") or ""))
            mask_id = int(float(row.get("mask_id") or ""))
        except ValueError:
            continue
        if mask_id > 0:
            out[chunk_id].add((frame_id, mask_id))
    return out


def _unique_pair_mapping(pairs: set[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping = {pair: idx + 1 for idx, pair in enumerate(sorted(pairs))}
    return mapping, {
        "selected_mask_count": int(len(mapping)),
        "support_pair_count": int(len(mapping)),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "max_objects_per_frame_mask": 1 if mapping else 0,
    }


def _same_numeric_mask_id_mapping(frame_data: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping: dict[tuple[int, int], int] = {}
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        for mask_id in np.unique(mask):
            mask_id_i = int(mask_id)
            if mask_id_i > 0:
                mapping[(frame_id, mask_id_i)] = mask_id_i
    return mapping, {
        "selected_mask_count": int(len(mapping)),
        "support_pair_count": int(len(mapping)),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "max_objects_per_frame_mask": 1 if mapping else 0,
    }


def _oracle_majority_mapping(
    *,
    frame_data: list[dict[str, Any]],
    allowed_pairs: set[tuple[int, int]] | None,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping: dict[tuple[int, int], int] = {}
    considered = 0
    skipped_no_gt = 0
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        stats_by_mask = _frame_mask_stats(np.asarray(mask, dtype=np.int64), np.asarray(item["gt"], dtype=np.int64))
        for mask_id, stats in stats_by_mask.items():
            key = (frame_id, int(mask_id))
            if allowed_pairs is not None and key not in allowed_pairs:
                continue
            considered += 1
            label = int(stats.get("majority_gt") or 0)
            if label <= 0:
                skipped_no_gt += 1
                continue
            mapping[key] = label
    return mapping, {
        "selected_mask_count": int(considered),
        "support_pair_count": int(len(mapping)),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "max_objects_per_frame_mask": 1 if mapping else 0,
        "oracle_skipped_no_gt_overlap": int(skipped_no_gt),
    }


def _oracle_majority_mapping_bundle(
    *,
    frame_data: list[dict[str, Any]],
    selected_pairs: set[tuple[int, int]],
    representative_pairs: set[tuple[int, int]],
) -> dict[str, tuple[dict[tuple[int, int], int], dict[str, Any]]]:
    mappings: dict[str, dict[tuple[int, int], int]] = {
        "raw": {},
        "selected": {},
        "representative": {},
    }
    considered = {"raw": 0, "selected": 0, "representative": 0}
    skipped = {"raw": 0, "selected": 0, "representative": 0}
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        stats_by_mask = _frame_mask_stats(np.asarray(mask, dtype=np.int64), np.asarray(item["gt"], dtype=np.int64))
        for mask_id, stats in stats_by_mask.items():
            key = (frame_id, int(mask_id))
            label = int(stats.get("majority_gt") or 0)
            target_names = ["raw"]
            if key in selected_pairs:
                target_names.append("selected")
            if key in representative_pairs:
                target_names.append("representative")
            for name in target_names:
                considered[name] += 1
                if label <= 0:
                    skipped[name] += 1
                    continue
                mappings[name][key] = label
    out: dict[str, tuple[dict[tuple[int, int], int], dict[str, Any]]] = {}
    for name, mapping in mappings.items():
        out[name] = (
            mapping,
            {
                "selected_mask_count": int(considered[name]),
                "support_pair_count": int(len(mapping)),
                "duplicate_frame_mask_conflict_pairs": 0,
                "duplicate_frame_mask_conflict_rate": 0.0,
                "max_objects_per_frame_mask": 1 if mapping else 0,
                "oracle_skipped_no_gt_overlap": int(skipped[name]),
            },
        )
    return out


def _score_free_at(summary: dict[str, Any], threshold: str) -> float | None:
    value = (summary.get(f"score_free_match_at_{threshold}") or {}).get("recall")
    return None if value is None else float(value)


def _row_from_eval(
    *,
    scene: str,
    chunk_id: str,
    variant: str,
    frame_ids: list[int],
    chunk: dict[str, Any],
    frame_data: list[dict[str, Any]],
    mapping: dict[tuple[int, int], int] | None,
    raw_per_frame_masks: bool,
    diag: dict[str, Any],
    uses_gt_for_prediction: bool,
    forbidden_for_method_table: bool,
    pipeline_root: Path,
) -> dict[str, Any]:
    summary, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
        frame_data=frame_data,
        variant=variant,
        mapping=mapping,
        raw_per_frame_masks=raw_per_frame_masks,
    )
    frag_mean, over_mean = _frag_overmerge_means(iou)
    object_stats = _object_frame_stats(frame_data=frame_data, mapping=mapping, raw_per_frame_masks=raw_per_frame_masks)
    return {
        "scene_id": scene,
        "chunk_id": chunk_id,
        "variant": variant,
        "chunk_frame_count": int(len(frame_ids)),
        "frame_min": int(frame_ids[0]),
        "frame_max": int(frame_ids[-1]),
        "mask_count": int(float(chunk.get("mask_count") or 0)),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "local_object_count": summary.get("evaluated_pred_count"),
        "local_gt_count": summary.get("evaluated_gt_count"),
        "local_AP": summary.get("ap"),
        "local_AP50": summary.get("ap50"),
        "local_AP25": summary.get("ap25"),
        "local_SF25": _score_free_at(summary, "025"),
        "local_SF50": _score_free_at(summary, "050"),
        "local_score_free_match50_recall": _score_free(summary),
        "GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "local_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "local_pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "fragmentation_mean": frag_mean,
        "local_fragmentation_mean": frag_mean,
        "overmerge_mean": over_mean,
        "local_overmerge_mean": over_mean,
        "duplicate_frame_mask_conflict_rate": diag.get("duplicate_frame_mask_conflict_rate", 0.0),
        "local_duplicate_frame_mask_conflict_rate": diag.get("duplicate_frame_mask_conflict_rate", 0.0),
        "duplicate_frame_mask_conflict_pairs": diag.get("duplicate_frame_mask_conflict_pairs", ""),
        "support_pair_count": diag.get("support_pair_count", object_stats.get("support_pair_count", "")),
        "selected_mask_count": diag.get("selected_mask_count", ""),
        "mean_masks_per_object": object_stats["mean_masks_per_object"],
        "single_frame_object_rate": object_stats["single_frame_object_rate"],
        "uses_gt_for_prediction": bool(uses_gt_for_prediction),
        "forbidden_for_method_table": bool(forbidden_for_method_table),
        "diagnostic_only": True,
        "source_scope": "current_run_v67_reanalysis",
        "pipeline_root": _rel(pipeline_root),
    }


def _summarize_variant_all(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    return {
        "variant": variant,
        "chunk_count": len(subset),
        "scene_count": len({row["scene_id"] for row in subset}),
        "local_AP50_mean": _mean([_float_or_none(row.get("local_AP50")) for row in subset]),
        "local_AP25_mean": _mean([_float_or_none(row.get("local_AP25")) for row in subset]),
        "local_SF50_mean": _mean([_float_or_none(row.get("local_SF50")) for row in subset]),
        "local_score_free_match50_recall_mean": _mean(
            [_float_or_none(row.get("local_score_free_match50_recall")) for row in subset]
        ),
        "local_GT_best_IoU_mean_mean": _mean([_float_or_none(row.get("local_GT_best_IoU_mean")) for row in subset]),
        "local_pred_best_IoU_median_mean": _mean([_float_or_none(row.get("local_pred_best_IoU_median")) for row in subset]),
        "local_duplicate_frame_mask_conflict_rate_mean": _mean(
            [_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate")) for row in subset]
        ),
        "local_object_count_mean": _mean([_float_or_none(row.get("local_object_count")) for row in subset]),
        "local_gt_count_mean": _mean([_float_or_none(row.get("local_gt_count")) for row in subset]),
        "mean_masks_per_object_mean": _mean([_float_or_none(row.get("mean_masks_per_object")) for row in subset]),
        "single_frame_object_rate_mean": _mean([_float_or_none(row.get("single_frame_object_rate")) for row in subset]),
        "uses_gt_for_prediction": any(bool(row.get("uses_gt_for_prediction")) for row in subset),
        "forbidden_for_method_table": any(bool(row.get("forbidden_for_method_table")) for row in subset),
        "diagnostic_only": True,
    }


def _variant_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant"]): row for row in rows}


def _best_non_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if not bool(row.get("uses_gt_for_prediction"))
        and str(row.get("variant", "")).startswith(("B1_", "B2_", "B3_", "B4_", "B5_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("local_score_free_match50_recall_mean") or 0.0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    stride = int(args.stride)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    variants_seen: set[tuple[str, str]] = set()
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        best_variant = _best_variant(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=stride, max_frames=None)
        objectlet_rows = _load_csv_rows(pipeline_root / "local_objectlets/objectlet_rows.csv")
        ledger_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv")
        candidate_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/candidate_rows.csv")
        representative_by_chunk = _representative_pairs_by_chunk(pipeline_root, scene)
        representative_scene_pairs = _representative_pairs(pipeline_root, scene)
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            data = _frame_data(scene, frame_ids, mask_dir)
            candidate_to_object_idx, _object_idx_to_id = _selected_candidates(
                objectlet_rows=objectlet_rows,
                scene=scene,
                chunk_id=chunk_id,
                variant=best_variant,
            )
            b1_mapping, b1_diag = _mapping_from_candidates(
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                candidate_to_object_idx=candidate_to_object_idx,
                wta=False,
            )
            b2_mapping, b2_diag = _mapping_from_candidates(
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                candidate_to_object_idx=candidate_to_object_idx,
                wta=True,
            )
            b3_mapping, b3_diag = _all_candidate_mapping(
                candidate_rows=candidate_rows,
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                wta=False,
            )
            rep_pairs = set(representative_by_chunk.get(chunk_id, set()))
            if not rep_pairs:
                rep_pairs = {
                    pair
                    for pair in representative_scene_pairs
                    if raw_start <= int(pair[0]) <= raw_end
                }
            b4_mapping, b4_diag = _unique_pair_mapping(rep_pairs)
            b5_mapping, b5_diag = _same_numeric_mask_id_mapping(data)
            oracle_bundle = _oracle_majority_mapping_bundle(
                frame_data=data,
                selected_pairs=set(b1_mapping.keys()),
                representative_pairs=rep_pairs,
            )
            b7_mapping, b7_diag = oracle_bundle["raw"]
            b8_mapping, b8_diag = oracle_bundle["selected"]
            b4_oracle_mapping, b4_oracle_diag = oracle_bundle["representative"]
            eval_specs = [
                {
                    "variant": "B0_raw_per_frame_masks",
                    "mapping": None,
                    "raw": True,
                    "diag": {
                        "selected_mask_count": "",
                        "support_pair_count": "",
                        "duplicate_frame_mask_conflict_rate": 0.0,
                    },
                    "uses_gt": False,
                    "forbidden": True,
                },
                {
                    "variant": f"B1_current_{best_variant}",
                    "mapping": b1_mapping,
                    "raw": False,
                    "diag": {**b1_diag, "selected_mask_count": len(candidate_to_object_idx)},
                    "uses_gt": False,
                    "forbidden": False,
                },
                {
                    "variant": f"B2_WTA_current_{best_variant}",
                    "mapping": b2_mapping,
                    "raw": False,
                    "diag": {**b2_diag, "selected_mask_count": len(candidate_to_object_idx)},
                    "uses_gt": False,
                    "forbidden": False,
                },
                {
                    "variant": "B3_all_reprojection_success_candidates",
                    "mapping": b3_mapping,
                    "raw": False,
                    "diag": b3_diag,
                    "uses_gt": False,
                    "forbidden": False,
                },
                {
                    "variant": "B4_all_representative_masks_identity",
                    "mapping": b4_mapping,
                    "raw": False,
                    "diag": b4_diag,
                    "uses_gt": False,
                    "forbidden": False,
                },
                {
                    "variant": "B5_all_CropFormer_masks_same_numeric_id_placeholder",
                    "mapping": b5_mapping,
                    "raw": False,
                    "diag": b5_diag,
                    "uses_gt": False,
                    "forbidden": True,
                },
                {
                    "variant": "B7_oracle_raw_mask_grouping_majority_GT",
                    "mapping": b7_mapping,
                    "raw": False,
                    "diag": b7_diag,
                    "uses_gt": True,
                    "forbidden": True,
                },
                {
                    "variant": "B8_oracle_selected_mask_grouping_majority_GT",
                    "mapping": b8_mapping,
                    "raw": False,
                    "diag": b8_diag,
                    "uses_gt": True,
                    "forbidden": True,
                },
                {
                    "variant": "B4_oracle_representative_mask_grouping_majority_GT",
                    "mapping": b4_oracle_mapping,
                    "raw": False,
                    "diag": b4_oracle_diag,
                    "uses_gt": True,
                    "forbidden": True,
                },
            ]
            for spec in eval_specs:
                row = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=str(spec["variant"]),
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=data,
                    mapping=spec["mapping"],
                    raw_per_frame_masks=bool(spec["raw"]),
                    diag=dict(spec["diag"]),
                    uses_gt_for_prediction=bool(spec["uses_gt"]),
                    forbidden_for_method_table=bool(spec["forbidden"]),
                    pipeline_root=pipeline_root,
                )
                rows.append(row)
                variants_seen.add((scene, str(spec["variant"])))
    missing_rows.append(
        {
            "scene_id": "ALL",
            "missing": "B6_Stream3D_rendered_local_diagnostic",
            "reason": "No chunk-local Stream3D rendered local-object artifact was found in the current v66/v67 audit inputs; scene-level Stream3D remains available from Phase 1 only.",
        }
    )
    scene_summary_rows = [_summarize_variant(rows, scene, variant) for scene, variant in sorted(variants_seen)]
    variant_summary_rows = [_summarize_variant_all(rows, variant) for variant in sorted({row["variant"] for row in rows})]
    lookup = _variant_lookup(variant_summary_rows)
    b1_rows = [row for key, row in lookup.items() if key.startswith("B1_current_")]
    b8 = lookup.get("B8_oracle_selected_mask_grouping_majority_GT", {})
    b7 = lookup.get("B7_oracle_raw_mask_grouping_majority_GT", {})
    b4_oracle = lookup.get("B4_oracle_representative_mask_grouping_majority_GT", {})
    b1_sf50 = _mean([_float_or_none(row.get("local_score_free_match50_recall_mean")) for row in b1_rows])
    b1_gt_best = _mean([_float_or_none(row.get("local_GT_best_IoU_mean_mean")) for row in b1_rows])
    b1_ap50 = _mean([_float_or_none(row.get("local_AP50_mean")) for row in b1_rows])
    b1_dup = _mean([_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate_mean")) for row in b1_rows])
    b7_sf50 = _float_or_none(b7.get("local_score_free_match50_recall_mean"))
    b8_sf50 = _float_or_none(b8.get("local_score_free_match50_recall_mean"))
    b4_oracle_sf50 = _float_or_none(b4_oracle.get("local_score_free_match50_recall_mean"))
    raw_minus_selected = b7_sf50 - b8_sf50 if b7_sf50 is not None and b8_sf50 is not None else None
    rep_minus_selected = (
        b4_oracle_sf50 - b8_sf50 if b4_oracle_sf50 is not None and b8_sf50 is not None else None
    )
    best_non_oracle = _best_non_oracle(variant_summary_rows) or {}
    best_non_oracle_sf50 = _float_or_none(best_non_oracle.get("local_score_free_match50_recall_mean"))
    best_non_oracle_gt_best = _float_or_none(best_non_oracle.get("local_GT_best_IoU_mean_mean"))
    best_non_oracle_ap50 = _float_or_none(best_non_oracle.get("local_AP50_mean"))
    best_non_oracle_dup = _float_or_none(best_non_oracle.get("local_duplicate_frame_mask_conflict_rate_mean"))
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "b6_stream3d_local_diagnostic_available": False,
        "b1_current_local_sf50_lt_0p10": b1_sf50 is not None and b1_sf50 < 0.10,
        "b1_current_local_ap50_ge_0p05": b1_ap50 is not None and b1_ap50 >= 0.05,
        "b1_current_duplicate_rate_le_0p02": b1_dup is not None and b1_dup <= 0.02,
        "b7_raw_oracle_minus_b8_selected_oracle_ge_0p20": raw_minus_selected is not None and raw_minus_selected >= 0.20,
        "b4_rep_oracle_minus_b8_selected_oracle_ge_0p20": rep_minus_selected is not None and rep_minus_selected >= 0.20,
        "best_non_oracle_local_sf50_ge_0p30": best_non_oracle_sf50 is not None and best_non_oracle_sf50 >= 0.30,
        "best_non_oracle_GT_best_IoU_ge_0p25": best_non_oracle_gt_best is not None and best_non_oracle_gt_best >= 0.25,
        "best_non_oracle_AP50_ge_0p05": best_non_oracle_ap50 is not None and best_non_oracle_ap50 >= 0.05,
        "best_non_oracle_duplicate_rate_le_0p02": best_non_oracle_dup is not None and best_non_oracle_dup <= 0.02,
    }
    gate["best_non_oracle_local_gate_pass"] = (
        gate["best_non_oracle_local_sf50_ge_0p30"]
        and gate["best_non_oracle_GT_best_IoU_ge_0p25"]
        and gate["best_non_oracle_AP50_ge_0p05"]
        and gate["best_non_oracle_duplicate_rate_le_0p02"]
    )
    if not gate["all_pipeline_roots_available"]:
        decision = "MISSING_INPUT"
    elif gate["best_non_oracle_local_gate_pass"]:
        decision = "PASS_LOCAL_BASELINE_GATE"
    elif gate["b7_raw_oracle_minus_b8_selected_oracle_ge_0p20"] or gate[
        "b4_rep_oracle_minus_b8_selected_oracle_ge_0p20"
    ]:
        decision = "ORACLE_HEADROOM_HIGH_NON_ORACLE_LOCAL_GATE_FAIL"
    elif b7_sf50 is not None and b7_sf50 < 0.20:
        decision = "RAW_LOCAL_ORACLE_LOW"
    else:
        decision = "LOCAL_BASELINE_INCONCLUSIVE"
    _write_csv(output_root / "local_baseline_rows.csv", rows)
    _write_csv(output_root / "local_scene_summary_rows.csv", scene_summary_rows)
    _write_csv(output_root / "local_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v67_local_object_formation_baselines",
        "decision": decision,
        "diagnostic_only": True,
        "scenes": scenes,
        "stride": stride,
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "b1_current_means": {
            "local_score_free_match50_recall_mean": b1_sf50,
            "local_GT_best_IoU_mean": b1_gt_best,
            "local_duplicate_frame_mask_conflict_rate": b1_dup,
            "local_AP50": b1_ap50,
        },
        "oracle_headroom": {
            "b7_raw_oracle_sf50": b7_sf50,
            "b8_selected_oracle_sf50": b8_sf50,
            "b4_representative_oracle_sf50": b4_oracle_sf50,
            "b7_raw_minus_b8_selected_sf50": raw_minus_selected,
            "b4_representative_minus_b8_selected_sf50": rep_minus_selected,
        },
        "best_non_oracle": best_non_oracle,
        "rows": {
            "local_baseline_rows_csv": _rel(output_root / "local_baseline_rows.csv"),
            "local_scene_summary_rows_csv": _rel(output_root / "local_scene_summary_rows.csv"),
            "local_variant_summary_rows_csv": _rel(output_root / "local_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "B7, B8, and B4_oracle use 2D GT majority labels for diagnostic upper bounds and are forbidden for method tables.",
            "B4 identity keeps every representative mask observation as its own local object and does not use GT; it is a lower-bound/non-clustering baseline.",
            "B5 same-numeric-mask-id is a placeholder sanity baseline because raw CropFormer label ids are not guaranteed to persist across frames.",
            "B6 Stream3D chunk-local rendered diagnostic is recorded as missing rather than fabricated.",
        ],
    }
    _write_json(output_root / "local_baseline_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "local_baseline_summary.json",
        output_root / "local_baseline_rows.csv",
        output_root / "local_scene_summary_rows.csv",
        output_root / "local_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stream4D v67 chunk-local object formation baselines.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v67_local_baselines")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
