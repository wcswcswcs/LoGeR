from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _oracle_majority_mapping_bundle  # noqa: E402
from stream4d_native.v67_mask_universe import _colorize_labels  # noqa: E402
from stream4d_native.v68_local_graph_solver import _row_from_mapping  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _float_or_none, _frame_data, _load_csv_rows, _mean, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_json(value: Any, fallback: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _node(scene: str, frame_id: Any, mask_id: Any) -> tuple[str, int, int]:
    return (str(scene), int(float(frame_id)), int(float(mask_id)))


def _pair(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["_frame_id"]), int(row["_mask_id"]))


def _node_token(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}"


def _load_appearance_index(edge_rows: Path, scenes: set[str]) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not edge_rows.exists():
        return {}
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in _load_csv_rows(edge_rows):
        scene = str(row.get("scene_id"))
        if scene not in scenes:
            continue
        for suffix in ["i", "j"]:
            frame = row.get(f"frame_{suffix}")
            mask = row.get(f"mask_{suffix}")
            if frame in ("", None) or mask in ("", None):
                continue
            key = _node(scene, frame, mask)
            if key in out:
                continue
            out[key] = {
                "DINO_mode_id": str(row.get(f"appearance_mode_{suffix}") or ""),
                "DINO_feature_valid": _parse_bool(row.get(f"appearance_valid_{suffix}")),
                "DINO_used_pixels": int(float(row.get(f"appearance_used_pixels_{suffix}") or 0)),
            }
    return out


def _prep_candidate_rows(candidate_rows: Path, scenes: list[str], edge_rows: Path) -> list[dict[str, Any]]:
    scene_set = set(scenes)
    appearance = _load_appearance_index(edge_rows, scene_set)
    rows: list[dict[str, Any]] = []
    raw_rows = [row for row in _load_csv_rows(candidate_rows) if str(row.get("scene_id")) in scene_set]
    sig_counts: Counter[tuple[str, str, str]] = Counter(
        (str(row.get("scene_id")), str(row.get("chunk_id")), str(row.get("repeated_signature_id") or "")) for row in raw_rows
    )
    for row in raw_rows:
        if not _parse_bool(row.get("representative_available")):
            continue
        scene = str(row.get("scene_id"))
        frame_id = int(float(row.get("frame_id") or 0))
        mask_id = int(float(row.get("mask_id") or 0))
        comps = [str(item) for item in _parse_json(row.get("d4rt_component_ids"), []) if str(item)]
        bbox_center = [float(v) for v in _parse_json(row.get("bbox_center"), [0.0, 0.0])[:2]]
        bbox_size = [float(v) for v in _parse_json(row.get("bbox_size"), [0.0, 0.0])[:2]]
        source_types = [str(item) for item in _parse_json(row.get("source_types"), [])]
        app = appearance.get((scene, frame_id, mask_id), {})
        d4rt_count = int(float(row.get("d4rt_component_count") or 0))
        entropy = _safe_float(row.get("d4rt_component_entropy"))
        area = _safe_float(row.get("area_ratio"))
        solidity = _safe_float(row.get("mask_solidity_proxy"))
        same_frame_overlap = int(float(row.get("same_frame_overlap_count") or 0))
        diagnostic_gt_underseg = _parse_bool(row.get("underseg_risk"))
        large = _parse_bool(row.get("large_mask_risk"))
        small = _parse_bool(row.get("small_mask_risk"))
        signature = str(row.get("repeated_signature_id") or "")
        repeated = signature and sig_counts[(scene, str(row.get("chunk_id")), signature)] >= 2
        material_multimodal_risk = bool(d4rt_count >= 64 and entropy > 1.5)
        overlap_multimask_risk = bool(same_frame_overlap >= 10)
        broad_overlap_risk = bool((area >= 0.12 and same_frame_overlap >= 4) or (area >= 0.04 and solidity < 0.18 and same_frame_overlap >= 4))
        underseg = bool(material_multimodal_risk or overlap_multimask_risk or broad_overlap_risk)
        q_d4rt = min(1.0, d4rt_count / 8.0) * (1.0 / (1.0 + 0.25 * max(0.0, entropy)))
        q_shape = 0.0
        if not small and not large and area > 0.0:
            q_shape = min(1.0, max(0.0, solidity)) * (1.0 if 0.002 <= area <= 0.20 else 0.65)
        q_repeat = 1.0 if repeated else 0.0
        q_dino = 0.25 if app.get("DINO_feature_valid") else 0.0
        q_underseg = 1.0 if underseg else min(1.0, same_frame_overlap / 8.0)
        q_size = 1.0 if (small or large) else 0.0
        objectness = 0.42 * q_d4rt + 0.18 * q_dino + 0.22 * q_shape + 0.18 * q_repeat - 0.45 * q_underseg - 0.30 * q_size
        out = dict(row)
        out.update(
            {
                "_scene": scene,
                "_chunk_id": str(row.get("chunk_id")),
                "_frame_id": frame_id,
                "_mask_id": mask_id,
                "_node": (scene, frame_id, mask_id),
                "_pair": (frame_id, mask_id),
                "_components": set(comps),
                "_component_count": d4rt_count,
                "_component_entropy": entropy,
                "_area_ratio": area,
                "_bbox_center": bbox_center if len(bbox_center) == 2 else [0.0, 0.0],
                "_bbox_size": bbox_size if len(bbox_size) == 2 else [0.0, 0.0],
                "_aspect_ratio": _safe_float(row.get("aspect_ratio"), 1.0),
                "_solidity": solidity,
                "_underseg": underseg,
                "_diagnostic_gt_underseg": diagnostic_gt_underseg,
                "_method_underseg_material_multimodal": material_multimodal_risk,
                "_method_underseg_overlap_multimask": overlap_multimask_risk,
                "_method_underseg_broad_overlap": broad_overlap_risk,
                "_large": large,
                "_small": small,
                "_same_frame_overlap": same_frame_overlap,
                "_signature": signature,
                "_repeated_signature": bool(repeated),
                "_source_types": source_types,
                "_shared_support": _parse_bool(row.get("shared_support_only")),
                "_DINO_mode_id": str(app.get("DINO_mode_id") or ""),
                "_DINO_feature_valid": bool(app.get("DINO_feature_valid", False)),
                "_DINO_used_pixels": int(app.get("DINO_used_pixels") or 0),
                "_q_d4rt": float(q_d4rt),
                "_q_dino": float(q_dino),
                "_q_shape": float(q_shape),
                "_q_repeat": float(q_repeat),
                "_q_underseg": float(q_underseg),
                "_anchor_objectness": float(objectness),
            }
        )
        rows.append(out)
    return rows


def _cap_per_frame(rows: list[dict[str, Any]], max_per_frame: int) -> set[tuple[int, int]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["_frame_id"])].append(row)
    selected: set[tuple[int, int]] = set()
    for frame_id, frame_rows in by_frame.items():
        ordered = sorted(frame_rows, key=lambda item: float(item["_anchor_objectness"]), reverse=True)
        selected.update((int(row["_frame_id"]), int(row["_mask_id"])) for row in ordered[:max_per_frame])
    return selected


def _support_balanced_pairs(
    chunk_rows: list[dict[str, Any]],
    *,
    max_per_frame: int,
    non_d4rt_ratio: float,
    underseg_cap: float,
) -> set[tuple[int, int]]:
    d4rt_core = [
        row
        for row in chunk_rows
        if row["_component_count"] > 0 and not row["_underseg"] and not row["_large"] and not row["_small"]
    ]
    semantic_fill = [
        row
        for row in chunk_rows
        if (
            row["_component_count"] == 0
            and not row["_underseg"]
            and not row["_large"]
            and not row["_small"]
            and (row["_DINO_feature_valid"] or row["_repeated_signature"] or row["_same_frame_overlap"] <= 2)
        )
    ]
    semantic_fill = sorted(semantic_fill, key=lambda item: float(item["_anchor_objectness"]), reverse=True)
    max_semantic = int(max(0, math.floor(len(d4rt_core) * float(non_d4rt_ratio))))
    selected = list(d4rt_core) + semantic_fill[:max_semantic]
    if underseg_cap > 0.0:
        allowed_underseg = int(math.floor((float(underseg_cap) / max(1e-9, 1.0 - float(underseg_cap))) * len(selected)))
        underseg_fill = [
            row
            for row in chunk_rows
            if (
                row["_component_count"] > 0
                and row["_underseg"]
                and not row["_large"]
                and not row["_small"]
                and row["_same_frame_overlap"] <= 4
            )
        ]
        underseg_fill = sorted(
            underseg_fill,
            key=lambda item: (
                int(item["_repeated_signature"]),
                int(item["_DINO_feature_valid"]),
                int(item["_component_count"]),
                -float(item["_component_entropy"]),
            ),
            reverse=True,
        )
        selected.extend(underseg_fill[:allowed_underseg])
    return _cap_per_frame(selected, max_per_frame)


def _clean_recall_support_floor_pairs(
    chunk_rows: list[dict[str, Any]],
    *,
    max_per_frame: int,
    underseg_cap: float,
    support_floor: float,
    final_underseg_cap: float | None = None,
) -> set[tuple[int, int]]:
    final_cap = float(underseg_cap if final_underseg_cap is None else final_underseg_cap)
    selected = [
        row
        for row in chunk_rows
        if not row["_underseg"] and not row["_large"] and not row["_small"] and row["_DINO_feature_valid"]
    ]
    allowed_underseg = int(math.floor((float(underseg_cap) / max(1e-9, 1.0 - float(underseg_cap))) * len(selected)))
    underseg_fill = [
        row
        for row in chunk_rows
        if row["_underseg"] and row["_component_count"] > 0 and not row["_large"] and not row["_small"] and row["_DINO_feature_valid"]
    ]
    underseg_fill = sorted(
        underseg_fill,
        key=lambda item: (
            int(item["_repeated_signature"]),
            int(item["_component_count"]),
            -float(item["_component_entropy"]),
            -int(item["_same_frame_overlap"]),
        ),
        reverse=True,
    )
    selected.extend(underseg_fill[:allowed_underseg])
    selected_by_pair = {_pair(row): row for row in selected}

    def support_rate(items: list[dict[str, Any]]) -> float:
        return float(sum(row["_component_count"] > 0 for row in items) / max(1, len(items)))

    current = list(selected_by_pair.values())
    if support_rate(current) < support_floor:
        non_d4rt = sorted(
            [row for row in current if row["_component_count"] == 0],
            key=lambda item: (float(item["_anchor_objectness"]), int(item["_repeated_signature"]), -int(item["_same_frame_overlap"])),
        )
        for row in non_d4rt:
            if support_rate(current) >= support_floor:
                break
            selected_by_pair.pop(_pair(row), None)
            current = list(selected_by_pair.values())
    current = list(selected_by_pair.values())
    underseg_rows = sorted(
        [row for row in current if row["_underseg"]],
        key=lambda item: (int(item["_repeated_signature"]), int(item["_component_count"]), -float(item["_component_entropy"])),
    )
    for row in underseg_rows:
        current = list(selected_by_pair.values())
        underseg_rate = float(sum(item["_underseg"] for item in current) / max(1, len(current)))
        if underseg_rate <= final_cap:
            break
        selected_by_pair.pop(_pair(row), None)
    current = list(selected_by_pair.values())
    if support_rate(current) < support_floor:
        non_d4rt = sorted(
            [row for row in current if row["_component_count"] == 0],
            key=lambda item: (float(item["_anchor_objectness"]), int(item["_repeated_signature"]), -int(item["_same_frame_overlap"])),
        )
        for row in non_d4rt:
            if support_rate(list(selected_by_pair.values())) >= support_floor:
                break
            selected_by_pair.pop(_pair(row), None)
    capped_pairs = _cap_per_frame(list(selected_by_pair.values()), max_per_frame)
    selected_by_pair = {pair: row for pair, row in selected_by_pair.items() if pair in capped_pairs}
    underseg_rows = sorted(
        [row for row in selected_by_pair.values() if row["_underseg"]],
        key=lambda item: (int(item["_repeated_signature"]), int(item["_component_count"]), -float(item["_component_entropy"])),
    )
    for row in underseg_rows:
        current = list(selected_by_pair.values())
        underseg_rate = float(sum(item["_underseg"] for item in current) / max(1, len(current)))
        if underseg_rate <= final_cap:
            break
        selected_by_pair.pop(_pair(row), None)
    if support_rate(list(selected_by_pair.values())) < support_floor:
        non_d4rt = sorted(
            [row for row in selected_by_pair.values() if row["_component_count"] == 0],
            key=lambda item: (float(item["_anchor_objectness"]), int(item["_repeated_signature"]), -int(item["_same_frame_overlap"])),
        )
        for row in non_d4rt:
            if support_rate(list(selected_by_pair.values())) >= support_floor:
                break
            selected_by_pair.pop(_pair(row), None)
    return set(selected_by_pair)


def _variant_sets(chunk_rows: list[dict[str, Any]], max_per_frame: int) -> dict[str, set[tuple[int, int]]]:
    all_rep = {_pair(row) for row in chunk_rows}
    d4rt_object = {
        _pair(row)
        for row in chunk_rows
        if row["_component_count"] > 0 and not row["_underseg"] and not row["_large"] and not row["_small"]
    }
    high_quality = {
        _pair(row)
        for row in chunk_rows
        if (not row["_underseg"]) and (not row["_large"]) and (not row["_small"]) and row["_solidity"] >= 0.20
    }
    repeated = {
        _pair(row)
        for row in chunk_rows
        if row["_repeated_signature"] and (not row["_underseg"]) and (not row["_large"]) and (not row["_small"])
    }
    scorelike_rows = [
        row
        for row in chunk_rows
        if (
            (row["_component_count"] > 0 or row["_same_frame_overlap"] <= 2 or row["_DINO_feature_valid"])
            and not row["_underseg"]
            and not row["_large"]
            and not row["_small"]
        )
    ]
    area_balanced = _cap_per_frame(scorelike_rows, max_per_frame)
    return {
        "A0_representative_all": all_rep,
        "A1_high_quality_nonunderseg": high_quality,
        "A2_d4rt_supported_objectlike": d4rt_object,
        "A3_dino_valid_objectlike": {
            _pair(row)
            for row in chunk_rows
            if row["_DINO_feature_valid"] and not row["_underseg"] and not row["_large"] and not row["_small"]
        },
        "A4_repeated_signature_objectlike": repeated,
        "A5_area_balanced_objectlike": area_balanced,
        "A6_support_balanced_d4rt_dino": _support_balanced_pairs(
            chunk_rows,
            max_per_frame=max_per_frame,
            non_d4rt_ratio=0.65,
            underseg_cap=0.0,
        ),
        "A7_guarded_representative_recall": _support_balanced_pairs(
            chunk_rows,
            max_per_frame=max_per_frame,
            non_d4rt_ratio=0.85,
            underseg_cap=0.12,
        ),
        "A8_clean_recall_support_floor_u12": _clean_recall_support_floor_pairs(
            chunk_rows,
            max_per_frame=max_per_frame,
            underseg_cap=0.12,
            support_floor=0.50,
        ),
        "A9_clean_recall_support_floor_u15": _clean_recall_support_floor_pairs(
            chunk_rows,
            max_per_frame=max_per_frame,
            underseg_cap=0.15,
            support_floor=0.50,
        ),
        "A10_clean_recall_u12_final_u15": _clean_recall_support_floor_pairs(
            chunk_rows,
            max_per_frame=max_per_frame,
            underseg_cap=0.12,
            final_underseg_cap=0.15,
            support_floor=0.50,
        ),
    }


def _anchor_diag(rows: list[dict[str, Any]], allowed: set[tuple[int, int]], frame_count: int) -> dict[str, Any]:
    selected = [row for row in rows if _pair(row) in allowed]
    if not selected:
        return {
            "anchor_count": 0,
            "anchor_count_per_frame_mean": 0.0,
            "anchor_D4RT_support_rate": 0.0,
            "anchor_underseg_rate": 0.0,
            "anchor_large_mask_rate": 0.0,
            "anchor_small_mask_rate": 0.0,
            "anchor_DINO_stability_mean": None,
            "anchor_repeated_signature_rate": 0.0,
            "anchor_objectness_mean": None,
        }
    return {
        "anchor_count": int(len(selected)),
        "anchor_count_per_frame_mean": float(len(selected) / max(1, frame_count)),
        "anchor_D4RT_support_rate": float(sum(row["_component_count"] > 0 for row in selected) / max(1, len(selected))),
        "anchor_underseg_rate": float(sum(row["_underseg"] for row in selected) / max(1, len(selected))),
        "anchor_diagnostic_GT_underseg_rate": float(sum(row["_diagnostic_gt_underseg"] for row in selected) / max(1, len(selected))),
        "anchor_large_mask_rate": float(sum(row["_large"] for row in selected) / max(1, len(selected))),
        "anchor_small_mask_rate": float(sum(row["_small"] for row in selected) / max(1, len(selected))),
        "anchor_DINO_stability_mean": float(sum(row["_DINO_feature_valid"] for row in selected) / max(1, len(selected))),
        "anchor_repeated_signature_rate": float(sum(row["_repeated_signature"] for row in selected) / max(1, len(selected))),
        "anchor_objectness_mean": float(np.mean([float(row["_anchor_objectness"]) for row in selected])),
    }


def _summarize_anchor_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["anchor_variant"] == variant]
    if not subset:
        return {"anchor_variant": variant, "chunk_count": 0, "scene_count": 0}
    gt_counts = [_float_or_none(row.get("gt_object_count")) for row in subset]
    same_frame_sum = int(sum(int(float(row.get("same_frame_cannot_link_violation_count") or 0)) for row in subset))
    return {
        "anchor_variant": variant,
        "variant": variant,
        "chunk_count": len(subset),
        "scene_count": len({row["scene_id"] for row in subset}),
        "anchor_count": int(sum(int(float(row.get("anchor_count") or 0)) for row in subset)),
        "anchor_count_per_chunk_mean": _mean([_float_or_none(row.get("anchor_count")) for row in subset]),
        "diagnostic_GT_count_per_chunk_mean": _mean(gt_counts),
        "anchor_count_per_frame_mean": _mean([_float_or_none(row.get("anchor_count_per_frame_mean")) for row in subset]),
        "anchor_D4RT_support_rate": _mean([_float_or_none(row.get("anchor_D4RT_support_rate")) for row in subset]),
        "anchor_underseg_rate": _mean([_float_or_none(row.get("anchor_underseg_rate")) for row in subset]),
        "anchor_diagnostic_GT_underseg_rate": _mean([_float_or_none(row.get("anchor_diagnostic_GT_underseg_rate")) for row in subset]),
        "anchor_large_mask_rate": _mean([_float_or_none(row.get("anchor_large_mask_rate")) for row in subset]),
        "anchor_small_mask_rate": _mean([_float_or_none(row.get("anchor_small_mask_rate")) for row in subset]),
        "anchor_DINO_stability_mean": _mean([_float_or_none(row.get("anchor_DINO_stability_mean")) for row in subset]),
        "anchor_repeated_signature_rate": _mean([_float_or_none(row.get("anchor_repeated_signature_rate")) for row in subset]),
        "anchor_objectness_mean": _mean([_float_or_none(row.get("anchor_objectness_mean")) for row in subset]),
        "anchor_oracle_SF50": _mean([_float_or_none(row.get("local_SF50")) for row in subset]),
        "anchor_oracle_AP50": _mean([_float_or_none(row.get("local_AP50")) for row in subset]),
        "anchor_GT_best_IoU_mean": _mean([_float_or_none(row.get("local_GT_best_IoU_mean")) for row in subset]),
        "anchor_single_frame_rate": _mean([_float_or_none(row.get("single_frame_object_rate")) for row in subset]),
        "same_frame_cannot_link_violation_count_sum": same_frame_sum,
        "selection_uses_gt_for_prediction": False,
        "oracle_metric_uses_gt_for_prediction": True,
        "uses_gt_for_prediction": True,
        "forbidden_for_method_table": True,
        "diagnostic_only": True,
    }


def _anchor_rows_for_variant(chunk_rows: list[dict[str, Any]], variant: str, allowed: set[tuple[int, int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in chunk_rows:
        if _pair(row) not in allowed:
            continue
        out.append(
            {
                "anchor_variant": variant,
                "scene_id": row["_scene"],
                "chunk_id": row["_chunk_id"],
                "frame_id": row["_frame_id"],
                "mask_id": row["_mask_id"],
                "mask_observation_id": _node_token(row["_scene"], row["_frame_id"], row["_mask_id"]),
                "source_type": "+".join(sorted(set(row["_source_types"]))),
                "area_ratio": row["_area_ratio"],
                "bbox": row.get("bbox"),
                "bbox_center": json.dumps(row["_bbox_center"]),
                "bbox_size": json.dumps(row["_bbox_size"]),
                "aspect_ratio": row["_aspect_ratio"],
                "mask_solidity_proxy": row["_solidity"],
                "boundary_length_proxy": row.get("mask_boundary_length"),
                "D4RT_material_count": row["_component_count"],
                "D4RT_visible_count": row["_component_count"],
                "D4RT_support_entropy": row["_component_entropy"],
                "DINO_feature_norm": "",
                "DINO_feature_norm_available": False,
                "DINO_mode_id": row["_DINO_mode_id"],
                "DINO_feature_valid": row["_DINO_feature_valid"],
                "DINO_intra_mask_variance": "",
                "DINO_intra_mask_variance_available": False,
                "repeated_signature_id": row["_signature"],
                "same_frame_overlap_count": row["_same_frame_overlap"],
                "underseg_risk": row["_underseg"],
                "diagnostic_gt_underseg_risk": row["_diagnostic_gt_underseg"],
                "method_underseg_material_multimodal": row["_method_underseg_material_multimodal"],
                "method_underseg_overlap_multimask": row["_method_underseg_overlap_multimask"],
                "method_underseg_broad_overlap": row["_method_underseg_broad_overlap"],
                "large_mask_risk": row["_large"],
                "small_mask_risk": row["_small"],
                "anchor_objectness_score": row["_anchor_objectness"],
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": False,
                "diagnostic_only": True,
            }
        )
    return out


def _write_visual(
    *,
    visual_root: Path,
    scene: str,
    frame_data: list[dict[str, Any]],
    variants: dict[str, set[tuple[int, int]]],
    wrote: int,
    max_images: int,
) -> list[dict[str, Any]]:
    if wrote >= max_images:
        return []
    rows: list[dict[str, Any]] = []
    for item in frame_data:
        if wrote >= max_images:
            break
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        gt = item["gt"]
        if mask is None:
            continue
        panels = [_colorize_labels(gt), _colorize_labels(mask)]
        for variant in ["A2_d4rt_supported_objectlike", "A5_area_balanced_objectlike"]:
            allowed = variants.get(variant, set())
            pred = np.zeros(mask.shape, dtype=np.int64)
            for mask_id in np.unique(mask):
                mask_id_i = int(mask_id)
                if mask_id_i > 0 and (frame_id, mask_id_i) in allowed:
                    pred[mask == mask_id_i] = mask_id_i
            panels.append(_colorize_labels(pred))
        panel = np.concatenate(panels, axis=1)
        path = visual_root / f"{scene}_frame{frame_id:06d}_gt_raw_A2_A5.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        rows.append({"scene_id": scene, "frame_id": frame_id, "visualization_path": _rel(path)})
        wrote += 1
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    variant_filter = {item.strip() for item in str(args.variant_filter or "").split(",") if item.strip()}
    candidate_path = _rooted(args.candidate_rows)
    edge_path = _rooted(args.edge_rows)
    all_candidates = _prep_candidate_rows(candidate_path, scenes, edge_path)
    candidates_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_candidates:
        candidates_by_chunk[str(row["_chunk_id"])].append(row)

    anchor_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    visualization_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    wrote_visual_count = 0

    for scene in scenes:
        print(f"[v69r2-anchor-bank] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            chunk_rows = candidates_by_chunk.get(chunk_id, [])
            if not chunk_rows:
                missing_rows.append({"scene_id": scene, "chunk_id": chunk_id, "missing": "candidate_rows"})
                continue
            variants = _variant_sets(chunk_rows, int(args.max_anchors_per_frame))
            if variant_filter:
                variants = {name: allowed for name, allowed in variants.items() if name in variant_filter}
            if wrote_visual_count < int(args.visual_frames):
                new_visuals = _write_visual(
                    visual_root=visual_root,
                    scene=scene,
                    frame_data=frame_data,
                    variants=variants,
                    wrote=wrote_visual_count,
                    max_images=int(args.visual_frames),
                )
                wrote_visual_count += len(new_visuals)
                visualization_rows.extend(new_visuals)
            for variant, allowed in sorted(variants.items()):
                anchor_rows.extend(_anchor_rows_for_variant(chunk_rows, variant, allowed))
                mapping, oracle_diag = _oracle_majority_mapping_bundle(
                    frame_data=frame_data,
                    selected_pairs=set(),
                    representative_pairs=allowed,
                )["representative"]
                diag = {
                    **oracle_diag,
                    **_anchor_diag(chunk_rows, allowed, len(frame_ids)),
                    "same_frame_cannot_link_violation_count": 0,
                    "shared_mask_count": int(sum(1 for row in chunk_rows if _pair(row) in allowed and row["_shared_support"])),
                    "reject_mask_count": int(sum(1 for row in chunk_rows if _pair(row) not in allowed)),
                    "unknown_mask_count": 0,
                }
                eval_row = _row_from_mapping(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    diag=diag,
                    pipeline_root=pipeline_root,
                    uses_gt_for_prediction=True,
                    forbidden_for_method_table=True,
                )
                eval_row["anchor_variant"] = variant
                eval_row.update(diag)
                eval_row["anchor_oracle_SF50"] = eval_row.get("local_SF50")
                eval_row["anchor_oracle_AP50"] = eval_row.get("local_AP50")
                eval_row["anchor_GT_best_IoU_mean"] = eval_row.get("local_GT_best_IoU_mean")
                eval_row["selection_uses_gt_for_prediction"] = False
                eval_row["oracle_metric_uses_gt_for_prediction"] = True
                eval_row["source_scope"] = "v69r2_anchor_bank_diagnostic_oracle"
                metric_rows.append(eval_row)

    variant_rows = [_summarize_anchor_variant(metric_rows, variant) for variant in sorted({row["anchor_variant"] for row in metric_rows})]
    non_oracle = [row for row in variant_rows if not str(row.get("anchor_variant", "")).startswith("A6_")]

    def _passes_anchor_core(row: dict[str, Any]) -> bool:
        count = _float_or_none(row.get("anchor_count_per_chunk_mean"))
        gt = _float_or_none(row.get("diagnostic_GT_count_per_chunk_mean"))
        per_frame = _float_or_none(row.get("anchor_count_per_frame_mean"))
        support = _float_or_none(row.get("anchor_D4RT_support_rate"))
        underseg = _float_or_none(row.get("anchor_underseg_rate"))
        sf50 = _float_or_none(row.get("anchor_oracle_SF50"))
        return bool(
            count is not None
            and gt is not None
            and count >= 0.5 * gt
            and per_frame is not None
            and per_frame <= 20.0
            and support is not None
            and support >= 0.50
            and underseg is not None
            and underseg <= 0.15
            and sf50 is not None
            and sf50 >= 0.20
        )

    passing = [row for row in non_oracle if _passes_anchor_core(row)]
    if passing:
        best = max(passing, key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0))
    else:
        best = max(non_oracle, key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0), default={})

    count = _float_or_none(best.get("anchor_count_per_chunk_mean"))
    gt = _float_or_none(best.get("diagnostic_GT_count_per_chunk_mean"))
    per_frame = _float_or_none(best.get("anchor_count_per_frame_mean"))
    support = _float_or_none(best.get("anchor_D4RT_support_rate"))
    underseg = _float_or_none(best.get("anchor_underseg_rate"))
    sf50 = _float_or_none(best.get("anchor_oracle_SF50"))
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "anchor_count_per_chunk_ge_half_GT": count is not None and gt is not None and count >= 0.5 * gt,
        "anchor_count_per_frame_mean_le_20": per_frame is not None and per_frame <= 20.0,
        "anchor_D4RT_support_rate_ge_0p50": support is not None and support >= 0.50,
        "anchor_underseg_rate_le_0p15": underseg is not None and underseg <= 0.15,
        "anchor_oracle_SF50_ge_0p20": sf50 is not None and sf50 >= 0.20,
        "non_oracle_selection_uses_gt_for_prediction_false": True,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    decision = "PASS_ANCHOR_BANK" if gate["pass"] else "NO_GO_ANCHOR_BANK"

    _write_csv(output_root / "anchor_rows.csv", anchor_rows)
    _write_csv(output_root / "anchor_metric_rows.csv", metric_rows)
    _write_csv(output_root / "anchor_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_csv(output_root / "visualization_rows.csv", visualization_rows)
    summary = {
        "phase": "v69r2_anchor_bank",
        "decision": decision,
        "gate": gate,
        "best_anchor_variant": best,
        "candidate_rows": _rel(candidate_path),
        "edge_rows_for_dino_metadata": _rel(edge_path),
        "scenes": scenes,
        "stride": int(args.stride),
        "pipeline_roots": pipeline_roots,
        "rows": {
            "anchor_rows_csv": _rel(output_root / "anchor_rows.csv"),
            "anchor_metric_rows_csv": _rel(output_root / "anchor_metric_rows.csv"),
            "anchor_variant_summary_rows_csv": _rel(output_root / "anchor_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
            "visualization_rows_csv": _rel(output_root / "visualization_rows.csv"),
        },
        "visual_root": _rel(visual_root),
        "notes": [
            "Anchor selection rows do not use GT labels.",
            "Anchor oracle metrics use GT majority mapping for diagnostic evaluation only and are forbidden for method tables.",
            "DINO mode metadata is imported from v68 frozen-appearance edge rows; DINO feature norm and intra-mask variance are marked unavailable instead of fabricated.",
        ],
    }
    _write_json(output_root / "anchor_bank_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "anchor_bank_summary.json",
        output_root / "anchor_rows.csv",
        output_root / "anchor_metric_rows.csv",
        output_root / "anchor_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
        output_root / "visualization_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v69-r2 Phase 1: build object-like anchor bank.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--edge-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v69r2_anchor_bank")
    parser.add_argument("--visual-root", default="outputs/audit/v69r2_visualizations/anchor_bank")
    parser.add_argument("--max-anchors-per-frame", type=int, default=20)
    parser.add_argument("--visual-frames", type=int, default=12)
    parser.add_argument("--variant-filter", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
