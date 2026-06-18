from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.measurement_bank import MaskMeasurement
from stream4d_native.semantic_material_mask_split import backfill_masks_by_material_support
from stream4d_native.semantic_material_part_graph import build_token_material_support
from tools.run_v42_semantic_part_audit import (
    _label_maps_from_masks,
    _load_d4rt_records,
    _npz_source_masks,
    _prepared_masks,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReasonSet:
    name: str
    reasons: tuple[str, ...] | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_frame_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_bool(value: str) -> bool | None:
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"true", "1"}:
        return True
    if text.lower() in {"false", "0"}:
        return False
    return None


def _parse_float(value: str, default: float = 0.0) -> float:
    text = str(value).strip()
    if not text:
        return float(default)
    return float(text)


def _parse_int(value: str, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return int(default)
    return int(text)


def _load_query_rows(query_root: Path, scene: str, variant: str) -> dict[str, dict[str, Any]]:
    rows = _read_csv(query_root / scene / "query_rows.csv")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("variant", "")) != str(variant):
            continue
        frame_id = _parse_int(row.get("frame_id", ""))
        x = _parse_int(row.get("x", ""))
        y = _parse_int(row.get("y", ""))
        key = f"{frame_id}:{x}:{y}"
        out[key] = {
            "reason": str(row.get("reason", "unknown")),
            "score": _parse_float(row.get("score", ""), default=0.0),
            "src_mask_id": _parse_int(row.get("src_mask_id", ""), default=0),
            "query_index": _parse_int(row.get("query_index", ""), default=-1),
        }
    return out


def _load_tokens(part_graph_root: Path, variant: str, scene: str, source: str) -> list[SimpleNamespace]:
    rows = _read_csv(part_graph_root / variant / scene / "part_token_rows.csv")
    tokens: list[SimpleNamespace] = []
    for row in rows:
        if str(row.get("source", "")) != str(source):
            continue
        tokens.append(
            SimpleNamespace(
                token_id=_parse_int(row.get("token_id", "")),
                frame_id=_parse_int(row.get("frame_id", "")),
                mask_id=_parse_int(row.get("mask_id", "")),
                diagnostic_gt_instance=None
                if not str(row.get("diagnostic_gt_instance", "")).strip()
                else _parse_int(row.get("diagnostic_gt_instance", "")),
                diagnostic_gt_iou=None
                if not str(row.get("diagnostic_gt_iou", "")).strip()
                else _parse_float(row.get("diagnostic_gt_iou", "")),
            )
        )
    return tokens


def _load_edges(part_graph_root: Path, variant: str, scene: str, source: str) -> list[SimpleNamespace]:
    rows = _read_csv(part_graph_root / variant / scene / "part_edge_rows.csv")
    edges: list[SimpleNamespace] = []
    for row in rows:
        if str(row.get("source", "")) != str(source):
            continue
        edges.append(
            SimpleNamespace(
                token_i=_parse_int(row.get("token_i", "")),
                token_j=_parse_int(row.get("token_j", "")),
                object_affinity=_parse_float(row.get("object_affinity", "")),
                diagnostic_same_gt=_parse_bool(row.get("diagnostic_same_gt", "")),
                semantic_affinity=_parse_float(row.get("semantic_affinity", "")),
                same_frame_cannot_link=_parse_bool(row.get("same_frame_cannot_link", "")),
                frame_i=None if not str(row.get("frame_i", "")).strip() else _parse_int(row.get("frame_i", "")),
                frame_j=None if not str(row.get("frame_j", "")).strip() else _parse_int(row.get("frame_j", "")),
                mask_i=None if not str(row.get("mask_i", "")).strip() else _parse_int(row.get("mask_i", "")),
                mask_j=None if not str(row.get("mask_j", "")).strip() else _parse_int(row.get("mask_j", "")),
            )
        )
    return edges


def _source_masks(
    *,
    source: str,
    stream: ScanNetStream,
    scene: str,
    frame_ids: list[int],
    min_area: int,
    sample_frames: int,
    external_root: Path,
    d4rt_records: list[Any],
    backfill_overlap_iou: float,
    backfill_max_masks_per_frame: int,
    material_backfill_min_tubes: int,
    material_backfill_max_candidate_area_fraction: float,
    material_min_visibility: float,
    material_min_confidence: float,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[str, Any]]:
    if source == "dinov2_maskcut":
        return (
            _npz_source_masks(external_root, scene, "dinov2_maskcut", frame_ids, min_area, sample_frames),
            {"strategy": "external_dinov2_maskcut"},
        )
    if source == "dinov2_maskcut_prepared_material_backfill":
        primary = _npz_source_masks(external_root, scene, "dinov2_maskcut", frame_ids, min_area, sample_frames)
        prepared = _prepared_masks(stream, frame_ids, min_area)
        masks, diag = backfill_masks_by_material_support(
            primary,
            [prepared],
            d4rt_records,
            overlap_iou=float(backfill_overlap_iou),
            max_backfill_per_frame=int(backfill_max_masks_per_frame),
            min_tubes=int(material_backfill_min_tubes),
            max_candidate_area_fraction=float(material_backfill_max_candidate_area_fraction),
            min_visibility=float(material_min_visibility),
            min_confidence=float(material_min_confidence),
        )
        return masks, {"strategy": "dinov2_maskcut_prepared_material_backfill", "backfill_diag": diag}
    raise ValueError(f"unsupported source for reason diagnostic: {source}")


def _reason_sets() -> list[ReasonSet]:
    return [
        ReasonSet("all_reasons", None),
        ReasonSet("fixed_grid", ("fixed_grid",)),
        ReasonSet("mask_boundary", ("mask_boundary",)),
        ReasonSet("overlap_anchor", ("overlap_anchor",)),
        ReasonSet("disagreement_proxy", ("disagreement_proxy",)),
        ReasonSet("mask_interior", ("mask_interior",)),
        ReasonSet("exploration", ("exploration", "exploration_fill")),
        ReasonSet(
            "q5_semantic_nonexplore",
            ("mask_boundary", "overlap_anchor", "disagreement_proxy", "mask_interior"),
        ),
        ReasonSet(
            "q5_boundary_overlap_disagreement",
            ("mask_boundary", "overlap_anchor", "disagreement_proxy"),
        ),
    ]


def _tube_visible_at(record: Any, local_idx: int, *, min_visibility: float, min_confidence: float) -> bool:
    visibility = np.asarray(record.get_geometry_for_measurement(field="visibility"), dtype=np.float32)
    confidence = np.asarray(record.get_geometry_for_measurement(field="confidence"), dtype=np.float32)
    return bool(visibility[local_idx] >= float(min_visibility) and confidence[local_idx] >= float(min_confidence))


def _build_fast_material_measurements(
    records: list[Any],
    *,
    masks_by_frame: dict[int, np.ndarray],
    min_visibility: float,
    min_confidence: float,
) -> tuple[list[MaskMeasurement], dict[str, Any]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    visible_by_frame: dict[int, set[int]] = {}
    skipped_no_mask = 0
    skipped_background = 0
    skipped_invalid_uv = 0
    for record in records:
        frames = np.asarray(record.target_frames_global, dtype=np.int64).reshape(-1)
        uv_all = np.asarray(record.get_geometry_for_measurement(field="uv"), dtype=np.float32)
        for local_idx, frame_value in enumerate(frames.tolist()):
            if not _tube_visible_at(
                record,
                local_idx,
                min_visibility=float(min_visibility),
                min_confidence=float(min_confidence),
            ):
                continue
            frame_id = int(frame_value)
            uv = uv_all[local_idx]
            if not (np.isfinite(uv).all() and 0.0 <= float(uv[0]) <= 1.0 and 0.0 <= float(uv[1]) <= 1.0):
                skipped_invalid_uv += 1
                continue
            visible_by_frame.setdefault(frame_id, set()).add(int(record.tube_id))
            mask = masks_by_frame.get(frame_id)
            if mask is None:
                skipped_no_mask += 1
                continue
            height, width = mask.shape[:2]
            x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
            y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
            mask_id = int(mask[y, x])
            if mask_id <= 0:
                skipped_background += 1
                continue
            grouped.setdefault((frame_id, mask_id), set()).add(int(record.tube_id))

    measurements: list[MaskMeasurement] = []
    for (frame_id, mask_id), inside_values in sorted(grouped.items()):
        inside = sorted(int(v) for v in inside_values)
        outside = sorted(visible_by_frame.get(int(frame_id), set()) - set(inside))
        mask_area = 0
        if int(frame_id) in masks_by_frame:
            mask_area = int(np.count_nonzero(masks_by_frame[int(frame_id)] == int(mask_id)))
        measurements.append(
            MaskMeasurement(
                measurement_id=f"f{int(frame_id):06d}_m{int(mask_id):04d}",
                frame_global=int(frame_id),
                mask_id=int(mask_id),
                tube_ids=inside,
                inside_tube_ids=inside,
                mask_area=mask_area,
                boundary_tube_ids=[],
                outside_visible_tube_ids=outside,
                metadata={"fast_reason_diagnostic": True},
            )
        )
    diagnostics = {
        "tube_count": int(len(records)),
        "measurement_count": int(len(measurements)),
        "skipped_no_mask_projection_count": int(skipped_no_mask),
        "skipped_background_projection_count": int(skipped_background),
        "skipped_invalid_uv_projection_count": int(skipped_invalid_uv),
        "measurement_uses_metric_geometry": False,
        "measurement_geometry_fields": ["uv", "visibility", "confidence"],
        "fast_reason_diagnostic": True,
    }
    return measurements, diagnostics


def _filter_ids(ids: tuple[int, ...], tube_reason: dict[int, str], reason_set: ReasonSet) -> set[int]:
    raw = {int(v) for v in ids}
    if reason_set.reasons is None:
        return raw
    allowed = set(reason_set.reasons)
    return {tube_id for tube_id in raw if tube_reason.get(int(tube_id), "unmatched") in allowed}


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _fast_auc_from_scores(labels: list[bool], scores: list[float]) -> float | None:
    if not labels:
        return None
    label_arr = np.asarray(labels, dtype=bool)
    score_arr = np.asarray(scores, dtype=np.float64)
    positive_count = int(np.count_nonzero(label_arr))
    negative_count = int(label_arr.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(score_arr, kind="mergesort")
    sorted_scores = score_arr[order]
    ranks = np.empty(score_arr.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average 1-based rank for tied scores.
        ranks[order[start:end]] = 0.5 * (float(start + 1) + float(end))
        start = end
    positive_rank_sum = float(np.sum(ranks[label_arr]))
    auc = (positive_rank_sum - float(positive_count * (positive_count + 1)) / 2.0) / float(
        positive_count * negative_count
    )
    return float(auc)


def _edge_scores_for_reason(
    *,
    edges: list[SimpleNamespace],
    support_by_token: dict[int, Any],
    tube_reason: dict[int, str],
    reason_set: ReasonSet,
    material_weight: float,
    conflict_weight: float,
    min_shared_tube_count: int,
    material_support_shrinkage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    false_rows: list[dict[str, Any]] = []
    labels: list[bool] = []
    p3_scores: list[float] = []
    p4_scores: list[float] = []
    p5_scores: list[float] = []
    positive_p3: list[float] = []
    negative_p3: list[float] = []
    predicted_p3 = 0
    predicted_p4 = 0
    predicted_p5 = 0
    false_p3 = 0
    false_p4 = 0
    false_p5 = 0
    labeled_count = 0
    p3_label_pairs: list[tuple[float, bool]] = []
    for edge in edges:
        left = support_by_token.get(int(edge.token_i))
        right = support_by_token.get(int(edge.token_j))
        left_inside = _filter_ids(left.inside_tube_ids if left else (), tube_reason, reason_set)
        right_inside = _filter_ids(right.inside_tube_ids if right else (), tube_reason, reason_set)
        shared = left_inside & right_inside
        union = left_inside | right_inside
        left_outside = _filter_ids(left.outside_visible_tube_ids if left else (), tube_reason, reason_set)
        right_outside = _filter_ids(right.outside_visible_tube_ids if right else (), tube_reason, reason_set)
        conflict = (left_inside & right_outside) | (right_inside & left_outside)
        shared_count = int(len(shared))
        material_jaccard = float(shared_count / max(len(union), 1)) if union else 0.0
        if shared_count < int(min_shared_tube_count):
            material_jaccard = 0.0
        if material_jaccard > 0.0 and float(material_support_shrinkage) > 0.0:
            shrink = float(shared_count) / (float(shared_count) + float(material_support_shrinkage))
            material_jaccard *= shrink
        conflict_ratio = float(len(conflict) / max(len(union), 1)) if union else 0.0
        p3 = float(material_jaccard - float(conflict_weight) * conflict_ratio)
        p4 = float(edge.object_affinity) + float(material_weight) * material_jaccard
        p5 = p4 - float(conflict_weight) * conflict_ratio
        label = edge.diagnostic_same_gt
        if label is not None:
            is_same = bool(label)
            labels.append(is_same)
            p3_scores.append(p3)
            p4_scores.append(p4)
            p5_scores.append(p5)
            p3_label_pairs.append((p3, is_same))
            labeled_count += 1
            if p3 >= 0.5:
                predicted_p3 += 1
                false_p3 += int(not is_same)
            if p4 >= 0.5:
                predicted_p4 += 1
                false_p4 += int(not is_same)
            if p5 >= 0.5:
                predicted_p5 += 1
                false_p5 += int(not is_same)
            if is_same:
                positive_p3.append(p3)
            else:
                negative_p3.append(p3)
                if union:
                    false_rows.append(
                        {
                            "token_i": int(edge.token_i),
                            "token_j": int(edge.token_j),
                            "frame_i": edge.frame_i,
                            "frame_j": edge.frame_j,
                            "mask_i": edge.mask_i,
                            "mask_j": edge.mask_j,
                            "diagnostic_same_gt": False,
                            "object_affinity": float(edge.object_affinity),
                            "material_jaccard": material_jaccard,
                            "shared_tube_count": shared_count,
                            "material_union_count": int(len(union)),
                            "visible_outside_conflict_ratio": conflict_ratio,
                            "p3_d4rt_only_affinity": p3,
                            "p4_semantic_material_affinity": p4,
                            "p5_semantic_material_boundary_affinity": p5,
                        }
                    )
    top_k = min(len(p3_scores), max(10, int(np.ceil(len(p3_scores) * 0.01)))) if p3_scores else 0
    top_negative_rate = None
    if top_k:
        top = sorted(p3_label_pairs, key=lambda item: float(item[0]), reverse=True)[:top_k]
        top_negative_rate = float(sum(1 for _score, is_same in top if not is_same) / max(len(top), 1))
    neg_p90 = _safe_quantile(negative_p3, 0.90)
    pos_below_neg_p90 = None
    if neg_p90 is not None and positive_p3:
        pos_below_neg_p90 = float(sum(1 for score in positive_p3 if score < float(neg_p90)) / max(len(positive_p3), 1))
    summary = {
        "reason_set": reason_set.name,
        "reason_filter": "*" if reason_set.reasons is None else ",".join(reason_set.reasons),
        "material_min_shared_tubes": int(min_shared_tube_count),
        "material_support_shrinkage": float(material_support_shrinkage),
        "edge_count": int(len(edges)),
        "gt_labeled_edge_count": int(labeled_count),
        "positive_edge_count": int(sum(1 for item in labels if item)),
        "negative_edge_count": int(sum(1 for item in labels if not item)),
        "p3_d4rt_only_AUC": _fast_auc_from_scores(labels, p3_scores) if labels else None,
        "p4_semantic_material_AUC": _fast_auc_from_scores(labels, p4_scores) if labels else None,
        "p5_semantic_material_boundary_AUC": _fast_auc_from_scores(labels, p5_scores) if labels else None,
        "p3_positive_mean": _safe_mean(positive_p3),
        "p3_negative_mean": _safe_mean(negative_p3),
        "p3_positive_p50": _safe_quantile(positive_p3, 0.50),
        "p3_negative_p90": neg_p90,
        "p3_positive_below_negative_p90_rate": pos_below_neg_p90,
        "p3_top_1pct_negative_rate": top_negative_rate,
        "p3_predicted_merge_count_at_050": int(predicted_p3),
        "p3_false_merge_count_at_050": int(false_p3),
        "p3_false_merge_rate_at_050": float(false_p3 / max(predicted_p3, 1)),
        "p4_predicted_merge_count_at_050": int(predicted_p4),
        "p4_false_merge_count_at_050": int(false_p4),
        "p4_false_merge_rate_at_050": float(false_p4 / max(predicted_p4, 1)),
        "p5_predicted_merge_count_at_050": int(predicted_p5),
        "p5_false_merge_count_at_050": int(false_p5),
        "p5_false_merge_rate_at_050": float(false_p5 / max(predicted_p5, 1)),
    }
    return summary, false_rows


def _tube_support_rows(
    *,
    scene: str,
    variant: str,
    source: str,
    records: list[Any],
    measurements: list[Any],
    support_by_token: dict[int, Any],
    tube_reason: dict[int, str],
    matched_keys: set[str],
) -> list[dict[str, Any]]:
    all_reasons = sorted({tube_reason.get(int(record.tube_id), "unmatched") for record in records})
    rows = []
    for reason in all_reasons:
        tube_ids = {int(record.tube_id) for record in records if tube_reason.get(int(record.tube_id), "unmatched") == reason}
        inside_occurrences = 0
        boundary_occurrences = 0
        outside_occurrences = 0
        supported_tokens = 0
        for support in support_by_token.values():
            inside = set(int(v) for v in support.inside_tube_ids) & tube_ids
            boundary = set(int(v) for v in support.boundary_tube_ids) & tube_ids
            outside = set(int(v) for v in support.outside_visible_tube_ids) & tube_ids
            inside_occurrences += len(inside)
            boundary_occurrences += len(boundary)
            outside_occurrences += len(outside)
            supported_tokens += int(bool(inside))
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "source": source,
                "reason": reason,
                "tube_count": int(len(tube_ids)),
                "matched_query_pixel_count": int(len(matched_keys)),
                "measurement_count": int(len(measurements)),
                "token_inside_occurrence_count": int(inside_occurrences),
                "token_boundary_occurrence_count": int(boundary_occurrences),
                "token_outside_occurrence_count": int(outside_occurrences),
                "supported_token_count": int(supported_tokens),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v42 material edge quality by semantic occupancy query reason.")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variants", default="Q0,Q5")
    parser.add_argument("--sources", default="dinov2_maskcut,dinov2_maskcut_prepared_material_backfill")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--part-graph-root", required=True)
    parser.add_argument("--material-cache-root", required=True)
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external_stride1_smoke")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--material-max-tubes-per-window", type=int, default=320)
    parser.add_argument("--material-image-width", type=int, default=1296)
    parser.add_argument("--material-image-height", type=int, default=968)
    parser.add_argument("--material-min-visibility", type=float, default=0.5)
    parser.add_argument("--material-min-confidence", type=float, default=0.5)
    parser.add_argument("--material-weight", type=float, default=0.35)
    parser.add_argument("--material-conflict-weight", type=float, default=0.35)
    parser.add_argument("--material-min-shared-tubes", type=int, default=1)
    parser.add_argument("--material-support-shrinkage", type=float, default=0.0)
    parser.add_argument("--backfill-overlap-iou", type=float, default=0.10)
    parser.add_argument("--backfill-max-masks-per-frame", type=int, default=8)
    parser.add_argument("--material-backfill-min-tubes", type=int, default=1)
    parser.add_argument("--material-backfill-max-candidate-area-fraction", type=float, default=1.0)
    args = parser.parse_args()

    frame_ids = _parse_frame_ids(args.frame_ids)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    query_root = ROOT / str(args.query_root)
    part_graph_root = ROOT / str(args.part_graph_root)
    material_cache_root = ROOT / str(args.material_cache_root)
    external_root = ROOT / str(args.external_source_root)
    output_root = ROOT / str(args.output_root)

    summary_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    top_false_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene)
        for variant in variants:
            records, d4rt_diag = _load_d4rt_records(
                cache_root=material_cache_root / variant,
                scene=scene,
                max_tubes_per_window=int(args.material_max_tubes_per_window),
                image_width=int(args.material_image_width),
                image_height=int(args.material_image_height),
            )
            query_rows = _load_query_rows(query_root, scene, variant)
            matched_keys = {record.source_pixel_key for record in records if record.source_pixel_key in query_rows}
            tube_reason = {
                int(record.tube_id): str(query_rows.get(record.source_pixel_key, {}).get("reason", "unmatched"))
                for record in records
            }
            for source in sources:
                masks_by_frame, mask_diag = _source_masks(
                    source=source,
                    stream=stream,
                    scene=scene,
                    frame_ids=frame_ids,
                    min_area=int(args.min_area),
                    sample_frames=int(args.sample_frames),
                    external_root=external_root,
                    d4rt_records=records,
                    backfill_overlap_iou=float(args.backfill_overlap_iou),
                    backfill_max_masks_per_frame=int(args.backfill_max_masks_per_frame),
                    material_backfill_min_tubes=int(args.material_backfill_min_tubes),
                    material_backfill_max_candidate_area_fraction=float(
                        args.material_backfill_max_candidate_area_fraction
                    ),
                    material_min_visibility=float(args.material_min_visibility),
                    material_min_confidence=float(args.material_min_confidence),
                )
                label_maps = _label_maps_from_masks(masks_by_frame)
                measurements, measurement_diag = _build_fast_material_measurements(
                    records,
                    masks_by_frame=label_maps,
                    min_visibility=float(args.material_min_visibility),
                    min_confidence=float(args.material_min_confidence),
                )
                tokens = _load_tokens(part_graph_root, variant, scene, source)
                edges = _load_edges(part_graph_root, variant, scene, source)
                support_by_token = build_token_material_support(tokens, measurements)
                support_rows.extend(
                    _tube_support_rows(
                        scene=scene,
                        variant=variant,
                        source=source,
                        records=records,
                        measurements=measurements,
                        support_by_token=support_by_token,
                        tube_reason=tube_reason,
                        matched_keys=matched_keys,
                    )
                )
                for reason_set in _reason_sets():
                    summary, false_edges = _edge_scores_for_reason(
                        edges=edges,
                        support_by_token=support_by_token,
                        tube_reason=tube_reason,
                        reason_set=reason_set,
                        material_weight=float(args.material_weight),
                        conflict_weight=float(args.material_conflict_weight),
                        min_shared_tube_count=int(args.material_min_shared_tubes),
                        material_support_shrinkage=float(args.material_support_shrinkage),
                    )
                    summary_rows.append(
                        {
                            "scene": scene,
                            "variant": variant,
                            "source": source,
                            "token_count": int(len(tokens)),
                            "d4rt_record_count": int(len(records)),
                            "query_row_count": int(len(query_rows)),
                            "matched_record_query_count": int(len(matched_keys)),
                            "measurement_count": int(measurement_diag.get("measurement_count", 0)),
                            "source_mask_strategy": mask_diag.get("strategy", ""),
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                            "diagnostic_only": True,
                            **summary,
                        }
                    )
                    for rank, row in enumerate(
                        sorted(false_edges, key=lambda item: float(item["p3_d4rt_only_affinity"]), reverse=True)[:25],
                        start=1,
                    ):
                        top_false_rows.append(
                            {
                                "scene": scene,
                                "variant": variant,
                                "source": source,
                                "reason_set": reason_set.name,
                                "rank": int(rank),
                                **row,
                            }
                        )
                manifest_rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "source": source,
                        "records": d4rt_diag,
                        "measurement_diag": measurement_diag,
                        "mask_diag": mask_diag,
                        "query_reason_counts": {
                            reason: int(sum(1 for value in tube_reason.values() if value == reason))
                            for reason in sorted(set(tube_reason.values()))
                        },
                    }
                )

    _write_csv(output_root / "reason_material_summary.csv", summary_rows)
    _write_csv(output_root / "reason_tube_support.csv", support_rows)
    _write_csv(output_root / "reason_top_false_edges.csv", top_false_rows)
    _write_json(
        output_root / "material_query_reason_manifest.json",
        {
            "scenes": scenes,
            "variants": variants,
            "sources": sources,
            "frame_ids": frame_ids,
            "query_root": str(query_root),
            "part_graph_root": str(part_graph_root),
            "material_cache_root": str(material_cache_root),
            "material_min_shared_tubes": int(args.material_min_shared_tubes),
            "material_support_shrinkage": float(args.material_support_shrinkage),
            "summary_row_count": int(len(summary_rows)),
            "support_row_count": int(len(support_rows)),
            "top_false_row_count": int(len(top_false_rows)),
            "rows": manifest_rows,
            "note": "Diagnostic-only reason provenance analysis. GT is used only for labels/AUC, never prediction.",
        },
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "summary_rows": len(summary_rows),
                    "support_rows": len(support_rows),
                    "top_false_rows": len(top_false_rows),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
