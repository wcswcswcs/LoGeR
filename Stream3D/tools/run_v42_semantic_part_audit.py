from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.frozen_feature_adapter import (
    FrozenFeatureAdapter,
    locate_default_dinov2_checkpoint,
    locate_default_radio_checkpoint,
)
from stream4d_native.measurement_bank import build_measurement_bank
from stream4d_native.semantic_material_mask_split import backfill_masks_by_material_support, split_masks_by_material_uv
from stream4d_native.semantic_material_part_graph import (
    build_material_part_graph_edges,
    build_token_material_support,
    summarize_material_part_graph,
)
from stream4d_native.semantic_part_graph import build_part_graph_edges, summarize_part_graph
from stream4d_native.semantic_part_tokens import (
    build_semantic_part_tokens,
    label_map_to_masks,
    merge_masks_by_feature_affinity,
    split_masks_by_feature_clusters,
    stack_to_masks,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _sample_ids(ids: list[int], count: int) -> list[int]:
    if len(ids) <= int(count):
        return ids
    idx = np.linspace(0, len(ids) - 1, num=int(count), dtype=np.int64)
    return [ids[int(i)] for i in idx.tolist()]


def _parse_frame_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_gt(stream: ScanNetStream, frame_id: int) -> np.ndarray | None:
    path = stream.root / "instance" / "instance" / f"{int(frame_id)}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int32)


def _prepared_masks(stream: ScanNetStream, frame_ids: list[int], min_area: int) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame_id in frame_ids:
        try:
            label = stream.load_mask(frame_id)
        except FileNotFoundError:
            out[int(frame_id)] = []
            continue
        out[int(frame_id)] = label_map_to_masks(label, min_area=min_area)
    return out


def _npz_source_masks(
    root: Path,
    scene: str,
    source: str,
    frame_ids: list[int],
    min_area: int,
    sample_count: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    scene_source_dir = root / str(scene) / source / "sample8"
    flat_source_dir = root / source / "sample8"
    source_dir = scene_source_dir if scene_source_dir.exists() else flat_source_dir
    requested = [source_dir / f"{source}_frame{int(frame_id):06d}_masks.npz" for frame_id in frame_ids]
    paths = [path for path in requested if path.exists()]
    if len(paths) < int(sample_count):
        available = sorted(source_dir.glob(f"{source}_frame*_masks.npz"))
        if len(available) <= int(sample_count):
            paths = available
        elif available:
            indexes = np.linspace(0, len(available) - 1, num=int(sample_count), dtype=np.int64)
            paths = [available[int(index)] for index in indexes.tolist()]
    for path in paths:
        frame_text = path.stem.replace(f"{source}_frame", "").replace("_masks", "")
        frame_id = int(frame_text)
        if not path.exists():
            continue
        with np.load(path) as data:
            masks = data["masks"]
        out[int(frame_id)] = stack_to_masks(masks, min_area=min_area)
    return out


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / max(union, 1))


def _dedupe_masks(
    groups: list[dict[int, list[tuple[int, np.ndarray]]]],
    *,
    overlap_iou: float = 0.70,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    next_id = 1
    frame_ids = sorted({int(frame_id) for group in groups for frame_id in group})
    for frame_id in frame_ids:
        selected: list[tuple[int, np.ndarray]] = []
        for group in groups:
            for _mask_id, mask in group.get(int(frame_id), []):
                if any(_mask_iou(mask, existing) >= float(overlap_iou) for _existing_id, existing in selected):
                    continue
                selected.append((int(next_id), np.asarray(mask, dtype=bool)))
                next_id += 1
        out[int(frame_id)] = selected
    return out


def _backfill_masks(
    primary: dict[int, list[tuple[int, np.ndarray]]],
    supplements: list[dict[int, list[tuple[int, np.ndarray]]]],
    *,
    overlap_iou: float,
    max_backfill_per_frame: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_ids = sorted({int(frame_id) for frame_id in primary} | {int(frame_id) for group in supplements for frame_id in group})
    for frame_id in frame_ids:
        selected: list[tuple[int, np.ndarray]] = [
            (int(mask_id), np.asarray(mask, dtype=bool)) for mask_id, mask in primary.get(int(frame_id), [])
        ]
        next_id = max([int(mask_id) for mask_id, _mask in selected], default=0) + 1
        candidates: list[tuple[int, np.ndarray]] = []
        for group in supplements:
            for _mask_id, mask in group.get(int(frame_id), []):
                candidate = np.asarray(mask, dtype=bool)
                if any(_mask_iou(candidate, existing) >= float(overlap_iou) for _existing_id, existing in selected):
                    continue
                candidates.append((int(candidate.sum()), candidate))
        candidates.sort(key=lambda item: item[0], reverse=True)
        added = 0
        for _area, mask in candidates:
            if added >= int(max_backfill_per_frame):
                break
            if any(_mask_iou(mask, existing) >= float(overlap_iou) for _existing_id, existing in selected):
                continue
            selected.append((int(next_id), mask))
            next_id += 1
            added += 1
        out[int(frame_id)] = selected
    return out


def _label_maps_from_masks(masks_by_frame: dict[int, list[tuple[int, np.ndarray]]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for frame_id, masks in masks_by_frame.items():
        shape = None
        for _mask_id, mask in masks:
            shape = np.asarray(mask).shape
            break
        if shape is None:
            continue
        label = np.zeros(shape, dtype=np.int32)
        for mask_id, mask in masks:
            label[np.asarray(mask, dtype=bool)] = int(mask_id)
        out[int(frame_id)] = label
    return out


def _load_d4rt_records(
    *,
    cache_root: Path,
    scene: str,
    max_tubes_per_window: int,
    image_width: int,
    image_height: int,
) -> tuple[list[Any], dict[str, Any]]:
    from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache

    chunks, load_diag = load_scene_chunks_from_cache(
        cache_root / str(scene),
        max_tubes_per_window=int(max_tubes_per_window),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    builder = D4RTNativeSceneBuilder(
        object(),
        {"model": {"input": {"clip_frames": 32}}},
        temporal_chunk_size=32,
        temporal_chunk_stride=16,
    )
    stitched = builder.stitch_to_canonical(chunks)
    records = chunks_to_records(stitched)
    diag = dict(load_diag)
    diag.update(stitched.get("diagnostics", {}))
    diag["record_count"] = int(len(records))
    return records, diag


def _coverage_metrics(tokens: list[Any]) -> dict[str, Any]:
    gt_ids = {int(t.diagnostic_gt_instance) for t in tokens if t.diagnostic_gt_instance is not None}
    best: dict[int, float] = {}
    for token in tokens:
        if token.diagnostic_gt_instance is None or token.diagnostic_gt_iou is None:
            continue
        key = int(token.diagnostic_gt_instance)
        best[key] = max(best.get(key, 0.0), float(token.diagnostic_gt_iou))
    return {
        "diagnostic_gt_instance_count": int(len(gt_ids)),
        "coverage@0.10": float(sum(1 for gt in gt_ids if best.get(gt, 0.0) >= 0.10) / max(len(gt_ids), 1)),
        "coverage@0.25": float(sum(1 for gt in gt_ids if best.get(gt, 0.0) >= 0.25) / max(len(gt_ids), 1)),
    }


def _source_tokens(
    *,
    stream: ScanNetStream,
    scene: str,
    source: str,
    frame_ids: list[int],
    masks_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    adapter: FrozenFeatureAdapter,
    max_tokens: int,
    feature_maps_by_frame: dict[int, Any] | None = None,
    token_feature_mode: str = "pooled",
) -> list[Any]:
    tokens: list[Any] = []
    token_id = 0
    for frame_id in frame_ids:
        masks = masks_by_frame.get(int(frame_id), [])
        if not masks:
            continue
        rgb = stream.load_rgb(frame_id)
        fmap = feature_maps_by_frame.get(int(frame_id)) if feature_maps_by_frame else None
        if fmap is None:
            fmap = adapter.extract_dense_features(rgb)
        gt = _load_gt(stream, frame_id)
        current = build_semantic_part_tokens(
            frame_id=frame_id,
            frame=rgb,
            masks=masks,
            adapter=adapter,
            feature_map=fmap,
            gt_instance=gt,
            start_token_id=token_id,
            feature_mode=str(token_feature_mode),
        )
        tokens.extend(current)
        token_id += len(current)
        if len(tokens) >= int(max_tokens):
            return tokens[: int(max_tokens)]
    return tokens


def _feature_split_masks(
    *,
    stream: ScanNetStream,
    masks_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    frame_ids: list[int],
    adapter: FrozenFeatureAdapter,
    min_area: int,
    max_splits: int,
    spatial_weight: float,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, Any]]:
    split_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    feature_maps_by_frame: dict[int, Any] = {}
    for frame_id in frame_ids:
        masks = masks_by_frame.get(int(frame_id), [])
        if not masks:
            continue
        rgb = stream.load_rgb(frame_id)
        fmap = adapter.extract_dense_features(rgb)
        feature_maps_by_frame[int(frame_id)] = fmap
        split_by_frame[int(frame_id)] = split_masks_by_feature_clusters(
            masks,
            fmap,
            image_shape=rgb.shape[:2],
            min_area=int(min_area),
            max_splits=int(max_splits),
            spatial_weight=float(spatial_weight),
        )
    return split_by_frame, feature_maps_by_frame


def _feature_merge_masks(
    *,
    stream: ScanNetStream,
    masks_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    frame_ids: list[int],
    adapter: FrozenFeatureAdapter,
    min_area: int,
    affinity_threshold: float,
    max_center_distance: float,
    max_group_size: int,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, Any]]:
    merged_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    feature_maps_by_frame: dict[int, Any] = {}
    for frame_id in frame_ids:
        masks = masks_by_frame.get(int(frame_id), [])
        if not masks:
            continue
        rgb = stream.load_rgb(frame_id)
        fmap = adapter.extract_dense_features(rgb)
        feature_maps_by_frame[int(frame_id)] = fmap
        merged_by_frame[int(frame_id)] = merge_masks_by_feature_affinity(
            masks,
            fmap,
            image_shape=rgb.shape[:2],
            min_area=int(min_area),
            affinity_threshold=float(affinity_threshold),
            max_center_distance=float(max_center_distance),
            max_group_size=int(max_group_size),
        )
    return merged_by_frame, feature_maps_by_frame


def _material_split_masks(
    *,
    masks_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    d4rt_records: list[Any],
    min_area: int,
    max_splits: int,
    min_tubes: int,
    min_cluster_distance_px: float,
    max_mask_area_ratio: float,
    min_visibility: float,
    min_confidence: float,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[str, Any]]:
    if not d4rt_records:
        raise ValueError("material split source requires --material-cache-root with loadable D4RT records")
    return split_masks_by_material_uv(
        masks_by_frame,
        d4rt_records,
        min_area=int(min_area),
        max_splits=int(max_splits),
        min_tubes=int(min_tubes),
        min_cluster_distance_px=float(min_cluster_distance_px),
        max_mask_area_ratio=float(max_mask_area_ratio),
        min_visibility=float(min_visibility),
        min_confidence=float(min_confidence),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0081_01")
    parser.add_argument("--sources", default="prepared,watershed,dinov2_maskcut")
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external")
    parser.add_argument("--feature-backend", choices=["rgb_stats", "dinov2_timm", "radio_radseg"], default="radio_radseg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--sample-frames", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--feature-split-max-splits", type=int, default=3)
    parser.add_argument("--feature-split-spatial-weight", type=float, default=0.15)
    parser.add_argument("--feature-merge-affinity", type=float, default=0.95)
    parser.add_argument("--feature-merge-max-center-distance", type=float, default=0.35)
    parser.add_argument("--feature-merge-max-group-size", type=int, default=6)
    parser.add_argument("--backfill-overlap-iou", type=float, default=0.10)
    parser.add_argument("--backfill-max-masks-per-frame", type=int, default=8)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--token-feature-mode", choices=["pooled", "pooled_local_contrast"], default="pooled")
    parser.add_argument(
        "--semantic-affinity-mode",
        choices=[
            "cosine",
            "twohop_structure",
            "widest_structure",
            "temporal_widest_structure",
            "temporal_chain_structure",
        ],
        default="cosine",
    )
    parser.add_argument("--structure-topk", type=int, default=8)
    parser.add_argument("--structure-min-affinity", type=float, default=0.25)
    parser.add_argument("--structure-decay", type=float, default=0.95)
    parser.add_argument("--structure-temporal-window", type=int, default=300)
    parser.add_argument("--structure-temporal-rank-window", type=int, default=1)
    parser.add_argument("--material-cache-root", default="")
    parser.add_argument("--material-max-tubes-per-window", type=int, default=160)
    parser.add_argument("--material-image-width", type=int, default=1296)
    parser.add_argument("--material-image-height", type=int, default=968)
    parser.add_argument("--material-min-visibility", type=float, default=0.5)
    parser.add_argument("--material-min-confidence", type=float, default=0.5)
    parser.add_argument("--material-weight", type=float, default=0.35)
    parser.add_argument("--material-conflict-weight", type=float, default=0.35)
    parser.add_argument("--material-min-shared-tubes", type=int, default=1)
    parser.add_argument("--material-support-shrinkage", type=float, default=0.0)
    parser.add_argument("--material-backfill-min-tubes", type=int, default=1)
    parser.add_argument("--material-backfill-max-candidate-area-fraction", type=float, default=1.0)
    parser.add_argument("--material-split-max-splits", type=int, default=3)
    parser.add_argument("--material-split-min-tubes", type=int, default=2)
    parser.add_argument("--material-split-min-cluster-distance-px", type=float, default=16.0)
    parser.add_argument("--material-split-max-mask-area-ratio", type=float, default=0.50)
    parser.add_argument("--output-root", default="outputs/audit/v42_semantic_part_graph")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.scene)
    mask_ids = sorted(int(path.stem) for path in stream.mask_dir.glob("*.png"))
    frame_ids = _parse_frame_ids(str(args.frame_ids)) if str(args.frame_ids).strip() else _sample_ids(mask_ids, int(args.sample_frames))
    if args.feature_backend == "dinov2_timm":
        checkpoint = args.checkpoint or locate_default_dinov2_checkpoint()
    elif args.feature_backend == "radio_radseg":
        checkpoint = args.checkpoint or locate_default_radio_checkpoint()
    else:
        checkpoint = args.checkpoint
    adapter = FrozenFeatureAdapter(
        backend=args.feature_backend,
        device=str(args.device),
        checkpoint=checkpoint,
        radio_lang_model=args.radio_lang_model,
        radio_lang_align=bool(args.radio_lang_align),
    )
    external_root = ROOT / args.external_source_root
    source_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    all_summaries: dict[str, Any] = {}
    all_material_summaries: dict[str, Any] = {}
    d4rt_records: list[Any] = []
    d4rt_diag: dict[str, Any] = {}
    if str(args.material_cache_root).strip():
        d4rt_records, d4rt_diag = _load_d4rt_records(
            cache_root=ROOT / str(args.material_cache_root),
            scene=str(args.scene),
            max_tubes_per_window=int(args.material_max_tubes_per_window),
            image_width=int(args.material_image_width),
            image_height=int(args.material_image_height),
        )

    for source in [item.strip() for item in str(args.sources).split(",") if item.strip()]:
        feature_maps_by_frame: dict[int, Any] | None = None
        repair_strategy = ""
        material_split_diag: dict[str, Any] = {}
        material_backfill_diag: dict[str, Any] = {}
        if source == "prepared":
            masks_by_frame = _prepared_masks(stream, frame_ids, int(args.min_area))
            source_frame_ids = frame_ids
        elif source == "prepared_material_split":
            base_masks = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame, material_split_diag = _material_split_masks(
                masks_by_frame=base_masks,
                d4rt_records=d4rt_records,
                min_area=int(args.min_area),
                max_splits=int(args.material_split_max_splits),
                min_tubes=int(args.material_split_min_tubes),
                min_cluster_distance_px=float(args.material_split_min_cluster_distance_px),
                max_mask_area_ratio=float(args.material_split_max_mask_area_ratio),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            source_frame_ids = frame_ids
            repair_strategy = "d4rt_material_uv_split"
        elif source == "prepared_feature_split":
            base_masks = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame, feature_maps_by_frame = _feature_split_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                max_splits=int(args.feature_split_max_splits),
                spatial_weight=float(args.feature_split_spatial_weight),
            )
            source_frame_ids = frame_ids
            repair_strategy = "boundary_feature_cluster_split"
        elif source == "dinov2_maskcut_material_split":
            base_masks = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, material_split_diag = _material_split_masks(
                masks_by_frame=base_masks,
                d4rt_records=d4rt_records,
                min_area=int(args.min_area),
                max_splits=int(args.material_split_max_splits),
                min_tubes=int(args.material_split_min_tubes),
                min_cluster_distance_px=float(args.material_split_min_cluster_distance_px),
                max_mask_area_ratio=float(args.material_split_max_mask_area_ratio),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            repair_strategy = "d4rt_material_uv_split"
        elif source == "dinov2_maskcut_prepared_backfill":
            primary = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            prepared = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame = _backfill_masks(
                primary,
                [prepared],
                overlap_iou=float(args.backfill_overlap_iou),
                max_backfill_per_frame=int(args.backfill_max_masks_per_frame),
            )
            source_frame_ids = sorted(masks_by_frame)
            repair_strategy = "dinov2_low_overlap_prepared_backfill"
        elif source == "dinov2_maskcut_prepared_material_backfill":
            primary = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            prepared = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame, material_backfill_diag = backfill_masks_by_material_support(
                primary,
                [prepared],
                d4rt_records,
                overlap_iou=float(args.backfill_overlap_iou),
                max_backfill_per_frame=int(args.backfill_max_masks_per_frame),
                min_tubes=int(args.material_backfill_min_tubes),
                max_candidate_area_fraction=float(args.material_backfill_max_candidate_area_fraction),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            source_frame_ids = sorted(masks_by_frame)
            repair_strategy = "dinov2_material_supported_prepared_backfill"
        elif source == "dinov2_maskcut_hybrid_backfill":
            primary = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            watershed = _npz_source_masks(
                external_root,
                str(args.scene),
                "watershed",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            prepared = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame = _backfill_masks(
                primary,
                [watershed, prepared],
                overlap_iou=float(args.backfill_overlap_iou),
                max_backfill_per_frame=int(args.backfill_max_masks_per_frame),
            )
            source_frame_ids = sorted(masks_by_frame)
            repair_strategy = "dinov2_low_overlap_hybrid_backfill"
        elif source == "dinov2_maskcut_hybrid_material_backfill":
            primary = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            watershed = _npz_source_masks(
                external_root,
                str(args.scene),
                "watershed",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            prepared = _prepared_masks(stream, frame_ids, int(args.min_area))
            masks_by_frame, material_backfill_diag = backfill_masks_by_material_support(
                primary,
                [watershed, prepared],
                d4rt_records,
                overlap_iou=float(args.backfill_overlap_iou),
                max_backfill_per_frame=int(args.backfill_max_masks_per_frame),
                min_tubes=int(args.material_backfill_min_tubes),
                max_candidate_area_fraction=float(args.material_backfill_max_candidate_area_fraction),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            source_frame_ids = sorted(masks_by_frame)
            repair_strategy = "dinov2_material_supported_hybrid_backfill"
        elif source == "watershed_material_split":
            base_masks = _npz_source_masks(
                external_root,
                str(args.scene),
                "watershed",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, material_split_diag = _material_split_masks(
                masks_by_frame=base_masks,
                d4rt_records=d4rt_records,
                min_area=int(args.min_area),
                max_splits=int(args.material_split_max_splits),
                min_tubes=int(args.material_split_min_tubes),
                min_cluster_distance_px=float(args.material_split_min_cluster_distance_px),
                max_mask_area_ratio=float(args.material_split_max_mask_area_ratio),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            repair_strategy = "d4rt_material_uv_split"
        elif source == "watershed_feature_split":
            base_masks = _npz_source_masks(
                external_root,
                str(args.scene),
                "watershed",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, feature_maps_by_frame = _feature_split_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                max_splits=int(args.feature_split_max_splits),
                spatial_weight=float(args.feature_split_spatial_weight),
            )
            repair_strategy = "boundary_feature_cluster_split"
        elif source == "hybrid_union_feature_split":
            base_masks = _dedupe_masks(
                [
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "dinov2_maskcut",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "watershed",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _prepared_masks(stream, frame_ids, int(args.min_area)),
                ],
                overlap_iou=0.70,
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, feature_maps_by_frame = _feature_split_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                max_splits=int(args.feature_split_max_splits),
                spatial_weight=float(args.feature_split_spatial_weight),
            )
            repair_strategy = "hybrid_union_boundary_feature_cluster_split"
        elif source == "hybrid_union_material_split":
            base_masks = _dedupe_masks(
                [
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "dinov2_maskcut",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "watershed",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _prepared_masks(stream, frame_ids, int(args.min_area)),
                ],
                overlap_iou=0.70,
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, material_split_diag = _material_split_masks(
                masks_by_frame=base_masks,
                d4rt_records=d4rt_records,
                min_area=int(args.min_area),
                max_splits=int(args.material_split_max_splits),
                min_tubes=int(args.material_split_min_tubes),
                min_cluster_distance_px=float(args.material_split_min_cluster_distance_px),
                max_mask_area_ratio=float(args.material_split_max_mask_area_ratio),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            repair_strategy = "hybrid_union_d4rt_material_uv_split"
        elif source == "dinov2_maskcut_feature_merge":
            base_masks = _npz_source_masks(
                external_root,
                str(args.scene),
                "dinov2_maskcut",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, feature_maps_by_frame = _feature_merge_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                affinity_threshold=float(args.feature_merge_affinity),
                max_center_distance=float(args.feature_merge_max_center_distance),
                max_group_size=int(args.feature_merge_max_group_size),
            )
            repair_strategy = "semantic_layout_feature_merge"
        elif source == "watershed_feature_merge":
            base_masks = _npz_source_masks(
                external_root,
                str(args.scene),
                "watershed",
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, feature_maps_by_frame = _feature_merge_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                affinity_threshold=float(args.feature_merge_affinity),
                max_center_distance=float(args.feature_merge_max_center_distance),
                max_group_size=int(args.feature_merge_max_group_size),
            )
            repair_strategy = "semantic_layout_feature_merge"
        elif source == "hybrid_union_feature_merge":
            base_masks = _dedupe_masks(
                [
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "dinov2_maskcut",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "watershed",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _prepared_masks(stream, frame_ids, int(args.min_area)),
                ],
                overlap_iou=0.70,
            )
            source_frame_ids = sorted(base_masks)
            masks_by_frame, feature_maps_by_frame = _feature_merge_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                affinity_threshold=float(args.feature_merge_affinity),
                max_center_distance=float(args.feature_merge_max_center_distance),
                max_group_size=int(args.feature_merge_max_group_size),
            )
            repair_strategy = "hybrid_union_semantic_layout_feature_merge"
        elif source == "hybrid_union_feature_merge_material_split":
            base_masks = _dedupe_masks(
                [
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "dinov2_maskcut",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "watershed",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _prepared_masks(stream, frame_ids, int(args.min_area)),
                ],
                overlap_iou=0.70,
            )
            source_frame_ids = sorted(base_masks)
            merged_masks, feature_maps_by_frame = _feature_merge_masks(
                stream=stream,
                masks_by_frame=base_masks,
                frame_ids=source_frame_ids,
                adapter=adapter,
                min_area=int(args.min_area),
                affinity_threshold=float(args.feature_merge_affinity),
                max_center_distance=float(args.feature_merge_max_center_distance),
                max_group_size=int(args.feature_merge_max_group_size),
            )
            masks_by_frame, material_split_diag = _material_split_masks(
                masks_by_frame=merged_masks,
                d4rt_records=d4rt_records,
                min_area=int(args.min_area),
                max_splits=int(args.material_split_max_splits),
                min_tubes=int(args.material_split_min_tubes),
                min_cluster_distance_px=float(args.material_split_min_cluster_distance_px),
                max_mask_area_ratio=float(args.material_split_max_mask_area_ratio),
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            repair_strategy = "hybrid_union_feature_merge_d4rt_material_uv_split"
        elif source == "hybrid_union":
            masks_by_frame = _dedupe_masks(
                [
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "dinov2_maskcut",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _npz_source_masks(
                        external_root,
                        str(args.scene),
                        "watershed",
                        frame_ids,
                        int(args.min_area),
                        int(args.sample_frames),
                    ),
                    _prepared_masks(stream, frame_ids, int(args.min_area)),
                ],
                overlap_iou=0.70,
            )
            source_frame_ids = sorted(masks_by_frame)
        else:
            masks_by_frame = _npz_source_masks(
                external_root,
                str(args.scene),
                source,
                frame_ids,
                int(args.min_area),
                int(args.sample_frames),
            )
            source_frame_ids = sorted(masks_by_frame)
        tokens = _source_tokens(
            stream=stream,
            scene=str(args.scene),
            source=source,
            frame_ids=source_frame_ids,
            masks_by_frame=masks_by_frame,
            adapter=adapter,
            max_tokens=int(args.max_tokens),
            feature_maps_by_frame=feature_maps_by_frame,
            token_feature_mode=str(args.token_feature_mode),
        )
        edges = build_part_graph_edges(
            tokens,
            semantic_affinity_mode=str(args.semantic_affinity_mode),
            structure_topk=int(args.structure_topk),
            structure_min_affinity=float(args.structure_min_affinity),
            structure_decay=float(args.structure_decay),
            structure_temporal_window=int(args.structure_temporal_window),
            structure_temporal_rank_window=int(args.structure_temporal_rank_window),
        )
        token_by_id = {int(token.token_id): token for token in tokens}
        summary = summarize_part_graph(tokens, edges)
        summary.update(
            {
                "source": source,
                "scene": str(args.scene),
                "source_available": bool(tokens),
                "feature_backend": str(args.feature_backend),
                "feature_checkpoint": checkpoint or "",
                "radio_lang_model": args.radio_lang_model if args.feature_backend == "radio_radseg" else "",
                "radio_lang_align": bool(args.radio_lang_align) if args.feature_backend == "radio_radseg" else "",
                "sampled_frame_ids": source_frame_ids,
                "requested_frame_ids": frame_ids,
                "diagnostic_only": bool(args.diagnostic_only),
                "failure_stage": "" if tokens else "mask_source_or_tokens",
                "counts_as_current_v42_measurement": not bool(args.diagnostic_only),
                "repair_strategy": repair_strategy,
                "feature_split_max_splits": int(args.feature_split_max_splits) if "feature_cluster_split" in repair_strategy else "",
                "feature_split_spatial_weight": float(args.feature_split_spatial_weight) if "feature_cluster_split" in repair_strategy else "",
                "feature_merge_affinity": float(args.feature_merge_affinity) if "merge" in repair_strategy else "",
                "feature_merge_max_center_distance": float(args.feature_merge_max_center_distance) if "merge" in repair_strategy else "",
                "feature_merge_max_group_size": int(args.feature_merge_max_group_size) if "merge" in repair_strategy else "",
                "backfill_overlap_iou": float(args.backfill_overlap_iou) if "backfill" in repair_strategy else "",
                "backfill_max_masks_per_frame": int(args.backfill_max_masks_per_frame) if "backfill" in repair_strategy else "",
                "material_backfill_candidate_count": material_backfill_diag.get("candidate_count", ""),
                "material_backfill_selected_count": material_backfill_diag.get("selected_backfill_count", ""),
                "material_backfill_rejected_no_support_count": material_backfill_diag.get(
                    "rejected_no_material_support_count", ""
                ),
                "material_backfill_rejected_overlap_count": material_backfill_diag.get("rejected_overlap_count", ""),
                "material_backfill_selected_visible_tube_anchor_count": material_backfill_diag.get(
                    "selected_visible_tube_anchor_count", ""
                ),
                "material_backfill_min_tubes": int(args.material_backfill_min_tubes) if material_backfill_diag else "",
                "material_backfill_max_candidate_area_fraction": float(
                    args.material_backfill_max_candidate_area_fraction
                )
                if material_backfill_diag
                else "",
                "material_backfill_diag": material_backfill_diag,
                "material_split_input_mask_count": material_split_diag.get("input_mask_count", ""),
                "material_split_output_mask_count": material_split_diag.get("output_mask_count", ""),
                "material_split_split_mask_count": material_split_diag.get("split_mask_count", ""),
                "material_split_created_fragment_count": material_split_diag.get("created_fragment_count", ""),
                "material_split_total_visible_tube_anchors_inside_masks": material_split_diag.get(
                    "total_visible_tube_anchors_inside_masks", ""
                ),
                "material_split_max_splits": int(args.material_split_max_splits) if material_split_diag else "",
                "material_split_min_tubes": int(args.material_split_min_tubes) if material_split_diag else "",
                "material_split_min_cluster_distance_px": float(args.material_split_min_cluster_distance_px)
                if material_split_diag
                else "",
                "material_split_max_mask_area_ratio": float(args.material_split_max_mask_area_ratio)
                if material_split_diag
                else "",
                "material_split_diag": material_split_diag,
                "token_feature_mode": str(args.token_feature_mode),
                "semantic_affinity_mode": str(args.semantic_affinity_mode),
                "structure_topk": int(args.structure_topk) if str(args.semantic_affinity_mode) != "cosine" else "",
                "structure_min_affinity": float(args.structure_min_affinity)
                if str(args.semantic_affinity_mode) != "cosine"
                else "",
                "structure_decay": float(args.structure_decay) if str(args.semantic_affinity_mode) != "cosine" else "",
                "structure_temporal_window": int(args.structure_temporal_window)
                if str(args.semantic_affinity_mode) == "temporal_widest_structure"
                else "",
                "structure_temporal_rank_window": int(args.structure_temporal_rank_window)
                if str(args.semantic_affinity_mode) == "temporal_chain_structure"
                else "",
            }
        )
        summary.update(_coverage_metrics(tokens))
        if d4rt_records:
            label_maps = _label_maps_from_masks(masks_by_frame)
            measurements, measurement_diag = build_measurement_bank(
                d4rt_records,
                masks_by_frame=label_maps,
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            support_by_token = build_token_material_support(tokens, measurements)
            material_edges = build_material_part_graph_edges(
                edges,
                support_by_token,
                material_weight=float(args.material_weight),
                conflict_weight=float(args.material_conflict_weight),
                min_shared_tube_count=int(args.material_min_shared_tubes),
                material_support_shrinkage=float(args.material_support_shrinkage),
            )
            material_summary = summarize_material_part_graph(
                tokens,
                material_edges,
                support_by_token,
                semantic_false_merge_rate=summary.get("same_frame_same_class_false_merge_rate"),
                coverage_at_010=float(summary.get("coverage@0.10", 0.0)),
            )
            material_summary.update(
                {
                    "source": source,
                    "scene": str(args.scene),
                    "material_cache_root": str(args.material_cache_root),
                    "d4rt_record_count": int(len(d4rt_records)),
                    "measurement_count": int(measurement_diag.get("measurement_count", 0)),
                    "same_mask_pair_count": int(measurement_diag.get("same_mask_pair_count", 0)),
                    "num_same_frame_cannot_link_pairs": int(measurement_diag.get("num_same_frame_cannot_link_pairs", 0)),
                    "num_visible_outside_negative_pairs": int(measurement_diag.get("num_visible_outside_negative_pairs", 0)),
                    "material_min_shared_tubes": int(args.material_min_shared_tubes),
                    "material_support_shrinkage": float(args.material_support_shrinkage),
                    "d4rt_cache_diag": d4rt_diag,
                    "counts_as_current_v42_measurement": not bool(args.diagnostic_only),
                }
            )
            all_material_summaries[source] = material_summary
            for row in material_summary["variant_rows"]:
                material_rows.append(
                    {
                        "source": source,
                        "scene": str(args.scene),
                        "material_cache_root": str(args.material_cache_root),
                        "d4rt_record_count": int(len(d4rt_records)),
                        "measurement_count": int(measurement_diag.get("measurement_count", 0)),
                        "material_min_shared_tubes": int(args.material_min_shared_tubes),
                        "material_support_shrinkage": float(args.material_support_shrinkage),
                        **row,
                    }
                )
        summary["gate_pass_phase1"] = bool(
            not bool(args.diagnostic_only)
            and summary.get("semantic_affinity_AUC") is not None
            and float(summary.get("semantic_affinity_AUC")) >= 0.75
            and float(summary.get("coverage@0.10", 0.0)) >= 0.70
        )
        source_rows.append(summary)
        all_summaries[source] = summary
        for token in tokens:
            token_rows.append(
                {
                    "source": source,
                    "scene": str(args.scene),
                    "token_id": int(token.token_id),
                    "frame_id": int(token.frame_id),
                    "mask_id": int(token.mask_id),
                    "area": int(token.area),
                    "boundary_contrast": float(token.boundary_contrast),
                    "diagnostic_gt_instance": "" if token.diagnostic_gt_instance is None else int(token.diagnostic_gt_instance),
                    "diagnostic_gt_purity": "" if token.diagnostic_gt_purity is None else float(token.diagnostic_gt_purity),
                    "diagnostic_gt_iou": "" if token.diagnostic_gt_iou is None else float(token.diagnostic_gt_iou),
                }
            )
        for edge in edges:
            left = token_by_id.get(int(edge.token_i))
            right = token_by_id.get(int(edge.token_j))
            edge_rows.append(
                {
                    "source": source,
                    "scene": str(args.scene),
                    "token_i": int(edge.token_i),
                    "token_j": int(edge.token_j),
                    "frame_i": "" if left is None else int(left.frame_id),
                    "frame_j": "" if right is None else int(right.frame_id),
                    "mask_i": "" if left is None else int(left.mask_id),
                    "mask_j": "" if right is None else int(right.mask_id),
                    "area_i": "" if left is None else int(left.area),
                    "area_j": "" if right is None else int(right.area),
                    "semantic_affinity": float(edge.semantic_affinity),
                    "boundary_penalty": float(edge.boundary_penalty),
                    "spatial_distance": float(edge.spatial_distance),
                    "same_frame_cannot_link": bool(edge.same_frame_cannot_link),
                    "object_affinity": float(edge.object_affinity),
                    "diagnostic_same_gt": "" if edge.diagnostic_same_gt is None else bool(edge.diagnostic_same_gt),
                    "gt_i": ""
                    if left is None or left.diagnostic_gt_instance is None
                    else int(left.diagnostic_gt_instance),
                    "gt_j": ""
                    if right is None or right.diagnostic_gt_instance is None
                    else int(right.diagnostic_gt_instance),
                    "purity_i": ""
                    if left is None or left.diagnostic_gt_purity is None
                    else float(left.diagnostic_gt_purity),
                    "purity_j": ""
                    if right is None or right.diagnostic_gt_purity is None
                    else float(right.diagnostic_gt_purity),
                    "iou_i": "" if left is None or left.diagnostic_gt_iou is None else float(left.diagnostic_gt_iou),
                    "iou_j": "" if right is None or right.diagnostic_gt_iou is None else float(right.diagnostic_gt_iou),
                }
            )

    best = max(source_rows, key=lambda row: (row.get("gate_pass_phase1", False), row.get("semantic_affinity_AUC") or -1, row.get("coverage@0.10") or -1), default={})
    phase1_pass = any(bool(row.get("gate_pass_phase1")) for row in source_rows)
    summary = {
        "phase": "v42_semantic_part_audit",
        "scene": str(args.scene),
        "feature_backend": str(args.feature_backend),
        "feature_checkpoint": checkpoint or "",
        "radio_lang_model": args.radio_lang_model if args.feature_backend == "radio_radseg" else "",
        "radio_lang_align": bool(args.radio_lang_align) if args.feature_backend == "radio_radseg" else "",
        "token_feature_mode": str(args.token_feature_mode),
        "requested_frame_ids": frame_ids,
        "diagnostic_only": bool(args.diagnostic_only),
        "semantic_affinity_mode": str(args.semantic_affinity_mode),
        "structure_topk": int(args.structure_topk) if str(args.semantic_affinity_mode) != "cosine" else "",
        "structure_min_affinity": float(args.structure_min_affinity) if str(args.semantic_affinity_mode) != "cosine" else "",
        "structure_decay": float(args.structure_decay) if str(args.semantic_affinity_mode) != "cosine" else "",
        "structure_temporal_window": int(args.structure_temporal_window)
        if str(args.semantic_affinity_mode) == "temporal_widest_structure"
        else "",
        "structure_temporal_rank_window": int(args.structure_temporal_rank_window)
        if str(args.semantic_affinity_mode) == "temporal_chain_structure"
        else "",
        "phase1_gate_pass": bool(phase1_pass),
        "best_source": best.get("source"),
        "best_source_summary": best,
        "source_rows_csv": str(ROOT / args.output_root / "source_audit_rows.csv"),
        "part_token_rows_csv": str(ROOT / args.output_root / "part_token_rows.csv"),
        "part_edge_rows_csv": str(ROOT / args.output_root / "part_edge_rows.csv"),
        "material_graph_rows_csv": str(ROOT / args.output_root / "material_graph_rows.csv") if material_rows else "",
        "all_source_summaries": all_summaries,
        "all_material_summaries": all_material_summaries,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / args.output_root
    _write_csv(out / "source_audit_rows.csv", source_rows)
    _write_csv(out / "part_token_rows.csv", token_rows)
    _write_csv(out / "part_edge_rows.csv", edge_rows)
    if material_rows:
        _write_csv(out / "material_graph_rows.csv", material_rows)
    _write_json(out / "part_graph_summary.json", summary)
    print(json.dumps({"summary": str(out / "part_graph_summary.json"), "phase1_gate_pass": phase1_pass, "best_source": best.get("source")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
