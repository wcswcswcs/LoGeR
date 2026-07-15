#!/usr/bin/env python3
"""Run v108 Phase3 controlled SAM2 appearance-capsule diagnostics.

All numeric ranks are diagnostic-only. Quality conclusions must come from the
generated visual casebook and manual review notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
GROUNDED_SAM2_ROOT = REPO_ROOT / "Grounded-SAM-2"
for path in (GROUNDED_SAM2_ROOT, REPO_ROOT, STREAM3D_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Stream3D.stream4d_v108.appearance_capsule import (  # noqa: E402
    AppearanceDescriptor,
    bbox_xyxy,
    cosine_similarity,
    descriptor_memory_bytes,
    interior_core_mask,
    pool_feature_descriptor,
    rgb_shape_descriptor,
)
from Stream3D.stream4d_v108.artifacts import ArtifactWriter, sha256_file  # noqa: E402


DEFAULT_SUMMARIES = [
    "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/"
    "v106_stateful_sam2_rolling_scene_stream/summary.json",
    "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/"
    "v106_stateful_sam2_rolling_scene_stream/summary.json",
    "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_appearance_only_control90_20260714_0410/"
    "v107_phase8_g3_rolling_scheduler_smoke/summary.json",
]


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    case_name: str
    summary_path: Path
    scene_id: str
    frame_id: int
    chunk_frame_index: int
    object_id: int
    rgb_path: Path
    label_path: Path
    tag: str
    forced: bool


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def resolve_rgb_path(summary: dict[str, Any], row: dict[str, Any]) -> Path:
    if row.get("rgb_path"):
        return resolve_path(str(row["rgb_path"]))
    rgb_root = summary.get("rgb_root") or "Stream3D/data/scannet/processed"
    return resolve_path(rgb_root) / str(summary["scene_id"]) / "color" / f"{int(row['frame_id'])}.jpg"


def parse_forced_specs(values: list[str]) -> list[dict[str, Any]]:
    out = []
    for value in values:
        parts = value.split(":")
        if len(parts) < 3:
            raise ValueError(f"forced spec must be scene_id:frame_id:object_id[:tag], got {value!r}")
        out.append(
            {
                "scene_id": parts[0],
                "frame_id": int(parts[1]),
                "object_id": int(parts[2]),
                "tag": parts[3] if len(parts) > 3 else "forced",
            }
        )
    return out


def load_cases(summary_paths: list[Path]) -> list[dict[str, Any]]:
    cases = []
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        rows_by_frame = {int(row["frame_id"]): row for row in summary.get("records", [])}
        if not rows_by_frame:
            raise ValueError(f"summary has no records: {summary_path}")
        case_name = summary_path.parent.parent.name if summary_path.parent.name == "v106_stateful_sam2_rolling_scene_stream" else summary_path.parent.name
        cases.append(
            {
                "summary_path": summary_path,
                "summary": summary,
                "case_name": case_name,
                "rows": list(summary.get("records", [])),
                "rows_by_frame": rows_by_frame,
                "scene_id": str(summary["scene_id"]),
            }
        )
    return cases


def object_rows_for_case(case: dict[str, Any], *, min_area_px: int, max_area_ratio: float) -> dict[int, list[dict[str, Any]]]:
    rows_by_obj: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in case["rows"]:
        label_path = resolve_path(str(row["label_path"]))
        label = read_label(label_path)
        h, w = label.shape[:2]
        max_area = float(h * w) * float(max_area_ratio)
        ids, counts = np.unique(label, return_counts=True)
        count_by_id = {int(obj_id): int(count) for obj_id, count in zip(ids, counts, strict=True) if int(obj_id) > 0}
        for obj_id, area in count_by_id.items():
            if area < int(min_area_px) or area > max_area:
                continue
            mask = label == int(obj_id)
            x0, y0, x1, y1 = bbox_xyxy(mask)
            rows_by_obj[int(obj_id)].append(
                {
                    "frame_id": int(row["frame_id"]),
                    "chunk_frame_index": int(row.get("chunk_frame_index", 0)),
                    "object_id": int(obj_id),
                    "area_px": int(area),
                    "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "row": row,
                }
            )
    return rows_by_obj


def choose_indices(count: int, snapshots_per_object: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1 or snapshots_per_object <= 1:
        return [0]
    candidates = [0, 1 if count > 2 else 0, count // 2, count - 1]
    out = []
    for idx in candidates:
        idx = int(max(0, min(count - 1, idx)))
        if idx not in out:
            out.append(idx)
        if len(out) >= int(snapshots_per_object):
            break
    return out


def select_snapshots(
    cases: list[dict[str, Any]],
    *,
    forced: list[dict[str, Any]],
    min_area_px: int,
    max_area_ratio: float,
    max_objects_per_case: int,
    snapshots_per_object: int,
) -> tuple[list[SnapshotSpec], list[dict[str, Any]]]:
    selected: dict[str, SnapshotSpec] = {}
    missing_forced: list[dict[str, Any]] = []
    cases_by_scene = {case["scene_id"]: case for case in cases}

    for case in cases:
        rows_by_obj = object_rows_for_case(case, min_area_px=min_area_px, max_area_ratio=max_area_ratio)
        scored = []
        for obj_id, rows in rows_by_obj.items():
            if len(rows) < 2:
                continue
            rows_sorted = sorted(rows, key=lambda item: int(item["frame_id"]))
            span = int(rows_sorted[-1]["frame_id"]) - int(rows_sorted[0]["frame_id"])
            median_area = float(np.median([row["area_px"] for row in rows_sorted]))
            scored.append((len(rows_sorted), span, median_area, int(obj_id), rows_sorted))
        scored.sort(reverse=True)
        for _, _, _, obj_id, rows_sorted in scored[: int(max_objects_per_case)]:
            for idx in choose_indices(len(rows_sorted), int(snapshots_per_object)):
                row_info = rows_sorted[idx]
                row = row_info["row"]
                frame_id = int(row["frame_id"])
                key = f"{case['scene_id']}|{frame_id}|{obj_id}"
                selected[key] = SnapshotSpec(
                    snapshot_id=key.replace("|", "_"),
                    case_name=str(case["case_name"]),
                    summary_path=case["summary_path"],
                    scene_id=str(case["scene_id"]),
                    frame_id=frame_id,
                    chunk_frame_index=int(row.get("chunk_frame_index", 0)),
                    object_id=int(obj_id),
                    rgb_path=resolve_rgb_path(case["summary"], row),
                    label_path=resolve_path(str(row["label_path"])),
                    tag="auto_stable_object",
                    forced=False,
                )

    for item in forced:
        case = cases_by_scene.get(str(item["scene_id"]))
        if case is None:
            missing_forced.append({**item, "reason": "scene_not_loaded"})
            continue
        row = case["rows_by_frame"].get(int(item["frame_id"]))
        if row is None:
            missing_forced.append({**item, "reason": "frame_not_in_summary"})
            continue
        label_path = resolve_path(str(row["label_path"]))
        label = read_label(label_path)
        if not bool(np.any(label == int(item["object_id"]))):
            missing_forced.append({**item, "reason": "object_id_not_visible_in_label"})
            continue
        key = f"{case['scene_id']}|{int(item['frame_id'])}|{int(item['object_id'])}"
        selected[key] = SnapshotSpec(
            snapshot_id=key.replace("|", "_"),
            case_name=str(case["case_name"]),
            summary_path=case["summary_path"],
            scene_id=str(case["scene_id"]),
            frame_id=int(item["frame_id"]),
            chunk_frame_index=int(row.get("chunk_frame_index", 0)),
            object_id=int(item["object_id"]),
            rgb_path=resolve_rgb_path(case["summary"], row),
            label_path=label_path,
            tag=str(item.get("tag") or "forced"),
            forced=True,
        )

    return sorted(selected.values(), key=lambda spec: (spec.scene_id, spec.frame_id, spec.object_id)), missing_forced


def build_sam2_predictor(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_cfg = str(args.sam2_model_cfg)
    checkpoint = resolve_path(args.sam2_checkpoint)
    t0 = time.time()
    model = build_sam2(
        model_cfg,
        str(checkpoint),
        device=str(args.device),
        apply_postprocessing=not bool(args.disable_sam2_postprocessing),
    )
    predictor = SAM2ImagePredictor(model)
    payload = {
        "sam2_available": True,
        "sam2_import_file": getattr(sys.modules.get("sam2"), "__file__", ""),
        "sam2_model_cfg": model_cfg,
        "sam2_model_cfg_path": rel(GROUNDED_SAM2_ROOT / "sam2" / model_cfg),
        "sam2_checkpoint": rel(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "build_runtime_sec": float(time.time() - t0),
        "torch_version": str(torch.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    return predictor, payload


def autocast_context(dtype_name: str):
    import torch

    lowered = str(dtype_name).lower()
    if not torch.cuda.is_available() or lowered in {"float32", "fp32", "none"}:
        from contextlib import nullcontext

        return nullcontext()
    if lowered in {"bfloat16", "bf16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if lowered in {"float16", "fp16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"unsupported model dtype: {dtype_name}")


def descriptor_rows_without_vectors(descriptors: list[AppearanceDescriptor], snapshots: dict[str, SnapshotSpec]) -> list[dict[str, Any]]:
    rows = []
    for desc in descriptors:
        spec = snapshots[f"{desc.scene_id}_{desc.frame_id}_{desc.object_id}"]
        rows.append(
            {
                "snapshot_id": spec.snapshot_id,
                "case_name": spec.case_name,
                "scene_id": desc.scene_id,
                "frame_id": int(desc.frame_id),
                "chunk_frame_index": int(spec.chunk_frame_index),
                "object_id": int(desc.object_id),
                "tag": spec.tag,
                "forced": bool(spec.forced),
                "variant": desc.variant,
                "feature_source": desc.feature_source,
                "feature_dim": int(desc.feature_dim),
                "mask_area_px": int(desc.mask_area_px),
                "core_area_px": int(desc.core_area_px),
                "quality": float(desc.quality),
                "bbox_xyxy": json.dumps(list(desc.bbox_xyxy)),
                "metadata": json.dumps(desc.metadata, sort_keys=True),
            }
        )
    return rows


def build_descriptors(
    specs: list[SnapshotSpec],
    args: argparse.Namespace,
) -> tuple[list[AppearanceDescriptor], dict[str, Any], dict[str, dict[str, Any]]]:
    specs_by_frame: dict[tuple[str, int], list[SnapshotSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_frame[(spec.scene_id, int(spec.frame_id))].append(spec)

    predictor = None
    sam2_probe: dict[str, Any] = {
        "sam2_available": False,
        "sam2_disabled": bool(args.disable_sam2),
        "feature_shapes": {},
    }
    if not bool(args.disable_sam2):
        predictor, sam2_probe = build_sam2_predictor(args)

    descriptors: list[AppearanceDescriptor] = []
    snapshot_meta: dict[str, dict[str, Any]] = {}
    first_feature_shapes: dict[str, Any] = {}
    import torch

    for (_, _), frame_specs in sorted(specs_by_frame.items(), key=lambda item: (item[0][0], item[0][1])):
        rgb = read_rgb(frame_specs[0].rgb_path)
        label = read_label(frame_specs[0].label_path)
        image_embed = None
        if predictor is not None:
            t_frame = time.time()
            with autocast_context(str(args.model_dtype)):
                predictor.set_image(rgb)
            image_embed = predictor.get_image_embedding()
            if not first_feature_shapes:
                first_feature_shapes["image_embed"] = list(image_embed.shape)
                features = getattr(predictor, "_features", {}) or {}
                first_feature_shapes["high_res_feats"] = [list(feat.shape) for feat in features.get("high_res_feats", [])]
            sam2_probe.setdefault("frame_feature_runtime_sec", {})[
                f"{frame_specs[0].scene_id}:{frame_specs[0].frame_id}"
            ] = float(time.time() - t_frame)

        for spec in frame_specs:
            mask = label == int(spec.object_id)
            core, core_meta = interior_core_mask(mask, min_core_area_px=int(args.min_core_area_px))
            mask_area = int(np.count_nonzero(mask))
            core_area = int(np.count_nonzero(core))
            quality = float(core_area) / float(max(mask_area, 1))
            x0, y0, x1, y1 = bbox_xyxy(mask)
            snapshot_meta[spec.snapshot_id] = {
                "snapshot_id": spec.snapshot_id,
                "case_name": spec.case_name,
                "scene_id": spec.scene_id,
                "frame_id": int(spec.frame_id),
                "object_id": int(spec.object_id),
                "tag": spec.tag,
                "forced": bool(spec.forced),
                "rgb_path": rel(spec.rgb_path),
                "label_path": rel(spec.label_path),
                "mask_area_px": mask_area,
                "core_area_px": core_area,
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "core_meta": core_meta,
            }
            rgb_vec, rgb_meta = rgb_shape_descriptor(rgb, mask, core_mask=core)
            descriptors.append(
                AppearanceDescriptor(
                    scene_id=spec.scene_id,
                    frame_id=int(spec.frame_id),
                    object_id=int(spec.object_id),
                    variant="A0_rgb_shape_control",
                    feature_source="rgb_mean_std_hist_shape_eroded_core",
                    vector=rgb_vec,
                    mask_area_px=mask_area,
                    core_area_px=core_area,
                    bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                    quality=quality,
                    metadata=rgb_meta,
                )
            )
            if image_embed is not None:
                a1_vec, a1_meta = pool_feature_descriptor(image_embed, mask, core_mask=core, use_core=False)
                descriptors.append(
                    AppearanceDescriptor(
                        scene_id=spec.scene_id,
                        frame_id=int(spec.frame_id),
                        object_id=int(spec.object_id),
                        variant="A1_sam2_mean_feature",
                        feature_source="sam2_image_embed_full_mask_mean",
                        vector=a1_vec,
                        mask_area_px=mask_area,
                        core_area_px=core_area,
                        bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                        quality=quality,
                        metadata=a1_meta,
                    )
                )
                a2_vec, a2_meta = pool_feature_descriptor(image_embed, mask, core_mask=core, use_core=True)
                descriptors.append(
                    AppearanceDescriptor(
                        scene_id=spec.scene_id,
                        frame_id=int(spec.frame_id),
                        object_id=int(spec.object_id),
                        variant="A2_eroded_core_sam2_feature",
                        feature_source="sam2_image_embed_eroded_core_mean",
                        vector=a2_vec,
                        mask_area_px=mask_area,
                        core_area_px=core_area,
                        bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                        quality=quality,
                        metadata=a2_meta,
                    )
                )
        if predictor is not None:
            predictor.reset_predictor()
            if torch.cuda.is_available() and int(args.empty_cache_every_frame):
                torch.cuda.empty_cache()

    sam2_probe["feature_shapes"] = first_feature_shapes
    if torch.cuda.is_available():
        sam2_probe["peak_cuda_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    return descriptors, sam2_probe, snapshot_meta


def desc_key(desc: AppearanceDescriptor) -> str:
    return f"{desc.scene_id}_{desc.frame_id}_{desc.object_id}"


def build_single_snapshot_ranks(
    descriptors: list[AppearanceDescriptor],
    *,
    variant: str,
    snapshot_meta: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    by_scene = defaultdict(list)
    for desc in descriptors:
        if desc.variant == variant:
            by_scene[desc.scene_id].append(desc)
    for scene_id, scene_descs in by_scene.items():
        scene_descs = sorted(scene_descs, key=lambda d: (d.frame_id, d.object_id))
        for query in scene_descs:
            candidates = [cand for cand in scene_descs if cand.frame_id < query.frame_id]
            scored = []
            for cand in candidates:
                score = cosine_similarity(query.vector, cand.vector)
                scored.append((score, cand))
            scored.sort(key=lambda item: item[0], reverse=True)
            for rank, (score, cand) in enumerate(scored[: int(top_k)], start=1):
                qid = desc_key(query)
                cid = desc_key(cand)
                rows.append(
                    {
                        "schema_version": "stream4d_v108_phase3_appearance_rank_row_v1",
                        "variant": variant,
                        "history_mode": "single_snapshot",
                        "scene_id": scene_id,
                        "query_snapshot_id": qid,
                        "query_frame_id": int(query.frame_id),
                        "query_object_id": int(query.object_id),
                        "query_tag": snapshot_meta[qid]["tag"],
                        "candidate_snapshot_id": cid,
                        "candidate_frame_id": int(cand.frame_id),
                        "candidate_object_id": int(cand.object_id),
                        "candidate_identity_object_id": int(cand.object_id),
                        "best_proto_snapshot_id": cid,
                        "rank": int(rank),
                        "score": float(score),
                        "adjusted_score": float(score),
                        "same_identity_diagnostic": bool(query.object_id == cand.object_id),
                        "cannot_link_conflict": False,
                        "prototype_count": 1,
                        "metric_role": "diagnostic_only_selects_visual_cases",
                        "may_set_acceptance": False,
                    }
                )
    return rows


def select_viewset_prototypes(history: list[AppearanceDescriptor], *, max_prototypes: int, min_quality: float) -> list[AppearanceDescriptor]:
    candidates = [desc for desc in history if float(desc.quality) >= float(min_quality)]
    candidates.sort(key=lambda desc: (float(desc.quality), int(desc.mask_area_px)), reverse=True)
    selected: list[AppearanceDescriptor] = []
    for desc in candidates:
        if len(selected) >= int(max_prototypes):
            break
        if not selected:
            selected.append(desc)
            continue
        max_sim = max(cosine_similarity(desc.vector, prev.vector) for prev in selected)
        if max_sim < 0.985:
            selected.append(desc)
    if not selected and history:
        selected.append(max(history, key=lambda desc: (float(desc.quality), int(desc.mask_area_px))))
    return selected[: int(max_prototypes)]


def bbox_fill_ratio(desc: AppearanceDescriptor) -> float:
    x0, y0, x1, y1 = desc.bbox_xyxy
    bbox_area = max(1, int(x1 - x0 + 1) * int(y1 - y0 + 1))
    return float(desc.mask_area_px) / float(bbox_area)


def select_quality_guarded_prototypes(
    history: list[AppearanceDescriptor],
    *,
    max_prototypes: int,
    min_quality: float,
    min_bbox_fill: float,
) -> list[AppearanceDescriptor]:
    guarded = [
        desc
        for desc in history
        if float(desc.quality) >= float(min_quality) and bbox_fill_ratio(desc) >= float(min_bbox_fill)
    ]
    return select_viewset_prototypes(guarded, max_prototypes=max_prototypes, min_quality=min_quality)


def build_viewset_ranks(
    descriptors: list[AppearanceDescriptor],
    *,
    source_variant: str,
    output_variant: str,
    snapshot_meta: dict[str, dict[str, Any]],
    top_k: int,
    max_prototypes: int,
    min_quality: float,
    use_cannot_link: bool,
    min_bbox_fill: float | None = None,
    identity_confirmation_allowed: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    by_scene = defaultdict(list)
    for desc in descriptors:
        if desc.variant == source_variant:
            by_scene[desc.scene_id].append(desc)
    for scene_id, scene_descs in by_scene.items():
        scene_descs = sorted(scene_descs, key=lambda d: (d.frame_id, d.object_id))
        present_by_frame: dict[int, set[int]] = defaultdict(set)
        for desc in scene_descs:
            present_by_frame[int(desc.frame_id)].add(int(desc.object_id))
        for query in scene_descs:
            histories_by_obj: dict[int, list[AppearanceDescriptor]] = defaultdict(list)
            for cand in scene_descs:
                if cand.frame_id < query.frame_id:
                    histories_by_obj[int(cand.object_id)].append(cand)
            scored = []
            for candidate_obj, history in histories_by_obj.items():
                if min_bbox_fill is None:
                    prototypes = select_viewset_prototypes(
                        history,
                        max_prototypes=int(max_prototypes),
                        min_quality=float(min_quality),
                    )
                else:
                    prototypes = select_quality_guarded_prototypes(
                        history,
                        max_prototypes=int(max_prototypes),
                        min_quality=float(min_quality),
                        min_bbox_fill=float(min_bbox_fill),
                    )
                if not prototypes:
                    continue
                proto_scores = [(cosine_similarity(query.vector, proto.vector), proto) for proto in prototypes]
                raw_score, best_proto = max(proto_scores, key=lambda item: item[0])
                cannot_link = bool(
                    use_cannot_link
                    and int(candidate_obj) != int(query.object_id)
                    and int(candidate_obj) in present_by_frame.get(int(query.frame_id), set())
                )
                adjusted = -2.0 if cannot_link else float(raw_score)
                scored.append((adjusted, raw_score, best_proto, candidate_obj, prototypes, cannot_link))
            scored.sort(key=lambda item: item[0], reverse=True)
            for rank, (adjusted, raw_score, best_proto, candidate_obj, prototypes, cannot_link) in enumerate(
                scored[: int(top_k)], start=1
            ):
                qid = desc_key(query)
                cid = desc_key(best_proto)
                rows.append(
                    {
                        "schema_version": "stream4d_v108_phase3_appearance_rank_row_v1",
                        "variant": output_variant,
                        "history_mode": "multi_view_viewset",
                        "scene_id": scene_id,
                        "query_snapshot_id": qid,
                        "query_frame_id": int(query.frame_id),
                        "query_object_id": int(query.object_id),
                        "query_tag": snapshot_meta[qid]["tag"],
                        "candidate_snapshot_id": cid,
                        "candidate_frame_id": int(best_proto.frame_id),
                        "candidate_object_id": int(best_proto.object_id),
                        "candidate_identity_object_id": int(candidate_obj),
                        "best_proto_snapshot_id": cid,
                        "rank": int(rank),
                        "score": float(raw_score),
                        "adjusted_score": float(adjusted),
                        "same_identity_diagnostic": bool(int(query.object_id) == int(candidate_obj)),
                        "cannot_link_conflict": bool(cannot_link),
                        "prototype_count": int(len(prototypes)),
                        "best_proto_bbox_fill_ratio": float(bbox_fill_ratio(best_proto)),
                        "prototype_min_bbox_fill_guard": None if min_bbox_fill is None else float(min_bbox_fill),
                        "appearance_use_recommendation": (
                            "IDENTITY_CONFIRMATION_REQUIRES_GEOMETRY_OR_WATCHER"
                            if identity_confirmation_allowed
                            else "ROI_RETRIEVAL_ONLY_NOT_IDENTITY_CONFIRMATION"
                        ),
                        "identity_confirmation_allowed_by_appearance_alone": bool(identity_confirmation_allowed),
                        "metric_role": "diagnostic_only_selects_visual_cases",
                        "may_set_acceptance": False,
                    }
                )
    return rows


def rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out = {}
    for variant, items in sorted(by_variant.items()):
        rank1 = [row for row in items if int(row["rank"]) == 1]
        forced_rank1 = [row for row in rank1 if "event00" in str(row.get("query_tag", ""))]
        out[variant] = {
            "rank_row_count": int(len(items)),
            "query_count_with_rank1": int(len(rank1)),
            "top1_same_identity_rate_diagnostic_only": (
                float(np.mean([bool(row["same_identity_diagnostic"]) for row in rank1])) if rank1 else None
            ),
            "forced_event_query_count_with_rank1": int(len(forced_rank1)),
            "forced_event_top1_same_identity_rate_diagnostic_only": (
                float(np.mean([bool(row["same_identity_diagnostic"]) for row in forced_rank1])) if forced_rank1 else None
            ),
            "diagnostic_only": True,
            "may_set_acceptance": False,
        }
    return out


def overlay_crop(rgb: np.ndarray, mask: np.ndarray, core: np.ndarray, title: str, lines: list[str]) -> Image.Image:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy(mask)
    pad = 36
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    crop_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    crop_core = core[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    color = np.array([230, 57, 70], dtype=np.uint8)
    core_color = np.array([255, 209, 102], dtype=np.uint8)
    overlay = crop_rgb.copy()
    overlay[crop_mask] = (0.48 * overlay[crop_mask].astype(np.float32) + 0.52 * color.astype(np.float32)).astype(np.uint8)
    overlay[crop_core] = (0.25 * overlay[crop_core].astype(np.float32) + 0.75 * core_color.astype(np.float32)).astype(np.uint8)
    edge = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    overlay[edge] = np.array([255, 255, 255], dtype=np.uint8)

    image = Image.fromarray(overlay)
    target_w = 620
    if image.width < target_w:
        scale = float(target_w) / float(max(image.width, 1))
        image = image.resize((target_w, int(round(image.height * scale))), Image.Resampling.NEAREST)
    elif image.width > target_w:
        scale = float(target_w) / float(max(image.width, 1))
        image = image.resize((target_w, int(round(image.height * scale))), Image.Resampling.LANCZOS)

    header_h = 84
    panel = Image.new("RGB", (image.width, image.height + header_h), (18, 20, 24))
    panel.paste(image, (0, header_h))
    draw = ImageDraw.Draw(panel)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_b = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((8, 6), title[:72], fill=(245, 245, 245), font=font_b)
    y = 31
    for line in lines[:3]:
        draw.text((8, y), line[:98], fill=(220, 224, 232), font=font)
        y += 16
    return panel


def make_casebook_panel(row: dict[str, Any], snapshots: dict[str, SnapshotSpec], out_path: Path) -> None:
    query_spec = snapshots[str(row["query_snapshot_id"])]
    cand_spec = snapshots[str(row["best_proto_snapshot_id"])]
    panels = []
    for role, spec in (("history candidate", cand_spec), ("query current", query_spec)):
        rgb = read_rgb(spec.rgb_path)
        label = read_label(spec.label_path)
        mask = label == int(spec.object_id)
        core, _ = interior_core_mask(mask)
        title = f"{role}: {spec.scene_id} f{spec.frame_id} obj{spec.object_id}"
        lines = [
            f"tag={spec.tag} forced={spec.forced}",
            f"mask px={int(np.count_nonzero(mask))} core px={int(np.count_nonzero(core))}",
            f"path={rel(spec.label_path)}",
        ]
        panels.append(overlay_crop(rgb, mask, core, title, lines))

    gap = 12
    header_h = 58
    width = panels[0].width + panels[1].width + gap
    height = max(panel.height for panel in panels) + header_h
    canvas = Image.new("RGB", (width, height), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_b = ImageFont.load_default()
        font = ImageFont.load_default()
    title = (
        f"{row['variant']} rank={row['rank']} score={float(row['score']):.4f} "
        f"same_id_diagnostic={row['same_identity_diagnostic']}"
    )
    draw.text((8, 6), title, fill=(245, 245, 245), font=font_b)
    draw.text(
        (8, 32),
        "Numeric score is diagnostic-only; judge the pair by visual inspection of the masks and cores.",
        fill=(218, 222, 230),
        font=font,
    )
    canvas.paste(panels[0], (0, header_h))
    canvas.paste(panels[1], (panels[0].width + gap, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def choose_casebook_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rank1 = [row for row in rows if int(row["rank"]) == 1]
    preferred_variants = [
        "A2_eroded_core_sam2_feature",
        "A3_multi_view_viewset",
        "A4_viewset_plus_cannot_link",
        "A4_repair_quality_guard_roi_only",
        "A1_sam2_mean_feature",
        "A0_rgb_shape_control",
    ]
    selected = []
    seen = set()
    per_variant_target = max(1, int(limit) // max(1, len(preferred_variants)))
    for variant in preferred_variants:
        variant_rows = [row for row in rank1 if str(row["variant"]) == variant]
        variant_added = 0
        buckets = (
            [row for row in variant_rows if "event00" in str(row.get("query_tag", ""))],
            [row for row in variant_rows if not bool(row["same_identity_diagnostic"])],
            [row for row in variant_rows if bool(row["same_identity_diagnostic"])],
        )
        for bucket in buckets:
            for row in bucket:
                key = (row["variant"], row["query_snapshot_id"])
                if key in seen:
                    continue
                selected.append(row)
                seen.add(key)
                variant_added += 1
                if len(selected) >= int(limit):
                    return selected
                if variant_added >= per_variant_target:
                    break
            if variant_added >= per_variant_target:
                break
    if len(selected) < int(limit):
        for variant in preferred_variants:
            for row in [item for item in rank1 if str(item["variant"]) == variant]:
                key = (row["variant"], row["query_snapshot_id"])
                if key in seen:
                    continue
                selected.append(row)
                seen.add(key)
                if len(selected) >= int(limit):
                    return selected
    return selected[: int(limit)]


def write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return True


def save_vectors(path: Path, descriptors: list[AppearanceDescriptor]) -> None:
    payload = {}
    for desc in descriptors:
        key = f"{desc.variant}__{desc.scene_id}_{desc.frame_id}_{desc.object_id}".replace(".", "_")
        payload[key] = desc.vector.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], help="Summary JSON with records/labels; repeatable.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--disable-sam2", action="store_true", default=False)
    parser.add_argument("--disable-sam2-postprocessing", action="store_true", default=False)
    parser.add_argument("--min-area-px", type=int, default=2048)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--min-core-area-px", type=int, default=24)
    parser.add_argument("--max-objects-per-case", type=int, default=5)
    parser.add_argument("--snapshots-per-object", type=int, default=3)
    parser.add_argument("--viewset-max-prototypes", type=int, default=4)
    parser.add_argument("--viewset-min-quality", type=float, default=0.002)
    parser.add_argument("--prototype-min-bbox-fill", type=float, default=0.18)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--casebook-limit", type=int, default=12)
    parser.add_argument("--forced-object-frame", action="append", default=[])
    parser.add_argument("--empty-cache-every-frame", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    started = time.time()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(output_root)

    summary_values = args.summary or DEFAULT_SUMMARIES
    summary_paths = [resolve_path(value) for value in summary_values]
    forced = parse_forced_specs(args.forced_object_frame)
    cases = load_cases(summary_paths)
    specs, missing_forced = select_snapshots(
        cases,
        forced=forced,
        min_area_px=int(args.min_area_px),
        max_area_ratio=float(args.max_area_ratio),
        max_objects_per_case=int(args.max_objects_per_case),
        snapshots_per_object=int(args.snapshots_per_object),
    )
    snapshots = {spec.snapshot_id: spec for spec in specs}

    descriptors, sam2_probe, snapshot_meta = build_descriptors(specs, args)
    desc_rows = descriptor_rows_without_vectors(descriptors, snapshots)
    writer.write_csv("appearance_descriptor_rows.csv", desc_rows, "stream4d_v108_phase3_descriptor_rows_v1")
    writer.write_jsonl("appearance_descriptor_rows.jsonl", desc_rows, "stream4d_v108_phase3_descriptor_rows_v1")
    vec_path = output_root / "appearance_descriptor_vectors_fp16.npz"
    save_vectors(vec_path, descriptors)
    writer.record_existing("appearance_descriptor_vectors_fp16.npz", "stream4d_v108_phase3_descriptor_vectors_npz_v1")

    rank_rows: list[dict[str, Any]] = []
    for variant in ("A0_rgb_shape_control", "A1_sam2_mean_feature", "A2_eroded_core_sam2_feature"):
        rank_rows.extend(
            build_single_snapshot_ranks(
                descriptors,
                variant=variant,
                snapshot_meta=snapshot_meta,
                top_k=int(args.top_k),
            )
        )
    rank_rows.extend(
        build_viewset_ranks(
            descriptors,
            source_variant="A2_eroded_core_sam2_feature",
            output_variant="A3_multi_view_viewset",
            snapshot_meta=snapshot_meta,
            top_k=int(args.top_k),
            max_prototypes=int(args.viewset_max_prototypes),
            min_quality=float(args.viewset_min_quality),
            use_cannot_link=False,
            identity_confirmation_allowed=False,
        )
    )
    rank_rows.extend(
        build_viewset_ranks(
            descriptors,
            source_variant="A2_eroded_core_sam2_feature",
            output_variant="A4_viewset_plus_cannot_link",
            snapshot_meta=snapshot_meta,
            top_k=int(args.top_k),
            max_prototypes=int(args.viewset_max_prototypes),
            min_quality=float(args.viewset_min_quality),
            use_cannot_link=True,
            identity_confirmation_allowed=False,
        )
    )
    rank_rows.extend(
        build_viewset_ranks(
            descriptors,
            source_variant="A2_eroded_core_sam2_feature",
            output_variant="A4_repair_quality_guard_roi_only",
            snapshot_meta=snapshot_meta,
            top_k=int(args.top_k),
            max_prototypes=int(args.viewset_max_prototypes),
            min_quality=float(args.viewset_min_quality),
            use_cannot_link=True,
            min_bbox_fill=float(args.prototype_min_bbox_fill),
            identity_confirmation_allowed=False,
        )
    )
    writer.write_csv("appearance_candidate_rank_rows.csv", rank_rows, "stream4d_v108_phase3_candidate_rank_rows_v1")
    writer.write_jsonl("appearance_candidate_rank_rows.jsonl", rank_rows, "stream4d_v108_phase3_candidate_rank_rows_v1")
    if write_parquet_if_available(output_root / "appearance_candidate_rank_rows.parquet", rank_rows):
        writer.record_existing("appearance_candidate_rank_rows.parquet", "stream4d_v108_phase3_candidate_rank_rows_v1")

    casebook_rows = choose_casebook_rows(rank_rows, limit=int(args.casebook_limit))
    visual_rows = []
    for idx, row in enumerate(casebook_rows):
        safe_variant = str(row["variant"]).replace("/", "_")
        out_rel = f"topk_casebook/case_{idx:02d}_{safe_variant}_{row['query_snapshot_id']}.jpg"
        out_path = output_root / out_rel
        make_casebook_panel(row, snapshots, out_path)
        visual_rows.append(
            {
                **row,
                "visual_path": rel(out_path),
                "visual_sha256": sha256_file(out_path),
                "manual_visual_status": "USER_REVIEW_PENDING",
                "quality_judgment_from_metrics": "FORBIDDEN",
            }
        )
    writer.write_json("topk_casebook_rows.json", visual_rows, "stream4d_v108_phase3_visual_casebook_rows_v1")
    for row in visual_rows:
        writer.record_existing(
            str(Path(row["visual_path"]).relative_to(rel(output_root))),
            "stream4d_v108_phase3_visual_casebook_image_v1",
        )

    descriptor_memory = {
        "descriptor_count": int(len(descriptors)),
        "snapshot_count": int(len(specs)),
        "fp16_vector_memory_bytes": int(descriptor_memory_bytes(descriptors, dtype_bytes=2)),
        "fp32_vector_memory_bytes": int(descriptor_memory_bytes(descriptors, dtype_bytes=4)),
        "variant_descriptor_counts": {
            variant: int(sum(1 for desc in descriptors if desc.variant == variant))
            for variant in sorted({desc.variant for desc in descriptors})
        },
        "runtime_sec": float(time.time() - started),
        "diagnostic_only": True,
        "may_set_acceptance": False,
    }
    writer.write_json(
        "descriptor_memory_runtime.json",
        descriptor_memory,
        "stream4d_v108_phase3_descriptor_memory_runtime_v1",
    )
    writer.write_json("sam2_feature_probe.json", sam2_probe, "stream4d_v108_phase3_sam2_feature_probe_v1")
    writer.write_json(
        "snapshot_selection.json",
        {
            "schema_version": "stream4d_v108_phase3_snapshot_selection_v1",
            "summary_paths": [rel(path) for path in summary_paths],
            "snapshot_count": int(len(specs)),
            "snapshots": [snapshot_meta[spec.snapshot_id] for spec in specs],
            "missing_forced": missing_forced,
            "selection_config": {
                "min_area_px": int(args.min_area_px),
                "max_area_ratio": float(args.max_area_ratio),
                "max_objects_per_case": int(args.max_objects_per_case),
                "snapshots_per_object": int(args.snapshots_per_object),
            },
        },
        "stream4d_v108_phase3_snapshot_selection_v1",
    )

    summary = {
        "schema_version": "stream4d_v108_phase3_appearance_benchmark_summary_v1",
        "status": "CONTROLLED_DIAGNOSTIC_COMPLETE_VISUAL_REVIEW_PENDING",
        "phase": "Phase3 SAM2 appearance capsule",
        "plan_variants": [
            "A0_rgb_shape_control",
            "A1_sam2_mean_feature",
            "A2_eroded_core_sam2_feature",
            "A3_multi_view_viewset",
            "A4_viewset_plus_cannot_link",
            "A4_repair_quality_guard_roi_only",
        ],
        "repair_variants": {
            "A4_repair_quality_guard_roi_only": {
                "prototype_min_bbox_fill": float(args.prototype_min_bbox_fill),
                "appearance_use_recommendation": "ROI_RETRIEVAL_ONLY_NOT_IDENTITY_CONFIRMATION",
                "reason": "failure repair for visually mixed historical prototypes; no margin sweep",
            }
        },
        "metrics_are_diagnostic_only": True,
        "quality_judgment_requires_visual_confirmation": True,
        "no_output_mask_mutation": True,
        "no_sam2_memory_mutation": True,
        "summary_paths": [rel(path) for path in summary_paths],
        "snapshot_count": int(len(specs)),
        "descriptor_count": int(len(descriptors)),
        "rank_row_count": int(len(rank_rows)),
        "visual_casebook_count": int(len(visual_rows)),
        "visual_casebook_rows_path": rel(output_root / "topk_casebook_rows.json"),
        "rank_summary_diagnostic_only": rank_summary(rank_rows),
        "missing_forced": missing_forced,
        "sam2_probe": sam2_probe,
        "descriptor_memory_runtime": descriptor_memory,
        "runtime_sec": float(time.time() - started),
        "acceptance_boundary": (
            "Do not judge Phase3 by rank metrics. Inspect topk_casebook images and record manual visual notes."
        ),
    }
    writer.write_json("phase3_appearance_summary.json", summary, "stream4d_v108_phase3_appearance_summary_v1")
    writer.write_json("artifact_manifest.json", writer.manifest(), "stream4d_v108_artifact_manifest_v1")
    print(json.dumps({"summary": rel(output_root / "phase3_appearance_summary.json"), **summary}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
