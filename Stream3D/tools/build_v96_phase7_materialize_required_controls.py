#!/usr/bin/env python3
"""Materialize required v96 Phase7 controls as Phase5-style roots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase7_materialize_required_controls"
RUN_ID = "v96_phase7_materialize_required_controls"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SEMANTIC_ROWS = ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _source_selected_rows(phase5_root: Path, family: str) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(phase5_root / "selected_masklet_rows.csv") if row.get("family") == family]
    if not rows:
        raise ValueError(f"no selected_masklet_rows for family={family} in {phase5_root}")
    return rows


def _source_object_rows(phase5_root: Path, family: str) -> dict[str, dict[str, str]]:
    path = phase5_root / "object_candidate_rows.csv"
    if not path.exists():
        return {}
    return {row.get("object_id", ""): row for row in _read_csv(path) if row.get("family") == family}


def _source_mask_lookup(source_rows: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    out: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in _read_csv(source_rows):
        key = (
            row.get("scene_id", ""),
            row.get("window_id", ""),
            int(_num(row.get("frame_id"))),
            int(_num(row.get("source_mask_id"))),
        )
        if key not in out:
            out[key] = row
    return out


def _frame_mask_lookup(source_rows: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    out: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        if key not in out and row.get("mask_path"):
            out[key] = row
    return out


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _semantic_lookup(path: Path, wanted: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.exists() or not wanted:
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("scene_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("mask_id"))))
            if key in wanted and key not in out:
                out[key] = row
    return out


def _area_bin(area_ratio: float) -> str:
    thresholds = [
        (0.002, "lt0002"),
        (0.005, "lt0005"),
        (0.010, "lt0010"),
        (0.020, "lt0020"),
        (0.050, "lt0050"),
        (0.100, "lt0100"),
    ]
    for threshold, name in thresholds:
        if area_ratio < threshold:
            return name
    return "ge0100"


def _semantic_score(feature: dict[str, str]) -> float:
    margin = max(0.0, _num(feature.get("semantic_prototype_margin")))
    entropy = min(1.0, max(0.0, _num(feature.get("semantic_entropy"))))
    broad_weight = 0.5 if _bool(feature.get("broad_background_risk")) else 1.0
    return float(margin * (1.0 - entropy) * broad_weight)


def _area_score(area_ratio: float, broad_risk: bool, rank_in_frame_bin: int, target_area_ratio: float) -> float:
    ratio = max(1e-6, area_ratio) / max(1e-6, target_area_ratio)
    specificity = 1.0 / (1.0 + abs(math.log(ratio, 2.0)))
    risk_weight = 0.45 if broad_risk else 1.0
    rank_weight = 1.0 / math.sqrt(max(1, rank_in_frame_bin))
    return float(specificity * risk_weight * rank_weight)


def _finalize_grouped_rows(
    grouped: dict[str, list[dict[str, Any]]],
    output_family: str,
    *,
    phase5_root: Path,
    output_root: Path,
    summary_extra: dict[str, Any],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for idx, (group_id, rows) in enumerate(sorted(grouped.items()), start=1):
        object_id = f"{output_family}_obj_{idx:06d}"
        support_sum = sum(int(_num(row.get("masklet_support_query_count"))) for row in rows)
        q_sum = sum(int(_num(row.get("object_query_count"))) for row in rows)
        score_vals = [float(_num(row.get("masklet_score"))) for row in rows]
        for row in rows:
            selected_rows.append(
                {
                    **row,
                    "schema_version": "stream4d_v96_selected_masklet_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": output_family,
                    "object_id": object_id,
                    "group_id": group_id,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": output_family,
                "object_id": object_id,
                "group_id": group_id,
                "micro_query_count": q_sum,
                "selected_frame_count_before_collision_resolution": len(rows),
                "masklet_support_query_count_sum": support_sum,
                "object_score": float(sum(score_vals) / max(1, len(score_vals))),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "selected_frame_count": len({(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id")))) for row in rows}),
            }
        )
    summary = {
        "schema": "stream4d_v96_required_control_materialization_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": f"MATERIALIZED_V96_{output_family.upper()}_CONTROL",
        "source_phase5_root": _rel(phase5_root),
        "output_root": _rel(output_root),
        "output_family": output_family,
        "selected_masklet_count": len(selected_rows),
        "object_count": len(object_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        **summary_extra,
    }
    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_rows)
    _write_csv(output_root / "control_materialization_rows.csv", control_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "object_count": len(object_rows), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def _resolve_frame_mask_collisions(
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    keyed: dict[tuple[str, str, int, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for group_id, rows in grouped.items():
        for row in rows:
            key = (
                str(row.get("scene_id", "")),
                str(row.get("window_id", "")),
                int(_num(row.get("frame_id"))),
                int(_num(row.get("selected_mask_id"))),
            )
            keyed[key].append((group_id, row))
    resolved: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_count = 0
    dropped_count = 0
    for vals in keyed.values():
        duplicate_count += max(0, len(vals) - 1)
        group_id, row = max(
            vals,
            key=lambda item: (
                int(_num(item[1].get("masklet_support_query_count"))),
                float(_num(item[1].get("masklet_score"))),
                str(item[0]),
            ),
        )
        out = dict(row)
        out["selection_status"] = f"{out.get('selection_status','control_selected')}_after_collision_resolution"
        resolved[group_id].append(out)
        dropped_count += len(vals) - 1
    return resolved, duplicate_count, dropped_count


def _materialize_c0(args: argparse.Namespace, selected: list[dict[str, str]], phase5_root: Path, output_root: Path) -> dict[str, Any]:
    wanted = {
        (row.get("scene_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id"))))
        for row in selected
    }
    semantic_rows = _semantic_lookup(_project(args.semantic_feature_rows), wanted)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    control_rows: list[dict[str, Any]] = []
    missing = 0
    unavailable = 0
    for row in selected:
        key = (row.get("scene_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id"))))
        feature = semantic_rows.get(key)
        if feature and _bool(feature.get("feature_available")):
            proto = feature.get("semantic_prototype_id") or "prototype_missing"
            broad = _bool(feature.get("broad_background_risk"))
            score = _semantic_score(feature)
            group_id = f"C0_semantic_only:{row.get('scene_id','')}:{row.get('window_id','')}:proto:{proto}"
            status = "semantic_prototype_grouped"
        else:
            missing += int(feature is None)
            unavailable += int(feature is not None)
            proto = "missing_or_unavailable"
            broad = False
            score = 0.0
            group_id = (
                f"C0_semantic_only_missing:{row.get('scene_id','')}:{row.get('window_id','')}:"
                f"f{row.get('frame_id','')}:m{row.get('selected_mask_id','')}"
            )
            status = "semantic_feature_missing_identity_fallback"
        out = {
            **row,
            "masklet_score": score,
            "selection_status": status,
            "semantic_prototype_id": proto,
            "semantic_broad_background_risk": broad,
        }
        grouped[group_id].append(out)
        control_rows.append(
            {
                "control": "C0_semantic_only",
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "frame_id": row.get("frame_id", ""),
                "selected_mask_id": row.get("selected_mask_id", ""),
                "group_id": group_id,
                "semantic_prototype_id": proto,
                "semantic_score": score,
                "feature_available": bool(feature and _bool(feature.get("feature_available"))),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return _finalize_grouped_rows(
        grouped,
        "C0_semantic_only",
        phase5_root=phase5_root,
        output_root=output_root,
        summary_extra={
            "control_definition": "Masks are grouped only by DINO semantic prototype id across frames; D4RT temporal micro-track identity is not used.",
            "semantic_feature_rows": _rel(_project(args.semantic_feature_rows)),
            "semantic_feature_backend": "dinov2_timm",
            "semantic_lookup_key_count": len(wanted),
            "semantic_matched_key_count": len(semantic_rows),
            "semantic_missing_row_count": missing,
            "semantic_unavailable_row_count": unavailable,
        },
        control_rows=control_rows,
    )


def _materialize_c1(args: argparse.Namespace, selected: list[dict[str, str]], phase5_root: Path, output_root: Path) -> dict[str, Any]:
    source_rows_path = _project(args.source_rows)
    source_lookup = _source_mask_lookup(source_rows_path)
    frame_lookup = _frame_mask_lookup(source_rows_path)
    label_cache: dict[tuple[str, str, int], np.ndarray] = {}
    by_frame: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_frame[(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))].append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    control_rows: list[dict[str, Any]] = []
    missing_area = 0
    raster_area_fallback = 0
    for frame_key, rows in sorted(by_frame.items()):
        bucketed: dict[tuple[str, int], list[tuple[float, dict[str, str], dict[str, str]]]] = defaultdict(list)
        for row in rows:
            source = source_lookup.get((row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id")))), {})
            area_ratio = _num(source.get("mask_area_ratio"), -1.0)
            if area_ratio < 0.0:
                frame_source = frame_lookup.get(frame_key, {})
                mask_path_raw = frame_source.get("mask_path", "")
                if mask_path_raw:
                    if frame_key not in label_cache:
                        label_cache[frame_key] = _load_label(_project(mask_path_raw))
                    label = label_cache[frame_key]
                    mask_area = int(np.count_nonzero(label == int(_num(row.get("selected_mask_id")))))
                    image_area = int(label.size)
                    area_ratio = float(mask_area / max(1, image_area))
                    source = {**frame_source, "mask_area_px": str(mask_area), "image_area_px": str(image_area), "mask_area_ratio": str(area_ratio)}
                    raster_area_fallback += 1
                else:
                    missing_area += 1
                    area_ratio = 0.0
            broad_risk = area_ratio >= float(args.broad_area_ratio)
            bucketed[(_area_bin(area_ratio), int(broad_risk))].append((area_ratio, row, source))
        for (area_bin, risk_flag), vals in sorted(bucketed.items()):
            vals.sort(key=lambda item: (-item[0], int(_num(item[1].get("selected_mask_id"))), item[1].get("object_id", "")))
            for rank, (area_ratio, row, source) in enumerate(vals, start=1):
                score = _area_score(area_ratio, bool(risk_flag), rank, float(args.area_target_ratio))
                group_id = (
                    f"C1_mask_area_risk:{frame_key[0]}:{frame_key[1]}:"
                    f"area{area_bin}:risk{risk_flag}:rank{rank:03d}"
                )
                out = {
                    **row,
                    "masklet_score": score,
                    "selection_status": "control_area_risk_rank_grouped",
                    "mask_area_ratio": area_ratio,
                    "area_bin": area_bin,
                    "broad_area_risk": bool(risk_flag),
                    "area_rank_in_frame_bin": rank,
                }
                grouped[group_id].append(out)
                control_rows.append(
                    {
                        "control": "C1_mask_area_risk",
                        "scene_id": row.get("scene_id", ""),
                        "window_id": row.get("window_id", ""),
                        "frame_id": row.get("frame_id", ""),
                        "selected_mask_id": row.get("selected_mask_id", ""),
                        "group_id": group_id,
                        "mask_area_ratio": area_ratio,
                        "mask_area_px": source.get("mask_area_px", ""),
                        "area_source": "source_registry" if source_lookup.get((row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id")))), {}) else "mask_raster_fallback",
                        "area_bin": area_bin,
                        "broad_area_risk": bool(risk_flag),
                        "area_rank_in_frame_bin": rank,
                        "area_risk_score": score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return _finalize_grouped_rows(
        grouped,
        "C1_mask_area_risk",
        phase5_root=phase5_root,
        output_root=output_root,
        summary_extra={
            "control_definition": "Object ids and scores use only mask area bin, broad-area risk, and per-frame area rank; no D4RT temporal identity or semantic feature is used.",
            "source_rows": _rel(_project(args.source_rows)),
            "broad_area_ratio": float(args.broad_area_ratio),
            "area_target_ratio": float(args.area_target_ratio),
            "missing_area_row_count": missing_area,
            "raster_area_fallback_row_count": raster_area_fallback,
        },
        control_rows=control_rows,
    )


def _load_incidence_by_query(root: Path, decode_variant: str) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[dict[str, str]]]]:
    qids_by_scope: dict[tuple[str, str], list[str]] = defaultdict(list)
    events_by_qid: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in _read_csv(root / "incidence_event_rows.csv"):
        if row.get("decode_variant") != decode_variant:
            continue
        qid = row.get("query_id", "")
        if not qid:
            continue
        events_by_qid[qid].append(row)
        scope = (row.get("scene_id", ""), row.get("window_id", ""))
        if qid not in seen:
            qids_by_scope[scope].append(qid)
            seen.add(qid)
    return qids_by_scope, events_by_qid


def _materialize_c4(args: argparse.Namespace, selected: list[dict[str, str]], phase5_root: Path, output_root: Path) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    source_objects = _source_object_rows(phase5_root, args.source_family)
    selected_by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        selected_by_object[row.get("object_id", "")].append(row)
    qids_by_scope, events_by_qid = _load_incidence_by_query(_project(args.incidence_root), args.decode_variant)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    control_rows: list[dict[str, Any]] = []
    total_source_query_count = 0
    total_random_query_count = 0
    objects_without_pool = 0
    for idx, (source_object_id, rows) in enumerate(sorted(selected_by_object.items()), start=1):
        first = rows[0]
        scope = (first.get("scene_id", ""), first.get("window_id", ""))
        pool = qids_by_scope.get(scope, [])
        if not pool:
            objects_without_pool += 1
            continue
        source_object = source_objects.get(source_object_id, {})
        query_count = int(_num(source_object.get("micro_query_count"), max(_num(row.get("object_query_count")) for row in rows)))
        query_count = max(1, query_count)
        target_frame_count = max(1, len({int(_num(row.get("frame_id"))) for row in rows}))
        if query_count <= len(pool):
            sampled = rng.sample(pool, query_count)
        else:
            sampled = [rng.choice(pool) for _ in range(query_count)]
        total_source_query_count += query_count
        total_random_query_count += len(sampled)
        per_frame_masks: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
        for qid in sampled:
            for event in events_by_qid.get(qid, []):
                mask_id = int(_num(event.get("center_mask_id")))
                if mask_id <= 0:
                    continue
                key = (event.get("scene_id", ""), event.get("window_id", ""), int(_num(event.get("target_frame_id"))))
                per_frame_masks[key][mask_id] += 1
        frame_candidates: list[tuple[int, tuple[str, str, int], int, int]] = []
        for key, counts in per_frame_masks.items():
            mask_id, support = counts.most_common(1)[0]
            if support >= int(args.min_random_mask_support):
                frame_candidates.append((int(support), key, int(mask_id), len(counts)))
        frame_candidates.sort(key=lambda item: (-item[0], item[1][2], item[2]))
        group_id = f"C4_random_micro_primitives:seed{int(args.seed)}:rand_obj_{idx:06d}"
        for support, key, mask_id, distinct_count in frame_candidates[:target_frame_count]:
            scene, window, frame_id = key
            score = float(support / max(1, len(sampled)))
            out = {
                "schema_version": "stream4d_v96_selected_masklet_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": "C4_random_micro_primitives",
                "object_id": "",
                "group_id": group_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": int(frame_id),
                "selected_mask_id": int(mask_id),
                "masklet_support_query_count": int(support),
                "object_query_count": int(len(sampled)),
                "masklet_score": score,
                "selection_status": "control_random_micro_primitive_majority_mask",
                "source_object_id": source_object_id,
                "random_seed": int(args.seed),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            grouped[group_id].append(out)
            control_rows.append(
                {
                    "control": "C4_random_micro_primitives",
                    "source_object_id": source_object_id,
                    "group_id": group_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "selected_mask_id": int(mask_id),
                    "random_query_count": int(len(sampled)),
                    "masklet_support_query_count": int(support),
                    "distinct_positive_mask_count": int(distinct_count),
                    "random_seed": int(args.seed),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    resolved_grouped, duplicate_count, dropped_count = _resolve_frame_mask_collisions(grouped)
    return _finalize_grouped_rows(
        resolved_grouped,
        "C4_random_micro_primitives",
        phase5_root=phase5_root,
        output_root=output_root,
        summary_extra={
            "control_definition": "For each real C object, sample the same number of decoded D3 micro-primitives at random from the same scene/window and select majority masks per frame; object birth uses no D4RT object identity.",
            "incidence_root": _rel(_project(args.incidence_root)),
            "decode_variant": args.decode_variant,
            "random_seed": int(args.seed),
            "min_random_mask_support": int(args.min_random_mask_support),
            "source_object_count": len(selected_by_object),
            "objects_without_random_pool": objects_without_pool,
            "source_query_count_sum": total_source_query_count,
            "random_query_count_sum": total_random_query_count,
            "pre_resolution_duplicate_frame_mask_count": duplicate_count,
            "dropped_duplicate_frame_mask_count": dropped_count,
            "limitation": "This control reuses the already decoded D3 pool instead of launching a fresh independent random-pixel D4RT decode.",
        },
        control_rows=control_rows,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    phase5_root = _project(args.phase5_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _source_selected_rows(phase5_root, args.source_family)
    if args.control == "C0_semantic_only":
        return _materialize_c0(args, selected, phase5_root, output_root)
    if args.control == "C1_mask_area_risk":
        return _materialize_c1(args, selected, phase5_root, output_root)
    if args.control == "C4_random_micro_primitives":
        return _materialize_c4(args, selected, phase5_root, output_root)
    raise ValueError(f"unknown control: {args.control}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v96 C0/C1/C4 required controls as Phase5-style roots.")
    parser.add_argument("--control", required=True, choices=["C0_semantic_only", "C1_mask_area_risk", "C4_random_micro_primitives"])
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--source-family", default="C_hybrid_cover_cluster")
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--semantic-feature-rows", default=str(DEFAULT_SEMANTIC_ROWS))
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--seed", type=int, default=9604)
    parser.add_argument("--broad-area-ratio", type=float, default=0.05)
    parser.add_argument("--area-target-ratio", type=float, default=0.015)
    parser.add_argument("--min-random-mask-support", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
