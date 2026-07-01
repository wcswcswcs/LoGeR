from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _oracle_majority_mapping_bundle  # noqa: E402
from stream4d_native.v68_local_graph_solver import _row_from_mapping, _same_frame_violation_count, _summarize_variant_all  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _float_or_none, _frame_data, _load_csv_rows, _mean, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


METHOD_VARIANTS = [
    "TC0_DINO_only",
    "TC1_carrier_inside",
    "TC2_carrier_inside_outside_residual",
    "TC3_carrier_inside_outside_DINO_guard",
    "TC4_carrier_DINO_signature",
    "TC5_TC4_underseg_shared",
    "TC6_TC5_temporal_adjacency",
    "TC7_shuffled_carrier_control",
    "TC8_no_temporal_carrier_control",
]
CONTROL_VARIANTS = {"TC7_shuffled_carrier_control", "TC8_no_temporal_carrier_control"}
DIAGNOSTIC_VARIANTS = {"TC9_component_proxy_C10_reproduction", "TC10_oracle_closure_diagnostic"}


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _stable_unit_interval(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _load_anchor_pairs(path: Path, variant: str, scenes: set[str]) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in _load_csv_rows(path):
        if str(row.get("anchor_variant")) != variant:
            continue
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        out[str(row.get("chunk_id"))].add((_safe_int(row.get("frame_id")), _safe_int(row.get("mask_id"))))
    return out


def _load_witness_rows(path: Path, scenes: set[str]) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    out: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        anchor = (_safe_int(row.get("anchor_frame")), _safe_int(row.get("anchor_mask")))
        row["_candidate"] = (_safe_int(row.get("candidate_frame")), _safe_int(row.get("candidate_mask")))
        row["_inside_ratio"] = _safe_float(row.get("inside_ratio"))
        row["_outside_ratio"] = _safe_float(row.get("outside_ratio"))
        row["_visible_carrier_count"] = _safe_int(row.get("visible_carrier_count"))
        row["_inside_candidate_count"] = _safe_int(row.get("inside_candidate_count"))
        row["_candidate_underseg_risk"] = _parse_bool(row.get("candidate_underseg_risk"))
        out[str(row.get("chunk_id"))][anchor].append(row)
    return out


def _temporal_score(frame_delta: int) -> float:
    delta_steps = max(1.0, abs(float(frame_delta)) / 5.0)
    return float(1.0 / (1.0 + 0.18 * max(0.0, delta_steps - 1.0)))


def _dino_match(row: dict[str, Any]) -> float:
    left = str(row.get("anchor_DINO_mode_id") or "")
    right = str(row.get("candidate_DINO_mode_id") or "")
    return 1.0 if left and right and left == right else 0.0


def _signature_match(row: dict[str, Any]) -> float:
    left = str(row.get("anchor_repeated_signature_id") or "")
    right = str(row.get("candidate_repeated_signature_id") or "")
    return 1.0 if left and right and left == right else 0.0


def _variant_score(variant: str, row: dict[str, Any]) -> tuple[float, str]:
    inside = float(row["_inside_ratio"])
    outside = float(row["_outside_ratio"])
    residual = max(0.0, inside - outside)
    dino = _dino_match(row)
    signature = _signature_match(row)
    temporal = _temporal_score(_safe_int(row.get("frame_delta")))
    underseg = bool(row["_candidate_underseg_risk"])
    shuffled = _stable_unit_interval("v70_true_carrier_shuffle", row.get("scene_id"), row.get("chunk_id"), row.get("anchor_frame"), row.get("anchor_mask"), row.get("candidate_frame"), row.get("candidate_mask"))
    no_temporal_score = 0.62 * residual + 0.22 * inside + 0.10 * dino + 0.06 * signature
    if variant == "TC0_DINO_only":
        score = 0.82 * dino + 0.18 * signature
        return score, "core" if score >= 0.82 and not underseg else "reject"
    if variant == "TC1_carrier_inside":
        score = inside
        if underseg:
            return score, "shared" if score >= 0.86 else "reject"
        if inside >= 0.72:
            return score, "core"
        return score, "support" if inside >= 0.50 else "reject"
    if variant == "TC2_carrier_inside_outside_residual":
        score = 0.72 * residual + 0.28 * inside
        if underseg:
            return score, "shared" if score >= 0.58 and inside >= 0.70 else "reject"
        if residual >= 0.32 and inside >= 0.66:
            return score, "core"
        return score, "support" if residual >= 0.08 and inside >= 0.54 else "reject"
    if variant == "TC3_carrier_inside_outside_DINO_guard":
        score = 0.62 * residual + 0.24 * inside + 0.14 * dino
        if underseg:
            return score, "shared" if score >= 0.58 and inside >= 0.70 else "reject"
        if score >= 0.52 and inside >= 0.64 and (dino > 0.0 or inside >= 0.82):
            return score, "core"
        return score, "support" if score >= 0.38 and inside >= 0.56 else "reject"
    if variant == "TC4_carrier_DINO_signature":
        score = 0.58 * residual + 0.20 * inside + 0.12 * dino + 0.10 * signature
        if underseg:
            return score, "shared" if score >= 0.58 and inside >= 0.70 else "reject"
        if score >= 0.50 and inside >= 0.62 and (dino > 0.0 or signature > 0.0 or inside >= 0.82):
            return score, "core"
        return score, "support" if score >= 0.36 and inside >= 0.54 else "reject"
    if variant == "TC5_TC4_underseg_shared":
        score = 0.58 * residual + 0.20 * inside + 0.12 * dino + 0.10 * signature
        if underseg:
            return score, "shared" if score >= 0.48 and inside >= 0.62 else "reject"
        if score >= 0.48 and inside >= 0.60 and (dino > 0.0 or signature > 0.0 or inside >= 0.80):
            return score, "core"
        return score, "support" if score >= 0.34 and inside >= 0.52 else "reject"
    if variant == "TC6_TC5_temporal_adjacency":
        score = 0.50 * residual + 0.18 * inside + 0.10 * dino + 0.08 * signature + 0.14 * temporal
        if underseg:
            return score, "shared" if score >= 0.48 and inside >= 0.62 else "reject"
        if score >= 0.46 and inside >= 0.58 and (dino > 0.0 or signature > 0.0 or inside >= 0.78):
            return score, "core"
        return score, "support" if score >= 0.34 and inside >= 0.50 else "reject"
    if variant == "TC7_shuffled_carrier_control":
        return shuffled, "core" if shuffled >= 0.985 and not underseg else "reject"
    if variant == "TC8_no_temporal_carrier_control":
        if underseg:
            return no_temporal_score, "shared" if no_temporal_score >= 0.50 and inside >= 0.62 else "reject"
        if no_temporal_score >= 0.48 and inside >= 0.58 and (dino > 0.0 or signature > 0.0 or inside >= 0.78):
            return no_temporal_score, "core"
        return no_temporal_score, "support" if no_temporal_score >= 0.34 and inside >= 0.50 else "reject"
    if variant == "TC11_adjacent15_residual_strict":
        if _safe_int(row.get("frame_delta")) > 15:
            return 0.0, "reject"
        score = 0.64 * residual + 0.22 * inside + 0.14 * temporal
        if underseg:
            return score, "reject"
        if residual >= 0.28 and inside >= 0.64:
            return score, "core"
        return score, "support" if residual >= 0.04 and inside >= 0.50 else "reject"
    if variant == "TC12_adjacent30_support_recall":
        if _safe_int(row.get("frame_delta")) > 30:
            return 0.0, "reject"
        score = 0.46 * residual + 0.30 * inside + 0.14 * temporal + 0.10 * signature
        if underseg:
            return score, "shared" if inside >= 0.86 and residual >= 0.20 else "reject"
        if inside >= 0.66 and residual >= 0.18:
            return score, "core"
        return score, "support" if inside >= 0.42 and residual >= -0.05 else "reject"
    if variant == "TC13_adjacent30_no_underseg_signature_veto":
        if _safe_int(row.get("frame_delta")) > 30:
            return 0.0, "reject"
        score = 0.50 * residual + 0.20 * inside + 0.16 * temporal + 0.14 * signature
        if underseg:
            return score, "reject"
        if inside >= 0.62 and residual >= 0.14 and (signature > 0.0 or inside >= 0.78):
            return score, "core"
        return score, "support" if inside >= 0.48 and residual >= 0.0 and (signature > 0.0 or inside >= 0.70) else "reject"
    raise ValueError(f"unknown v70 closure variant: {variant}")


def _object_temporal_span(mapping: dict[tuple[int, int], int]) -> float | None:
    frames_by_object: dict[int, set[int]] = defaultdict(set)
    for (frame_id, _mask_id), object_id in mapping.items():
        frames_by_object[int(object_id)].add(int(frame_id))
    spans = [len(frames) for frames in frames_by_object.values()]
    return float(np.mean(spans)) if spans else None


def _build_mapping_for_variant(
    *,
    variant: str,
    anchors: set[tuple[int, int]],
    witness_by_anchor: dict[tuple[int, int], list[dict[str, Any]]],
    max_supports_per_anchor: int,
    closure_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping: dict[tuple[int, int], int] = {}
    used_masks: set[tuple[int, int]] = set()
    object_id = 0
    core_counts: list[int] = []
    support_counts: list[int] = []
    shared_counts: list[int] = []
    reject_counts: list[int] = []
    candidate_counts: list[int] = []
    visible_material_counts: list[int] = []
    visible_anchor_count = 0
    underseg_shared = 0
    for anchor in sorted(anchors):
        object_id += 1
        object_members: set[tuple[int, int]] = {anchor}
        used_frames: set[int] = {int(anchor[0])}
        candidate_rows = list(witness_by_anchor.get(anchor, []))
        candidate_counts.append(len(candidate_rows))
        anchor_carrier_count = int(max([_safe_int(row.get("anchor_carrier_count")) for row in candidate_rows] or [0]))
        visible_material_counts.append(anchor_carrier_count)
        if anchor_carrier_count > 0:
            visible_anchor_count += 1
        scored: list[tuple[float, str, tuple[int, int], dict[str, Any]]] = []
        for row in candidate_rows:
            other = row["_candidate"]
            if other == anchor or int(other[0]) == int(anchor[0]):
                continue
            score, role = _variant_score(variant, row)
            scored.append((score, role, other, row))
        selected = 0
        reject_count = 0
        shared_count = 0
        support_count = 0
        core_count = 1
        for score, role, other, row in sorted(scored, reverse=True, key=lambda item: item[0]):
            if selected >= int(max_supports_per_anchor):
                break
            if role == "reject":
                reject_count += 1
                continue
            if other in used_masks or int(other[0]) in used_frames:
                reject_count += 1
                continue
            if role == "shared":
                shared_count += 1
                underseg_shared += 1
            else:
                object_members.add(other)
                used_frames.add(int(other[0]))
                if role == "core":
                    core_count += 1
                else:
                    support_count += 1
            selected += 1
            if len(closure_rows) < 1_200_000:
                closure_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "closure_variant": variant,
                        "anchor_frame": int(anchor[0]),
                        "anchor_mask": int(anchor[1]),
                        "candidate_frame": int(other[0]),
                        "candidate_mask": int(other[1]),
                        "role": role,
                        "closure_score": float(score),
                        "inside_ratio": row.get("inside_ratio"),
                        "outside_ratio": row.get("outside_ratio"),
                        "inside_candidate_count": row.get("inside_candidate_count"),
                        "visible_carrier_count": row.get("visible_carrier_count"),
                        "frame_delta": row.get("frame_delta"),
                        "dino_match": _dino_match(row),
                        "signature_match": _signature_match(row),
                        "candidate_underseg_risk": bool(row["_candidate_underseg_risk"]),
                        "uses_gt_for_prediction": False,
                        "diagnostic_only": False,
                        "forbidden_for_method_table": False,
                    }
                )
        for member in object_members:
            if member not in used_masks:
                mapping[member] = object_id
                used_masks.add(member)
        core_counts.append(core_count)
        support_counts.append(support_count)
        shared_counts.append(shared_count)
        reject_counts.append(reject_count)
    diag = {
        "core_mask_count_mean": _mean([float(v) for v in core_counts]),
        "support_mask_count_mean": _mean([float(v) for v in support_counts]),
        "shared_mask_count_mean": _mean([float(v) for v in shared_counts]),
        "reject_mask_count_mean": _mean([float(v) for v in reject_counts]),
        "candidate_masks_per_anchor": _mean([float(v) for v in candidate_counts]),
        "anchor_with_carrier_witness_rate": float(visible_anchor_count / max(1, len(visible_material_counts))),
        "mean_visible_material_per_anchor": _mean([float(v) for v in visible_material_counts]),
        "underseg_shared_rate": float(underseg_shared / max(1, sum(candidate_counts))),
        "same_frame_cannot_link_violation_count": _same_frame_violation_count(mapping),
        "shared_mask_count": int(sum(shared_counts)),
        "reject_mask_count": int(sum(reject_counts)),
        "unknown_mask_count": 0,
    }
    return mapping, diag


def _summarize_closure_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "closure_variant": variant,
            "anchor_count": int(sum(int(float(row.get("anchor_count") or 0)) for row in subset)),
            "anchor_with_carrier_witness_rate": _mean([_float_or_none(row.get("anchor_with_carrier_witness_rate")) for row in subset]),
            "mean_visible_material_per_anchor": _mean([_float_or_none(row.get("mean_visible_material_per_anchor")) for row in subset]),
            "candidate_masks_per_anchor": _mean([_float_or_none(row.get("candidate_masks_per_anchor")) for row in subset]),
            "core_mask_count_mean": _mean([_float_or_none(row.get("core_mask_count_mean")) for row in subset]),
            "support_mask_count_mean": _mean([_float_or_none(row.get("support_mask_count_mean")) for row in subset]),
            "shared_mask_count_mean": _mean([_float_or_none(row.get("shared_mask_count_mean")) for row in subset]),
            "reject_mask_count_mean": _mean([_float_or_none(row.get("reject_mask_count_mean")) for row in subset]),
            "single_anchor_SF50": base.get("local_score_free_match50_recall_mean"),
            "single_anchor_AP50": base.get("local_AP50_mean"),
            "single_anchor_GT_best_IoU_mean": base.get("local_GT_best_IoU_mean_mean"),
            "single_anchor_temporal_span_mean": _mean([_float_or_none(row.get("single_anchor_temporal_span_mean")) for row in subset]),
            "single_anchor_single_frame_rate": base.get("single_frame_object_rate_mean"),
            "underseg_false_bridge_rate": _mean([_float_or_none(row.get("underseg_shared_rate")) for row in subset]),
        }
    )
    return base


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _import_c10_metric(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    best = summary.get("best_closure_variant") or {}
    return {
        "variant": "TC9_component_proxy_C10_reproduction",
        "closure_variant": "TC9_component_proxy_C10_reproduction",
        "imported_from": _rel(summary_path),
        "candidate_source": "v69r2_component_index_proxy",
        "single_anchor_SF50": best.get("single_anchor_SF50"),
        "single_anchor_AP50": best.get("single_anchor_AP50"),
        "single_anchor_GT_best_IoU_mean": best.get("single_anchor_GT_best_IoU_mean"),
        "single_anchor_temporal_span_mean": best.get("single_anchor_temporal_span_mean"),
        "single_anchor_single_frame_rate": best.get("single_anchor_single_frame_rate"),
        "same_frame_cannot_link_violation_count_sum": best.get("same_frame_cannot_link_violation_count_sum"),
        "support_mask_count_mean": best.get("support_mask_count_mean"),
        "shared_mask_count_mean": best.get("shared_mask_count_mean"),
        "uses_gt_for_prediction": False,
        "forbidden_for_method_table": False,
        "diagnostic_only": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    scene_set = set(scenes)
    variants = _parse_csv_list(args.variants) or METHOD_VARIANTS
    anchor_path = _rooted(args.anchor_rows)
    witness_path = _rooted(args.witness_rows)
    anchors_by_chunk = _load_anchor_pairs(anchor_path, str(args.anchor_variant), scene_set)
    witness_by_chunk = _load_witness_rows(witness_path, scene_set)
    closure_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    include_oracle = bool(args.include_oracle)
    for scene in scenes:
        print(f"[v70-true-material-closure] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            t0 = time.time()
            chunk_id = str(chunk.get("chunk_id"))
            anchors = anchors_by_chunk.get(chunk_id, set())
            witness_for_chunk = witness_by_chunk.get(chunk_id, {})
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids or not anchors:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for variant in variants:
                mapping, diag = _build_mapping_for_variant(
                    variant=variant,
                    anchors=anchors,
                    witness_by_anchor=witness_for_chunk,
                    max_supports_per_anchor=int(args.max_supports_per_anchor),
                    closure_rows=closure_rows,
                    scene=scene,
                    chunk_id=chunk_id,
                )
                diag["anchor_count"] = int(len(anchors))
                row = _row_from_mapping(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    diag=diag,
                    pipeline_root=pipeline_root,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                )
                row.update(diag)
                row["single_anchor_temporal_span_mean"] = _object_temporal_span(mapping)
                row["runtime_sec"] = float(time.time() - t0)
                row["anchor_variant"] = str(args.anchor_variant)
                row["witness_rows"] = _rel(witness_path)
                chunk_rows.append(row)
            if include_oracle:
                nodes = set(anchors)
                for rows in witness_for_chunk.values():
                    nodes.update(row["_candidate"] for row in rows)
                if nodes:
                    oracle_mapping = _oracle_majority_mapping_bundle(
                        frame_data=frame_data,
                        selected_pairs=set(),
                        representative_pairs=nodes,
                    )["representative"][0]
                    diag = {
                        "same_frame_cannot_link_violation_count": _same_frame_violation_count(oracle_mapping),
                        "shared_mask_count": 0,
                        "reject_mask_count": 0,
                        "unknown_mask_count": 0,
                    }
                    row = _row_from_mapping(
                        scene=scene,
                        chunk_id=chunk_id,
                        variant="TC10_oracle_closure_diagnostic",
                        frame_ids=frame_ids,
                        chunk=chunk,
                        frame_data=frame_data,
                        mapping=oracle_mapping,
                        diag=diag,
                        pipeline_root=pipeline_root,
                        uses_gt_for_prediction=True,
                        forbidden_for_method_table=True,
                    )
                    row.update(diag)
                    row["single_anchor_temporal_span_mean"] = _object_temporal_span(oracle_mapping)
                    row["runtime_sec"] = float(time.time() - t0)
                    row["anchor_variant"] = str(args.anchor_variant)
                    row["witness_rows"] = _rel(witness_path)
                    chunk_rows.append(row)

    metric_rows = [_summarize_closure_variant(chunk_rows, variant) for variant in variants]
    if include_oracle:
        metric_rows.append(_summarize_closure_variant(chunk_rows, "TC10_oracle_closure_diagnostic"))
    c10 = _import_c10_metric(_rooted(args.v69r2_c10_summary))
    metric_rows.append(c10)
    method_metrics = [
        row
        for row in metric_rows
        if row.get("closure_variant") not in CONTROL_VARIANTS and row.get("closure_variant") not in DIAGNOSTIC_VARIANTS
    ]
    best = max(method_metrics, key=lambda row: float(row.get("single_anchor_SF50") or 0.0), default={})
    by_variant = {str(row.get("closure_variant")): row for row in metric_rows}
    shuffled = by_variant.get("TC7_shuffled_carrier_control", {})
    no_temporal = by_variant.get("TC8_no_temporal_carrier_control", {})
    best_sf50 = _float_or_none(best.get("single_anchor_SF50"))
    best_gt = _float_or_none(best.get("single_anchor_GT_best_IoU_mean"))
    best_span = _float_or_none(best.get("single_anchor_temporal_span_mean"))
    best_single = _float_or_none(best.get("single_anchor_single_frame_rate"))
    best_violation = int(float(best.get("same_frame_cannot_link_violation_count_sum") or 0)) if best else 0
    shuffled_sf50 = _float_or_none(shuffled.get("single_anchor_SF50"))
    no_temporal_sf50 = _float_or_none(no_temporal.get("single_anchor_SF50"))
    c10_sf50 = _float_or_none(c10.get("single_anchor_SF50"))
    c10_gt = _float_or_none(c10.get("single_anchor_GT_best_IoU_mean"))
    c10_single = _float_or_none(c10.get("single_anchor_single_frame_rate"))
    real_minus_shuffled = None if best_sf50 is None or shuffled_sf50 is None else float(best_sf50 - shuffled_sf50)
    real_minus_no_temporal = None if best_sf50 is None or no_temporal_sf50 is None else float(best_sf50 - no_temporal_sf50)
    strong_pass = {
        "single_anchor_SF50_ge_v69r2_C10_plus_0p08": best_sf50 is not None and c10_sf50 is not None and best_sf50 >= c10_sf50 + 0.08,
        "single_anchor_GT_best_IoU_ge_v69r2_C10_plus_0p08": best_gt is not None and c10_gt is not None and best_gt >= c10_gt + 0.08,
        "single_anchor_single_frame_rate_le_v69r2_C10_minus_0p20": best_single is not None and c10_single is not None and best_single <= c10_single - 0.20,
    }
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "single_anchor_SF50_ge_0p15": best_sf50 is not None and best_sf50 >= 0.15,
        "single_anchor_GT_best_IoU_ge_0p20": best_gt is not None and best_gt >= 0.20,
        "single_anchor_temporal_span_mean_ge_2p0": best_span is not None and best_span >= 2.0,
        "single_anchor_single_frame_rate_le_0p65": best_single is not None and best_single <= 0.65,
        "same_frame_violation_count_eq_0": best_violation == 0,
        "real_minus_shuffled_SF50_ge_0p05": real_minus_shuffled is not None and real_minus_shuffled >= 0.05,
        "real_minus_no_temporal_SF50_ge_0p03": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.03,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    summary = {
        "phase": "v70_true_material_closure",
        "decision": "PASS_TRUE_MATERIAL_CLOSURE" if gate["pass"] else "NO_GO_TRUE_MATERIAL_CLOSURE",
        "gate": gate,
        "strong_pass_gate": strong_pass,
        "strong_pass": all(bool(value) for value in strong_pass.values()),
        "best_closure_variant": best,
        "real_minus_shuffled_SF50": real_minus_shuffled,
        "real_minus_no_temporal_SF50": real_minus_no_temporal,
        "v69r2_C10_reproduction": c10,
        "anchor_variant": str(args.anchor_variant),
        "anchor_rows": _rel(anchor_path),
        "witness_rows": _rel(witness_path),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "variants": variants,
        "max_supports_per_anchor": int(args.max_supports_per_anchor),
        "include_oracle": include_oracle,
        "rows": {
            "closure_summary_json": _rel(output_root / "closure_summary.json"),
            "closure_rows_csv": _rel(output_root / "closure_rows.csv"),
            "closure_metric_rows_csv": _rel(output_root / "closure_metric_rows.csv"),
            "closure_chunk_rows_csv": _rel(output_root / "closure_chunk_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "TC0-TC8 are built from v70 carrier witness rows. TC9 imports v69-r2 C10 component proxy baseline for comparison only.",
            "TC10 oracle is diagnostic-only and forbidden for method tables when enabled.",
        ],
    }
    _write_csv(output_root / "closure_rows.csv", closure_rows)
    _write_csv(output_root / "closure_metric_rows.csv", metric_rows)
    _write_csv(output_root / "closure_chunk_rows.csv", chunk_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "closure_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "closure_summary.json",
        output_root / "closure_rows.csv",
        output_root / "closure_metric_rows.csv",
        output_root / "closure_chunk_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 true carrier material closure evaluator.")
    parser.add_argument("--output-root", default="outputs/audit/v70_true_material_closure")
    parser.add_argument("--anchor-rows", default="outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_rows.csv")
    parser.add_argument("--anchor-variant", default="A9_clean_recall_support_floor_u15")
    parser.add_argument("--witness-rows", default="outputs/audit/v70_carrier_witness/witness_rows.csv")
    parser.add_argument("--v69r2-c10-summary", default="outputs/audit/v69r2_material_closure_repair2_shared_no_bridge_probe5/closure_summary.json")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--variants", default=",".join(METHOD_VARIANTS))
    parser.add_argument("--max-supports-per-anchor", type=int, default=6)
    parser.add_argument("--include-oracle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
