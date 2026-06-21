from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from .v47_common import bbox_iou, cosine, parse_bool, parse_float, parse_int, rank_auc, safe_mean


def _mask_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["scene"]), parse_int(row["frame_id"]), parse_int(row["mask_id"])


def _frame_rank(mask_rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    by_scene: dict[str, set[int]] = defaultdict(set)
    for row in mask_rows:
        by_scene[str(row["scene"])].add(parse_int(row["frame_id"]))
    for scene, frames in by_scene.items():
        for rank, frame_id in enumerate(sorted(frames)):
            out[(scene, int(frame_id))] = int(rank)
    return out


def _carrier_indexes(carrier_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int, int], set[int]], dict[tuple[str, int, int], dict[str, Any]]]:
    carriers_by_mask: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    obs_by_carrier_frame: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in carrier_rows:
        scene = str(row["scene"])
        frame_id = parse_int(row["frame_id"])
        carrier_id = parse_int(row["carrier_id"])
        if parse_bool(row.get("visible")) and parse_bool(row.get("valid_uv")):
            obs_by_carrier_frame[(scene, frame_id, carrier_id)] = row
            mask_id = parse_int(row.get("observed_mask_id"), 0)
            if mask_id > 0:
                carriers_by_mask[(scene, frame_id, mask_id)].add(carrier_id)
    return carriers_by_mask, obs_by_carrier_frame


def _directed_containment(
    source_key: tuple[str, int, int],
    target_key: tuple[str, int, int],
    carriers_by_mask: dict[tuple[str, int, int], set[int]],
    obs_by_carrier_frame: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[float, float, int]:
    scene, _source_frame, _source_mask = source_key
    _target_scene, target_frame, target_mask = target_key
    source_carriers = carriers_by_mask.get(source_key, set())
    denom = 0
    inside = 0
    for carrier_id in source_carriers:
        obs = obs_by_carrier_frame.get((scene, target_frame, carrier_id))
        if obs is None:
            continue
        denom += 1
        if parse_int(obs.get("observed_mask_id"), 0) == int(target_mask):
            inside += 1
    if denom <= 0:
        return 0.0, 1.0, 0
    containment = float(inside / denom)
    return containment, float(1.0 - containment), int(denom)


def _deterministic_target(rows: list[dict[str, Any]], seed_text: str) -> dict[int, dict[str, Any]]:
    if not rows:
        return {}
    keyed = []
    for idx, row in enumerate(rows):
        raw = f"{seed_text}:{row.get('node_id')}:{row.get('scene')}:{row.get('frame_id')}:{row.get('mask_id')}"
        keyed.append((hashlib.sha1(raw.encode("utf-8")).hexdigest(), idx))
    keyed.sort()
    order = [idx for _key, idx in keyed]
    return {parse_int(rows[idx]["node_id"]): rows[order[(pos + 1) % len(order)]] for pos, idx in enumerate(order)}


def _edge_scores(
    left: dict[str, Any],
    right: dict[str, Any],
    carriers_by_mask: dict[tuple[str, int, int], set[int]],
    obs_by_carrier_frame: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    left_key = _mask_key(left)
    right_key = _mask_key(right)
    fwd, outside_fwd, fwd_visible = _directed_containment(left_key, right_key, carriers_by_mask, obs_by_carrier_frame)
    bwd, outside_bwd, bwd_visible = _directed_containment(right_key, left_key, carriers_by_mask, obs_by_carrier_frame)
    symmetric = 0.5 * (fwd + bwd)
    outside_values = [outside_fwd if fwd_visible > 0 else None, outside_bwd if bwd_visible > 0 else None]
    usable_outside = [float(value) for value in outside_values if value is not None]
    visible_outside = float(sum(usable_outside) / len(usable_outside)) if usable_outside else 1.0
    semantic = cosine(left.get("core_feature"), right.get("core_feature"))
    bbox = bbox_iou(
        [left.get("bbox_x0", 0), left.get("bbox_y0", 0), left.get("bbox_x1", 0), left.get("bbox_y1", 0)],
        [right.get("bbox_x0", 0), right.get("bbox_y0", 0), right.get("bbox_x1", 0), right.get("bbox_y1", 0)],
    )
    a4 = symmetric * max(0.0, 1.0 - 0.5 * visible_outside)
    a5 = 0.75 * a4 + 0.25 * semantic
    return {
        "d4rt_forward_containment": fwd,
        "d4rt_backward_containment": bwd,
        "visible_outside_forward": outside_fwd,
        "visible_outside_backward": outside_bwd,
        "forward_visible_carrier_count": fwd_visible,
        "backward_visible_carrier_count": bwd_visible,
        "symmetric_geometry_score": symmetric,
        "visible_outside": visible_outside,
        "semantic_similarity": semantic,
        "bbox_iou": bbox,
        "A0_bbox_overlap": bbox,
        "A1_semantic_feature_only": semantic,
        "A2_d4rt_forward": fwd,
        "A3_d4rt_symmetric": symmetric,
        "A4_d4rt_visible_veto": a4,
        "A5_d4rt_semantic_confirmation": a5,
        "edge_cost": float(1.0 - a5 + 0.5 * visible_outside),
        "edge_accept_candidate": bool(a5 >= 0.30 and symmetric >= 0.20 and visible_outside <= 0.85),
    }


def build_temporal_candidate_edges(
    *,
    mask_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
    gap_max: int = 2,
    reactivation_gap_max: int = 5,
    min_forward_visible_carriers: int = 0,
    min_backward_visible_carriers: int = 0,
    max_visible_outside: float = 1.0,
) -> dict[str, Any]:
    carriers_by_mask, obs_by_carrier_frame = _carrier_indexes(carrier_rows)
    rank_by_scene_frame = _frame_rank(mask_rows)
    by_scene_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        by_scene_frame[(str(row["scene"]), parse_int(row["frame_id"]))].append(row)
    by_scene: dict[str, list[int]] = defaultdict(list)
    for scene, frame_id in by_scene_frame:
        by_scene[scene].append(frame_id)
    shuffled_by_frame = {
        key: _deterministic_target(rows, seed_text=f"v47_shuffle:{key[0]}:{key[1]}") for key, rows in by_scene_frame.items()
    }
    edge_rows: list[dict[str, Any]] = []
    for scene, frames_raw in by_scene.items():
        frames = sorted(set(frames_raw))
        for src_frame in frames:
            src_rank = rank_by_scene_frame[(scene, src_frame)]
            for dst_frame in frames:
                dst_rank = rank_by_scene_frame[(scene, dst_frame)]
                delta_rank = dst_rank - src_rank
                if delta_rank <= 0 or delta_rank > int(reactivation_gap_max):
                    continue
                if delta_rank == 1:
                    edge_type = "adjacent"
                elif delta_rank <= int(gap_max):
                    edge_type = "skip"
                else:
                    edge_type = "short_reactivation"
                for left in by_scene_frame[(scene, src_frame)]:
                    for right in by_scene_frame[(scene, dst_frame)]:
                        scores = _edge_scores(left, right, carriers_by_mask, obs_by_carrier_frame)
                        shuffled_target = shuffled_by_frame[(scene, dst_frame)].get(parse_int(right["node_id"]), right)
                        shuffled_scores = _edge_scores(left, shuffled_target, carriers_by_mask, obs_by_carrier_frame)
                        same_gt = (
                            str(left.get("diagnostic_gt_instance", "")) != ""
                            and str(right.get("diagnostic_gt_instance", "")) != ""
                            and str(left.get("diagnostic_gt_instance")) == str(right.get("diagnostic_gt_instance"))
                        )
                        edge_rows.append(
                            {
                                "src_node_id": parse_int(left["node_id"]),
                                "dst_node_id": parse_int(right["node_id"]),
                                "scene": scene,
                                "src_mask_observation_id": left.get("mask_observation_id"),
                                "dst_mask_observation_id": right.get("mask_observation_id"),
                                "src_frame_id": src_frame,
                                "dst_frame_id": dst_frame,
                                "src_mask_id": parse_int(left["mask_id"]),
                                "dst_mask_id": parse_int(right["mask_id"]),
                                "delta_frame": int(dst_frame - src_frame),
                                "delta_observation_rank": int(delta_rank),
                                "edge_type": edge_type,
                                "scale_weak_flag": False,
                                "same_frame_conflict_proxy": False,
                                "diagnostic_same_gt": same_gt,
                                "src_diagnostic_gt": left.get("diagnostic_gt_instance"),
                                "dst_diagnostic_gt": right.get("diagnostic_gt_instance"),
                                "A7_shuffled_D4RT": shuffled_scores["A5_d4rt_semantic_confirmation"],
                                "A8_no_temporal_control": scores["A1_semantic_feature_only"],
                                **scores,
                                "uses_gt_for_prediction": False,
                                "uses_gt_for_diagnostic_labels": True,
                            }
                        )
    raw_edge_count = len(edge_rows)
    edge_rows = [
        row
        for row in edge_rows
        if parse_int(row.get("forward_visible_carrier_count")) >= int(min_forward_visible_carriers)
        and parse_int(row.get("backward_visible_carrier_count")) >= int(min_backward_visible_carriers)
        and parse_float(row.get("visible_outside"), 1.0) <= float(max_visible_outside)
    ]
    edge_filter = {
        "raw_edge_count": raw_edge_count,
        "filtered_edge_count": len(edge_rows),
        "min_forward_visible_carriers": int(min_forward_visible_carriers),
        "min_backward_visible_carriers": int(min_backward_visible_carriers),
        "max_visible_outside": float(max_visible_outside),
    }
    summary_rows = summarize_edges(edge_rows)
    return {
        "edge_rows": edge_rows,
        "summary_rows": summary_rows,
        "summary": _edge_summary_payload(edge_rows, summary_rows, edge_filter=edge_filter),
    }


def _precision_top1_per_node(rows: list[dict[str, Any]], score_key: str) -> float | None:
    by_src: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_src[(str(row["scene"]), parse_int(row["src_node_id"]))].append(row)
    hits = []
    for group in by_src.values():
        best = max(group, key=lambda item: parse_float(item.get(score_key)))
        hits.append(1.0 if parse_bool(best.get("diagnostic_same_gt")) else 0.0)
    return safe_mean(hits)


def _recall_topk_per_node(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    by_src: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_src[(str(row["scene"]), parse_int(row["src_node_id"]))].append(row)
    hits = []
    for group in by_src.values():
        if not any(parse_bool(row.get("diagnostic_same_gt")) for row in group):
            continue
        ranked = sorted(group, key=lambda item: parse_float(item.get(score_key)), reverse=True)[: int(k)]
        hits.append(1.0 if any(parse_bool(row.get("diagnostic_same_gt")) for row in ranked) else 0.0)
    return safe_mean(hits)


def summarize_edges(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = [parse_bool(row.get("diagnostic_same_gt")) for row in edge_rows]
    for score_key in [
        "A0_bbox_overlap",
        "A1_semantic_feature_only",
        "A2_d4rt_forward",
        "A3_d4rt_symmetric",
        "A4_d4rt_visible_veto",
        "A5_d4rt_semantic_confirmation",
        "A7_shuffled_D4RT",
        "A8_no_temporal_control",
    ]:
        scores = [parse_float(row.get(score_key)) for row in edge_rows]
        rows.append(
            {
                "score_key": score_key,
                "edge_count": len(edge_rows),
                "edge_candidate_density": None,
                "edge_precision@top1_per_node": _precision_top1_per_node(edge_rows, score_key),
                "edge_recall@top3_per_node": _recall_topk_per_node(edge_rows, score_key, 3),
                "edge_AUC": rank_auc(labels, scores),
                "same_frame_false_merge_proxy": 0,
                "visible_outside_veto_precision": _precision_top1_per_node(
                    [row for row in edge_rows if parse_float(row.get("visible_outside")) >= 0.5], score_key
                ),
                "scene0081_edge_precision": _precision_top1_per_node(
                    [row for row in edge_rows if str(row.get("scene")) == "scene0081_01"], score_key
                ),
                "scene0591_edge_precision": _precision_top1_per_node(
                    [row for row in edge_rows if str(row.get("scene")) == "scene0591_00"], score_key
                ),
            }
        )
    by_key = {row["score_key"]: row for row in rows}
    real = by_key.get("A5_d4rt_semantic_confirmation", {})
    shuffled = by_key.get("A7_shuffled_D4RT", {})
    no_temporal = by_key.get("A8_no_temporal_control", {})
    real_auc = real.get("edge_AUC")
    shuffled_auc = shuffled.get("edge_AUC")
    no_temporal_auc = no_temporal.get("edge_AUC")
    real["real_minus_shuffled_edge_AUC"] = None if real_auc is None or shuffled_auc is None else float(real_auc - shuffled_auc)
    real["real_minus_no_temporal_edge_AUC"] = None if real_auc is None or no_temporal_auc is None else float(real_auc - no_temporal_auc)
    real["gate_pass"] = bool(
        (real.get("edge_precision@top1_per_node") or 0.0) >= 0.80
        and (real.get("edge_recall@top3_per_node") or 0.0) >= 0.55
        and (real.get("real_minus_shuffled_edge_AUC") or 0.0) >= 0.10
        and (real.get("real_minus_no_temporal_edge_AUC") or 0.0) >= 0.08
    )
    return rows


def _edge_summary_payload(
    edge_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], edge_filter: dict[str, Any] | None = None
) -> dict[str, Any]:
    best = next((row for row in summary_rows if row["score_key"] == "A5_d4rt_semantic_confirmation"), {})
    return {
        "phase": "v47_adjacent_edge_audit",
        "edge_count": len(edge_rows),
        "edge_filter": edge_filter or {
            "raw_edge_count": len(edge_rows),
            "filtered_edge_count": len(edge_rows),
            "min_forward_visible_carriers": 0,
            "min_backward_visible_carriers": 0,
            "max_visible_outside": 1.0,
        },
        "summary_rows": summary_rows,
        "gate": {
            "best_score_key": "A5_d4rt_semantic_confirmation",
            "A5_edge_precision_top1_pass": bool((best.get("edge_precision@top1_per_node") or 0.0) >= 0.80),
            "A5_edge_recall_top3_pass": bool((best.get("edge_recall@top3_per_node") or 0.0) >= 0.55),
            "real_minus_shuffled_pass": bool((best.get("real_minus_shuffled_edge_AUC") or 0.0) >= 0.10),
            "real_minus_no_temporal_pass": bool((best.get("real_minus_no_temporal_edge_AUC") or 0.0) >= 0.08),
            "pass": bool(best.get("gate_pass", False)),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
