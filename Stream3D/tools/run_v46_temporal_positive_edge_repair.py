from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import (
    ROOT,
    MaskNode,
    WindowTrace,
    _build_nodes,
    _deterministic_permutation,
    _json_safe,
    _load_scene_windows,
    _rank_auc,
    _safe_mean,
    _safe_median,
    _safe_quantile,
    _shared_jaccard,
)
from tools.run_v46_supporter_quality_raw_repair import _supporter_quality_rows, _weighted_view_consensus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _temporal_containment(
    left: MaskNode,
    right: MaskNode,
    *,
    frame_rank_by_id: dict[int, int],
    max_observation_gap: int,
) -> tuple[float, int, float]:
    left_rank = frame_rank_by_id.get(int(left.frame_id))
    right_rank = frame_rank_by_id.get(int(right.frame_id))
    if left_rank is None or right_rank is None:
        return 0.0, 0, 0.0
    gap = abs(int(left_rank) - int(right_rank))
    if gap <= 0 or gap > int(max_observation_gap):
        return 0.0, 0, 0.0
    scores: list[float] = []
    common_windows = sorted(set(left.inc_by_window) & set(right.inc_by_window))
    for window_index in common_windows:
        left_idx = set(left.inc_by_window.get(window_index, set()))
        right_idx = set(right.inc_by_window.get(window_index, set()))
        if not left_idx or not right_idx:
            continue
        inter = len(left_idx & right_idx)
        score = 0.5 * (inter / max(len(left_idx), 1) + inter / max(len(right_idx), 1))
        scores.append(float(score))
    if not scores:
        return 0.0, 0, 0.0
    return float(max(scores)), int(len(scores)), float(np.mean(scores))


def _directed_visible_outside(
    source: MaskNode,
    target: MaskNode,
    windows_by_index: dict[int, WindowTrace],
    *,
    min_visible_carriers: int,
) -> tuple[float | None, int, float | None]:
    outside_values: list[float] = []
    coverage_values: list[float] = []
    for window_index, source_indices in source.inc_by_window.items():
        window = windows_by_index.get(window_index)
        if window is None or int(target.frame_id) not in window.frame_ids:
            continue
        local_index = window.frame_ids.index(int(target.frame_id))
        labels_at_carrier = window.labels_by_frame.get(int(target.frame_id))
        if labels_at_carrier is None:
            continue
        src_idx = np.asarray(sorted(source_indices), dtype=np.int64)
        visible_idx = src_idx[window.visible[local_index, src_idx]]
        if visible_idx.size < int(min_visible_carriers):
            continue
        coverage = float(np.mean(labels_at_carrier[visible_idx] == int(target.mask_id)))
        coverage_values.append(coverage)
        outside_values.append(float(1.0 - coverage))
    if not outside_values:
        return None, 0, None
    return float(np.mean(outside_values)), int(len(outside_values)), float(np.mean(coverage_values))


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    ranked = sorted(rows, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(rows))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if row.get("diagnostic_same_gt") is True) / len(ranked))


def _edge_rows_for_scene(
    *,
    scene: str,
    nodes: list[MaskNode],
    windows: list[WindowTrace],
    frame_rank_by_id: dict[int, int],
    quality_variant: str,
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
    max_temporal_observation_gap: int,
    temporal_weight: float,
    visible_outside_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        node
        for node in nodes
        if node.support_count >= int(min_node_carriers) and node.dominant_gt is not None and node.dominant_gt_purity is not None
    ]
    eligible.sort(key=lambda node: (node.support_count, node.area), reverse=True)
    capped = eligible[: int(max_edge_nodes)]
    windows_by_index = {window.window_index: window for window in windows}
    quality_by_key, quality_rows = _supporter_quality_rows(
        scene=scene,
        nodes=nodes,
        windows_by_index=windows_by_index,
        min_visible_carriers=int(min_visible_carriers),
        underseg_purity_threshold=0.70,
        low_q_threshold=0.55,
    )
    permuted_indices = _deterministic_permutation(capped, seed_text=f"v46_temporal_positive_shuffle:{scene}") if capped else []
    shuffled_by_node_id: dict[int, MaskNode] = {}
    for idx, node in enumerate(capped):
        shuffled_by_node_id[node.node_id] = capped[permuted_indices[idx % len(permuted_indices)]]
    edge_rows: list[dict[str, Any]] = []
    for i, left in enumerate(capped):
        for right in capped[i + 1 :]:
            same_gt = bool(left.dominant_gt == right.dominant_gt)
            shuffled_right = shuffled_by_node_id.get(right.node_id, right)
            vc_q0, observer_count, vc_q0_max, _mean_q0 = _weighted_view_consensus(
                left,
                right,
                windows_by_index,
                quality_by_key,
                variant="Q0_no_filter",
                min_visible_carriers=int(min_visible_carriers),
                observer_frame_mode="all",
                near_endpoint_frame_gap=10,
            )
            vc_q, q_observer_count, vc_q_max, mean_q = _weighted_view_consensus(
                left,
                right,
                windows_by_index,
                quality_by_key,
                variant=quality_variant,
                min_visible_carriers=int(min_visible_carriers),
                observer_frame_mode="all",
                near_endpoint_frame_gap=10,
            )
            shuffled_vc_q, shuffled_observer_count, _shuffled_vc_q_max, shuffled_mean_q = _weighted_view_consensus(
                left,
                shuffled_right,
                windows_by_index,
                quality_by_key,
                variant=quality_variant,
                min_visible_carriers=int(min_visible_carriers),
                observer_frame_mode="all",
                near_endpoint_frame_gap=10,
            )
            temporal, temporal_window_count, temporal_mean = _temporal_containment(
                left,
                right,
                frame_rank_by_id=frame_rank_by_id,
                max_observation_gap=int(max_temporal_observation_gap),
            )
            shuffled_temporal, shuffled_temporal_window_count, _shuffled_temporal_mean = _temporal_containment(
                left,
                shuffled_right,
                frame_rank_by_id=frame_rank_by_id,
                max_observation_gap=int(max_temporal_observation_gap),
            )
            left_to_right_outside, left_to_right_count, left_to_right_coverage = _directed_visible_outside(
                left,
                right,
                windows_by_index,
                min_visible_carriers=int(min_visible_carriers),
            )
            right_to_left_outside, right_to_left_count, right_to_left_coverage = _directed_visible_outside(
                right,
                left,
                windows_by_index,
                min_visible_carriers=int(min_visible_carriers),
            )
            outside_values = [value for value in [left_to_right_outside, right_to_left_outside] if value is not None]
            symmetric_visible_outside = float(min(outside_values)) if len(outside_values) == 2 else None
            visible_outside_hard_negative = bool(
                symmetric_visible_outside is not None and symmetric_visible_outside >= float(visible_outside_threshold)
            )
            tw = float(temporal_weight)
            p4_q = (1.0 - tw) * float(vc_q) + tw * float(temporal)
            p4_q0 = (1.0 - tw) * float(vc_q0) + tw * float(temporal)
            p4_shuffled = (1.0 - tw) * float(shuffled_vc_q) + tw * float(shuffled_temporal)
            p4_q_visible_veto = 0.0 if visible_outside_hard_negative else p4_q
            p4_q0_visible_veto = 0.0 if visible_outside_hard_negative else p4_q0
            edge_rows.append(
                {
                    "scene": scene,
                    "left_node_id": left.node_id,
                    "right_node_id": right.node_id,
                    "left_frame_id": left.frame_id,
                    "right_frame_id": right.frame_id,
                    "frame_gap": abs(int(left.frame_id) - int(right.frame_id)),
                    "left_frame_rank": frame_rank_by_id.get(int(left.frame_id)),
                    "right_frame_rank": frame_rank_by_id.get(int(right.frame_id)),
                    "observation_frame_gap": None
                    if frame_rank_by_id.get(int(left.frame_id)) is None or frame_rank_by_id.get(int(right.frame_id)) is None
                    else abs(int(frame_rank_by_id[int(left.frame_id)]) - int(frame_rank_by_id[int(right.frame_id)])),
                    "left_mask_id": left.mask_id,
                    "right_mask_id": right.mask_id,
                    "left_support_count": left.support_count,
                    "right_support_count": right.support_count,
                    "left_gt": left.dominant_gt,
                    "right_gt": right.dominant_gt,
                    "diagnostic_same_gt": same_gt,
                    "shared_carrier_jaccard": _shared_jaccard(left, right),
                    "P1_adjacent_temporal": temporal,
                    "temporal_window_count": temporal_window_count,
                    "temporal_window_mean": temporal_mean,
                    "P2_raw_view_consensus": vc_q0,
                    "P2_raw_view_consensus_max_observer": vc_q0_max,
                    "P2_observer_count": observer_count,
                    "P3_view_consensus_q": vc_q,
                    "P3_view_consensus_q_max_observer": vc_q_max,
                    "P3_observer_count": q_observer_count,
                    "P3_mean_supporter_q_used": mean_q,
                    "P4_vc_q_temporal": p4_q,
                    "P4_vc_q0_temporal": p4_q0,
                    "P4_vc_q_temporal_visible_veto": p4_q_visible_veto,
                    "P4_vc_q0_temporal_visible_veto": p4_q0_visible_veto,
                    "P4_shuffled_vc_q_temporal": p4_shuffled,
                    "P4_shuffled_observer_count": shuffled_observer_count,
                    "P4_shuffled_mean_supporter_q_used": shuffled_mean_q,
                    "P4_shuffled_temporal": shuffled_temporal,
                    "P4_shuffled_temporal_window_count": shuffled_temporal_window_count,
                    "left_to_right_visible_outside": left_to_right_outside,
                    "right_to_left_visible_outside": right_to_left_outside,
                    "left_to_right_visible_sample_count": left_to_right_count,
                    "right_to_left_visible_sample_count": right_to_left_count,
                    "left_to_right_target_mask_coverage": left_to_right_coverage,
                    "right_to_left_target_mask_coverage": right_to_left_coverage,
                    "symmetric_visible_outside": symmetric_visible_outside,
                    "visible_outside_threshold": float(visible_outside_threshold),
                    "visible_outside_hard_negative": visible_outside_hard_negative,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return edge_rows, quality_rows


def _summarize_edges(
    *,
    scene: str,
    edge_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    quality_variant: str,
) -> list[dict[str, Any]]:
    labels = [bool(row["diagnostic_same_gt"]) for row in edge_rows]
    metric_keys = [
        "shared_carrier_jaccard",
        "P1_adjacent_temporal",
        "P2_raw_view_consensus",
        "P3_view_consensus_q",
        "P4_vc_q_temporal",
        "P4_vc_q0_temporal",
        "P4_vc_q_temporal_visible_veto",
        "P4_vc_q0_temporal_visible_veto",
        "P4_shuffled_vc_q_temporal",
    ]
    by_key: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        scores = [float(row.get(key) or 0.0) for row in edge_rows]
        by_key[key] = {
            "scene": scene,
            "variant": key,
            "quality_variant": quality_variant,
            "edge_count": len(edge_rows),
            "edge_same_gt_AUC": _rank_auc(labels, scores),
            "edge_precision@top1k": _precision_at_k(edge_rows, key, 1000),
            "edge_precision@top5k": _precision_at_k(edge_rows, key, 5000),
            "score_mean": _safe_mean(scores),
            "score_p90": _safe_quantile(scores, 0.90),
            "mean_observer_count": _safe_mean(row.get("P3_observer_count") for row in edge_rows),
            "median_observer_count": _safe_median(row.get("P3_observer_count") for row in edge_rows),
            "temporal_edge_density": float(sum(1 for row in edge_rows if float(row.get("P1_adjacent_temporal") or 0.0) > 0.0) / max(len(edge_rows), 1)),
            "temporal_nonzero_edge_count": int(sum(1 for row in edge_rows if float(row.get("P1_adjacent_temporal") or 0.0) > 0.0)),
            "supporter_reliability_mean": _safe_mean(row.get(quality_variant) for row in quality_rows),
            "supporter_reliability_p10": _safe_quantile([row.get(quality_variant) for row in quality_rows], 0.10),
            "visible_outside_hard_negative_count": int(sum(1 for row in edge_rows if bool(row.get("visible_outside_hard_negative")))),
            "visible_outside_hard_negative_precision": _negative_precision(edge_rows, "visible_outside_hard_negative"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    shared = by_key["shared_carrier_jaccard"]
    p3 = by_key["P3_view_consensus_q"]
    shuffled = by_key["P4_shuffled_vc_q_temporal"]
    for key in [
        "P3_view_consensus_q",
        "P4_vc_q_temporal",
        "P4_vc_q0_temporal",
        "P4_vc_q_temporal_visible_veto",
        "P4_vc_q0_temporal_visible_veto",
    ]:
        row = by_key[key]
        auc = row.get("edge_same_gt_AUC")
        p5 = row.get("edge_precision@top5k")
        row["real_minus_shared_edge_AUC"] = None if auc is None or shared.get("edge_same_gt_AUC") is None else float(auc - shared["edge_same_gt_AUC"])
        row["precision_top5k_minus_shared"] = None if p5 is None or shared.get("edge_precision@top5k") is None else float(p5 - shared["edge_precision@top5k"])
        row["real_minus_shuffled_edge_AUC"] = (
            None if auc is None or shuffled.get("edge_same_gt_AUC") is None else float(auc - shuffled["edge_same_gt_AUC"])
        )
        row["real_minus_no_temporal_edge_AUC"] = None if auc is None or p3.get("edge_same_gt_AUC") is None else float(auc - p3["edge_same_gt_AUC"])
        is_temporal_variant = key.startswith("P4")
        row["gate_pass"] = bool(
            is_temporal_variant
            and row["real_minus_shared_edge_AUC"] is not None
            and row["real_minus_shared_edge_AUC"] >= 0.08
            and row["precision_top5k_minus_shared"] is not None
            and row["precision_top5k_minus_shared"] >= 0.10
            and row["real_minus_shuffled_edge_AUC"] is not None
            and row["real_minus_shuffled_edge_AUC"] >= 0.10
            and row["real_minus_no_temporal_edge_AUC"] is not None
            and row["real_minus_no_temporal_edge_AUC"] >= 0.08
        )
    return list(by_key.values())


def _negative_precision(edge_rows: list[dict[str, Any]], flag_key: str) -> float | None:
    flagged = [row for row in edge_rows if bool(row.get(flag_key))]
    if not flagged:
        return None
    return float(sum(1 for row in flagged if row.get("diagnostic_same_gt") is False) / len(flagged))


def _scene_payload(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_threshold: float,
    confidence_threshold: float,
    min_mask_area: int,
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
    max_temporal_frame_gap: int,
    temporal_weight: float,
    visible_outside_threshold: float,
    quality_variant: str,
) -> dict[str, Any]:
    windows, window_rows, manifest_diag = _load_scene_windows(
        scene=scene,
        carrier_cache_root=carrier_cache_root,
        visibility_threshold=float(visibility_threshold),
        confidence_threshold=float(confidence_threshold),
        min_mask_area=int(min_mask_area),
    )
    nodes, frame_rows, node_diag = _build_nodes(scene, windows, min_mask_area=int(min_mask_area))
    frame_rank_by_id = _frame_rank_by_id(windows)
    edge_rows, quality_rows = _edge_rows_for_scene(
        scene=scene,
        nodes=nodes,
        windows=windows,
        frame_rank_by_id=frame_rank_by_id,
        quality_variant=quality_variant,
        max_edge_nodes=int(max_edge_nodes),
        min_node_carriers=int(min_node_carriers),
        min_visible_carriers=int(min_visible_carriers),
        max_temporal_observation_gap=int(max_temporal_frame_gap),
        temporal_weight=float(temporal_weight),
        visible_outside_threshold=float(visible_outside_threshold),
    )
    summary_rows = _summarize_edges(
        scene=scene,
        edge_rows=edge_rows,
        quality_rows=quality_rows,
        quality_variant=quality_variant,
    )
    return {
        "edge_rows": edge_rows,
        "quality_rows": quality_rows,
        "summary_rows": summary_rows,
        "frame_rows": frame_rows,
        "window_rows": window_rows,
        "diag": {**manifest_diag, **node_diag},
    }


def _frame_rank_by_id(windows: list[WindowTrace]) -> dict[int, int]:
    frame_ids = sorted({int(frame_id) for window in windows for frame_id in window.frame_ids})
    return {frame_id: rank for rank, frame_id in enumerate(frame_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-cache P1/P4 adjacent temporal positive-edge repair for v46.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--max-edge-nodes", type=int, default=120)
    parser.add_argument("--min-node-carriers", type=int, default=5)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument(
        "--max-temporal-frame-gap",
        type=int,
        default=2,
        help="Maximum gap in sorted observation-frame rank, not raw ScanNet frame-id difference.",
    )
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--visible-outside-threshold", type=float, default=0.80)
    parser.add_argument("--quality-variant", default="Q5_split_outside_fragment_soft")
    parser.add_argument("--output-root", default="outputs/audit/v46_temporal_positive_edge_repair")
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    all_edge_rows: list[dict[str, Any]] = []
    all_quality_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    diags: dict[str, Any] = {}
    for scene in scenes:
        payload = _scene_payload(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_threshold=float(args.visibility_threshold),
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            max_edge_nodes=int(args.max_edge_nodes),
            min_node_carriers=int(args.min_node_carriers),
            min_visible_carriers=int(args.min_visible_carriers),
            max_temporal_frame_gap=int(args.max_temporal_frame_gap),
            temporal_weight=float(args.temporal_weight),
            visible_outside_threshold=float(args.visible_outside_threshold),
            quality_variant=str(args.quality_variant),
        )
        all_edge_rows.extend(payload["edge_rows"])
        all_quality_rows.extend(payload["quality_rows"])
        all_summary_rows.extend(payload["summary_rows"])
        all_frame_rows.extend(payload["frame_rows"])
        all_window_rows.extend(payload["window_rows"])
        diags[scene] = payload["diag"]
    p4_rows = [row for row in all_summary_rows if str(row.get("variant")) in {"P4_vc_q_temporal", "P4_vc_q0_temporal"}]
    gate = {
        "any_scene_variant_gate_pass": any(bool(row.get("gate_pass")) for row in p4_rows),
        "all_scene_variant_gate_pass": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    for variant in [
        "P4_vc_q_temporal",
        "P4_vc_q0_temporal",
        "P4_vc_q_temporal_visible_veto",
        "P4_vc_q0_temporal_visible_veto",
    ]:
        selected = [row for row in p4_rows if str(row.get("variant")) == variant]
        if selected and all(bool(row.get("gate_pass")) for row in selected):
            gate["all_scene_variant_gate_pass"] = True
    gate["pass"] = bool(gate["all_scene_variant_gate_pass"])
    payload = {
        "phase": "v46_temporal_positive_edge_repair",
        "created_at": _utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "visibility_threshold": float(args.visibility_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "max_edge_nodes": int(args.max_edge_nodes),
        "min_node_carriers": int(args.min_node_carriers),
        "min_visible_carriers": int(args.min_visible_carriers),
        "max_temporal_frame_gap": int(args.max_temporal_frame_gap),
        "max_temporal_gap_uses_observation_rank": True,
        "temporal_weight": float(args.temporal_weight),
        "visible_outside_threshold": float(args.visible_outside_threshold),
        "quality_variant": str(args.quality_variant),
        "summary_rows": all_summary_rows,
        "diag": diags,
        "gate": gate,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "temporal_positive_edge_repair.json", payload)
    _write_csv(out / "temporal_positive_edge_rows.csv", all_edge_rows)
    _write_csv(out / "temporal_positive_summary_rows.csv", all_summary_rows)
    _write_csv(out / "temporal_positive_quality_rows.csv", all_quality_rows)
    _write_csv(out / "temporal_positive_frame_rows.csv", all_frame_rows)
    _write_csv(out / "temporal_positive_window_rows.csv", all_window_rows)
    print(json.dumps({"summary": str(out / "temporal_positive_edge_repair.json"), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
