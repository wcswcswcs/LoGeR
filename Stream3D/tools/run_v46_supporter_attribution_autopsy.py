from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import (
    ROOT,
    MaskNode,
    WindowTrace,
    _build_nodes,
    _json_safe,
    _load_scene_windows,
    _positive_label_counts,
)
from tools.run_v46_supporter_quality_raw_repair import _supporter_quality_rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_node_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    return {int(float(row["node_id"])): row for row in _read_csv(path)}


def _frame_rank_by_id(windows: list[WindowTrace]) -> dict[int, int]:
    frame_ids = sorted({int(frame_id) for window in windows for frame_id in window.frame_ids})
    return {frame_id: rank for rank, frame_id in enumerate(frame_ids)}


def _select_edges(path: Path, scene: str, max_edges: int, edge_group: str) -> list[dict[str, Any]]:
    rows = [row for row in _read_csv(path) if str(row.get("scene")) == str(scene)]
    selected = rows[: max(0, int(max_edges))]
    for rank, row in enumerate(selected, start=1):
        row["_edge_group"] = edge_group
        row["_edge_rank"] = rank
    return selected


def _node_by_id(nodes: list[MaskNode], node_map: dict[int, dict[str, Any]]) -> dict[int, MaskNode]:
    out = {int(node.node_id): node for node in nodes}
    for node_id, row in node_map.items():
        node = out.get(int(node_id))
        if node is None:
            continue
        if int(float(row.get("frame_id", node.frame_id))) != int(node.frame_id):
            raise ValueError(f"node_id={node_id} frame mismatch: rebuilt={node.frame_id} csv={row.get('frame_id')}")
        if int(float(row.get("mask_id", node.mask_id))) != int(node.mask_id):
            raise ValueError(f"node_id={node_id} mask mismatch: rebuilt={node.mask_id} csv={row.get('mask_id')}")
    return out


def _supporter_quality(quality_by_key: dict[tuple[int, int], dict[str, Any]], frame_id: int, mask_id: int) -> dict[str, Any]:
    return quality_by_key.get((int(frame_id), int(mask_id)), {})


def _endpoint_flags(observer_frame_id: int, left: MaskNode, right: MaskNode, frame_rank: dict[int, int]) -> dict[str, Any]:
    left_rank = frame_rank.get(int(left.frame_id))
    right_rank = frame_rank.get(int(right.frame_id))
    obs_rank = frame_rank.get(int(observer_frame_id))
    rank_gap_left = None if left_rank is None or obs_rank is None else abs(int(obs_rank) - int(left_rank))
    rank_gap_right = None if right_rank is None or obs_rank is None else abs(int(obs_rank) - int(right_rank))
    raw_gap_left = abs(int(observer_frame_id) - int(left.frame_id))
    raw_gap_right = abs(int(observer_frame_id) - int(right.frame_id))
    return {
        "observer_is_left_endpoint": bool(int(observer_frame_id) == int(left.frame_id)),
        "observer_is_right_endpoint": bool(int(observer_frame_id) == int(right.frame_id)),
        "observer_rank_gap_to_left": rank_gap_left,
        "observer_rank_gap_to_right": rank_gap_right,
        "observer_raw_frame_gap_to_left": raw_gap_left,
        "observer_raw_frame_gap_to_right": raw_gap_right,
        "observer_near_endpoint_rank_gap_le1": bool(
            (rank_gap_left is not None and rank_gap_left <= 1) or (rank_gap_right is not None and rank_gap_right <= 1)
        ),
        "observer_near_endpoint_raw_gap_le10": bool(raw_gap_left <= 10 or raw_gap_right <= 10),
    }


def _observer_rows_for_edge(
    *,
    edge: dict[str, Any],
    left: MaskNode,
    right: MaskNode,
    windows_by_index: dict[int, WindowTrace],
    quality_by_key: dict[tuple[int, int], dict[str, Any]],
    quality_variant: str,
    min_visible_carriers: int,
    frame_rank: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_windows = sorted(set(left.inc_by_window) & set(right.inc_by_window))
    for window_index in common_windows:
        window = windows_by_index[window_index]
        left_idx = np.asarray(sorted(left.inc_by_window[window_index]), dtype=np.int64)
        right_idx = np.asarray(sorted(right.inc_by_window[window_index]), dtype=np.int64)
        if left_idx.size == 0 or right_idx.size == 0:
            continue
        for local_index, frame_id in enumerate(window.frame_ids):
            labels_at_carrier = window.labels_by_frame.get(int(frame_id))
            if labels_at_carrier is None:
                continue
            left_visible = left_idx[window.visible[local_index, left_idx]]
            right_visible = right_idx[window.visible[local_index, right_idx]]
            if left_visible.size < int(min_visible_carriers) or right_visible.size < int(min_visible_carriers):
                continue
            left_counts = _positive_label_counts(labels_at_carrier[left_visible])
            right_counts = _positive_label_counts(labels_at_carrier[right_visible])
            common_labels = sorted(set(left_counts) & set(right_counts))
            best_label = 0
            best_raw = 0.0
            best_q = 0.0
            best_q_score = 0.0
            for label in common_labels:
                raw_score = min(left_counts[label] / left_visible.size, right_counts[label] / right_visible.size)
                quality = _supporter_quality(quality_by_key, int(frame_id), int(label))
                q = float(quality.get(quality_variant, 0.0))
                q_score = float(q * raw_score)
                if q_score > best_q_score:
                    best_label = int(label)
                    best_raw = float(raw_score)
                    best_q = float(q)
                    best_q_score = float(q_score)
            quality = _supporter_quality(quality_by_key, int(frame_id), int(best_label)) if best_label > 0 else {}
            rows.append(
                {
                    "edge_group": edge.get("_edge_group"),
                    "edge_rank": int(edge.get("_edge_rank", 0)),
                    "scene": edge.get("scene"),
                    "left_node_id": left.node_id,
                    "right_node_id": right.node_id,
                    "left_frame_id": left.frame_id,
                    "right_frame_id": right.frame_id,
                    "left_mask_id": left.mask_id,
                    "right_mask_id": right.mask_id,
                    "left_gt": left.dominant_gt,
                    "right_gt": right.dominant_gt,
                    "diagnostic_same_gt": bool(left.dominant_gt == right.dominant_gt),
                    "edge_P5_p4_semantic_boost_capped": _parse_float(edge.get("P5_p4_semantic_boost_capped")),
                    "edge_P6_feature_only": _parse_float(edge.get("P6_feature_only")),
                    "window_index": int(window_index),
                    "observer_frame_id": int(frame_id),
                    **_endpoint_flags(int(frame_id), left, right, frame_rank),
                    "left_visible_carrier_count": int(left_visible.size),
                    "right_visible_carrier_count": int(right_visible.size),
                    "common_supporter_label_count": int(len(common_labels)),
                    "best_supporter_mask_id": int(best_label),
                    "best_supporter_raw_score": float(best_raw),
                    "best_supporter_q": float(best_q),
                    "best_supporter_q_score": float(best_q_score),
                    "best_supporter_gt": quality.get("diagnostic_gt_instance"),
                    "best_supporter_gt_purity": quality.get("diagnostic_gt_purity"),
                    "best_supporter_split_entropy": quality.get("split_entropy"),
                    "best_supporter_visible_outside": quality.get("visible_outside"),
                    "best_supporter_fragmentation_rate": quality.get("fragmentation_rate"),
                    "best_supporter_Q5_soft": quality.get("Q5_split_outside_fragment_soft"),
                    "best_supporter_Q5_threshold_055": quality.get("Q5_threshold_055"),
                    "best_supporter_Q5_threshold_070": quality.get("Q5_threshold_070"),
                    "best_supporter_gt_matches_left": bool(quality.get("diagnostic_gt_instance") == left.dominant_gt),
                    "best_supporter_gt_matches_right": bool(quality.get("diagnostic_gt_instance") == right.dominant_gt),
                    "best_supporter_gt_matches_both": bool(
                        quality.get("diagnostic_gt_instance") == left.dominant_gt
                        and quality.get("diagnostic_gt_instance") == right.dominant_gt
                    ),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return rows


def _mean(values: list[float]) -> float | None:
    nums = [float(value) for value in values if np.isfinite(float(value))]
    return None if not nums else float(np.mean(nums))


def _edge_summary(rows: list[dict[str, Any]], edge: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            "edge_group": edge.get("_edge_group"),
            "edge_rank": int(edge.get("_edge_rank", 0)),
            "scene": edge.get("scene"),
            "left_node_id": edge.get("left_node_id"),
            "right_node_id": edge.get("right_node_id"),
            "observer_count": 0,
        }
    top = max(rows, key=lambda row: float(row.get("best_supporter_q_score") or 0.0))
    labels = [int(row["best_supporter_mask_id"]) for row in rows if int(row.get("best_supporter_mask_id") or 0) > 0]
    top_label_count = Counter(labels).most_common(1)[0] if labels else (0, 0)
    endpoint_count = sum(1 for row in rows if row.get("observer_is_left_endpoint") or row.get("observer_is_right_endpoint"))
    near_rank_count = sum(1 for row in rows if row.get("observer_near_endpoint_rank_gap_le1"))
    near_raw_count = sum(1 for row in rows if row.get("observer_near_endpoint_raw_gap_le10"))
    left_match_count = sum(1 for row in rows if row.get("best_supporter_gt_matches_left"))
    right_match_count = sum(1 for row in rows if row.get("best_supporter_gt_matches_right"))
    both_match_count = sum(1 for row in rows if row.get("best_supporter_gt_matches_both"))
    q_scores = [float(row.get("best_supporter_q_score") or 0.0) for row in rows]
    raw_scores = [float(row.get("best_supporter_raw_score") or 0.0) for row in rows]
    return {
        "edge_group": edge.get("_edge_group"),
        "edge_rank": int(edge.get("_edge_rank", 0)),
        "scene": top.get("scene"),
        "left_node_id": top.get("left_node_id"),
        "right_node_id": top.get("right_node_id"),
        "left_gt": top.get("left_gt"),
        "right_gt": top.get("right_gt"),
        "diagnostic_same_gt": top.get("diagnostic_same_gt"),
        "edge_P5_p4_semantic_boost_capped": top.get("edge_P5_p4_semantic_boost_capped"),
        "edge_P6_feature_only": top.get("edge_P6_feature_only"),
        "observer_count": len(rows),
        "best_q_score_mean": _mean(q_scores),
        "best_q_score_max": max(q_scores) if q_scores else None,
        "best_raw_score_mean": _mean(raw_scores),
        "best_raw_score_max": max(raw_scores) if raw_scores else None,
        "top_observer_frame_id": top.get("observer_frame_id"),
        "top_supporter_mask_id": top.get("best_supporter_mask_id"),
        "top_supporter_q": top.get("best_supporter_q"),
        "top_supporter_q_score": top.get("best_supporter_q_score"),
        "top_supporter_gt": top.get("best_supporter_gt"),
        "top_supporter_gt_purity": top.get("best_supporter_gt_purity"),
        "top_supporter_gt_matches_left": top.get("best_supporter_gt_matches_left"),
        "top_supporter_gt_matches_right": top.get("best_supporter_gt_matches_right"),
        "top_supporter_gt_matches_both": top.get("best_supporter_gt_matches_both"),
        "mode_supporter_mask_id": int(top_label_count[0]),
        "mode_supporter_fraction": float(top_label_count[1] / max(len(rows), 1)),
        "endpoint_observer_fraction": float(endpoint_count / max(len(rows), 1)),
        "near_endpoint_rank_le1_fraction": float(near_rank_count / max(len(rows), 1)),
        "near_endpoint_raw_le10_fraction": float(near_raw_count / max(len(rows), 1)),
        "supporter_gt_matches_left_fraction": float(left_match_count / max(len(rows), 1)),
        "supporter_gt_matches_right_fraction": float(right_match_count / max(len(rows), 1)),
        "supporter_gt_matches_both_fraction": float(both_match_count / max(len(rows), 1)),
    }


def _group_summary(edge_rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    rows = [row for row in edge_rows if row.get("edge_group") == group]
    if not rows:
        return {"edge_group": group, "edge_count": 0}
    return {
        "edge_group": group,
        "edge_count": len(rows),
        "observer_count_mean": _mean([float(row.get("observer_count") or 0.0) for row in rows]),
        "best_q_score_mean": _mean([float(row.get("best_q_score_mean") or 0.0) for row in rows]),
        "best_q_score_max_mean": _mean([float(row.get("best_q_score_max") or 0.0) for row in rows]),
        "endpoint_observer_fraction_mean": _mean([float(row.get("endpoint_observer_fraction") or 0.0) for row in rows]),
        "near_endpoint_rank_le1_fraction_mean": _mean([float(row.get("near_endpoint_rank_le1_fraction") or 0.0) for row in rows]),
        "supporter_gt_matches_left_fraction_mean": _mean([float(row.get("supporter_gt_matches_left_fraction") or 0.0) for row in rows]),
        "supporter_gt_matches_right_fraction_mean": _mean([float(row.get("supporter_gt_matches_right_fraction") or 0.0) for row in rows]),
        "supporter_gt_matches_both_fraction_mean": _mean([float(row.get("supporter_gt_matches_both_fraction") or 0.0) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute v46 view-consensus edge scores to concrete observer masks.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--false-positive-csv", required=True)
    parser.add_argument("--false-negative-csv", required=True)
    parser.add_argument("--node-rows-csv", default="")
    parser.add_argument("--max-edges-per-group", type=int, default=20)
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--underseg-purity-threshold", type=float, default=0.70)
    parser.add_argument("--low-q-threshold", type=float, default=0.55)
    parser.add_argument("--quality-variant", default="Q5_split_outside_fragment_soft")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    windows, _window_rows, manifest_diag = _load_scene_windows(
        scene=str(args.scene),
        carrier_cache_root=carrier_cache_root,
        visibility_threshold=float(args.visibility_threshold),
        confidence_threshold=float(args.confidence_threshold),
        min_mask_area=int(args.min_mask_area),
    )
    nodes, _frame_rows, node_diag = _build_nodes(str(args.scene), windows, min_mask_area=int(args.min_mask_area))
    windows_by_index = {window.window_index: window for window in windows}
    frame_rank = _frame_rank_by_id(windows)
    quality_by_key, quality_rows = _supporter_quality_rows(
        scene=str(args.scene),
        nodes=nodes,
        windows_by_index=windows_by_index,
        min_visible_carriers=int(args.min_visible_carriers),
        underseg_purity_threshold=float(args.underseg_purity_threshold),
        low_q_threshold=float(args.low_q_threshold),
    )
    node_map = _load_node_map(Path(args.node_rows_csv) if str(args.node_rows_csv).strip() else None)
    nodes_by_id = _node_by_id(nodes, node_map)
    selected_edges = _select_edges(Path(args.false_positive_csv), str(args.scene), int(args.max_edges_per_group), "top_false_positive")
    selected_edges.extend(
        _select_edges(Path(args.false_negative_csv), str(args.scene), int(args.max_edges_per_group), "top_false_negative")
    )

    observer_rows: list[dict[str, Any]] = []
    edge_summary_rows: list[dict[str, Any]] = []
    for edge in selected_edges:
        left = nodes_by_id[int(float(edge["left_node_id"]))]
        right = nodes_by_id[int(float(edge["right_node_id"]))]
        rows = _observer_rows_for_edge(
            edge=edge,
            left=left,
            right=right,
            windows_by_index=windows_by_index,
            quality_by_key=quality_by_key,
            quality_variant=str(args.quality_variant),
            min_visible_carriers=int(args.min_visible_carriers),
            frame_rank=frame_rank,
        )
        observer_rows.extend(rows)
        edge_summary_rows.append(_edge_summary(rows, edge))

    group_summary_rows = [
        _group_summary(edge_summary_rows, "top_false_positive"),
        _group_summary(edge_summary_rows, "top_false_negative"),
    ]
    payload = {
        "phase": "v46_supporter_attribution_autopsy",
        "created_at": _utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scene": str(args.scene),
        "false_positive_csv": str(args.false_positive_csv),
        "false_negative_csv": str(args.false_negative_csv),
        "node_rows_csv": str(args.node_rows_csv),
        "max_edges_per_group": int(args.max_edges_per_group),
        "visibility_threshold": float(args.visibility_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "min_visible_carriers": int(args.min_visible_carriers),
        "quality_variant": str(args.quality_variant),
        "edge_count": len(edge_summary_rows),
        "observer_row_count": len(observer_rows),
        "group_summary_rows": group_summary_rows,
        "diag": {**manifest_diag, **node_diag, "quality_row_count": len(quality_rows)},
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "supporter_attribution_autopsy.json", payload)
    _write_csv(out / "supporter_observer_rows.csv", observer_rows)
    _write_csv(out / "supporter_edge_summary_rows.csv", edge_summary_rows)
    _write_csv(out / "supporter_group_summary_rows.csv", group_summary_rows)
    print(json.dumps({"summary": str(out / "supporter_attribution_autopsy.json"), "group_summary_rows": group_summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
