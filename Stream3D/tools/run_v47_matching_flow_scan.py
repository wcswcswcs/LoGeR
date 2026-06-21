from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from stream4d_native.v47_common import (
    ROOT,
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    safe_quantile,
    write_csv,
    write_json,
)


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_score_keys(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _node_index(mask_rows: list[dict[str, Any]]) -> tuple[list[int], dict[int, int]]:
    node_ids = [parse_int(row.get("node_id")) for row in mask_rows]
    return node_ids, {node_id: idx for idx, node_id in enumerate(node_ids)}


def _select_matching_edges(
    *,
    edge_rows: list[dict[str, Any]],
    node_to_index: dict[int, int],
    score_key: str,
    min_score: float,
    edge_types: set[str],
    respect_edge_accept_candidate: bool,
    max_visible_outside: float,
    min_visible_carriers: int,
) -> list[dict[str, Any]]:
    n = len(node_to_index)
    cost = np.zeros((n, n), dtype=np.float32)
    best: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for row in edge_rows:
        if str(row.get("edge_type")) not in edge_types:
            continue
        if respect_edge_accept_candidate and not parse_bool(row.get("edge_accept_candidate", True)):
            continue
        if parse_float(row.get("visible_outside"), 1.0) > float(max_visible_outside):
            continue
        if parse_int(row.get("forward_visible_carrier_count")) < int(min_visible_carriers):
            continue
        if parse_int(row.get("backward_visible_carrier_count")) < int(min_visible_carriers):
            continue
        score = parse_float(row.get(score_key))
        if score < float(min_score):
            continue
        src_idx = node_to_index.get(parse_int(row.get("src_node_id")))
        dst_idx = node_to_index.get(parse_int(row.get("dst_node_id")))
        if src_idx is None or dst_idx is None or src_idx == dst_idx:
            continue
        key = (src_idx, dst_idx)
        if key not in best or score > best[key][0]:
            best[key] = (score, row)
            cost[src_idx, dst_idx] = -float(score)

    row_ind, col_ind = linear_sum_assignment(cost)
    selected: list[dict[str, Any]] = []
    for src_idx, dst_idx in zip(row_ind.tolist(), col_ind.tolist()):
        item = best.get((int(src_idx), int(dst_idx)))
        if item is not None:
            selected.append(dict(item[1], matching_flow_score=float(item[0]), selected_for_matching_flow=True))
    return selected


def _evaluate(mask_rows: list[dict[str, Any]], selected_edges: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = [parse_int(row.get("node_id")) for row in mask_rows]
    uf = UnionFind(node_ids)
    for row in selected_edges:
        uf.union(parse_int(row.get("src_node_id")), parse_int(row.get("dst_node_id")))

    labels_by_root: dict[int, str] = {}
    pred_labels: list[str] = []
    true_labels: list[str] = []
    node_rows: list[dict[str, Any]] = []
    for row in mask_rows:
        node_id = parse_int(row.get("node_id"))
        root = uf.find(node_id)
        if root not in labels_by_root:
            labels_by_root[root] = f"f{len(labels_by_root):05d}"
        pred = labels_by_root[root]
        gt = str(row.get("diagnostic_gt_instance", ""))
        node_rows.append(
            {
                "node_id": node_id,
                "matching_flow_track_id": pred,
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "mask_id": row.get("mask_id"),
                "diagnostic_gt_instance": gt,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)

    track_sizes = Counter(row["matching_flow_track_id"] for row in node_rows)
    track_frames: dict[str, set[int]] = defaultdict(set)
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    for row in node_rows:
        track_frames[str(row["matching_flow_track_id"])].add(parse_int(row.get("frame_id")))
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            scene_true[str(row.get("scene"))].append(gt)
            scene_pred[str(row.get("scene"))].append(str(row["matching_flow_track_id"]))

    return {
        "track_rows": node_rows,
        "metrics": {
            "selected_edge_count": len(selected_edges),
            "track_count": len(track_sizes),
            "temporal_span_mean": safe_mean(len(frames) for frames in track_frames.values()),
            "track_length_mean": safe_mean(track_sizes.values()),
            "track_length_p50": safe_quantile(track_sizes.values(), 0.50),
            "track_length_p90": safe_quantile(track_sizes.values(), 0.90),
            "ARI": adjusted_rand_score(true_labels, pred_labels),
            "purity": cluster_purity(true_labels, pred_labels),
            "completeness": cluster_completeness(true_labels, pred_labels),
            "scene0081_ARI": adjusted_rand_score(scene_true["scene0081_01"], scene_pred["scene0081_01"])
            if scene_true.get("scene0081_01")
            else None,
            "scene0011_purity": cluster_purity(scene_true["scene0011_00"], scene_pred["scene0011_00"])
            if scene_true.get("scene0011_00")
            else None,
            "scene0050_purity": cluster_purity(scene_true["scene0050_00"], scene_pred["scene0050_00"])
            if scene_true.get("scene0050_00")
            else None,
            "scene0591_completeness": cluster_completeness(scene_true["scene0591_00"], scene_pred["scene0591_00"])
            if scene_true.get("scene0591_00")
            else None,
            "birth_from_d4rt_tube_count": 0,
            "maskless_object_count": 0,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 global bipartite matching sparse-flow proxy scan.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--real-score-keys", default="A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto")
    parser.add_argument("--control-score-keys", default="A8_no_temporal_control,A7_shuffled_D4RT")
    parser.add_argument("--min-scores", default="0.97,0.90,0.80,0.70,0.50,0.30")
    parser.add_argument("--max-visible-outside-values", default="1.0")
    parser.add_argument("--min-visible-carrier-values", default="0")
    parser.add_argument("--respect-edge-accept-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default="outputs/audit/v47_matching_flow_gap2")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    edge_rows = read_csv(ROOT / str(args.edge_table))
    node_ids, node_to_index = _node_index(mask_rows)
    del node_ids
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    score_keys = _parse_score_keys(args.real_score_keys) + _parse_score_keys(args.control_score_keys)

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[str, float, float, int], list[dict[str, Any]]] = {}
    track_rows_by_signature: dict[tuple[str, float, float, int], list[dict[str, Any]]] = {}
    for score_key in score_keys:
        for min_score in _parse_float_list(args.min_scores):
            for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                for min_visible_carriers in [int(v) for v in _parse_float_list(args.min_visible_carrier_values)]:
                    selected = _select_matching_edges(
                        edge_rows=edge_rows,
                        node_to_index=node_to_index,
                        score_key=score_key,
                        min_score=min_score,
                        edge_types=edge_types,
                        respect_edge_accept_candidate=bool(args.respect_edge_accept_candidate),
                        max_visible_outside=max_visible_outside,
                        min_visible_carriers=min_visible_carriers,
                    )
                    evaluated = _evaluate(mask_rows, selected)
                    signature = (score_key, float(min_score), float(max_visible_outside), int(min_visible_carriers))
                    selected_by_signature[signature] = selected
                    track_rows_by_signature[signature] = evaluated["track_rows"]
                    metrics = evaluated["metrics"]
                    rows.append(
                        {
                            "score_key": score_key,
                            "min_score": float(min_score),
                            "edge_types": ",".join(sorted(edge_types)),
                            "max_visible_outside": float(max_visible_outside),
                            "min_visible_carriers": int(min_visible_carriers),
                            **metrics,
                        }
                    )

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in set(_parse_score_keys(args.real_score_keys))]
    no_temporal_rows = [row for row in rows if row["score_key"] == "A8_no_temporal_control"]
    shuffled_rows = [row for row in rows if row["score_key"] == "A7_shuffled_D4RT"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}

    def signature(row: dict[str, Any]) -> tuple[str, float, float, int]:
        return (
            str(row.get("score_key")),
            parse_float(row.get("min_score")),
            parse_float(row.get("max_visible_outside")),
            parse_int(row.get("min_visible_carriers")),
        )

    summary = {
        "phase": "v47_matching_flow_scan",
        "solver_note": "Global bipartite matching one-predecessor/one-successor sparse-flow proxy; no GT used for prediction.",
        "observation_root": str(ROOT / str(args.observation_root)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "edge_types": sorted(edge_types),
        "respect_edge_accept_candidate": bool(args.respect_edge_accept_candidate),
        "rows": len(rows),
        "best_row": rows[0] if rows else None,
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
        "best_real_minus_best_shuffled_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")),
        "gate": {
            "ARI_pass": bool(parse_float(best_real.get("ARI")) >= 0.465),
            "purity_pass": bool(parse_float(best_real.get("purity")) >= 0.865),
            "completeness_pass": bool(parse_float(best_real.get("completeness")) >= 0.535),
            "temporal_span_mean_pass": bool(parse_float(best_real.get("temporal_span_mean")) >= 1.70),
            "scene0081_ARI_pass": bool(parse_float(best_real.get("scene0081_ARI")) >= 0.270),
            "birth_from_d4rt_tube_count_pass": bool(parse_int(best_real.get("birth_from_d4rt_tube_count")) == 0),
            "maskless_object_count_pass": bool(parse_int(best_real.get("maskless_object_count")) == 0),
            "real_minus_no_temporal_pass": bool(
                parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")) >= 0.15
            ),
            "real_minus_shuffled_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")) >= 0.25),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "matching_flow_scan_rows.csv", rows)
    write_csv(out_root / "matching_flow_best_real_selected_edges.csv", selected_by_signature.get(signature(best_real), []))
    write_csv(out_root / "matching_flow_best_no_temporal_selected_edges.csv", selected_by_signature.get(signature(best_no_temporal), []))
    write_csv(out_root / "matching_flow_best_shuffled_selected_edges.csv", selected_by_signature.get(signature(best_shuffled), []))
    write_csv(out_root / "matching_flow_best_real_track_rows.csv", track_rows_by_signature.get(signature(best_real), []))
    write_json(out_root / "matching_flow_summary.json", summary)
    print({"summary": str(out_root / "matching_flow_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
