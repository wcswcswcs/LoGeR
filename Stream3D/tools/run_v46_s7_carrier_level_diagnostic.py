from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import (
    ROOT,
    WindowTrace,
    _build_nodes,
    _load_scene_windows,
)
from tools.run_v46_raw_signed_solver_diagnostic import (
    UnionFind,
    _adjusted_rand_score,
    _cluster_completeness,
    _cluster_purity,
    _pairwise_metrics,
    _write_csv,
    _write_json,
)


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos_count = sum(1 for label, _score in pairs if label)
    neg_count = len(pairs) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    pairs.sort(key=lambda item: item[1])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum_pos += avg_rank * sum(1 for label, _score in pairs[idx:end] if label)
        idx = end
    return float((rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    labeled = [row for row in rows if row.get("diagnostic_same_gt") is not None]
    ranked = sorted(labeled, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(labeled))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if row.get("diagnostic_same_gt") is True) / len(ranked))


def _node_by_frame_mask(nodes: list[Any]) -> dict[tuple[int, int], Any]:
    return {(int(node.frame_id), int(node.mask_id)): node for node in nodes}


def _select_carriers(nodes: list[Any], *, carriers_per_mask: int, max_carriers: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    ranked_nodes = sorted(nodes, key=lambda node: (int(node.support_count), int(node.area), -int(node.node_id)), reverse=True)
    for node in ranked_nodes:
        keys = sorted(node.carrier_keys)[: int(carriers_per_mask)]
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            selected.append(key)
            if len(selected) >= int(max_carriers):
                return selected
    return selected


def _carrier_diag_label(
    carrier_key: tuple[int, int],
    windows_by_index: dict[int, WindowTrace],
    node_lookup: dict[tuple[int, int], Any],
) -> tuple[int | None, float | None, int]:
    window_index, carrier_index = carrier_key
    window = windows_by_index[window_index]
    counts: Counter[int] = Counter()
    observed = 0
    for local_index, frame_id in enumerate(window.frame_ids):
        labels_at_carrier = window.labels_by_frame.get(int(frame_id))
        if labels_at_carrier is None or carrier_index >= labels_at_carrier.shape[0]:
            continue
        mask_id = int(labels_at_carrier[carrier_index])
        if mask_id <= 0:
            continue
        observed += 1
        node = node_lookup.get((int(frame_id), mask_id))
        if node is not None and node.dominant_gt is not None:
            counts[int(node.dominant_gt)] += 1
    if not counts:
        return None, None, observed
    label, count = counts.most_common(1)[0]
    return int(label), float(count / max(sum(counts.values()), 1)), observed


def _carrier_pair_scores(
    left_key: tuple[int, int],
    right_key: tuple[int, int],
    windows_by_index: dict[int, WindowTrace],
) -> tuple[float, float, int, int, int]:
    if left_key[0] != right_key[0]:
        return 0.0, 0.0, 0, 0, 0
    window = windows_by_index[left_key[0]]
    left_idx = int(left_key[1])
    right_idx = int(right_key[1])
    both_inside = 0
    same_mask = 0
    diff_mask = 0
    both_visible = 0
    for local_index, frame_id in enumerate(window.frame_ids):
        if not bool(window.visible[local_index, left_idx]) or not bool(window.visible[local_index, right_idx]):
            continue
        both_visible += 1
        labels_at_carrier = window.labels_by_frame.get(int(frame_id))
        if labels_at_carrier is None:
            continue
        left_mask = int(labels_at_carrier[left_idx])
        right_mask = int(labels_at_carrier[right_idx])
        if left_mask <= 0 or right_mask <= 0:
            continue
        both_inside += 1
        if left_mask == right_mask:
            same_mask += 1
        else:
            diff_mask += 1
    positive_score = float(same_mask / max(both_inside, 1))
    negative_score = float(diff_mask / max(both_inside, 1))
    return positive_score, negative_score, both_visible, both_inside, same_mask


def _would_violate(uf: UnionFind, left: int, right: int, hard_negative_pairs: set[tuple[int, int]]) -> bool:
    root_l = uf.find(left)
    root_r = uf.find(right)
    if root_l == root_r:
        return False
    for node_l in uf.members[root_l]:
        for node_r in uf.members[root_r]:
            if _pair_key(node_l, node_r) in hard_negative_pairs:
                return True
    return False


def _cluster_labels(uf: UnionFind, node_ids: list[int]) -> list[str]:
    root_to_label: dict[int, str] = {}
    labels: list[str] = []
    for node_id in node_ids:
        root = uf.find(node_id)
        if root not in root_to_label:
            root_to_label[root] = f"c{len(root_to_label):04d}"
        labels.append(root_to_label[root])
    return labels


def _solve_carrier_graph(
    *,
    scene: str,
    carrier_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    positive_threshold: float,
    negative_threshold: float,
    soft_negative_lambda: float,
    merge_margin: float,
) -> dict[str, Any]:
    node_ids = [int(row["carrier_node_id"]) for row in carrier_rows]
    gt_by_node = {int(row["carrier_node_id"]): row.get("diagnostic_gt") for row in carrier_rows}
    hard_negative_pairs = {
        _pair_key(int(row["left_carrier_node_id"]), int(row["right_carrier_node_id"]))
        for row in edge_rows
        if float(row["negative_score"]) >= float(negative_threshold)
    }
    candidate_edges = [row for row in edge_rows if float(row["positive_score"]) >= float(positive_threshold)]
    rows_out: list[dict[str, Any]] = []
    variants = [
        "S0_carrier_positive_cc",
        "S7_carrier_hard_veto_cc",
        "S7_carrier_soft_penalty_cc",
        "S7_carrier_soft_penalty_strong_hard_veto_cc",
    ]
    for variant in variants:
        uf = UnionFind(node_ids)
        accepted = 0
        rejected = 0
        margin_rejected = 0
        if variant == "S0_carrier_positive_cc":
            merge_edges = sorted(candidate_edges, key=lambda row: float(row["positive_score"]), reverse=True)
        elif variant == "S7_carrier_hard_veto_cc":
            merge_edges = sorted(candidate_edges, key=lambda row: float(row["positive_score"]), reverse=True)
        else:
            scored_edges: list[tuple[float, dict[str, Any]]] = []
            for row in candidate_edges:
                score = float(row["positive_score"]) - float(soft_negative_lambda) * float(row["negative_score"])
                if score < float(merge_margin):
                    margin_rejected += 1
                    continue
                scored_edges.append((score, row))
            scored_edges.sort(key=lambda item: item[0], reverse=True)
            merge_edges = [row for _score, row in scored_edges]
        for row in merge_edges:
            left = int(row["left_carrier_node_id"])
            right = int(row["right_carrier_node_id"])
            use_hard_veto = variant in {
                "S7_carrier_hard_veto_cc",
                "S7_carrier_soft_penalty_strong_hard_veto_cc",
            }
            if use_hard_veto and _pair_key(left, right) in hard_negative_pairs:
                rejected += 1
                continue
            if use_hard_veto and _would_violate(uf, left, right, hard_negative_pairs):
                rejected += 1
                continue
            if uf.union(left, right):
                accepted += 1
        labeled_ids = [node_id for node_id in node_ids if gt_by_node.get(node_id) is not None]
        true_labels = [str(gt_by_node[node_id]) for node_id in labeled_ids]
        pred_all = _cluster_labels(uf, node_ids)
        pred_by_node = {node_id: pred_all[index] for index, node_id in enumerate(node_ids)}
        pred_labels = [pred_by_node[node_id] for node_id in labeled_ids]
        pairwise_precision, pairwise_recall = _pairwise_metrics(true_labels, pred_labels)
        hard_negative_violation_count = sum(1 for left, right in hard_negative_pairs if uf.find(left) == uf.find(right))
        row_out = {
            "scene": scene,
            "solver_variant": variant,
            "carrier_node_count": len(node_ids),
            "diagnostic_labeled_carrier_count": len(labeled_ids),
            "carrier_edge_count": len(edge_rows),
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
            "soft_negative_lambda": None
            if variant in {"S0_carrier_positive_cc", "S7_carrier_hard_veto_cc"}
            else float(soft_negative_lambda),
            "merge_margin": None
            if variant in {"S0_carrier_positive_cc", "S7_carrier_hard_veto_cc"}
            else float(merge_margin),
            "positive_candidate_count": len(candidate_edges),
            "signed_merge_candidate_count": len(merge_edges),
            "hard_negative_count": len(hard_negative_pairs),
            "accepted_merge_count": accepted,
            "rejected_negative_veto_count": rejected,
            "rejected_soft_margin_count": margin_rejected,
            "cluster_count": len(uf.members),
            "diagnostic_subset_purity": _cluster_purity(true_labels, pred_labels),
            "diagnostic_subset_completeness": _cluster_completeness(true_labels, pred_labels),
            "diagnostic_subset_ari": _adjusted_rand_score(true_labels, pred_labels),
            "diagnostic_pairwise_precision": pairwise_precision,
            "diagnostic_pairwise_recall": pairwise_recall,
            "hard_negative_violation_count": hard_negative_violation_count,
            "hard_negative_violation_rate": float(hard_negative_violation_count / max(len(hard_negative_pairs), 1)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "diagnostic_only": True,
        }
        rows_out.append(row_out)
    base = rows_out[0]
    base["purity_minus_s0"] = None
    base["ari_minus_s0"] = None
    base["completeness_minus_s0"] = None
    base["carrier_phase5_gate_pass"] = False
    signed_rows = rows_out[1:]
    for signed in signed_rows:
        signed["purity_minus_s0"] = float(signed["diagnostic_subset_purity"] - base["diagnostic_subset_purity"])
        signed["ari_minus_s0"] = float(signed["diagnostic_subset_ari"] - base["diagnostic_subset_ari"])
        signed["completeness_minus_s0"] = float(
            signed["diagnostic_subset_completeness"] - base["diagnostic_subset_completeness"]
        )
        signed["carrier_phase5_gate_pass"] = bool(
            signed["purity_minus_s0"] >= 0.05
            and signed["ari_minus_s0"] >= -0.02
            and signed["diagnostic_subset_completeness"] >= 0.555
            and signed["hard_negative_violation_rate"] <= 0.05
        )
    signed_best = sorted(
        signed_rows,
        key=lambda row: (
            bool(row.get("carrier_phase5_gate_pass")),
            float(row.get("diagnostic_subset_ari") or -999.0),
            float(row.get("diagnostic_subset_completeness") or -999.0),
            float(row.get("diagnostic_subset_purity") or -999.0),
        ),
        reverse=True,
    )[0]
    return {"rows": rows_out, "signed": signed_best}


def _scene_payload(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_threshold: float,
    confidence_threshold: float,
    min_mask_area: int,
    carriers_per_mask: int,
    max_carriers: int,
    positive_thresholds: list[float],
    negative_thresholds: list[float],
    soft_negative_lambdas: list[float],
    merge_margins: list[float],
) -> dict[str, Any]:
    windows, _window_rows, _manifest_diag = _load_scene_windows(
        scene=scene,
        carrier_cache_root=carrier_cache_root,
        visibility_threshold=float(visibility_threshold),
        confidence_threshold=float(confidence_threshold),
        min_mask_area=int(min_mask_area),
    )
    nodes, _frame_rows, _node_diag = _build_nodes(scene, windows, min_mask_area=int(min_mask_area))
    windows_by_index = {window.window_index: window for window in windows}
    node_lookup = _node_by_frame_mask(nodes)
    selected_keys = _select_carriers(nodes, carriers_per_mask=int(carriers_per_mask), max_carriers=int(max_carriers))
    carrier_rows: list[dict[str, Any]] = []
    for carrier_node_id, key in enumerate(selected_keys):
        gt, gt_purity, observed_mask_count = _carrier_diag_label(key, windows_by_index, node_lookup)
        window = windows_by_index[key[0]]
        carrier_rows.append(
            {
                "scene": scene,
                "carrier_node_id": int(carrier_node_id),
                "window_index": int(key[0]),
                "local_carrier_index": int(key[1]),
                "carrier_id": int(window.carrier_ids[key[1]]),
                "diagnostic_gt": gt,
                "diagnostic_gt_purity": gt_purity,
                "observed_mask_count": int(observed_mask_count),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    edge_rows: list[dict[str, Any]] = []
    labels: list[bool] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for i, left_key in enumerate(selected_keys):
        for j in range(i + 1, len(selected_keys)):
            right_key = selected_keys[j]
            if left_key[0] != right_key[0]:
                continue
            positive_score, negative_score, both_visible, both_inside, same_mask = _carrier_pair_scores(
                left_key, right_key, windows_by_index
            )
            left_gt = carrier_rows[i].get("diagnostic_gt")
            right_gt = carrier_rows[j].get("diagnostic_gt")
            same_gt = None if left_gt is None or right_gt is None else bool(left_gt == right_gt)
            if same_gt is not None:
                labels.append(bool(same_gt))
                positive_scores.append(positive_score)
                negative_scores.append(negative_score)
            edge_rows.append(
                {
                    "scene": scene,
                    "left_carrier_node_id": int(i),
                    "right_carrier_node_id": int(j),
                    "left_carrier_id": carrier_rows[i]["carrier_id"],
                    "right_carrier_id": carrier_rows[j]["carrier_id"],
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "both_visible_count": int(both_visible),
                    "both_inside_mask_count": int(both_inside),
                    "same_observed_mask_count": int(same_mask),
                    "left_gt": left_gt,
                    "right_gt": right_gt,
                    "diagnostic_same_gt": same_gt,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    solver_rows: list[dict[str, Any]] = []
    best_signed: dict[str, Any] | None = None
    for positive_threshold in positive_thresholds:
        for negative_threshold in negative_thresholds:
            for soft_negative_lambda in soft_negative_lambdas:
                for merge_margin in merge_margins:
                    solved = _solve_carrier_graph(
                        scene=scene,
                        carrier_rows=carrier_rows,
                        edge_rows=edge_rows,
                        positive_threshold=float(positive_threshold),
                        negative_threshold=float(negative_threshold),
                        soft_negative_lambda=float(soft_negative_lambda),
                        merge_margin=float(merge_margin),
                    )
                    solver_rows.extend(solved["rows"])
                    signed = solved["signed"]
                    if best_signed is None or (
                        bool(signed.get("carrier_phase5_gate_pass")),
                        float(signed.get("diagnostic_subset_ari") or -999.0),
                        float(signed.get("diagnostic_subset_completeness") or -999.0),
                        float(signed.get("diagnostic_subset_purity") or -999.0),
                    ) > (
                        bool(best_signed.get("carrier_phase5_gate_pass")),
                        float(best_signed.get("diagnostic_subset_ari") or -999.0),
                        float(best_signed.get("diagnostic_subset_completeness") or -999.0),
                        float(best_signed.get("diagnostic_subset_purity") or -999.0),
                    ):
                        best_signed = dict(signed)
    summary = {
        "scene": scene,
        "carrier_node_count": len(carrier_rows),
        "carrier_edge_count": len(edge_rows),
        "diagnostic_labeled_carrier_count": sum(1 for row in carrier_rows if row.get("diagnostic_gt") is not None),
        "positive_score_same_gt_auc": _rank_auc(labels, positive_scores),
        "negative_score_diff_gt_auc": _rank_auc([not label for label in labels], negative_scores),
        "positive_precision@top1k": _precision_at_k(edge_rows, "positive_score", 1000),
        "best_signed": best_signed,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    return {
        "summary": summary,
        "carrier_rows": carrier_rows,
        "edge_rows": edge_rows,
        "solver_rows": solver_rows,
        "best_row": best_signed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 S7 carrier-level signed graph diagnostic.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--carriers-per-mask", type=int, default=3)
    parser.add_argument("--max-carriers", type=int, default=160)
    parser.add_argument("--positive-thresholds", default="0.2,0.4,0.6")
    parser.add_argument("--negative-thresholds", default="0.5,0.7")
    parser.add_argument("--soft-negative-lambdas", default="0.25,0.5,0.75")
    parser.add_argument("--merge-margins", default="0.0")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    positive_thresholds = [float(item) for item in str(args.positive_thresholds).split(",") if item]
    negative_thresholds = [float(item) for item in str(args.negative_thresholds).split(",") if item]
    soft_negative_lambdas = [float(item) for item in str(args.soft_negative_lambdas).split(",") if item]
    merge_margins = [float(item) for item in str(args.merge_margins).split(",") if item]
    all_summary_rows: list[dict[str, Any]] = []
    all_carrier_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_solver_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for scene in scenes:
        payload = _scene_payload(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_threshold=float(args.visibility_threshold),
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            carriers_per_mask=int(args.carriers_per_mask),
            max_carriers=int(args.max_carriers),
            positive_thresholds=positive_thresholds,
            negative_thresholds=negative_thresholds,
            soft_negative_lambdas=soft_negative_lambdas,
            merge_margins=merge_margins,
        )
        all_summary_rows.append(payload["summary"])
        for row in payload["carrier_rows"]:
            all_carrier_rows.append(row)
        for row in payload["edge_rows"]:
            all_edge_rows.append(row)
        for row in payload["solver_rows"]:
            all_solver_rows.append(row)
        if payload["best_row"]:
            best_rows.append(payload["best_row"])
    gate = {
        "pass": False,
        "any_scene_carrier_phase5_gate_pass": any(bool(row.get("carrier_phase5_gate_pass")) for row in best_rows),
        "all_scene_carrier_phase5_gate_pass": bool(best_rows and all(bool(row.get("carrier_phase5_gate_pass")) for row in best_rows)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_s7_carrier_level_diagnostic",
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "visibility_threshold": float(args.visibility_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "carriers_per_mask": int(args.carriers_per_mask),
        "max_carriers": int(args.max_carriers),
        "positive_thresholds": positive_thresholds,
        "negative_thresholds": negative_thresholds,
        "soft_negative_lambdas": soft_negative_lambdas,
        "merge_margins": merge_margins,
        "gate": gate,
        "summary_rows": all_summary_rows,
        "best_rows": best_rows,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "s7_carrier_level_diagnostic.json", payload)
    _write_csv(out / "s7_carrier_summary_rows.csv", all_summary_rows)
    _write_csv(out / "s7_carrier_rows.csv", all_carrier_rows)
    _write_csv(out / "s7_carrier_edge_rows.csv", all_edge_rows)
    _write_csv(out / "s7_carrier_solver_rows.csv", all_solver_rows)
    _write_csv(out / "s7_carrier_best_rows.csv", best_rows)
    print(json.dumps({"gate": gate, "summary": str(out / "s7_carrier_level_diagnostic.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
