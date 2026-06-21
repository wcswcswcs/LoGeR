from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.run_v46_correlation_local_search_solver import PartitionView, _energy_terms, _positive_cc_labels
from tools.run_v46_local_candidate_signed_solver import _local_rank_maps, _passes_local_filter
from tools.run_v46_raw_signed_solver_diagnostic import (
    _evaluate_clusters,
    _is_negative_row,
    _node_info,
    _pair_key,
    _parse_bool,
    _parse_derived_negative_specs,
    _parse_float,
    _parse_int,
    _read_edge_rows,
    _scene_rows,
    _write_csv,
    _write_json,
)


def _subset_rows(rows: list[dict[str, Any]], nodes: set[int]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _parse_int(row["left_node_id"]) in nodes and _parse_int(row["right_node_id"]) in nodes
    ]


def _build_weights(
    *,
    rows: list[dict[str, Any]],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    local_mode: str,
    local_topk: int,
    derived_specs: dict[str, tuple[float, float]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    ranks = _local_rank_maps(rows, positive_key)
    positive: dict[tuple[int, int], float] = {}
    negative: dict[tuple[int, int], float] = {}
    for row in rows:
        left = _parse_int(row["left_node_id"])
        right = _parse_int(row["right_node_id"])
        pair = _pair_key(left, right)
        score = _parse_float(row.get(positive_key))
        if score >= float(positive_threshold) and _passes_local_filter(
            row,
            ranks=ranks,
            local_mode=local_mode,
            local_topk=int(local_topk),
        ):
            positive[pair] = max(score - float(positive_threshold), 0.0) + 1.0e-6
        if negative_key != "none" and _is_negative_row(row, negative_key, derived_specs):
            negative[pair] = 1.0
    return positive, negative


def _select_micro_nodes(
    *,
    rows: list[dict[str, Any]],
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    micro_size: int,
) -> list[int]:
    node_scores: dict[int, float] = defaultdict(float)
    frame_rank: dict[int, int] = {}
    for row in rows:
        for prefix in ["left", "right"]:
            node_id = _parse_int(row[f"{prefix}_node_id"])
            frame_rank.setdefault(node_id, _parse_int(row.get(f"{prefix}_frame_rank")))
    for (left, right), weight in positive.items():
        node_scores[left] += float(weight)
        node_scores[right] += float(weight)
    for (left, right), weight in negative.items():
        node_scores[left] += float(weight)
        node_scores[right] += float(weight)
    ranked = sorted(node_scores, key=lambda node_id: (-node_scores[node_id], frame_rank.get(node_id, 9999), node_id))
    return ranked[: int(micro_size)]


def _enumerate_partitions(node_ids: list[int]):
    labels = [-1 for _ in node_ids]
    labels[0] = 0

    def rec(index: int, max_label: int):
        if index == len(node_ids):
            yield {node_ids[i]: labels[i] for i in range(len(node_ids))}
            return
        for label in range(max_label + 2):
            labels[index] = label
            yield from rec(index + 1, max(max_label, label))
        labels[index] = -1

    yield from rec(1, 0)


def _exact_partition(
    *,
    node_ids: list[int],
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    negative_lambda: float,
    cluster_lambda: float,
    max_partitions: int | None,
) -> tuple[dict[int, int], dict[str, float], int, bool]:
    best_labels: dict[int, int] | None = None
    best_terms: dict[str, float] | None = None
    visited = 0
    truncated = False
    for labels in _enumerate_partitions(node_ids):
        visited += 1
        terms = _energy_terms(
            node_ids,
            labels,
            positive,
            negative,
            negative_lambda=float(negative_lambda),
            cluster_lambda=float(cluster_lambda),
        )
        if best_terms is None or terms["energy_total"] < best_terms["energy_total"] - 1.0e-12:
            best_terms = terms
            best_labels = labels
        if max_partitions is not None and visited >= int(max_partitions):
            truncated = True
            break
    if best_labels is None or best_terms is None:
        raise RuntimeError("no partitions enumerated")
    return best_labels, best_terms, visited, truncated


def _metric_row(
    *,
    input_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    labels: dict[int, int],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    solver_variant: str,
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    negative_lambda: float,
    cluster_lambda: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    node_infos = _node_info(rows)
    node_ids = sorted(node_infos)
    hard_negative_pairs = set(negative)
    hard_negative_rows = [
        row for row in rows if _pair_key(_parse_int(row["left_node_id"]), _parse_int(row["right_node_id"])) in hard_negative_pairs
    ]
    hard_negative_false_same_gt_count = int(sum(1 for row in hard_negative_rows if _parse_bool(row.get("diagnostic_same_gt"))))
    hard_negative_true_diff_gt_count = int(len(hard_negative_rows) - hard_negative_false_same_gt_count)
    hard_negative_precision = (
        None if not hard_negative_rows else float(hard_negative_true_diff_gt_count / len(hard_negative_rows))
    )
    view = PartitionView(node_ids, labels)
    row = _evaluate_clusters(
        input_root=input_root,
        scene=scene,
        rows=rows,
        node_infos=node_infos,
        uf=view,  # type: ignore[arg-type]
        positive_key=positive_key,
        positive_threshold=positive_threshold,
        negative_key=negative_key,
        negative_mode="none" if negative_key == "none" else "exact_micro",
        negative_weight=float(negative_lambda) if negative_key != "none" else None,
        hard_negative_precision=hard_negative_precision,
        hard_negative_false_same_gt_count=hard_negative_false_same_gt_count,
        hard_negative_true_diff_gt_count=hard_negative_true_diff_gt_count,
        solver_variant=solver_variant,
        accepted_merge_count=0,
        rejected_negative_veto_count=0,
        skipped_positive_hard_negative_count=0,
        hard_negative_pairs=hard_negative_pairs,
        positive_candidate_count=len(positive),
    )
    row.update(
        {
            "negative_lambda": float(negative_lambda),
            "cluster_lambda": float(cluster_lambda),
            **_energy_terms(
                node_ids,
                labels,
                positive,
                negative,
                negative_lambda=float(negative_lambda),
                cluster_lambda=float(cluster_lambda),
            ),
            **extra,
            "diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 S6 exact micro partition diagnostic.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--scene", default="scene0591_00")
    parser.add_argument("--positive-key", default="P5_p4_semantic_boost_capped")
    parser.add_argument("--positive-thresholds", default="0.0,0.02")
    parser.add_argument("--negative-key", default="N4_semantic_contradiction_guarded_le_0p8")
    parser.add_argument("--negative-lambdas", default="0.5,1.0,2.0")
    parser.add_argument("--cluster-lambdas", default="0.0,0.05")
    parser.add_argument("--local-mode", default="topk_union")
    parser.add_argument("--topks", default="5,12")
    parser.add_argument("--micro-sizes", default="10")
    parser.add_argument("--max-partitions", type=int, default=0, help="0 means exhaustive for the selected micro size.")
    parser.add_argument("--derived-negative-specs", default="")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    scene = str(args.scene)
    rows = _scene_rows(_read_edge_rows(input_root))[scene]
    derived_specs = _parse_derived_negative_specs(str(args.derived_negative_specs))
    thresholds = [float(item) for item in str(args.positive_thresholds).split(",") if item]
    negative_lambdas = [float(item) for item in str(args.negative_lambdas).split(",") if item]
    cluster_lambdas = [float(item) for item in str(args.cluster_lambdas).split(",") if item]
    topks = [int(item) for item in str(args.topks).split(",") if item]
    micro_sizes = [int(item) for item in str(args.micro_sizes).split(",") if item]
    max_partitions = None if int(args.max_partitions) <= 0 else int(args.max_partitions)

    scene_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for topk in topks:
            positive_all, negative_all = _build_weights(
                rows=rows,
                positive_key=str(args.positive_key),
                positive_threshold=float(threshold),
                negative_key=str(args.negative_key),
                local_mode=str(args.local_mode),
                local_topk=int(topk),
                derived_specs=derived_specs,
            )
            for micro_size in micro_sizes:
                micro_nodes = _select_micro_nodes(
                    rows=rows,
                    positive=positive_all,
                    negative=negative_all,
                    micro_size=int(micro_size),
                )
                micro_node_set = set(micro_nodes)
                micro_edge_rows = _subset_rows(rows, micro_node_set)
                positive = {pair: weight for pair, weight in positive_all.items() if pair[0] in micro_node_set and pair[1] in micro_node_set}
                negative = {pair: weight for pair, weight in negative_all.items() if pair[0] in micro_node_set and pair[1] in micro_node_set}
                baseline_labels = _positive_cc_labels(micro_nodes, positive)
                for negative_lambda in negative_lambdas:
                    for cluster_lambda in cluster_lambdas:
                        exact_labels, exact_terms, visited, truncated = _exact_partition(
                            node_ids=micro_nodes,
                            positive=positive,
                            negative=negative,
                            negative_lambda=float(negative_lambda),
                            cluster_lambda=float(cluster_lambda),
                            max_partitions=max_partitions,
                        )
                        base = _metric_row(
                            input_root=input_root,
                            scene=scene,
                            rows=micro_edge_rows,
                            labels=baseline_labels,
                            positive_key=str(args.positive_key),
                            positive_threshold=float(threshold),
                            negative_key="none",
                            solver_variant="S0_micro_positive_cc",
                            positive=positive,
                            negative={},
                            negative_lambda=0.0,
                            cluster_lambda=0.0,
                            extra={
                                "local_mode": str(args.local_mode),
                                "local_topk": int(topk),
                                "micro_size": int(micro_size),
                                "micro_node_ids": ",".join(str(node_id) for node_id in micro_nodes),
                                "partition_count_visited": 0,
                                "partition_search_truncated": False,
                            },
                        )
                        exact = _metric_row(
                            input_root=input_root,
                            scene=scene,
                            rows=micro_edge_rows,
                            labels=exact_labels,
                            positive_key=str(args.positive_key),
                            positive_threshold=float(threshold),
                            negative_key=str(args.negative_key),
                            solver_variant="S6_exact_micro_partition",
                            positive=positive,
                            negative=negative,
                            negative_lambda=float(negative_lambda),
                            cluster_lambda=float(cluster_lambda),
                            extra={
                                "local_mode": str(args.local_mode),
                                "local_topk": int(topk),
                                "micro_size": int(micro_size),
                                "micro_node_ids": ",".join(str(node_id) for node_id in micro_nodes),
                                "partition_count_visited": int(visited),
                                "partition_search_truncated": bool(truncated),
                                "exact_positive_cut_cost": exact_terms["positive_cut_cost"],
                                "exact_negative_inside_cost": exact_terms["negative_inside_cost"],
                                "exact_cluster_cost": exact_terms["cluster_cost"],
                                "exact_energy_total": exact_terms["energy_total"],
                            },
                        )
                        exact["purity_minus_s0"] = float(exact["diagnostic_subset_purity"] - base["diagnostic_subset_purity"])
                        exact["ari_minus_s0"] = float(exact["diagnostic_subset_ari"] - base["diagnostic_subset_ari"])
                        exact["completeness_minus_s0"] = float(
                            exact["diagnostic_subset_completeness"] - base["diagnostic_subset_completeness"]
                        )
                        exact["micro_phase5_gate_pass"] = bool(
                            exact["purity_minus_s0"] >= 0.05
                            and exact["ari_minus_s0"] >= -0.02
                            and exact["diagnostic_subset_completeness"] >= 0.555
                            and exact["hard_negative_violation_rate"] <= 0.05
                            and not exact["partition_search_truncated"]
                        )
                        scene_rows.append(exact)
                        micro_rows.append(base)
                        micro_rows.append(exact)

    best_rows = sorted(
        scene_rows,
        key=lambda row: (
            bool(row.get("micro_phase5_gate_pass")),
            float(row.get("diagnostic_subset_ari") or -999.0),
            float(row.get("diagnostic_subset_completeness") or -999.0),
            -float(row.get("hard_negative_violation_rate") or 999.0),
        ),
        reverse=True,
    )[:20]
    gate = {
        "pass": False,
        "any_micro_phase5_gate_pass": any(bool(row.get("micro_phase5_gate_pass")) for row in scene_rows),
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "full_scene_exact": False,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_exact_micro_partition",
        "input_root": str(input_root),
        "scene": scene,
        "positive_key": str(args.positive_key),
        "positive_thresholds": thresholds,
        "negative_key": str(args.negative_key),
        "negative_lambdas": negative_lambdas,
        "cluster_lambdas": cluster_lambdas,
        "local_mode": str(args.local_mode),
        "topks": topks,
        "micro_sizes": micro_sizes,
        "max_partitions": max_partitions,
        "gate": gate,
        "scene_rows": scene_rows,
        "best_rows": best_rows,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = Path(args.output_root)
    _write_json(out / "exact_micro_partition.json", payload)
    _write_csv(out / "exact_micro_partition_scene_rows.csv", scene_rows)
    _write_csv(out / "exact_micro_partition_rows.csv", micro_rows)
    _write_csv(out / "exact_micro_partition_best_rows.csv", best_rows)
    print(json.dumps({"gate": gate, "summary": str(out / "exact_micro_partition.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
