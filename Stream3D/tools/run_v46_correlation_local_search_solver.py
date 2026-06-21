from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.run_v46_local_candidate_signed_solver import _local_rank_maps, _passes_local_filter
from tools.run_v46_raw_signed_solver_diagnostic import (
    _component_negative_conflicts,
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
    _would_violate,
    _write_csv,
    _write_json,
    UnionFind,
)


class PartitionView:
    def __init__(self, node_ids: list[int], labels: dict[int, int]) -> None:
        normalized: dict[int, int] = {}
        self.parent: dict[int, int] = {}
        self.members: dict[int, set[int]] = {}
        for node_id in node_ids:
            raw_label = labels[node_id]
            if raw_label not in normalized:
                normalized[raw_label] = node_id
                self.members[node_id] = set()
            root = normalized[raw_label]
            self.parent[node_id] = root
            self.members[root].add(node_id)

    def find(self, node_id: int) -> int:
        return self.parent[node_id]


def _normalize_labels(labels: dict[int, int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    out: dict[int, int] = {}
    next_label = 0
    for node_id in sorted(labels):
        label = labels[node_id]
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        out[node_id] = mapping[label]
    return out


def _cluster_members(labels: dict[int, int]) -> dict[int, set[int]]:
    members: dict[int, set[int]] = defaultdict(set)
    for node_id, label in labels.items():
        members[label].add(node_id)
    return dict(members)


def _build_pair_weights(
    *,
    rows: list[dict[str, Any]],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    local_mode: str,
    local_topk: int,
    derived_specs: dict[str, tuple[float, float]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], set[tuple[int, int]], int, int]:
    ranks = _local_rank_maps(rows, positive_key)
    positive: dict[tuple[int, int], float] = {}
    negative: dict[tuple[int, int], float] = {}
    hard_negative_pairs: set[tuple[int, int]] = set()
    positive_candidate_count = 0
    positive_candidate_same_gt_count = 0
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
            positive_candidate_count += 1
            positive_candidate_same_gt_count += int(_parse_bool(row.get("diagnostic_same_gt")))
        if negative_key != "none" and _is_negative_row(row, negative_key, derived_specs):
            negative[pair] = 1.0
            hard_negative_pairs.add(pair)
    return positive, negative, hard_negative_pairs, positive_candidate_count, positive_candidate_same_gt_count


def _positive_cc_labels(node_ids: list[int], positive: dict[tuple[int, int], float]) -> dict[int, int]:
    uf = UnionFind(node_ids)
    for left, right in sorted(positive):
        uf.union(left, right)
    return _normalize_labels({node_id: uf.find(node_id) for node_id in node_ids})


def _hard_veto_cc_labels(
    node_ids: list[int],
    positive: dict[tuple[int, int], float],
    hard_negative_pairs: set[tuple[int, int]],
) -> dict[int, int]:
    uf = UnionFind(node_ids)
    for (left, right), _score in sorted(positive.items(), key=lambda item: item[1], reverse=True):
        if _would_violate(uf, left, right, hard_negative_pairs):
            continue
        uf.union(left, right)
    return _normalize_labels({node_id: uf.find(node_id) for node_id in node_ids})


def _initial_labels(
    *,
    node_ids: list[int],
    positive: dict[tuple[int, int], float],
    hard_negative_pairs: set[tuple[int, int]],
    init_mode: str,
) -> dict[int, int]:
    if init_mode == "singleton":
        return {node_id: index for index, node_id in enumerate(node_ids)}
    if init_mode == "positive_cc":
        return _positive_cc_labels(node_ids, positive)
    if init_mode == "hard_veto_cc":
        return _hard_veto_cc_labels(node_ids, positive, hard_negative_pairs)
    raise ValueError(f"unknown init mode: {init_mode}")


def _energy_terms(
    node_ids: list[int],
    labels: dict[int, int],
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    *,
    negative_lambda: float,
    cluster_lambda: float,
) -> dict[str, float]:
    positive_cut_cost = 0.0
    negative_inside_cost = 0.0
    for i, left in enumerate(node_ids):
        for right in node_ids[i + 1 :]:
            pair = _pair_key(left, right)
            same = labels[left] == labels[right]
            if same:
                negative_inside_cost += float(negative_lambda) * negative.get(pair, 0.0)
            else:
                positive_cut_cost += positive.get(pair, 0.0)
    cluster_count = len(set(labels.values()))
    cluster_cost = float(cluster_lambda) * float(cluster_count)
    return {
        "positive_cut_cost": positive_cut_cost,
        "negative_inside_cost": negative_inside_cost,
        "cluster_cost": cluster_cost,
        "energy_total": positive_cut_cost + negative_inside_cost + cluster_cost,
    }


def _move_delta(
    *,
    node_id: int,
    target_label: int,
    node_ids: list[int],
    labels: dict[int, int],
    members: dict[int, set[int]],
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    negative_lambda: float,
    cluster_lambda: float,
) -> float:
    old_label = labels[node_id]
    if old_label == target_label:
        return 0.0
    delta = 0.0
    for other in node_ids:
        if other == node_id:
            continue
        pair = _pair_key(node_id, other)
        old_same = labels[other] == old_label
        new_same = labels[other] == target_label
        if old_same == new_same:
            continue
        pos = positive.get(pair, 0.0)
        neg = float(negative_lambda) * negative.get(pair, 0.0)
        old_cost = neg if old_same else pos
        new_cost = neg if new_same else pos
        delta += new_cost - old_cost
    old_cluster_will_vanish = len(members[old_label]) == 1
    target_is_new = target_label not in members
    cluster_delta = 0
    if old_cluster_will_vanish:
        cluster_delta -= 1
    if target_is_new:
        cluster_delta += 1
    delta += float(cluster_lambda) * float(cluster_delta)
    return delta


def _local_search(
    *,
    node_ids: list[int],
    labels: dict[int, int],
    positive: dict[tuple[int, int], float],
    negative: dict[tuple[int, int], float],
    negative_lambda: float,
    cluster_lambda: float,
    max_iters: int,
) -> tuple[dict[int, int], int]:
    labels = _normalize_labels(labels)
    move_count = 0
    next_new_label = max(labels.values(), default=-1) + 1
    for _iteration in range(int(max_iters)):
        changed = False
        members = _cluster_members(labels)
        for node_id in node_ids:
            members = _cluster_members(labels)
            candidate_labels = sorted(members)
            candidate_labels.append(next_new_label)
            best_label = labels[node_id]
            best_delta = 0.0
            for target_label in candidate_labels:
                delta = _move_delta(
                    node_id=node_id,
                    target_label=target_label,
                    node_ids=node_ids,
                    labels=labels,
                    members=members,
                    positive=positive,
                    negative=negative,
                    negative_lambda=float(negative_lambda),
                    cluster_lambda=float(cluster_lambda),
                )
                if delta < best_delta - 1.0e-9:
                    best_delta = delta
                    best_label = target_label
            if best_label != labels[node_id]:
                labels[node_id] = best_label
                if best_label == next_new_label:
                    next_new_label += 1
                labels = _normalize_labels(labels)
                next_new_label = max(labels.values(), default=-1) + 1
                move_count += 1
                changed = True
        if not changed:
            break
    return _normalize_labels(labels), move_count


def _solve_scene(
    *,
    input_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    local_mode: str,
    local_topk: int,
    negative_lambda: float,
    cluster_lambda: float,
    init_mode: str,
    max_iters: int,
    derived_specs: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    node_infos = _node_info(rows)
    node_ids = sorted(node_infos)
    positive, negative, hard_negative_pairs, candidate_count, candidate_same_gt_count = _build_pair_weights(
        rows=rows,
        positive_key=positive_key,
        positive_threshold=positive_threshold,
        negative_key=negative_key,
        local_mode=local_mode,
        local_topk=int(local_topk),
        derived_specs=derived_specs,
    )
    if negative_key == "none":
        labels = _positive_cc_labels(node_ids, positive)
        move_count = 0
    else:
        labels = _initial_labels(
            node_ids=node_ids,
            positive=positive,
            hard_negative_pairs=hard_negative_pairs,
            init_mode=init_mode,
        )
        labels, move_count = _local_search(
            node_ids=node_ids,
            labels=labels,
            positive=positive,
            negative=negative,
            negative_lambda=float(negative_lambda),
            cluster_lambda=float(cluster_lambda),
            max_iters=int(max_iters),
        )
    view = PartitionView(node_ids, labels)
    hard_negative_rows = [
        row
        for row in rows
        if negative_key != "none" and _is_negative_row(row, negative_key, derived_specs)
    ]
    hard_negative_false_same_gt_count = int(sum(1 for row in hard_negative_rows if _parse_bool(row.get("diagnostic_same_gt"))))
    hard_negative_true_diff_gt_count = int(len(hard_negative_rows) - hard_negative_false_same_gt_count)
    hard_negative_precision = (
        None if not hard_negative_rows else float(hard_negative_true_diff_gt_count / len(hard_negative_rows))
    )
    metrics = _evaluate_clusters(
        input_root=input_root,
        scene=scene,
        rows=rows,
        node_infos=node_infos,
        uf=view,  # type: ignore[arg-type]
        positive_key=positive_key,
        positive_threshold=positive_threshold,
        negative_key=negative_key,
        negative_mode="none" if negative_key == "none" else "correlation_local_search",
        negative_weight=float(negative_lambda) if negative_key != "none" else None,
        hard_negative_precision=hard_negative_precision,
        hard_negative_false_same_gt_count=hard_negative_false_same_gt_count,
        hard_negative_true_diff_gt_count=hard_negative_true_diff_gt_count,
        solver_variant="S0_local_positive_cc" if negative_key == "none" else "S5_correlation_local_search",
        accepted_merge_count=move_count,
        rejected_negative_veto_count=0,
        skipped_positive_hard_negative_count=0,
        hard_negative_pairs=hard_negative_pairs,
        positive_candidate_count=candidate_count,
    )
    terms = _energy_terms(
        node_ids,
        labels,
        positive,
        negative,
        negative_lambda=float(negative_lambda),
        cluster_lambda=float(cluster_lambda),
    )
    metrics.update(
        {
            "local_mode": local_mode,
            "local_topk": int(local_topk),
            "negative_lambda": float(negative_lambda),
            "cluster_lambda": float(cluster_lambda),
            "init_mode": init_mode if negative_key != "none" else "positive_cc",
            "move_count": int(move_count),
            "positive_candidate_count": candidate_count,
            "local_candidate_same_gt_count": candidate_same_gt_count,
            "local_candidate_same_gt_precision": (
                None if candidate_count == 0 else float(candidate_same_gt_count / candidate_count)
            ),
            **terms,
            "diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    )
    return metrics


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, float, str, int], dict[str, Any]] = {}
    for row in rows:
        if row["negative_key"] == "none":
            baselines[
                (
                    str(row["input_root"]),
                    str(row["scene"]),
                    str(row["positive_key"]),
                    float(row["positive_threshold"]),
                    str(row["local_mode"]),
                    int(row["local_topk"]),
                )
            ] = row
    for row in rows:
        base = baselines.get(
            (
                str(row["input_root"]),
                str(row["scene"]),
                str(row["positive_key"]),
                float(row["positive_threshold"]),
                str(row["local_mode"]),
                int(row["local_topk"]),
            )
        )
        if base is None or row["negative_key"] == "none":
            row["purity_minus_s0"] = None
            row["ari_minus_s0"] = None
            row["completeness_minus_s0"] = None
            row["phase5_solver_gate_pass"] = False
            row["strict_local_gate_pass"] = False
            continue
        row["purity_minus_s0"] = float(row["diagnostic_subset_purity"] - base["diagnostic_subset_purity"])
        row["ari_minus_s0"] = float(row["diagnostic_subset_ari"] - base["diagnostic_subset_ari"])
        row["completeness_minus_s0"] = float(
            row["diagnostic_subset_completeness"] - base["diagnostic_subset_completeness"]
        )
        row["phase5_solver_gate_pass"] = bool(
            row["purity_minus_s0"] >= 0.05
            and row["ari_minus_s0"] >= -0.02
            and row["diagnostic_subset_completeness"] >= 0.555
            and row["hard_negative_violation_rate"] <= 0.05
        )
        row["strict_local_gate_pass"] = bool(
            row["purity_minus_s0"] >= 0.05
            and row["ari_minus_s0"] >= -0.02
            and row["diagnostic_subset_completeness"] >= 0.80
            and row["hard_negative_violation_rate"] <= 0.01
        )
        row["diagnostic_gate_pass"] = bool(row["strict_local_gate_pass"])


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["input_root"]), str(row["scene"]))].append(row)
    best: list[dict[str, Any]] = []
    for (_root, _scene), group in grouped.items():
        signed = [row for row in group if row["negative_key"] != "none"]
        candidates = signed or group
        candidates.sort(
            key=lambda row: (
                bool(row.get("strict_local_gate_pass")),
                bool(row.get("phase5_solver_gate_pass")),
                float(row.get("diagnostic_subset_ari") or -999.0),
                float(row.get("diagnostic_subset_completeness") or -999.0),
                -float(row.get("hard_negative_violation_rate") or 999.0),
            ),
            reverse=True,
        )
        best.append(dict(candidates[0]))
    return best


def _joint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str, str, int, float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["negative_key"] == "none":
            continue
        grouped[
            (
                str(row["input_root"]),
                str(row["positive_key"]),
                float(row["positive_threshold"]),
                str(row["negative_key"]),
                str(row["local_mode"]),
                int(row["local_topk"]),
                float(row["negative_lambda"]),
                float(row["cluster_lambda"]),
                str(row["init_mode"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for key, group in grouped.items():
        (
            input_root,
            positive_key,
            positive_threshold,
            negative_key,
            local_mode,
            local_topk,
            negative_lambda,
            cluster_lambda,
            init_mode,
        ) = key
        out.append(
            {
                "input_root": input_root,
                "input_root_name": Path(input_root).name,
                "positive_key": positive_key,
                "positive_threshold": positive_threshold,
                "negative_key": negative_key,
                "local_mode": local_mode,
                "local_topk": local_topk,
                "negative_lambda": negative_lambda,
                "cluster_lambda": cluster_lambda,
                "init_mode": init_mode,
                "scene_count": len({str(row["scene"]) for row in group}),
                "all_scene_phase5_solver_gate_pass": bool(group and all(bool(row.get("phase5_solver_gate_pass")) for row in group)),
                "all_scene_strict_local_gate_pass": bool(group and all(bool(row.get("strict_local_gate_pass")) for row in group)),
                "mean_purity_minus_s0": _mean([row.get("purity_minus_s0") for row in group]),
                "min_completeness": _min([row.get("diagnostic_subset_completeness") for row in group]),
                "mean_ari": _mean([row.get("diagnostic_subset_ari") for row in group]),
                "max_hard_negative_violation_rate": _max([row.get("hard_negative_violation_rate") for row in group]),
                "diagnostic_only": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    out.sort(
        key=lambda row: (
            row["all_scene_strict_local_gate_pass"],
            row["all_scene_phase5_solver_gate_pass"],
            row["mean_ari"] if row["mean_ari"] is not None else -999.0,
            row["min_completeness"] if row["min_completeness"] is not None else -999.0,
            -(row["max_hard_negative_violation_rate"] if row["max_hard_negative_violation_rate"] is not None else 999.0),
        ),
        reverse=True,
    )
    return out


def _numbers(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        number = _parse_float(value, default=float("nan"))
        if number == number:
            out.append(number)
    return out


def _mean(values: list[Any]) -> float | None:
    numbers = _numbers(values)
    return None if not numbers else float(sum(numbers) / len(numbers))


def _min(values: list[Any]) -> float | None:
    numbers = _numbers(values)
    return None if not numbers else float(min(numbers))


def _max(values: list[Any]) -> float | None:
    numbers = _numbers(values)
    return None if not numbers else float(max(numbers))


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 S5 correlation/local-search diagnostic on raw edge rows.")
    parser.add_argument("--input-roots", required=True)
    parser.add_argument("--positive-keys", default="P5_p4_semantic_boost_capped")
    parser.add_argument("--positive-thresholds", default="0.0,0.01,0.02,0.03")
    parser.add_argument("--negative-keys", default="none,N4_semantic_contradiction_guarded_le_0p75,N4_semantic_contradiction_guarded_le_0p8")
    parser.add_argument("--local-modes", default="topk_union")
    parser.add_argument("--topks", default="5,8,12")
    parser.add_argument("--negative-lambdas", default="0.5,1.0,2.0")
    parser.add_argument("--cluster-lambdas", default="0.0,0.05")
    parser.add_argument("--init-modes", default="hard_veto_cc,positive_cc")
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--derived-negative-specs", default="")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_roots = [Path(item) for item in str(args.input_roots).split(",") if item]
    positive_keys = [item for item in str(args.positive_keys).split(",") if item]
    positive_thresholds = [float(item) for item in str(args.positive_thresholds).split(",") if item]
    negative_keys = [item for item in str(args.negative_keys).split(",") if item]
    local_modes = [item for item in str(args.local_modes).split(",") if item]
    topks = [int(item) for item in str(args.topks).split(",") if item]
    negative_lambdas = [float(item) for item in str(args.negative_lambdas).split(",") if item]
    cluster_lambdas = [float(item) for item in str(args.cluster_lambdas).split(",") if item]
    init_modes = [item for item in str(args.init_modes).split(",") if item]
    derived_specs = _parse_derived_negative_specs(str(args.derived_negative_specs))
    out = Path(args.output_root)

    scene_rows: list[dict[str, Any]] = []
    for input_root in input_roots:
        rows_by_scene = _scene_rows(_read_edge_rows(input_root))
        for scene, rows in rows_by_scene.items():
            for positive_key in positive_keys:
                for positive_threshold in positive_thresholds:
                    for local_mode in local_modes:
                        for local_topk in topks:
                            for negative_key in negative_keys:
                                if negative_key == "none":
                                    scene_rows.append(
                                        _solve_scene(
                                            input_root=input_root,
                                            scene=scene,
                                            rows=rows,
                                            positive_key=positive_key,
                                            positive_threshold=positive_threshold,
                                            negative_key=negative_key,
                                            local_mode=local_mode,
                                            local_topk=int(local_topk),
                                            negative_lambda=0.0,
                                            cluster_lambda=0.0,
                                            init_mode="positive_cc",
                                            max_iters=0,
                                            derived_specs=derived_specs,
                                        )
                                    )
                                    continue
                                for negative_lambda in negative_lambdas:
                                    for cluster_lambda in cluster_lambdas:
                                        for init_mode in init_modes:
                                            scene_rows.append(
                                                _solve_scene(
                                                    input_root=input_root,
                                                    scene=scene,
                                                    rows=rows,
                                                    positive_key=positive_key,
                                                    positive_threshold=positive_threshold,
                                                    negative_key=negative_key,
                                                    local_mode=local_mode,
                                                    local_topk=int(local_topk),
                                                    negative_lambda=float(negative_lambda),
                                                    cluster_lambda=float(cluster_lambda),
                                                    init_mode=init_mode,
                                                    max_iters=int(args.max_iters),
                                                    derived_specs=derived_specs,
                                                )
                                            )

    _add_baseline_deltas(scene_rows)
    best_rows = _best_rows(scene_rows)
    joint_rows = _joint_rows(scene_rows)
    gate = {
        "pass": False,
        "diagnostic_only": True,
        "any_scene_phase5_solver_gate_pass": any(bool(row.get("phase5_solver_gate_pass")) for row in scene_rows),
        "all_best_scene_phase5_solver_gate_pass": bool(best_rows and all(bool(row.get("phase5_solver_gate_pass")) for row in best_rows)),
        "any_joint_config_all_scene_phase5_solver_gate_pass": any(
            bool(row.get("all_scene_phase5_solver_gate_pass")) for row in joint_rows
        ),
        "any_scene_strict_local_gate_pass": any(bool(row.get("strict_local_gate_pass")) for row in scene_rows),
        "any_joint_config_all_scene_strict_local_gate_pass": any(
            bool(row.get("all_scene_strict_local_gate_pass")) for row in joint_rows
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_correlation_local_search_solver",
        "input_roots": [str(root) for root in input_roots],
        "positive_keys": positive_keys,
        "positive_thresholds": positive_thresholds,
        "negative_keys": negative_keys,
        "local_modes": local_modes,
        "topks": topks,
        "negative_lambdas": negative_lambdas,
        "cluster_lambdas": cluster_lambdas,
        "init_modes": init_modes,
        "max_iters": int(args.max_iters),
        "gate": gate,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "scene_rows": scene_rows,
        "best_rows": best_rows,
        "joint_config_rows": joint_rows,
    }
    _write_json(out / "correlation_local_search_solver.json", payload)
    _write_csv(out / "correlation_local_search_solver_scene_rows.csv", scene_rows)
    _write_csv(out / "correlation_local_search_solver_best_rows.csv", best_rows)
    _write_csv(out / "correlation_local_search_solver_joint_config_rows.csv", joint_rows)
    print(json.dumps({"gate": gate, "summary": str(out / "correlation_local_search_solver.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
