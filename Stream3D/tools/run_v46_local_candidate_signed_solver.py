from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.run_v46_raw_signed_solver_diagnostic import (
    UnionFind,
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
)


def _local_rank_maps(rows: list[dict[str, Any]], positive_key: str) -> dict[tuple[int, int], int]:
    by_node: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        left = _parse_int(row["left_node_id"])
        right = _parse_int(row["right_node_id"])
        score = _parse_float(row.get(positive_key))
        by_node[left].append((-score, right, index))
        by_node[right].append((-score, left, index))
    ranks: dict[tuple[int, int], int] = {}
    for node_id, candidates in by_node.items():
        for rank, (_neg_score, other_id, _index) in enumerate(sorted(candidates), start=1):
            ranks[(node_id, other_id)] = rank
    return ranks


def _local_rank_pair(
    row: dict[str, Any],
    ranks: dict[tuple[int, int], int],
) -> tuple[int | None, int | None, int | None, int | None]:
    left = _parse_int(row["left_node_id"])
    right = _parse_int(row["right_node_id"])
    left_rank = ranks.get((left, right))
    right_rank = ranks.get((right, left))
    rank_min = None if left_rank is None or right_rank is None else min(left_rank, right_rank)
    rank_max = None if left_rank is None or right_rank is None else max(left_rank, right_rank)
    return left_rank, right_rank, rank_min, rank_max


def _passes_local_filter(
    row: dict[str, Any],
    *,
    ranks: dict[tuple[int, int], int],
    local_mode: str,
    local_topk: int,
) -> bool:
    if local_mode == "none":
        return True
    left_rank, right_rank, rank_min, rank_max = _local_rank_pair(row, ranks)
    if left_rank is None or right_rank is None or rank_min is None or rank_max is None:
        return False
    if local_mode == "topk_union":
        return rank_min <= int(local_topk)
    if local_mode == "mutual_topk":
        return rank_max <= int(local_topk)
    if local_mode == "rank_sum_lte_2k":
        return (left_rank + right_rank) <= int(2 * local_topk)
    raise ValueError(f"unknown local mode: {local_mode}")


def _solve_scene_local(
    *,
    input_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    positive_key: str,
    positive_threshold: float,
    negative_key: str,
    negative_mode: str,
    negative_weight: float | None,
    local_mode: str,
    local_topk: int,
    derived_specs: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    node_infos = _node_info(rows)
    node_ids = sorted(node_infos)
    ranks = _local_rank_maps(rows, positive_key)
    hard_negative_rows = [
        row
        for row in rows
        if negative_key != "none" and _is_negative_row(row, negative_key, derived_specs)
    ]
    hard_negative_pairs = {
        _pair_key(_parse_int(row["left_node_id"]), _parse_int(row["right_node_id"]))
        for row in hard_negative_rows
    }
    hard_negative_false_same_gt_count = int(sum(1 for row in hard_negative_rows if _parse_bool(row.get("diagnostic_same_gt"))))
    hard_negative_true_diff_gt_count = int(len(hard_negative_rows) - hard_negative_false_same_gt_count)
    hard_negative_precision = (
        None if not hard_negative_rows else float(hard_negative_true_diff_gt_count / len(hard_negative_rows))
    )

    threshold_rows = [
        row for row in rows if _parse_float(row.get(positive_key)) >= float(positive_threshold)
    ]
    candidate_rows = [
        row
        for row in threshold_rows
        if _passes_local_filter(row, ranks=ranks, local_mode=local_mode, local_topk=int(local_topk))
    ]
    candidate_rows.sort(
        key=lambda row: (
            _parse_float(row.get(positive_key)),
            -float(_local_rank_pair(row, ranks)[2] or 10**9),
        ),
        reverse=True,
    )

    uf = UnionFind(node_ids)
    accepted_merge_count = 0
    rejected_negative_veto_count = 0
    skipped_positive_hard_negative_count = 0
    negative_conflict_checks = 0
    negative_conflict_count_total = 0
    negative_cross_pair_count_total = 0

    for row in candidate_rows:
        left = _parse_int(row["left_node_id"])
        right = _parse_int(row["right_node_id"])
        pair = _pair_key(left, right)
        score = _parse_float(row.get(positive_key))
        decision_score = score
        if negative_key != "none" and negative_mode in {"hard_veto", "hard_veto_expand"} and pair in hard_negative_pairs:
            skipped_positive_hard_negative_count += 1
            continue
        if uf.find(left) == uf.find(right):
            continue
        if negative_key != "none":
            conflict_count, cross_count, _conflict_ratio = _component_negative_conflicts(
                uf, left, right, hard_negative_pairs
            )
            negative_conflict_checks += 1
            negative_conflict_count_total += int(conflict_count)
            negative_cross_pair_count_total += int(cross_count)
            if negative_mode == "soft_penalty":
                decision_score = float(score - float(negative_weight or 0.0) * _conflict_ratio)
            if negative_mode in {"hard_veto", "hard_veto_expand"} and _would_violate(uf, left, right, hard_negative_pairs):
                rejected_negative_veto_count += 1
                continue
            if negative_mode == "soft_penalty" and decision_score < float(positive_threshold):
                rejected_negative_veto_count += 1
                continue
        if uf.union(left, right):
            accepted_merge_count += 1

    if negative_key != "none" and negative_mode == "hard_veto_expand":
        max_conflict_ratio = float(negative_weight or 0.0)
        for row in candidate_rows:
            left = _parse_int(row["left_node_id"])
            right = _parse_int(row["right_node_id"])
            if uf.find(left) == uf.find(right):
                continue
            conflict_count, cross_count, conflict_ratio = _component_negative_conflicts(
                uf, left, right, hard_negative_pairs
            )
            negative_conflict_checks += 1
            negative_conflict_count_total += int(conflict_count)
            negative_cross_pair_count_total += int(cross_count)
            if conflict_ratio <= max_conflict_ratio:
                if uf.union(left, right):
                    accepted_merge_count += 1
            else:
                rejected_negative_veto_count += 1

    metrics = _evaluate_clusters(
        input_root=input_root,
        scene=scene,
        rows=rows,
        node_infos=node_infos,
        uf=uf,
        positive_key=positive_key,
        positive_threshold=positive_threshold,
        negative_key=negative_key,
        negative_mode=negative_mode,
        negative_weight=negative_weight,
        hard_negative_precision=hard_negative_precision,
        hard_negative_false_same_gt_count=hard_negative_false_same_gt_count,
        hard_negative_true_diff_gt_count=hard_negative_true_diff_gt_count,
        solver_variant=(
            "S0_local_positive_cc"
            if negative_key == "none"
            else f"S6_local_candidate_signed_{negative_mode}_cc"
        ),
        accepted_merge_count=accepted_merge_count,
        rejected_negative_veto_count=rejected_negative_veto_count,
        skipped_positive_hard_negative_count=skipped_positive_hard_negative_count,
        hard_negative_pairs=hard_negative_pairs,
        positive_candidate_count=len(candidate_rows),
    )
    candidate_same_gt_count = int(sum(1 for row in candidate_rows if _parse_bool(row.get("diagnostic_same_gt"))))
    candidate_ranks = [
        float(_local_rank_pair(row, ranks)[2])
        for row in candidate_rows
        if _local_rank_pair(row, ranks)[2] is not None
    ]
    metrics.update(
        {
            "local_mode": local_mode,
            "local_topk": int(local_topk),
            "positive_candidate_count_before_local_filter": len(threshold_rows),
            "local_candidate_same_gt_count": candidate_same_gt_count,
            "local_candidate_same_gt_precision": (
                None if not candidate_rows else float(candidate_same_gt_count / len(candidate_rows))
            ),
            "local_candidate_rank_min_mean": (
                None if not candidate_ranks else float(sum(candidate_ranks) / len(candidate_ranks))
            ),
            "negative_conflict_checks": negative_conflict_checks,
            "negative_conflict_count_total": negative_conflict_count_total,
            "negative_cross_pair_count_total": negative_cross_pair_count_total,
            "diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    )
    return metrics


def _add_local_baseline_deltas(rows: list[dict[str, Any]]) -> None:
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
            row["hard_negative_violation_rate_minus_s0"] = None
            row["diagnostic_gate_pass"] = False
            continue
        row["purity_minus_s0"] = float(row["diagnostic_subset_purity"] - base["diagnostic_subset_purity"])
        row["ari_minus_s0"] = float(row["diagnostic_subset_ari"] - base["diagnostic_subset_ari"])
        row["completeness_minus_s0"] = float(
            row["diagnostic_subset_completeness"] - base["diagnostic_subset_completeness"]
        )
        row["hard_negative_violation_rate_minus_s0"] = float(
            row["hard_negative_violation_rate"] - base["hard_negative_violation_rate"]
        )
        row["diagnostic_gate_pass"] = bool(
            row["purity_minus_s0"] >= 0.05
            and row["ari_minus_s0"] >= -0.02
            and row["completeness_minus_s0"] >= -0.10
            and row["diagnostic_subset_completeness"] >= 0.80
            and row["hard_negative_violation_rate"] <= 0.01
        )


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["input_root"]), str(row["scene"]))].append(row)
    best: list[dict[str, Any]] = []
    for (_root, _scene), group in grouped.items():
        signed = [row for row in group if row["negative_key"] != "none"]
        candidates = signed or group
        candidates = sorted(
            candidates,
            key=lambda row: (
                row.get("diagnostic_gate_pass") is True,
                row.get("purity_minus_s0") if row.get("purity_minus_s0") is not None else -999.0,
                row.get("diagnostic_subset_ari") if row.get("diagnostic_subset_ari") is not None else -999.0,
                row.get("diagnostic_subset_completeness") if row.get("diagnostic_subset_completeness") is not None else -999.0,
            ),
            reverse=True,
        )
        best.append(dict(candidates[0]))
    return best


def _joint_config_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str, str, float | None, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["negative_key"] == "none":
            continue
        grouped[
            (
                str(row["input_root"]),
                str(row["positive_key"]),
                float(row["positive_threshold"]),
                str(row["negative_key"]),
                str(row["negative_mode"]),
                None if row.get("negative_weight") is None else float(row["negative_weight"]),
                str(row["local_mode"]),
                int(row["local_topk"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (
        input_root,
        positive_key,
        positive_threshold,
        negative_key,
        negative_mode,
        negative_weight,
        local_mode,
        local_topk,
    ), group in grouped.items():
        scenes = sorted({str(row["scene"]) for row in group})
        out.append(
            {
                "input_root": input_root,
                "input_root_name": Path(input_root).name,
                "positive_key": positive_key,
                "positive_threshold": positive_threshold,
                "negative_key": negative_key,
                "negative_mode": negative_mode,
                "negative_weight": negative_weight,
                "local_mode": local_mode,
                "local_topk": int(local_topk),
                "scene_count": len(scenes),
                "scenes": ",".join(scenes),
                "all_scene_diagnostic_gate_pass": bool(group and all(bool(row.get("diagnostic_gate_pass")) for row in group)),
                "mean_purity_minus_s0": _mean_optional([row.get("purity_minus_s0") for row in group]),
                "min_purity_minus_s0": _min_optional([row.get("purity_minus_s0") for row in group]),
                "mean_ari": _mean_optional([row.get("diagnostic_subset_ari") for row in group]),
                "mean_completeness": _mean_optional([row.get("diagnostic_subset_completeness") for row in group]),
                "min_completeness": _min_optional([row.get("diagnostic_subset_completeness") for row in group]),
                "max_hard_negative_violation_rate": _max_optional([row.get("hard_negative_violation_rate") for row in group]),
                "diagnostic_only": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    out.sort(
        key=lambda row: (
            row["all_scene_diagnostic_gate_pass"],
            row["mean_purity_minus_s0"] if row["mean_purity_minus_s0"] is not None else -999.0,
            row["mean_ari"] if row["mean_ari"] is not None else -999.0,
            row["min_completeness"] if row["min_completeness"] is not None else -999.0,
        ),
        reverse=True,
    )
    return out


def _mean_optional(values: list[Any]) -> float | None:
    valid = [_parse_float(value, default=float("nan")) for value in values if value is not None]
    valid = [value for value in valid if value == value]
    return None if not valid else float(sum(valid) / len(valid))


def _min_optional(values: list[Any]) -> float | None:
    valid = [_parse_float(value, default=float("nan")) for value in values if value is not None]
    valid = [value for value in valid if value == value]
    return None if not valid else float(min(valid))


def _max_optional(values: list[Any]) -> float | None:
    valid = [_parse_float(value, default=float("nan")) for value in values if value is not None]
    valid = [value for value in valid if value == value]
    return None if not valid else float(max(valid))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v46 local top-k candidate signed solver on raw visual semantic edge rows."
    )
    parser.add_argument("--input-roots", required=True, help="Comma-separated roots with raw_visual_semantic_edge_rows.csv.")
    parser.add_argument("--positive-keys", default="P5_p4_semantic_boost_capped")
    parser.add_argument("--positive-thresholds", default="0.0,0.005,0.01,0.02,0.03,0.05,0.10,0.20")
    parser.add_argument(
        "--negative-keys",
        default="none,N4_semantic_contradiction_guarded_le_0p7,N4_semantic_contradiction_guarded_le_0p75,N4_semantic_contradiction_guarded_le_0p8",
    )
    parser.add_argument("--local-modes", default="topk_union,mutual_topk,rank_sum_lte_2k")
    parser.add_argument("--topks", default="1,2,3,5,8,12")
    parser.add_argument(
        "--negative-modes",
        default="hard_veto",
        help="Comma-separated modes for non-none negative keys: hard_veto,soft_penalty,hard_veto_expand.",
    )
    parser.add_argument("--soft-negative-weights", default="0.10,0.20,0.30,0.50")
    parser.add_argument("--expansion-conflict-ratios", default="0.02,0.05,0.10,0.20")
    parser.add_argument(
        "--derived-negative-specs",
        default="",
        help="Comma-separated specs name:semantic_threshold:p4_max or semantic_threshold:p4_max.",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_roots = [Path(item) for item in str(args.input_roots).split(",") if item]
    positive_keys = [item for item in str(args.positive_keys).split(",") if item]
    positive_thresholds = [float(item) for item in str(args.positive_thresholds).split(",") if item]
    negative_keys = [item for item in str(args.negative_keys).split(",") if item]
    local_modes = [item for item in str(args.local_modes).split(",") if item]
    topks = [int(item) for item in str(args.topks).split(",") if item]
    negative_modes = [item for item in str(args.negative_modes).split(",") if item]
    soft_negative_weights = [float(item) for item in str(args.soft_negative_weights).split(",") if item]
    expansion_conflict_ratios = [float(item) for item in str(args.expansion_conflict_ratios).split(",") if item]
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
                                    mode_configs: list[tuple[str, float | None]] = [("none", None)]
                                else:
                                    mode_configs = []
                                    if "hard_veto" in negative_modes:
                                        mode_configs.append(("hard_veto", None))
                                    if "soft_penalty" in negative_modes:
                                        mode_configs.extend(("soft_penalty", weight) for weight in soft_negative_weights)
                                    if "hard_veto_expand" in negative_modes:
                                        mode_configs.extend(
                                            ("hard_veto_expand", ratio) for ratio in expansion_conflict_ratios
                                        )
                                for negative_mode, negative_weight in mode_configs:
                                    scene_rows.append(
                                        _solve_scene_local(
                                            input_root=input_root,
                                            scene=scene,
                                            rows=rows,
                                            positive_key=positive_key,
                                            positive_threshold=positive_threshold,
                                            negative_key=negative_key,
                                            negative_mode=negative_mode,
                                            negative_weight=negative_weight,
                                            local_mode=local_mode,
                                            local_topk=int(local_topk),
                                            derived_specs=derived_specs,
                                        )
                                    )

    _add_local_baseline_deltas(scene_rows)
    best_rows = _best_rows(scene_rows)
    joint_rows = _joint_config_rows(scene_rows)
    gate = {
        "pass": False,
        "diagnostic_only": True,
        "any_scene_config_diagnostic_gate_pass": any(bool(row.get("diagnostic_gate_pass")) for row in scene_rows),
        "all_best_scene_diagnostic_gate_pass": bool(best_rows and all(bool(row.get("diagnostic_gate_pass")) for row in best_rows)),
        "any_joint_config_all_scene_diagnostic_gate_pass": any(
            bool(row.get("all_scene_diagnostic_gate_pass")) for row in joint_rows
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_local_candidate_signed_solver",
        "input_roots": [str(root) for root in input_roots],
        "positive_keys": positive_keys,
        "positive_thresholds": positive_thresholds,
        "negative_keys": negative_keys,
        "negative_modes": negative_modes,
        "soft_negative_weights": soft_negative_weights,
        "expansion_conflict_ratios": expansion_conflict_ratios,
        "local_modes": local_modes,
        "topks": topks,
        "derived_negative_specs": {
            name: {"semantic_threshold": spec[0], "p4_max": spec[1]} for name, spec in derived_specs.items()
        },
        "gate": gate,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "scene_rows": scene_rows,
        "best_rows": best_rows,
        "joint_config_rows": joint_rows,
    }
    _write_json(out / "local_candidate_signed_solver.json", payload)
    _write_csv(out / "local_candidate_signed_solver_scene_rows.csv", scene_rows)
    _write_csv(out / "local_candidate_signed_solver_best_rows.csv", best_rows)
    _write_csv(out / "local_candidate_signed_solver_joint_config_rows.csv", joint_rows)
    print(json.dumps({"gate": gate, "summary": str(out / "local_candidate_signed_solver.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
