from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
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
from stream4d_native.v48_data_contract import utc_now


class StringUnionFind:
    def __init__(self, nodes: list[str], frames: dict[str, set[str]]) -> None:
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}
        self.frames = {node: set(frames.get(node, set())) for node in nodes}

    def find(self, node: str) -> str:
        node = str(node)
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node}
            self.frames[node] = set()
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def can_union(self, left: str, right: str, *, conflict_policy: str) -> bool:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        if conflict_policy == "hard" and (self.frames.get(root_l, set()) & self.frames.get(root_r, set())):
            return False
        return True

    def union(self, left: str, right: str, *, conflict_policy: str) -> bool:
        if not self.can_union(left, right, conflict_policy=conflict_policy):
            return False
        root_l = self.find(left)
        root_r = self.find(right)
        if len(self.members[root_l]) < len(self.members[root_r]):
            root_l, root_r = root_r, root_l
        for node in self.members[root_r]:
            self.parent[node] = root_l
        self.members[root_l].update(self.members[root_r])
        self.frames[root_l].update(self.frames[root_r])
        del self.members[root_r]
        del self.frames[root_r]
        return True


def _parse_csv_values(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _parse_float_values(spec: str) -> list[float]:
    return [float(item.strip()) for item in str(spec).split(",") if item.strip()]


def _parse_threshold_profiles(spec: str) -> list[tuple[str, list[float]]]:
    profiles: list[tuple[str, list[float]]] = []
    for item in str(spec).split(";"):
        if not item.strip():
            continue
        name, values = item.split(":", 1)
        profiles.append((name.strip(), _parse_float_values(values)))
    return profiles


def _real_seed_value(value: str) -> bool:
    return bool(value) and not str(value).startswith("uncovered:")


def _seed_key(row: dict[str, Any], init_mode: str) -> str:
    scene = str(row.get("scene"))
    if init_mode == "component":
        value = str(row.get("predicted_component_object_id") or "")
    elif init_mode == "supertrack":
        value = str(row.get("predicted_supertrack_object_id") or "")
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")
    if _real_seed_value(value):
        return f"{scene}|{value}"
    return f"{scene}|uncovered:{row.get('mask_observation_id') or row.get('node_id')}"


def _component_key(scene: str, component: str) -> str:
    return f"{scene}|{component}"


def _is_uncovered_seed(seed: str) -> bool:
    return "|uncovered:" in str(seed)


def _build_seed_maps(
    mask_vote_rows: list[dict[str, Any]],
    *,
    init_mode: str,
) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    seed_frames: dict[str, set[str]] = defaultdict(set)
    component_to_seeds: dict[str, set[str]] = defaultdict(set)
    seed_support: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        scene = str(row.get("scene"))
        frame = parse_int(row.get("frame_id"))
        component = str(row.get("predicted_component_object_id") or "")
        seed = _seed_key(row, init_mode)
        component_to_seeds[_component_key(scene, component)].add(seed)
        seed_frames[seed].add(f"{scene}:{frame}")
        support = seed_support.setdefault(
            seed,
            {
                "seed": seed,
                "mask_count": 0,
                "supporting_unique_carrier_count": 0.0,
                "supporting_carrier_observation_count": 0.0,
            },
        )
        support["mask_count"] += 1
        support["supporting_unique_carrier_count"] += parse_float(row.get("supporting_unique_carrier_count"))
        support["supporting_carrier_observation_count"] += parse_float(row.get("supporting_carrier_observation_count"))
    seeds = sorted(seed_frames)
    return seeds, seed_frames, component_to_seeds, seed_support


def _top_seeds(seeds: set[str], seed_support: dict[str, dict[str, Any]], limit: int) -> list[str]:
    rows = [seed_support.get(seed, {"seed": seed, "mask_count": 0}) for seed in seeds]
    rows.sort(
        key=lambda row: (
            parse_float(row.get("supporting_unique_carrier_count")),
            parse_float(row.get("supporting_carrier_observation_count")),
            parse_float(row.get("mask_count")),
            str(row.get("seed")),
        ),
        reverse=True,
    )
    return [str(row["seed"]) for row in rows[: int(limit)]]


def _score(row: dict[str, Any], score_key: str) -> float:
    a5 = parse_float(row.get("max_A5_d4rt_semantic_confirmation"))
    a4 = parse_float(row.get("max_A4_d4rt_visible_veto"))
    a8 = parse_float(row.get("max_A8_no_temporal_control"))
    a7 = parse_float(row.get("max_A7_shuffled_D4RT"))
    edge_count = parse_float(row.get("edge_count"))
    if score_key == "A5_minus_no_temporal":
        return a5 - a8
    if score_key == "A5_minus_max_control":
        return a5 - max(a8, a7)
    if score_key == "A5_minus_shuffled":
        return a5 - a7
    if score_key == "A4_minus_no_temporal":
        return a4 - a8
    if score_key == "A5_margin_max_control_x_log_edges":
        return (a5 - max(a8, a7)) * math.log1p(max(edge_count, 0.0))
    if score_key == "A5_margin_no_temporal_x_log_edges":
        return (a5 - a8) * math.log1p(max(edge_count, 0.0))
    return parse_float(row.get(f"max_{score_key}"))


def _threshold_profiles_for_key(
    score_key: str,
    *,
    margin_profiles: list[tuple[str, list[float]]],
    raw_profiles: list[tuple[str, list[float]]],
) -> list[tuple[str, list[float]]]:
    if "minus" in score_key or "margin" in score_key:
        return margin_profiles
    return raw_profiles


def _candidate_pairs(
    pair_rows: list[dict[str, Any]],
    component_to_seeds: dict[str, set[str]],
    seed_support: dict[str, dict[str, Any]],
    *,
    init_mode: str,
    score_key: str,
    threshold: float,
    min_edge_count: int,
    max_visible_outside: float,
    filter_pair_conflict: bool,
    max_seeds_per_component: int,
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pair_rows:
        scene = str(row.get("scene"))
        if parse_int(row.get("edge_count")) < int(min_edge_count):
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > float(max_visible_outside):
            continue
        if filter_pair_conflict and parse_bool(row.get("same_frame_conflict")):
            continue
        score = _score(row, score_key)
        if score < float(threshold):
            continue
        left_component = _component_key(scene, str(row.get("component_left")))
        right_component = _component_key(scene, str(row.get("component_right")))
        left_seeds = _top_seeds(component_to_seeds.get(left_component, set()), seed_support, max_seeds_per_component)
        right_seeds = _top_seeds(component_to_seeds.get(right_component, set()), seed_support, max_seeds_per_component)
        for left in left_seeds:
            for right in right_seeds:
                if left == right:
                    continue
                a, b = sorted([left, right])
                key = (a, b)
                previous = by_pair.get(key)
                candidate = {
                    "init_mode": init_mode,
                    "component_left": str(row.get("component_left")),
                    "component_right": str(row.get("component_right")),
                    "seed_left": a,
                    "seed_right": b,
                    "scene": scene,
                    "selected_score": score,
                    "score_key": score_key,
                    "threshold": threshold,
                    "edge_count": row.get("edge_count"),
                    "same_frame_conflict": row.get("same_frame_conflict"),
                    "min_visible_outside": row.get("min_visible_outside"),
                    "max_A5_d4rt_semantic_confirmation": row.get("max_A5_d4rt_semantic_confirmation"),
                    "max_A4_d4rt_visible_veto": row.get("max_A4_d4rt_visible_veto"),
                    "max_A8_no_temporal_control": row.get("max_A8_no_temporal_control"),
                    "max_A7_shuffled_D4RT": row.get("max_A7_shuffled_D4RT"),
                    "diagnostic_same_gt_edge_count": row.get("diagnostic_same_gt_edge_count"),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
                if previous is None or (
                    parse_float(candidate.get("selected_score")),
                    parse_int(candidate.get("edge_count")),
                ) > (
                    parse_float(previous.get("selected_score")),
                    parse_int(previous.get("edge_count")),
                ):
                    by_pair[key] = candidate
    out = list(by_pair.values())
    out.sort(key=lambda item: (parse_float(item.get("selected_score")), parse_int(item.get("edge_count"))), reverse=True)
    return out


def _evaluate(
    mask_vote_rows: list[dict[str, Any]],
    uf: StringUnionFind,
    *,
    init_mode: str,
    selected_score_sum: float,
    merge_count: int,
    unknown_seed_count: int,
    object_count_penalty: float,
    unknown_penalty: float,
) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    cluster_frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    uncovered_rows = 0
    for row in mask_vote_rows:
        scene = str(row.get("scene"))
        seed = _seed_key(row, init_mode)
        pred = uf.find(seed)
        gt = str(row.get("diagnostic_gt_instance") or "")
        cluster_frames[pred].add(parse_int(row.get("frame_id")))
        if _is_uncovered_seed(seed):
            uncovered_rows += 1
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            object_gt[pred][gt] += 1
            scene_true[scene].append(gt)
            scene_pred[scene].append(pred)
    cluster_count = len(cluster_frames)
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    energy = -float(selected_score_sum) + float(object_count_penalty) * cluster_count + float(unknown_penalty) * unknown_seed_count
    return {
        "cluster_count": cluster_count,
        "object_count": cluster_count,
        "selected_candidate_count": cluster_count,
        "merge_count": merge_count,
        "unknown_ratio": float(uncovered_rows / max(len(mask_vote_rows), 1)),
        "unknown_tube_ratio": float(uncovered_rows / max(len(mask_vote_rows), 1)),
        "temporal_span_mean": safe_mean(len(frames) for frames in cluster_frames.values()),
        "mean_predictions_per_scene": safe_mean(len(set(scene_pred[scene])) for scene in sorted(scene_pred)),
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "duplicate_rate": 0.0,
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
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
        "energy": energy,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _run_variant(
    *,
    mask_vote_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    init_mode: str,
    score_key: str,
    profile_name: str,
    thresholds: list[float],
    min_edge_count: int,
    max_visible_outside: float,
    filter_pair_conflict: bool,
    conflict_policy: str,
    max_merges_per_iter: int,
    max_seeds_per_component: int,
    object_count_penalty: float,
    unknown_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seeds, seed_frames, component_to_seeds, seed_support = _build_seed_maps(mask_vote_rows, init_mode=init_mode)
    uf = StringUnionFind(seeds, seed_frames)
    unknown_seed_count = sum(1 for seed in seeds if _is_uncovered_seed(seed))
    variant = (
        f"C_repair_{init_mode}_{score_key}_{profile_name}"
        f"_edge{min_edge_count}_vis{max_visible_outside}_{conflict_policy}_m{max_merges_per_iter}"
    )
    iteration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_score_sum = 0.0
    merge_count = 0
    base = _evaluate(
        mask_vote_rows,
        uf,
        init_mode=init_mode,
        selected_score_sum=selected_score_sum,
        merge_count=merge_count,
        unknown_seed_count=unknown_seed_count,
        object_count_penalty=object_count_penalty,
        unknown_penalty=unknown_penalty,
    )
    iteration_rows.append(
        {
            "variant": variant,
            "init_mode": init_mode,
            "score_key": score_key,
            "profile": profile_name,
            "iteration": 0,
            "threshold": None,
            "accepted_this_iteration": 0,
            "energy_delta": None,
            "control_gap_per_iteration": None,
            "assignment_margin_p10": None,
            "assignment_entropy": None,
            **base,
        }
    )
    previous_energy = parse_float(base.get("energy"))
    for iteration, threshold in enumerate(thresholds, start=1):
        accepted = 0
        rejected = Counter()
        accepted_scores: list[float] = []
        candidates = _candidate_pairs(
            pair_rows,
            component_to_seeds,
            seed_support,
            init_mode=init_mode,
            score_key=score_key,
            threshold=threshold,
            min_edge_count=min_edge_count,
            max_visible_outside=max_visible_outside,
            filter_pair_conflict=filter_pair_conflict,
            max_seeds_per_component=max_seeds_per_component,
        )
        for row in candidates:
            left = str(row.get("seed_left"))
            right = str(row.get("seed_right"))
            if uf.find(left) == uf.find(right):
                rejected["already_same_root"] += 1
                continue
            if not uf.can_union(left, right, conflict_policy=conflict_policy):
                rejected["root_frame_conflict"] += 1
                continue
            if not uf.union(left, right, conflict_policy=conflict_policy):
                rejected["union_noop"] += 1
                continue
            accepted += 1
            merge_count += 1
            score = parse_float(row.get("selected_score"))
            accepted_scores.append(score)
            selected_score_sum += score
            selected_rows.append(
                {
                    "variant": variant,
                    "iteration": iteration,
                    "accepted_index": accepted,
                    "threshold": threshold,
                    "min_edge_count": min_edge_count,
                    "max_visible_outside": max_visible_outside,
                    "filter_pair_conflict": filter_pair_conflict,
                    "conflict_policy": conflict_policy,
                    "max_merges_per_iter": max_merges_per_iter,
                    **row,
                }
            )
            if accepted >= int(max_merges_per_iter):
                rejected["max_merges_per_iter"] += 1
                break
        metrics = _evaluate(
            mask_vote_rows,
            uf,
            init_mode=init_mode,
            selected_score_sum=selected_score_sum,
            merge_count=merge_count,
            unknown_seed_count=unknown_seed_count,
            object_count_penalty=object_count_penalty,
            unknown_penalty=unknown_penalty,
        )
        energy = parse_float(metrics.get("energy"))
        iteration_rows.append(
            {
                "variant": variant,
                "init_mode": init_mode,
                "score_key": score_key,
                "profile": profile_name,
                "iteration": iteration,
                "threshold": threshold,
                "accepted_this_iteration": accepted,
                "candidate_count": len(candidates),
                "rejected": dict(rejected),
                "energy_delta": energy - previous_energy,
                "control_gap_per_iteration": None,
                "assignment_margin_p10": safe_quantile(accepted_scores, 0.10) if accepted_scores else None,
                "assignment_entropy": None,
                **metrics,
            }
        )
        previous_energy = energy
    summary = _summarize_variant(iteration_rows)
    return iteration_rows, selected_rows, summary


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    final = rows[-1]
    accepted_rows = [row for row in rows[1:] if parse_int(row.get("accepted_this_iteration")) > 0]
    monotonic_hits = [parse_float(row.get("energy_delta")) <= 1e-12 for row in accepted_rows]
    monotonic_rate = float(sum(1 for item in monotonic_hits if item) / max(len(monotonic_hits), 1))
    selected_scores = [parse_float(row.get("assignment_margin_p10")) for row in rows[1:] if row.get("assignment_margin_p10") not in (None, "")]
    delta_ari = parse_float(final.get("ARI")) - parse_float(first.get("ARI"))
    delta_completeness = parse_float(final.get("completeness")) - parse_float(first.get("completeness"))
    out = {
        "variant": final.get("variant"),
        "init_mode": final.get("init_mode"),
        "score_key": final.get("score_key"),
        "profile": final.get("profile"),
        "iteration_count": len(rows) - 1,
        "accepted_iteration_count": len(accepted_rows),
        "energy_monotonic_decrease_rate": monotonic_rate,
        "initial_ARI": first.get("ARI"),
        "final_ARI": final.get("ARI"),
        "delta_ARI": delta_ari,
        "initial_completeness": first.get("completeness"),
        "final_completeness": final.get("completeness"),
        "delta_completeness": delta_completeness,
        "initial_purity": first.get("purity"),
        "final_purity": final.get("purity"),
        "final_unknown_ratio": final.get("unknown_ratio"),
        "final_unknown_tube_ratio": final.get("unknown_tube_ratio"),
        "temporal_span_mean": final.get("temporal_span_mean"),
        "mean_predictions_per_scene": final.get("mean_predictions_per_scene"),
        "conflict_rate": final.get("conflict_rate"),
        "duplicate_rate": final.get("duplicate_rate"),
        "birth_from_d4rt_tube_count": final.get("birth_from_d4rt_tube_count"),
        "maskless_object_count": final.get("maskless_object_count"),
        "cluster_count": final.get("cluster_count"),
        "selected_candidate_count": final.get("selected_candidate_count"),
        "merge_count": final.get("merge_count"),
        "scene0081_ARI": final.get("scene0081_ARI"),
        "scene0011_purity": final.get("scene0011_purity"),
        "scene0050_purity": final.get("scene0050_purity"),
        "scene0591_completeness": final.get("scene0591_completeness"),
        "assignment_margin_p10": safe_quantile(selected_scores, 0.10) if selected_scores else None,
        "assignment_entropy": None,
        "control_gap_per_iteration": None,
        "gate_energy_pass": monotonic_rate >= 0.90,
        "gate_delta_ARI_pass": delta_ari >= 0.04,
        "gate_delta_completeness_pass": delta_completeness >= 0.08,
        "gate_purity_pass": parse_float(final.get("purity")) >= 0.875,
        "gate_unknown_ratio_pass": parse_float(final.get("unknown_ratio")) <= 0.35,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out["gate_pass"] = bool(
        out["gate_energy_pass"]
        and out["gate_delta_ARI_pass"]
        and out["gate_delta_completeness_pass"]
        and out["gate_purity_pass"]
        and out["gate_unknown_ratio_pass"]
    )
    return out


def _delta_if_present(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if not left or not right or left.get("final_ARI") in (None, "") or right.get("final_ARI") in (None, ""):
        return None
    return parse_float(left.get("final_ARI")) - parse_float(right.get("final_ARI"))


def _row_signature(row: dict[str, Any]) -> str:
    return str(row.get("variant"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair v48 Scheme C latent EM with D4RT-specific terms and high-temporal-support thresholds.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--init-modes", default="component,supertrack")
    parser.add_argument(
        "--real-score-keys",
        default="A5_minus_max_control,A5_minus_no_temporal,A5_minus_shuffled,A5_margin_max_control_x_log_edges,A5_margin_no_temporal_x_log_edges,A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto",
    )
    parser.add_argument("--control-score-keys", default="A8_no_temporal_control,A7_shuffled_D4RT")
    parser.add_argument("--no-temporal-score-keys", default="A8_no_temporal_control")
    parser.add_argument("--shuffled-score-keys", default="A7_shuffled_D4RT")
    parser.add_argument("--margin-threshold-profiles", default="margin_strict:0.30,0.20,0.10;margin_relaxed:0.20,0.10,0.00;margin_floor:0.10,0.05,0.00")
    parser.add_argument("--raw-threshold-profiles", default="raw_strict:0.97,0.90,0.80;raw_relaxed:0.90,0.75,0.60;raw_floor:0.75,0.60,0.45")
    parser.add_argument("--min-edge-counts", default="2,3")
    parser.add_argument("--max-visible-outside-values", default="0.6,1.0")
    parser.add_argument("--conflict-policies", default="hard")
    parser.add_argument("--filter-pair-conflict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-merges-per-iter-values", default="40,80,160")
    parser.add_argument("--max-seeds-per-component", type=int, default=4)
    parser.add_argument("--object-count-penalty", type=float, default=0.02)
    parser.add_argument("--unknown-penalty", type=float, default=0.20)
    parser.add_argument("--output-root", default="outputs/audit/v48_latent_em_d4rt_specific_repair")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    pair_rows = read_csv(ROOT / str(args.pair_stats))
    init_modes = _parse_csv_values(args.init_modes)
    real_keys = _parse_csv_values(args.real_score_keys)
    control_keys = _parse_csv_values(args.control_score_keys)
    score_keys = real_keys + control_keys
    no_temporal_keys = set(_parse_csv_values(args.no_temporal_score_keys))
    shuffled_keys = set(_parse_csv_values(args.shuffled_score_keys))
    margin_profiles = _parse_threshold_profiles(args.margin_threshold_profiles)
    raw_profiles = _parse_threshold_profiles(args.raw_threshold_profiles)
    min_edge_counts = [int(value) for value in _parse_float_values(args.min_edge_counts)]
    max_visible_outside_values = _parse_float_values(args.max_visible_outside_values)
    conflict_policies = _parse_csv_values(args.conflict_policies)
    max_merges_values = [int(value) for value in _parse_float_values(args.max_merges_per_iter_values)]

    iteration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_by_variant: dict[str, list[dict[str, Any]]] = {}
    for init_mode in init_modes:
        for score_key in score_keys:
            for profile_name, thresholds in _threshold_profiles_for_key(score_key, margin_profiles=margin_profiles, raw_profiles=raw_profiles):
                for min_edge_count in min_edge_counts:
                    for max_visible_outside in max_visible_outside_values:
                        for conflict_policy in conflict_policies:
                            for max_merges in max_merges_values:
                                rows, selected, summary = _run_variant(
                                    mask_vote_rows=mask_vote_rows,
                                    pair_rows=pair_rows,
                                    init_mode=init_mode,
                                    score_key=score_key,
                                    profile_name=profile_name,
                                    thresholds=thresholds,
                                    min_edge_count=min_edge_count,
                                    max_visible_outside=max_visible_outside,
                                    filter_pair_conflict=bool(args.filter_pair_conflict),
                                    conflict_policy=conflict_policy,
                                    max_merges_per_iter=max_merges,
                                    max_seeds_per_component=int(args.max_seeds_per_component),
                                    object_count_penalty=float(args.object_count_penalty),
                                    unknown_penalty=float(args.unknown_penalty),
                                )
                                iteration_rows.extend(rows)
                                selected_rows.extend(selected)
                                summary_rows.append(summary)
                                selected_by_variant[_row_signature(summary)] = selected

    summary_rows.sort(
        key=lambda row: (
            parse_float(row.get("final_ARI")),
            parse_float(row.get("final_completeness")),
            parse_float(row.get("final_purity")),
        ),
        reverse=True,
    )
    real_key_set = set(real_keys)
    real_rows = [row for row in summary_rows if row.get("score_key") in real_key_set]
    no_temporal_rows = [row for row in summary_rows if row.get("score_key") in no_temporal_keys]
    shuffled_rows = [row for row in summary_rows if row.get("score_key") in shuffled_keys]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    real_minus_no_temporal = _delta_if_present(best_real, best_no_temporal)
    real_minus_shuffled = _delta_if_present(best_real, best_shuffled)
    gate = {
        "best_real_core_em_gate_pass": parse_bool(best_real.get("gate_pass")),
        "best_real_energy_pass": parse_bool(best_real.get("gate_energy_pass")),
        "best_real_delta_ARI_pass": parse_bool(best_real.get("gate_delta_ARI_pass")),
        "best_real_delta_completeness_pass": parse_bool(best_real.get("gate_delta_completeness_pass")),
        "best_real_purity_pass": parse_bool(best_real.get("gate_purity_pass")),
        "best_real_unknown_ratio_pass": parse_bool(best_real.get("gate_unknown_ratio_pass")),
        "partial_ARI_pass": parse_float(best_real.get("final_ARI")) >= 0.45,
        "partial_completeness_pass": parse_float(best_real.get("final_completeness")) >= 0.50,
        "real_minus_shuffled_pass": real_minus_shuffled is not None and real_minus_shuffled >= 0.20,
        "real_minus_no_temporal_pass": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.10,
    }
    gate["pass"] = bool(
        gate["best_real_core_em_gate_pass"]
        and gate["partial_ARI_pass"]
        and gate["partial_completeness_pass"]
        and gate["real_minus_shuffled_pass"]
        and gate["real_minus_no_temporal_pass"]
    )
    summary = {
        "phase": "v48_latent_em_d4rt_specific_repair",
        "created_at": utc_now(),
        "repair_basis": (
            "Phase 5 Scheme C repair: when EM does little or controls fail, lower thresholds only for "
            "high-temporal-support pairs and add D4RT-specific margins against no-temporal/shuffled controls."
        ),
        "solver_type": "coordinate_descent_proxy_not_full_probabilistic_EM",
        "init_modes": init_modes,
        "row_count": len(summary_rows),
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_LATENT_EM_D4RT_SPECIFIC_REPAIR",
        "thresholds": {
            "energy_monotonic_decrease_rate": 0.90,
            "delta_ARI": 0.04,
            "delta_completeness": 0.08,
            "purity": 0.875,
            "unknown_ratio": 0.35,
            "partial_ARI": 0.45,
            "partial_completeness": 0.50,
            "real_minus_shuffled_ARI": 0.20,
            "real_minus_no_temporal_ARI": 0.10,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "latent_em_d4rt_specific_repair_summary.json", summary)
    write_csv(out / "latent_em_d4rt_specific_repair_summary_rows.csv", summary_rows)
    write_csv(out / "latent_em_d4rt_specific_repair_iteration_rows.csv", iteration_rows)
    write_csv(out / "latent_em_d4rt_specific_repair_selected_pairs.csv", selected_rows)
    for name, row in [
        ("best_real", best_real),
        ("best_no_temporal", best_no_temporal),
        ("best_shuffled", best_shuffled),
    ]:
        if row:
            write_csv(out / f"latent_em_d4rt_specific_repair_{name}_selected_pairs.csv", selected_by_variant.get(_row_signature(row), []))
    print({"summary": str(out / "latent_em_d4rt_specific_repair_summary.json"), "gate": gate, "failure_label": summary["failure_label"]})


if __name__ == "__main__":
    main()
