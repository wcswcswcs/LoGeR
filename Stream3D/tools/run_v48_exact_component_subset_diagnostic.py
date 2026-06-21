from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


class StringUnionFind:
    def __init__(self, nodes: list[str], frames: dict[str, set[str]]) -> None:
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}
        self.frames = {node: set(frames.get(node, set())) for node in nodes}

    def copy(self) -> "StringUnionFind":
        other = object.__new__(StringUnionFind)
        other.parent = dict(self.parent)
        other.members = {key: set(value) for key, value in self.members.items()}
        other.frames = {key: set(value) for key, value in self.frames.items()}
        return other

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


def _delta_if_present(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if not left or not right or left.get("ARI") in (None, "") or right.get("ARI") in (None, ""):
        return None
    return parse_float(left.get("ARI")) - parse_float(right.get("ARI"))


def _real_component(component: str) -> bool:
    return bool(component) and not str(component).startswith("uncovered:")


def _component_key(scene: str, component: str) -> str:
    return f"{scene}|{component}"


def _component_support(mask_vote_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    support: dict[str, dict[str, Any]] = {}
    frames: dict[str, set[str]] = defaultdict(set)
    for row in mask_vote_rows:
        component = str(row.get("predicted_component_object_id") or "")
        if not _real_component(component):
            continue
        scene = str(row.get("scene"))
        key = _component_key(scene, component)
        item = support.setdefault(
            key,
            {
                "scene": scene,
                "component": component,
                "component_key": key,
                "mask_count": 0,
                "supporting_unique_carrier_count": 0.0,
                "supporting_carrier_observation_count": 0.0,
            },
        )
        item["mask_count"] += 1
        item["supporting_unique_carrier_count"] += parse_float(row.get("supporting_unique_carrier_count"))
        item["supporting_carrier_observation_count"] += parse_float(row.get("supporting_carrier_observation_count"))
        frames[key].add(f"{scene}:{parse_int(row.get('frame_id'))}")
    return support, frames


def _score(row: dict[str, Any], score_key: str) -> float:
    if score_key == "A5_minus_no_temporal":
        return parse_float(row.get("max_A5_d4rt_semantic_confirmation")) - parse_float(row.get("max_A8_no_temporal_control"))
    if score_key == "A5_minus_max_control":
        return parse_float(row.get("max_A5_d4rt_semantic_confirmation")) - max(
            parse_float(row.get("max_A8_no_temporal_control")),
            parse_float(row.get("max_A7_shuffled_D4RT")),
        )
    if score_key == "A4_minus_no_temporal":
        return parse_float(row.get("max_A4_d4rt_visible_veto")) - parse_float(row.get("max_A8_no_temporal_control"))
    return parse_float(row.get(f"max_{score_key}"))


def _candidate_edges(
    pair_rows: list[dict[str, Any]],
    top_components: set[str],
    *,
    score_key: str,
    merge_penalty: float,
    min_raw_score: float,
    max_edges_per_scene: int,
    filter_pair_conflict: bool,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row in pair_rows:
        scene = str(row.get("scene"))
        left = _component_key(scene, str(row.get("component_left")))
        right = _component_key(scene, str(row.get("component_right")))
        if left not in top_components or right not in top_components or left == right:
            continue
        if filter_pair_conflict and str(row.get("same_frame_conflict")).lower() == "true":
            continue
        raw_score = _score(row, score_key)
        if raw_score < float(min_raw_score):
            continue
        weight = float(raw_score - merge_penalty)
        if weight <= 0.0:
            continue
        edges.append(
            {
                "scene": scene,
                "component_left": left,
                "component_right": right,
                "raw_score": raw_score,
                "weight": weight,
                "score_key": score_key,
                "merge_penalty": merge_penalty,
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
        )
    edges.sort(key=lambda item: (parse_float(item.get("weight")), parse_int(item.get("edge_count"))), reverse=True)
    return edges[: int(max_edges_per_scene)]


def _solve_exact_subset(
    nodes: list[str],
    frames: dict[str, set[str]],
    edges: list[dict[str, Any]],
    *,
    conflict_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges = [edge for edge in edges if parse_float(edge.get("weight")) > 0.0]
    suffix: list[float] = [0.0] * (len(edges) + 1)
    for idx in range(len(edges) - 1, -1, -1):
        suffix[idx] = suffix[idx + 1] + max(0.0, parse_float(edges[idx].get("weight")))
    best_score = 0.0
    best_selected: list[int] = []
    visited = 0
    pruned = 0
    initial_uf = StringUnionFind(nodes, frames)

    def dfs(idx: int, score: float, uf: StringUnionFind, selected: list[int]) -> None:
        nonlocal best_score, best_selected, visited, pruned
        visited += 1
        if score + suffix[idx] <= best_score + 1e-12:
            pruned += 1
            return
        if idx >= len(edges):
            if score > best_score:
                best_score = float(score)
                best_selected = list(selected)
            return
        dfs(idx + 1, score, uf, selected)
        edge = edges[idx]
        left = str(edge.get("component_left"))
        right = str(edge.get("component_right"))
        if uf.can_union(left, right, conflict_policy=conflict_policy):
            next_uf = uf.copy()
            next_uf.union(left, right, conflict_policy=conflict_policy)
            dfs(idx + 1, score + parse_float(edge.get("weight")), next_uf, [*selected, idx])

    dfs(0, 0.0, initial_uf, [])
    selected_edges = [dict(edges[idx], selected_index=rank) for rank, idx in enumerate(best_selected)]
    return selected_edges, {
        "candidate_edge_count": len(edges),
        "enumerated_subset_upper_bound": 2 ** len(edges),
        "branch_nodes_visited": visited,
        "branch_nodes_pruned": pruned,
        "best_objective": best_score,
    }


def _evaluate(mask_vote_rows: list[dict[str, Any]], selected_edges: list[dict[str, Any]], support_frames: dict[str, set[str]], *, conflict_policy: str) -> dict[str, Any]:
    seeds = sorted({_component_key(str(row.get("scene")), str(row.get("predicted_component_object_id"))) for row in mask_vote_rows if _real_component(str(row.get("predicted_component_object_id") or ""))})
    uf = StringUnionFind(seeds, support_frames)
    for edge in selected_edges:
        uf.union(str(edge.get("component_left")), str(edge.get("component_right")), conflict_policy=conflict_policy)
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_count = 0
    for row in mask_vote_rows:
        scene = str(row.get("scene"))
        component = str(row.get("predicted_component_object_id") or "")
        if _real_component(component):
            pred = uf.find(_component_key(scene, component))
        else:
            pred = f"{scene}|uncovered:{row.get('mask_observation_id') or row.get('node_id')}"
            unknown_count += 1
        gt = str(row.get("diagnostic_gt_instance") or "")
        frames[pred].add(parse_int(row.get("frame_id")))
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            object_gt[pred][gt] += 1
            scene_true[scene].append(gt)
            scene_pred[scene].append(pred)
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(value) for value in frames.values()),
        "selected_candidate_count": len(frames),
        "selected_object_count": len(frames),
        "mean_predictions_per_scene": safe_mean(len(set(scene_pred[scene])) for scene in sorted(scene_pred)),
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "unknown_tube_ratio": float(unknown_count / max(len(mask_vote_rows), 1)),
        "duplicate_rate": 0.0,
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
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
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded exact component-subset diagnostic for Stream4D v48 Scheme A.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--score-keys", default="A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto,A5_minus_no_temporal,A5_minus_max_control,A8_no_temporal_control,A7_shuffled_D4RT,A0_bbox_overlap")
    parser.add_argument("--real-score-keys", default="A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto,A5_minus_no_temporal,A5_minus_max_control")
    parser.add_argument("--no-temporal-score-keys", default="A8_no_temporal_control")
    parser.add_argument("--shuffled-score-keys", default="A7_shuffled_D4RT")
    parser.add_argument("--mask-only-score-keys", default="A0_bbox_overlap")
    parser.add_argument("--merge-penalties", default="0.0,0.10,0.20,0.30")
    parser.add_argument("--conflict-policies", default="hard,soft")
    parser.add_argument("--top-components-per-scene", type=int, default=40)
    parser.add_argument("--max-edges-per-scene", type=int, default=16)
    parser.add_argument("--min-raw-score", type=float, default=0.0)
    parser.add_argument("--filter-pair-conflict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default="outputs/audit/v48_exact_component_subset_diagnostic")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    pair_rows = read_csv(ROOT / str(args.pair_stats))
    support, frames = _component_support(mask_vote_rows)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in support.values():
        by_scene[str(item["scene"])].append(item)
    top_components: dict[str, set[str]] = {}
    for scene, rows in by_scene.items():
        rows.sort(
            key=lambda row: (
                parse_float(row.get("supporting_unique_carrier_count")),
                parse_float(row.get("supporting_carrier_observation_count")),
                parse_float(row.get("mask_count")),
            ),
            reverse=True,
        )
        top_components[scene] = {str(row["component_key"]) for row in rows[: int(args.top_components_per_scene)]}

    score_keys = _parse_csv_values(args.score_keys)
    real_keys = set(_parse_csv_values(args.real_score_keys))
    no_temporal_keys = set(_parse_csv_values(args.no_temporal_score_keys))
    shuffled_keys = set(_parse_csv_values(args.shuffled_score_keys))
    mask_only_keys = set(_parse_csv_values(args.mask_only_score_keys))
    penalties = _parse_float_values(args.merge_penalties)
    conflict_policies = _parse_csv_values(args.conflict_policies)

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for score_key in score_keys:
        for merge_penalty in penalties:
            for conflict_policy in conflict_policies:
                selected_all: list[dict[str, Any]] = []
                solver_stats: list[dict[str, Any]] = []
                for scene, comps in sorted(top_components.items()):
                    scene_pairs = [row for row in pair_rows if str(row.get("scene")) == scene]
                    edges = _candidate_edges(
                        scene_pairs,
                        comps,
                        score_key=score_key,
                        merge_penalty=merge_penalty,
                        min_raw_score=float(args.min_raw_score),
                        max_edges_per_scene=int(args.max_edges_per_scene),
                        filter_pair_conflict=bool(args.filter_pair_conflict),
                    )
                    for edge in edges:
                        candidate_rows.append(
                            {
                                "variant": f"exact_{score_key}_pen{merge_penalty}_{conflict_policy}",
                                "scene": scene,
                                "conflict_policy": conflict_policy,
                                **edge,
                            }
                        )
                    selected, stats = _solve_exact_subset(sorted(comps), frames, edges, conflict_policy=conflict_policy)
                    solver_stats.append({"scene": scene, **stats})
                    selected_all.extend(selected)
                metrics = _evaluate(mask_vote_rows, selected_all, frames, conflict_policy=conflict_policy)
                row = {
                    "variant": f"A6_exact_subset_{score_key}_pen{merge_penalty}_{conflict_policy}",
                    "score_key": score_key,
                    "merge_penalty": merge_penalty,
                    "conflict_policy": conflict_policy,
                    "top_components_per_scene": int(args.top_components_per_scene),
                    "max_edges_per_scene": int(args.max_edges_per_scene),
                    "filter_pair_conflict": bool(args.filter_pair_conflict),
                    "selected_edge_count": len(selected_all),
                    "candidate_edge_count": sum(parse_int(item.get("candidate_edge_count")) for item in solver_stats),
                    "branch_nodes_visited": sum(parse_int(item.get("branch_nodes_visited")) for item in solver_stats),
                    "branch_nodes_pruned": sum(parse_int(item.get("branch_nodes_pruned")) for item in solver_stats),
                    "exact_subset_upper_bound": sum(parse_int(item.get("enumerated_subset_upper_bound")) for item in solver_stats),
                    "objective": safe_mean(item.get("best_objective") for item in solver_stats),
                    **metrics,
                }
                rows.append(row)
                for edge in selected_all:
                    selected_rows.append({"variant": row["variant"], **edge})

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in real_keys]
    no_temporal_rows = [row for row in rows if row["score_key"] in no_temporal_keys]
    shuffled_rows = [row for row in rows if row["score_key"] in shuffled_keys]
    mask_only_rows = [row for row in rows if row["score_key"] in mask_only_keys]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    best_mask_only = mask_only_rows[0] if mask_only_rows else {}
    raw_ari = 0.4247026471350924
    raw_completeness = 0.41711229946524064
    raw_purity = 0.9013125911521633
    real_minus_shuffled = _delta_if_present(best_real, best_shuffled)
    real_minus_no_temporal = _delta_if_present(best_real, best_no_temporal)
    real_minus_mask_only = _delta_if_present(best_real, best_mask_only)
    gate = {
        "exact_diagnostic_claimed": True,
        "full_ilp_claimed": False,
        "beats_raw_ARI_pass": parse_float(best_real.get("ARI")) - raw_ari >= 0.04,
        "beats_raw_completeness_pass": parse_float(best_real.get("completeness")) - raw_completeness >= 0.08,
        "purity_pass": parse_float(best_real.get("purity")) >= 0.875,
        "purity_drop_pass": raw_purity - parse_float(best_real.get("purity")) <= 0.02,
        "real_minus_shuffled_pass": real_minus_shuffled is not None and real_minus_shuffled >= 0.20,
        "real_minus_no_temporal_pass": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.10,
        "real_minus_mask_only_pass": real_minus_mask_only is not None and real_minus_mask_only >= 0.10,
        "partial_ARI_pass": parse_float(best_real.get("ARI")) >= 0.45,
        "partial_completeness_pass": parse_float(best_real.get("completeness")) >= 0.50,
    }
    gate["pass"] = bool(
        gate["beats_raw_ARI_pass"]
        and gate["beats_raw_completeness_pass"]
        and gate["purity_pass"]
        and gate["purity_drop_pass"]
        and gate["real_minus_shuffled_pass"]
        and gate["real_minus_no_temporal_pass"]
    )
    summary = {
        "phase": "v48_exact_component_subset_diagnostic",
        "created_at": utc_now(),
        "diagnostic_scope": "bounded exact branch-and-bound over top component-pair subsets per scene; not a full large-scale ILP.",
        "top_components_per_scene": int(args.top_components_per_scene),
        "max_edges_per_scene": int(args.max_edges_per_scene),
        "row_count": len(rows),
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_mask_only_row": best_mask_only,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "real_minus_mask_only_ARI": real_minus_mask_only,
        "raw_reference": {"ARI": raw_ari, "purity": raw_purity, "completeness": raw_completeness},
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_EXACT_COMPONENT_SUBSET_DIAGNOSTIC",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "exact_component_subset_summary.json", summary)
    write_csv(out / "exact_component_subset_rows.csv", rows)
    write_csv(out / "exact_component_subset_candidate_edges.csv", candidate_rows)
    write_csv(out / "exact_component_subset_selected_edges.csv", selected_rows)
    print({"summary": str(out / "exact_component_subset_summary.json"), "gate": gate, "failure_label": summary["failure_label"]})


if __name__ == "__main__":
    main()
