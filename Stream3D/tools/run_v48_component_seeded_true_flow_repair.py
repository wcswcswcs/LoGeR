from __future__ import annotations

import argparse
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
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now
from stream4d_native.v48_true_min_cost_flow import select_min_cost_circulation_edges


class StringUnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {str(node): str(node) for node in nodes}
        self.members = {str(node): {str(node)} for node in nodes}

    def find(self, node: str) -> str:
        node = str(node)
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node}
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> bool:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        if len(self.members[root_l]) < len(self.members[root_r]):
            root_l, root_r = root_r, root_l
        for node in self.members[root_r]:
            self.parent[node] = root_l
        self.members[root_l].update(self.members[root_r])
        del self.members[root_r]
        return True


def _parse_csv_values(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _parse_float_values(spec: str) -> list[float]:
    return [float(item.strip()) for item in str(spec).split(",") if item.strip()]


def _seed_id(row: dict[str, Any]) -> str:
    component = str(row.get("predicted_component_object_id") or "")
    if component and not component.startswith("uncovered:"):
        return f"{row.get('scene')}|{component}"
    return f"{row.get('scene')}|uncovered:{row.get('mask_observation_id') or row.get('node_id')}"


def _is_uncovered_seed(seed: str) -> bool:
    return "|uncovered:" in str(seed)


def _build_node_maps(mask_vote_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, set[str]], dict[str, set[int]]]:
    node_to_seed: dict[int, str] = {}
    seed_frames: dict[str, set[str]] = defaultdict(set)
    seed_nodes: dict[str, set[int]] = defaultdict(set)
    for row in mask_vote_rows:
        node = parse_int(row.get("node_id"))
        seed = _seed_id(row)
        node_to_seed[node] = seed
        seed_frames[seed].add(f"{row.get('scene')}:{parse_int(row.get('frame_id'))}")
        seed_nodes[seed].add(node)
    return node_to_seed, seed_frames, seed_nodes


def _root_frames(uf: StringUnionFind, seed_frames: dict[str, set[str]], root: str) -> set[str]:
    frames: set[str] = set()
    for member in uf.members.get(root, {root}):
        frames.update(seed_frames.get(member, set()))
    return frames


def _evaluate(mask_vote_rows: list[dict[str, Any]], uf: StringUnionFind, node_to_seed: dict[int, str]) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    uncovered_rows = 0
    for row in mask_vote_rows:
        node = parse_int(row.get("node_id"))
        seed = node_to_seed.get(node, _seed_id(row))
        pred = uf.find(seed)
        gt = str(row.get("diagnostic_gt_instance") or "")
        frames[pred].add(parse_int(row.get("frame_id")))
        if _is_uncovered_seed(seed):
            uncovered_rows += 1
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            object_gt[pred][gt] += 1
            scene = str(row.get("scene"))
            scene_true[scene].append(gt)
            scene_pred[scene].append(pred)
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(value) for value in frames.values()),
        "track_count": len(frames),
        "mean_predictions_per_scene": safe_mean(len(set(scene_pred[scene])) for scene in sorted(scene_pred)),
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "unknown_tube_ratio": float(uncovered_rows / max(len(mask_vote_rows), 1)),
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
    }


def _component_pair_rows(
    selected_edges: list[dict[str, Any]],
    node_to_seed: dict[int, str],
    *,
    score_key: str,
    include_uncovered: bool,
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in selected_edges:
        left = node_to_seed.get(parse_int(edge.get("src_node_id")))
        right = node_to_seed.get(parse_int(edge.get("dst_node_id")))
        if not left or not right or left == right:
            continue
        if not include_uncovered and (_is_uncovered_seed(left) or _is_uncovered_seed(right)):
            continue
        a, b = sorted([left, right])
        key = (a, b)
        row = by_pair.setdefault(
            key,
            {
                "component_left": a,
                "component_right": b,
                "selected_flow_edge_count": 0,
                "max_score": 0.0,
                "min_visible_outside": None,
                "max_forward_visible_carrier_count": 0,
                "max_backward_visible_carrier_count": 0,
                "diagnostic_same_gt_edge_count": 0,
                "edge_type_counts": Counter(),
            },
        )
        row["selected_flow_edge_count"] += 1
        row["max_score"] = max(parse_float(row.get("max_score")), parse_float(edge.get(score_key)))
        visible = parse_float(edge.get("visible_outside"), 1.0)
        row["min_visible_outside"] = visible if row["min_visible_outside"] is None else min(row["min_visible_outside"], visible)
        row["max_forward_visible_carrier_count"] = max(
            parse_int(row.get("max_forward_visible_carrier_count")),
            parse_int(edge.get("forward_visible_carrier_count")),
        )
        row["max_backward_visible_carrier_count"] = max(
            parse_int(row.get("max_backward_visible_carrier_count")),
            parse_int(edge.get("backward_visible_carrier_count")),
        )
        if parse_bool(edge.get("diagnostic_same_gt")):
            row["diagnostic_same_gt_edge_count"] += 1
        row["edge_type_counts"][str(edge.get("edge_type"))] += 1
    rows: list[dict[str, Any]] = []
    for row in by_pair.values():
        counts = dict(row.pop("edge_type_counts"))
        rows.append({**row, "edge_type_counts": counts})
    rows.sort(key=lambda row: (parse_int(row.get("selected_flow_edge_count")), parse_float(row.get("max_score"))), reverse=True)
    return rows


def _apply_component_flow(
    *,
    seeds: list[str],
    seed_frames: dict[str, set[str]],
    pair_rows: list[dict[str, Any]],
    min_component_edge_count: int,
    conflict_policy: str,
    max_component_merges: int,
) -> tuple[StringUnionFind, list[dict[str, Any]], dict[str, Any]]:
    uf = StringUnionFind(seeds)
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    for row in pair_rows:
        if parse_int(row.get("selected_flow_edge_count")) < int(min_component_edge_count):
            rejected["min_component_edge_count"] += 1
            continue
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        root_l = uf.find(left)
        root_r = uf.find(right)
        if root_l == root_r:
            rejected["already_same_root"] += 1
            continue
        if conflict_policy == "hard":
            if _root_frames(uf, seed_frames, root_l) & _root_frames(uf, seed_frames, root_r):
                rejected["same_frame_component_conflict"] += 1
                continue
        if not uf.union(root_l, root_r):
            rejected["union_noop"] += 1
            continue
        selected.append(
            {
                **row,
                "merge_index": len(selected),
                "min_component_edge_count": min_component_edge_count,
                "conflict_policy": conflict_policy,
            }
        )
        if len(selected) >= int(max_component_merges):
            rejected["max_component_merges"] += 1
            break
    return uf, selected, dict(rejected)


def _row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("variant"),
        row.get("score_key"),
        row.get("min_score"),
        row.get("include_uncovered"),
        row.get("min_component_edge_count"),
        row.get("conflict_policy"),
        row.get("max_component_merges"),
    )


def _with_derived_scores(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in edge_rows:
        item = dict(row)
        bbox = parse_float(row.get("A0_bbox_overlap"))
        real_d4rt = max(parse_float(row.get("A5_d4rt_semantic_confirmation")), parse_float(row.get("A4_d4rt_visible_veto")))
        shuffled = parse_float(row.get("A7_shuffled_D4RT"))
        no_temporal = parse_float(row.get("A8_no_temporal_control"))
        item["A9_d4rt_confirmed_bbox"] = bbox if real_d4rt >= 0.60 else 0.0
        item["A9_no_temporal_confirmed_bbox"] = bbox if no_temporal >= 0.60 else 0.0
        item["A9_shuffled_confirmed_bbox"] = bbox if shuffled >= 0.60 else 0.0
        item["A10_d4rt_over_shuffled_bbox"] = bbox if real_d4rt >= 0.60 and real_d4rt >= shuffled else 0.0
        item["A10_shuffled_over_d4rt_bbox"] = bbox if shuffled >= 0.60 and shuffled >= real_d4rt else 0.0
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair v48 Scheme B by seeding true flow with carrier components.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-type-sets", default="B2_component_seeded_adjacent_skip:adjacent,skip")
    parser.add_argument("--real-score-keys", default="A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto")
    parser.add_argument("--control-score-keys", default="A8_no_temporal_control,A7_shuffled_D4RT,A0_bbox_overlap")
    parser.add_argument("--min-scores", default="0.10,0.20,0.30,0.50")
    parser.add_argument("--include-uncovered-values", default="false,true")
    parser.add_argument("--min-component-edge-counts", default="1,2,3")
    parser.add_argument("--conflict-policies", default="hard,soft")
    parser.add_argument("--max-component-merges", default="40,80,160")
    parser.add_argument("--add-derived-bbox-d4rt-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default="outputs/audit/v48_component_seeded_true_flow_repair")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    edge_rows = read_csv(ROOT / str(args.edge_table))
    if args.add_derived_bbox_d4rt_scores:
        edge_rows = _with_derived_scores(edge_rows)
    node_to_seed, seed_frames, _seed_nodes = _build_node_maps(mask_vote_rows)
    seeds = sorted(set(node_to_seed.values()))
    real_keys = _parse_csv_values(args.real_score_keys)
    control_keys = _parse_csv_values(args.control_score_keys)
    score_keys = real_keys + control_keys
    min_scores = _parse_float_values(args.min_scores)
    include_uncovered_values = [parse_bool(value) for value in _parse_csv_values(args.include_uncovered_values)]
    min_component_edge_counts = [int(value) for value in _parse_float_values(args.min_component_edge_counts)]
    conflict_policies = _parse_csv_values(args.conflict_policies)
    max_component_merges_values = [int(value) for value in _parse_float_values(args.max_component_merges)]
    edge_type_sets: list[tuple[str, set[str]]] = []
    for item in str(args.edge_type_sets).split(";"):
        if not item.strip():
            continue
        name, values = item.split(":", 1)
        edge_type_sets.append((name.strip(), set(_parse_csv_values(values))))

    baseline_uf = StringUnionFind(seeds)
    baseline = _evaluate(mask_vote_rows, baseline_uf, node_to_seed)

    scan_rows: list[dict[str, Any]] = []
    flow_solver_rows: list[dict[str, Any]] = []
    pair_rows_all: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for variant, edge_types in edge_type_sets:
        for score_key in score_keys:
            for min_score in min_scores:
                selected_edges, solver_info = select_min_cost_circulation_edges(
                    edge_rows=edge_rows,
                    score_key=score_key,
                    min_score=float(min_score),
                    edge_types=edge_types,
                    max_visible_outside=1.0,
                    min_visible_carriers=0,
                    respect_edge_accept_candidate=False,
                )
                flow_solver_rows.append(
                    {
                        "variant": variant,
                        "score_key": score_key,
                        "min_score": min_score,
                        "edge_types": ",".join(sorted(edge_types)),
                        **solver_info,
                    }
                )
                for include_uncovered in include_uncovered_values:
                    component_pairs = _component_pair_rows(
                        selected_edges,
                        node_to_seed,
                        score_key=score_key,
                        include_uncovered=include_uncovered,
                    )
                    for pair in component_pairs:
                        pair_rows_all.append(
                            {
                                "variant": variant,
                                "score_key": score_key,
                                "min_score": min_score,
                                "include_uncovered": include_uncovered,
                                **pair,
                            }
                        )
                    for min_count in min_component_edge_counts:
                        for conflict_policy in conflict_policies:
                            for max_merges in max_component_merges_values:
                                uf, selected_pairs, rejected = _apply_component_flow(
                                    seeds=seeds,
                                    seed_frames=seed_frames,
                                    pair_rows=component_pairs,
                                    min_component_edge_count=min_count,
                                    conflict_policy=conflict_policy,
                                    max_component_merges=max_merges,
                                )
                                metrics = _evaluate(mask_vote_rows, uf, node_to_seed)
                                row = {
                                    "variant": variant,
                                    "score_key": score_key,
                                    "min_score": min_score,
                                    "edge_types": ",".join(sorted(edge_types)),
                                    "include_uncovered": include_uncovered,
                                    "min_component_edge_count": min_count,
                                    "conflict_policy": conflict_policy,
                                    "max_component_merges": max_merges,
                                    "component_pair_count": len(component_pairs),
                                    "selected_component_pair_count": len(selected_pairs),
                                    "component_flow_rejected": rejected,
                                    "delta_ARI_vs_component_seed": metrics["ARI"] - baseline["ARI"],
                                    "delta_completeness_vs_component_seed": metrics["completeness"] - baseline["completeness"],
                                    "purity_drop_vs_component_seed": baseline["purity"] - metrics["purity"],
                                    "duplicate_rate": 0.0,
                                    **metrics,
                                }
                                scan_rows.append(row)
                                selected_by_signature[_row_signature(row)] = selected_pairs

    scan_rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("completeness")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in scan_rows if row["score_key"] in set(real_keys)]
    no_temporal_keys = {key for key in control_keys if "no_temporal" in str(key) or key == "A8_no_temporal_control"}
    shuffled_keys = {key for key in control_keys if "shuffled" in str(key) or key == "A7_shuffled_D4RT"}
    no_temporal_rows = [row for row in scan_rows if row["score_key"] in no_temporal_keys]
    shuffled_rows = [row for row in scan_rows if row["score_key"] in shuffled_keys]
    mask_only_rows = [row for row in scan_rows if row["score_key"] == "A0_bbox_overlap"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    best_mask_only = mask_only_rows[0] if mask_only_rows else {}
    real_minus_shuffled = parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI"))
    real_minus_no_temporal = parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI"))
    real_minus_mask_only = parse_float(best_real.get("ARI")) - parse_float(best_mask_only.get("ARI"))

    gate = {
        "beats_component_seed_ARI_pass": parse_float(best_real.get("delta_ARI_vs_component_seed")) >= 0.03,
        "beats_component_seed_completeness_pass": parse_float(best_real.get("delta_completeness_vs_component_seed")) >= 0.05,
        "purity_drop_vs_component_seed_pass": parse_float(best_real.get("purity_drop_vs_component_seed")) <= 0.01,
        "partial_ARI_pass": parse_float(best_real.get("ARI")) >= 0.45,
        "partial_purity_pass": parse_float(best_real.get("purity")) >= 0.87,
        "partial_completeness_pass": parse_float(best_real.get("completeness")) >= 0.50,
        "real_minus_shuffled_pass": real_minus_shuffled >= 0.20,
        "real_minus_no_temporal_pass": real_minus_no_temporal >= 0.10,
        "real_minus_mask_only_pass": real_minus_mask_only >= 0.10,
    }
    gate["pass"] = bool(
        gate["beats_component_seed_ARI_pass"]
        and gate["beats_component_seed_completeness_pass"]
        and gate["purity_drop_vs_component_seed_pass"]
        and gate["partial_ARI_pass"]
        and gate["partial_purity_pass"]
        and gate["partial_completeness_pass"]
        and gate["real_minus_shuffled_pass"]
        and gate["real_minus_no_temporal_pass"]
        and gate["real_minus_mask_only_pass"]
    )

    summary = {
        "phase": "v48_component_seeded_true_flow_repair",
        "created_at": utc_now(),
        "repair_basis": "Scheme B under-merge repair: seed true min-cost flow with carrier components and stitch component roots using selected sparse temporal-flow edges.",
        "add_derived_bbox_d4rt_scores": bool(args.add_derived_bbox_d4rt_scores),
        "baseline_component_seed": baseline,
        "row_count": len(scan_rows),
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_mask_only_row": best_mask_only,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "real_minus_mask_only_ARI": real_minus_mask_only,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_COMPONENT_SEEDED_TRUE_FLOW_REPAIR",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out = ROOT / str(args.output_root)
    write_json(out / "component_seeded_true_flow_repair_summary.json", summary)
    write_csv(out / "component_seeded_true_flow_repair_rows.csv", scan_rows)
    write_csv(out / "component_seeded_true_flow_solver_rows.csv", flow_solver_rows)
    write_csv(out / "component_seeded_true_flow_pair_rows.csv", pair_rows_all)
    passing = [row for row in scan_rows if row["score_key"] in set(real_keys) and parse_float(row.get("ARI")) >= 0.45 and parse_float(row.get("purity")) >= 0.87 and parse_float(row.get("completeness")) >= 0.50]
    write_csv(out / "component_seeded_true_flow_partial_metric_rows.csv", passing)
    for name, row in [
        ("best_real", best_real),
        ("best_no_temporal", best_no_temporal),
        ("best_shuffled", best_shuffled),
        ("best_mask_only", best_mask_only),
    ]:
        if row:
            write_csv(out / f"component_seeded_true_flow_{name}_selected_pairs.csv", selected_by_signature.get(_row_signature(row), []))
    print({"summary": str(out / "component_seeded_true_flow_repair_summary.json"), "gate": gate, "failure_label": summary["failure_label"]})


if __name__ == "__main__":
    main()
