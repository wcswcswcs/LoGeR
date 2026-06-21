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
    safe_quantile,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


class StringUnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {str(node): str(node) for node in nodes}
        self.rank = {str(node): 0 for node in nodes}

    def find(self, node: str) -> str:
        node = str(node)
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _evaluate(mask_vote_rows: list[dict[str, Any]], uf: StringUnionFind, *, selected_score_sum: float, merge_count: int) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    cluster_frames: dict[str, set[int]] = defaultdict(set)
    unknown_count = 0
    labeled_count = 0
    for row in mask_vote_rows:
        component = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id") or row.get("mask_observation_id"))
        pred = str(uf.find(component))
        gt = str(row.get("diagnostic_gt_instance", ""))
        if component.startswith("uncovered:"):
            unknown_count += 1
        cluster_frames[pred].add(parse_int(row.get("frame_id")))
        if gt:
            labeled_count += 1
            true_labels.append(gt)
            pred_labels.append(pred)
    cluster_count = len(cluster_frames)
    energy = float(-selected_score_sum + 0.02 * cluster_count + 0.2 * unknown_count)
    return {
        "cluster_count": cluster_count,
        "object_count": cluster_count,
        "merge_count": merge_count,
        "unknown_ratio": float(unknown_count / max(len(mask_vote_rows), 1)),
        "labeled_unknown_ratio": float(unknown_count / max(labeled_count, 1)),
        "temporal_span_mean": safe_mean(len(frames) for frames in cluster_frames.values()),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "assignment_margin_p10": None,
        "assignment_entropy": None,
        "energy": energy,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _candidate_pairs(
    pair_rows: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    max_visible_outside: float,
    forbid_same_frame_conflict: bool,
    min_edge_count: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in pair_rows:
        score = parse_float(row.get(f"max_{score_key}"))
        if score < float(threshold):
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > float(max_visible_outside):
            continue
        if parse_int(row.get("edge_count")) < int(min_edge_count):
            continue
        if forbid_same_frame_conflict and parse_bool(row.get("same_frame_conflict")):
            continue
        out.append(dict(row, selected_score=score))
    out.sort(key=lambda row: (parse_float(row.get("selected_score")), parse_int(row.get("edge_count"))), reverse=True)
    return out


def _run_variant(mask_vote_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], *, variant: str, thresholds: list[float], score_key: str, max_visible_outside: float, forbid_same_frame_conflict: bool, min_edge_count: int, max_merges_per_iter: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components = sorted({str(row.get("predicted_component_object_id") or row.get("mask_observation_id")) for row in mask_vote_rows})
    uf = StringUnionFind(components)
    iteration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_score_sum = 0.0
    merge_count = 0
    base = _evaluate(mask_vote_rows, uf, selected_score_sum=selected_score_sum, merge_count=merge_count)
    iteration_rows.append({"variant": variant, "iteration": 0, "threshold": None, **base})
    for iteration, threshold in enumerate(thresholds, start=1):
        accepted = 0
        for row in _candidate_pairs(
            pair_rows,
            score_key=score_key,
            threshold=threshold,
            max_visible_outside=max_visible_outside,
            forbid_same_frame_conflict=forbid_same_frame_conflict,
            min_edge_count=min_edge_count,
        ):
            left = str(row.get("component_left"))
            right = str(row.get("component_right"))
            if left not in uf.parent or right not in uf.parent:
                continue
            if uf.find(left) == uf.find(right):
                continue
            uf.union(left, right)
            accepted += 1
            merge_count += 1
            selected_score_sum += parse_float(row.get("selected_score"))
            selected_rows.append(
                {
                    "variant": variant,
                    "iteration": iteration,
                    "component_left": left,
                    "component_right": right,
                    "selected_score": parse_float(row.get("selected_score")),
                    "score_key": score_key,
                    "threshold": threshold,
                    "edge_count": row.get("edge_count"),
                    "same_frame_conflict": row.get("same_frame_conflict"),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
            if accepted >= int(max_merges_per_iter):
                break
        metrics = _evaluate(mask_vote_rows, uf, selected_score_sum=selected_score_sum, merge_count=merge_count)
        iteration_rows.append({"variant": variant, "iteration": iteration, "threshold": threshold, "accepted_this_iteration": accepted, **metrics})
    return iteration_rows, selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 latent object EM / coordinate-descent proxy.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--output-root", default="outputs/audit/v48_latent_em")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    pair_rows = read_csv(ROOT / str(args.pair_stats))
    variants = [
        {
            "variant": "C0_component_nuclei_only",
            "thresholds": [],
            "score_key": "A5_d4rt_semantic_confirmation",
            "max_visible_outside": 1.0,
            "forbid_same_frame_conflict": True,
            "min_edge_count": 1,
            "max_merges_per_iter": 0,
        },
        {
            "variant": "C3_component_semantic_temporal_strict",
            "thresholds": [0.97, 0.90, 0.80],
            "score_key": "A5_d4rt_semantic_confirmation",
            "max_visible_outside": 0.6,
            "forbid_same_frame_conflict": True,
            "min_edge_count": 1,
            "max_merges_per_iter": 80,
        },
        {
            "variant": "C3_component_semantic_temporal_relaxed",
            "thresholds": [0.90, 0.75, 0.60],
            "score_key": "A5_d4rt_semantic_confirmation",
            "max_visible_outside": 1.0,
            "forbid_same_frame_conflict": True,
            "min_edge_count": 1,
            "max_merges_per_iter": 120,
        },
        {
            "variant": "C5_shuffled_D4RT_control",
            "thresholds": [0.90, 0.75, 0.60],
            "score_key": "A7_shuffled_D4RT",
            "max_visible_outside": 1.0,
            "forbid_same_frame_conflict": True,
            "min_edge_count": 1,
            "max_merges_per_iter": 120,
        },
    ]
    iteration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for cfg in variants:
        rows, selected = _run_variant(mask_vote_rows, pair_rows, **cfg)
        iteration_rows.extend(rows)
        selected_rows.extend(selected)
        first = rows[0]
        final = rows[-1]
        energies = [parse_float(row.get("energy")) for row in rows]
        decreases = [b <= a for a, b in zip(energies, energies[1:])]
        monotonic_rate = float(sum(1 for item in decreases if item) / max(len(decreases), 1))
        summary_rows.append(
            {
                "variant": cfg["variant"],
                "iteration_count": len(rows) - 1,
                "energy_monotonic_decrease_rate": monotonic_rate,
                "initial_ARI": first.get("ARI"),
                "final_ARI": final.get("ARI"),
                "delta_ARI": parse_float(final.get("ARI")) - parse_float(first.get("ARI")),
                "initial_completeness": first.get("completeness"),
                "final_completeness": final.get("completeness"),
                "delta_completeness": parse_float(final.get("completeness")) - parse_float(first.get("completeness")),
                "final_purity": final.get("purity"),
                "final_unknown_ratio": final.get("unknown_ratio"),
                "merge_count": final.get("merge_count"),
                "gate_energy_pass": monotonic_rate >= 0.90,
                "gate_delta_ARI_pass": parse_float(final.get("ARI")) - parse_float(first.get("ARI")) >= 0.04,
                "gate_delta_completeness_pass": parse_float(final.get("completeness")) - parse_float(first.get("completeness")) >= 0.08,
                "gate_purity_pass": parse_float(final.get("purity")) >= 0.875,
                "gate_unknown_ratio_pass": parse_float(final.get("unknown_ratio")) <= 0.35,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    for row in summary_rows:
        row["gate_pass"] = bool(
            row["gate_energy_pass"]
            and row["gate_delta_ARI_pass"]
            and row["gate_delta_completeness_pass"]
            and row["gate_purity_pass"]
            and row["gate_unknown_ratio_pass"]
        )
    best_real = max([row for row in summary_rows if not str(row["variant"]).startswith("C5_")], key=lambda row: parse_float(row.get("final_ARI")))
    best_control = max([row for row in summary_rows if str(row["variant"]).startswith("C5_")], key=lambda row: parse_float(row.get("final_ARI")))
    gate = {
        "pass": any(row["gate_pass"] for row in summary_rows if not str(row["variant"]).startswith("C5_")),
        "best_real_variant": best_real.get("variant"),
        "best_real_final_ARI": best_real.get("final_ARI"),
        "best_control_final_ARI": best_control.get("final_ARI"),
        "real_minus_shuffled_ARI": parse_float(best_real.get("final_ARI")) - parse_float(best_control.get("final_ARI")),
        "failure_label": None,
    }
    if not gate["pass"]:
        gate["failure_label"] = "NO_GO_EM"
    payload = {
        "phase": "v48_latent_object_em",
        "created_at": utc_now(),
        "solver_type": "coordinate_descent_proxy_not_full_probabilistic_EM",
        "summary_rows": summary_rows,
        "gate": gate,
        "thresholds": {
            "energy_monotonic_decrease_rate": 0.90,
            "delta_ARI": 0.04,
            "delta_completeness": 0.08,
            "purity": 0.875,
            "unknown_ratio": 0.35,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "latent_em_summary.json", payload)
    write_csv(out / "latent_em_summary_rows.csv", summary_rows)
    write_csv(out / "latent_em_iteration_rows.csv", iteration_rows)
    write_csv(out / "latent_em_selected_pairs.csv", selected_rows)
    print({"summary": str(out / "latent_em_summary.json"), "gate": gate})


if __name__ == "__main__":
    main()
