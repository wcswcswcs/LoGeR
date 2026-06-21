from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    safe_quantile,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


class StringUF:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        rl, rr = self.find(left), self.find(right)
        if rl != rr:
            self.parent[rr] = rl


def _component(row: dict[str, Any]) -> str:
    return str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id") or row.get("mask_observation_id"))


def _build_uf(components: list[str], selected_pairs: list[dict[str, Any]]) -> StringUF:
    uf = StringUF(components)
    for row in selected_pairs:
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        if left in uf.parent and right in uf.parent:
            uf.union(left, right)
    return uf


def _metric_rows(mask_vote_rows: list[dict[str, Any]], uf: StringUF, variant: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    by_scene_gt: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_root_gt: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt:
            continue
        scene = str(row.get("scene"))
        comp = _component(row)
        root = uf.find(comp)
        label = f"{scene}:{gt}"
        true_labels.append(label)
        pred_labels.append(root)
        scene_true[scene].append(label)
        scene_pred[scene].append(root)
        by_scene_gt[(scene, gt)][root] += 1
        by_root_gt[root][(scene, gt)] += 1
    frag_rows: list[dict[str, Any]] = []
    for (scene, gt), counts in by_scene_gt.items():
        total = sum(counts.values())
        largest_root, largest_count = counts.most_common(1)[0]
        frag_rows.append(
            {
                "variant": variant,
                "scene": scene,
                "diagnostic_gt_instance": gt,
                "observation_count": total,
                "predicted_root_count": len(counts),
                "largest_root": largest_root,
                "largest_root_observation_count": largest_count,
                "largest_root_fraction": float(largest_count / max(total, 1)),
                "oracle_missing_merge_count": max(0, len(counts) - 1),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    conflict_rows: list[dict[str, Any]] = []
    for root, counts in by_root_gt.items():
        if len(counts) <= 1:
            continue
        total = sum(counts.values())
        dominant, dominant_count = counts.most_common(1)[0]
        conflict_rows.append(
            {
                "variant": variant,
                "predicted_root": root,
                "diagnostic_scene_gt_count": len(counts),
                "observation_count": total,
                "dominant_scene": dominant[0],
                "dominant_gt_instance": dominant[1],
                "dominant_fraction": float(dominant_count / max(total, 1)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    def scene_metric(scene: str, metric: str) -> float | None:
        if not scene_true.get(scene):
            return None
        if metric == "ARI":
            return adjusted_rand_score(scene_true[scene], scene_pred[scene])
        if metric == "purity":
            return cluster_purity(scene_true[scene], scene_pred[scene])
        if metric == "completeness":
            return cluster_completeness(scene_true[scene], scene_pred[scene])
        raise ValueError(metric)

    metric = {
        "variant": variant,
        "metric_scope": "scene_qualified_gt_fragmentation_diagnostic_not_stage1_gate",
        "scene_qualified_ARI": adjusted_rand_score(true_labels, pred_labels),
        "scene_qualified_purity": cluster_purity(true_labels, pred_labels),
        "scene_qualified_completeness": cluster_completeness(true_labels, pred_labels),
        "scene0011_completeness": scene_metric("scene0011_00", "completeness"),
        "scene0030_completeness": scene_metric("scene0030_00", "completeness"),
        "scene0050_completeness": scene_metric("scene0050_00", "completeness"),
        "scene0081_completeness": scene_metric("scene0081_01", "completeness"),
        "scene0591_completeness": scene_metric("scene0591_00", "completeness"),
        "fragmented_gt_count": sum(1 for row in frag_rows if parse_int(row.get("predicted_root_count")) > 1),
        "oracle_missing_merge_count": sum(parse_int(row.get("oracle_missing_merge_count")) for row in frag_rows),
        "predicted_root_count_per_gt_mean": safe_mean(parse_int(row.get("predicted_root_count")) for row in frag_rows),
        "predicted_root_count_per_gt_p90": safe_quantile((parse_int(row.get("predicted_root_count")) for row in frag_rows), 0.90),
        "largest_root_fraction_mean": safe_mean(parse_float(row.get("largest_root_fraction")) for row in frag_rows),
        "conflict_root_count": len(conflict_rows),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    frag_rows.sort(key=lambda row: (parse_int(row.get("predicted_root_count")), parse_int(row.get("observation_count"))), reverse=True)
    conflict_rows.sort(key=lambda row: (parse_int(row.get("diagnostic_scene_gt_count")), parse_int(row.get("observation_count"))), reverse=True)
    return metric, frag_rows, conflict_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v48 fragmentation autopsy for component-completion candidates.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--a5-selected-pairs", default="outputs/audit/v47_carrier_component_mdl_semantic_continued19/carrier_component_mdl_semantic_best_real_selected_pairs.csv")
    parser.add_argument("--contrast-selected-pairs", default="outputs/audit/v48_control_contrast_component_merge/control_contrast_best_real_selected_pairs.csv")
    parser.add_argument("--output-root", default="outputs/audit/v48_fragmentation_autopsy")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    components = sorted({_component(row) for row in mask_vote_rows})
    variants = [
        ("raw_P3_carrier_components", []),
        ("A5_semantic_complete_link", read_csv(ROOT / str(args.a5_selected_pairs))),
        ("control_contrast_best_real", read_csv(ROOT / str(args.contrast_selected_pairs))),
    ]
    metric_rows: list[dict[str, Any]] = []
    frag_rows_all: list[dict[str, Any]] = []
    conflict_rows_all: list[dict[str, Any]] = []
    for name, selected in variants:
        uf = _build_uf(components, selected)
        metrics, frag_rows, conflict_rows = _metric_rows(mask_vote_rows, uf, name)
        metric_rows.append(metrics)
        frag_rows_all.extend(frag_rows)
        conflict_rows_all.extend(conflict_rows)
    summary = {
        "phase": "v48_fragmentation_autopsy",
        "created_at": utc_now(),
        "metric_rows": metric_rows,
        "top_fragmented_rows": frag_rows_all[:50],
        "top_conflict_rows": conflict_rows_all[:50],
        "diagnostic_note": "GT labels are used only to quantify fragmentation and conflicts after method predictions are produced.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "fragmentation_autopsy_summary.json", summary)
    write_csv(out / "fragmentation_metric_rows.csv", metric_rows)
    write_csv(out / "fragmented_gt_rows.csv", frag_rows_all)
    write_csv(out / "conflict_root_rows.csv", conflict_rows_all)
    print({"summary": str(out / "fragmentation_autopsy_summary.json"), "metrics": metric_rows})


if __name__ == "__main__":
    main()
