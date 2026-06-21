from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    rank_auc,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


class StringUF:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        if node not in self.parent:
            self.parent[node] = node
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _mask_by_node(mask_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("node_id")): row for row in mask_rows}


def _component_tracks(mask_vote_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> dict[str, dict[int, tuple[float, float]]]:
    masks = _mask_by_node(mask_rows)
    accum: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in mask_vote_rows:
        comp = str(row.get("predicted_component_object_id") or "")
        if not comp or comp.startswith("uncovered:"):
            continue
        mask = masks.get(str(row.get("node_id")))
        if not mask:
            continue
        x0, y0 = parse_float(mask.get("bbox_x0")), parse_float(mask.get("bbox_y0"))
        x1, y1 = parse_float(mask.get("bbox_x1")), parse_float(mask.get("bbox_y1"))
        frame = parse_int(row.get("frame_id"))
        accum[comp][frame].append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    tracks: dict[str, dict[int, tuple[float, float]]] = {}
    for comp, by_frame in accum.items():
        tracks[comp] = {}
        for frame, centers in by_frame.items():
            tracks[comp][frame] = (
                sum(x for x, _y in centers) / len(centers),
                sum(y for _x, y in centers) / len(centers),
            )
    return tracks


def _common_fate(left: dict[int, tuple[float, float]], right: dict[int, tuple[float, float]], *, sigma_motion: float, sigma_layout: float) -> tuple[float | None, dict[str, Any]]:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return None, {"overlap_frame_count": len(common), "reason": "insufficient_overlap"}
    motion_diffs: list[float] = []
    distances: list[float] = []
    for a, b in zip(common, common[1:]):
        la, lb = left[a], left[b]
        ra, rb = right[a], right[b]
        dl = (lb[0] - la[0], lb[1] - la[1])
        dr = (rb[0] - ra[0], rb[1] - ra[1])
        motion_diffs.append(math.hypot(dl[0] - dr[0], dl[1] - dr[1]))
    for frame in common:
        distances.append(math.hypot(left[frame][0] - right[frame][0], left[frame][1] - right[frame][1]))
    mean_motion = safe_mean(motion_diffs) or 0.0
    mean_dist = safe_mean(distances) or 0.0
    dist_var = safe_mean((value - mean_dist) ** 2 for value in distances) or 0.0
    layout_std = math.sqrt(dist_var)
    motion_score = math.exp(-mean_motion / max(float(sigma_motion), 1e-6))
    layout_score = math.exp(-layout_std / max(float(sigma_layout), 1e-6))
    score = 0.65 * motion_score + 0.35 * layout_score
    return float(score), {
        "overlap_frame_count": len(common),
        "mean_motion_delta": mean_motion,
        "layout_std": layout_std,
        "motion_score": motion_score,
        "layout_score": layout_score,
    }


def _evaluate(mask_vote_rows: list[dict[str, Any]], uf: StringUF) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    frames: dict[str, set[int]] = defaultdict(set)
    for row in mask_vote_rows:
        comp = str(row.get("predicted_component_object_id") or row.get("mask_observation_id"))
        pred = uf.find(comp)
        gt = str(row.get("diagnostic_gt_instance", ""))
        frames[pred].add(parse_int(row.get("frame_id")))
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "cluster_count": len(frames),
        "temporal_span_mean": safe_mean(len(v) for v in frames.values()),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 common-fate motion diagnostic.")
    parser.add_argument("--mask-observation-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--pair-stats", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_pair_stats.csv")
    parser.add_argument("--sigma-motion", type=float, default=40.0)
    parser.add_argument("--sigma-layout", type=float, default=80.0)
    parser.add_argument("--output-root", default="outputs/audit/v48_common_fate")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.mask_observation_table))
    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    pair_rows = read_csv(ROOT / str(args.pair_stats))
    tracks = _component_tracks(mask_vote_rows, mask_rows)
    pair_diag_rows: list[dict[str, Any]] = []
    labels: list[bool] = []
    scores: list[float] = []
    for row in pair_rows:
        left = str(row.get("component_left"))
        right = str(row.get("component_right"))
        if left not in tracks or right not in tracks:
            continue
        score, diag = _common_fate(tracks[left], tracks[right], sigma_motion=args.sigma_motion, sigma_layout=args.sigma_layout)
        if score is None:
            continue
        same = parse_int(row.get("diagnostic_same_gt_edge_count")) > 0
        labels.append(same)
        scores.append(score)
        pair_diag_rows.append(
            {
                "component_left": left,
                "component_right": right,
                "scene": row.get("scene"),
                "common_fate_score": score,
                "diagnostic_same_gt": same,
                "A5_d4rt_semantic_confirmation": row.get("max_A5_d4rt_semantic_confirmation"),
                "same_frame_conflict": row.get("same_frame_conflict"),
                **diag,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    pair_diag_rows.sort(key=lambda row: parse_float(row.get("common_fate_score")), reverse=True)
    precision_top100 = None
    if pair_diag_rows:
        top = pair_diag_rows[: min(100, len(pair_diag_rows))]
        precision_top100 = sum(1 for row in top if row["diagnostic_same_gt"]) / len(top)
    auc = rank_auc(labels, scores)

    components = sorted({str(row.get("predicted_component_object_id") or row.get("mask_observation_id")) for row in mask_vote_rows})
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    base_uf = StringUF(components)
    base = _evaluate(mask_vote_rows, base_uf)
    for variant, min_motion, min_a5, veto_same_frame_conflict in [
        ("D0_no_motion_guard", 2.0, 2.0, True),
        ("D1_image_space_common_fate_relaxed", 0.45, 0.0, True),
        ("D1_image_space_common_fate", 0.75, 0.0, True),
        ("D1b_image_space_common_fate_conflict_soft", 0.95, 0.0, False),
        ("D3_motion_plus_semantic_temporal_relaxed", 0.45, 0.30, True),
        ("D3b_motion_semantic_temporal_conflict_soft", 0.90, 0.25, False),
        ("D4_strict_motion_plus_semantic_temporal", 0.80, 0.70, True),
        ("D4b_strict_semantic_conflict_soft", 0.90, 0.40, False),
    ]:
        uf = StringUF(components)
        count = 0
        for row in pair_diag_rows:
            if parse_float(row.get("common_fate_score")) < min_motion:
                continue
            if parse_float(row.get("A5_d4rt_semantic_confirmation")) < min_a5:
                continue
            if veto_same_frame_conflict and parse_bool(row.get("same_frame_conflict")):
                continue
            left = str(row["component_left"])
            right = str(row["component_right"])
            if uf.find(left) == uf.find(right):
                continue
            uf.union(left, right)
            count += 1
            selected_rows.append({"variant": variant, **row})
            if count >= 120:
                break
        metrics = _evaluate(mask_vote_rows, uf)
        summary_rows.append(
            {
                "variant": variant,
                "selected_pair_count": count,
                "same_frame_conflict_veto": veto_same_frame_conflict,
                "min_common_fate_score": min_motion,
                "min_A5_d4rt_semantic_confirmation": min_a5,
                "motion_pair_AUC": auc,
                "common_fate_precision@top100": precision_top100,
                "ARI": metrics["ARI"],
                "purity": metrics["purity"],
                "completeness": metrics["completeness"],
                "delta_ARI_vs_nomotion": metrics["ARI"] - base["ARI"],
                "delta_completeness_vs_nomotion": metrics["completeness"] - base["completeness"],
                "purity_drop_vs_nomotion": base["purity"] - metrics["purity"],
                "temporal_span_mean": metrics["temporal_span_mean"],
                "gate_AUC_pass": auc is not None and auc >= 0.80,
                "gate_topk_pass": precision_top100 is not None and precision_top100 >= 0.80,
                "gate_delta_ARI_pass": metrics["ARI"] - base["ARI"] >= 0.02,
                "gate_delta_completeness_pass": metrics["completeness"] - base["completeness"] >= 0.04,
                "gate_purity_drop_pass": base["purity"] - metrics["purity"] <= 0.005,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    for row in summary_rows:
        row["gate_pass"] = bool(
            (row["gate_AUC_pass"] or row["gate_topk_pass"])
            and row["gate_delta_ARI_pass"]
            and row["gate_delta_completeness_pass"]
            and row["gate_purity_drop_pass"]
        )
    best = max(summary_rows, key=lambda row: parse_float(row.get("ARI")))
    gate = {
        "pass": any(row["gate_pass"] for row in summary_rows),
        "best_variant": best["variant"],
        "best_ARI": best["ARI"],
        "motion_pair_AUC": auc,
        "common_fate_precision@top100": precision_top100,
        "failure_label": None,
    }
    if not gate["pass"]:
        gate["failure_label"] = "NO_GO_COMMON_FATE"
    payload = {
        "phase": "v48_common_fate_motion",
        "created_at": utc_now(),
        "summary_rows": summary_rows,
        "gate": gate,
        "base_metrics": base,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "common_fate_summary.json", payload)
    write_csv(out / "common_fate_summary_rows.csv", summary_rows)
    write_csv(out / "common_fate_pair_rows.csv", pair_diag_rows)
    write_csv(out / "common_fate_selected_pairs.csv", selected_rows)
    print({"summary": str(out / "common_fate_summary.json"), "gate": gate})


if __name__ == "__main__":
    main()
