from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v37_4d_if_allowed import (
    SceneState,
    _build_scene_state,
    _component_stats,
    _labels_for_components,
    _merge_components_rgb_temporal_topk,
    _rgb_similarity,
    _safe_div,
    _tube_error_proxy,
)
from tools.run_v37_temporal_curriculum import UnionFind


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _node_rgb(state: SceneState, node_id: int) -> list[float] | None:
    return state.diagnostics.get(int(node_id), {}).get("rgb_mean")


def _split_i4_components_by_rgb_incoherence(
    state: SceneState,
    components: list[list[int]],
    *,
    min_rgb_similarity: float,
    max_frame_gap: int,
    min_child_nodes: int,
    min_child_frames: int,
    max_parent_nodes: int,
) -> tuple[list[list[int]], list[int], dict[int, int], dict[str, Any], list[dict[str, Any]]]:
    refined: list[list[int]] = []
    child_to_parent: dict[int, int] = {}
    split_parent_ids: list[int] = []
    split_rows: list[dict[str, Any]] = []
    rejected_small_parent = 0
    rejected_no_rgb = 0
    rejected_single_child = 0
    rejected_child_floor = 0
    same_frame_conflict_components = 0
    same_frame_conflict_extra_nodes = 0

    for parent_id, component in enumerate(components):
        frame_counts = Counter(int(state.nodes[int(node_id)].frame_id) for node_id in component)
        conflict_extra = sum(int(count) - 1 for count in frame_counts.values() if int(count) > 1)
        if conflict_extra:
            same_frame_conflict_components += 1
            same_frame_conflict_extra_nodes += int(conflict_extra)

        if len(component) < max(2, int(min_child_nodes) * 2):
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(component))
            rejected_small_parent += 1
            continue
        if int(max_parent_nodes) > 0 and len(component) > int(max_parent_nodes):
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(component))
            rejected_small_parent += 1
            continue

        rgb_valid = [idx for idx, node_id in enumerate(component) if _node_rgb(state, int(node_id)) is not None]
        if len(rgb_valid) < 2:
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(component))
            rejected_no_rgb += 1
            continue

        uf = UnionFind(len(component))
        accepted_edges = 0
        candidate_edges = 0
        for i, left in enumerate(component):
            left_rgb = _node_rgb(state, int(left))
            if left_rgb is None:
                continue
            left_rank = int(state.frame_rank.get(int(state.nodes[int(left)].frame_id), int(state.nodes[int(left)].frame_id)))
            for j in range(i + 1, len(component)):
                right = int(component[j])
                right_rgb = _node_rgb(state, right)
                if right_rgb is None:
                    continue
                right_rank = int(
                    state.frame_rank.get(int(state.nodes[right].frame_id), int(state.nodes[right].frame_id))
                )
                if int(max_frame_gap) > 0 and abs(left_rank - right_rank) > int(max_frame_gap):
                    continue
                if int(state.nodes[int(left)].frame_id) == int(state.nodes[right].frame_id):
                    continue
                candidate_edges += 1
                sim = _rgb_similarity(left_rgb, right_rgb)
                if sim is not None and float(sim) >= float(min_rgb_similarity):
                    uf.union(i, j)
                    accepted_edges += 1

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, node_id in enumerate(component):
            groups[uf.find(idx)].append(int(node_id))
        parts = list(groups.values())
        if len(parts) < 2:
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(component))
            rejected_single_child += 1
            continue

        child_frames = [len({int(state.nodes[int(node_id)].frame_id) for node_id in part}) for part in parts]
        if min(len(part) for part in parts) < int(min_child_nodes) or min(child_frames) < int(min_child_frames):
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(component))
            rejected_child_floor += 1
            continue

        split_parent_ids.append(int(parent_id))
        row = {
            "scene": state.scene,
            "parent_component": int(parent_id),
            "parent_node_count": int(len(component)),
            "child_count": int(len(parts)),
            "child_node_counts": ",".join(str(len(part)) for part in sorted(parts, key=len, reverse=True)),
            "child_frame_counts": ",".join(str(value) for value in sorted(child_frames, reverse=True)),
            "candidate_rgb_edges": int(candidate_edges),
            "accepted_rgb_edges": int(accepted_edges),
            "same_frame_conflict_extra_nodes": int(conflict_extra),
        }
        split_rows.append(row)
        for part in parts:
            child_to_parent[len(refined)] = int(parent_id)
            refined.append(list(part))

    info = {
        "split_parent_count": int(len(split_parent_ids)),
        "split_new_component_count": int(len(refined) - len(components)),
        "split_rejected_small_parent": int(rejected_small_parent),
        "split_rejected_no_rgb": int(rejected_no_rgb),
        "split_rejected_single_child": int(rejected_single_child),
        "split_rejected_child_floor": int(rejected_child_floor),
        "same_frame_conflict_components": int(same_frame_conflict_components),
        "same_frame_conflict_extra_nodes": int(same_frame_conflict_extra_nodes),
    }
    return refined, split_parent_ids, child_to_parent, info, split_rows


def _component_majority_gt(labels_pred: dict[int, int], gt_labels: dict[int, int]) -> dict[int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt > 0:
            counts[int(pred)][gt] += 1
    return {comp: int(counter.most_common(1)[0][0]) for comp, counter in counts.items() if counter}


def _changed_tube_info(
    state: SceneState,
    base_labels: dict[int, int],
    split_labels: dict[int, int],
    child_to_parent: dict[int, int],
    split_parent_ids: set[int],
) -> dict[str, Any]:
    base_majority = _component_majority_gt(base_labels, state.gt_labels)
    split_majority = _component_majority_gt(split_labels, state.gt_labels)
    changed = 0
    improved = 0
    regressed = 0
    labeled = 0
    for tube_id, gt in sorted(state.gt_labels.items()):
        if int(gt) <= 0:
            continue
        labeled += 1
        split_label = int(split_labels.get(int(tube_id), -1))
        parent = child_to_parent.get(split_label)
        if parent is None or int(parent) not in split_parent_ids:
            continue
        changed += 1
        base_ok = int(base_majority.get(int(base_labels.get(int(tube_id), -1)), -1)) == int(gt)
        split_ok = int(split_majority.get(split_label, -1)) == int(gt)
        improved += int(split_ok and not base_ok)
        regressed += int(base_ok and not split_ok)
    return {
        "changed_tube_count": int(changed),
        "changed_object_ratio": _safe_div(changed, labeled),
        "diagnostic_changed_tube_improved_count": int(improved),
        "diagnostic_changed_tube_regressed_count": int(regressed),
        "diagnostic_changed_tube_net_improved": int(improved - regressed),
        "diagnostic_split_precision": _safe_div(improved, changed),
    }


def _evaluate_labels(
    state: SceneState,
    variant: str,
    components: list[list[int]],
    labels_pred: dict[int, int],
    info: dict[str, Any],
) -> dict[str, Any]:
    metrics = _cluster_metrics(labels_pred, state.gt_labels)
    labeled_ids = [int(tid) for tid in sorted(labels_pred) if int(state.gt_labels.get(int(tid), 0)) > 0]
    component_count = int(len(components))
    row = {
        "scene": state.scene,
        "variant": variant,
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "unknown_tube_ratio": _safe_div(info.get("unknown_count"), len(labeled_ids)),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
        "predicted_object_count_labeled": len(
            {int(labels_pred[tid]) for tid in labeled_ids if int(labels_pred[tid]) <= component_count}
        ),
        "predicted_unknown_count_labeled": len(
            {int(labels_pred[tid]) for tid in labeled_ids if int(labels_pred[tid]) > component_count}
        ),
        "component_count": component_count,
        "temporal_span_mean": _component_stats(state.nodes, components, state.frame_rank).get(
            "masklet_temporal_span_mean"
        ),
        **info,
        **_tube_error_proxy(labels_pred, state.gt_labels),
        "_labels_true": [int(state.gt_labels[int(tid)]) for tid in labeled_ids],
        "_labels_pred": [int(labels_pred[int(tid)]) for tid in labeled_ids],
    }
    return row


def _aggregate_rows(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        by_variant[str(row["variant"])].append(row)
    out = []
    for variant, items in sorted(by_variant.items()):
        all_true: list[int] = []
        all_pred: list[int] = []
        true_offset = 0
        pred_offset = 0
        for item in items:
            true_vals = [int(v) for v in item.get("_labels_true", [])]
            pred_vals = [int(v) for v in item.get("_labels_pred", [])]
            all_true.extend([value + true_offset for value in true_vals])
            all_pred.extend([value + pred_offset for value in pred_vals])
            true_offset += (max(true_vals) + 11) if true_vals else 11
            pred_offset += (max(pred_vals) + 11) if pred_vals else 11
        metrics = _cluster_metrics(
            {idx: pred for idx, pred in enumerate(all_pred)},
            {idx: true for idx, true in enumerate(all_true)},
        )
        total_labeled = sum(int(row.get("labeled_tube_count") or 0) for row in items)
        weighted_unknown = sum(float(row.get("unknown_tube_ratio") or 0.0) * int(row.get("labeled_tube_count") or 0) for row in items)
        weighted_changed = sum(float(row.get("changed_object_ratio") or 0.0) * int(row.get("labeled_tube_count") or 0) for row in items)
        changed = sum(int(row.get("changed_tube_count") or 0) for row in items)
        improved = sum(int(row.get("diagnostic_changed_tube_improved_count") or 0) for row in items)
        regressed = sum(int(row.get("diagnostic_changed_tube_regressed_count") or 0) for row in items)
        row = {
            "variant": variant,
            "4D_ARI": metrics.get("ari"),
            "4D_purity": metrics.get("purity"),
            "4D_completeness": metrics.get("completeness"),
            "unknown_tube_ratio": _safe_div(weighted_unknown, total_labeled),
            "changed_object_ratio": _safe_div(weighted_changed, total_labeled),
            "changed_tube_count": int(changed),
            "diagnostic_changed_tube_improved_count": int(improved),
            "diagnostic_changed_tube_regressed_count": int(regressed),
            "diagnostic_changed_tube_net_improved": int(improved - regressed),
            "diagnostic_split_precision": _safe_div(improved, changed),
            "split_parent_count": sum(int(item.get("split_parent_count") or 0) for item in items),
            "split_new_component_count": sum(int(item.get("split_new_component_count") or 0) for item in items),
            "same_frame_conflict_components": sum(int(item.get("same_frame_conflict_components") or 0) for item in items),
            "same_frame_conflict_extra_nodes": sum(int(item.get("same_frame_conflict_extra_nodes") or 0) for item in items),
            "scene0081_ARI": next((item.get("4D_ARI") for item in items if item.get("scene") == "scene0081_01"), None),
            "scene0591_ARI": next((item.get("4D_ARI") for item in items if item.get("scene") == "scene0591_00"), None),
            "temporal_span_mean": float(
                np.mean([float(item["temporal_span_mean"]) for item in items if item.get("temporal_span_mean") is not None])
            ),
            "mean_predictions_per_scene": float(np.mean([float(item["predicted_object_count_labeled"]) for item in items])),
            "mean_unknown_labels_per_scene": float(np.mean([float(item["predicted_unknown_count_labeled"]) for item in items])),
            "ID_switches": sum(int(item.get("ID_switches") or 0) for item in items),
            "fragmentation": float(np.mean([float(item["fragmentation"]) for item in items if item.get("fragmentation") is not None])),
            "merge_errors": sum(int(item.get("merge_errors") or 0) for item in items),
        }
        out.append(row)
    return out


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def _variant_name(rgb: float, gap: int, child_nodes: int, child_frames: int, parent_nodes: int) -> str:
    parent = "all" if int(parent_nodes) <= 0 else str(int(parent_nodes))
    return f"OMS_rgb{rgb:.3f}_gap{gap}_n{child_nodes}_f{child_frames}_p{parent}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--local-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_overmerge_split_residual_sweep")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-same-frame-pairs-per-frame", type=int, default=4000)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--max-shuffled-pair-rows-per-scene", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4323)
    parser.add_argument("--rgbs", default="0.990,0.992,0.995,0.997")
    parser.add_argument("--gaps", default="2,4,8,0")
    parser.add_argument("--min-child-nodes", default="2,3,5")
    parser.add_argument("--min-child-frames", default="1,2")
    parser.add_argument("--max-parent-nodes", default="0,20,50")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_row_count = 0
    states: list[SceneState] = []
    for scene in _read_split(Path(args.split)):
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        states.append(state)

    rgbs = [float(v) for v in str(args.rgbs).split(",") if v]
    gaps = [int(v) for v in str(args.gaps).split(",") if v]
    min_child_nodes = [int(v) for v in str(args.min_child_nodes).split(",") if v]
    min_child_frames = [int(v) for v in str(args.min_child_frames).split(",") if v]
    max_parent_nodes = [int(v) for v in str(args.max_parent_nodes).split(",") if v]

    scene_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    same_frame_scope_rows: list[dict[str, Any]] = []
    for state in states:
        base_components, memory_info = _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.99,
            max_frame_gap=2,
            max_rgb_fallback_per_component=1,
        )
        base_labels, base_unknown_ratio = _labels_for_components(
            base_components,
            state.support_by_tube,
            state.observation_count_by_tube,
            state.gt_labels,
            min_support=1,
            min_fraction=float(state.adaptive_fraction),
        )
        same_frame_scope_rows.append(
            {
                "scene": state.scene,
                "i4_component_count": int(len(base_components)),
                "same_frame_conflict_components": int(
                    sum(
                        1
                        for component in base_components
                        if any(
                            count > 1
                            for count in Counter(int(state.nodes[int(node_id)].frame_id) for node_id in component).values()
                        )
                    )
                ),
                "same_frame_conflict_extra_nodes": int(
                    sum(
                        sum(count - 1 for count in Counter(int(state.nodes[int(node_id)].frame_id) for node_id in component).values() if count > 1)
                        for component in base_components
                    )
                ),
            }
        )
        for rgb in rgbs:
            for gap in gaps:
                for child_nodes in min_child_nodes:
                    for child_frames in min_child_frames:
                        for parent_nodes in max_parent_nodes:
                            variant = _variant_name(rgb, gap, child_nodes, child_frames, parent_nodes)
                            components, split_parent_ids, child_to_parent, split_info, rows = _split_i4_components_by_rgb_incoherence(
                                state,
                                base_components,
                                min_rgb_similarity=rgb,
                                max_frame_gap=gap,
                                min_child_nodes=child_nodes,
                                min_child_frames=child_frames,
                                max_parent_nodes=parent_nodes,
                            )
                            labels, unknown_ratio = _labels_for_components(
                                components,
                                state.support_by_tube,
                                state.observation_count_by_tube,
                                state.gt_labels,
                                min_support=1,
                                min_fraction=float(state.adaptive_fraction),
                            )
                            info = {
                                **split_info,
                                "unknown_count": int(round(float(unknown_ratio) * len([tid for tid, gt in state.gt_labels.items() if int(gt) > 0]))),
                                **_changed_tube_info(
                                    state,
                                    base_labels,
                                    labels,
                                    child_to_parent,
                                    set(split_parent_ids),
                                ),
                                **{key: value for key, value in memory_info.items() if key.startswith("memory_")},
                                "split_min_rgb_similarity": float(rgb),
                                "split_max_frame_gap": int(gap),
                                "split_min_child_nodes": int(child_nodes),
                                "split_min_child_frames": int(child_frames),
                                "split_max_parent_nodes": int(parent_nodes),
                                "base_unknown_ratio": float(base_unknown_ratio),
                            }
                            scene_rows.append(_evaluate_labels(state, variant, components, labels, info))
                            for row in rows:
                                row["variant"] = variant
                                split_rows.append(row)

    summary_rows = _aggregate_rows(scene_rows)
    for row in summary_rows:
        row["semantic_phase_gate_proxy_pass"] = bool(
            float(row.get("4D_ARI") or -999.0) >= 0.42599481039581194 + 0.035
            and float(row.get("4D_completeness") or -999.0) >= 0.5056972999752292 + 0.015
            and float(row.get("4D_purity") or -999.0) >= 0.8673519940549913 - 0.003
            and float(row.get("changed_object_ratio") or 999.0) <= 0.20
        )
    best_by_ari = max(summary_rows, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    best_by_purity = max(summary_rows, key=lambda row: float(row.get("4D_purity") or -999.0), default={})
    passing = [row for row in summary_rows if row.get("semantic_phase_gate_proxy_pass")]
    best_passing = max(passing, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    same_frame_total = {
        "same_frame_conflict_components": sum(int(row.get("same_frame_conflict_components") or 0) for row in same_frame_scope_rows),
        "same_frame_conflict_extra_nodes": sum(int(row.get("same_frame_conflict_extra_nodes") or 0) for row in same_frame_scope_rows),
    }
    payload = {
        "phase": "v43_2_overmerge_split_residual_sweep",
        "status": "PASS_OVERMERGE_SPLIT_RESIDUAL_SWEEP" if passing else "NO_GO_OVERMERGE_SPLIT_RESIDUAL_SWEEP",
        "variant_count": int(len(summary_rows)),
        "scene_count": int(len(states)),
        "same_frame_scope": same_frame_total,
        "best_by_ari": best_by_ari,
        "best_by_purity": best_by_purity,
        "best_passing_semantic_phase_proxy": best_passing,
        "passing_semantic_phase_proxy_count": int(len(passing)),
        "policy": {
            "prediction_uses_gt": False,
            "gt_used_only_for_diagnostic_precision_and_scoring": True,
            "residual_scope": "I4 components only; split by RGB/temporal incoherence, with same-frame conflict counted as scope evidence",
            "component_source": "v37 I4 sparse rgb temporal gap2 rgb099 top1 components",
        },
    }
    _write_json(output_root / "overmerge_split_residual_sweep_summary.json", payload)
    _write_csv(output_root / "overmerge_split_residual_summary_rows.csv", summary_rows)
    _write_csv(output_root / "overmerge_split_residual_scene_rows.csv", _public_rows(scene_rows))
    _write_csv(output_root / "overmerge_split_accepted_rows.csv", split_rows)
    _write_csv(output_root / "same_frame_scope_rows.csv", same_frame_scope_rows)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
