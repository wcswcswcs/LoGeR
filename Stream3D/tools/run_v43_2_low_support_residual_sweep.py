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
    _component_descriptors,
    _component_stats,
    _merge_components_rgb_temporal_topk,
    _rgb_similarity,
    _safe_div,
    _tube_error_proxy,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _tube_rgb_mean(state: SceneState, tube_id: int) -> list[float] | None:
    values: list[list[float]] = []
    weights: list[int] = []
    for node_id, count in state.support_by_tube.get(int(tube_id), Counter()).items():
        rgb = state.diagnostics.get(int(node_id), {}).get("rgb_mean")
        if rgb is None:
            continue
        values.append([float(v) for v in rgb])
        weights.append(int(count))
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    return np.average(arr, axis=0, weights=w).tolist()


def _assign_low_support_residual(
    state: SceneState,
    components: list[list[int]],
    *,
    floor_fraction: float,
    min_margin: float,
    min_rgb_similarity: float,
    min_support: int,
) -> tuple[dict[int, int], dict[str, Any], list[dict[str, Any]]]:
    node_to_component: dict[int, int] = {}
    for comp_idx, component in enumerate(components):
        for node_id in component:
            node_to_component[int(node_id)] = int(comp_idx)
    _supports, _frames, component_rgbs = _component_descriptors(state, components)

    labels_pred: dict[int, int] = {}
    default_labels: dict[int, int] = {}
    next_unknown = len(components) + 1
    default_next_unknown = len(components) + 1
    attach_rows: list[dict[str, Any]] = []
    unknown_count = 0
    default_unknown_count = 0
    accepted_attach = 0
    rejected_floor = 0
    rejected_margin = 0
    rejected_rgb = 0
    changed = 0
    labeled = 0

    for tube_id, gt in sorted(state.gt_labels.items()):
        if int(gt) <= 0:
            continue
        labeled += 1
        comp_counts: Counter[int] = Counter()
        for node_id, count in state.support_by_tube.get(int(tube_id), Counter()).items():
            comp = node_to_component.get(int(node_id))
            if comp is not None:
                comp_counts[int(comp)] += int(count)
        if not comp_counts:
            labels_pred[int(tube_id)] = next_unknown
            default_labels[int(tube_id)] = default_next_unknown
            next_unknown += 1
            default_next_unknown += 1
            unknown_count += 1
            default_unknown_count += 1
            continue
        top = comp_counts.most_common(2)
        comp, count = int(top[0][0]), int(top[0][1])
        second = int(top[1][1]) if len(top) > 1 else 0
        obs = max(int(state.observation_count_by_tube.get(int(tube_id), 0)), 1)
        frac = float(count / obs)
        margin = float((count - second) / obs)

        if count >= 1 and frac >= float(state.adaptive_fraction):
            default_labels[int(tube_id)] = int(comp)
        else:
            default_labels[int(tube_id)] = default_next_unknown
            default_next_unknown += 1
            default_unknown_count += 1

        attach_reason = ""
        rgb_sim = None
        attach = False
        if count < int(min_support):
            rejected_floor += 1
            attach_reason = "reject_min_support"
        elif frac < float(floor_fraction):
            rejected_floor += 1
            attach_reason = "reject_floor_fraction"
        elif margin < float(min_margin):
            rejected_margin += 1
            attach_reason = "reject_margin"
        else:
            tube_rgb = _tube_rgb_mean(state, int(tube_id))
            rgb_sim = _rgb_similarity(tube_rgb, component_rgbs[comp])
            if min_rgb_similarity > 0.0 and (rgb_sim is None or float(rgb_sim) < float(min_rgb_similarity)):
                rejected_rgb += 1
                attach_reason = "reject_rgb"
            else:
                attach = True
                attach_reason = "accepted_low_support_attach"

        if default_labels[int(tube_id)] <= len(components):
            labels_pred[int(tube_id)] = default_labels[int(tube_id)]
        elif attach:
            labels_pred[int(tube_id)] = int(comp)
            accepted_attach += 1
            changed += 1
        else:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1

        if default_labels[int(tube_id)] > len(components) or attach:
            attach_rows.append(
                {
                    "scene": state.scene,
                    "tube_id": int(tube_id),
                    "gt_label_diagnostic": int(gt),
                    "top_component": int(comp),
                    "top_count": int(count),
                    "second_count": int(second),
                    "observation_count": int(obs),
                    "top_fraction": float(frac),
                    "margin_fraction": float(margin),
                    "rgb_similarity": rgb_sim,
                    "default_unknown": bool(default_labels[int(tube_id)] > len(components)),
                    "accepted_attach": bool(attach and default_labels[int(tube_id)] > len(components)),
                    "reason": attach_reason,
                }
            )

    precision_rows = [row for row in attach_rows if row.get("accepted_attach")]
    comp_majority_gt = _component_majority_gt(default_labels, state.gt_labels, len(components))
    correct = 0
    for row in precision_rows:
        majority = comp_majority_gt.get(int(row["top_component"]))
        row["diagnostic_component_majority_gt"] = majority
        row["diagnostic_attach_correct"] = bool(majority is not None and int(row["gt_label_diagnostic"]) == int(majority))
        if row["diagnostic_attach_correct"]:
            correct += 1
    info = {
        "floor_fraction": float(floor_fraction),
        "min_margin": float(min_margin),
        "min_rgb_similarity": float(min_rgb_similarity),
        "min_support": int(min_support),
        "default_unknown_count": int(default_unknown_count),
        "unknown_count": int(unknown_count),
        "accepted_low_support_attach_count": int(accepted_attach),
        "rejected_floor_count": int(rejected_floor),
        "rejected_margin_count": int(rejected_margin),
        "rejected_rgb_count": int(rejected_rgb),
        "changed_tube_count": int(changed),
        "changed_object_ratio": _safe_div(changed, labeled),
        "low_support_attach_precision": _safe_div(correct, len(precision_rows)),
        "low_support_attach_precision_denominator": int(len(precision_rows)),
    }
    return labels_pred, info, attach_rows


def _component_majority_gt(labels_pred: dict[int, int], gt_labels: dict[int, int], component_count: int) -> dict[int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt <= 0 or int(pred) > int(component_count):
            continue
        counts[int(pred)][gt] += 1
    return {comp: int(counter.most_common(1)[0][0]) for comp, counter in counts.items() if counter}


def _evaluate_labels(
    state: SceneState,
    variant: str,
    components: list[list[int]],
    labels_pred: dict[int, int],
    info: dict[str, Any],
) -> dict[str, Any]:
    metrics = _cluster_metrics(labels_pred, state.gt_labels)
    labeled_ids = [int(tid) for tid in sorted(labels_pred) if int(state.gt_labels.get(int(tid), 0)) > 0]
    row = {
        "scene": state.scene,
        "variant": variant,
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "unknown_tube_ratio": _safe_div(info.get("unknown_count"), len(labeled_ids)),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
        "predicted_object_count_labeled": len({int(labels_pred[tid]) for tid in labeled_ids if int(labels_pred[tid]) <= len(components)}),
        "predicted_unknown_count_labeled": len({int(labels_pred[tid]) for tid in labeled_ids if int(labels_pred[tid]) > len(components)}),
        "component_count": int(len(components)),
        "temporal_span_mean": _component_stats(state.nodes, components, state.frame_rank).get("masklet_temporal_span_mean"),
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
        accepted = sum(int(row.get("accepted_low_support_attach_count") or 0) for row in items)
        correct_num = sum(
            float(row.get("low_support_attach_precision") or 0.0)
            * int(row.get("low_support_attach_precision_denominator") or 0)
            for row in items
        )
        correct_den = sum(int(row.get("low_support_attach_precision_denominator") or 0) for row in items)
        row = {
            "variant": variant,
            "4D_ARI": metrics.get("ari"),
            "4D_purity": metrics.get("purity"),
            "4D_completeness": metrics.get("completeness"),
            "unknown_tube_ratio": _safe_div(weighted_unknown, total_labeled),
            "changed_object_ratio": _safe_div(weighted_changed, total_labeled),
            "low_support_attach_precision": _safe_div(correct_num, correct_den),
            "low_support_attach_precision_denominator": int(correct_den),
            "accepted_low_support_attach_count": int(accepted),
            "scene0081_ARI": next((item.get("4D_ARI") for item in items if item.get("scene") == "scene0081_01"), None),
            "scene0591_ARI": next((item.get("4D_ARI") for item in items if item.get("scene") == "scene0591_00"), None),
            "temporal_span_mean": float(np.mean([float(item["temporal_span_mean"]) for item in items if item.get("temporal_span_mean") is not None])),
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


def _variant_name(floor: float, margin: float, rgb: float, support: int) -> str:
    return f"LS_floor{floor:.3f}_margin{margin:.3f}_rgb{rgb:.2f}_s{support}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--local-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_low_support_residual_sweep")
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
    parser.add_argument("--seed", type=int, default=4322)
    parser.add_argument("--floors", default="0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30")
    parser.add_argument("--margins", default="0.00,0.02,0.05,0.10")
    parser.add_argument("--rgbs", default="0.00,0.85,0.90,0.95")
    parser.add_argument("--supports", default="1,2")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_row_count = 0
    states: list[SceneState] = []
    for scene in _read_split(Path(args.split)):
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        states.append(state)

    floors = [float(v) for v in str(args.floors).split(",") if v]
    margins = [float(v) for v in str(args.margins).split(",") if v]
    rgbs = [float(v) for v in str(args.rgbs).split(",") if v]
    supports = [int(v) for v in str(args.supports).split(",") if v]

    scene_rows: list[dict[str, Any]] = []
    attach_rows: list[dict[str, Any]] = []
    for state in states:
        components, memory_info = _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.99,
            max_frame_gap=2,
            max_rgb_fallback_per_component=1,
        )
        for floor in floors:
            for margin in margins:
                for rgb in rgbs:
                    for support in supports:
                        variant = _variant_name(floor, margin, rgb, support)
                        labels, info, rows = _assign_low_support_residual(
                            state,
                            components,
                            floor_fraction=floor,
                            min_margin=margin,
                            min_rgb_similarity=rgb,
                            min_support=support,
                        )
                        info.update({key: value for key, value in memory_info.items() if key.startswith("memory_")})
                        scene_rows.append(_evaluate_labels(state, variant, components, labels, info))
                        for row in rows:
                            if row.get("accepted_attach"):
                                row["variant"] = variant
                                attach_rows.append(row)

    summary_rows = _aggregate_rows(scene_rows)
    for row in summary_rows:
        row["semantic_phase_gate_proxy_pass"] = bool(
            float(row.get("4D_ARI") or -999.0) >= 0.42599481039581194 + 0.035
            and float(row.get("4D_completeness") or -999.0) >= 0.5056972999752292 + 0.015
            and float(row.get("4D_purity") or -999.0) >= 0.8673519940549913 - 0.003
            and float(row.get("changed_object_ratio") or 999.0) <= 0.20
        )
    best_by_ari = max(summary_rows, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    passing = [row for row in summary_rows if row.get("semantic_phase_gate_proxy_pass")]
    best_passing = max(passing, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    payload = {
        "phase": "v43_2_low_support_residual_sweep",
        "status": "PASS_LOW_SUPPORT_RESIDUAL_SWEEP" if passing else "NO_GO_LOW_SUPPORT_RESIDUAL_SWEEP",
        "variant_count": int(len(summary_rows)),
        "scene_count": int(len(states)),
        "best_by_ari": best_by_ari,
        "best_passing_semantic_phase_proxy": best_passing,
        "passing_semantic_phase_proxy_count": int(len(passing)),
        "policy": {
            "prediction_uses_gt": False,
            "gt_used_only_for_diagnostic_precision_and_scoring": True,
            "residual_scope": "default-unknown low-support tubes only",
            "component_source": "v37 I4 sparse rgb temporal gap2 rgb099 top1 components",
        },
    }
    _write_json(output_root / "low_support_residual_sweep_summary.json", payload)
    _write_csv(output_root / "low_support_residual_summary_rows.csv", summary_rows)
    _write_csv(output_root / "low_support_residual_scene_rows.csv", _public_rows(scene_rows))
    _write_csv(output_root / "accepted_low_support_attach_rows.csv", attach_rows)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
