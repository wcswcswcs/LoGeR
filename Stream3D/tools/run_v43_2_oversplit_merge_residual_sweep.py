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
    _safe_div,
    _tube_error_proxy,
)


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


def _component_majority(labels_pred: dict[int, int], labels_ref: dict[int, int]) -> dict[int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id, pred in labels_pred.items():
        ref = int(labels_ref.get(int(tube_id), 0))
        if ref > 0:
            counts[int(pred)][ref] += 1
    return {comp: int(counter.most_common(1)[0][0]) for comp, counter in counts.items() if counter}


def _changed_from_base(
    state: SceneState,
    base_labels: dict[int, int],
    labels: dict[int, int],
) -> dict[str, Any]:
    new_to_base = _component_majority(labels, base_labels)
    base_to_gt = _component_majority(base_labels, state.gt_labels)
    new_to_gt = _component_majority(labels, state.gt_labels)
    labeled = 0
    changed = 0
    improved = 0
    regressed = 0
    for tube_id, gt in sorted(state.gt_labels.items()):
        if int(gt) <= 0:
            continue
        labeled += 1
        new_label = int(labels.get(int(tube_id), -1))
        base_label = int(base_labels.get(int(tube_id), -1))
        mapped_base = int(new_to_base.get(new_label, -999))
        if mapped_base == base_label:
            continue
        changed += 1
        base_ok = int(base_to_gt.get(base_label, -1)) == int(gt)
        new_ok = int(new_to_gt.get(new_label, -1)) == int(gt)
        improved += int(new_ok and not base_ok)
        regressed += int(base_ok and not new_ok)
    return {
        "changed_tube_count": int(changed),
        "changed_object_ratio": _safe_div(changed, labeled),
        "diagnostic_changed_tube_improved_count": int(improved),
        "diagnostic_changed_tube_regressed_count": int(regressed),
        "diagnostic_changed_tube_net_improved": int(improved - regressed),
        "diagnostic_merge_precision": _safe_div(improved, changed),
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
            "diagnostic_merge_precision": _safe_div(improved, changed),
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
            "memory_candidate_pairs": sum(int(item.get("memory_candidate_pairs") or 0) for item in items),
            "memory_accepted_merges": sum(int(item.get("memory_accepted_merges") or 0) for item in items),
            "memory_rgb_fallback_edges": sum(int(item.get("memory_rgb_fallback_edges") or 0) for item in items),
        }
        out.append(row)
    return out


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def _variant_name(rgb: float, gap: int, topk: int) -> str:
    return f"OSM_rgb{rgb:.4f}_gap{gap}_top{topk}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--local-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_oversplit_merge_residual_sweep")
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
    parser.add_argument("--seed", type=int, default=4324)
    parser.add_argument("--rgbs", default="0.985,0.990,0.992,0.995,0.997")
    parser.add_argument("--gaps", default="1,2,3,4")
    parser.add_argument("--topks", default="1,2,3")
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
    topks = [int(v) for v in str(args.topks).split(",") if v]

    scene_rows: list[dict[str, Any]] = []
    for state in states:
        base_components, _base_memory_info = _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.99,
            max_frame_gap=2,
            max_rgb_fallback_per_component=1,
        )
        base_labels, _base_unknown_ratio = _labels_for_components(
            base_components,
            state.support_by_tube,
            state.observation_count_by_tube,
            state.gt_labels,
            min_support=1,
            min_fraction=float(state.adaptive_fraction),
        )
        for rgb in rgbs:
            for gap in gaps:
                for topk in topks:
                    variant = _variant_name(rgb, gap, topk)
                    components, memory_info = _merge_components_rgb_temporal_topk(
                        state,
                        state.components,
                        min_rgb_similarity=float(rgb),
                        max_frame_gap=int(gap),
                        max_rgb_fallback_per_component=int(topk),
                    )
                    labels, unknown_ratio = _labels_for_components(
                        components,
                        state.support_by_tube,
                        state.observation_count_by_tube,
                        state.gt_labels,
                        min_support=1,
                        min_fraction=float(state.adaptive_fraction),
                    )
                    labeled = len([tid for tid, gt in state.gt_labels.items() if int(gt) > 0])
                    info = {
                        **memory_info,
                        **_changed_from_base(state, base_labels, labels),
                        "unknown_count": int(round(float(unknown_ratio) * labeled)),
                        "merge_min_rgb_similarity": float(rgb),
                        "merge_max_frame_gap": int(gap),
                        "merge_max_rgb_fallback_per_component": int(topk),
                    }
                    scene_rows.append(_evaluate_labels(state, variant, components, labels, info))

    summary_rows = _aggregate_rows(scene_rows)
    for row in summary_rows:
        row["semantic_phase_gate_proxy_pass"] = bool(
            float(row.get("4D_ARI") or -999.0) >= 0.42599481039581194 + 0.035
            and float(row.get("4D_completeness") or -999.0) >= 0.5056972999752292 + 0.015
            and float(row.get("4D_purity") or -999.0) >= 0.8673519940549913 - 0.003
            and float(row.get("changed_object_ratio") or 999.0) <= 0.20
        )
    best_by_ari = max(summary_rows, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    best_by_gate_balance = max(
        summary_rows,
        key=lambda row: (
            float(row.get("4D_ARI") or -999.0)
            + float(row.get("4D_completeness") or -999.0)
            + float(row.get("4D_purity") or -999.0)
        ),
        default={},
    )
    passing = [row for row in summary_rows if row.get("semantic_phase_gate_proxy_pass")]
    best_passing = max(passing, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    payload = {
        "phase": "v43_2_oversplit_merge_residual_sweep",
        "status": "PASS_OVERSPLIT_MERGE_RESIDUAL_SWEEP" if passing else "NO_GO_OVERSPLIT_MERGE_RESIDUAL_SWEEP",
        "variant_count": int(len(summary_rows)),
        "scene_count": int(len(states)),
        "best_by_ari": best_by_ari,
        "best_by_gate_balance": best_by_gate_balance,
        "best_passing_semantic_phase_proxy": best_passing,
        "passing_semantic_phase_proxy_count": int(len(passing)),
        "policy": {
            "prediction_uses_gt": False,
            "gt_used_only_for_diagnostic_precision_and_scoring": True,
            "residual_scope": "oversplit/id-switch candidate merge using RGB temporal top-k over v37 components",
            "component_source": "v37 base components with temporal RGB fallback sweep",
        },
    }
    _write_json(output_root / "oversplit_merge_residual_sweep_summary.json", payload)
    _write_csv(output_root / "oversplit_merge_residual_summary_rows.csv", summary_rows)
    _write_csv(output_root / "oversplit_merge_residual_scene_rows.csv", _public_rows(scene_rows))
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
