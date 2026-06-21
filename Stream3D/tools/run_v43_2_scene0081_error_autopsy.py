from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v37_4d_if_allowed import (
    _build_scene_state,
    _merge_components_rgb_temporal_topk,
    _safe_div,
)
from tools.run_v37_temporal_curriculum import _labels_for_components


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


def _offset_labels(labels: dict[int, int], offset: int) -> dict[int, int]:
    return {int(k): int(v) + int(offset) for k, v in labels.items()}


def _oracle_unknown_reactivation(
    labels_pred: dict[int, int],
    gt_labels: dict[int, int],
    component_count: int,
) -> dict[int, int]:
    by_gt_known: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt > 0 and int(pred) <= int(component_count):
            by_gt_known[gt][int(pred)] += 1
    gt_to_best_pred = {gt: counter.most_common(1)[0][0] for gt, counter in by_gt_known.items() if counter}
    next_label = max([int(v) for v in labels_pred.values()] + [int(component_count)]) + 1
    gt_to_new: dict[int, int] = {}
    out: dict[int, int] = {}
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt <= 0:
            continue
        if int(pred) <= int(component_count):
            out[int(tube_id)] = int(pred)
            continue
        if gt in gt_to_best_pred:
            out[int(tube_id)] = int(gt_to_best_pred[gt])
        else:
            if gt not in gt_to_new:
                gt_to_new[gt] = next_label
                next_label += 1
            out[int(tube_id)] = int(gt_to_new[gt])
    return out


def _oracle_split_overmerged(
    labels_pred: dict[int, int],
    gt_labels: dict[int, int],
    component_count: int,
) -> dict[int, int]:
    pair_to_label: dict[tuple[int, int], int] = {}
    next_label = 1
    out: dict[int, int] = {}
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt <= 0:
            continue
        if int(pred) > int(component_count):
            out[int(tube_id)] = int(pred) + 1_000_000
            continue
        key = (int(pred), gt)
        if key not in pair_to_label:
            pair_to_label[key] = next_label
            next_label += 1
        out[int(tube_id)] = pair_to_label[key]
    return out


def _oracle_merge_known_oversplit(
    labels_pred: dict[int, int],
    gt_labels: dict[int, int],
    component_count: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    next_unknown = max(int(v) for v in gt_labels.values() if int(v) > 0) + 1
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt <= 0:
            continue
        if int(pred) <= int(component_count):
            out[int(tube_id)] = gt
        else:
            out[int(tube_id)] = next_unknown
            next_unknown += 1
    return out


def _oracle_full_gt(gt_labels: dict[int, int]) -> dict[int, int]:
    return {int(tube_id): int(gt) for tube_id, gt in gt_labels.items() if int(gt) > 0}


def _metrics_row(name: str, labels_pred: dict[int, int], gt_labels: dict[int, int]) -> dict[str, Any]:
    metrics = _cluster_metrics(labels_pred, gt_labels)
    return {
        "intervention": name,
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "overmerge": metrics.get("overmerge"),
        "oversplit": metrics.get("oversplit"),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
    }


def _gt_rows(labels_pred: dict[int, int], gt_labels: dict[int, int], component_count: int) -> list[dict[str, Any]]:
    gt_to_pred: dict[int, Counter[int]] = defaultdict(Counter)
    gt_to_unknown = Counter()
    for tube_id, gt in gt_labels.items():
        gt = int(gt)
        if gt <= 0:
            continue
        pred = int(labels_pred.get(int(tube_id), -1))
        gt_to_pred[gt][pred] += 1
        if pred > int(component_count):
            gt_to_unknown[gt] += 1
    rows = []
    for gt, counter in sorted(gt_to_pred.items()):
        total = sum(counter.values())
        top_pred, top_count = counter.most_common(1)[0]
        known_preds = [pred for pred in counter if int(pred) <= int(component_count)]
        rows.append(
            {
                "gt_label": int(gt),
                "tube_count": int(total),
                "top_pred_label": int(top_pred),
                "top_pred_count": int(top_count),
                "top_pred_fraction": _safe_div(top_count, total),
                "pred_cluster_count": int(len(counter)),
                "known_pred_cluster_count": int(len(known_preds)),
                "unknown_tube_count": int(gt_to_unknown[gt]),
                "unknown_fraction": _safe_div(gt_to_unknown[gt], total),
            }
        )
    return sorted(rows, key=lambda row: (float(row["top_pred_fraction"]), -int(row["tube_count"])))


def _pred_rows(labels_pred: dict[int, int], gt_labels: dict[int, int], component_count: int) -> list[dict[str, Any]]:
    pred_to_gt: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt > 0:
            pred_to_gt[int(pred)][gt] += 1
    rows = []
    for pred, counter in sorted(pred_to_gt.items()):
        total = sum(counter.values())
        top_gt, top_count = counter.most_common(1)[0]
        rows.append(
            {
                "pred_label": int(pred),
                "is_unknown_label": bool(int(pred) > int(component_count)),
                "tube_count": int(total),
                "gt_count": int(len(counter)),
                "top_gt": int(top_gt),
                "top_gt_count": int(top_count),
                "purity": _safe_div(top_count, total),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["gt_count"]), -int(row["tube_count"])))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state, _pair_count = _build_scene_state(args.scene, args, 0)
    components, memory_info = _merge_components_rgb_temporal_topk(
        state,
        state.components,
        min_rgb_similarity=float(args.i4_rgb),
        max_frame_gap=int(args.i4_gap),
        max_rgb_fallback_per_component=int(args.i4_topk),
    )
    labels_pred, unknown_ratio = _labels_for_components(
        components,
        state.support_by_tube,
        state.observation_count_by_tube,
        state.gt_labels,
        min_support=1,
        min_fraction=float(state.adaptive_fraction),
    )
    component_count = len(components)
    interventions = [
        _metrics_row("current_i4", labels_pred, state.gt_labels),
        _metrics_row(
            "oracle_unknown_reactivation_only",
            _oracle_unknown_reactivation(labels_pred, state.gt_labels, component_count),
            state.gt_labels,
        ),
        _metrics_row(
            "oracle_split_overmerged_known_components",
            _oracle_split_overmerged(labels_pred, state.gt_labels, component_count),
            state.gt_labels,
        ),
        _metrics_row(
            "oracle_merge_known_oversplit_components",
            _oracle_merge_known_oversplit(labels_pred, state.gt_labels, component_count),
            state.gt_labels,
        ),
        _metrics_row("oracle_full_gt_assignment", _oracle_full_gt(state.gt_labels), state.gt_labels),
    ]
    gt_rows = _gt_rows(labels_pred, state.gt_labels, component_count)
    pred_rows = _pred_rows(labels_pred, state.gt_labels, component_count)
    hardest_gt = gt_rows[:20]
    largest_overmerge = [row for row in pred_rows if not row["is_unknown_label"] and int(row["gt_count"]) > 1][:20]
    payload = {
        "phase": "v43_2_scene0081_error_autopsy",
        "scene": str(args.scene),
        "prediction_uses_gt": False,
        "gt_used_for_diagnostic_interventions": True,
        "current": interventions[0],
        "oracle_interventions": interventions,
        "component_count": int(component_count),
        "unknown_tube_ratio": float(unknown_ratio),
        "adaptive_fraction": float(state.adaptive_fraction),
        "memory_info": memory_info,
        "hardest_gt_rows_top20": hardest_gt,
        "largest_overmerge_rows_top20": largest_overmerge,
        "interpretation": {
            "unknown_oracle_delta_ari": (
                None
                if interventions[1]["4D_ARI"] is None or interventions[0]["4D_ARI"] is None
                else float(interventions[1]["4D_ARI"]) - float(interventions[0]["4D_ARI"])
            ),
            "split_oracle_delta_ari": (
                None
                if interventions[2]["4D_ARI"] is None or interventions[0]["4D_ARI"] is None
                else float(interventions[2]["4D_ARI"]) - float(interventions[0]["4D_ARI"])
            ),
            "merge_oracle_delta_ari": (
                None
                if interventions[3]["4D_ARI"] is None or interventions[0]["4D_ARI"] is None
                else float(interventions[3]["4D_ARI"]) - float(interventions[0]["4D_ARI"])
            ),
        },
    }
    _write_json(output_root / "scene0081_error_autopsy_summary.json", payload)
    _write_csv(output_root / "scene0081_oracle_intervention_rows.csv", interventions)
    _write_csv(output_root / "scene0081_gt_error_rows.csv", gt_rows)
    _write_csv(output_root / "scene0081_pred_error_rows.csv", pred_rows)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0081_01")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_scene0081_error_autopsy")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
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
    parser.add_argument("--seed", type=int, default=4325)
    parser.add_argument("--i4-rgb", type=float, default=0.99)
    parser.add_argument("--i4-gap", type=int, default=2)
    parser.add_argument("--i4-topk", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
