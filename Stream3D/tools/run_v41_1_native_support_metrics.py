from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.object_field_native_export import (
    NativeObjectFieldExportConfig,
    export_object_fields_to_native_points,
)
from stream4d_native.semantic_material_inference import SemanticMaterialInferenceConfig, run_semantic_material_inference
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_external_downstream_assignment import _load_gt
from tools.run_v41_1_native_object_field_export_smoke import _build_no_gt_components, _make_candidates_and_scores


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _labels_from_assignments(
    assignments: dict[int, int | str],
    gt_labels: dict[int, int],
) -> tuple[dict[int, int], float, dict[str, int]]:
    labels_pred: dict[int, int] = {}
    unknown_count = 0
    assigned_count = 0
    next_unknown = 1_000_000
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        value = assignments.get(int(tube_id), "unknown")
        if isinstance(value, str):
            labels_pred[int(tube_id)] = int(next_unknown)
            next_unknown += 1
            unknown_count += 1
        else:
            labels_pred[int(tube_id)] = int(value)
            assigned_count += 1
    labeled = int(sum(1 for value in gt_labels.values() if int(value) > 0))
    return labels_pred, float(unknown_count / max(labeled, 1)), {
        "labeled_tube_count": int(labeled),
        "assigned_labeled_tube_count": int(assigned_count),
        "unknown_labeled_tube_count": int(unknown_count),
    }


def _offset_labels(
    scene_index: int,
    labels_pred: dict[int, int],
    gt_labels: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    key_base = int(scene_index) * 10_000_000
    pred_base = int(scene_index) * 10_000_000
    gt_base = int(scene_index) * 10_000_000
    out_pred = {key_base + int(tube_id): pred_base + int(pred) for tube_id, pred in labels_pred.items()}
    out_gt = {
        key_base + int(tube_id): gt_base + int(gt)
        for tube_id, gt in gt_labels.items()
        if int(gt) > 0 and int(tube_id) in labels_pred
    }
    return out_pred, out_gt


def _read_v37_reference(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path_text in {
        "v37_3d_decision": args.v37_3d_decision,
        "v37_4d_decision": args.v37_4d_decision,
    }.items():
        path = Path(path_text)
        if not path.exists():
            out[name] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[name] = {
            "status": "ok",
            "path": str(path),
            "final_status": payload.get("final_status"),
            "best_metrics": payload.get("best_metrics", {}),
        }
    return out


def _run_scene(scene: str, scene_index: int, args: argparse.Namespace) -> tuple[dict[str, Any], dict[int, int], dict[int, int]]:
    state = _build_no_gt_components(scene, args)
    candidates, attachment_scores, candidate_rows = _make_candidates_and_scores(
        state["components"],
        state["support_by_region"],
        state["observation_count_by_tube"],
        max_candidates=int(args.max_candidates),
        max_tubes_per_candidate=int(args.max_tubes_per_candidate),
        include_forbidden_birth_probe=bool(args.include_forbidden_birth_probe),
    )
    inference = run_semantic_material_inference(
        candidates,
        attachment_scores,
        config=SemanticMaterialInferenceConfig(
            attach_threshold=float(args.attach_threshold),
            attach_margin=float(args.attach_margin),
            max_fields=int(args.max_fields),
            duplicate_support_jaccard=float(args.duplicate_support_jaccard),
            duplicate_material_jaccard=float(args.duplicate_material_jaccard),
            adaptive_attach_threshold=float(args.adaptive_attach_threshold),
            adaptive_attach_score_quantile=float(args.adaptive_attach_score_quantile),
            adaptive_attach_quantile_min=float(args.adaptive_attach_quantile_min),
        ),
        diagnostic_metrics={"AP_bridge": None},
    )
    native_export = export_object_fields_to_native_points(
        inference.object_fields,
        state["tubes"],
        config=NativeObjectFieldExportConfig(
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            require_semantic_birth=True,
            require_canonical=True,
            require_method_safe_alignment=True,
        ),
    )
    gt_labels = _load_gt(scene, state["tubes"], args)
    labels_pred, unknown_ratio, label_info = _labels_from_assignments(inference.tube_assignments, gt_labels)
    metrics = _cluster_metrics(labels_pred, gt_labels) if labels_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    global_pred, global_gt = _offset_labels(scene_index, labels_pred, gt_labels)
    scene_dir = Path(args.output_root) / "scene_details" / scene
    _write_json(
        scene_dir / "object_fields.json",
        [
            {
                "object_id": int(field.object_id),
                "primary_field_id": int(field.primary_field_id),
                "semantic_masklet_ids": [int(v) for v in field.semantic_masklet_ids],
                "attached_tube_ids": [int(v) for v in field.attached_tube_ids],
                "confidence": float(field.confidence),
            }
            for field in inference.object_fields
        ],
    )
    _write_json(scene_dir / "native_export_summary.json", native_export.summary)
    _write_csv(scene_dir / "candidate_rows.csv", candidate_rows)
    row = {
        "scene": scene,
        "status": "ok",
        "prediction_uses_gt": False,
        "prediction_uses_rgbd": False,
        "prediction_uses_pose": False,
        "prediction_uses_scannet_mesh": False,
        "gt_used_only_for_scoring": True,
        "region_count": int(len(state["nodes"])),
        "tube_count": int(len(state["tubes"])),
        "support_region_count": int(len(state["support_by_region"])),
        "support_tube_count": int(len(state["support_by_tube"])),
        "component_count": int(len(state["components"])),
        "boundary_split_used": bool(state.get("boundary_split_used", False)),
        "boundary_split_variant": str(state.get("boundary_split_variant", "")),
        "boundary_parent_region_count": int(state.get("boundary_parent_region_count", len(state["nodes"]))),
        "boundary_output_region_count": int(state.get("boundary_output_region_count", len(state["nodes"]))),
        "boundary_split_parent_count": int(state.get("boundary_split_parent_count", 0)),
        "boundary_added_region_count": int(state.get("boundary_added_region_count", 0)),
        "boundary_dropped_pixel_count": int(state.get("boundary_dropped_pixel_count", 0)),
        "rgb_filter_used": bool(state.get("rgb_filter_used", False)),
        "rgb_rejected_edge_count": int(state.get("rgb_rejected_edge_count", 0)),
        "rgb_split_components": int(state.get("rgb_split_components", 0)),
        "rgb_split_new_components": int(state.get("rgb_split_new_components", 0)),
        "rgb_outlier_components": int(state.get("rgb_outlier_components", 0)),
        "rgb_outlier_nodes": int(state.get("rgb_outlier_nodes", 0)),
        "rgb_unknown_components": int(state.get("rgb_unknown_components", 0)),
        "rgb_unknown_nodes": int(state.get("rgb_unknown_nodes", 0)),
        "candidate_count": int(len(candidates)),
        "selected_object_field_count": int(len(inference.object_fields)),
        "predictions_per_scene": int(inference.metrics.get("predictions_per_scene", len(inference.object_fields))),
        "conflict_rate": float(inference.metrics.get("conflict_rate", 0.0)),
        "duplicate_drop_count": int(inference.constraint_audit.get("duplicate_drop_count", 0)),
        "material_duplicate_drop_count": int(inference.constraint_audit.get("material_duplicate_drop_count", 0)),
        "duplicate_rate": float(inference.constraint_audit.get("duplicate_rate", 0.0)),
        "effective_attach_threshold": float(inference.constraint_audit.get("effective_attach_threshold", args.attach_threshold)),
        "adaptive_attach_used": bool(inference.constraint_audit.get("adaptive_attach_used", False)),
        "adaptive_attach_score_quantile": inference.constraint_audit.get("adaptive_attach_score_quantile"),
        "exported_object_count": int(native_export.summary.get("exported_object_count", 0)),
        "exported_tube_count": int(native_export.summary.get("exported_tube_count", 0)),
        "native_point_count": int(native_export.summary.get("native_point_count", 0)),
        "birth_from_d4rt_tube_count": int(inference.metrics.get("birth_from_d4rt_tube_count", 0)),
        "rejected_forbidden_birth_candidate_count": int(
            inference.metrics.get("rejected_forbidden_birth_candidate_count", 0)
        ),
        "unknown_tube_ratio_all_assignments": float(inference.metrics.get("unknown_tube_ratio", 0.0)),
        "unknown_tube_ratio_labeled": float(unknown_ratio),
        **{f"label_{key}": value for key, value in label_info.items()},
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "overmerge": metrics.get("overmerge"),
        "oversplit": metrics.get("oversplit"),
        "metric_labeled_tube_count": metrics.get("labeled_tube_count"),
    }
    return row, global_pred, global_gt


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = [args.scene] if args.scene else _read_split(Path(args.split))
    rows: list[dict[str, Any]] = []
    all_pred: dict[int, int] = {}
    all_gt: dict[int, int] = {}
    for scene_index, scene in enumerate(scenes):
        row, pred, gt = _run_scene(scene, scene_index, args)
        rows.append(row)
        all_pred.update(pred)
        all_gt.update(gt)
    aggregate_metrics = _cluster_metrics(all_pred, all_gt) if all_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    v37 = _read_v37_reference(args)
    v37_4d = v37.get("v37_4d_decision", {}).get("best_metrics", {})
    v37_3d = v37.get("v37_3d_decision", {}).get("best_metrics", {})
    birth_from_d4rt_tube_count_sum = int(sum(int(row.get("birth_from_d4rt_tube_count") or 0) for row in rows))
    rejected_forbidden_birth_candidate_count_sum = int(
        sum(int(row.get("rejected_forbidden_birth_candidate_count") or 0) for row in rows)
    )
    mean_predictions_per_scene = _mean([row.get("predictions_per_scene") for row in rows])
    duplicate_rate_mean = _mean([row.get("duplicate_rate") for row in rows])
    conflict_rate_mean = _mean([row.get("conflict_rate") for row in rows])
    summary = {
        "status": "PARTIAL_NATIVE_SUPPORT_METRICS_COMPUTED_AP_STILL_NOT_METHOD_COMPATIBLE",
        "scene_count": int(len(rows)),
        "scenes": scenes,
        "prediction_uses_gt": False,
        "prediction_uses_rgbd": False,
        "prediction_uses_pose": False,
        "prediction_uses_scannet_mesh": False,
        "gt_used_only_for_scoring": True,
        "metric_scope_note": "tube-level native-support object metrics; not ScanNet AP",
        "AP_bridge_status": "not_evaluated_native_support_metrics_only",
        "real_method_ap_status": "not_run",
        "aggregate_metrics": {
            "4D_ARI": aggregate_metrics.get("ari"),
            "4D_purity": aggregate_metrics.get("purity"),
            "4D_completeness": aggregate_metrics.get("completeness"),
            "overmerge": aggregate_metrics.get("overmerge"),
            "oversplit": aggregate_metrics.get("oversplit"),
            "labeled_tube_count": aggregate_metrics.get("labeled_tube_count"),
            "unknown_tube_ratio_labeled": _mean([row.get("unknown_tube_ratio_labeled") for row in rows]),
            "selected_object_field_count_mean": _mean([row.get("selected_object_field_count") for row in rows]),
            "mean_predictions_per_scene": mean_predictions_per_scene,
            "duplicate_rate_mean": duplicate_rate_mean,
            "conflict_rate_mean": conflict_rate_mean,
            "native_point_count_sum": int(sum(int(row.get("native_point_count") or 0) for row in rows)),
            "exported_tube_count_sum": int(sum(int(row.get("exported_tube_count") or 0) for row in rows)),
            "birth_from_d4rt_tube_count_sum": birth_from_d4rt_tube_count_sum,
            "rejected_forbidden_birth_candidate_count_sum": rejected_forbidden_birth_candidate_count_sum,
        },
        "gate": {
            "ari_pass": aggregate_metrics.get("ari") is not None and float(aggregate_metrics["ari"]) >= 0.40,
            "purity_pass": aggregate_metrics.get("purity") is not None and float(aggregate_metrics["purity"]) >= 0.85,
            "completeness_pass": aggregate_metrics.get("completeness") is not None
            and float(aggregate_metrics["completeness"]) >= 0.50,
            "unknown_pass": _mean([row.get("unknown_tube_ratio_labeled") for row in rows]) is not None
            and float(_mean([row.get("unknown_tube_ratio_labeled") for row in rows])) <= 0.40,
            "prediction_count_pass": mean_predictions_per_scene is not None
            and float(mean_predictions_per_scene) <= 300.0,
            "duplicate_rate_pass": duplicate_rate_mean is not None and float(duplicate_rate_mean) <= 0.10,
            "conflict_rate_pass": conflict_rate_mean is not None and float(conflict_rate_mean) <= 0.15,
            "no_d4rt_tube_birth_pass": birth_from_d4rt_tube_count_sum == 0,
            "tube_birth_negative_control_pass": birth_from_d4rt_tube_count_sum == 0
            and rejected_forbidden_birth_candidate_count_sum >= int(len(rows)),
            "no_forbidden_prediction_source": birth_from_d4rt_tube_count_sum == 0,
        },
        "v37_reference": v37,
        "comparison_to_v37": {
            "delta_vs_v37_4d_ARI": None
            if aggregate_metrics.get("ari") is None or v37_4d.get("4D_ARI") is None
            else float(aggregate_metrics["ari"]) - float(v37_4d["4D_ARI"]),
            "delta_vs_v37_4d_purity": None
            if aggregate_metrics.get("purity") is None or v37_4d.get("4D_purity") is None
            else float(aggregate_metrics["purity"]) - float(v37_4d["4D_purity"]),
            "delta_vs_v37_4d_completeness": None
            if aggregate_metrics.get("completeness") is None or v37_4d.get("4D_completeness") is None
            else float(aggregate_metrics["completeness"]) - float(v37_4d["4D_completeness"]),
            "delta_vs_v37_3d_ARI": None
            if aggregate_metrics.get("ari") is None or v37_3d.get("ARI") is None
            else float(aggregate_metrics["ari"]) - float(v37_3d["ARI"]),
        },
        "scene_rows": rows,
    }
    summary["gate"]["pass_native_support_metric_gate"] = bool(
        summary["gate"]["ari_pass"]
        and summary["gate"]["purity_pass"]
        and summary["gate"]["completeness_pass"]
        and summary["gate"]["unknown_pass"]
        and summary["gate"]["prediction_count_pass"]
        and summary["gate"]["duplicate_rate_pass"]
        and summary["gate"]["conflict_rate_pass"]
        and summary["gate"]["no_d4rt_tube_birth_pass"]
        and summary["gate"]["tube_birth_negative_control_pass"]
        and summary["gate"]["no_forbidden_prediction_source"]
    )
    if summary["gate"]["pass_native_support_metric_gate"]:
        summary["status"] = "PARTIAL_NATIVE_SUPPORT_METRIC_GATE_PASS_AP_STILL_NOT_METHOD_COMPATIBLE"
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "native_support_metrics_summary.json", summary)
    _write_csv(out_root / "native_support_scene_rows.csv", rows)
    comparison_rows = [
        {
            "row": "v41_1_native_support_metrics",
            "ARI": summary["aggregate_metrics"]["4D_ARI"],
            "purity": summary["aggregate_metrics"]["4D_purity"],
            "completeness": summary["aggregate_metrics"]["4D_completeness"],
            "unknown_tube_ratio": summary["aggregate_metrics"]["unknown_tube_ratio_labeled"],
            "scope": "native_support_tube_level",
        },
        {
            "row": "v37_4d_memory_reference",
            "ARI": v37_4d.get("4D_ARI"),
            "purity": v37_4d.get("4D_purity"),
            "completeness": v37_4d.get("4D_completeness"),
            "unknown_tube_ratio": v37_4d.get("unknown_tube_ratio"),
            "scope": "v37_4d_tube_level_reference",
        },
        {
            "row": "v37_3d_local_reference",
            "ARI": v37_3d.get("ARI"),
            "purity": v37_3d.get("purity"),
            "completeness": v37_3d.get("completeness"),
            "unknown_tube_ratio": v37_3d.get("unknown_tube_ratio"),
            "scope": "v37_3d_tube_level_reference",
        },
    ]
    _write_csv(out_root / "native_support_stream3d_comparison.csv", comparison_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v41_1_native_support_metrics")
    parser.add_argument("--v37-3d-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--v37-4d-decision", default="outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=41101)
    parser.add_argument("--short-max-delta", type=int, default=1)
    parser.add_argument("--short-min-shared", type=int, default=2)
    parser.add_argument("--short-min-jaccard", type=float, default=0.0)
    parser.add_argument("--disable-closure", action="store_true", default=True)
    parser.add_argument("--enable-closure", dest="disable_closure", action="store_false")
    parser.add_argument("--closure-min-delta", type=int, default=9)
    parser.add_argument("--closure-min-shared", type=int, default=2)
    parser.add_argument("--closure-min-jaccard", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-fields", type=int, default=64)
    parser.add_argument("--max-tubes-per-candidate", type=int, default=128)
    parser.add_argument("--attach-threshold", type=float, default=0.25)
    parser.add_argument("--attach-margin", type=float, default=0.05)
    parser.add_argument("--duplicate-support-jaccard", type=float, default=0.90)
    parser.add_argument("--duplicate-material-jaccard", type=float, default=1.01)
    parser.add_argument("--adaptive-attach-threshold", type=float, default=0.0)
    parser.add_argument("--adaptive-attach-score-quantile", type=float, default=0.25)
    parser.add_argument("--adaptive-attach-quantile-min", type=float, default=1.01)
    parser.add_argument("--include-forbidden-birth-probe", action="store_true", default=True)
    parser.add_argument("--no-forbidden-birth-probe", dest="include_forbidden_birth_probe", action="store_false")
    parser.add_argument("--compute-rgb-filter", action="store_true", default=False)
    parser.add_argument("--rgb-min-similarity", type=float, default=0.90)
    parser.add_argument("--rgb-split-min-similarity", type=float, default=0.90)
    parser.add_argument("--rgb-drop-incoherent", action="store_true", default=False)
    parser.add_argument("--rgb-drop-min-pairwise-similarity", type=float, default=0.85)
    parser.add_argument("--rgb-drop-max-component-nodes", type=int, default=0)
    parser.add_argument("--rgb-isolate-outliers", action="store_true", default=False)
    parser.add_argument("--rgb-outlier-min-center-similarity", type=float, default=0.85)
    parser.add_argument("--rgb-outlier-max-component-nodes", type=int, default=0)
    parser.add_argument("--enable-boundary-split", action="store_true", default=False)
    parser.add_argument("--boundary-variant", default="boundary_watershed_q85_split")
    parser.add_argument("--min-child-area", type=int, default=64)
    parser.add_argument("--boundary-min-split-area", type=int, default=1024)
    parser.add_argument("--boundary-gradient-quantile", type=float, default=0.90)
    parser.add_argument("--boundary-min-gradient", type=float, default=0.08)
    parser.add_argument("--boundary-edge-dilate", type=int, default=1)
    parser.add_argument("--boundary-min-child-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-min-core-coverage", type=float, default=0.35)
    parser.add_argument("--boundary-max-child-count", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
