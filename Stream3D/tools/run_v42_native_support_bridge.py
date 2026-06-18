from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.object_field import ObjectField
from stream4d_native.object_field_native_export import (
    NativeObjectFieldExportConfig,
    export_object_fields_to_native_points,
)
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_external_downstream_assignment import _load_gt, _load_tubes
from tools.run_v42_full_factor_graph import _json_safe, _parse_json_list


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_v42_object_fields(
    object_field_root: Path,
    *,
    scene: str,
    variant: str,
    source: str,
) -> list[ObjectField]:
    rows = _read_csv(object_field_root / "object_field_rows.csv")
    fields: list[ObjectField] = []
    for row in rows:
        if str(row.get("scene", "")) != str(scene):
            continue
        if str(row.get("variant", "")) != str(variant):
            continue
        if str(row.get("source", "")) != str(source):
            continue
        fields.append(
            ObjectField(
                object_id=int(row["object_id"]),
                primary_field_id=int(row["primary_field_id"]),
                semantic_masklet_ids=_parse_json_list(row.get("semantic_masklet_ids", "")),
                attached_tube_ids=_parse_json_list(row.get("attached_tube_ids", "")),
                confidence=float(row.get("confidence", 0.0) or 0.0),
            )
        )
    return fields


def _labels_from_tube_assignments(
    object_fields: list[ObjectField],
    gt_labels: dict[int, int],
    *,
    tube_filter: set[int] | None = None,
    unknown_label_base: int = 1_000_000,
) -> tuple[dict[int, int], float, dict[str, int]]:
    assignments: dict[int, int] = {}
    for field in object_fields:
        for tube_id in field.attached_tube_ids:
            tid = int(tube_id)
            if tube_filter is not None and tid not in tube_filter:
                continue
            assignments.setdefault(tid, int(field.object_id))
    labels_pred: dict[int, int] = {}
    assigned = 0
    unknown = 0
    next_unknown = int(unknown_label_base)
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        if tube_filter is not None and int(tube_id) not in tube_filter:
            continue
        if int(tube_id) in assignments:
            labels_pred[int(tube_id)] = int(assignments[int(tube_id)])
            assigned += 1
        else:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown += 1
    labeled = int(assigned + unknown)
    return labels_pred, float(unknown / max(labeled, 1)), {
        "labeled_tube_count": int(labeled),
        "assigned_labeled_tube_count": int(assigned),
        "unknown_labeled_tube_count": int(unknown),
    }


def _offset_labels(
    scene_index: int,
    pred: dict[int, int],
    gt: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    key_base = int(scene_index) * 10_000_000
    pred_base = int(scene_index) * 10_000_000
    gt_base = int(scene_index) * 10_000_000
    return (
        {key_base + int(tube_id): pred_base + int(label) for tube_id, label in pred.items()},
        {
            key_base + int(tube_id): gt_base + int(label)
            for tube_id, label in gt.items()
            if int(label) > 0 and int(tube_id) in pred
        },
    )


def _mean(values: list[Any]) -> float | None:
    out: list[float] = []
    for value in values:
        try:
            current = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(current):
            out.append(current)
    return float(np.mean(out)) if out else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge v42 object fields to native D4RT support diagnostics.")
    parser.add_argument("--object-field-root", required=True)
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v42_semantic_occupancy_real_dino_q5_mf32_b1024/Q5")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--source", default="dinov2_maskcut")
    parser.add_argument("--max-tubes-per-window", type=int, default=1024)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    object_field_root = ROOT / str(args.object_field_root)
    output_root = ROOT / str(args.output_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    scene_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    aggregate_pred: dict[int, int] = {}
    aggregate_gt: dict[int, int] = {}

    for scene_index, scene in enumerate(scenes):
        fields = load_v42_object_fields(
            object_field_root,
            scene=scene,
            variant=str(args.variant),
            source=str(args.source),
        )
        tubes = _load_tubes(scene, args)
        native = export_object_fields_to_native_points(
            fields,
            tubes,
            config=NativeObjectFieldExportConfig(
                min_visibility=float(args.min_visibility),
                min_confidence=float(args.min_confidence),
                require_semantic_birth=True,
                require_canonical=True,
                require_method_safe_alignment=True,
            ),
        )
        gt_labels = _load_gt(scene, tubes, args)
        labels_pred, unknown_ratio, label_info = _labels_from_tube_assignments(
            fields,
            gt_labels,
            unknown_label_base=(scene_index + 1) * 1_000_000,
        )
        gt_labeled = {int(tube_id): int(gt) for tube_id, gt in gt_labels.items() if int(gt) > 0 and int(tube_id) in labels_pred}
        metrics = _cluster_metrics(labels_pred, gt_labeled) if labels_pred else {
            "ari": None,
            "purity": None,
            "completeness": None,
            "overmerge": None,
            "oversplit": None,
            "labeled_tube_count": 0,
        }
        global_pred, global_gt = _offset_labels(scene_index, labels_pred, gt_labeled)
        aggregate_pred.update(global_pred)
        aggregate_gt.update(global_gt)
        for row in native.object_rows:
            object_rows.append({"scene": scene, "variant": str(args.variant), "source": str(args.source), **row})
        for row in native.point_rows:
            point_rows.append({"scene": scene, "variant": str(args.variant), "source": str(args.source), **row})
        scene_rows.append(
            {
                "scene": scene,
                "variant": str(args.variant),
                "source": str(args.source),
                "input_object_field_count": int(len(fields)),
                "cache_tube_count": int(len(tubes)),
                "native_export_smoke_pass": bool(native.summary["native_export_smoke_pass"]),
                "exported_object_count": int(native.summary["exported_object_count"]),
                "exported_tube_count": int(native.summary["exported_tube_count"]),
                "native_point_count": int(native.summary["native_point_count"]),
                "missing_tube_count": int(native.summary["missing_tube_count"]),
                "rejected_tube_count": int(native.summary["rejected_tube_count"]),
                "tube_4D_ARI": metrics.get("ari"),
                "tube_purity": metrics.get("purity"),
                "tube_completeness": metrics.get("completeness"),
                "tube_overmerge": metrics.get("overmerge"),
                "tube_oversplit": metrics.get("oversplit"),
                "unknown_labeled_tube_ratio": float(unknown_ratio),
                **label_info,
                "uses_gt_for_prediction": False,
                "uses_gt_for_scoring": True,
                "is_method_ap_result": False,
                "AP_bridge_status": native.summary["AP_bridge_status"],
            }
        )

    aggregate_metrics = _cluster_metrics(aggregate_pred, aggregate_gt) if aggregate_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    summary = {
        "phase": "v42_native_support_bridge",
        "object_field_root": str(object_field_root),
        "cache_root": str(ROOT / str(args.cache_root)),
        "scene_count": int(len(scene_rows)),
        "input_object_field_count": int(sum(int(row["input_object_field_count"]) for row in scene_rows)),
        "exported_object_count": int(sum(int(row["exported_object_count"]) for row in scene_rows)),
        "exported_tube_count": int(sum(int(row["exported_tube_count"]) for row in scene_rows)),
        "native_point_count": int(sum(int(row["native_point_count"]) for row in scene_rows)),
        "native_export_smoke_pass": bool(scene_rows and all(bool(row["native_export_smoke_pass"]) for row in scene_rows)),
        "aggregate_tube_4D_ARI": aggregate_metrics.get("ari"),
        "aggregate_tube_purity": aggregate_metrics.get("purity"),
        "aggregate_tube_completeness": aggregate_metrics.get("completeness"),
        "aggregate_tube_overmerge": aggregate_metrics.get("overmerge"),
        "aggregate_tube_oversplit": aggregate_metrics.get("oversplit"),
        "mean_unknown_labeled_tube_ratio": _mean([row["unknown_labeled_tube_ratio"] for row in scene_rows]),
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "is_method_ap_result": False,
        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
        "phase8_gate_pass": False,
        "phase8_gate_blocker": "native D4RT support bridge is not ScanNet AP/native method AP; AP materialization remains unrun",
    }
    _write_csv(output_root / "native_bridge_scene_rows.csv", scene_rows)
    _write_csv(output_root / "native_bridge_object_rows.csv", object_rows)
    _write_csv(output_root / "native_bridge_point_rows.csv", point_rows)
    _write_json(output_root / "native_bridge_summary.json", summary)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "scene_rows": len(scene_rows),
                "native_export_smoke_pass": summary["native_export_smoke_pass"],
                "phase8_gate_pass": summary["phase8_gate_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
