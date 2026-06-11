from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.evidence_terms import ExplanationParams
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.object_explanation import explain_objects
from stream4d.scannet_stream import ScanNetStream
from stream4d.export_scannet import ScanNetExporter
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm": "v12_object_explanation",
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
        "scenes": rows,
    }


def _write_summary(args: argparse.Namespace, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_config}_summary.json"
    json_path.write_text(json.dumps(json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (out_dir / f"{args.output_config}_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | slots | rejected | objects | points | conflict | assigned | core | reject | explained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("seq_name")),
                    str(row.get("num_active_slots")),
                    str(row.get("num_rejected_slots")),
                    f"{float(row.get('num_exported_objects', 0.0)):.0f}",
                    f"{float(row.get('num_exported_points', 0.0)):.0f}",
                    f"{float(row.get('export_conflict_rate', 0.0)):.6f}",
                    f"{float(row.get('assigned_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('core_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('reject_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('measurement_explained_ratio', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    (out_dir / f"{args.output_config}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_scene(args: argparse.Namespace, scene: str, params: ExplanationParams) -> dict[str, Any]:
    bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
    slots, diag = explain_objects(bank, params=params, mode=args.mode, seed=int(args.seed))
    object_dict = {int(slot.object_id): slot.to_object_dict_record() for slot in slots}
    stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.min_export_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    export_diag = exporter.export_object_dict_mask_backproject(object_dict)
    slot_summary = [slot.summary() for slot in slots]
    scene_summary = {
        "seq_name": scene,
        "algorithm": "v12_object_explanation",
        "mode": args.mode,
        "uses_gt": False,
        "is_method_result": not bool(args.diagnostic_candidate_only),
        "is_diagnostic_only": bool(args.diagnostic_candidate_only),
        "bank_path": str(Path(args.bank_root) / scene / "measurement_bank.npz"),
        "slot_summary": slot_summary,
        **diag,
        **export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / f"{args.output_config}_{scene}_summary.json"
    scene_summary["summary_path"] = str(scene_path)
    scene_path.write_text(json.dumps(json_safe(scene_summary), indent=2, sort_keys=True), encoding="utf-8")
    return scene_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--mode", required=True, choices=["no_negative", "with_negative", "shuffled_d4rt", "no_d4rt_temporal", "surfel_cluster_candidate"])
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v12_object_explanation")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--birth-min-surfels", type=int, default=16)
    parser.add_argument("--birth-min-boundary-safe-ratio", type=float, default=0.65)
    parser.add_argument("--birth-max-ambiguous-ratio", type=float, default=0.25)
    parser.add_argument("--core-posterior-threshold", type=float, default=0.70)
    parser.add_argument("--fringe-posterior-threshold", type=float, default=0.45)
    parser.add_argument("--reject-negative-threshold", type=float, default=0.40)
    parser.add_argument("--visible-outside-negative-weight", type=float, default=1.0)
    parser.add_argument("--boundary-risk-weight", type=float, default=0.5)
    parser.add_argument("--appearance-weight", type=float, default=0.3)
    parser.add_argument("--d4rt-temporal-weight", type=float, default=0.5)
    parser.add_argument("--max-slots-per-frame-mask", type=int, default=3)
    parser.add_argument("--min-core-surfels-per-object", type=int, default=12)
    parser.add_argument("--min-export-points-per-object", type=int, default=100)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--measurement-min-surfels", type=int, default=4)
    parser.add_argument("--measurement-min-core-ratio", type=float, default=0.08)
    parser.add_argument("--enable-target-births", action="store_true")
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-score-mode", choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"], default="reliability")
    parser.add_argument("--diagnostic-candidate-only", action="store_true")
    args = parser.parse_args()

    params = ExplanationParams(
        birth_min_surfels=int(args.birth_min_surfels),
        birth_min_boundary_safe_ratio=float(args.birth_min_boundary_safe_ratio),
        birth_max_ambiguous_ratio=float(args.birth_max_ambiguous_ratio),
        core_posterior_threshold=float(args.core_posterior_threshold),
        fringe_posterior_threshold=float(args.fringe_posterior_threshold),
        reject_negative_threshold=float(args.reject_negative_threshold),
        visible_outside_negative_weight=float(args.visible_outside_negative_weight),
        boundary_risk_weight=float(args.boundary_risk_weight),
        appearance_weight=float(args.appearance_weight),
        d4rt_temporal_weight=float(args.d4rt_temporal_weight),
        max_slots_per_frame_mask=int(args.max_slots_per_frame_mask),
        min_core_surfels_per_object=int(args.min_core_surfels_per_object),
        min_export_points_per_object=int(args.min_export_points_per_object),
        boundary_safe_px=float(args.boundary_safe_px),
        measurement_min_surfels=int(args.measurement_min_surfels),
        measurement_min_core_ratio=float(args.measurement_min_core_ratio),
        enable_target_births=bool(args.enable_target_births),
    )
    rows = [_export_scene(args, scene, params) for scene in read_seq_list(Path(args.seq_list))]
    aggregate = _aggregate(rows)
    _write_summary(args, rows, aggregate)
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=not bool(args.diagnostic_candidate_only),
        is_diagnostic_only=bool(args.diagnostic_candidate_only),
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.bank_root)],
        pre_points_policy="recompute",
        support_policy=f"v12_object_explanation:{args.mode}:mask_backproject_eval_adapter",
        notes=(
            "v12 deterministic object-slot explanation over D4RT measurement bank. "
            "GT is not read; ScanNet npz is an RGB-D evaluation adapter for the slot ownership field."
        ),
        extra={
            "algorithm": "v12_object_explanation",
            "method_family": "object_explanation",
            "mode": args.mode,
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": not bool(args.diagnostic_candidate_only),
            "is_diagnostic_only": bool(args.diagnostic_candidate_only),
            "summary_path": str(Path(args.summary_root) / f"{args.output_config}_summary.json"),
            "seq_list": str(args.seq_list),
            "params": params.__dict__,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
