from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.object_explanation_mdl import MDLParams, explain_objects_mdl, slot_to_posterior_record
from stream4d.scannet_stream import ScanNetStream
from stream4d.video_masklet import VideoMaskletBank
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
        "algorithm": "v13_object_explanation_mdl",
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
        "| scene | candidates | selected | objects | points | conflict | assigned | core | fringe | unknown | reject | explained | energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("seq_name")),
                    f"{float(row.get('num_candidate_slots', 0.0)):.0f}",
                    f"{float(row.get('num_selected_slots', 0.0)):.0f}",
                    f"{float(row.get('num_exported_objects', 0.0)):.0f}",
                    f"{float(row.get('num_exported_points', 0.0)):.0f}",
                    f"{float(row.get('export_conflict_rate', 0.0)):.6f}",
                    f"{float(row.get('assigned_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('core_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('fringe_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('unknown_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('reject_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('explained_measurement_ratio', 0.0)):.4f}",
                    f"{float(row.get('energy_mean', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    (out_dir / f"{args.output_config}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_masklets(args: argparse.Namespace, scene: str) -> VideoMaskletBank | None:
    if not args.masklet_root or not args.masklet_mode:
        return None
    path = Path(args.masklet_root) / args.masklet_mode / scene / "masklets.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return VideoMaskletBank.load(path)


def _export_scene(args: argparse.Namespace, scene: str, params: MDLParams) -> dict[str, Any]:
    bank_path = Path(args.bank_root) / scene / "measurement_bank.npz"
    bank = MeasurementBank.load(bank_path)
    masklets = _load_masklets(args, scene)
    slots, diag = explain_objects_mdl(bank, masklets=masklets, params=params)
    object_dict = {int(slot.object_id): slot_to_posterior_record(slot) for slot in slots}
    stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode=args.export_support_mode,
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.min_export_points_per_object),
        export_score_mode=args.export_score_mode,
        export_core_nn_radius=float(args.export_core_nn_radius),
        export_fringe_nn_radius=float(args.export_fringe_nn_radius),
        export_fringe_radius=float(args.export_fringe_radius),
        export_fringe_max_ratio=float(args.export_fringe_max_ratio),
        export_enable_wta=bool(args.export_enable_wta),
    )
    if args.export_support_mode == "posterior_support":
        export_diag = exporter.export_object_slot_posterior_support(object_dict, bank)
    elif args.export_support_mode == "mask_backproject":
        export_diag = exporter.export_object_dict_mask_backproject(object_dict)
    else:
        raise ValueError(f"Unsupported v13 export support mode: {args.export_support_mode}")
    slot_summary = [slot.summary() for slot in slots]
    scene_summary = {
        "seq_name": scene,
        "algorithm": "v13_object_explanation_mdl",
        "mode": "mdl",
        "uses_gt": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
        "bank_path": str(bank_path),
        "masklet_path": "" if masklets is None else str(Path(args.masklet_root) / args.masklet_mode / scene / "masklets.npz"),
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
    parser.add_argument("--masklet-root", default="outputs/v13_masklet_measurements")
    parser.add_argument("--masklet-mode", default="C3")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v13_object_explanation_mdl")
    parser.add_argument("--birth-min-surfels", type=int, default=12)
    parser.add_argument("--birth-min-boundary-safe-ratio", type=float, default=0.50)
    parser.add_argument("--birth-max-ambiguous-ratio", type=float, default=0.60)
    parser.add_argument("--core-posterior-threshold", type=float, default=0.62)
    parser.add_argument("--fringe-posterior-threshold", type=float, default=0.40)
    parser.add_argument("--reject-negative-threshold", type=float, default=0.45)
    parser.add_argument("--visible-outside-negative-weight", type=float, default=1.0)
    parser.add_argument("--boundary-risk-weight", type=float, default=0.45)
    parser.add_argument("--appearance-weight", type=float, default=0.25)
    parser.add_argument("--d4rt-temporal-weight", type=float, default=0.60)
    parser.add_argument("--min-core-surfels-per-object", type=int, default=8)
    parser.add_argument("--min-export-points-per-object", type=int, default=60)
    parser.add_argument("--measurement-min-surfels", type=int, default=3)
    parser.add_argument("--measurement-min-core-ratio", type=float, default=0.05)
    parser.add_argument("--max-slots-per-frame-mask", type=int, default=4)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--model-cost", type=float, default=0.80)
    parser.add_argument("--overlap-penalty", type=float, default=2.5)
    parser.add_argument("--unexplained-penalty", type=float, default=0.30)
    parser.add_argument("--measurement-reward", type=float, default=0.65)
    parser.add_argument("--surfel-reward", type=float, default=0.35)
    parser.add_argument("--negative-cost", type=float, default=1.1)
    parser.add_argument("--boundary-cost", type=float, default=0.45)
    parser.add_argument("--motion-reward", type=float, default=0.25)
    parser.add_argument("--appearance-reward", type=float, default=0.15)
    parser.add_argument("--max-core-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--max-slots", type=int, default=96)
    parser.add_argument("--export-support-mode", choices=["posterior_support", "mask_backproject"], default="posterior_support")
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-max-ratio", type=float, default=0.35)
    parser.add_argument("--export-enable-wta", action="store_true")
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-score-mode", choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"], default="reliability")
    args = parser.parse_args()

    params = MDLParams(
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
        min_core_surfels_per_object=int(args.min_core_surfels_per_object),
        min_export_points_per_object=int(args.min_export_points_per_object),
        measurement_min_surfels=int(args.measurement_min_surfels),
        measurement_min_core_ratio=float(args.measurement_min_core_ratio),
        max_slots_per_frame_mask=int(args.max_slots_per_frame_mask),
        boundary_safe_px=float(args.boundary_safe_px),
        model_cost=float(args.model_cost),
        overlap_penalty=float(args.overlap_penalty),
        unexplained_penalty=float(args.unexplained_penalty),
        measurement_reward=float(args.measurement_reward),
        surfel_reward=float(args.surfel_reward),
        negative_cost=float(args.negative_cost),
        boundary_cost=float(args.boundary_cost),
        motion_reward=float(args.motion_reward),
        appearance_reward=float(args.appearance_reward),
        max_core_overlap_ratio=float(args.max_core_overlap_ratio),
        max_slots=int(args.max_slots),
    )
    rows = [_export_scene(args, scene, params) for scene in read_seq_list(Path(args.seq_list))]
    aggregate = _aggregate(rows)
    _write_summary(args, rows, aggregate)
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.bank_root), str(args.masklet_root)],
        pre_points_policy="recompute",
        support_policy=f"v13_mdl:{args.export_support_mode}",
        notes=(
            "v13 MDL-style object explanation over D4RT measurement bank and video masklet measurements. "
            "GT is not read; ScanNet npz is an RGB-D evaluation adapter for posterior surfel ownership."
        ),
        extra={
            "algorithm": "v13_object_explanation_mdl",
            "method_family": "object_explanation_mdl",
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": True,
            "is_diagnostic_only": False,
            "summary_path": str(Path(args.summary_root) / f"{args.output_config}_summary.json"),
            "seq_list": str(args.seq_list),
            "params": params.__dict__,
            "masklet_mode": args.masklet_mode,
            "export_support_mode": args.export_support_mode,
            "export_enable_wta": bool(args.export_enable_wta),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
