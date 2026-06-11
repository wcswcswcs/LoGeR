from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.scannet_stream import ScanNetStream
from stream4d.surfel_atom_bank import atom_to_object_record, build_surfel_atom_bank
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
        "algorithm": "v14_surfel_atom_bank",
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
        "scenes": rows,
    }


def _write_summary(summary_root: Path, output_config: str, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / f"{output_config}_summary.json").write_text(
        json.dumps(json_safe(aggregate), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if key != "atom_preview"})
        with (summary_root / f"{output_config}_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        f"# {output_config}",
        "",
        "| scene | atoms | known | unknown | surfels/atom | known support % | entropy mean/p90 | traj var mean/p90 | boundary safe | neg outside | objects | points | conflict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    f"{float(row.get('num_atoms', 0)):.0f}",
                    f"{float(row.get('num_known_atoms', 0)):.0f}",
                    f"{float(row.get('num_unknown_atoms', 0)):.0f}",
                    f"{float(row.get('mean_surfels_per_atom', 0)):.2f}",
                    f"{float(row.get('known_atom_support_ratio', 0)) * 100.0:.4f}",
                    f"{float(row.get('mask_entropy_mean', 0)):.4f}/{float(row.get('mask_entropy_p90', 0)):.4f}",
                    f"{float(row.get('trajectory_variance_mean', 0)):.6f}/{float(row.get('trajectory_variance_p90', 0)):.6f}",
                    f"{float(row.get('boundary_safe_ratio_mean', 0)):.4f}",
                    f"{float(row.get('negative_visible_outside_ratio_mean', 0)):.4f}",
                    f"{float(row.get('num_exported_objects', 0)):.0f}",
                    f"{float(row.get('num_exported_points', 0)):.0f}",
                    f"{float(row.get('export_conflict_rate', 0)):.6f}",
                ]
            )
            + " |"
        )
    (summary_root / f"{output_config}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_scene(args: argparse.Namespace, scene: str, output_config: str) -> dict[str, Any]:
    bank_path = Path(args.bank_root) / scene / "measurement_bank.npz"
    bank = MeasurementBank.load(bank_path)
    atom_bank, diag = build_surfel_atom_bank(
        bank,
        variant=args.variant,
        base_mode=args.base_mode,
        min_surfels=int(args.min_surfels),
        merge_small_surfels=int(args.merge_small_surfels),
        boundary_safe_px=float(args.boundary_safe_px),
        trajectory_bins=int(args.trajectory_bins),
        rgb_bins=int(args.rgb_bins),
        max_mask_votes=int(args.max_mask_votes),
        max_neighbors=int(args.max_neighbors),
    )
    atom_dir = Path(args.atom_root) / args.variant / scene
    atom_bank.save(atom_dir / "atom_bank.npz")
    (atom_dir / "atom_bank_summary.json").write_text(json.dumps(json_safe(diag), indent=2, sort_keys=True), encoding="utf-8")

    object_dict: dict[int, dict[str, Any]] = {}
    for atom_idx in range(atom_bank.num_atoms):
        if bool(atom_bank.is_unknown[atom_idx]) and not bool(args.export_unknown_atoms):
            continue
        if int(atom_bank.atom_size[atom_idx]) < int(args.min_export_surfels):
            continue
        record = atom_to_object_record(
            atom_bank,
            bank,
            atom_idx,
            max_mask_votes=int(args.max_mask_votes),
            fringe_from_neighbors=bool(args.fringe_from_neighbors),
        )
        if not record["mask_list"]:
            continue
        object_dict[len(object_dict)] = record

    stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="posterior_support",
        export_core_nn_radius=float(args.export_core_nn_radius),
        export_fringe_nn_radius=float(args.export_fringe_nn_radius),
        export_fringe_radius=float(args.export_fringe_radius),
        export_fringe_max_ratio=float(args.export_fringe_max_ratio),
        export_min_points_per_object=int(args.min_export_points_per_object),
        export_score_mode=args.export_score_mode,
        export_enable_wta=bool(args.export_enable_wta),
    )
    export_diag = exporter.export_object_slot_posterior_support(object_dict, bank)
    row = {
        **diag,
        **export_diag,
        "scene": scene,
        "variant": args.variant,
        "bank_path": str(bank_path),
        "atom_bank_path": str(atom_dir / "atom_bank.npz"),
        "num_atom_candidate_records": int(len(object_dict)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "is_method_result": False,
        "is_diagnostic_only": True,
    }
    return row


def _evaluate(args: argparse.Namespace, output_config: str) -> None:
    output_file = Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(Path("data/prediction") / f"{output_config}_class_agnostic"),
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(output_file),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--variant", choices=["A0", "A1", "A2", "A3", "A4"], required=True)
    parser.add_argument(
        "--base-mode",
        choices=["source", "source_or_target", "target_dominant"],
        default="source",
        help="Atom birth key source. target_dominant is diagnostic-only and uses predicted 2D masks, not GT.",
    )
    parser.add_argument("--output-config", default="")
    parser.add_argument("--atom-root", default="outputs/v14_surfel_atom_bank")
    parser.add_argument("--summary-root", default="outputs/v14_surfel_atom_bank")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-surfels", type=int, default=4)
    parser.add_argument("--min-export-surfels", type=int, default=4)
    parser.add_argument("--merge-small-surfels", type=int, default=12)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--trajectory-bins", type=int, default=4)
    parser.add_argument("--rgb-bins", type=int, default=4)
    parser.add_argument("--max-mask-votes", type=int, default=8)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--fringe-from-neighbors", action="store_true")
    parser.add_argument("--export-unknown-atoms", action="store_true")
    parser.add_argument("--export-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-max-ratio", type=float, default=0.20)
    parser.add_argument("--min-export-points-per-object", type=int, default=20)
    parser.add_argument("--export-score-mode", choices=["one", "area", "reliability", "observations"], default="reliability")
    parser.add_argument("--export-enable-wta", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_config = args.output_config or f"stream4d_v14_{args.variant.lower()}_atom_candidate_probe5"
    rows = [_export_scene(args, scene, output_config) for scene in read_seq_list(Path(args.seq_list))]
    aggregate = _aggregate(rows)
    _write_summary(Path(args.summary_root) / args.variant, output_config, rows, aggregate)
    manifest = build_prediction_manifest(
        root=".",
        output_config=output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.bank_root)],
        pre_points_policy="recompute",
        support_policy=f"v14_surfel_atom_candidate:{args.variant}",
        notes="v14 surfel atom candidate bank. Diagnostic candidate artifact; GT is not read.",
        extra={
            "algorithm": "v14_surfel_atom_bank",
            "variant": args.variant,
            "base_mode": args.base_mode,
            "eval_policy": "candidate_bank_diagnostic",
            "prediction_config": output_config,
            "pre_points_config": output_config,
            "support_source": "own_atom_candidate",
            "geometry_source": "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "summary_path": str(Path(args.summary_root) / args.variant / f"{output_config}_summary.json"),
        },
    )
    write_prediction_manifest(output_config, manifest, root=".", pred_suffix="class_agnostic")
    if not args.skip_eval:
        _evaluate(args, output_config)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
