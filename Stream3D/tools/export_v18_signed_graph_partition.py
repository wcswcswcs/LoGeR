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
from stream4d.signed_boundary_evidence import SignedBoundaryEvidence
from stream4d.signed_graph_partition import partition_signed_graph, partition_to_object_dict
from stream4d.signed_surfel_graph import SignedSurfelGraph
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _parse_metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}


def _export_scene(args: argparse.Namespace, scene: str, output_config: str, export_mode: str) -> dict[str, Any]:
    bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
    graph = SignedSurfelGraph.load(Path(args.graph_root) / scene / "signed_surfel_graph.npz")
    evidence = SignedBoundaryEvidence.load(Path(args.evidence_root) / args.variant / scene / "signed_boundary_evidence.npz")
    result = partition_signed_graph(
        graph,
        evidence,
        mode=args.partition_mode,
        cut_threshold=float(args.cut_threshold),
        merge_threshold=float(args.merge_threshold),
        min_component_size=int(args.min_component_size),
        max_component_ratio=float(args.max_component_ratio),
        max_fringe_ratio=float(args.max_fringe_ratio),
        use_graph_precut=not bool(args.disable_graph_precut),
    )
    object_dict = partition_to_object_dict(bank, result, export_mode=export_mode, max_mask_votes=int(args.max_mask_votes))
    stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="posterior_support",
        export_core_nn_radius=float(args.export_core_nn_radius),
        export_fringe_nn_radius=float(args.export_fringe_nn_radius),
        export_fringe_radius=float(args.export_fringe_radius),
        export_fringe_max_ratio=float(args.export_fringe_max_ratio),
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    export_diag = exporter.export_object_slot_posterior_support(object_dict, bank)
    row = {
        **result.diagnostics,
        **export_diag,
        "scene": scene,
        "variant": args.variant,
        "output_config": output_config,
        "export_mode": export_mode,
        "num_object_records_before_export": int(len(object_dict)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
    }
    scene_dir = Path(args.summary_root) / output_config / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "partition_summary.json").write_text(json.dumps(json_safe(row), indent=2, sort_keys=True), encoding="utf-8")
    return row


def _evaluate(output_config: str, *, require_manifest: bool) -> dict[str, float | None]:
    metric_path = Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        f"data/prediction/{output_config}_class_agnostic",
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(metric_path),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output_config,
        "--no_class",
    ]
    if require_manifest:
        cmd.append("--require-manifest")
    subprocess.run(cmd, check=True)
    return _parse_metric(metric_path)


def _write_summary(path: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    path.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), dict)})
        with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [f"# {aggregate['output_config']}", "", "## Aggregate", ""]
    for key, value in aggregate.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | components | exported objects | exported points | largest comp | conflict |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_kept_components")),
                    str(row.get("num_exported_objects")),
                    str(row.get("num_exported_points")),
                    f"{float(row.get('largest_component_ratio') or 0.0):.6f}",
                    f"{float(row.get('export_conflict_rate') or 0.0):.6f}",
                ]
            )
            + " |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate(rows: list[dict[str, Any]], metrics: dict[str, float | None], output_config: str, export_mode: str) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm": "v18_signed_graph_partition",
        "output_config": output_config,
        "export_mode": export_mode,
        "ap": metrics.get("ap"),
        "ap50": metrics.get("ap50"),
        "ap25": metrics.get("ap25"),
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }


def _manifest(output_config: str, args: argparse.Namespace, export_mode: str) -> None:
    manifest = build_prediction_manifest(
        root=".",
        output_config=output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.graph_root, str(Path(args.evidence_root) / args.variant), args.bank_root],
        pre_points_policy="recompute",
        support_policy=f"v18_signed_graph_partition:{args.partition_mode}:{export_mode}",
        notes="v18 non-GT signed boundary graph partition method. GT is not read for prediction.",
        extra={
            "algorithm": "v18_signed_graph_partition",
            "variant": args.variant,
            "partition_mode": args.partition_mode,
            "export_mode": export_mode,
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": output_config,
            "pre_points_config": output_config,
            "support_source": "own_signed_partition",
            "geometry_source": "rgbd_eval_bridge_for_export",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": True,
            "is_diagnostic_only": False,
        },
    )
    write_prediction_manifest(output_config, manifest, root=".", pred_suffix="class_agnostic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--graph-root", default="outputs/audit/v18_phase1")
    parser.add_argument("--evidence-root", default="outputs/audit/v18_phase3")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--variant", default="E5_full_signed")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--export-mode", choices=["G_core", "G_region_fill"], required=True)
    parser.add_argument("--partition-mode", choices=["P1_signed_watershed", "P2_agglomerative_signed", "P3_seeded_graph_partition"], default="P2_agglomerative_signed")
    parser.add_argument("--cut-threshold", type=float, default=0.62)
    parser.add_argument("--merge-threshold", type=float, default=0.55)
    parser.add_argument("--min-component-size", type=int, default=16)
    parser.add_argument("--max-component-ratio", type=float, default=0.40)
    parser.add_argument("--max-fringe-ratio", type=float, default=0.35)
    parser.add_argument("--disable-graph-precut", action="store_true")
    parser.add_argument("--max-mask-votes", type=int, default=8)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--export-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-max-ratio", type=float, default=0.20)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--export-score-mode", choices=["one", "area", "reliability", "observations"], default="observations")
    parser.add_argument("--summary-root", default="outputs/audit/v18_phase4")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    rows = [
        _export_scene(args, scene, args.output_config, args.export_mode)
        for scene in read_seq_list(Path(args.seq_list))
    ]
    _manifest(args.output_config, args, args.export_mode)
    metrics = {"ap": None, "ap50": None, "ap25": None} if args.skip_eval else _evaluate(args.output_config, require_manifest=True)
    aggregate = _aggregate(rows, metrics, args.output_config, args.export_mode)
    _write_summary(Path(args.summary_root) / f"{args.output_config}_summary", rows, aggregate, args)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
