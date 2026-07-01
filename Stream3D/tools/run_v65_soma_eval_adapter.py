from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v65_soma_eval_adapter import (
    V65SOMAEvalAdapterConfig,
    build_eval_adapter_summary,
    build_scene_object_dicts,
    read_inputs,
    read_split,
    write_v65_soma_eval_adapter,
)
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _export_scene(args: argparse.Namespace, scene: str, object_dict: dict[int, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene": scene,
        "input_object_count": len(object_dict),
        "input_mask_observation_count": int(sum(len(value.get("mask_list", [])) for value in object_dict.values())),
        "ok": False,
        "error": "",
    }
    try:
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
            export_min_points_per_object=int(args.export_min_points_per_object),
            export_score_mode=args.export_score_mode,
        )
        diag = exporter.export_object_dict_mask_backproject(object_dict)
        row.update({key: _json_safe(value) for key, value in diag.items()})
        row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _write_manifest(args: argparse.Namespace, summary_path: str) -> None:
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="scannet_rgbd_pose_mesh_eval_adapter",
        source_configs=[args.object_bank_rows, args.object_support_rows],
        pre_points_policy="diagnostic_recompute_from_backprojected_view_masks",
        support_policy="soma_object_bank_view_mask_backproject_eval_only",
        notes=(
            "Evaluation-only SOMA object-bank adapter. Inference artifact is object->view-mask support; "
            "this adapter uses ScanNet RGB-D/pose/mesh only to materialize evaluator vertex masks."
        ),
        extra={
            "algorithm": "v65_soma_object_bank_eval_adapter",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "soma_object_bank_view_mask_support",
            "geometry_source": "scannet_rgbd_pose_mesh_eval_adapter",
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "uses_rgbd_for_evaluation_support": True,
            "uses_rgbd_pose_mesh_for_export": True,
            "uses_gt_for_diagnostic": True,
            "forbidden_for_method_table": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "summary_path": summary_path,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".", pred_suffix="class_agnostic")


def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_file = Path("data/evaluation/scannet") / f"{args.output_config}_class_agnostic.txt"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        f"data/prediction/{args.output_config}_class_agnostic",
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(output_file),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        args.output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    proc = subprocess.run(cmd, cwd=".", text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "output_file": str(output_file),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize SOMA object-bank view masks as diagnostic AP predictions.")
    parser.add_argument("--object-bank-rows", default="outputs/audit/v65_soma_object_bank/soma_object_bank_rows.csv")
    parser.add_argument("--object-support-rows", default="outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-config", default="v65_soma_object_bank_eval_bridge")
    parser.add_argument("--output-root", default="outputs/audit/v65_soma_eval_adapter")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-min-points-per-object", type=int, default=1)
    parser.add_argument("--export-score-mode", default="area")
    parser.add_argument("--run-eval", action="store_true")
    args = parser.parse_args()

    cfg = V65SOMAEvalAdapterConfig(
        object_bank_rows_path=args.object_bank_rows,
        object_support_rows_path=args.object_support_rows,
        output_config=args.output_config,
        output_root=args.output_root,
        split_path=args.split,
        export_nn_radius=float(args.export_nn_radius),
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.export_min_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    object_rows, support_rows = read_inputs(cfg)
    scene_dicts = build_scene_object_dicts(object_rows, support_rows)
    split_scenes = read_split(args.split)
    scene_rows = [
        _export_scene(args, scene, scene_dicts.get(scene, {}))
        for scene in split_scenes
    ]
    summary = build_eval_adapter_summary(cfg=cfg, scene_dicts=scene_dicts, scene_rows=scene_rows)
    payload = {"summary": summary, "scene_rows": scene_rows}
    paths = write_v65_soma_eval_adapter(args.output_root, payload)
    _write_manifest(args, paths["summary"])
    eval_result = _run_eval(args) if args.run_eval else {"skipped": True}
    if args.run_eval:
        summary["eval_result"] = eval_result
        payload = {"summary": summary, "scene_rows": scene_rows}
        paths = write_v65_soma_eval_adapter(args.output_root, payload)
        _write_manifest(args, paths["summary"])
    print(
        json.dumps(
            _json_safe(
                {
                    "summary": paths["summary"],
                    "output_config": args.output_config,
                    "object_count": summary["object_count"],
                    "mask_observation_count": summary["mask_observation_count"],
                    "ok_scene_count": summary["ok_scene_count"],
                    "num_exported_objects_total": summary["num_exported_objects_total"],
                    "num_exported_points_total": summary["num_exported_points_total"],
                    "method_ap_available": summary["method_ap_available"],
                    "diagnostic_ap_export_available": summary["diagnostic_ap_export_available"],
                    "gate": summary["gate"],
                    "eval_result": eval_result,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
