from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


ROOT = Path(__file__).resolve().parents[1]
PROBE5_SCENES = ["scene0030_00", "scene0081_01", "scene0591_00", "scene0011_00", "scene0050_00"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in keys})


def _parse_scene_list(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return list(PROBE5_SCENES)
    if text == "probe5":
        return list(PROBE5_SCENES)
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_mask_ref(ref: str) -> tuple[int, int] | None:
    if ":" not in ref:
        return None
    left, right = ref.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _parse_absorbed(value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    refs: list[tuple[int, int]] = []
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            ref = _parse_mask_ref(str(item))
            if ref is not None:
                refs.append(ref)
    return refs


def _load_objectlet_masks(path: Path) -> dict[int, list[tuple[int, int, float]]]:
    by_object: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                object_id = int(row.get("object_id", "-1"))
            except ValueError:
                continue
            if object_id < 0:
                continue
            refs: list[tuple[int, int]] = []
            primary = _parse_mask_ref(str(row.get("primary", "")))
            if primary is not None:
                refs.append(primary)
            refs.extend(_parse_absorbed(str(row.get("absorbed", ""))))
            try:
                weight = float(row.get("support_tube_count") or 0.0) + float(row.get("support_observation_count") or 0.0)
            except ValueError:
                weight = 1.0
            weight = max(float(weight), 1.0)
            for ref in refs:
                by_object[object_id][ref] = max(float(by_object[object_id].get(ref, 0.0)), weight)
    return {
        object_id: [(frame_id, mask_id, weight) for (frame_id, mask_id), weight in sorted(items.items())]
        for object_id, items in sorted(by_object.items())
    }


def _gt_coverage_ratio(gt_path: Path, point_union: np.ndarray) -> tuple[float | None, int, int]:
    if not gt_path.exists():
        return None, 0, 0
    gt = np.loadtxt(gt_path, dtype=np.int64)
    valid_gt = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    if not valid_gt:
        return None, 0, 0
    covered = set(int(v) for v in np.unique(gt[point_union]) if int(v) > 0) if point_union.size else set()
    return float(len(covered) / len(valid_gt)), int(len(covered)), int(len(valid_gt))


def _export_scene(
    *,
    scene: str,
    native_root: Path,
    output_config: str,
    export_nn_radius: float,
    export_mask_sample_stride: int,
    export_mask_max_pixels: int,
    export_min_points_per_object: int,
) -> dict[str, Any]:
    objectlet_path = native_root / "scene_details" / scene / "objectlets.csv"
    if not objectlet_path.exists():
        raise FileNotFoundError(objectlet_path)
    object_masks = _load_objectlet_masks(objectlet_path)
    stream = ScanNetStream(seq_name=scene)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_nn_radius=float(export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(export_mask_sample_stride),
        export_mask_max_pixels=int(export_mask_max_pixels),
        export_min_points_per_object=int(export_min_points_per_object),
        export_score_mode="area",
    )
    object_records: list[dict[str, Any]] = []
    object_dict: dict[int, dict[str, Any]] = {}
    backproject_queries = 0
    backproject_hits = 0
    mask_observation_count = 0
    for object_id, refs in object_masks.items():
        point_ids: set[int] = set()
        kept_refs: list[tuple[int, int, float]] = []
        for frame_id, mask_id, weight in refs:
            hit_ids, query_count = exporter._backproject_mask(
                int(frame_id), int(mask_id), nn_radius=float(export_nn_radius)
            )
            backproject_queries += int(query_count)
            backproject_hits += int(hit_ids.shape[0])
            if hit_ids.size == 0:
                continue
            point_ids.update(int(v) for v in hit_ids.tolist())
            kept_refs.append((int(frame_id), int(mask_id), float(weight)))
            mask_observation_count += 1
        score = float(len(point_ids))
        object_records.append(
            {
                "object_id": int(object_id),
                "point_ids": set(point_ids),
                "score": score,
                "area_score": score,
            }
        )
        object_dict[int(object_id)] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": kept_refs,
            "repre_mask_list": kept_refs[: min(8, len(kept_refs))],
            "score": score,
            "area_score": score,
            "source": "v44_native_typed_mask_assembly",
        }
    diag = exporter._write_outputs(
        object_records,
        object_dict,
        np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16),
    )
    pred_path = ROOT / "data/prediction" / f"{output_config}_class_agnostic" / f"{scene}.npz"
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
        point_union = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    gt_ratio, covered_gt, total_gt = _gt_coverage_ratio(ROOT / "data/scannet/gt" / f"{scene}.txt", point_union)
    diag.update(
        {
            "scene": scene,
            "objectlet_path": str(objectlet_path),
            "object_count_input": int(len(object_masks)),
            "mask_observation_count": int(mask_observation_count),
            "backproject_query_count": int(backproject_queries),
            "backproject_hit_count": int(backproject_hits),
            "backproject_hit_rate": float(backproject_hits / max(backproject_queries, 1)),
            "mesh_coverage": float(point_union.shape[0] / max(masks.shape[0], 1)),
            "covered_GT_instance_ratio": gt_ratio,
            "covered_GT_instances": int(covered_gt),
            "total_GT_instances": int(total_gt),
            "prediction_path": str(pred_path),
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
        }
    )
    return diag


def _parse_metric_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("class,"):
        reader = csv.DictReader(lines)
        numeric_rows: list[tuple[float, float, float]] = []
        trailing: tuple[float, float, float] | None = None
        for row in reader:
            name = str(row.get("class", "")).strip().lower()
            try:
                values = (float(row["ap"]), float(row["ap50"]), float(row["ap25"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(np.isfinite(v) for v in values):
                numeric_rows.append(values)
            if name in {"average", "mean"}:
                trailing = values
        if trailing is None and numeric_rows:
            trailing = numeric_rows[-1]
        if trailing is not None:
            return {"AP": float(trailing[0]), "AP50": float(trailing[1]), "AP25": float(trailing[2])}
    metrics: dict[str, Any] = {}
    for raw in lines:
        line = raw.strip()
        if not line.startswith("average"):
            continue
        parts = line.split()
        values: list[float] = []
        for item in parts[1:]:
            try:
                values.append(float(item))
            except ValueError:
                pass
        if len(values) >= 3:
            metrics["AP"] = float(values[0])
            metrics["AP50"] = float(values[1])
            metrics["AP25"] = float(values[2])
    return metrics


def _run_eval(output_config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{output_config}_class_agnostic.txt"
    log_path = eval_dir / f"{output_config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{output_config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        output_config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return {
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": _parse_metric_file(metric_file),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    native_root = ROOT / args.native_root
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_scene_list(args.scenes)
    rows = [
        _export_scene(
            scene=scene,
            native_root=native_root,
            output_config=args.output_config,
            export_nn_radius=float(args.export_nn_radius),
            export_mask_sample_stride=int(args.export_mask_sample_stride),
            export_mask_max_pixels=int(args.export_mask_max_pixels),
            export_min_points_per_object=int(args.export_min_points_per_object),
        )
        for scene in scenes
    ]
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(native_root)],
        pre_points_policy="rgbd_pose_mesh_mask_backproject",
        support_policy="v44_native_objectlet_mask_backproject",
        notes=(
            "v44 native typed-mask AP bridge diagnostic. This is not a method AP row: "
            "it materializes 2D mask objectlets onto ScanNet mesh vertices using RGB-D, pose, and mesh."
        ),
        extra={
            "phase": "v44_native_ap_bridge_diagnostic",
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
            "native_stage1_root": str(native_root),
            "scene_count": int(len(scenes)),
            "scenes": list(scenes),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=ROOT, pred_suffix="class_agnostic")
    eval_result = _run_eval(args.output_config, output_root)
    numeric_keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.integer, np.floating))
    )
    aggregate: dict[str, Any] = {"scene_count": int(len(rows)), **eval_result.get("metrics", {})}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
        if vals:
            aggregate[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    summary = {
        "phase": "v44_native_ap_bridge_diagnostic",
        "status": "ok" if eval_result["exit_code"] == 0 else "eval_failed",
        "output_config": args.output_config,
        "native_root": str(native_root),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
        "scene_rows": rows,
        "aggregate": aggregate,
        "eval": eval_result,
        "manifest": str(ROOT / "data/prediction" / f"{args.output_config}_class_agnostic" / "config_manifest.json"),
    }
    _write_json(output_root / "v44_native_ap_bridge_summary.json", summary)
    _write_csv(output_root / "v44_native_ap_bridge_scene_rows.csv", rows)
    print(json.dumps(_json_safe({"status": summary["status"], "aggregate": aggregate, "eval": eval_result}), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic AP bridge for v44 native typed-mask objectlets.")
    parser.add_argument("--native-root", default="outputs/audit/v44_native_full_probe5_core_first_l034")
    parser.add_argument("--scenes", default="probe5")
    parser.add_argument("--output-config", default="v44_native_core_first_l034_diag_ap_probe5")
    parser.add_argument("--output-root", default="outputs/audit/v44_native_ap_bridge_core_first_l034_probe5")
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-min-points-per-object", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
