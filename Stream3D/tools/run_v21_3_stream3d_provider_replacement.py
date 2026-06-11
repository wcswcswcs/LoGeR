from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from geometry_provider import D4RTCarrierProjectionProvider, RGBDGeometryProvider
from main import main as stream3d_main
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _debug_frame_ids(debug_root: str | Path, scene: str) -> list[int]:
    scene_dir = Path(debug_root) / scene
    out: set[int] = set()
    for carrier_path in sorted(scene_dir.glob("carriers_window*.npz")):
        with np.load(carrier_path) as data:
            if "src_frame_global" in data.files:
                out.update(int(v) for v in np.unique(np.asarray(data["src_frame_global"], dtype=np.int64)).tolist())
                continue
            manifest = carrier_path.with_name(f"{carrier_path.stem}_manifest.json")
            if manifest.exists():
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    for key in ("raw_frame_ids", "frame_indices", "frame_ids"):
                        vals = [int(v) for v in payload.get(key, [])]
                        if vals:
                            out.update(vals)
                            break
                except Exception:
                    pass
    return sorted(out)


def _base_args(config_name: str, base_config: dict[str, Any], seq_name: str, backbone: str, debug: bool) -> SimpleNamespace:
    payload = {
        "seq_name": seq_name,
        "seq_name_list": seq_name,
        "config": config_name,
        "backbone": backbone,
        "debug": debug,
        "para": 1,
    }
    payload.update(base_config)
    return SimpleNamespace(**payload)


def _metric_from_file(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    try:
        return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}
    except ValueError:
        return {"ap": None, "ap50": None, "ap25": None}


def _prediction_stats(root: Path, config: str, scenes: list[str]) -> dict[str, Any]:
    pred_dir = root / "data" / "prediction" / f"{config}_class_agnostic"
    tmp_dir = root / "data" / "TMP" / config
    num_pred = []
    union_ratio = []
    pre_ratio = []
    for scene in scenes:
        pred_path = pred_dir / f"{scene}.npz"
        pre_path = tmp_dir / f"{scene}_pre_points.npy"
        if pred_path.exists():
            data = np.load(pred_path)
            masks = np.asarray(data["pred_masks"], dtype=bool)
            num_pred.append(int(masks.shape[1]))
            union_ratio.append(float(np.count_nonzero(np.any(masks, axis=1)) / max(masks.shape[0], 1)))
        if pre_path.exists() and pred_path.exists():
            pre = np.load(pre_path)
            data = np.load(pred_path)
            masks = np.asarray(data["pred_masks"], dtype=bool)
            pre_ratio.append(float(pre.shape[0] / max(masks.shape[0], 1)))
    return {
        "num_pred_per_scene": float(np.mean(num_pred)) if num_pred else None,
        "prediction_union_ratio": float(np.mean(union_ratio)) if union_ratio else None,
        "pre_points_ratio": float(np.mean(pre_ratio)) if pre_ratio else None,
    }


def _make_provider(spec: dict[str, Any], args: argparse.Namespace):
    if spec["provider"] == "rgbd":
        return RGBDGeometryProvider()
    return D4RTCarrierProjectionProvider(
        debug_root=args.debug_root,
        mode=spec["mode"],
        nn_radius=args.nn_radius,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
        max_anchors=args.max_anchors,
        robust_trim_percentile=args.robust_trim_percentile,
        density_alpha=args.density_alpha,
        local_outlier_filter=bool(spec.get("local_outlier_filter", False)),
        min_mask_interior_px=float(spec.get("min_mask_interior_px", args.min_mask_interior_px)),
        overlap_policy=str(spec.get("overlap_policy", args.overlap_policy)),
    )


VARIANTS: dict[str, dict[str, Any]] = {
    "G0": {"label": "G0 RGBD baseline", "provider": "rgbd", "geometry_source": "stream3d_rgbd_pose_mesh"},
    "G1": {"label": "G1 D4RT raw", "provider": "d4rt", "mode": "raw", "geometry_source": "d4rt_raw_carrier_provider"},
    "G2": {
        "label": "G2 D4RT self-stitched",
        "provider": "d4rt",
        "mode": "self_stitched",
        "geometry_source": "d4rt_self_stitched_carrier_provider",
        "overlap_policy": "best_confidence",
    },
    "G3": {
        "label": "G3 D4RT eval-Sim3",
        "provider": "d4rt",
        "mode": "eval_sim3",
        "geometry_source": "d4rt_eval_sim3_carrier_provider",
    },
    "G4": {
        "label": "G4 D4RT eval-Sim3 + outlier filter",
        "provider": "d4rt",
        "mode": "eval_sim3",
        "local_outlier_filter": True,
        "geometry_source": "d4rt_eval_sim3_outlier_filtered_carrier_provider",
    },
    "G5": {
        "label": "G5 D4RT eval-Sim3 + density thresholds",
        "provider": "d4rt",
        "mode": "eval_sim3_density",
        "geometry_source": "d4rt_eval_sim3_density_carrier_provider",
    },
    "G6": {
        "label": "G6 D4RT self-stitched + density thresholds",
        "provider": "d4rt",
        "mode": "self_stitched_density",
        "geometry_source": "d4rt_self_stitched_density_carrier_provider",
        "overlap_policy": "best_confidence",
    },
    "G7": {
        "label": "G7 D4RT eval-Sim3 + mask interior gate",
        "provider": "d4rt",
        "mode": "eval_sim3",
        "min_mask_interior_px": 2.0,
        "geometry_source": "d4rt_eval_sim3_mask_interior_carrier_provider",
    },
    "G8": {
        "label": "G8 D4RT eval-Sim3 + density + mask interior gate",
        "provider": "d4rt",
        "mode": "eval_sim3_density",
        "min_mask_interior_px": 2.0,
        "geometry_source": "d4rt_eval_sim3_density_mask_interior_carrier_provider",
    },
    "G9": {
        "label": "G9 D4RT self-stitched scale-normalized bundle",
        "provider": "d4rt",
        "mode": "self_stitched_scale_normalized",
        "geometry_source": "d4rt_self_stitched_scale_normalized_carrier_provider",
        "overlap_policy": "best_confidence",
    },
    "G10": {
        "label": "G10 D4RT self-stitched scale-normalized density",
        "provider": "d4rt",
        "mode": "self_stitched_scale_normalized_density",
        "geometry_source": "d4rt_self_stitched_scale_normalized_density_carrier_provider",
        "overlap_policy": "best_confidence",
    },
}


def _write_outputs(out_prefix: Path, rows: list[dict[str, Any]], scenes: list[str], args: argparse.Namespace) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    deltas = {}
    by_variant = {row["variant"]: row for row in rows}
    if by_variant.get("G0", {}).get("ap") is not None and by_variant.get("G3", {}).get("ap") is not None:
        deltas["delta_d4rt_eval_sim3"] = float(by_variant["G0"]["ap"] - by_variant["G3"]["ap"])
    if by_variant.get("G3", {}).get("ap") is not None and by_variant.get("G2", {}).get("ap") is not None:
        deltas["delta_self_stitch"] = float(by_variant["G3"]["ap"] - by_variant["G2"]["ap"])
    if by_variant.get("G4", {}).get("ap") is not None and by_variant.get("G3", {}).get("ap") is not None:
        deltas["delta_outlier"] = float(by_variant["G4"]["ap"] - by_variant["G3"]["ap"])
    if by_variant.get("G5", {}).get("ap") is not None and by_variant.get("G3", {}).get("ap") is not None:
        deltas["delta_density_threshold"] = float(by_variant["G5"]["ap"] - by_variant["G3"]["ap"])

    payload = {
        "summary": {
            "scenes": scenes,
            "variants_requested": args.variants,
            "num_rows": int(len(rows)),
            **deltas,
        },
        "rows": rows,
    }
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_prefix.with_suffix(".csv")
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# D4RT Geometry Replacement Stream3D Probe5",
        "",
        "This table is produced by rerunning Stream3D internal graph construction with a geometry_provider hook. Rows are diagnostic-only and must not enter a method table.",
        "",
        "| variant | config | AP | AP50 | AP25 | pre% | union% | #pred | projection hit | empty mask | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(value: Any, scale: float = 1.0) -> str:
            if value is None:
                return "NA"
            return f"{float(value) * scale:.6f}"

        lines.append(
            "| {variant} | {config} | {ap} | {ap50} | {ap25} | {pre} | {union} | {pred} | {hit} | {empty} | {status} |".format(
                variant=row.get("variant"),
                config=row.get("output_config"),
                ap=fmt(row.get("ap")),
                ap50=fmt(row.get("ap50")),
                ap25=fmt(row.get("ap25")),
                pre=fmt(row.get("pre_points_ratio")),
                union=fmt(row.get("prediction_union_ratio")),
                pred=fmt(row.get("num_pred_per_scene")),
                hit=fmt(row.get("projection_hit_rate_mean")),
                empty=fmt(row.get("mask_projection_empty_rate_mean")),
                status=row.get("status"),
            )
        )
    lines.extend(["", "## Deltas", ""])
    for key, value in deltas.items():
        lines.append(f"- {key}: `{value:.6f}`")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- seq_list: `{Path(args.seq_list).resolve()}`",
            f"- debug_root: `{Path(args.debug_root).resolve()}`",
            f"- output_prefix: `{args.output_prefix}`",
        ]
    )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v21.3 Stream3D GeometryProvider replacement diagnostics.")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--debug-root", default="outputs/stream4d_debug_full_32f_ioc075_fixmem")
    parser.add_argument("--output-prefix", default="stream4d_v21_3_provider_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v21_3_phaseD")
    parser.add_argument("--base-config", default="configs/scannet.json")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--variants", default="G0,G1,G2,G3,G4,G5,G6")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--density-alpha", type=float, default=2.0)
    parser.add_argument("--min-mask-interior-px", type=float, default=0.0)
    parser.add_argument(
        "--overlap-policy",
        choices=["all", "all_window_union", "best_confidence", "lowest_residual", "newest_window"],
        default="all_window_union",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit-to-debug-frames", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    scenes = _read_seq_list(Path(args.seq_list))
    base_config = _load_config(Path(args.base_config))
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for variant in requested:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choices={sorted(VARIANTS)}")
        spec = VARIANTS[variant]
        output_config = f"{args.output_prefix}_{variant.lower()}"
        provider = _make_provider(spec, args)
        if hasattr(provider, "reset_diagnostics"):
            provider.reset_diagnostics()
        row: dict[str, Any] = {
            "variant": variant,
            "label": spec["label"],
            "output_config": output_config,
            "status": "ok",
            "error": "",
            "geometry_source": spec["geometry_source"],
            "runtime_seconds": None,
        }
        started = time.time()
        scene_errors = []
        for scene in scenes:
            run_args = _base_args(output_config, base_config, scene, args.backbone, args.debug)
            run_args.geometry_provider = provider
            if args.limit_to_debug_frames:
                run_args.frame_id_allowlist = _debug_frame_ids(args.debug_root, scene)
            try:
                stream3d_main(run_args, para=1)
            except Exception as exc:
                scene_errors.append(f"{scene}: {type(exc).__name__}: {exc}")
        row["runtime_seconds"] = float(time.time() - started)
        if scene_errors:
            row["status"] = "failed"
            row["error"] = "; ".join(scene_errors)

        provider_diag = provider.aggregate_diagnostics() if hasattr(provider, "aggregate_diagnostics") else {}
        row.update(provider_diag)
        diag_path = Path(args.audit_root) / f"{output_config}_provider_diagnostics.json"
        if hasattr(provider, "write_diagnostics"):
            provider.write_diagnostics(diag_path)
            row["provider_diagnostics_path"] = str(diag_path)
        row.update(_prediction_stats(root, output_config, scenes))

        uses_gt_sim3 = bool(getattr(provider, "uses_gt_sim3_for_prediction", False))
        uses_d4rt_self_sim3 = bool(getattr(provider, "uses_d4rt_self_sim3", False))
        uses_rgbd = bool(getattr(provider, "uses_rgbd_for_prediction", False))
        manifest = build_prediction_manifest(
            root=root,
            output_config=output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=False,
            gt_usage="none",
            source_configs=[args.debug_root],
            pre_points_policy="stream3d_provider_replacement_diagnostic",
            support_policy=f"v21_3_phaseD:{variant}",
            notes="v21.3 diagnostic-only Stream3D GeometryProvider replacement rerun.",
            extra={
                "algorithm": "v21_3_stream3d_geometry_provider_replacement",
                "variant": variant,
                "prediction_config": output_config,
                "pre_points_config": output_config,
                "support_source": "own",
                "geometry_source": spec["geometry_source"],
                "eval_policy": "v21_3_provider_replacement_diagnostic",
                "uses_rgbd_for_prediction": uses_rgbd,
                "uses_pose_for_prediction": uses_rgbd,
                "uses_scannet_mesh_for_prediction": True,
                "uses_gt_for_prediction": uses_gt_sim3,
                "uses_gt_sim3_for_prediction": uses_gt_sim3,
                "uses_d4rt_self_sim3": uses_d4rt_self_sim3,
                "uses_gt_for_diagnostic": uses_gt_sim3 or uses_rgbd,
                "alignment_used_for_prediction": uses_gt_sim3,
                "alignment_used_for_diagnostic": uses_gt_sim3,
                "forbidden_for_method_table": True,
            },
        )
        write_prediction_manifest(output_config, manifest, root=root, pred_suffix="class_agnostic")

        metric_path = root / "data" / "evaluation" / "scannet" / f"{output_config}_class_agnostic.txt"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
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
            "--require-manifest",
            "--allow-oracle-eval",
        ]
        if row["status"] == "ok":
            proc = subprocess.run(eval_cmd, cwd=str(root), text=True, capture_output=True)
            row["eval_returncode"] = int(proc.returncode)
            row["eval_stdout_tail"] = "\n".join(proc.stdout.splitlines()[-8:])
            row["eval_stderr_tail"] = "\n".join(proc.stderr.splitlines()[-8:])
            if proc.returncode != 0:
                row["status"] = "eval_failed"
                row["error"] = row["eval_stderr_tail"] or row["eval_stdout_tail"]
        row.update(_metric_from_file(metric_path))
        rows.append(row)
        _write_outputs(Path(args.audit_root) / "D4RT_geometry_replacement_stream3d_probe5", rows, scenes, args)
        print(json.dumps(_json_safe(row), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
