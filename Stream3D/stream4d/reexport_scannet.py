from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .diagnostics import write_json
from .export_scannet import ScanNetExporter
from .scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _point_set(value: dict) -> set[int]:
    return set(int(v) for v in np.asarray(value.get("point_ids", []), dtype=np.int64).tolist())


def _merge_by_point_ioc(object_dict: dict[int, dict], threshold: float) -> dict[int, dict]:
    if threshold <= 0.0 or len(object_dict) <= 1:
        return object_dict
    items = [(int(k), v, _point_set(v)) for k, v in sorted(object_dict.items(), key=lambda item: int(item[0]))]
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(items)):
        a = items[i][2]
        if not a:
            continue
        for j in range(i + 1, len(items)):
            b = items[j][2]
            if not b:
                continue
            denom = max(1, min(len(a), len(b)))
            if len(a.intersection(b)) / denom >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(items)):
        groups.setdefault(find(idx), []).append(idx)

    merged: dict[int, dict] = {}
    next_id = 0
    for indices in groups.values():
        point_ids: set[int] = set()
        mask_list = []
        carrier_ids = []
        for idx in indices:
            value = items[idx][1]
            point_ids.update(items[idx][2])
            mask_list.extend(list(value.get("mask_list", [])))
            carrier_ids.extend(np.asarray(value.get("carrier_ids", []), dtype=np.int64).tolist())
        merged[next_id] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": mask_list,
            "carrier_ids": np.asarray(sorted(set(int(v) for v in carrier_ids)), dtype=np.int64),
        }
        next_id += 1
    return merged


def _seq_names(args: argparse.Namespace) -> list[str]:
    if args.seq_name:
        return [args.seq_name]
    if args.seq_list:
        with Path(args.seq_list).open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    raise ValueError("Provide --seq-name or --seq-list")


def _process_sequence(args: argparse.Namespace, seq_name: str) -> dict:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    object_path = stream.object_dir / args.input_config / "object_dict.npy"
    if not object_path.exists():
        raise FileNotFoundError(f"Missing Stream4D object_dict: {object_path}")

    pred_path = Path("data/prediction") / f"{args.output_config}_class_agnostic" / f"{seq_name}.npz"
    tmp_path = Path("data/TMP") / args.output_config / f"{seq_name}_pre_points.npy"
    if args.skip_existing and pred_path.exists() and tmp_path.exists():
        print(f"[stream4d-reexport] seq={seq_name} skip existing prediction={pred_path}", flush=True)
        return {"seq_name": seq_name, "skipped_existing": True}

    object_dict = np.load(object_path, allow_pickle=True).item()
    object_dict = _merge_by_point_ioc(object_dict, args.merge_point_ioc_threshold)
    if args.reexport_mode == "mask_backproject":
        export_support_mode = "mask_backproject"
    elif args.reexport_mode == "reliable_densify":
        export_support_mode = "reliable_densify"
    elif args.export_point_dilate_radius > 0.0:
        export_support_mode = "point_dilate"
    else:
        export_support_mode = "reuse_point_ids"
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=args.export_nn_radius,
        export_support_mode=export_support_mode,
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
        export_max_masks_per_object=args.export_max_masks_per_object,
        export_point_dilate_radius=args.export_point_dilate_radius,
        export_min_points_per_object=args.export_min_points_per_object,
        export_score_mode=args.export_score_mode,
        densify_boundary_erosion=args.densify_boundary_erosion,
        densify_small_mask_area=args.densify_small_mask_area,
        densify_seed_distance_px=args.densify_seed_distance_px,
        densify_min_seed_pixels=args.densify_min_seed_pixels,
        densify_enable_wta=not args.disable_densify_wta,
        densify_seed_keep_mode=args.densify_seed_keep_mode,
        densify_seed_min_support_views=args.densify_seed_min_support_views,
        densify_mask_selection_mode=args.densify_mask_selection_mode,
    )
    if args.reexport_mode == "mask_backproject":
        diag = exporter.export_object_dict_mask_backproject(object_dict)
    elif args.reexport_mode == "point_dilate":
        diag = exporter.export_object_dict_points(object_dict)
    elif args.reexport_mode == "reliable_densify":
        diag = exporter.export_object_dict_reliable_densify(object_dict)
    else:
        raise ValueError(f"Unsupported reexport mode: {args.reexport_mode}")
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config],
        pre_points_policy="recompute",
        support_policy=export_support_mode,
        notes="Generated by stream4d.reexport_scannet from an existing Stream4D object_dict.",
        extra={
            "seq_scope": seq_name,
            "reexport_mode": args.reexport_mode,
            "input_config": args.input_config,
        },
    )
    write_prediction_manifest(args.output_config, manifest)
    hit_rate = diag.get("export_nn_hit_rate")
    hit_text = "NA" if hit_rate is None else f"{float(hit_rate):.4f}"
    print(
        f"[stream4d-reexport] seq={seq_name} objects={int(diag['num_exported_objects'])} "
        f"points={int(diag['num_exported_points'])} hit_rate={hit_text} "
        f"conflict={diag['export_conflict_rate']:.4f}",
        flush=True,
    )
    return {"seq_name": seq_name, **diag}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-list", default="")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--reexport-mode",
        default="mask_backproject",
        choices=["mask_backproject", "point_dilate", "reliable_densify"],
    )
    parser.add_argument("--export-nn-radius", type=float, default=0.08)
    parser.add_argument("--export-mask-sample-stride", type=int, default=1)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-max-masks-per-object", type=int, default=0)
    parser.add_argument("--export-point-dilate-radius", type=float, default=0.0)
    parser.add_argument("--export-min-points-per-object", type=int, default=0)
    parser.add_argument(
        "--export-score-mode",
        default="one",
        choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"],
    )
    parser.add_argument("--merge-point-ioc-threshold", type=float, default=0.0)
    parser.add_argument("--densify-boundary-erosion", type=int, default=1)
    parser.add_argument("--densify-small-mask-area", type=int, default=400)
    parser.add_argument("--densify-seed-distance-px", type=float, default=32.0)
    parser.add_argument("--densify-min-seed-pixels", type=int, default=1)
    parser.add_argument(
        "--densify-seed-keep-mode",
        default="none",
        choices=["none", "supported", "boundary", "component", "all"],
    )
    parser.add_argument("--densify-seed-min-support-views", type=int, default=1)
    parser.add_argument(
        "--densify-mask-selection-mode",
        default="coverage",
        choices=[
            "coverage",
            "seed_density",
            "component_seed_density",
            "kept_seed_density",
            "coverage_component_density",
            "coverage_kept_density",
            "kept_ratio",
        ],
    )
    parser.add_argument("--disable-densify-wta", action="store_true")
    parser.add_argument("--debug-root", default="outputs/stream4d_reexport")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = []
    errors = []
    for seq_name in _seq_names(args):
        try:
            summaries.append(_process_sequence(args, seq_name))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[stream4d-reexport][ERROR] seq={seq_name} {message}", flush=True)
            errors.append({"seq_name": seq_name, "error": message})
    out_dir = Path(args.debug_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.output_config}_summary.json", {"summaries": summaries, "errors": errors})


if __name__ == "__main__":
    main()
