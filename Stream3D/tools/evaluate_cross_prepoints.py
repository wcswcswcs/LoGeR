from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_dir(root: Path, config: str, suffix: str) -> Path:
    if config.endswith(suffix):
        dirname = config
    else:
        dirname = f"{config}{suffix}"
    return root / "data" / "prediction" / dirname


def _strip_suffix(config: str, suffix: str) -> str:
    if suffix and config.endswith(suffix):
        return config[: -len(suffix)]
    return config


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _count_gt_instances(gt_ids: np.ndarray) -> int:
    valid = gt_ids[gt_ids >= 1000]
    if valid.size == 0:
        return 0
    return int(np.unique(valid.astype(np.int64)).shape[0])


def _replace_with_symlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel_src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _parse_metric_file(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    try:
        return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}
    except ValueError:
        return {"ap": None, "ap50": None, "ap25": None}


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get("ok") and row.get(key) is not None]
    return float(mean(values)) if values else None


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows if row.get("ok") and row.get(key) is not None))


def _materialize_scene(
    args: argparse.Namespace,
    root: Path,
    scene_id: str,
    source_pred_dir: Path,
    output_pred_dir: Path,
    output_tmp_dir: Path,
    source_pre_points_config: str,
) -> dict[str, Any]:
    pred_path = source_pred_dir / f"{scene_id}.npz"
    source_tmp_path = _tmp_path(root, source_pre_points_config, scene_id)
    target_tmp_path = _tmp_path(root, args.pre_points_config, scene_id)
    gt_path = root / args.gt_root / f"{scene_id}.txt"
    row: dict[str, Any] = {
        "scene_id": scene_id,
        "source_pred_path": str(pred_path),
        "source_pre_points_path": str(source_tmp_path),
        "target_pre_points_path": str(target_tmp_path),
        "gt_path": str(gt_path),
        "ok": False,
        "error": None,
    }
    missing = [str(path) for path in (pred_path, source_tmp_path, target_tmp_path, gt_path) if not path.exists()]
    if missing:
        row["error"] = "missing files: " + "; ".join(missing)
        return row

    with np.load(pred_path) as pred:
        pred_masks = np.asarray(pred["pred_masks"])
        pred_score = np.asarray(pred["pred_score"])
        pred_classes = np.asarray(pred["pred_classes"])
    source_pre_points = np.load(source_tmp_path).astype(np.int64)
    target_pre_points = np.load(target_tmp_path).astype(np.int64)
    gt_ids_full = np.loadtxt(gt_path).astype(np.int64)
    scene_vertices = int(gt_ids_full.shape[0])

    if source_pre_points.size and (
        int(source_pre_points.min()) < 0 or int(source_pre_points.max()) >= scene_vertices
    ):
        row["error"] = "source pre_points has out-of-range entries"
        return row
    if target_pre_points.size and (
        int(target_pre_points.min()) < 0 or int(target_pre_points.max()) >= scene_vertices
    ):
        row["error"] = "target pre_points has out-of-range entries"
        return row

    expanded_prediction = False
    link_mode = "none"
    output_pred_path = output_pred_dir / f"{scene_id}.npz"
    if pred_masks.shape[0] == scene_vertices:
        full_masks = pred_masks
        mask_shape_mode = "full_scene"
        link_mode = _replace_with_symlink_or_copy(pred_path, output_pred_path)
    elif pred_masks.shape[0] == source_pre_points.shape[0]:
        full_masks = np.zeros((scene_vertices, pred_masks.shape[1]), dtype=pred_masks.dtype)
        full_masks[source_pre_points, :] = pred_masks
        mask_shape_mode = "source_pre_points_cropped"
        expanded_prediction = True
        output_pred_dir.mkdir(parents=True, exist_ok=True)
        if output_pred_path.exists() or output_pred_path.is_symlink():
            output_pred_path.unlink()
        np.savez_compressed(
            output_pred_path,
            pred_masks=full_masks,
            pred_score=pred_score,
            pred_classes=pred_classes,
        )
        link_mode = "expanded_npz"
    else:
        row["error"] = (
            f"unsupported pred mask first dimension {pred_masks.shape[0]}; "
            f"scene vertices={scene_vertices}; source pre_points={source_pre_points.shape[0]}"
        )
        return row

    output_tmp_path = output_tmp_dir / f"{scene_id}_pre_points.npy"
    tmp_link_mode = _replace_with_symlink_or_copy(target_tmp_path, output_tmp_path)

    prediction_union = np.flatnonzero(full_masks.any(axis=1)).astype(np.int64)
    target_union_count = int(np.count_nonzero(full_masks[target_pre_points, :].any(axis=1)))
    gt_ids_crop = gt_ids_full[target_pre_points]
    row.update(
        {
            "ok": True,
            "source_pred_config": args.pred_config,
            "source_pre_points_config": source_pre_points_config,
            "target_pre_points_config": args.pre_points_config,
            "output_config": args.output_config,
            "output_pred_path": str(output_pred_path),
            "output_tmp_path": str(output_tmp_path),
            "prediction_link_mode": link_mode,
            "tmp_link_mode": tmp_link_mode,
            "mask_shape_mode": mask_shape_mode,
            "expanded_prediction": bool(expanded_prediction),
            "num_scene_vertices": scene_vertices,
            "num_source_pre_points": int(source_pre_points.shape[0]),
            "source_pre_points_ratio": float(source_pre_points.shape[0] / max(scene_vertices, 1)),
            "num_target_pre_points": int(target_pre_points.shape[0]),
            "target_pre_points_ratio": float(target_pre_points.shape[0] / max(scene_vertices, 1)),
            "num_prediction_union": int(prediction_union.shape[0]),
            "prediction_union_ratio": float(prediction_union.shape[0] / max(scene_vertices, 1)),
            "prediction_union_in_target_count": target_union_count,
            "prediction_union_in_target_ratio_of_scene": float(target_union_count / max(scene_vertices, 1)),
            "prediction_union_in_target_ratio_of_target": float(
                target_union_count / max(int(target_pre_points.shape[0]), 1)
            ),
            "num_gt_instances_in_target_pre_points": _count_gt_instances(gt_ids_crop),
            "num_gt_instances_fullmesh": _count_gt_instances(gt_ids_full),
            "num_pred_instances": int(pred_masks.shape[1]),
        }
    )
    return row


def _aggregate(args: argparse.Namespace, rows: list[dict[str, Any]], metrics: dict[str, float | None]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    modes = Counter(str(row.get("mask_shape_mode")) for row in ok_rows)
    expanded_count = int(sum(1 for row in ok_rows if row.get("expanded_prediction")))
    return {
        "output_config": args.output_config,
        "source_pred_config": args.pred_config,
        "source_pre_points_config": args.source_pre_points_config or _strip_suffix(args.pred_config, args.pred_suffix),
        "target_pre_points_config": args.pre_points_config,
        "pred_suffix": args.pred_suffix,
        "dataset": args.dataset,
        "scenes": len(rows),
        "ok_scenes": len(ok_rows),
        "missing_or_error_scenes": len(rows) - len(ok_rows),
        "ap": metrics["ap"],
        "ap50": metrics["ap50"],
        "ap25": metrics["ap25"],
        "mask_shape_modes": dict(modes),
        "expanded_prediction_scenes": expanded_count,
        "mean_source_pre_points_ratio": _mean(ok_rows, "source_pre_points_ratio"),
        "mean_target_pre_points_ratio": _mean(ok_rows, "target_pre_points_ratio"),
        "mean_prediction_union_ratio": _mean(ok_rows, "prediction_union_ratio"),
        "mean_prediction_union_in_target_ratio_of_scene": _mean(
            ok_rows, "prediction_union_in_target_ratio_of_scene"
        ),
        "mean_prediction_union_in_target_ratio_of_target": _mean(
            ok_rows, "prediction_union_in_target_ratio_of_target"
        ),
        "mean_gt_instances_in_target_pre_points": _mean(ok_rows, "num_gt_instances_in_target_pre_points"),
        "mean_gt_instances_fullmesh": _mean(ok_rows, "num_gt_instances_fullmesh"),
        "mean_num_pred_instances": _mean(ok_rows, "num_pred_instances"),
        "sum_num_prediction_union": _sum(ok_rows, "num_prediction_union"),
        "sum_target_pre_points": _sum(ok_rows, "num_target_pre_points"),
    }


def _fmt(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value * scale:.{digits}f}"
    return str(value)


def _update_combined_audit(audit_root: Path, aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    json_path = audit_root / "cross_prepoints_audit.json"
    payload: dict[str, Any]
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        payload = {"aggregates": [], "rows_by_config": {}}
    aggregates = [
        item for item in payload.get("aggregates", []) if item.get("output_config") != aggregate["output_config"]
    ]
    aggregates.append(aggregate)
    aggregates = sorted(aggregates, key=lambda item: str(item.get("output_config")))
    payload["aggregates"] = aggregates
    payload.setdefault("rows_by_config", {})[aggregate["output_config"]] = rows
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = audit_root / "cross_prepoints_audit.csv"
    fieldnames = [
        "output_config",
        "source_pred_config",
        "source_pre_points_config",
        "target_pre_points_config",
        "ap",
        "ap50",
        "ap25",
        "ok_scenes",
        "scenes",
        "mean_source_pre_points_ratio",
        "mean_target_pre_points_ratio",
        "mean_prediction_union_ratio",
        "mean_prediction_union_in_target_ratio_of_scene",
        "mean_prediction_union_in_target_ratio_of_target",
        "mean_gt_instances_in_target_pre_points",
        "mean_gt_instances_fullmesh",
        "mean_num_pred_instances",
        "expanded_prediction_scenes",
        "mask_shape_modes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in aggregates:
            row = {key: item.get(key) for key in fieldnames}
            row["mask_shape_modes"] = json.dumps(item.get("mask_shape_modes", {}), ensure_ascii=False)
            writer.writerow(row)

    md_path = audit_root / "cross_prepoints_audit.md"
    lines = [
        "# Cross Pre-Points Audit",
        "",
        "| output_config | source pred | source pre_points | target pre_points | AP | AP50 | AP25 | OK | target pre % | union % | union in target % scene/target | GT crop/full | #pred | modes | expanded |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("output_config")),
                    str(item.get("source_pred_config")),
                    str(item.get("source_pre_points_config")),
                    str(item.get("target_pre_points_config")),
                    _fmt(item.get("ap"), 100.0, 4),
                    _fmt(item.get("ap50"), 100.0, 4),
                    _fmt(item.get("ap25"), 100.0, 4),
                    f"{item.get('ok_scenes')}/{item.get('scenes')}",
                    _fmt(item.get("mean_target_pre_points_ratio"), 100.0, 4),
                    _fmt(item.get("mean_prediction_union_ratio"), 100.0, 4),
                    f"{_fmt(item.get('mean_prediction_union_in_target_ratio_of_scene'), 100.0, 4)}/{_fmt(item.get('mean_prediction_union_in_target_ratio_of_target'), 100.0, 4)}",
                    f"{_fmt(item.get('mean_gt_instances_in_target_pre_points'), 1.0, 2)}/{_fmt(item.get('mean_gt_instances_fullmesh'), 1.0, 2)}",
                    _fmt(item.get("mean_num_pred_instances"), 1.0, 2),
                    json.dumps(item.get("mask_shape_modes", {}), ensure_ascii=False),
                    str(item.get("expanded_prediction_scenes")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--source-pre-points-config", default="")
    parser.add_argument("--pre-points-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument("--gt-root", default="data/scannet/gt")
    parser.add_argument("--no-class", action="store_true")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument(
        "--eval-policy",
        default="cross_fixed_support",
        help="Human-readable unified evaluation policy to store in the diagnostic manifest.",
    )
    parser.add_argument(
        "--allow-diagnostic-eval",
        action="store_true",
        help="pass evaluator's --allow-oracle-eval gate for this diagnostic-only cross-support AP",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    seq_list = root / args.seq_list
    scene_ids = _read_seq_list(seq_list)
    source_pre_points_config = args.source_pre_points_config or _strip_suffix(args.pred_config, args.pred_suffix)

    source_pred_dir = _prediction_dir(root, args.pred_config, args.pred_suffix)
    output_pred_dir = _prediction_dir(root, args.output_config, args.pred_suffix)
    output_tmp_dir = root / "data" / "TMP" / args.output_config
    output_file = root / args.output_file

    rows = [
        _materialize_scene(
            args=args,
            root=root,
            scene_id=scene_id,
            source_pred_dir=source_pred_dir,
            output_pred_dir=output_pred_dir,
            output_tmp_dir=output_tmp_dir,
            source_pre_points_config=source_pre_points_config,
        )
        for scene_id in scene_ids
    ]
    error_rows = [row for row in rows if not row.get("ok")]
    if error_rows:
        examples = "; ".join(f"{row['scene_id']}: {row.get('error')}" for row in error_rows[:5])
        raise RuntimeError(f"cross-prepoints materialization failed for {len(error_rows)} scenes: {examples}")

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.pred_config, source_pre_points_config, args.pre_points_config],
        pre_points_policy="fixed_path",
        support_policy=f"cross_prepoints:{args.pre_points_config}",
        notes=(
            "Cross-pre_points diagnostic: prediction masks come from source config, "
            "TMP support comes from target pre_points config. Not a standalone method result."
        ),
        extra={
            "diagnostic_eval_requires_allow_oracle_eval": True,
            "eval_policy": args.eval_policy,
            "prediction_config": args.pred_config,
            "pre_points_config": args.pre_points_config,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix=args.pred_suffix)

    if not args.skip_eval:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "evaluation.evaluate",
            "--pred_path",
            str(output_pred_dir),
            "--gt_path",
            str(root / args.gt_root),
            "--dataset",
            args.dataset,
            "--output_file",
            str(output_file),
            "--tmp_root",
            str(root / "data" / "TMP"),
            "--tmp_config",
            args.output_config,
        ]
        if args.no_class:
            cmd.append("--no_class")
        if args.require_manifest:
            cmd.append("--require-manifest")
        if args.allow_diagnostic_eval:
            cmd.append("--allow-oracle-eval")
        print("[cross-prepoints] running evaluation:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    metrics = _parse_metric_file(output_file)
    aggregate = _aggregate(args, rows, metrics)
    audit_root = root / args.audit_root
    per_config_path = audit_root / "cross_prepoints" / f"{args.output_config}_summary.json"
    per_config_path.parent.mkdir(parents=True, exist_ok=True)
    per_config_payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    per_config_path.write_text(json.dumps(_json_safe(per_config_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _update_combined_audit(audit_root, aggregate, rows)
    print(f"[cross-prepoints] wrote {per_config_path}")
    print(f"[cross-prepoints] updated {audit_root / 'cross_prepoints_audit.md'}")


if __name__ == "__main__":
    main()
