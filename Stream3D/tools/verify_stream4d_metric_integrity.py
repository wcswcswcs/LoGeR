from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from stream4d.rescore_scannet import verify_object_dict_prediction_alignment
from tools.audit_stream3d_eval_protocol import (
    _aggregate,
    _audit_scene,
    _compare_evaluators,
    _json_safe,
    _parse_metric_file,
    _read_seq_list,
    _result_file_for_config,
)
from tools.scan_reportable_configs import scan_configs


def _prediction_dir(current_root: Path, config: str) -> Path:
    if config.endswith("_class_agnostic"):
        dirname = config
    else:
        dirname = f"{config}_class_agnostic"
    return current_root / "data" / "prediction" / dirname


def _object_dict_path(current_root: Path, backbone: str, config: str, scene_id: str) -> Path:
    return (
        current_root
        / "data"
        / "scannet"
        / "processed"
        / scene_id
        / f"output_{backbone}"
        / "object"
        / config
        / "object_dict.npy"
    )


def _summary_object_dict_path(current_root: Path, config: str, scene_id: str) -> Path | None:
    summary_path = current_root / "outputs" / "v7_carrier_tracklet_graph" / f"{config}_{scene_id}_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    object_dict_path = payload.get("object_dict_write_path")
    if not object_dict_path:
        return None
    path = Path(str(object_dict_path))
    if not path.is_absolute():
        path = current_root / path
    return path


def _rescore_gt_read_check(current_root: Path) -> dict[str, Any]:
    path = current_root / "stream4d" / "rescore_scannet.py"
    result = {
        "path": str(path),
        "exists": path.exists(),
        "gt_files_read_by_rescore": False,
        "matches": [],
        "error": None,
    }
    if not path.exists():
        result["error"] = "stream4d/rescore_scannet.py missing"
        return result
    source = path.read_text(encoding="utf-8")
    forbidden = ["gt_path", "data/scannet/gt", "/gt/", "np.loadtxt", "loadtxt("]
    matches = [item for item in forbidden if item in source]
    result["matches"] = matches
    result["gt_files_read_by_rescore"] = bool(matches)
    return result


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _object_alignment_for_config(
    current_root: Path,
    backbone: str,
    config: str,
    scene_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    scene_rows: list[dict[str, Any]] = []
    instance_ious: list[float] = []
    for scene_id in scene_ids:
        pred_path = _prediction_dir(current_root, config) / f"{scene_id}.npz"
        processed_object_path = _object_dict_path(current_root, backbone, config, scene_id)
        summary_object_path = _summary_object_dict_path(current_root, config, scene_id)
        object_path = processed_object_path
        object_path_source = "processed"
        if not object_path.exists() and summary_object_path is not None:
            object_path = summary_object_path
            object_path_source = "summary_fallback"
        row: dict[str, Any] = {
            "scene_id": scene_id,
            "pred_path": str(pred_path),
            "object_dict_path": str(object_path),
            "object_dict_processed_path": str(processed_object_path),
            "object_dict_summary_fallback_path": str(summary_object_path) if summary_object_path else None,
            "object_dict_path_source": object_path_source,
            "alignment_checked": False,
            "cannot_verify_alignment": True,
            "alignment_mean_iou": None,
            "alignment_min_iou": None,
            "alignment_failed_instances": 0,
            "error": None,
        }
        if not pred_path.exists() or not object_path.exists():
            missing = [str(path) for path in (pred_path, object_path) if not path.exists()]
            row["error"] = "missing files: " + "; ".join(missing)
            scene_rows.append(row)
            continue
        try:
            with np.load(pred_path) as pred:
                pred_masks = np.asarray(pred["pred_masks"])
            object_dict = np.load(object_path, allow_pickle=True).item()
            object_items = [(int(k), v) for k, v in sorted(object_dict.items(), key=lambda item: int(item[0]))]
            alignment = verify_object_dict_prediction_alignment(
                pred_masks,
                object_items,
                threshold=threshold,
                include_records=True,
            )
            ious = [float(item["point_iou"]) for item in alignment["alignment_records"]]
            instance_ious.extend(ious)
            row.update(
                {
                    "alignment_checked": bool(alignment["alignment_checked"]),
                    "cannot_verify_alignment": bool(alignment["cannot_verify_alignment"]),
                    "alignment_num_checked": int(alignment["alignment_num_checked"]),
                    "alignment_mean_iou": alignment["alignment_mean_iou"],
                    "alignment_min_iou": alignment["alignment_min_iou"],
                    "alignment_failed_instances": int(alignment["alignment_failed_instances"]),
                    "error": None,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        scene_rows.append(row)

    checked_rows = [row for row in scene_rows if row.get("alignment_checked")]
    failed_instances = int(sum(int(row.get("alignment_failed_instances") or 0) for row in checked_rows))
    return {
        "config": config,
        "scenes": len(scene_rows),
        "alignment_checked_scenes": len(checked_rows),
        "cannot_verify_scenes": int(sum(1 for row in scene_rows if row.get("cannot_verify_alignment"))),
        "missing_or_error_scenes": int(sum(1 for row in scene_rows if row.get("error"))),
        "object_dict_pred_alignment_mean_iou": _mean(
            [float(row["alignment_mean_iou"]) for row in checked_rows if row.get("alignment_mean_iou") is not None]
        ),
        "object_dict_pred_alignment_min_iou": (
            float(min(float(row["alignment_min_iou"]) for row in checked_rows if row.get("alignment_min_iou") is not None))
            if any(row.get("alignment_min_iou") is not None for row in checked_rows)
            else None
        ),
        "object_dict_pred_alignment_failed_instances": failed_instances,
        "scene_rows": scene_rows,
        "instance_iou_count": int(len(instance_ious)),
        "instance_ious": instance_ious,
    }


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    counts = Counter(str(row.get("pre_points_policy_like")) for row in ok_rows)
    return {
        "pre_points_policy_counts": dict(counts),
        "pre_points_equals_prediction_union": bool(
            ok_rows and all(bool(row.get("pre_points_equals_prediction_union")) for row in ok_rows)
        ),
        "prediction_union_subset_of_pre_points": bool(
            ok_rows and all(bool(row.get("prediction_union_subset_of_pre_points")) for row in ok_rows)
        ),
    }


def _write_plots(output_dir: Path, rows: list[dict[str, Any]], alignment_payloads: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    ok_rows = [row for row in rows if row.get("ok")]
    configs = sorted({row["config"] for row in ok_rows})
    if ok_rows and configs:
        for key, path_name, ylabel in (
            ("pre_points_ratio", "pre_points_ratio_by_config.png", "pre_points ratio"),
            ("prediction_union_ratio", "union_ratio_by_config.png", "prediction union ratio"),
        ):
            fig, ax = plt.subplots(figsize=(10, 4))
            data = [[row[key] for row in ok_rows if row["config"] == config] for config in configs]
            ax.boxplot(data, tick_labels=configs, showfliers=False)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=35)
            fig.tight_layout()
            path = output_dir / path_name
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 4))
        crop = [
            float(mean(row["num_gt_instances_in_pre_points"] for row in ok_rows if row["config"] == config))
            for config in configs
        ]
        full = [
            float(mean(row["num_gt_instances_fullmesh"] for row in ok_rows if row["config"] == config))
            for config in configs
        ]
        x = np.arange(len(configs))
        width = 0.38
        ax.bar(x - width / 2, crop, width, label="GT in pre_points")
        ax.bar(x + width / 2, full, width, label="GT fullmesh")
        ax.set_ylabel("mean GT instances")
        ax.set_xticks(x, configs, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "gt_crop_full_by_config.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    instance_ious: list[float] = []
    for payload in alignment_payloads:
        instance_ious.extend(float(v) for v in payload.get("instance_ious", []))
    if instance_ious:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(instance_ious, bins=40, range=(0.0, 1.0))
        ax.set_xlabel("object_dict / prediction-column point IoU")
        ax.set_ylabel("instances")
        fig.tight_layout()
        path = output_dir / "object_dict_alignment_iou_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def _fmt(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value * scale:.{digits}f}"
    return str(value)


def _write_markdown(
    output: Path,
    args: argparse.Namespace,
    evaluator_compare: dict[str, Any],
    rescore_check: dict[str, Any],
    config_summaries: list[dict[str, Any]],
    alignment_summaries: dict[str, dict[str, Any]],
    plots: list[str],
    reportable_scan: dict[str, Any],
    phase0_pass: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Stream4D v4 Metric Integrity Audit")
    lines.append("")
    lines.append("## Required Fields")
    lines.append("")
    lines.append(f"- evaluator_ap_core_equal_by_hash: `{evaluator_compare['all_ap_core_equal']}`")
    lines.append(f"- has_pre_points_load_original: `{evaluator_compare['original']['has_pre_points_load']}`")
    lines.append(f"- has_pre_points_load_current: `{evaluator_compare['current']['has_pre_points_load']}`")
    lines.append(f"- gt_files_read_by_rescore: `{rescore_check['gt_files_read_by_rescore']}`")
    lines.append(f"- num_configs_missing_manifest: `{reportable_scan['summary']['num_configs_missing_manifest']}`")
    lines.append(f"- num_oracle_configs: `{reportable_scan['summary']['num_oracle_configs']}`")
    lines.append(f"- num_reportable_method_configs: `{reportable_scan['summary']['num_reportable_method_configs']}`")
    lines.append(f"- num_suspicious_configs: `{reportable_scan['summary']['num_suspicious_configs']}`")
    lines.append(f"- num_uses_gt_for_prediction: `{reportable_scan['summary'].get('num_uses_gt_for_prediction', 0)}`")
    lines.append(f"- num_uses_gt_for_diagnostic_and_method_result: `{reportable_scan['summary'].get('num_uses_gt_for_diagnostic_and_method_result', 0)}`")
    lines.append(f"- phase0_pass: `{phase0_pass}`")
    if rescore_check["matches"]:
        lines.append(f"- rescore forbidden-token matches: `{rescore_check['matches']}`")
    lines.append("")
    lines.append("## Config Summary")
    lines.append("")
    lines.append(
        "| Config | AP | AP50 | AP25 | OK | pre % | union % | policy | equals union | union subset | GT crop/full | #pred | alignment mean/min | alignment failed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|")
    for item in config_summaries:
        alignment = alignment_summaries.get(item["config"], {})
        policy = json.dumps(item.get("pre_points_policy_counts", {}), ensure_ascii=False)
        lines.append(
            "| "
            + " | ".join(
                [
                    item["config"],
                    _fmt(item.get("ap"), 100.0, 4),
                    _fmt(item.get("ap50"), 100.0, 4),
                    _fmt(item.get("ap25"), 100.0, 4),
                    f"{item['ok_scenes']}/{item['scenes']}",
                    _fmt(item.get("mean_pre_points_ratio"), 100.0, 4),
                    _fmt(item.get("mean_prediction_union_ratio"), 100.0, 4),
                    policy,
                    str(item.get("pre_points_equals_prediction_union")),
                    str(item.get("prediction_union_subset_of_pre_points")),
                    f"{_fmt(item.get('mean_gt_instances_in_pre_points'), 1.0, 2)}/{_fmt(item.get('mean_gt_instances_fullmesh'), 1.0, 2)}",
                    _fmt(item.get("mean_num_pred_instances"), 1.0, 2),
                    f"{_fmt(alignment.get('object_dict_pred_alignment_mean_iou'), 1.0, 6)}/{_fmt(alignment.get('object_dict_pred_alignment_min_iou'), 1.0, 6)}",
                    str(alignment.get("object_dict_pred_alignment_failed_instances", "NA")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for path in plots:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- current_root: `{Path(args.current_root).resolve()}`")
    lines.append(f"- orig_stream3d_root: `{Path(args.orig_stream3d_root).resolve()}`")
    lines.append(f"- seq_list: `{Path(args.seq_list).resolve()}`")
    lines.append(f"- configs: `{args.configs}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-stream3d-root", required=True)
    parser.add_argument("--current-root", default=".")
    parser.add_argument("--configs", required=True, help="comma-separated config names")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output", default="outputs/audit/stream4d_v4_metric_integrity.md")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--alignment-iou-threshold", type=float, default=0.99)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="fail phase0_pass when any scanned config is missing a v5 manifest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    current_root = Path(args.current_root).resolve()
    orig_root = Path(args.orig_stream3d_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = current_root / output
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    scene_ids = _read_seq_list((current_root / args.seq_list).resolve())
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    evaluator_compare = _compare_evaluators(orig_root, current_root)
    rescore_check = _rescore_gt_read_check(current_root)
    reportable_scan = scan_configs(root=current_root, configs=configs)
    all_rows: list[dict[str, Any]] = []
    config_summaries: list[dict[str, Any]] = []
    alignment_payloads: list[dict[str, Any]] = []
    alignment_summaries: dict[str, dict[str, Any]] = {}

    for config in configs:
        rows = [_audit_scene(current_root=current_root, config=config, scene_id=scene_id) for scene_id in scene_ids]
        metrics = _parse_metric_file(_result_file_for_config(current_root, config))
        aggregate = _aggregate(config, rows, metrics)
        aggregate.update(_policy_summary(rows))
        all_rows.extend(rows)
        config_summaries.append(aggregate)

        alignment = _object_alignment_for_config(
            current_root=current_root,
            backbone=args.backbone,
            config=config,
            scene_ids=scene_ids,
            threshold=float(args.alignment_iou_threshold),
        )
        alignment_summaries[config] = {
            key: value for key, value in alignment.items() if key not in {"scene_rows", "instance_ious"}
        }
        alignment_payloads.append(alignment)

    plots = _write_plots(output.parent, all_rows, alignment_payloads)
    rows_ok = all(item["missing_or_error_scenes"] == 0 for item in config_summaries)
    alignment_ok = all(
        int(item.get("object_dict_pred_alignment_failed_instances", 0)) == 0
        for item in alignment_summaries.values()
        if int(item.get("alignment_checked_scenes", 0)) > 0
    )
    phase0_pass = bool(
        evaluator_compare["all_ap_core_equal"]
        and evaluator_compare["current"]["has_pre_points_load"]
        and not rescore_check["gt_files_read_by_rescore"]
        and rows_ok
        and alignment_ok
        and int(reportable_scan["summary"]["num_uses_gt_and_method_result"]) == 0
        and int(reportable_scan["summary"].get("num_uses_gt_for_prediction", 0)) == 0
        and int(reportable_scan["summary"].get("num_uses_gt_for_diagnostic_and_method_result", 0)) == 0
        and (not args.require_manifest or int(reportable_scan["summary"]["num_configs_missing_manifest"]) == 0)
    )

    payload = {
        "args": vars(args),
        "evaluator_compare": evaluator_compare,
        "rescore_gt_read_check": rescore_check,
        "config_summaries": config_summaries,
        "rows": all_rows,
        "alignment_summaries": alignment_summaries,
        "alignment_scene_rows": {
            item["config"]: item["scene_rows"] for item in alignment_payloads
        },
        "reportable_config_scan": reportable_scan,
        "plots": plots,
        "phase0_pass": phase0_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        output=output,
        args=args,
        evaluator_compare=evaluator_compare,
        rescore_check=rescore_check,
        config_summaries=config_summaries,
        alignment_summaries=alignment_summaries,
        plots=plots,
        reportable_scan=reportable_scan,
        phase0_pass=phase0_pass,
    )
    print(f"[metric-integrity] wrote {output}")
    print(f"[metric-integrity] wrote {output.with_suffix('.json')}")
    for plot in plots:
        print(f"[metric-integrity] wrote {plot}")
    print(f"[metric-integrity] phase0_pass={phase0_pass}")


if __name__ == "__main__":
    main()
