from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


AP_CORE_FUNCTIONS = ("evaluate_matches", "compute_averages")


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_function_source(source: str, function_name: str) -> str | None:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def _evaluator_checks(root: Path) -> dict[str, Any]:
    path = root / "evaluation" / "evaluate.py"
    result: dict[str, Any] = {
        "root": str(root),
        "path": str(path),
        "exists": path.exists(),
        "file_sha256": None,
        "has_pre_points_load": False,
        "has_tmp_root_arg": False,
        "has_tmp_config_arg": False,
        "ap_core": {},
        "error": None,
    }
    if not path.exists():
        result["error"] = "evaluation/evaluate.py missing"
        return result
    try:
        source = path.read_text(encoding="utf-8")
        result["file_sha256"] = _sha256_text(source)
        result["has_pre_points_load"] = "_pre_points.npy" in source and "np.load" in source
        result["has_tmp_root_arg"] = "--tmp_root" in source
        result["has_tmp_config_arg"] = "--tmp_config" in source
        for function_name in AP_CORE_FUNCTIONS:
            function_source = _extract_function_source(source, function_name)
            result["ap_core"][function_name] = {
                "present": function_source is not None,
                "sha256": _sha256_text(function_source) if function_source is not None else None,
            }
    except Exception as exc:  # pragma: no cover - audit should report, not hide, parse failures.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _compare_evaluators(orig_root: Path, current_root: Path) -> dict[str, Any]:
    orig = _evaluator_checks(orig_root)
    current = _evaluator_checks(current_root)
    function_equal: dict[str, bool | None] = {}
    for function_name in AP_CORE_FUNCTIONS:
        orig_hash = orig.get("ap_core", {}).get(function_name, {}).get("sha256")
        current_hash = current.get("ap_core", {}).get(function_name, {}).get("sha256")
        function_equal[function_name] = None if orig_hash is None or current_hash is None else orig_hash == current_hash
    return {
        "original": orig,
        "current": current,
        "ap_core_equal_by_function": function_equal,
        "all_ap_core_equal": all(v is True for v in function_equal.values()),
    }


def _result_file_for_config(current_root: Path, config: str) -> Path:
    return current_root / "data" / "evaluation" / "scannet" / f"{config}_class_agnostic.txt"


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


def _count_gt_instances(gt_ids: np.ndarray) -> int:
    valid = gt_ids[gt_ids >= 1000]
    if valid.size == 0:
        return 0
    return int(np.unique(valid.astype(np.int64)).shape[0])


def _prediction_dir(current_root: Path, config: str) -> Path:
    if config.endswith("_class_agnostic"):
        return current_root / "data" / "prediction" / config
    return current_root / "data" / "prediction" / f"{config}_class_agnostic"


def _tmp_path_for_config(current_root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        current_root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        current_root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _audit_scene(current_root: Path, config: str, scene_id: str) -> dict[str, Any]:
    pred_path = _prediction_dir(current_root, config) / f"{scene_id}.npz"
    tmp_path = _tmp_path_for_config(current_root, config, scene_id)
    gt_path = current_root / "data" / "scannet" / "gt" / f"{scene_id}.txt"
    row: dict[str, Any] = {
        "config": config,
        "scene_id": scene_id,
        "pred_path": str(pred_path),
        "tmp_path": str(tmp_path),
        "gt_path": str(gt_path),
        "ok": False,
        "error": None,
    }
    missing = [str(path) for path in (pred_path, tmp_path, gt_path) if not path.exists()]
    if missing:
        row["error"] = "missing files: " + "; ".join(missing)
        return row
    try:
        with np.load(pred_path) as pred:
            pred_masks = np.asarray(pred["pred_masks"])
            pred_score = np.asarray(pred["pred_score"]) if "pred_score" in pred else np.empty((pred_masks.shape[1],))
        pre_points = np.load(tmp_path).astype(np.int64)
        gt_ids_full = np.loadtxt(gt_path).astype(np.int64)
        scene_vertices = int(gt_ids_full.shape[0])
        if pred_masks.shape[0] != scene_vertices:
            row["error"] = f"pred vertex count {pred_masks.shape[0]} != gt vertex count {scene_vertices}"
            return row
        valid_pre_points = pre_points[(pre_points >= 0) & (pre_points < scene_vertices)]
        if valid_pre_points.shape[0] != pre_points.shape[0]:
            row["error"] = f"pre_points has {pre_points.shape[0] - valid_pre_points.shape[0]} out-of-range entries"
            return row
        prediction_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
        pre_set = set(pre_points.tolist())
        union_set = set(prediction_union.tolist())
        if pre_set == union_set:
            policy_like = "recompute_like"
        elif union_set.issubset(pre_set):
            policy_like = "inherit_or_fixed_superset"
        else:
            policy_like = "inconsistent_union_not_subset"
        gt_ids_crop = gt_ids_full[pre_points]
        row.update(
            {
                "ok": True,
                "num_scene_vertices": scene_vertices,
                "num_pre_points": int(pre_points.shape[0]),
                "pre_points_ratio": float(pre_points.shape[0] / max(scene_vertices, 1)),
                "num_prediction_union": int(prediction_union.shape[0]),
                "prediction_union_ratio": float(prediction_union.shape[0] / max(scene_vertices, 1)),
                "pre_points_equals_prediction_union": bool(pre_set == union_set),
                "prediction_union_subset_of_pre_points": bool(union_set.issubset(pre_set)),
                "pre_points_policy_like": policy_like,
                "num_pred_instances": int(pred_masks.shape[1]),
                "num_pred_scores": int(pred_score.shape[0]),
                "num_gt_instances_in_pre_points": _count_gt_instances(gt_ids_crop),
                "num_gt_instances_fullmesh": _count_gt_instances(gt_ids_full),
            }
        )
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get("ok") and row.get(key) is not None]
    return float(mean(values)) if values else None


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows if row.get("ok") and row.get(key) is not None))


def _aggregate(config: str, rows: list[dict[str, Any]], metrics: dict[str, float | None]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    return {
        "config": config,
        "scenes": len(rows),
        "ok_scenes": len(ok_rows),
        "missing_or_error_scenes": len(rows) - len(ok_rows),
        "ap": metrics["ap"],
        "ap50": metrics["ap50"],
        "ap25": metrics["ap25"],
        "mean_pre_points_ratio": _mean(ok_rows, "pre_points_ratio"),
        "mean_prediction_union_ratio": _mean(ok_rows, "prediction_union_ratio"),
        "mean_num_pred_instances": _mean(ok_rows, "num_pred_instances"),
        "sum_num_pre_points": _sum(ok_rows, "num_pre_points"),
        "sum_num_prediction_union": _sum(ok_rows, "num_prediction_union"),
        "equals_union_scenes": int(sum(1 for row in ok_rows if row.get("pre_points_equals_prediction_union"))),
        "union_subset_scenes": int(sum(1 for row in ok_rows if row.get("prediction_union_subset_of_pre_points"))),
        "mean_gt_instances_in_pre_points": _mean(ok_rows, "num_gt_instances_in_pre_points"),
        "mean_gt_instances_fullmesh": _mean(ok_rows, "num_gt_instances_fullmesh"),
    }


def _fmt(value: Any, scale: float = 1.0, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value * scale:.{digits}f}"
    return str(value)


def _write_markdown(
    output: Path,
    args: argparse.Namespace,
    evaluator_compare: dict[str, Any],
    aggregates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Stream4D ScanNet Evaluator Protocol Audit")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- current_root: `{Path(args.current_root).resolve()}`")
    lines.append(f"- orig_stream3d_root: `{Path(args.orig_stream3d_root).resolve()}`")
    lines.append(f"- seq_list: `{Path(args.seq_list).resolve()}`")
    lines.append(f"- configs: `{args.configs}`")
    lines.append("")
    lines.append("## Evaluator Code Checks")
    lines.append("")
    orig = evaluator_compare["original"]
    current = evaluator_compare["current"]
    lines.append(f"- original evaluate.py exists: `{orig['exists']}`")
    lines.append(f"- current evaluate.py exists: `{current['exists']}`")
    lines.append(f"- original has pre_points load: `{orig['has_pre_points_load']}`")
    lines.append(f"- current has pre_points load: `{current['has_pre_points_load']}`")
    lines.append(f"- current has tmp root/config arguments: `{current['has_tmp_root_arg']}` / `{current['has_tmp_config_arg']}`")
    lines.append(f"- AP core functions equal by hash: `{evaluator_compare['all_ap_core_equal']}`")
    for function_name, is_equal in evaluator_compare["ap_core_equal_by_function"].items():
        lines.append(f"  - {function_name}: `{is_equal}`")
    if orig.get("error"):
        lines.append(f"- original evaluator error: `{orig['error']}`")
    if current.get("error"):
        lines.append(f"- current evaluator error: `{current['error']}`")
    lines.append("")
    lines.append("## Config Summary")
    lines.append("")
    lines.append(
        "| Config | AP | AP50 | AP25 | OK scenes | mean pre_points % | mean union % | equals union scenes | union subset scenes | mean #pred | mean GT crop/full |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["config"],
                    _fmt(item["ap"], 100.0),
                    _fmt(item["ap50"], 100.0),
                    _fmt(item["ap25"], 100.0),
                    f"{item['ok_scenes']}/{item['scenes']}",
                    _fmt(item["mean_pre_points_ratio"], 100.0),
                    _fmt(item["mean_prediction_union_ratio"], 100.0),
                    str(item["equals_union_scenes"]),
                    str(item["union_subset_scenes"]),
                    _fmt(item["mean_num_pred_instances"]),
                    f"{_fmt(item['mean_gt_instances_in_pre_points'])}/{_fmt(item['mean_gt_instances_fullmesh'])}",
                ]
            )
            + " |"
        )
    error_rows = [row for row in rows if not row.get("ok")]
    if error_rows:
        lines.append("")
        lines.append("## Missing Or Error Rows")
        lines.append("")
        for row in error_rows[:50]:
            lines.append(f"- `{row['config']}` `{row['scene_id']}`: {row.get('error')}")
        if len(error_rows) > 50:
            lines.append(f"- ... {len(error_rows) - 50} more rows omitted from markdown; see JSON.")
    lines.append("")
    lines.append("## Scene-Level Sample")
    lines.append("")
    lines.append("| Config | Scene | pre_points | union | pre % | union % | equal | subset | #pred | GT crop/full |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---:|---:|")
    for row in [row for row in rows if row.get("ok")][:40]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config"],
                    row["scene_id"],
                    str(row["num_pre_points"]),
                    str(row["num_prediction_union"]),
                    _fmt(row["pre_points_ratio"], 100.0),
                    _fmt(row["prediction_union_ratio"], 100.0),
                    str(row["pre_points_equals_prediction_union"]),
                    str(row["prediction_union_subset_of_pre_points"]),
                    str(row["num_pred_instances"]),
                    f"{row['num_gt_instances_in_pre_points']}/{row['num_gt_instances_fullmesh']}",
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_plots(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written: list[str] = []
    ok_rows = [row for row in rows if row.get("ok")]
    if not ok_rows:
        return written
    for key, ylabel in (("pre_points_ratio", "pre_points ratio"), ("prediction_union_ratio", "prediction union ratio")):
        fig, ax = plt.subplots(figsize=(9, 4))
        configs = sorted({row["config"] for row in ok_rows})
        data = [[row[key] for row in ok_rows if row["config"] == config] for config in configs]
        ax.boxplot(data, tick_labels=configs, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        plot_path = output.with_name(f"{output.stem}_{key}.png")
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        written.append(str(plot_path))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-stream3d-root", required=True)
    parser.add_argument("--current-root", default=".")
    parser.add_argument("--configs", required=True, help="comma-separated config names")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    current_root = Path(args.current_root).resolve()
    orig_root = Path(args.orig_stream3d_root).resolve()
    output = Path(args.output)
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    scene_ids = _read_seq_list(Path(args.seq_list))
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    evaluator_compare = _compare_evaluators(orig_root, current_root)
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for config in configs:
        config_rows = [_audit_scene(current_root=current_root, config=config, scene_id=scene_id) for scene_id in scene_ids]
        metrics = _parse_metric_file(_result_file_for_config(current_root, config))
        rows.extend(config_rows)
        aggregates.append(_aggregate(config, config_rows, metrics))

    plots = _write_plots(output, rows)
    payload = {
        "args": vars(args),
        "evaluator_compare": evaluator_compare,
        "aggregates": aggregates,
        "rows": rows,
        "plots": plots,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output, args, evaluator_compare, aggregates, rows)
    print(f"[audit] wrote {output}")
    print(f"[audit] wrote {output.with_suffix('.json')}")
    for plot in plots:
        print(f"[audit] wrote {plot}")


if __name__ == "__main__":
    main()
