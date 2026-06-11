from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def process_scene(root: Path, scene_id: str, input_configs: list[str], output_config: str) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    input_counts: dict[str, int] = {}
    for config in input_configs:
        path = _tmp_path(root, config, scene_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing pre_points for {config}/{scene_id}: {path}")
        arr = np.load(path).astype(np.int64, copy=False)
        arrays.append(arr)
        input_counts[config] = int(arr.shape[0])
    union = np.unique(np.concatenate(arrays) if arrays else np.zeros((0,), dtype=np.int64)).astype(np.int64)
    out_dir = root / "data" / "TMP" / output_config
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scene_id}_pre_points.npy", union)
    return {
        "scene_id": scene_id,
        "input_counts": input_counts,
        "union_count": int(union.shape[0]),
        "union_over_max_input": float(union.shape[0] / max(max(input_counts.values()), 1)) if input_counts else 0.0,
        "union_over_sum_inputs": float(union.shape[0] / max(sum(input_counts.values()), 1)) if input_counts else 0.0,
    }


def aggregate(rows: list[dict[str, Any]], input_configs: list[str], output_config: str) -> dict[str, Any]:
    input_means: dict[str, float] = {}
    for config in input_configs:
        values = [float(row["input_counts"][config]) for row in rows]
        input_means[f"mean_{config}_points"] = float(np.mean(values)) if values else 0.0
    union_values = [float(row["union_count"]) for row in rows]
    return {
        "output_config": output_config,
        "input_configs": input_configs,
        "num_scenes": len(rows),
        "mean_union_points": float(np.mean(union_values)) if union_values else 0.0,
        "min_union_points": int(min(union_values)) if union_values else 0,
        "max_union_points": int(max(union_values)) if union_values else 0,
        "mean_union_over_max_input": float(np.mean([row["union_over_max_input"] for row in rows])) if rows else 0.0,
        "mean_union_over_sum_inputs": float(np.mean([row["union_over_sum_inputs"] for row in rows])) if rows else 0.0,
        **input_means,
    }


def write_summary(root: Path, output_config: str, summary_root: str, payload: dict[str, Any]) -> None:
    out_dir = root / summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / output_config
    (prefix.with_suffix(".json")).write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = payload["rows"]
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scene_id", "union_count", "union_over_max_input", "union_over_sum_inputs", "input_counts"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["input_counts"] = json.dumps(row["input_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow({key: item.get(key) for key in fieldnames})
    lines = [
        "# Union Pre-Points Summary",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scene | union | union/max input | union/sum input | input counts |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row['union_count']} | "
            f"{row['union_over_max_input']:.6f} | {row['union_over_sum_inputs']:.6f} | "
            f"{json.dumps(row['input_counts'], ensure_ascii=False, sort_keys=True)} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TMP support config from the union of existing pre_points configs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-configs", required=True, help="Comma-separated pre_points config names.")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--summary-root", default="outputs/audit/v9_support")
    parser.add_argument("--eval-policy", default="support_union_only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    input_configs = [item.strip() for item in args.input_configs.split(",") if item.strip()]
    if len(input_configs) < 2:
        raise ValueError("--input-configs must contain at least two configs")
    scene_ids = _read_seq_list(root / args.seq_list)
    rows = [process_scene(root, scene_id, input_configs, args.output_config) for scene_id in scene_ids]
    payload = {
        "args": vars(args),
        "aggregate": aggregate(rows, input_configs, args.output_config),
        "rows": rows,
    }
    write_summary(root, args.output_config, args.summary_root, payload)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=input_configs,
        pre_points_policy="union",
        support_policy="union_prepoints:" + ",".join(input_configs),
        notes="Support-only config built by unioning existing pre_points arrays; no prediction masks are generated.",
        extra={
            "eval_policy": args.eval_policy,
            "input_configs": input_configs,
            "support_only": True,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root)
    print(f"[build-union-prepoints] wrote {root / args.summary_root / (args.output_config + '.md')}")


if __name__ == "__main__":
    main()
