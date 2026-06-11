from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.d4rt_stream3d_geometry_adapter import D4RTStream3DGeometryAdapter


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
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
        }
    )
    numeric_mean = {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in numeric_keys
        if any(row.get(key) is not None for row in rows)
    }
    return {
        "algorithm": "v11_d4rt_stream3d_geometry_adapter",
        "mode": args.mode,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": args.mode != "raw",
        "is_method_result": False,
        "is_complete_stream3d_replacement": False,
        "num_scenes": int(len(rows)),
        "numeric_mean": numeric_mean,
    }


def _write_outputs(summary_root: Path, output_name: str, payload: dict[str, Any]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / f"{output_name}_summary.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["scenes"]
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
        }
    )
    with (summary_root / f"{output_name}_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene"] + numeric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ["scene"] + numeric_keys})
    lines = [
        f"# {output_name}",
        "",
        "This is a diagnostic geometry adapter. It materializes D4RT per-frame point clouds and mask mappings, but does not rerun original Stream3D local proposal/set-cover/manifold stages.",
        "",
        "| scene | anchors | median residual | p90 residual | spacing q50 | frames | empty mask mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {anchors} | {median} | {p90} | {spacing} | {frames} | {empty} |".format(
                scene=row.get("scene"),
                anchors=int(row.get("anchor_count") or 0),
                median="NA" if row.get("median_residual") is None else f"{float(row['median_residual']):.6g}",
                p90="NA" if row.get("p90_residual") is None else f"{float(row['p90_residual']):.6g}",
                spacing="NA" if row.get("point_spacing_q50") is None else f"{float(row['point_spacing_q50']):.6g}",
                frames=int(row.get("num_frames_materialized") or 0),
                empty="NA" if row.get("empty_mask_ratio_mean") is None else f"{float(row['empty_mask_ratio_mean']):.6g}",
            )
        )
    (summary_root / f"{output_name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--mode", choices=["raw", "scene_sim3", "window_sim3"], required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--output-root", default="outputs/v11_d4rt_stream3d_geometry_adapter")
    parser.add_argument("--summary-root", default="outputs/audit/v11_d4rt_stream3d_geometry")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    args = parser.parse_args()

    adapter = D4RTStream3DGeometryAdapter(
        debug_root=args.debug_root,
        output_root=Path(args.output_root) / args.output_name,
        mode=args.mode,
        backbone=args.backbone,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_anchors=int(args.max_anchors),
        robust_trim_percentile=float(args.robust_trim_percentile),
    )
    scenes = [adapter.materialize_scene(scene) for scene in _read_seq_list(Path(args.seq_list))]
    payload = {"summary": _aggregate(scenes, args), "scenes": scenes}
    _write_outputs(Path(args.summary_root), args.output_name, payload)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
