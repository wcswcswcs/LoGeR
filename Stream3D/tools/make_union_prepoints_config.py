from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _scene_vertices(root: Path, gt_root: str, scene_id: str) -> int | None:
    path = root / gt_root / f"{scene_id}.txt"
    if not path.exists():
        return None
    return int(np.loadtxt(path).shape[0])


def _build_scene_union(args: argparse.Namespace, root: Path, scene_id: str, configs: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_id": scene_id,
        "ok": False,
        "error": None,
        "input_counts": {},
        "input_paths": {},
    }
    arrays: list[np.ndarray] = []
    for config in configs:
        path = _tmp_path(root, config, scene_id)
        row["input_paths"][config] = str(path)
        if not path.exists():
            row["error"] = f"missing pre_points: {path}"
            return row
        values = np.load(path).astype(np.int64).reshape(-1)
        row["input_counts"][config] = int(values.shape[0])
        arrays.append(values)
    union = np.unique(np.concatenate(arrays).astype(np.int64)) if arrays else np.empty((0,), dtype=np.int64)
    vertices = _scene_vertices(root, args.gt_root, scene_id)
    if vertices is not None:
        valid = (union >= 0) & (union < int(vertices))
        if not np.all(valid):
            row["error"] = f"union has {int(np.count_nonzero(~valid))} out-of-range entries"
            return row
    output_dir = root / "data" / "TMP" / args.output_config
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_id}_pre_points.npy"
    np.save(output_path, union)
    row.update(
        {
            "ok": True,
            "output_path": str(output_path),
            "num_union_pre_points": int(union.shape[0]),
            "scene_vertices": vertices,
            "union_pre_points_ratio": float(union.shape[0] / max(int(vertices or 0), 1)),
        }
    )
    return row


def _aggregate(rows: list[dict[str, Any]], configs: list[str]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    out: dict[str, Any] = {
        "scenes": len(rows),
        "ok_scenes": len(ok_rows),
        "missing_or_error_scenes": len(rows) - len(ok_rows),
        "mean_union_pre_points": float(mean(row["num_union_pre_points"] for row in ok_rows)) if ok_rows else None,
        "mean_union_pre_points_ratio": (
            float(mean(row["union_pre_points_ratio"] for row in ok_rows)) if ok_rows else None
        ),
    }
    for config in configs:
        vals = [int(row["input_counts"][config]) for row in ok_rows if config in row.get("input_counts", {})]
        out[f"mean_input_count_{config}"] = float(mean(vals)) if vals else None
    return out


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Union Pre-Points Config",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| Scene | OK | union pre_points | union % | error |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        ratio = row.get("union_pre_points_ratio")
        ratio_text = "NA" if ratio is None else f"{float(ratio) * 100.0:.4f}"
        lines.append(
            f"| {row['scene_id']} | {row.get('ok')} | {row.get('num_union_pre_points', 'NA')} | {ratio_text} | {row.get('error') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--configs", required=True, help="comma-separated TMP configs to union")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--gt-root", default="data/scannet/gt")
    parser.add_argument("--summary-root", default="outputs/audit/v7_gap_matrix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    if not configs:
        raise ValueError("--configs must contain at least one config")
    scene_ids = _read_seq_list(root / args.seq_list)
    rows = [_build_scene_union(args, root, scene_id, configs) for scene_id in scene_ids]
    errors = [row for row in rows if not row.get("ok")]
    if errors:
        examples = "; ".join(f"{row['scene_id']}: {row.get('error')}" for row in errors[:5])
        raise RuntimeError(f"union pre_points failed for {len(errors)} scenes: {examples}")

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=configs,
        pre_points_policy="union_prepoints",
        support_policy="union_prepoints",
        notes="Diagnostic support universe built as the union of multiple TMP pre_points configs.",
    )
    write_prediction_manifest(args.output_config, manifest, root=root)

    payload = {"args": vars(args), "aggregate": _aggregate(rows, configs), "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_config}.json"
    csv_path = out_dir / f"{args.output_config}.csv"
    md_path = out_dir / f"{args.output_config}.md"
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    _write_markdown(md_path, payload)
    print(f"[union-prepoints] wrote {json_path}")
    print(f"[union-prepoints] wrote {csv_path}")
    print(f"[union-prepoints] wrote {md_path}")


if __name__ == "__main__":
    main()
