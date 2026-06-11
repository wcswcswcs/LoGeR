from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.summarize_v9_unified_eval import (
    _aggregate_row,
    _attach_same_support_gaps,
    _fmt,
    _json_safe,
    _read_seq_list,
)


def _manifest_integrity_pass(row: dict[str, Any]) -> bool:
    if not row.get("manifest_path"):
        return False
    if not row.get("eval_policy"):
        return False
    if bool(row.get("uses_gt_for_prediction", False)):
        return False
    if bool(row.get("is_method_result", False)):
        if bool(row.get("is_diagnostic_only", False)):
            return False
        if bool(row.get("uses_gt_for_diagnostic", False)):
            return False
    return True


def _method_table_allowed(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("is_method_result", False))
        and not bool(row.get("is_diagnostic_only", False))
        and not bool(row.get("uses_gt_for_prediction", False))
        and not bool(row.get("uses_gt_for_diagnostic", False))
    )


def _add_v10_fields(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    manifest = row.get("manifest") or {}
    row["uses_gt_for_prediction"] = bool(manifest.get("uses_gt_for_prediction", row.get("uses_gt", False)))
    row["uses_gt_for_diagnostic"] = bool(manifest.get("uses_gt_for_diagnostic", row.get("uses_gt", False)))
    row["support_source"] = str(spec.get("support_source", manifest.get("support_source", "")))
    row["geometry_source"] = str(spec.get("geometry_source", manifest.get("geometry_source", "")))
    row["runtime_seconds"] = spec.get("runtime_seconds", manifest.get("runtime_seconds"))
    row["manifest_integrity_pass"] = _manifest_integrity_pass(row)
    row["method_table_allowed"] = _method_table_allowed(row)
    row["mean_points_per_object"] = row.get("mean_points_per_pred")
    if row["mean_points_per_object"] is None and row.get("points_per_scene") is not None:
        row["mean_points_per_object"] = float(row["points_per_scene"]) / max(float(row.get("num_pred_per_scene") or 0.0), 1.0)
    return row


def _write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Stream4D v10 Unified Evaluation Matrix",
        "",
        "Official AP columns are parsed from `evaluation.evaluate` output. Diagnostic fields are computed from the corresponding prediction and TMP support artifacts. `NA` means the value was not measured rather than inferred.",
        "",
        "## Required Matrix",
        "",
        "| method | prediction | pre_points | policy | AP | AP50 | AP25 | support source | geometry | pre% | union% | union target scene/target | GT crop/full | #pred | mean pts/object | conflict | tiny<100 | large>1000 | best IoU | GT>=.25 | GT>=.50 | missed | dup/GT | runtime sec | manifest pass | method table |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    str(row.get("prediction_config", "")),
                    str(row.get("pre_points_config", "")),
                    str(row.get("eval_policy", "")),
                    _fmt(row.get("ap"), 100.0),
                    _fmt(row.get("ap50"), 100.0),
                    _fmt(row.get("ap25"), 100.0),
                    str(row.get("support_source", "")),
                    str(row.get("geometry_source", "")),
                    _fmt(row.get("pre_points_ratio"), 100.0),
                    _fmt(row.get("prediction_union_ratio"), 100.0),
                    f"{_fmt(row.get('union_in_target_ratio_of_scene'), 100.0)}/{_fmt(row.get('union_in_target_ratio_of_target'), 100.0)}",
                    f"{_fmt(row.get('gt_crop'), 1.0, 2)}/{_fmt(row.get('gt_full'), 1.0, 2)}",
                    _fmt(row.get("num_pred_per_scene"), 1.0, 2),
                    _fmt(row.get("mean_points_per_object"), 1.0, 2),
                    _fmt(row.get("conflict_rate"), 100.0),
                    _fmt(row.get("tiny_mask_ratio_lt100_vertices"), 100.0),
                    _fmt(row.get("large_mask_ratio_gt1000_vertices"), 100.0),
                    _fmt(row.get("per_gt_best_iou_mean"), 1.0),
                    str(row.get("gt_iou_ge_025_count")),
                    str(row.get("gt_iou_ge_050_count")),
                    str(row.get("missed_eval_gt_count_iou_lt_025")),
                    _fmt(row.get("duplicate_predictions_per_gt_mean_at_025"), 1.0),
                    _fmt(row.get("runtime_seconds"), 1.0, 2),
                    str(row.get("manifest_integrity_pass")),
                    str(row.get("method_table_allowed")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Same-Support Delta",
            "",
            "| method | pre_points | Stream3D AP/AP50/AP25 | method - Stream3D AP/AP50/AP25 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    str(row.get("pre_points_config", "")),
                    f"{_fmt(row.get('stream3d_same_support_ap'), 100.0)}/{_fmt(row.get('stream3d_same_support_ap50'), 100.0)}/{_fmt(row.get('stream3d_same_support_ap25'), 100.0)}",
                    f"{_fmt(row.get('same_support_gap_ap'), 100.0)}/{_fmt(row.get('same_support_gap_ap50'), 100.0)}/{_fmt(row.get('same_support_gap_ap25'), 100.0)}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- matrix: `{Path(args.matrix_json).resolve()}`",
            f"- seq_list: `{Path(args.seq_list).resolve()}`",
            f"- root: `{Path(args.root).resolve()}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def heat(metric: str, filename: str, title: str) -> None:
        preds = sorted({str(row.get("prediction_config", "")) for row in rows})
        supports = sorted({str(row.get("pre_points_config", "")) for row in rows})
        values = np.full((len(preds), len(supports)), np.nan, dtype=np.float64)
        for row in rows:
            if row.get(metric) is None:
                continue
            i = preds.index(str(row.get("prediction_config", "")))
            j = supports.index(str(row.get("pre_points_config", "")))
            values[i, j] = float(row[metric]) * 100.0
        fig, ax = plt.subplots(figsize=(max(8, len(supports) * 1.1), max(4, len(preds) * 0.55)))
        im = ax.imshow(values, cmap="viridis")
        ax.set_xticks(range(len(supports)), supports, rotation=35, ha="right")
        ax.set_yticks(range(len(preds)), preds)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    heat("ap", "v10_gap_matrix_heatmap_AP.png", "AP")
    heat("ap50", "v10_gap_matrix_heatmap_AP50.png", "AP50")

    labels = [str(row.get("method", row.get("output_config", ""))) for row in rows]
    if rows:
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 4))
        ax.bar(x, [float(row.get("pre_points_ratio") or 0.0) * 100.0 for row in rows])
        ax.set_ylabel("pre_points %")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        fig.tight_layout()
        path = output_dir / "v10_support_ratio_bar.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 4))
        ax.bar(x - 0.18, [float(row.get("gt_crop") or 0.0) for row in rows], width=0.36, label="crop")
        ax.bar(x + 0.18, [float(row.get("gt_full") or 0.0) for row in rows], width=0.36, label="full")
        ax.set_ylabel("mean GT instances")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "v10_gt_crop_full_bar.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 4))
        gaps = [np.nan if row.get("same_support_gap_ap") is None else float(row["same_support_gap_ap"]) * 100.0 for row in rows]
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(x, gaps)
        ax.set_ylabel("method - Stream3D AP")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        fig.tight_layout()
        path = output_dir / "v10_method_vs_stream3d_same_support_delta.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--stream3d-config", default="scannet")
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--plot-dir", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scene_ids = _read_seq_list((root / args.seq_list).resolve())
    matrix_path = Path(args.matrix_json)
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    specs = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise ValueError("--matrix-json must contain a list of row specs")

    rows = []
    for spec in specs:
        row = _aggregate_row(
            root=root,
            dataset=args.dataset,
            suffix=args.pred_suffix,
            scene_ids=scene_ids,
            row_spec=spec,
            min_region_size=int(args.min_region_size),
        )
        manifest_path = row.get("manifest_path")
        if manifest_path:
            try:
                row["manifest"] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                row["manifest"] = {}
        else:
            row["manifest"] = {}
        rows.append(_add_v10_fields(row, spec))
    _attach_same_support_gaps(rows, stream3d_config=args.stream3d_config)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_prefix.parent
    if not plot_dir.is_absolute():
        plot_dir = root / plot_dir
    plots = _plot_heatmaps(rows, plot_dir)
    payload = {"args": vars(args), "rows": rows, "plots": plots}
    output_prefix.with_suffix(".json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat_rows = [{key: value for key, value in row.items() if key not in {"scene_rows", "manifest"}} for row in rows]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else ["method"]
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(flat_rows))
    _write_markdown(output_prefix.with_suffix(".md"), rows, args)
    print(f"[v10-unified-eval] wrote {output_prefix.with_suffix('.json')}")
    print(f"[v10-unified-eval] wrote {output_prefix.with_suffix('.csv')}")
    print(f"[v10-unified-eval] wrote {output_prefix.with_suffix('.md')}")
    for plot in plots:
        print(f"[v10-unified-eval] wrote {plot}")


if __name__ == "__main__":
    main()
