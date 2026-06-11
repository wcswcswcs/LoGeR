from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            row = dict(row)
            row["source_matrix"] = str(path)
            rows.append(row)
    return rows


def _tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    ap = row.get("ap")
    pre = row.get("pre_points_ratio")
    gap = row.get("same_support_gap_ap")
    conflict = row.get("conflict_rate")
    if ap is not None and float(ap) < 0.18:
        tags.append("low_ap")
    if pre is not None and float(pre) < 0.05:
        tags.append("tiny_support")
    if gap is not None and float(gap) < -0.08:
        tags.append("stream3d_same_support_stronger")
    if conflict is not None and float(conflict) > 0.25:
        tags.append("high_conflict")
    if "on S0" in str(row.get("method", "")) or "on S1" in str(row.get("method", "")):
        tags.append("cross_support")
    return tags


def _fmt(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value) * scale:.4f}"
    except Exception:
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-json", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/audit/v13_visuals")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = _load_rows([Path(path) for path in args.matrix_json])
    rows = sorted(
        rows,
        key=lambda row: (
            0 if _tags(row) else 1,
            float(row.get("ap") or 0.0),
            str(row.get("method", "")),
        ),
    )[: int(args.limit)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for v13 diagnostic visuals: {exc}") from exc

    manifest = []
    for idx, row in enumerate(rows):
        metrics = [
            ("AP", row.get("ap"), 100.0),
            ("AP50", row.get("ap50"), 100.0),
            ("AP25", row.get("ap25"), 100.0),
            ("pre%", row.get("pre_points_ratio"), 100.0),
            ("union%", row.get("prediction_union_ratio"), 100.0),
            ("conflict%", row.get("conflict_rate"), 100.0),
            ("best IoU", row.get("per_gt_best_iou_mean"), 1.0),
        ]
        labels = [item[0] for item in metrics]
        values = [0.0 if item[1] is None else float(item[1]) * float(item[2]) for item in metrics]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#eeca3b", "#e45756", "#b279a2"]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, max(100.0, max(values) * 1.15 if values else 100.0))
        ax.set_title(str(row.get("method", ""))[:90])
        ax.tick_params(axis="x", labelrotation=30)
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        png_path = out_dir / f"v13_failure_panel_{idx:02d}.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        sidecar = {
            "scene": "probe5_aggregate",
            "method": row.get("method"),
            "prediction_config": row.get("prediction_config"),
            "pre_points_config": row.get("pre_points_config"),
            "eval_policy": row.get("eval_policy"),
            "AP": row.get("ap"),
            "AP50": row.get("ap50"),
            "AP25": row.get("ap25"),
            "pre_points_ratio": row.get("pre_points_ratio"),
            "prediction_union_ratio": row.get("prediction_union_ratio"),
            "GT_crop": row.get("gt_crop"),
            "GT_full": row.get("gt_full"),
            "failure_tags": _tags(row),
            "source_matrix": row.get("source_matrix"),
            "visual_type": "aggregate_metric_bar_panel",
        }
        sidecar_path = png_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(json_safe(sidecar), indent=2, sort_keys=True), encoding="utf-8")
        manifest.append({"png": str(png_path), "sidecar": str(sidecar_path), **sidecar})
    (out_dir / "v13_failure_visuals_manifest.json").write_text(
        json.dumps(json_safe({"num_visuals": len(manifest), "visuals": manifest}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"num_visuals": len(manifest), "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
