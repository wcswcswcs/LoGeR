from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe, read_seq_list
from tools.prediction_manifest import load_prediction_manifest


ROWS = [
    ("P0 on S0", "stream4d_v10_p0_on_s0_probe5", "S0"),
    ("P0 on S1", "stream4d_v10_p0_on_s1_probe5", "S1"),
    ("P0 on B1 support", "stream4d_v10_p0_on_s2_b1_probe5", "B1"),
    ("P0 on O38 support", "stream4d_v10_p0_on_o38_c055_probe5", "O38"),
    ("P0 on repair_cmask support", "stream4d_v17_p0_on_m17_repair_cmask_probe5", "repair_cmask"),
    ("B1 own", "stream4d_v8_b1_surfacelet_singlemask_probe5", "B1"),
    ("B1 on S0", "stream4d_v10_b1_on_s0_probe5", "S0"),
    ("B1 on S1", "stream4d_v10_b1_on_s1_probe5", "S1"),
    ("O38 own", "stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5", "O38"),
    ("O38 on S0", "stream4d_v10_o38_c055_on_s0_probe5", "S0"),
    ("O38 on S1", "stream4d_v10_o38_c055_on_s1_probe5", "S1"),
    ("repair_cmask own", "stream4d_v17_m17_repair_cmask_probe5", "repair_cmask"),
    ("repair_cmask on S0", "stream4d_v17_m17_repair_cmask_on_s0_probe5", "S0"),
    ("repair_cmask on S1", "stream4d_v17_m17_repair_cmask_on_s1_probe5", "S1"),
    ("P_v6compact on S1", "stream4d_v10_v6compact_on_s1_probe5", "S1"),
]


def _metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}


def _tmp_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene}_pre_points.npy"


def _pred_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "prediction" / f"{config}_class_agnostic" / f"{scene}.npz"


def _gt_count(gt: np.ndarray) -> int:
    return int(np.unique(gt[gt >= 1000]).shape[0])


def _scene_stats(root: Path, config: str, scene: str) -> dict[str, Any]:
    pred_path = _pred_path(root, config, scene)
    tmp_path = _tmp_path(root, config, scene)
    gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
    row: dict[str, Any] = {"scene": scene, "ok": False, "error": None}
    missing = [str(p) for p in (pred_path, tmp_path, gt_path) if not p.exists()]
    if missing:
        row["error"] = "missing: " + "; ".join(missing)
        return row
    gt = np.loadtxt(gt_path, dtype=np.int64)
    pre = np.load(tmp_path).astype(np.int64)
    with np.load(pred_path) as data:
        masks = np.asarray(data["pred_masks"], dtype=bool)
    if masks.shape[0] == gt.shape[0]:
        full = masks
    elif masks.shape[0] == pre.shape[0]:
        full = np.zeros((gt.shape[0], masks.shape[1]), dtype=bool)
        full[pre] = masks
    else:
        row["error"] = f"shape mismatch pred={masks.shape[0]} gt={gt.shape[0]} pre={pre.shape[0]}"
        return row
    union = np.any(full, axis=1) if full.shape[1] else np.zeros((gt.shape[0],), dtype=bool)
    pred_areas = np.count_nonzero(full, axis=0) if full.shape[1] else np.zeros((0,), dtype=np.int64)
    conflict = np.count_nonzero(np.count_nonzero(full, axis=1) > 1)
    gt_crop = gt[pre] if pre.size else np.zeros((0,), dtype=np.int64)
    gt_masks = []
    for gid in np.unique(gt_crop[gt_crop >= 1000]):
        gt_masks.append(gt == int(gid))
    best_ious: list[float] = []
    dup_counts: list[int] = []
    for gt_mask in gt_masks:
        if full.shape[1] == 0:
            best_ious.append(0.0)
            dup_counts.append(0)
            continue
        inter = np.count_nonzero(full & gt_mask[:, None], axis=0).astype(np.float64)
        pred_area = pred_areas.astype(np.float64)
        gt_area = float(np.count_nonzero(gt_mask))
        iou = inter / np.maximum(gt_area + pred_area - inter, 1.0)
        best_ious.append(float(np.max(iou)) if iou.size else 0.0)
        dup_counts.append(int(np.count_nonzero(iou >= 0.25)))
    row.update(
        {
            "ok": True,
            "num_scene_vertices": int(gt.shape[0]),
            "num_pre_points": int(pre.shape[0]),
            "pre_points_ratio": float(pre.shape[0] / max(gt.shape[0], 1)),
            "num_prediction_union": int(np.count_nonzero(union)),
            "prediction_union_ratio": float(np.count_nonzero(union) / max(gt.shape[0], 1)),
            "prediction_union_in_target_ratio_of_scene": float(np.count_nonzero(union[pre]) / max(gt.shape[0], 1))
            if pre.size
            else 0.0,
            "prediction_union_in_target_ratio_of_target": float(np.count_nonzero(union[pre]) / max(pre.shape[0], 1))
            if pre.size
            else 0.0,
            "gt_crop": _gt_count(gt_crop),
            "gt_full": _gt_count(gt),
            "num_pred": int(full.shape[1]),
            "mean_points_per_object": float(np.mean(pred_areas)) if pred_areas.size else 0.0,
            "tiny_mask_ratio_lt100": float(np.mean(pred_areas < 100)) if pred_areas.size else 0.0,
            "large_mask_ratio_gt1000": float(np.mean(pred_areas > 1000)) if pred_areas.size else 0.0,
            "conflict_rate": float(conflict / max(np.count_nonzero(union), 1)),
            "per_gt_best_iou_mean": float(np.mean(best_ious)) if best_ious else 0.0,
            "gt_iou_ge_025_count": int(np.count_nonzero(np.asarray(best_ious) >= 0.25)),
            "gt_iou_ge_050_count": int(np.count_nonzero(np.asarray(best_ious) >= 0.50)),
            "missed_gt_count": int(np.count_nonzero(np.asarray(best_ious) < 0.25)),
            "duplicate_predictions_per_matched_gt": float(np.mean([max(v - 1, 0) for v in dup_counts])) if dup_counts else 0.0,
        }
    )
    return row


def _aggregate_scene_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("ok")]
    out: dict[str, Any] = {"manifest_integrity_pass": None}
    numeric_keys = sorted(
        {
            key
            for row in ok
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        out[key] = float(mean(float(row[key]) for row in ok if row.get(key) is not None))
    return out


def _row(root: Path, method: str, config: str, support: str, scenes: list[str]) -> dict[str, Any]:
    metrics = _metric(root / "data" / "evaluation" / "scannet" / f"{config}_class_agnostic.txt")
    manifest, _ = load_prediction_manifest(root, config, "class_agnostic")
    scene_rows = [_scene_stats(root, config, scene) for scene in scenes]
    stats = _aggregate_scene_stats(scene_rows)
    if manifest is not None:
        stats["manifest_integrity_pass"] = bool(
            not manifest.get("uses_gt_for_prediction", manifest.get("uses_gt", False))
            and bool(manifest.get("eval_policy", manifest.get("pre_points_policy", "")))
        )
    return {
        "method": method,
        "config": config,
        "support": support,
        **metrics,
        **stats,
        "scene_rows": scene_rows,
    }


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["method"] for row in rows]
    for key, filename in (("ap", "gap_matrix_heatmap_AP.png"), ("ap50", "gap_matrix_heatmap_AP50.png")):
        values = np.asarray([[row.get(key) or 0.0] for row in rows], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6, max(4, len(rows) * 0.25)))
        ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xticks([0], [key])
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        path.with_suffix(".json").write_text(json.dumps(json_safe({"phase": "v18_phase0", "metric": key}), indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, [float(row.get("pre_points_ratio") or 0.0) for row in rows])
    ax.tick_params(axis="x", rotation=75)
    ax.set_ylabel("support ratio")
    fig.tight_layout()
    path = output_dir / "support_ratio_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(json_safe({"phase": "v18_phase0", "metric": "pre_points_ratio"}), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-prefix", default="outputs/audit/v18_phase0/unified_eval_matrix_probe5")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    scenes = read_seq_list(root / args.seq_list)
    rows = [_row(root, method, config, support, scenes) for method, config, support in ROWS]
    p0_by_support = {row["support"]: row for row in rows if row["method"].startswith("P0 on")}
    for row in rows:
        p0 = p0_by_support.get(row["support"])
        if p0 is not None and row.get("ap") is not None and p0.get("ap") is not None:
            row["same_support_gap_to_stream3d_AP"] = float((p0.get("ap") or 0.0) - (row.get("ap") or 0.0))
            row["same_support_gap_to_stream3d_AP50"] = float((p0.get("ap50") or 0.0) - (row.get("ap50") or 0.0))
            row["same_support_gap_to_stream3d_AP25"] = float((p0.get("ap25") or 0.0) - (row.get("ap25") or 0.0))
        else:
            row["same_support_gap_to_stream3d_AP"] = None
            row["same_support_gap_to_stream3d_AP50"] = None
            row["same_support_gap_to_stream3d_AP25"] = None
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "rows": rows}
    output.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "scene_rows"})
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v18 Phase 0 Unified Eval Matrix",
        "",
        "| method | AP | AP50 | AP25 | pre% | union% | GT crop/full | #pred | best IoU | gap AP/AP50/AP25 | manifest |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row.get("ap")),
                    str(row.get("ap50")),
                    str(row.get("ap25")),
                    f"{100.0 * float(row.get('pre_points_ratio') or 0.0):.4f}",
                    f"{100.0 * float(row.get('prediction_union_ratio') or 0.0):.4f}",
                    f"{row.get('gt_crop')}/{row.get('gt_full')}",
                    f"{float(row.get('num_pred') or 0.0):.2f}",
                    f"{float(row.get('per_gt_best_iou_mean') or 0.0):.6f}",
                    f"{row.get('same_support_gap_to_stream3d_AP')}/{row.get('same_support_gap_to_stream3d_AP50')}/{row.get('same_support_gap_to_stream3d_AP25')}",
                    str(row.get("manifest_integrity_pass")),
                ]
            )
            + " |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_plots(output.parent / "figures", rows)
    print(json.dumps(json_safe({"num_rows": len(rows), "output": str(output)}), indent=2))


if __name__ == "__main__":
    main()
