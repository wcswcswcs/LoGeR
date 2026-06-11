"""v14 GT-read-only failure decomposition diagnostic.

This tool reads GT to explain why candidate pools and final predictions fail.
It does not write prediction artifacts and must not be used as a method result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from evaluation.constants import SCANNET_IDS
from tools.prediction_manifest import load_prediction_manifest


MIN_REGION_SIZE = 100


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


def _prediction_dir(root: Path, config: str, suffix: str) -> Path:
    suffix_norm = suffix[1:] if suffix.startswith("_") else suffix
    if config.endswith(suffix_norm):
        return root / "data" / "prediction" / config
    return root / "data" / "prediction" / f"{config}_{suffix_norm}"


def _tmp_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene}_pre_points.npy"


def _metric_path(root: Path, config: str, suffix: str) -> Path:
    suffix_norm = suffix if suffix.startswith("_") else f"_{suffix}"
    name = config if config.endswith(suffix_norm) else f"{config}{suffix_norm}"
    return root / "data" / "evaluation" / "scannet" / f"{name}.txt"


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


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids % 1000 + int(SCANNET_IDS[0]) * 1000


def _gt_instances(gt_ids: np.ndarray) -> tuple[list[int], list[int], list[np.ndarray]]:
    ids: list[int] = []
    counts: list[int] = []
    masks: list[np.ndarray] = []
    valid = gt_ids[gt_ids >= 1000].astype(np.int64)
    for gt_id, count in zip(*np.unique(valid, return_counts=True)):
        if int(count) < MIN_REGION_SIZE:
            continue
        ids.append(int(gt_id))
        counts.append(int(count))
        masks.append(gt_ids == int(gt_id))
    return ids, counts, masks


def _load_prediction_full(
    root: Path,
    config: str,
    suffix: str,
    scene: str,
    scene_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = _prediction_dir(root, config, suffix) / f"{scene}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        masks = np.asarray(data["pred_masks"], dtype=bool)
        scores = np.asarray(data.get("pred_score", np.ones((masks.shape[1],), dtype=np.float32)), dtype=np.float64)
    if scores.shape[0] != masks.shape[1]:
        scores = np.ones((masks.shape[1],), dtype=np.float64)
    if masks.shape[0] == scene_vertices:
        return masks, scores
    pre_points = np.load(_tmp_path(root, config, scene)).astype(np.int64)
    if masks.shape[0] != pre_points.shape[0]:
        raise ValueError(
            f"{scene}: {config} pred dim {masks.shape[0]} is neither full scene "
            f"{scene_vertices} nor own pre_points {pre_points.shape[0]}"
        )
    full = np.zeros((scene_vertices, masks.shape[1]), dtype=bool)
    full[pre_points, :] = masks
    return full, scores


def _support_ratio(root: Path, config: str, scene: str, scene_vertices: int) -> float | None:
    path = _tmp_path(root, config, scene)
    if not path.exists():
        return None
    pre = np.load(path).astype(np.int64)
    return float(pre.shape[0] / max(scene_vertices, 1))


def _best_and_counts(gt_mask: np.ndarray, pred_masks: np.ndarray) -> dict[str, Any]:
    if pred_masks.shape[1] == 0:
        return {
            "best_iou": 0.0,
            "best_index": -1,
            "count_ge_0p10": 0,
            "count_ge_0p25": 0,
            "count_ge_0p50": 0,
        }
    pred_areas = pred_masks.sum(axis=0).astype(np.float64)
    gt_area = float(np.count_nonzero(gt_mask))
    inter = pred_masks[gt_mask, :].sum(axis=0).astype(np.float64)
    union = gt_area + pred_areas - inter
    iou = np.zeros((pred_masks.shape[1],), dtype=np.float64)
    valid = union > 0.0
    iou[valid] = inter[valid] / union[valid]
    best_index = int(np.argmax(iou)) if iou.size else -1
    return {
        "best_iou": float(iou[best_index]) if best_index >= 0 else 0.0,
        "best_index": int(best_index),
        "count_ge_0p10": int(np.count_nonzero(iou >= 0.10)),
        "count_ge_0p25": int(np.count_nonzero(iou >= 0.25)),
        "count_ge_0p50": int(np.count_nonzero(iou >= 0.50)),
        "ious": iou,
    }


def _rank_of_index(scores: np.ndarray, index: int) -> int:
    if index < 0 or scores.size == 0:
        return -1
    order = sorted(range(scores.shape[0]), key=lambda idx: (-float(scores[idx]), int(idx)))
    ranks = {int(idx): rank + 1 for rank, idx in enumerate(order)}
    return int(ranks.get(int(index), -1))


def _candidate_class(best_iou: float) -> str:
    if best_iou < 0.10:
        return "no_candidate"
    if best_iou < 0.25:
        return "weak_candidate"
    if best_iou < 0.50:
        return "good_candidate"
    return "high_candidate"


def _failure_class(pool_iou: float, method_iou: float, method_counts: dict[str, Any]) -> str:
    if method_iou >= 0.50:
        return "selected_good"
    if pool_iou >= 0.50:
        if method_iou >= 0.10 or int(method_counts.get("count_ge_0p25", 0)) > 1:
            return "assignment_error"
        return "filtered_good"
    if pool_iou >= 0.25:
        return "boundary_error"
    if pool_iou >= 0.10:
        return "weak_candidate"
    return "no_candidate"


def _parse_source_specs(specs: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("--source entries must be label:config or label:config:oracle_config")
        label, config = parts[:2]
        out.append({"label": label, "config": config, "oracle_config": parts[2] if len(parts) == 3 else ""})
    return out


def _parse_method_specs(specs: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("--method entries must be label:config")
        out.append({"label": parts[0], "config": parts[1]})
    return out


def _source_metrics(root: Path, suffix: str, source: dict[str, str]) -> dict[str, Any]:
    config = source["config"]
    manifest, manifest_path = load_prediction_manifest(root, config, suffix.lstrip("_"))
    metrics = _parse_metric_file(_metric_path(root, config, suffix))
    oracle_metrics = (
        _parse_metric_file(_metric_path(root, source["oracle_config"], suffix))
        if source.get("oracle_config")
        else {"ap": None, "ap50": None, "ap25": None}
    )
    return {
        "label": source["label"],
        "config": config,
        "oracle_config": source.get("oracle_config", ""),
        "ap": metrics["ap"],
        "ap50": metrics["ap50"],
        "ap25": metrics["ap25"],
        "oracle_ap": oracle_metrics["ap"],
        "oracle_ap50": oracle_metrics["ap50"],
        "oracle_ap25": oracle_metrics["ap25"],
        "manifest_path": str(manifest_path) if manifest_path else "",
        "uses_gt_for_prediction": bool(manifest.get("uses_gt_for_prediction", False)) if manifest else None,
        "is_method_result": bool(manifest.get("is_method_result", False)) if manifest else None,
    }


def _process_scene(
    root: Path,
    suffix: str,
    scene: str,
    sources: list[dict[str, str]],
    methods: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
    gt_ids = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    scene_vertices = int(gt_ids.shape[0])
    gt_ids_list, gt_counts, gt_masks = _gt_instances(gt_ids)

    source_masks: dict[str, np.ndarray] = {}
    source_scores: dict[str, np.ndarray] = {}
    source_offsets: dict[str, tuple[int, int]] = {}
    pool_parts: list[np.ndarray] = []
    cursor = 0
    for source in sources:
        masks, scores = _load_prediction_full(root, source["config"], suffix, scene, scene_vertices)
        source_masks[source["label"]] = masks
        source_scores[source["label"]] = scores
        source_offsets[source["label"]] = (cursor, cursor + masks.shape[1])
        pool_parts.append(masks)
        cursor += masks.shape[1]
    pool = np.concatenate(pool_parts, axis=1) if pool_parts else np.zeros((scene_vertices, 0), dtype=bool)

    method_masks: dict[str, np.ndarray] = {}
    method_scores: dict[str, np.ndarray] = {}
    for method in methods:
        masks, scores = _load_prediction_full(root, method["config"], suffix, scene, scene_vertices)
        method_masks[method["label"]] = masks
        method_scores[method["label"]] = scores

    source_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for gt_id, gt_count, gt_mask in zip(gt_ids_list, gt_counts, gt_masks):
        per_source: dict[str, dict[str, Any]] = {}
        for source in sources:
            label = source["label"]
            stats = _best_and_counts(gt_mask, source_masks[label])
            per_source[label] = stats
            source_rows.append(
                {
                    "scene": scene,
                    "gt_id": int(gt_id),
                    "gt_vertices": int(gt_count),
                    "source_label": label,
                    "source_config": source["config"],
                    "best_iou": float(stats["best_iou"]),
                    "best_index": int(stats["best_index"]),
                    "candidate_class": _candidate_class(float(stats["best_iou"])),
                    "candidate_count_ge_0p10": int(stats["count_ge_0p10"]),
                    "candidate_count_ge_0p25": int(stats["count_ge_0p25"]),
                    "candidate_count_ge_0p50": int(stats["count_ge_0p50"]),
                }
            )

        pool_stats = _best_and_counts(gt_mask, pool)
        best_source = ""
        best_source_local_index = -1
        pool_best_index = int(pool_stats["best_index"])
        for label, (start, end) in source_offsets.items():
            if start <= pool_best_index < end:
                best_source = label
                best_source_local_index = int(pool_best_index - start)
                break

        for method in methods:
            label = method["label"]
            stats = _best_and_counts(gt_mask, method_masks[label])
            method_iou = float(stats["best_iou"])
            pool_iou = float(pool_stats["best_iou"])
            method_rows.append(
                {
                    "scene": scene,
                    "gt_id": int(gt_id),
                    "gt_vertices": int(gt_count),
                    "method_label": label,
                    "method_config": method["config"],
                    "pool_best_iou": pool_iou,
                    "pool_best_source": best_source,
                    "pool_best_index": pool_best_index,
                    "pool_best_source_local_index": best_source_local_index,
                    "pool_candidate_class": _candidate_class(pool_iou),
                    "pool_candidate_count_ge_0p10": int(pool_stats["count_ge_0p10"]),
                    "pool_candidate_count_ge_0p25": int(pool_stats["count_ge_0p25"]),
                    "pool_candidate_count_ge_0p50": int(pool_stats["count_ge_0p50"]),
                    "method_best_iou": method_iou,
                    "method_best_index": int(stats["best_index"]),
                    "method_best_rank": _rank_of_index(method_scores[label], int(stats["best_index"])),
                    "method_duplicate_predictions_ge_0p25": max(0, int(stats["count_ge_0p25"]) - 1),
                    "failure_class": _failure_class(pool_iou, method_iou, stats),
                }
            )

    scene_summary = {
        "scene": scene,
        "num_gt_instances": int(len(gt_ids_list)),
        "num_scene_vertices": scene_vertices,
        "source_support_ratio": {
            source["label"]: _support_ratio(root, source["config"], scene, scene_vertices) for source in sources
        },
        "method_support_ratio": {
            method["label"]: _support_ratio(root, method["config"], scene, scene_vertices) for method in methods
        },
    }
    return source_rows, method_rows, scene_summary


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(mean(valid)) if valid else None


def _aggregate(
    *,
    root: Path,
    suffix: str,
    sources: list[dict[str, str]],
    methods: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    scene_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_summary: list[dict[str, Any]] = []
    for source in sources:
        rows = [row for row in source_rows if row["source_label"] == source["label"]]
        counts = Counter(str(row["candidate_class"]) for row in rows)
        metrics = _source_metrics(root, suffix, source)
        source_summary.append(
            {
                **metrics,
                "num_gt_rows": int(len(rows)),
                "mean_best_iou": float(mean([float(row["best_iou"]) for row in rows])) if rows else 0.0,
                "candidate_count_mean_ge_0p10": float(mean([int(row["candidate_count_ge_0p10"]) for row in rows])) if rows else 0.0,
                "candidate_count_mean_ge_0p25": float(mean([int(row["candidate_count_ge_0p25"]) for row in rows])) if rows else 0.0,
                "candidate_count_mean_ge_0p50": float(mean([int(row["candidate_count_ge_0p50"]) for row in rows])) if rows else 0.0,
                "no_candidate": int(counts.get("no_candidate", 0)),
                "weak_candidate": int(counts.get("weak_candidate", 0)),
                "good_candidate": int(counts.get("good_candidate", 0)),
                "high_candidate": int(counts.get("high_candidate", 0)),
                "support_pre_ratio_mean": _mean_or_none(
                    [
                        scene["source_support_ratio"].get(source["label"])
                        for scene in scene_summaries
                    ]
                ),
            }
        )

    method_summary: list[dict[str, Any]] = []
    for method in methods:
        rows = [row for row in method_rows if row["method_label"] == method["label"]]
        counts = Counter(str(row["failure_class"]) for row in rows)
        method_summary.append(
            {
                "label": method["label"],
                "config": method["config"],
                "num_gt_rows": int(len(rows)),
                "mean_pool_best_iou": float(mean([float(row["pool_best_iou"]) for row in rows])) if rows else 0.0,
                "mean_method_best_iou": float(mean([float(row["method_best_iou"]) for row in rows])) if rows else 0.0,
                "selected_good": int(counts.get("selected_good", 0)),
                "filtered_good": int(counts.get("filtered_good", 0)),
                "assignment_error": int(counts.get("assignment_error", 0)),
                "boundary_error": int(counts.get("boundary_error", 0)),
                "weak_candidate": int(counts.get("weak_candidate", 0)),
                "no_candidate": int(counts.get("no_candidate", 0)),
                "mean_duplicate_predictions_ge_0p25": float(
                    mean([int(row["method_duplicate_predictions_ge_0p25"]) for row in rows])
                )
                if rows
                else 0.0,
                "support_pre_ratio_mean": _mean_or_none(
                    [
                        scene["method_support_ratio"].get(method["label"])
                        for scene in scene_summaries
                    ]
                ),
            }
        )

    source_vote = Counter(str(row["pool_best_source"]) for row in method_rows if row["pool_best_source"])
    return {
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "num_scenes": int(len(scene_summaries)),
        "num_source_gt_rows": int(len(source_rows)),
        "num_method_gt_rows": int(len(method_rows)),
        "source_summary": source_summary,
        "method_summary": method_summary,
        "pool_best_source_counts": dict(source_vote),
        "scene_summaries": scene_summaries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _fmt(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value) * scale:.{digits}f}"
    except Exception:
        return str(value)


def _write_markdown(prefix: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Stream4D v14 Failure Decomposition",
        "",
        "GT is read only for diagnostic attribution. This output is diagnostic-only and must not enter the method table.",
        "",
        "## Candidate Sources",
        "",
        "| source | AP/AP50/AP25 | oracle AP/AP50/AP25 | support % | best IoU | no/weak/good/high | cand>=.25/.50 | method result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["source_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    f"{_fmt(row.get('ap'), 100.0)}/{_fmt(row.get('ap50'), 100.0)}/{_fmt(row.get('ap25'), 100.0)}",
                    f"{_fmt(row.get('oracle_ap'), 100.0)}/{_fmt(row.get('oracle_ap50'), 100.0)}/{_fmt(row.get('oracle_ap25'), 100.0)}",
                    _fmt(row.get("support_pre_ratio_mean"), 100.0),
                    _fmt(row.get("mean_best_iou")),
                    f"{row.get('no_candidate')}/{row.get('weak_candidate')}/{row.get('good_candidate')}/{row.get('high_candidate')}",
                    f"{_fmt(row.get('candidate_count_mean_ge_0p25'))}/{_fmt(row.get('candidate_count_mean_ge_0p50'))}",
                    str(row.get("is_method_result")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Final Method Attribution",
            "",
            "| method | support % | pool IoU | method IoU | selected | filtered_good | assignment | boundary | weak | no_candidate | dup>=.25 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["method_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    _fmt(row.get("support_pre_ratio_mean"), 100.0),
                    _fmt(row.get("mean_pool_best_iou")),
                    _fmt(row.get("mean_method_best_iou")),
                    str(row.get("selected_good")),
                    str(row.get("filtered_good")),
                    str(row.get("assignment_error")),
                    str(row.get("boundary_error")),
                    str(row.get("weak_candidate")),
                    str(row.get("no_candidate")),
                    _fmt(row.get("mean_duplicate_predictions_ge_0p25")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pool Best Source Counts",
            "",
            "| source | count |",
            "|---|---:|",
        ]
    )
    for source, count in sorted(summary["pool_best_source_counts"].items()):
        lines.append(f"| {source} | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{prefix.with_suffix('.json')}`",
            f"- source GT CSV: `{prefix.parent / (prefix.name + '_source_gt.csv')}`",
            f"- method GT CSV: `{prefix.parent / (prefix.name + '_method_gt.csv')}`",
            f"- visualization manifest: `{prefix.parent / 'failure_visuals_manifest.json'}`",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_visuals(out_dir: Path, payload: dict[str, Any], limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for failure visuals: {exc}") from exc

    manifest: list[dict[str, Any]] = []
    summary = payload["summary"]
    method_rows = payload["method_rows"]
    source_rows = payload["source_rows"]

    def save_panel(name: str, title: str, labels: list[str], values: list[float], ylabel: str, sidecar: dict[str, Any]) -> None:
        idx = len(manifest)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"][: len(labels)])
        ax.set_title(title[:100])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        ymax = max(values) * 1.15 if values else 1.0
        ax.set_ylim(0, max(1.0, ymax))
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        png_path = out_dir / f"v14_failure_panel_{idx:02d}_{name}.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        sidecar_path = png_path.with_suffix(".json")
        sidecar_payload = {"png": str(png_path), "visual_type": name, **sidecar}
        sidecar_path.write_text(json.dumps(_json_safe(sidecar_payload), indent=2, sort_keys=True), encoding="utf-8")
        manifest.append({"png": str(png_path), "sidecar": str(sidecar_path), **sidecar})

    for row in summary["source_summary"]:
        labels = ["no", "weak", "good", "high"]
        values = [float(row.get(key, 0)) for key in ["no_candidate", "weak_candidate", "good_candidate", "high_candidate"]]
        save_panel(
            "source_candidate_counts",
            f"{row['label']} candidate class counts",
            labels,
            values,
            "GT count",
            {"source_label": row["label"], "source_config": row["config"], "diagnostic_only": True},
        )
        if len(manifest) >= limit:
            break

    if len(manifest) < limit:
        for row in summary["method_summary"]:
            labels = ["selected", "filtered", "assign", "boundary", "weak", "none"]
            values = [
                float(row.get("selected_good", 0)),
                float(row.get("filtered_good", 0)),
                float(row.get("assignment_error", 0)),
                float(row.get("boundary_error", 0)),
                float(row.get("weak_candidate", 0)),
                float(row.get("no_candidate", 0)),
            ]
            save_panel(
                "method_failure_counts",
                f"{row['label']} failure categories",
                labels,
                values,
                "GT count",
                {"method_label": row["label"], "method_config": row["config"], "diagnostic_only": True},
            )
            if len(manifest) >= limit:
                break

    if len(manifest) < limit:
        by_scene = defaultdict(Counter)
        for row in method_rows:
            by_scene[(str(row["scene"]), str(row["method_label"]))][str(row["failure_class"])] += 1
        for (scene, method_label), counts in sorted(by_scene.items()):
            labels = ["selected_good", "filtered_good", "assignment_error", "boundary_error", "weak_candidate", "no_candidate"]
            values = [float(counts.get(label, 0)) for label in labels]
            save_panel(
                "scene_method_failure_counts",
                f"{scene} {method_label}",
                [label.replace("_", "\n") for label in labels],
                values,
                "GT count",
                {"scene": scene, "method_label": method_label, "diagnostic_only": True},
            )
            if len(manifest) >= limit:
                break

    if len(manifest) < limit:
        labels = [row["label"] for row in summary["source_summary"]]
        values = [float(row.get("mean_best_iou", 0.0)) for row in summary["source_summary"]]
        save_panel(
            "source_mean_best_iou",
            "Mean per-GT best IoU by source",
            labels,
            values,
            "IoU",
            {"diagnostic_only": True},
        )

    if len(manifest) < limit:
        best_source_counts = summary["pool_best_source_counts"]
        save_panel(
            "pool_best_source_counts",
            "Best candidate source over pooled candidates",
            list(best_source_counts.keys()),
            [float(v) for v in best_source_counts.values()],
            "GT rows",
            {"diagnostic_only": True},
        )

    (out_dir / "failure_visuals_manifest.json").write_text(
        json.dumps(_json_safe({"num_visuals": len(manifest), "visuals": manifest}), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--source", action="append", required=True, help="label:config or label:config:oracle_config")
    parser.add_argument("--method", action="append", required=True, help="label:config")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--visual-dir", required=True)
    parser.add_argument("--visual-limit", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    seq_list = root / args.seq_list
    sources = _parse_source_specs(args.source)
    methods = _parse_method_specs(args.method)

    source_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in _read_seq_list(seq_list):
        scene_source_rows, scene_method_rows, scene_summary = _process_scene(root, args.pred_suffix, scene, sources, methods)
        source_rows.extend(scene_source_rows)
        method_rows.extend(scene_method_rows)
        scene_summaries.append(scene_summary)

    summary = _aggregate(
        root=root,
        suffix=args.pred_suffix,
        sources=sources,
        methods=methods,
        source_rows=source_rows,
        method_rows=method_rows,
        scene_summaries=scene_summaries,
    )
    payload = {
        "args": vars(args),
        "summary": summary,
        "source_rows": source_rows,
        "method_rows": method_rows,
    }

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(prefix.parent / f"{prefix.name}_source_gt.csv", source_rows)
    _write_csv(prefix.parent / f"{prefix.name}_method_gt.csv", method_rows)
    _write_markdown(prefix, payload)
    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = root / visual_dir
    _write_visuals(visual_dir, payload, limit=int(args.visual_limit))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(f"[v14-failure] wrote {prefix.with_suffix('.json')}")
    print(f"[v14-failure] wrote {prefix.with_suffix('.md')}")
    print(f"[v14-failure] wrote {visual_dir / 'failure_visuals_manifest.json'}")


if __name__ == "__main__":
    main()
