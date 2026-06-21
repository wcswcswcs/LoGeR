from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, load_mask_label, parse_bool, parse_int, read_csv, utc_now, write_csv, write_json
from .v53_mask_component_support import _build_components, _carrier_global_id, _collect_support, _is_visible_row


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _chunk_windows(frames: list[int], chunk_size: int, chunk_stride: int) -> list[tuple[int, int, list[int]]]:
    if not frames:
        return []
    windows: list[tuple[int, int, list[int]]] = []
    start_rank = 0
    while start_rank < len(frames):
        end_rank = min(start_rank + int(chunk_size), len(frames))
        selected = frames[start_rank:end_rank]
        if selected:
            windows.append((start_rank, end_rank - 1, selected))
        if end_rank >= len(frames):
            break
        start_rank += int(chunk_stride)
    return windows


def build_chunk_universe(
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    max_union_unique_carriers: int = 32,
    min_visibility_prob: float = 0.5,
    min_confidence: float = 0.5,
    chunk_size: int = 32,
    chunk_stride: int = 16,
) -> dict[str, Any]:
    carrier_rows = read_csv(_project(carrier_table_path))
    mask_rows = read_csv(_project(mask_table_path))
    component_payload = _build_components(
        carrier_rows=carrier_rows,
        mask_rows=mask_rows,
        max_union_unique_carriers=max_union_unique_carriers,
        min_visibility_prob=min_visibility_prob,
        min_confidence=min_confidence,
    )
    support_payload = _collect_support(
        visible_rows=component_payload["visible_rows"],
        mask_rows=mask_rows,
        component_by_carrier=component_payload["component_by_carrier"],
    )
    component_by_carrier: dict[str, str] = component_payload["component_by_carrier"]
    support_by_mask: dict[str, Counter[str]] = support_payload["support_by_mask"]

    frames_by_scene: dict[str, set[int]] = defaultdict(set)
    carrier_by_scene_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    scale_weak_by_scene_frame: dict[tuple[str, int], bool] = defaultdict(bool)
    for row in carrier_rows:
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        frames_by_scene[scene].add(frame_id)
        if _is_visible_row(row, min_visibility_prob=min_visibility_prob, min_confidence=min_confidence):
            carrier_by_scene_frame[(scene, frame_id)].append(row)
        if not parse_bool(row.get("scale_guard_pass", True)) or not parse_bool(row.get("allow_metric_relation", True)):
            scale_weak_by_scene_frame[(scene, frame_id)] = True

    masks_by_scene_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        scene = str(row.get("scene"))
        frame_id = parse_int(row.get("frame_id"))
        frames_by_scene[scene].add(frame_id)
        masks_by_scene_frame[(scene, frame_id)].append(row)

    chunk_rows: list[dict[str, Any]] = []
    chunk_component_rows: list[dict[str, Any]] = []
    chunk_mask_rows: list[dict[str, Any]] = []
    visibility_rasters: dict[str, dict[str, Any]] = {}
    weak_scale_chunk_count = 0

    for scene in sorted(frames_by_scene):
        frames = sorted(frames_by_scene[scene])
        scene_raster_components: set[str] = set()
        scene_raster_values: dict[tuple[str, int], int] = Counter()
        for chunk_index, (start_rank, end_rank, chunk_frames) in enumerate(_chunk_windows(frames, chunk_size, chunk_stride)):
            chunk_id = f"{scene}:chunk{chunk_index:03d}"
            frame_set = set(chunk_frames)
            component_frame_counts: dict[str, set[int]] = defaultdict(set)
            component_visible_obs: Counter[str] = Counter()
            component_mask_support: Counter[str] = Counter()
            mask_count = 0
            supported_mask_count = 0
            chunk_weak_scale = False

            for frame_id in chunk_frames:
                chunk_weak_scale = chunk_weak_scale or bool(scale_weak_by_scene_frame[(scene, frame_id)])
                for row in carrier_by_scene_frame.get((scene, frame_id), []):
                    component_id = component_by_carrier.get(_carrier_global_id(row))
                    if not component_id:
                        continue
                    component_frame_counts[component_id].add(frame_id)
                    component_visible_obs[component_id] += 1
                    scene_raster_components.add(component_id)
                    scene_raster_values[(component_id, frame_id)] += 1
                for mask_row in masks_by_scene_frame.get((scene, frame_id), []):
                    mask_count += 1
                    mask_observation_id = str(mask_row.get("mask_observation_id"))
                    support_counter = support_by_mask.get(mask_observation_id, Counter())
                    if support_counter:
                        supported_mask_count += 1
                    for component_id, count in support_counter.items():
                        component_mask_support[component_id] += int(count)
                    chunk_mask_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "scene": scene,
                            "raw_frame_id": frame_id,
                            "observation_rank": frames.index(frame_id),
                            "mask_observation_id": mask_observation_id,
                            "mask_id": mask_row.get("mask_id"),
                            "raw_supported_component_count": int(len(support_counter)),
                            "raw_support_carrier_count": int(sum(support_counter.values())),
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )

            component_ids = sorted(component_frame_counts)
            supported_components = set(component_mask_support)
            visibility_counts = [len(component_frame_counts[component_id]) for component_id in component_ids]
            if chunk_weak_scale:
                weak_scale_chunk_count += 1
            for component_id in component_ids:
                frame_count = len(component_frame_counts[component_id])
                support_count = int(component_mask_support.get(component_id, 0))
                chunk_component_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "scene": scene,
                        "component_id": component_id,
                        "visible_frame_count": int(frame_count),
                        "visible_observation_count": int(component_visible_obs[component_id]),
                        "mask_support_carrier_count": support_count,
                        "has_mask_support": bool(support_count > 0),
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "scene": scene,
                    "chunk_index": int(chunk_index),
                    "raw_frame_start": int(chunk_frames[0]),
                    "raw_frame_end": int(chunk_frames[-1]),
                    "observation_rank_start": int(start_rank),
                    "observation_rank_end": int(end_rank),
                    "frame_count": int(len(chunk_frames)),
                    "component_count": int(len(component_ids)),
                    "mask_count": int(mask_count),
                    "component_visibility_frame_count_mean": _mean([float(value) for value in visibility_counts]),
                    "component_visibility_frame_count_p10": _quantile([float(value) for value in visibility_counts], 0.10),
                    "chunk_component_coverage": float(len(supported_components) / max(len(component_ids), 1)),
                    "chunk_mask_coverage": float(supported_mask_count / max(mask_count, 1)),
                    "weak_scale_chunk": bool(chunk_weak_scale),
                    "allow_metric_relation": bool(not chunk_weak_scale),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
        ordered_components = sorted(scene_raster_components)[:200]
        visibility_rasters[scene] = {
            "frames": frames,
            "components": ordered_components,
            "values": [[int(scene_raster_values[(component_id, frame_id)]) for frame_id in frames] for component_id in ordered_components],
        }

    components_per_chunk = [float(row["component_count"]) for row in chunk_rows]
    masks_per_chunk = [float(row["mask_count"]) for row in chunk_rows]
    visibility_means = [
        float(row["component_visibility_frame_count_mean"])
        for row in chunk_rows
        if row["component_visibility_frame_count_mean"] is not None
    ]
    visibility_p10s = [
        float(row["component_visibility_frame_count_p10"])
        for row in chunk_rows
        if row["component_visibility_frame_count_p10"] is not None
    ]
    component_coverages = [float(row["chunk_component_coverage"]) for row in chunk_rows]
    mask_coverages = [float(row["chunk_mask_coverage"]) for row in chunk_rows]
    summary = {
        "phase": "v53_chunk_universe",
        "created_at": utc_now(),
        "carrier_table_path": str(carrier_table_path),
        "mask_table_path": str(mask_table_path),
        "max_union_unique_carriers": int(max_union_unique_carriers),
        "min_visibility_prob": float(min_visibility_prob),
        "min_confidence": float(min_confidence),
        "chunk_size": int(chunk_size),
        "chunk_stride": int(chunk_stride),
        "chunk_count": len(chunk_rows),
        "components_per_chunk_mean": _mean(components_per_chunk),
        "components_per_chunk_p50": _quantile(components_per_chunk, 0.50),
        "components_per_chunk_p90": _quantile(components_per_chunk, 0.90),
        "masks_per_chunk_mean": _mean(masks_per_chunk),
        "component_visibility_frame_count_mean": _mean(visibility_means),
        "component_visibility_frame_count_p10": _mean(visibility_p10s),
        "chunk_component_coverage": _mean(component_coverages),
        "chunk_mask_coverage": _mean(mask_coverages),
        "weak_scale_chunk_count": int(weak_scale_chunk_count),
        "weak_scale_chunk_count_recorded": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "components_per_chunk_mean_ge_20": float(summary["components_per_chunk_mean"] or 0.0) >= 20.0,
        "chunk_component_coverage_ge_0.80": float(summary["chunk_component_coverage"] or 0.0) >= 0.80,
        "component_visibility_frame_count_mean_ge_2": float(summary["component_visibility_frame_count_mean"] or 0.0) >= 2.0,
        "weak_scale_chunk_count_recorded": True,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {
        "summary": summary,
        "chunk_rows": chunk_rows,
        "chunk_component_rows": chunk_component_rows,
        "chunk_mask_rows": chunk_mask_rows,
        "visibility_rasters": visibility_rasters,
    }


def _write_visualizations(output_root: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(output_root / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(output_root / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    chunks_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["chunk_rows"]:
        chunks_by_scene[str(row["scene"])].append(row)

    for scene, rows in sorted(chunks_by_scene.items()):
        frames = sorted({frame for row in rows for frame in range(int(row["raw_frame_start"]), int(row["raw_frame_end"]) + 1)})
        fig, ax = plt.subplots(figsize=(10, max(2.5, len(rows) * 0.6)))
        for y, row in enumerate(rows):
            ax.barh(y, int(row["raw_frame_end"]) - int(row["raw_frame_start"]) + 1, left=int(row["raw_frame_start"]), height=0.4)
            ax.text(int(row["raw_frame_start"]), y + 0.25, str(row["chunk_id"]).split(":")[-1], fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([str(row["chunk_id"]).split(":")[-1] for row in rows])
        ax.set_xlabel("raw_frame_id")
        ax.set_title(f"{scene} chunk timeline")
        path = vis_root / f"chunk_timeline_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "chunk_timeline", "scene": scene})

        raster = payload["visibility_rasters"].get(scene, {"values": [], "frames": [], "components": []})
        values = np.asarray(raster["values"], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(10, max(3, min(10, values.shape[0] * 0.04))))
        if values.size:
            ax.imshow(values > 0, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xlabel("observation frame rank")
        ax.set_ylabel("components (first 200)")
        ax.set_title(f"{scene} component visibility raster")
        path = vis_root / f"component_visibility_raster_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "component_visibility_raster", "scene": scene})

        labels = [str(row["chunk_id"]).split(":")[-1] for row in rows]
        component_counts = [float(row["component_count"]) for row in rows]
        mask_counts = [float(row["mask_count"]) for row in rows]
        weak = [1.0 if row["weak_scale_chunk"] else 0.0 for row in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 4))
        ax.bar(x - 0.2, component_counts, width=0.4, label="components")
        ax.bar(x + 0.2, mask_counts, width=0.4, label="masks")
        if any(weak):
            ax.scatter(x, [max(component_counts + mask_counts + [1.0]) for _ in x], c=weak, cmap="Reds", label="weak scale")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{scene} chunk geometry summary")
        ax.legend()
        path = vis_root / f"chunk_geometry_summary_panel_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "chunk_geometry_summary_panel", "scene": scene})
    return manifest


def write_chunk_universe(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v53_visualizations/local_objectlets",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "chunk_summary.json", payload["summary"])
    write_csv(out / "chunk_component_rows.csv", payload["chunk_component_rows"])
    write_csv(out / "chunk_mask_rows.csv", payload["chunk_mask_rows"])
    write_csv(out / "chunk_rows.csv", payload["chunk_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v53_chunk_universe", "files": manifest})


__all__ = ["build_chunk_universe", "write_chunk_universe"]
