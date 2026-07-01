from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import (  # noqa: E402
    SparseSceneIoU,
    _load_gt_2d,
    _read_label_png,
    _run_one,
    _sha256,
    _summarize_iou,
    _top_iou_rows,
    _write_csv,
    _write_json,
)


DEFAULT_SCENES = ["scene0011_00", "scene0030_00", "scene0050_00", "scene0081_01", "scene0591_00"]


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _numeric(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if value_f.is_integer():
        return int(value_f)
    return value_f


def _discover_pipeline_root(scene: str) -> Path | None:
    candidates = [
        ROOT / "outputs/audit" / f"v66_soma_fullscene_pipeline_{scene}_stride5_conf02_integrated_d4rt",
        ROOT / "outputs/audit" / f"v65_soma_fullscene_pipeline_{scene}_stride5_conf02_integrated_d4rt",
    ]
    if scene == "scene0050_00":
        candidates.append(ROOT / "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt")
    for root in candidates:
        if (root / "pipeline_summary.json").exists():
            try:
                summary = _read_json(root / "pipeline_summary.json")
            except Exception:
                continue
            if str(summary.get("scene")) == scene and bool((summary.get("pipeline_gate") or {}).get("ap_ready", False)):
                return root
    return None


def _mask_dir_from_pipeline(pipeline_root: Path) -> Path:
    summary = _read_json(pipeline_root / "pipeline_summary.json")
    coverage = summary.get("mask_frame_coverage") or {}
    mask_dir = coverage.get("mask_dir")
    if not mask_dir:
        raise ValueError(f"pipeline summary lacks mask_frame_coverage.mask_dir: {pipeline_root}")
    return _project(mask_dir)


def _frame_mask_duplicate_stats(objectlet_rows: Path, ledger_rows: Path, scene: str, variant: str) -> dict[str, Any]:
    selected: dict[str, str] = {}
    with objectlet_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") == scene and row.get("variant") == variant:
                selected[str(row.get("candidate_id") or "")] = str(row.get("objectlet_id") or "")
    owners: dict[tuple[int, int], set[str]] = {}
    with ledger_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("reprojection_success")) != "True":
                continue
            object_id = selected.get(str(row.get("candidate_id") or ""))
            if not object_id:
                continue
            parts = str(row.get("best_mask_observation_id") or "").split(":")
            if len(parts) != 3 or parts[0] != scene:
                continue
            key = (int(parts[1]), int(parts[2]))
            owners.setdefault(key, set()).add(object_id)
    duplicate = {key: vals for key, vals in owners.items() if len(vals) > 1}
    return {
        "support_pair_count": int(len(owners)),
        "duplicate_frame_mask_conflicts": int(len(duplicate)),
        "duplicate_frame_mask_conflict_rate": float(len(duplicate) / max(1, len(owners))),
        "max_objects_per_frame_mask": max((len(vals) for vals in owners.values()), default=0),
    }


def _best_variant(pipeline_root: Path) -> str:
    summary = _read_json(pipeline_root / "local_objectlets/local_objectlet_summary.json")
    return str(summary.get("best_real_variant") or summary.get("best_real_row", {}).get("variant") or "best")


def _score_free(summary: dict[str, Any], threshold: str) -> dict[str, Any]:
    return summary.get(f"score_free_match_at_{threshold}", {})


def _row_from_summary(
    *,
    row_id: str,
    method: str,
    scene: str,
    stride: int,
    score_mode: str,
    summary: dict[str, Any],
    source: str,
    geometry_dependency: str,
    diagnostic_only: bool,
    uses_gt_for_prediction: bool,
    forbidden_for_method_table: bool,
    support_stats: dict[str, Any] | None = None,
    output_summary: str = "",
) -> dict[str, Any]:
    sf25 = _score_free(summary, "025")
    sf50 = _score_free(summary, "050")
    row = {
        "row_id": row_id,
        "scene_id": scene,
        "method": method,
        "stride": int(stride),
        "frame_policy": f"stride{int(stride)}",
        "frame_count": summary.get("frame_count"),
        "AP": summary.get("ap"),
        "AP50": summary.get("ap50"),
        "AP25": summary.get("ap25"),
        "score_free_match25_TP": sf25.get("tp"),
        "score_free_match50_TP": sf50.get("tp"),
        "score_free_match25_recall": sf25.get("recall"),
        "score_free_match50_recall": sf50.get("recall"),
        "gt_best_iou_mean": summary.get("gt_best_iou_mean"),
        "gt_best_iou_median": summary.get("gt_best_iou_median"),
        "pred_best_iou_mean": summary.get("pred_best_iou_mean"),
        "pred_best_iou_median": summary.get("pred_best_iou_median"),
        "pred_count": summary.get("evaluated_pred_count"),
        "gt_count": summary.get("evaluated_gt_count"),
        "support_frame_count": summary.get("frame_count"),
        "duplicate_frame_mask_conflicts": "",
        "duplicate_frame_mask_conflict_rate": "",
        "missing_mask_frame_count": "",
        "diagnostic_only": diagnostic_only,
        "metric_name": "scene_level_multi_view_2d_AP",
        "geometry_dependency": geometry_dependency,
        "score_mode": score_mode,
        "matching_scope": "global_scene_not_per_view",
        "pixel_grid": "ScanNet depth resolution",
        "uses_gt_for_prediction": uses_gt_for_prediction,
        "forbidden_for_method_table": forbidden_for_method_table,
        "source": source,
        "output_summary": output_summary,
    }
    if support_stats:
        row.update({key: support_stats.get(key, row.get(key, "")) for key in support_stats})
    return row


def _load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    return _read_label_png(path, shape_hw)


def _raw_cropformer_summary(*, scene: str, stride: int, mask_dir: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    frame_ids = stream.frame_ids(stride=stride, max_frames=None)
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    missing = 0
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, frame_id, shape_hw)
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        raw_mask_pixels = 0
        if mask is None:
            missing += 1
        else:
            raw_mask_pixels = int(np.count_nonzero(mask > 0))
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                pred[mask == mask_id] = int(frame_id) * 100000 + mask_id
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene": scene,
                "frame_id": int(frame_id),
                "raw_mask_pixels": raw_mask_pixels,
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
            }
        )
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    summary["frame_count"] = len(frame_ids)
    summary["missing_mask_frame_count"] = int(missing)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = _top_iou_rows(iou, pred_ids, gt_ids, top_k=50)
    _write_csv(output_dir / "frame_rows.csv", frame_rows)
    _write_csv(output_dir / "top_iou_pairs.csv", top_rows)
    _write_json(output_dir / "summary.json", {"summary": summary, "top_iou_pairs": top_rows})
    return summary, top_rows


def _oracle_selected_mask_map(pipeline_root: Path, scene: str, variant: str) -> dict[tuple[int, int], int]:
    selected: set[str] = set()
    with (pipeline_root / "local_objectlets/objectlet_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") == scene and row.get("variant") == variant:
                selected.add(str(row.get("candidate_id") or ""))
    out: dict[tuple[int, int], int] = {}
    with (pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("candidate_id") or "") not in selected or str(row.get("reprojection_success")) != "True":
                continue
            parts = str(row.get("best_mask_observation_id") or "").split(":")
            if len(parts) != 3 or parts[0] != scene:
                continue
            gt_id = int(float(row.get("diagnostic_best_gt") or 0))
            if gt_id > 0:
                out[(int(parts[1]), int(parts[2]))] = gt_id
    return out


def _mapped_mask_summary(
    *,
    scene: str,
    stride: int,
    mask_dir: Path,
    mapping: dict[tuple[int, int], int],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    frame_ids = stream.frame_ids(stride=stride, max_frames=None)
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    missing = 0
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, frame_id, shape_hw)
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png", shape_hw)
        if mask is None:
            missing += 1
        else:
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                label = int(mapping.get((int(frame_id), mask_id), 0))
                if label > 0:
                    pred[mask == mask_id] = label
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene": scene,
                "frame_id": int(frame_id),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
            }
        )
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    summary["frame_count"] = len(frame_ids)
    summary["missing_mask_frame_count"] = int(missing)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = _top_iou_rows(iou, pred_ids, gt_ids, top_k=50)
    _write_csv(output_dir / "frame_rows.csv", frame_rows)
    _write_csv(output_dir / "top_iou_pairs.csv", top_rows)
    _write_json(output_dir / "summary.json", {"summary": summary, "top_iou_pairs": top_rows})
    return summary, top_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    strides = [int(value) for value in _parse_csv_list(args.strides)]
    score_modes = _parse_csv_list(args.score_modes)
    rows: list[dict[str, Any]] = []
    top_rows_all: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    dummy_pipeline = _discover_pipeline_root("scene0050_00") or output_root

    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is not None:
            pipeline_roots[scene] = _rel(pipeline_root)
        for stride in strides:
            for score_mode in score_modes:
                payload = _run_one(
                    scene=scene,
                    method="stream3d",
                    stride=stride,
                    output_root=output_root / "runs",
                    pipeline_root=dummy_pipeline,
                    stream3d_config=args.stream3d_config,
                    score_mode=score_mode,
                    min_pred_pixels=1,
                    min_gt_pixels=1,
                    vertex_nn_radius=float(args.vertex_nn_radius),
                    vertex_cache_root=_project(args.vertex_cache_root),
                    use_cache=True,
                    max_frames=0,
                )
                summary = payload["summary"]
                rows.append(
                    _row_from_summary(
                        row_id=f"M{'3' if score_mode == 'constant' else '4'}_{scene}_stride{stride}_{score_mode}",
                        method="Stream3D_constant" if score_mode == "constant" else "Stream3D_pred_area",
                        scene=scene,
                        stride=stride,
                        score_mode=score_mode,
                        summary=summary,
                        source="current_run_v66_stream3d_rendered_from_scannet_mesh",
                        geometry_dependency="ScanNet_depth_pose_mesh_render_for_Stream3D",
                        diagnostic_only=True,
                        uses_gt_for_prediction=False,
                        forbidden_for_method_table=True,
                        output_summary=payload["outputs"]["summary_json"],
                    )
                )
                rows.append(
                    _row_from_summary(
                        row_id=f"M5_{scene}_stride{stride}_{score_mode}",
                        method="Stream3D_score_free",
                        scene=scene,
                        stride=stride,
                        score_mode="score_free_from_" + score_mode,
                        summary=summary,
                        source="derived_from_current_run_v66_stream3d_summary",
                        geometry_dependency="ScanNet_depth_pose_mesh_render_for_Stream3D",
                        diagnostic_only=True,
                        uses_gt_for_prediction=False,
                        forbidden_for_method_table=True,
                        output_summary=payload["outputs"]["summary_json"],
                    )
                )
                for item in payload.get("top_iou_pairs", []):
                    top_rows_all.append({"scene_id": scene, "method": rows[-2]["method"], "stride": stride, "score_mode": score_mode, **item})

            if pipeline_root is None:
                missing_rows.append(
                    {
                        "scene_id": scene,
                        "missing": "soma_fullscene_pipeline_root",
                        "attempted_autodiscovery": "v66_soma_fullscene_pipeline_<scene>_stride5_conf02_integrated_d4rt; v65_soma_fullscene_pipeline_<scene>_stride5_conf02_integrated_d4rt",
                    }
                )
                continue

            variant = _best_variant(pipeline_root)
            support_stats = _frame_mask_duplicate_stats(
                pipeline_root / "local_objectlets/objectlet_rows.csv",
                pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv",
                scene,
                variant,
            )
            mask_dir = _mask_dir_from_pipeline(pipeline_root)
            for score_mode in score_modes:
                payload = _run_one(
                    scene=scene,
                    method="soma",
                    stride=stride,
                    output_root=output_root / "runs",
                    pipeline_root=pipeline_root,
                    stream3d_config=args.stream3d_config,
                    score_mode=score_mode,
                    min_pred_pixels=1,
                    min_gt_pixels=1,
                    vertex_nn_radius=float(args.vertex_nn_radius),
                    vertex_cache_root=_project(args.vertex_cache_root),
                    use_cache=True,
                    max_frames=0,
                )
                summary = payload["summary"]
                support_row = dict(support_stats)
                support_row["missing_mask_frame_count"] = sum(
                    1 for frame_row in (payload.get("frame_rows") or []) if not frame_row.get("mask_exists", True)
                )
                rows.append(
                    _row_from_summary(
                        row_id=f"M{'0' if score_mode == 'constant' else '1'}_{scene}_stride{stride}_{score_mode}",
                        method="SOMA_current_constant" if score_mode == "constant" else "SOMA_current_pred_area",
                        scene=scene,
                        stride=stride,
                        score_mode=score_mode,
                        summary=summary,
                        source="current_fullscene_pipeline_root",
                        geometry_dependency="none_for_SOMA",
                        diagnostic_only=True,
                        uses_gt_for_prediction=False,
                        forbidden_for_method_table=True,
                        support_stats=support_row,
                        output_summary=payload["outputs"]["summary_json"],
                    )
                )
                rows.append(
                    _row_from_summary(
                        row_id=f"M2_{scene}_stride{stride}_{score_mode}",
                        method="SOMA_current_score_free",
                        scene=scene,
                        stride=stride,
                        score_mode="score_free_from_" + score_mode,
                        summary=summary,
                        source="derived_from_current_fullscene_pipeline_summary",
                        geometry_dependency="none_for_SOMA",
                        diagnostic_only=True,
                        uses_gt_for_prediction=False,
                        forbidden_for_method_table=True,
                        support_stats=support_row,
                        output_summary=payload["outputs"]["summary_json"],
                    )
                )
                for item in payload.get("top_iou_pairs", []):
                    top_rows_all.append({"scene_id": scene, "method": rows[-2]["method"], "stride": stride, "score_mode": score_mode, **item})

            raw_summary, raw_top = _raw_cropformer_summary(
                scene=scene,
                stride=stride,
                mask_dir=mask_dir,
                output_dir=output_root / "runs" / f"{scene}_raw_cropformer_stride{stride}",
            )
            rows.append(
                _row_from_summary(
                    row_id=f"M6_{scene}_stride{stride}_raw_cropformer",
                    method="raw_CropFormer_mask_connected_component_diagnostic",
                    scene=scene,
                    stride=stride,
                    score_mode="constant",
                    summary=raw_summary,
                    source="pipeline_mask_dir_per_frame_mask_ids",
                    geometry_dependency="none_for_SOMA",
                    diagnostic_only=True,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=True,
                    output_summary=_rel(output_root / "runs" / f"{scene}_raw_cropformer_stride{stride}" / "summary.json"),
                )
            )
            for item in raw_top:
                top_rows_all.append({"scene_id": scene, "method": rows[-1]["method"], "stride": stride, "score_mode": "constant", **item})

            oracle_map = _oracle_selected_mask_map(pipeline_root, scene, variant)
            oracle_summary, oracle_top = _mapped_mask_summary(
                scene=scene,
                stride=stride,
                mask_dir=mask_dir,
                mapping=oracle_map,
                output_dir=output_root / "runs" / f"{scene}_oracle_selected_masks_stride{stride}",
            )
            rows.append(
                _row_from_summary(
                    row_id=f"M7_{scene}_stride{stride}_oracle_selected_masks",
                    method="oracle_gt_group_selected_masks_diagnostic_for_upper_bound",
                    scene=scene,
                    stride=stride,
                    score_mode="constant",
                    summary=oracle_summary,
                    source="pipeline_selected_masks_grouped_by_diagnostic_best_gt",
                    geometry_dependency="none_for_SOMA",
                    diagnostic_only=True,
                    uses_gt_for_prediction=True,
                    forbidden_for_method_table=True,
                    output_summary=_rel(output_root / "runs" / f"{scene}_oracle_selected_masks_stride{stride}" / "summary.json"),
                )
            )
            for item in oracle_top:
                top_rows_all.append({"scene_id": scene, "method": rows[-1]["method"], "stride": stride, "score_mode": "constant", **item})

    stream3d_sf50 = [
        float(row["score_free_match50_recall"])
        for row in rows
        if row["method"] == "Stream3D_constant" and row["stride"] == 5 and row["score_free_match50_recall"] not in ("", None)
    ]
    soma_const = [row for row in rows if row["method"] == "SOMA_current_constant" and row["stride"] == 5]
    stream3d_const = [row for row in rows if row["method"] == "Stream3D_constant" and row["stride"] == 5]
    stream3d_by_scene = {row["scene_id"]: row for row in stream3d_const}
    gap_rows = []
    for soma in soma_const:
        s3d = stream3d_by_scene.get(soma["scene_id"])
        if s3d is None:
            continue
        gap_rows.append(
            {
                "scene_id": soma["scene_id"],
                "soma_AP50": soma["AP50"],
                "stream3d_AP50": s3d["AP50"],
                "soma_score_free_match50_recall": soma["score_free_match50_recall"],
                "stream3d_score_free_match50_recall": s3d["score_free_match50_recall"],
                "AP50_gap_stream3d_minus_soma": float(s3d["AP50"]) - float(soma["AP50"]),
                "score_free_recall_gap_stream3d_minus_soma": float(s3d["score_free_match50_recall"]) - float(soma["score_free_match50_recall"]),
            }
        )
    gate = {
        "stream3d_mean_score_free_match50_recall_ge_0p45": bool(stream3d_sf50) and float(np.mean(stream3d_sf50)) >= 0.45,
        "stream3d_no_scene_below_0p20": bool(stream3d_sf50) and all(value >= 0.20 for value in stream3d_sf50),
        "soma_fullscene_available_all_probe5": len(pipeline_roots) == len(scenes),
        "has_soma_gap_rows": bool(gap_rows),
    }
    gate["stream3d_pass"] = gate["stream3d_mean_score_free_match50_recall_ge_0p45"] and gate["stream3d_no_scene_below_0p20"]
    gate["pass"] = gate["stream3d_pass"] and gate["soma_fullscene_available_all_probe5"]

    _write_csv(output_root / "mv_ap_rows.csv", rows)
    _write_csv(output_root / "top_iou_pairs_by_scene.csv", top_rows_all)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_csv(output_root / "soma_stream3d_gap_rows.csv", gap_rows)
    summary = {
        "phase": "v66_scene_mv_ap_probe5",
        "diagnostic_only": True,
        "scenes": scenes,
        "strides": strides,
        "score_modes": score_modes,
        "stream3d_config": args.stream3d_config,
        "pipeline_roots": pipeline_roots,
        "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        "gate": gate,
        "stream3d_mean_score_free_match50_recall": float(np.mean(stream3d_sf50)) if stream3d_sf50 else None,
        "soma_gap_scene_count": len(gap_rows),
        "rows_csv": _rel(output_root / "mv_ap_rows.csv"),
        "top_iou_pairs_csv": _rel(output_root / "top_iou_pairs_by_scene.csv"),
        "gap_rows_csv": _rel(output_root / "soma_stream3d_gap_rows.csv"),
        "notes": [
            "SOMA rows are current fullscene pipeline rows only when a matching scene pipeline root is present.",
            "Missing fullscene SOMA pipeline roots are recorded instead of substituting legacy probe5 artifacts.",
            "Stream3D rows use ScanNet depth/pose/mesh rendering and are diagnostic_only.",
        ],
    }
    _write_json(output_root / "mv_ap_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v66 probe5 scene-level multi-view AP diagnostics.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--strides", default="5")
    parser.add_argument("--score-modes", default="constant,pred_area")
    parser.add_argument("--stream3d-config", default="scannet")
    parser.add_argument("--output-root", default="outputs/audit/v66_scene_mv_ap_probe5")
    parser.add_argument("--vertex-nn-radius", type=float, default=0.08)
    parser.add_argument("--vertex-cache-root", default="outputs/cache/v66_scene_multiview_vertex_maps")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

