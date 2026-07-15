from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
V105_RUNNER = REPO_ROOT / "Stream3D" / "tools" / "run_v105_sgq_stream_pipeline.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Stream3D" / "outputs" / "audit" / "v105_specgap_phase1_baseline_x_review_20260711"


DEFAULT_RUNS: list[dict[str, str]] = [
    {
        "run_id": "X0_scene0011_baseline_x_sam2_twostage_sam2",
        "variant_id": "baseline_x_sam2_twostage_sam2",
        "label": "X0",
        "scene_id": "scene0011_00",
        "summary": "Stream3D/outputs/audit/v105_specgap_phase1_x0_scene0011_20260711/baseline_x_sam2_twostage_sam2/summary.json",
    },
    {
        "run_id": "X1_scene0011_baseline_x_gapadaptive_sam2",
        "variant_id": "baseline_x_gapadaptive_sam2",
        "label": "X1",
        "scene_id": "scene0011_00",
        "summary": "Stream3D/outputs/audit/v105_specgap_phase1_x1_scene0011_20260711/baseline_x_gapadaptive_sam2/summary.json",
    },
    {
        "run_id": "X0_scene0050_baseline_x_sam2_twostage_sam2",
        "variant_id": "baseline_x_sam2_twostage_sam2",
        "label": "X0",
        "scene_id": "scene0050_00",
        "summary": "Stream3D/outputs/audit/v105_specgap_phase1_x0_scene0050_20260711/baseline_x_sam2_twostage_sam2/summary.json",
    },
    {
        "run_id": "X1_scene0050_baseline_x_gapadaptive_sam2",
        "variant_id": "baseline_x_gapadaptive_sam2",
        "label": "X1",
        "scene_id": "scene0050_00",
        "summary": "Stream3D/outputs/audit/v105_specgap_phase1_x1_scene0050_20260711/baseline_x_gapadaptive_sam2/summary.json",
    },
]


VISUAL_RECORDS: list[dict[str, Any]] = [
    {
        "run_id": "X0_scene0011_baseline_x_sam2_twostage_sam2",
        "sheet_range": "00_07",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense object/surface coverage",
        "id_observation": "large chair/table/floor surfaces visible; many fine edge fragments",
        "risk": "over-segmentation and broad surface dominance",
    },
    {
        "run_id": "X0_scene0011_baseline_x_sam2_twostage_sam2",
        "sheet_range": "08_15",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage continues",
        "id_observation": "main chair remains visible; floor/table surfaces stable-looking; small fragments persist",
        "risk": "edge fragments and broad floor/table masks",
    },
    {
        "run_id": "X0_scene0011_baseline_x_sam2_twostage_sam2",
        "sheet_range": "16_23",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage, view rotates",
        "id_observation": "near chair partly leaves frame; object color continuity not fully conclusive from sheet",
        "risk": "possible ID/color discontinuity across view change",
    },
    {
        "run_id": "X0_scene0011_baseline_x_sam2_twostage_sam2",
        "sheet_range": "24_31",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage remains",
        "id_observation": "closest chair appears with different color family than early frames; no whole-frame loss",
        "risk": "visual ID continuity risk, not a blank/drop failure",
    },
    {
        "run_id": "X1_scene0011_baseline_x_gapadaptive_sam2",
        "sheet_range": "00_07",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage with fewer visible objects than X0 summary",
        "id_observation": "main furniture surfaces coherent; fine fragments still present",
        "risk": "surface dominance, edge fragments",
    },
    {
        "run_id": "X1_scene0011_baseline_x_gapadaptive_sam2",
        "sheet_range": "08_15",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage continues",
        "id_observation": "large chair/table/floor masks stable-looking within sheet",
        "risk": "broad masks dominate scene",
    },
    {
        "run_id": "X1_scene0011_baseline_x_gapadaptive_sam2",
        "sheet_range": "16_23",
        "blank_or_dropped_frames": False,
        "coverage_observation": "coverage remains nonzero",
        "id_observation": "view rotation; chair/floor/table still visible; no complete loss",
        "risk": "some object color/ID continuity uncertainty",
    },
    {
        "run_id": "X1_scene0011_baseline_x_gapadaptive_sam2",
        "sheet_range": "24_31",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage remains",
        "id_observation": "closest chair appears in different color family vs early front view",
        "risk": "visual ID continuity risk",
    },
    {
        "run_id": "X1_scene0050_baseline_x_gapadaptive_sam2",
        "sheet_range": "00_07",
        "blank_or_dropped_frames": False,
        "coverage_observation": "piano/table/wall/floor covered",
        "id_observation": "piano body changes from dark to red color family around early frames",
        "risk": "obvious visual ID/color discontinuity for piano",
    },
    {
        "run_id": "X1_scene0050_baseline_x_gapadaptive_sam2",
        "sheet_range": "08_15",
        "blank_or_dropped_frames": False,
        "coverage_observation": "table/floor plus a right-edge couch/sofa sliver are visible, but the contact sheet is not sufficient to clear couch tracking",
        "id_observation": "table remains magenta; couch/sofa enters at the right edge and needs focused GT/pred overlay audit",
        "risk": "edge artifacts at right/bottom; couch ownership is not visually cleared by this sheet",
    },
    {
        "run_id": "X1_scene0050_baseline_x_gapadaptive_sam2",
        "sheet_range": "16_23",
        "blank_or_dropped_frames": False,
        "coverage_observation": "couch/sofa and small object are visible, but original color overlay makes sofa fill hard to judge",
        "id_observation": "small object on sofa remains visible; couch/sofa large region appears mostly as boundary/low-contrast fill in this overlay",
        "risk": "coverage shifts from piano/table to sofa view; contact sheet can mislead couch segmentation judgment",
    },
    {
        "run_id": "X1_scene0050_baseline_x_gapadaptive_sam2",
        "sheet_range": "24_31",
        "blank_or_dropped_frames": False,
        "coverage_observation": "couch/sofa and surrounding surfaces visible, but sofa must be judged from focused GT/pred audit rather than this low-contrast sheet",
        "id_observation": "large couch/sofa contour is present; original overlay does not provide reliable visual evidence of object-level tracking",
        "risk": "large sofa treated as broad object, floor/bottom fragments, focused sofa audit required",
    },
    {
        "run_id": "X0_scene0050_baseline_x_sam2_twostage_sam2",
        "sheet_range": "00_07",
        "blank_or_dropped_frames": False,
        "coverage_observation": "piano/table/wall/floor covered",
        "id_observation": "piano body changes from dark to brown after early frames",
        "risk": "visual ID/color discontinuity for piano, broad object masks",
    },
    {
        "run_id": "X0_scene0050_baseline_x_sam2_twostage_sam2",
        "sheet_range": "08_15",
        "blank_or_dropped_frames": False,
        "coverage_observation": "dense coverage persists",
        "id_observation": "table stable magenta; couch/sofa enters; bottom/right fragments",
        "risk": "edge fragments, view-transition granularity, and unresolved couch ownership",
    },
    {
        "run_id": "X0_scene0050_baseline_x_sam2_twostage_sam2",
        "sheet_range": "16_23",
        "blank_or_dropped_frames": False,
        "coverage_observation": "couch/sofa and small-object region visible, but original color overlay is not reliable enough for sofa tracking clearance",
        "id_observation": "small object remains visible; couch/sofa broad region requires focused GT/pred overlay because X0 has early ID changes",
        "risk": "bottom fragments, coarse sofa grouping, and sofa ID continuity risk",
    },
    {
        "run_id": "X0_scene0050_baseline_x_sam2_twostage_sam2",
        "sheet_range": "24_31",
        "blank_or_dropped_frames": False,
        "coverage_observation": "couch/sofa and floor/bottom regions covered in broad masks",
        "id_observation": "no blank/drop; broad couch/sofa contour remains, but object-level tracking must be checked by focused couch audit",
        "risk": "coarse large-object grouping, residual fragments, and sofa tracking not cleared by contact sheet alone",
    },
]


def _load_v105_runner() -> Any:
    spec = importlib.util.spec_from_file_location("v105_runner_for_phase1_review", V105_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load v105 runner module: {V105_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v105_runner_for_phase1_review"] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else REPO_ROOT / path_obj


def _build_review(runs: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    mod = _load_v105_runner()
    mask_base = output_root / "mv_ap_bridge_masks"
    pipeline_base = output_root / "mv_ap_bridge_pipelines"
    eval_base = output_root / "mv_ap_diagnostic"

    runtime_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []

    for run in runs:
        summary_path = _repo_path(run["summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scene = run["scene_id"]
        frame_ids = [int(value) for value in summary["frame_ids"]]
        source_label_dir = summary_path.parent / "labels"
        mask_dir = mask_base / run["run_id"] / scene / "mask"
        mask_dir.mkdir(parents=True, exist_ok=True)

        copy_records: list[dict[str, Any]] = []
        missing_masks: list[str] = []
        for frame_id in frame_ids:
            src = source_label_dir / f"frame_{frame_id:06d}.png"
            dst = mask_dir / f"{frame_id}.png"
            if not src.exists():
                missing_masks.append(str(src))
                continue
            shutil.copy2(src, dst)
            copy_records.append({"frame_id": frame_id, "src": mod._rel(src), "dst": mod._rel(dst)})

        pipeline_root = pipeline_base / run["run_id"]
        support = mod._write_pipeline_support_root(
            ctx=None,
            pipeline_root=pipeline_root,
            variant_id=run["run_id"],
            mask_root=mod._rel(mask_base / run["run_id"]),
            mask_dir_by_scene={scene: mask_dir},
            frame_ids_by_scene={scene: frame_ids},
            object_id_policy="mask_id_is_track",
        )
        eval_rows = mod._run_v65_soma_eval(
            scene_id=scene,
            pipeline_root=pipeline_root,
            output_root=eval_base / run["run_id"],
            stride=5,
            max_frames=len(frame_ids),
        )
        if len(eval_rows) != 1:
            raise RuntimeError(f"{run['run_id']}: expected one v65 eval row, got {len(eval_rows)}")
        eval_row = dict(eval_rows[0])

        sheet_paths = summary.get("sheet_paths", [])
        runtime_rows.append(
            {
                "schema_version": "stream4d_v105_phase1_baseline_x_runtime_row_v1",
                "run_id": run["run_id"],
                "label": run["label"],
                "variant_id": run["variant_id"],
                "scene_id": scene,
                "frame_count": summary.get("frame_count"),
                "total_runtime_sec": summary.get("total_runtime_sec"),
                "setup_sec": summary.get("setup_sec"),
                "frame0_stage1_runtime_sec": summary.get("frame0_stage1_runtime_sec"),
                "frame0_stage2_runtime_sec": summary.get("frame0_stage2_runtime_sec"),
                "total_tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
                "total_gap_segmentation_runtime_sec": summary.get("total_gap_segmentation_runtime_sec"),
                "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
                "initial_stage1_mask_count": summary.get("initial_stage1_mask_count"),
                "initial_stage2_mask_count": summary.get("initial_stage2_mask_count"),
                "initial_mask_count": summary.get("initial_mask_count"),
                "total_object_id_count": summary.get("total_object_id_count"),
                "mean_visible_id_count": summary.get("mean_visible_id_count"),
                "mean_foreground_ratio": summary.get("mean_foreground_ratio"),
                "empty_propagation_frames": summary.get("empty_propagation_frames"),
                "video_path": summary.get("video_path"),
                "video_exists": bool(summary.get("video_path") and Path(summary["video_path"]).exists()),
                "sheet_count": len(sheet_paths),
                "sheets_exist": all(Path(path).exists() for path in sheet_paths),
                "summary_json": mod._rel(summary_path),
                "sam2_checkpoint": summary.get("sam2_checkpoint"),
                "sam2_checkpoint_sha256": summary.get("sam2_checkpoint_sha256"),
                "sam2_cfg": summary.get("sam2_cfg"),
            }
        )
        metric_rows.append(
            {
                "schema_version": "stream4d_v105_phase1_baseline_x_mv_ap_diagnostic_row_v1",
                "run_id": run["run_id"],
                "label": run["label"],
                "variant_id": run["variant_id"],
                "scene_id": scene,
                "metric_scope": "window_dev_smoke_v65_soma_2d_multiview_diagnostic",
                "diagnostic_only": True,
                "AP": eval_row.get("AP"),
                "MV_AP_window": eval_row.get("AP"),
                "MV_AP50_window": eval_row.get("AP50"),
                "MV_AP25_window": eval_row.get("AP25"),
                "frame_count": eval_row.get("frame_count"),
                "evaluated_pred_count": eval_row.get("evaluated_pred_count"),
                "evaluated_gt_count": eval_row.get("evaluated_gt_count"),
                "gt_best_iou_mean": eval_row.get("gt_best_iou_mean"),
                "gt_recall_best_iou_ge_025": eval_row.get("gt_recall_best_iou_ge_025"),
                "gt_recall_best_iou_ge_050": eval_row.get("gt_recall_best_iou_ge_050"),
                "score_mode": eval_row.get("score_mode"),
                "summary_json": eval_row.get("summary_json"),
                "pipeline_root": mod._rel(pipeline_root),
                "mask_root": mod._rel(mask_base / run["run_id"]),
                "object_id_policy": "mask_id_is_track",
                "source_summary_json": mod._rel(summary_path),
            }
        )
        bridge_rows.append(
            {
                "schema_version": "stream4d_v105_phase1_mv_ap_bridge_row_v1",
                "run_id": run["run_id"],
                "scene_id": scene,
                "source_label_dir": mod._rel(source_label_dir),
                "bridge_mask_dir": mod._rel(mask_dir),
                "frame_count": len(frame_ids),
                "copied_mask_count": len(copy_records),
                "missing_mask_count": len(missing_masks),
                "missing_masks": missing_masks,
                "support": support,
                "copy_records": copy_records,
            }
        )

    summary_payload = {
        "schema_version": "stream4d_v105_phase1_baseline_x_review_summary_v1",
        "status": "phase1_baseline_x_artifacts_and_mv_ap_diagnostics_complete_with_visual_risks_sofa_requires_focused_audit",
        "diagnostic_only": True,
        "mv_ap_note": (
            "MV_AP_window here is v65 scene-level 2D multiview AP over baseline-x label PNGs bridged "
            "into the v105 support-ledger contract; it is not official 3D AP and is not a hard success gate "
            "in the v105 SpecGap plan."
        ),
        "runtime_record_count": len(runtime_rows),
        "metric_record_count": len(metric_rows),
        "visual_sheet_record_count": len(VISUAL_RECORDS),
        "all_videos_exist": all(row["video_exists"] for row in runtime_rows),
        "all_four_sheets_exist": all(row["sheet_count"] == 4 and row["sheets_exist"] for row in runtime_rows),
        "all_empty_propagation_frames_zero": all(int(row["empty_propagation_frames"]) == 0 for row in runtime_rows),
        "visual_gate": "no_blank_or_drop_observed_but_contact_sheets_do_not_clear_scene0050_sofa_tracking",
        "sofa_blocker_audit": mod._rel(
            REPO_ROOT
            / "Stream3D/outputs/audit/v105_specgap_phase1_sofa_blocker_20260711/scene0050_couch_summary.json"
        ),
        "runtime_records_json": mod._rel(output_root / "phase1_runtime_records.json"),
        "mv_ap_metric_records_json": mod._rel(output_root / "phase1_mv_ap_diagnostic_records.json"),
        "visual_assessment_records_json": mod._rel(output_root / "phase1_visual_assessment_records.json"),
        "mv_ap_bridge_records_json": mod._rel(output_root / "phase1_mv_ap_bridge_records.json"),
    }
    _write_json(
        output_root / "phase1_runtime_records.json",
        {"schema_version": "stream4d_v105_phase1_baseline_x_runtime_table_v1", "row_count": len(runtime_rows), "rows": runtime_rows},
    )
    _write_json(
        output_root / "phase1_mv_ap_diagnostic_records.json",
        {"schema_version": "stream4d_v105_phase1_baseline_x_mv_ap_diagnostic_table_v1", "row_count": len(metric_rows), "rows": metric_rows},
    )
    _write_json(
        output_root / "phase1_visual_assessment_records.json",
        {"schema_version": "stream4d_v105_phase1_visual_assessment_table_v1", "row_count": len(VISUAL_RECORDS), "rows": VISUAL_RECORDS},
    )
    _write_json(
        output_root / "phase1_mv_ap_bridge_records.json",
        {"schema_version": "stream4d_v105_phase1_mv_ap_bridge_table_v1", "row_count": len(bridge_rows), "rows": bridge_rows},
    )
    _write_json(output_root / "phase1_summary.json", summary_payload)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v105 Phase 1 baseline-x review and MV_AP diagnostic artifacts.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--runs-json", default="", help="Optional JSON file containing run records; defaults to the 20260711 Phase 1 runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = DEFAULT_RUNS
    if args.runs_json:
        runs = json.loads(_repo_path(args.runs_json).read_text(encoding="utf-8"))
    summary = _build_review(runs, _repo_path(args.output_root))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
