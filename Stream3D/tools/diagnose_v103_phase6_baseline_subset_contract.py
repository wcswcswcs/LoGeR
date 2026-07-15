#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    PHASE_ID as PHASE6_ID,
    _evaluate_variant,
    _jsonable,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6_baseline_subset_contract_diagnostic"
DEFAULT_F2_ROOT = STREAM3D_ROOT / "outputs/audit/v100_phase2c_overlap3_local_repair"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6_baseline_subset_contract_r1"
DEFAULT_FULL_BASELINE = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"


def _load_full_dev_baseline(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    rows = df[(df["baseline_role"] == "current_strong_local_baseline") & (df["dataset_split"] == "dev")]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "MV_AP_window": float(row["MV_AP_window"]),
        "MV_AP50_window": float(row["MV_AP50_window"]),
        "MV_AP25_window": float(row["MV_AP25_window"]),
        "ScoreFreeMatch50_window": float(row["ScoreFreeMatch50_window"]),
        "source_artifact": str(row.get("source_artifact", "")),
        "metric_source": str(row.get("metric_source", "")),
    }


def _adapt_f2_rows(*, f2_root: Path, phase2_summaries: dict[str, dict[str, Any]], dataset_split: str, chunk_id: str) -> dict[str, list[dict[str, Any]]]:
    path = f2_root / "mv_object_frame_mask_rows.parquet"
    rows = pd.read_parquet(path)
    rows = rows[(rows["dataset_split"].astype(str) == dataset_split) & (rows["chunk_id"].astype(str) == chunk_id)].copy()
    if rows.empty:
        raise RuntimeError(f"no F2 rows for dataset_split={dataset_split} chunk_id={chunk_id}: {path}")
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_to_local = {
        scene: {int(frame_id): idx for idx, frame_id in enumerate(summary["frame_ids"])}
        for scene, summary in phase2_summaries.items()
    }
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_id = int(row["frame_id"])
        if scene not in frame_to_local or frame_id not in frame_to_local[scene]:
            continue
        oid = str(row["mv_object_id"])
        scene_rows[scene].append(
            {
                "schema_version": "stream4d_v103_phase6_baseline_subset_input_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": "F2_v100_phase2c_overlap3_c0000_first32_replay",
                "mv_object_id": oid,
                "object_id": oid,
                "scene_id": scene,
                "chunk_id": chunk_id,
                "window_id": str(row.get("window_id", chunk_id)),
                "frame_local_index": int(frame_to_local[scene][frame_id]),
                "frame_id": frame_id,
                "selected_mask_id": int(row["selected_mask_id"]),
                "mask_id_or_generated_id": int(row["mask_id_or_generated_id"]),
                "object_score": float(row.get("score", 0.0)),
                "score": float(row.get("score", 0.0)),
                "support_count": int(row.get("support_surfel_count", 0) or 0),
                "node_policy": "f2_phase2c_skeleton_replay",
                "emit_policy": str(row.get("eval_emit_policy", "f2_phase2c_primary_emit")),
                "readout_mode": str(row.get("readout_mode", "f2_phase2c_primary_emit")),
                "uses_gt_for_prediction": bool(row.get("uses_gt_for_prediction", False)),
                "uses_future": bool(row.get("uses_future", False)),
            }
        )
    return scene_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the locked F2 baseline on the exact v103 Phase6 first32 subset.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--full-baseline-rows", default=str(DEFAULT_FULL_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0000")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    scene_rows = _adapt_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    variant_id = "F2_v100_phase2c_overlap3_c0000_first32_replay"
    window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, frame_eval_count = _evaluate_variant(
        variant_id=variant_id,
        scene_rows=scene_rows,
        phase2_summaries=phase2_summaries,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
        use_cupy_iou=not bool(args.disable_cupy_iou),
        cupy_device_id=int(args.cupy_device_id),
    )
    full_baseline = _load_full_dev_baseline(_project(args.full_baseline_rows))
    aggregate.update(
        {
            "variant_id": variant_id,
            "phase_id": PHASE_ID,
            "dataset_split": str(args.dataset_split),
            "chunk_id": str(args.chunk_id),
            "metric_scope": "same_subset_as_v103_phase6_first32_c0000",
            "phase6_evaluator_source": PHASE6_ID,
        }
    )
    _write_csv(out / "baseline_subset_metric_rows.csv", [aggregate])
    _write_csv(out / "baseline_subset_window_rows.csv", window_rows)
    _write_csv(out / "baseline_subset_selected_rows.csv", selected_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6_baseline_subset_contract_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "variant_id": variant_id,
        "dataset_split": str(args.dataset_split),
        "chunk_id": str(args.chunk_id),
        "subset_MV_AP_window": aggregate.get("MV_AP_window"),
        "subset_MV_AP50_window": aggregate.get("MV_AP50_window"),
        "subset_MV_AP25_window": aggregate.get("MV_AP25_window"),
        "subset_ScoreFreeMatch50_window": aggregate.get("ScoreFreeMatch50_window"),
        "full_dev_baseline_contract": full_baseline,
        "subset_minus_full_dev_MV_AP_window": float(aggregate.get("MV_AP_window", 0.0)) - float(full_baseline.get("MV_AP_window", 0.0))
        if full_baseline
        else "",
        "pixel_collision_count": int(pixel_collision_count),
        "missing_mask_raster_count": int(missing_count),
        "frame_eval_count": int(frame_eval_count),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "truthfulness_note": "This diagnostic replays the locked v100 Phase2c F2 primary emit rows on the exact v103 Phase6 first32 subset; GT is used only by the canonical evaluator.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "baseline_subset_metric_rows": _rel(out / "baseline_subset_metric_rows.csv"),
            "baseline_subset_window_rows": _rel(out / "baseline_subset_window_rows.csv"),
            "baseline_subset_selected_rows": _rel(out / "baseline_subset_selected_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
