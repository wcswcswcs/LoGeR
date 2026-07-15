#!/usr/bin/env python3
"""Run v106 rolling SAM2 with a v107 Phase1 raw-logit trace hook."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base  # noqa: E402
import tools.run_v106_stateful_sam2_rolling_scene_stream as v106_runner  # noqa: E402

VARIANT_ID = "v106_stateful_sam2_rolling_scene_stream"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return ROOT / path


def imread_label(path: Path):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def compare_labels(reference_root: Path, instrumented_root: Path) -> dict[str, Any]:
    ref_summary_path = reference_root / "summary.json"
    cand_summary_path = instrumented_root / "summary.json"
    ref_summary = read_json(ref_summary_path)
    cand_summary = read_json(cand_summary_path)
    ref_records = {int(row["frame_id"]): row for row in ref_summary.get("records", [])}
    cand_records = {int(row["frame_id"]): row for row in cand_summary.get("records", [])}
    rows = []
    pixel_mismatch_count = 0
    missing_frames = []
    for frame_id in sorted(ref_records):
        if frame_id not in cand_records:
            missing_frames.append(int(frame_id))
            continue
        ref_label_path = resolve(str(ref_records[frame_id]["label_path"]), reference_root)
        cand_label_path = resolve(str(cand_records[frame_id]["label_path"]), instrumented_root)
        ref_label = imread_label(ref_label_path)
        cand_label = imread_label(cand_label_path)
        same_shape = ref_label.shape == cand_label.shape
        exact = bool(same_shape and (ref_label == cand_label).all())
        mismatch = 0 if exact else int((ref_label != cand_label).sum()) if same_shape else -1
        if mismatch > 0:
            pixel_mismatch_count += mismatch
        rows.append(
            {
                "frame_id": int(frame_id),
                "reference_label": rel(ref_label_path),
                "instrumented_label": rel(cand_label_path),
                "same_shape": same_shape,
                "pixel_exact_equal": exact,
                "pixel_mismatch_count": mismatch,
                "reference_sha256": sha256_file(ref_label_path),
                "instrumented_sha256": sha256_file(cand_label_path),
            }
        )
    exact_frame_count = int(sum(1 for row in rows if row["pixel_exact_equal"]))
    return {
        "schema_version": "stream4d_v107_live_instrumented_label_parity_v1",
        "reference_summary": {"path": rel(ref_summary_path), "sha256": sha256_file(ref_summary_path)},
        "instrumented_summary": {"path": rel(cand_summary_path), "sha256": sha256_file(cand_summary_path)},
        "frame_count": int(len(ref_records)),
        "compared_frame_count": int(len(rows)),
        "exact_frame_count": exact_frame_count,
        "missing_frames": missing_frames,
        "pixel_mismatch_count": int(pixel_mismatch_count),
        "label_exact_parity_pass": exact_frame_count == len(ref_records) and not missing_frames and pixel_mismatch_count == 0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--seed", type=int, default=105)
    args = parser.parse_args()

    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def trace_hook(frame_idx: int, obj_ids: Any, logits: Any, _masks: Any) -> None:
        import torch

        with torch.no_grad():
            tensor = logits.detach().float()
            flat = tensor.flatten(1)
            means = flat.mean(dim=1).detach().cpu().tolist()
            mins = flat.min(dim=1).values.detach().cpu().tolist()
            maxs = flat.max(dim=1).values.detach().cpu().tolist()
            stds = flat.std(dim=1, unbiased=False).detach().cpu().tolist()
            positive = (flat > 0.0).sum(dim=1).detach().cpu().tolist()
            ids = [int(v) for v in obj_ids]
            shape = [int(v) for v in tensor.shape]
        for i, obj_id in enumerate(ids):
            rows.append(
                {
                    "schema_version": "stream4d_v107_raw_logit_stats_v1",
                    "frame_index": int(frame_idx),
                    "runtime_id": int(obj_id),
                    "global_id": int(obj_id),
                    "logit_shape": shape,
                    "raw_logit_mean": float(means[i]),
                    "raw_logit_min": float(mins[i]),
                    "raw_logit_max": float(maxs[i]),
                    "raw_logit_std": float(stds[i]),
                    "positive_logit_pixel_count": int(positive[i]),
                }
            )

    old_hook = base.STREAM_INFER_TRACE_HOOK
    base.STREAM_INFER_TRACE_HOOK = trace_hook
    started = time.time()
    try:
        v106_args = [
            "--config",
            str(args.config),
            "--scene-id",
            str(args.scene_id),
            "--frame-start",
            str(args.frame_start),
            "--frame-stride",
            str(args.frame_stride),
            "--frame-count",
            str(args.frame_count),
            "--output-root",
            str(output_root),
            "--gpu",
            str(args.gpu),
            "--seed",
            str(args.seed),
        ]
        rc = v106_runner.main(v106_args)
    finally:
        base.STREAM_INFER_TRACE_HOOK = old_hook
    elapsed = float(time.time() - started)
    if rc != 0:
        raise RuntimeError(f"v106 runner exited with {rc}")

    run_root = output_root / VARIANT_ID
    trace_dir = output_root / "phase1_live_instrumentation"
    trace_dir.mkdir(parents=True, exist_ok=True)
    logit_path = trace_dir / "raw_logit_stats.parquet"
    pd.DataFrame(rows).to_parquet(logit_path, index=False)
    pd.DataFrame(rows).to_csv(trace_dir / "raw_logit_stats.csv", index=False)
    parity = compare_labels(reference_root, run_root)
    write_json(trace_dir / "live_instrumented_label_parity.json", parity)
    summary = {
        "schema_version": "stream4d_v107_phase1_live_instrumented_replay_summary_v1",
        "reference_run_root": rel(reference_root),
        "instrumented_run_root": rel(run_root),
        "raw_logit_stats_parquet": rel(logit_path),
        "raw_logit_row_count": int(len(rows)),
        "wall_time_sec": elapsed,
        "label_exact_parity_pass": parity["label_exact_parity_pass"],
        "pixel_mismatch_count": parity["pixel_mismatch_count"],
        "decision": (
            "PASS_PHASE1_LIVE_RAW_LOGIT_TRACE_LABEL_PARITY"
            if parity["label_exact_parity_pass"]
            else "NO_GO_PHASE1_LIVE_TRACE_LABEL_PARITY_FAILED"
        ),
        "honesty_note": (
            "The hook records per-object raw logit summary statistics only. It does not store dense logits and "
            "does not modify infer_stream_frame outputs."
        ),
    }
    write_json(trace_dir / "live_instrumented_replay_summary.json", summary)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase1_live_instrumented_replay_run_summary_v1",
            "summary": rel(trace_dir / "live_instrumented_replay_summary.json"),
            "decision": summary["decision"],
            "label_exact_parity_pass": summary["label_exact_parity_pass"],
            "raw_logit_row_count": summary["raw_logit_row_count"],
        },
    )
    print(json.dumps({"output_root": str(output_root), **summary}, sort_keys=True), flush=True)
    return 0 if parity["label_exact_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
