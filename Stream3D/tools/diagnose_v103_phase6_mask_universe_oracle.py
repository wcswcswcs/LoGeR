#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402

try:
    from tools.v99_cupy_sparse_iou import CuPySparseSceneIoU  # noqa: E402
except Exception:  # pragma: no cover
    CuPySparseSceneIoU = None  # type: ignore[assignment]


PHASE_ID = "v103_phase6_mask_universe_oracle_diagnostic"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_mask_level_pooling_q5c_phase4r7_r4_control_gate_strict_l2o"
DEFAULT_PHASE2_SCENE0011 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6b_mask_universe_oracle_diagnostic"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _acc(use_cupy: bool, device_id: int) -> tuple[Any, str]:
    if use_cupy and CuPySparseSceneIoU is not None:
        return CuPySparseSceneIoU(device_id=device_id), "cupy_v99_sparse_scene_iou"
    return SparseSceneIoU(), "cpu_v65_sparse_scene_iou"


def _mask_to_gt(label: np.ndarray, gt: np.ndarray, mask_id: int) -> tuple[int, int, float]:
    pixels = label == int(mask_id)
    area = int(np.count_nonzero(pixels))
    if area <= 0:
        return 0, 0, 0.0
    gt_vals, counts = np.unique(gt[pixels & (gt > 0)], return_counts=True)
    if gt_vals.size == 0:
        return 0, area, 0.0
    idx = int(np.argmax(counts))
    return int(gt_vals[idx]), area, float(counts[idx] / max(1, area))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GT-only oracle diagnostic for v103 Phase6 mask universe.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
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
    phase5_root = _project(args.phase5_root)
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    feature_payloads = {
        scene: torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
        for scene in phase2_roots
    }
    policies = {
        "oracle_object_like_masks": lambda payload: payload["mask_is_object_like"].cpu().numpy().astype(bool),
        "oracle_all_supported_masks": lambda payload: payload["support_count"].cpu().numpy().astype(np.int64) > 0,
        "oracle_all_masks": lambda payload: np.ones(payload["mask_frame"].shape[0], dtype=bool),
    }
    metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for policy, selector in policies.items():
        window_metrics: list[dict[str, Any]] = []
        for scene, payload in feature_payloads.items():
            summary = phase2_summaries[scene]
            frame_ids = [int(v) for v in summary["frame_ids"]]
            mask_root = _project(summary["mask_root"])
            mask_frame = payload["mask_frame"].cpu().numpy().astype(np.int64)
            mask_label = payload["mask_label"].cpu().numpy().astype(np.int64)
            node_mask = selector(payload)
            by_frame: dict[int, list[int]] = defaultdict(list)
            for idx in np.flatnonzero(node_mask).tolist():
                by_frame[int(mask_frame[int(idx)])].append(int(idx))
            acc, backend = _acc(not bool(args.disable_cupy_iou), int(args.cupy_device_id))
            pred_scores: dict[int, float] = defaultdict(float)
            for local_idx, frame_id in enumerate(frame_ids):
                label = _load_label_png(mask_root / f"{int(frame_id)}.png")
                gt = _load_gt_2d(scene, frame_id, label.shape)
                pred = np.zeros(label.shape, dtype=np.int64)
                candidates = []
                for obs_idx in by_frame.get(int(local_idx), []):
                    gt_id, area, purity = _mask_to_gt(label, gt, int(mask_label[obs_idx]))
                    if gt_id <= 0:
                        continue
                    candidates.append((gt_id, area, purity, obs_idx))
                    assignment_rows.append(
                        {
                            "schema_version": "stream4d_v103_phase6_oracle_assignment_row_v1",
                            "phase_id": PHASE_ID,
                            "policy": policy,
                            "scene_id": scene,
                            "frame_id": int(frame_id),
                            "mask_observation_index": int(obs_idx),
                            "mask_id": int(mask_label[obs_idx]),
                            "oracle_gt_id": int(gt_id),
                            "mask_area": int(area),
                            "mask_gt_purity": purity,
                            "uses_gt_for_prediction": True,
                            "diagnostic_only": True,
                        }
                    )
                # WTA per GT id in a frame: keep the purest/largest mask.
                best_by_gt: dict[int, tuple[int, float, int]] = {}
                for gt_id, area, purity, obs_idx in candidates:
                    cur = best_by_gt.get(int(gt_id))
                    if cur is None or (purity, area) > (cur[1], cur[0]):
                        best_by_gt[int(gt_id)] = (int(area), float(purity), int(obs_idx))
                for gt_id, (area, purity, obs_idx) in best_by_gt.items():
                    pixels = label == int(mask_label[obs_idx])
                    pred[pixels & (pred == 0)] = int(gt_id)
                    pred_scores[int(gt_id)] = max(pred_scores[int(gt_id)], float(purity))
                acc.add(pred, gt)
                frame_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase6_oracle_frame_row_v1",
                        "phase_id": PHASE_ID,
                        "policy": policy,
                        "scene_id": scene,
                        "window_id": "c0000",
                        "frame_id": int(frame_id),
                        "candidate_mask_count": int(len(candidates)),
                        "emitted_oracle_object_count": int(len(best_by_gt)),
                        "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                        "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                        "uses_gt_for_prediction": True,
                        "diagnostic_only": True,
                    }
                )
            max_id = max(pred_scores.keys(), default=0)
            scores = np.ones((max_id,), dtype=np.float32)
            for gid, score in pred_scores.items():
                if gid > 0:
                    scores[int(gid) - 1] = float(score)
            metric, _iou, _pred_ids, _gt_ids = _summarize_iou(
                accumulator=acc,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                score_mode="input",
                input_scores=scores,
            )
            window_metrics.append(
                {
                    "schema_version": "stream4d_v103_phase6_oracle_window_metric_row_v1",
                    "phase_id": PHASE_ID,
                    "policy": policy,
                    "scene_id": scene,
                    "window_id": "c0000",
                    "MV_AP_window": metric.get("ap"),
                    "MV_AP50_window": metric.get("ap50"),
                    "MV_AP25_window": metric.get("ap25"),
                    "ScoreFreeMatch50_window": metric.get("score_free_match_at_050", {}).get("f1"),
                    "evaluated_pred_count": metric.get("evaluated_pred_count"),
                    "evaluated_gt_count": metric.get("evaluated_gt_count"),
                    "gt_best_iou_mean": metric.get("gt_best_iou_mean"),
                    "pred_best_iou_mean": metric.get("pred_best_iou_mean"),
                    "iou_backend": backend,
                    "uses_gt_for_prediction": True,
                    "diagnostic_only": True,
                }
            )
        metric_rows.extend(window_metrics)
        aggregate = {
            "schema_version": "stream4d_v103_phase6_oracle_metric_aggregate_row_v1",
            "phase_id": PHASE_ID,
            "policy": policy,
            "scene_count": len(window_metrics),
            "MV_AP_window": float(np.mean([float(r["MV_AP_window"]) for r in window_metrics])),
            "MV_AP50_window": float(np.mean([float(r["MV_AP50_window"]) for r in window_metrics])),
            "MV_AP25_window": float(np.mean([float(r["MV_AP25_window"]) for r in window_metrics])),
            "ScoreFreeMatch50_window": float(np.mean([float(r["ScoreFreeMatch50_window"]) for r in window_metrics])),
            "uses_gt_for_prediction": True,
            "diagnostic_only": True,
        }
        metric_rows.append(aggregate)
    _write_csv(out / "oracle_metric_rows.csv", metric_rows)
    _write_csv(out / "oracle_frame_rows.csv", frame_rows)
    _write_csv(out / "oracle_assignment_rows.csv", assignment_rows[:5000])
    summary = {
        "schema_version": "stream4d_v103_phase6_mask_universe_oracle_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "DIAGNOSTIC_ONLY_GT_ORACLE_NOT_METHOD_RESULT",
        "uses_gt_for_prediction": True,
        "diagnostic_only": True,
        "policy_count": len(policies),
        "truthfulness_note": "This diagnostic uses GT to assign masks to GT objects and is forbidden as a method prediction. It only estimates mask-universe upper-bound quality.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "oracle_metric_rows": _rel(out / "oracle_metric_rows.csv"),
            "oracle_frame_rows": _rel(out / "oracle_frame_rows.csv"),
            "oracle_assignment_rows": _rel(out / "oracle_assignment_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
