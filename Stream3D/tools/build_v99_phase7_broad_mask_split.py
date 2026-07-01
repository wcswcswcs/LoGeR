#!/usr/bin/env python3
"""Conservative DA3 depth split for broad F2 masks."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase4_f2_da3_link_verifier as p4  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase7_broad_mask_split"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE2_SUMMARY = PHASE2_DIR / "best_variant_summary.json"
MIN_COMPONENT_AREA_PX = 4096


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no rows for Phase2 best variant {variant}")
    return variant, rows


def _load_depth(scene: str, frame: int, da3_maps: dict[str, dict[str, Any]]) -> np.ndarray | None:
    if scene not in da3_maps or frame not in da3_maps[scene]["frame_to_idx"]:
        return None
    idx = da3_maps[scene]["frame_to_idx"][frame]
    npz_path = da3_maps[scene]["output"] / "results_output" / f"frame_{idx}.npz"
    if not npz_path.exists():
        return None
    return np.asarray(np.load(npz_path)["depth"], dtype=np.float32)


def _split_mask_by_depth(label: np.ndarray, mask_id: int, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    mask = label == int(mask_id)
    if int(np.count_nonzero(mask)) < 2 * MIN_COMPONENT_AREA_PX:
        return None
    depth_resized = cv2.resize(depth, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
    vals = depth_resized[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 2 * MIN_COMPONENT_AREA_PX:
        return None
    threshold = float(np.median(vals))
    comp_a = mask & (depth_resized <= threshold)
    comp_b = mask & (depth_resized > threshold)
    if int(np.count_nonzero(comp_a)) < MIN_COMPONENT_AREA_PX or int(np.count_nonzero(comp_b)) < MIN_COMPONENT_AREA_PX:
        return None
    return comp_a, comp_b, threshold


def _should_split(row: dict[str, Any], area_ratio: float, variant: str) -> bool:
    score = _num(row.get("score"), 1.0)
    if variant == "P7_U1_DA3_component_split_broad_only":
        return area_ratio >= 0.30
    if variant == "P7_U2_DA3_semantic_conservative_split_broad_only":
        return area_ratio >= 0.30 and score < 0.95
    if variant == "P7_U3_DA3_D4RT_anchor_component_split_broad_only":
        return area_ratio >= 0.30 and score < 0.85
    if variant == "P7_U4_conservative_split_only_when_F2_conf_low":
        return area_ratio >= 0.25 and score < 0.75
    if variant == "P7_U5_no_split_when_F2_conf_high":
        return area_ratio >= 0.35 and score < 0.60
    return False


def _build_variant(
    variant: str,
    parent_rows: list[dict[str, Any]],
    scope: dict[str, Any],
    da3_maps: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Path], list[dict[str, Any]]]:
    if variant == "P7_B0_phase2_best_no_split":
        return [dict(row, variant_id=variant, phase7_parent_variant_id=row["variant_id"]) for row in parent_rows], {}, []

    rows_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        rows_by_frame[(str(row["scene_id"]), int(row["frame_id"]))].append(row)
    out_rows: list[dict[str, Any]] = []
    mask_paths: dict[str, Path] = {}
    split_rows: list[dict[str, Any]] = []
    mask_out_dir = OUT_DIR / "generated_masks" / variant
    for (scene, frame), rows in sorted(rows_by_frame.items()):
        src_path = scope["mask_path_by_frame"].get((scene, frame))
        if src_path is None or not src_path.exists():
            out_rows.extend(dict(row, variant_id=variant, phase7_parent_variant_id=row["variant_id"]) for row in rows)
            continue
        label = p1._read_label(src_path).copy()
        original = label.copy()
        depth = _load_depth(scene, frame, da3_maps)
        max_label = int(label.max()) + 1
        frame_changed = False
        for row in rows:
            mask_id = int(row["selected_mask_id"])
            area_px = int(np.count_nonzero(original == mask_id))
            area_ratio = float(area_px / max(1, original.size))
            split = None if depth is None or not _should_split(row, area_ratio, variant) else _split_mask_by_depth(original, mask_id, depth)
            if split is None:
                new = dict(row)
                new["variant_id"] = variant
                new["phase7_parent_variant_id"] = row["variant_id"]
                new["phase7_split_action"] = "keep_original"
                out_rows.append(new)
                continue
            comp_a, comp_b, threshold = split
            new_id_a = max_label
            new_id_b = max_label + 1
            max_label += 2
            label[original == mask_id] = 0
            label[comp_a] = new_id_a
            label[comp_b] = new_id_b
            frame_changed = True
            for comp_idx, (new_id, comp_mask) in enumerate([(new_id_a, comp_a), (new_id_b, comp_b)]):
                new = dict(row)
                new["variant_id"] = variant
                new["mv_object_id"] = f"{row['mv_object_id']}|phase7_depthsplit{comp_idx}"
                new["object_id"] = new["mv_object_id"]
                new["selected_mask_id"] = int(new_id)
                new["mask_id_or_generated_id"] = int(new_id)
                new["score"] = float(_num(row.get("score"), 1.0) * 0.75)
                new["score_policy"] = "phase7_depth_split_component_score_parent_times_0p75"
                new["phase7_parent_variant_id"] = row["variant_id"]
                new["phase7_parent_mask_id"] = mask_id
                new["phase7_split_action"] = "da3_depth_median_component"
                out_rows.append(new)
                split_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase7_split_row_v1",
                        "phase_id": "v99_phase7_broad_mask_split",
                        "variant_id": variant,
                        "scene_id": scene,
                        "frame_id": frame,
                        "parent_mask_id": mask_id,
                        "generated_mask_id": int(new_id),
                        "component_index": comp_idx,
                        "parent_area_ratio": area_ratio,
                        "component_area_px": int(np.count_nonzero(comp_mask)),
                        "depth_median_threshold": threshold,
                        "parent_score": _num(row.get("score"), 1.0),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        if frame_changed:
            out_path = mask_out_dir / scene / f"{frame}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), label.astype(np.uint16))
            mask_paths[f"{scene}|{frame}"] = out_path
    return out_rows, mask_paths, split_rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    parent_variant, parent_rows = _phase2_best_rows()
    source_scope = p1._load_source_scope()
    da3_maps = p4._load_da3_maps()
    variants = [
        "P7_B0_phase2_best_no_split",
        "P7_U1_DA3_component_split_broad_only",
        "P7_U2_DA3_semantic_conservative_split_broad_only",
        "P7_U3_DA3_D4RT_anchor_component_split_broad_only",
        "P7_U4_conservative_split_only_when_F2_conf_low",
        "P7_U5_no_split_when_F2_conf_high",
    ]
    all_rows: list[dict[str, Any]] = []
    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows, generated_masks, splits = _build_variant(variant, parent_rows, source_scope, da3_maps)
        variant_scope = dict(source_scope)
        mask_map = dict(source_scope["mask_path_by_frame"])
        for key, path in generated_masks.items():
            scene, frame = key.split("|")
            mask_map[(scene, int(frame))] = path
        variant_scope["mask_path_by_frame"] = mask_map
        metrics, frames = p1._evaluate_variant(variant, rows, variant_scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
        all_rows.extend(rows)
        split_rows.extend(splits)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase7_variant_config_v1",
                "phase_id": "v99_phase7_broad_mask_split",
                "variant_id": variant,
                "emitted_rows": len(rows),
                "split_component_rows": len(splits),
                "generated_mask_frame_count": len(generated_masks),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)
    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    best_real = max([row for row in aggregate_rows if row["variant_id"] != "P7_B0_phase2_best_no_split"], key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"])))
    baseline = next(row for row in aggregate_rows if row["variant_id"] == "P7_B0_phase2_best_no_split")
    broad_keys: set[tuple[str, int, int]] = set()
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    for row in parent_rows:
        scene = str(row["scene_id"])
        frame = int(row["frame_id"])
        mask_id = int(row["selected_mask_id"])
        key = (scene, frame)
        if key not in label_cache:
            mask_path = source_scope["mask_path_by_frame"].get(key)
            label_cache[key] = p1._read_label(mask_path) if mask_path is not None and mask_path.exists() else np.zeros((1, 1), dtype=np.int64)
        label = label_cache[key]
        area_ratio = float(np.count_nonzero(label == mask_id) / max(1, label.size))
        if area_ratio >= 0.25:
            broad_keys.add((scene, frame, mask_id))
    broad_mask_count = len(broad_keys)
    split_attempt_count = sum(int(row["split_component_rows"]) for row in config_rows)
    split_success_count = split_attempt_count // 2
    broad_success = bool(float(best_real["MV_AP_window"]) >= base_window + 0.003 and float(best_real["MV_AP_window"]) >= float(baseline["MV_AP_window"]) - 0.001)
    safety_pass = bool(
        int(best_real["same_frame_collision_count"]) == 0
        and float(best_real["pixel_collision_rate"]) <= 0.02
        and int(best_real["missing_mask_raster_count"]) == 0
    )
    phase7_pass = bool(broad_success and safety_pass)
    gate_rows = [
        {
            "gate_id": "overall_MV_AP_window_ge_F2_base_plus_0p003",
            "pass": float(best_real["MV_AP_window"]) >= base_window + 0.003,
            "expected": f">={base_window + 0.003}",
            "observed": best_real["MV_AP_window"],
            "severity": "plan_success",
        },
        {
            "gate_id": "normal_case_no_large_drop_proxy",
            "pass": float(best_real["MV_AP_window"]) >= float(baseline["MV_AP_window"]) - 0.001,
            "expected": f">={float(baseline['MV_AP_window']) - 0.001}",
            "observed": best_real["MV_AP_window"],
            "severity": "required",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(best_real["same_frame_collision_count"]) == 0,
            "expected": "0",
            "observed": best_real["same_frame_collision_count"],
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(best_real["missing_mask_raster_count"]) == 0,
            "expected": "0",
            "observed": best_real["missing_mask_raster_count"],
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "split overfragments or fails AP; keep split disabled by default and use DA3 only as verifier",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase7_broad_mask_split_summary_v1",
        "phase_id": "v99_phase7_broad_mask_split",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_BROAD_MASK_SPLIT" if phase7_pass else "NO_GO_BROAD_MASK_SPLIT_KEEP_DISABLED",
        "phase7_pass": phase7_pass,
        "parent_phase2_variant": parent_variant,
        "best_real_variant": best_real["variant_id"],
        "best_real_MV_AP_window": float(best_real["MV_AP_window"]),
        "best_real_MV_AP50_window": float(best_real["MV_AP50_window"]),
        "best_real_MV_AP_scene": float(best_real["MV_AP_scene"]),
        "best_real_MV_AP50_scene": float(best_real["MV_AP50_scene"]),
        "baseline_MV_AP_window": float(baseline["MV_AP_window"]),
        "baseline_MV_AP_scene": float(baseline["MV_AP_scene"]),
        "split_attempt_count": split_attempt_count,
        "split_success_count": split_success_count,
        "split_component_count_mean": 2.0 if split_success_count else 0.0,
        "broad_mask_count": broad_mask_count,
        "underseg_case_improved_count": "",
        "overfragment_case_count": split_success_count if not phase7_pass else 0,
        "blocking_failure_count": len(failure_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "split_rows": _rel(OUT_DIR / "split_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "split_rows.csv", split_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase7_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
