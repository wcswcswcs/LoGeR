#!/usr/bin/env python3
"""Evaluate DA3 geometry as a verifier on top of the Phase2 F2 rows."""

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


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase4_f2_da3_link_verifier"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE2_SUMMARY = PHASE2_DIR / "best_variant_summary.json"
PHASE3_SUMMARY = AUDIT_ROOT / "v99_phase3_da3_protocol/summary.json"
EPS = 1e-4
DA3_SCENES = {
    "scene0011_00": {
        "input": AUDIT_ROOT / "v99_phase1_da3_chunk32o3_scene0011_base_input177",
        "output": AUDIT_ROOT / "v99_phase1_da3_chunk32o3_scene0011_base",
    },
    "scene0050_00": {
        "input": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_d4rt32o3_scene0050_input119",
        "output": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_d4rt32o3_scene0050_base_input119",
    },
}


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


def _read_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) == 16:
            poses.append(np.asarray(vals, dtype=np.float32).reshape(4, 4))
    return poses


def _load_da3_maps() -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for scene, spec in DA3_SCENES.items():
        manifest = _read_csv(Path(spec["input"]) / "frame_manifest_rows.csv")
        maps[scene] = {
            "frame_to_idx": {int(row["frame_id"]): int(row["da3_frame_index"]) for row in manifest},
            "idx_to_frame": {int(row["da3_frame_index"]): int(row["frame_id"]) for row in manifest},
            "poses": _read_poses(Path(spec["output"]) / "camera_poses.txt"),
            "output": Path(spec["output"]),
            "input": Path(spec["input"]),
        }
    return maps


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no rows for Phase2 best variant {variant}")
    return variant, rows


def _mask_geometry(
    scene: str,
    frame: int,
    mask_id: int,
    *,
    da3_maps: dict[str, dict[str, Any]],
    mask_path_by_frame: dict[tuple[str, int], Path],
    cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (scene, frame, mask_id)
    if key in cache:
        return cache[key]
    empty = {
        "valid": False,
        "centroid": np.zeros(3, dtype=np.float32),
        "radius": 0.0,
        "mean_conf": 0.0,
        "point_count": 0,
    }
    if scene not in da3_maps or frame not in da3_maps[scene]["frame_to_idx"]:
        cache[key] = empty
        return empty
    mask_path = mask_path_by_frame.get((scene, frame))
    if mask_path is None or not mask_path.exists():
        cache[key] = empty
        return empty
    idx = da3_maps[scene]["frame_to_idx"][frame]
    npz_path = da3_maps[scene]["output"] / "results_output" / f"frame_{idx}.npz"
    if not npz_path.exists():
        cache[key] = empty
        return empty
    payload = np.load(npz_path)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    conf = np.asarray(payload["conf"], dtype=np.float32)
    intr = np.asarray(payload["intrinsics"], dtype=np.float32)
    pose = da3_maps[scene]["poses"][idx]
    label = p1._read_label(mask_path, depth.shape)
    inside = (label == int(mask_id)) & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(inside)
    if ys.size < 8:
        cache[key] = empty
        return empty
    if ys.size > 1024:
        stride = max(1, ys.size // 1024)
        ys = ys[::stride][:1024]
        xs = xs[::stride][:1024]
    z = depth[ys, xs]
    pix = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(z, dtype=np.float32)], axis=0)
    cam = np.linalg.inv(intr) @ pix
    cam = cam * z[None, :]
    homo = np.concatenate([cam, np.ones((1, cam.shape[1]), dtype=np.float32)], axis=0)
    world = (pose @ homo)[:3].T.astype(np.float32)
    centroid = np.mean(world, axis=0)
    radius = float(np.percentile(np.linalg.norm(world - centroid[None, :], axis=1), 75))
    out = {
        "valid": True,
        "centroid": centroid,
        "radius": radius,
        "mean_conf": float(np.mean(conf[ys, xs])),
        "point_count": int(ys.size),
    }
    cache[key] = out
    return out


def _build_link_metrics(rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    da3_maps = _load_da3_maps()
    geom_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    link_rows: list[dict[str, Any]] = []
    obj_scores: dict[str, list[float]] = defaultdict(list)
    obj_conflicts: dict[str, int] = defaultdict(int)
    for oid, vals in sorted(by_object.items()):
        vals_sorted = sorted(vals, key=lambda r: (str(r["scene_id"]), str(r["chunk_id"]), int(r["frame_id"]), int(r["selected_mask_id"])))
        for a, b in zip(vals_sorted[:-1], vals_sorted[1:]):
            if a["scene_id"] != b["scene_id"] or a["chunk_id"] != b["chunk_id"] or int(a["frame_id"]) == int(b["frame_id"]):
                continue
            ga = _mask_geometry(
                str(a["scene_id"]),
                int(a["frame_id"]),
                int(a["selected_mask_id"]),
                da3_maps=da3_maps,
                mask_path_by_frame=scope["mask_path_by_frame"],
                cache=geom_cache,
            )
            gb = _mask_geometry(
                str(b["scene_id"]),
                int(b["frame_id"]),
                int(b["selected_mask_id"]),
                da3_maps=da3_maps,
                mask_path_by_frame=scope["mask_path_by_frame"],
                cache=geom_cache,
            )
            confident = bool(ga["valid"] and gb["valid"] and ga["point_count"] >= 32 and gb["point_count"] >= 32)
            if confident:
                dist = float(np.linalg.norm(ga["centroid"] - gb["centroid"]))
                denom = max(0.10, float(ga["radius"]) + float(gb["radius"]) + 0.25)
                consistency = float(math.exp(-dist / denom))
                conf_mean = 0.5 * (float(ga["mean_conf"]) + float(gb["mean_conf"]))
            else:
                dist = ""
                consistency = 0.0
                conf_mean = 0.0
            conflict = bool(confident and consistency < 0.10)
            if confident:
                obj_scores[oid].append(consistency)
            if conflict:
                obj_conflicts[oid] += 1
            link_rows.append(
                {
                    "schema_version": "stream4d_v99_phase4_link_metric_v1",
                    "phase_id": "v99_phase4_f2_da3_link_verifier",
                    "mv_object_id": oid,
                    "scene_id": a["scene_id"],
                    "chunk_id": a["chunk_id"],
                    "frame_a": int(a["frame_id"]),
                    "mask_a": int(a["selected_mask_id"]),
                    "frame_b": int(b["frame_id"]),
                    "mask_b": int(b["selected_mask_id"]),
                    "da3_confident": confident,
                    "geometry_distance": dist,
                    "geometry_consistency": consistency,
                    "mean_da3_confidence": conf_mean,
                    "geo_conflict": conflict,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    obj_metric: dict[str, dict[str, float]] = {}
    for oid in by_object:
        vals = obj_scores.get(oid, [])
        obj_metric[oid] = {
            "mean_geometry_consistency": float(np.mean(vals)) if vals else 0.0,
            "confident_link_count": float(len(vals)),
            "conflict_count": float(obj_conflicts.get(oid, 0)),
        }
    return link_rows, obj_metric


def _norm_metric(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}


def _variant_rows(parent_rows: list[dict[str, Any]], obj_metric: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    geom_norm = _norm_metric({oid: vals["mean_geometry_consistency"] for oid, vals in obj_metric.items()})
    conflict_norm = _norm_metric({oid: vals["conflict_count"] for oid, vals in obj_metric.items()})
    variants = {
        "P4_B0_phase2_best_no_da3": ("no_da3_replay", lambda score, oid: score),
        "P4_B1_DA3_boost_only": ("phase2_score_plus_1e-4_da3_geometry_consistency", lambda score, oid: score + EPS * geom_norm.get(oid, 0.0)),
        "P4_B2_DA3_veto_only": ("phase2_score_minus_1e-4_da3_geometry_conflict", lambda score, oid: score - EPS * conflict_norm.get(oid, 0.0)),
        "P4_B3_DA3_boost_plus_veto": (
            "phase2_score_plus_1e-4_da3_geometry_minus_conflict",
            lambda score, oid: score + EPS * geom_norm.get(oid, 0.0) - EPS * conflict_norm.get(oid, 0.0),
        ),
        "P4_B4_DA3_uncertain_noop": ("phase2_score_da3_uncertain_noop", lambda score, oid: score),
    }
    out: list[dict[str, Any]] = []
    for variant, (policy, fn) in variants.items():
        for row in parent_rows:
            oid = str(row["mv_object_id"])
            score = _num(row.get("score"), 1.0)
            new = dict(row)
            new["variant_id"] = variant
            new["score"] = float(fn(score, oid))
            new["score_policy"] = policy
            new["phase4_parent_variant_id"] = row["variant_id"]
            new["da3_geometry_consistency"] = obj_metric.get(oid, {}).get("mean_geometry_consistency", 0.0)
            new["da3_confident_link_count"] = obj_metric.get(oid, {}).get("confident_link_count", 0.0)
            new["da3_conflict_count"] = obj_metric.get(oid, {}).get("conflict_count", 0.0)
            new["uses_future"] = False
            out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase2_summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    phase3_summary = json.loads(PHASE3_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase2_summary.get("phase2_full_pass")):
        raise RuntimeError("Phase4 requires Phase2 full pass")
    if not bool(phase3_summary.get("phase3_pass")):
        raise RuntimeError("Phase4 requires Phase3 pass")
    parent_variant, parent_rows = _phase2_best_rows()
    scope = p1._load_source_scope()
    link_rows, obj_metric = _build_link_metrics(parent_rows, scope)
    all_rows = _variant_rows(parent_rows, obj_metric)

    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant_id"] for row in all_rows}):
        rows = [row for row in all_rows if row["variant_id"] == variant]
        metrics, frames = p1._evaluate_variant(variant, rows, scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)
    confident_link_count = sum(1 for row in link_rows if row["da3_confident"])
    vetoed_link_count = sum(1 for row in link_rows if row["geo_conflict"])
    consistency_vals = [_num(row["geometry_consistency"]) for row in link_rows if row["da3_confident"]]
    duplicate_tube_count = 0
    fragmentation_rate = 0.0
    for row in aggregate_rows:
        variant = str(row["variant_id"])
        row["link_count"] = len(link_rows)
        row["da3_confident_link_count"] = confident_link_count
        row["da3_boosted_link_count"] = confident_link_count if "boost" in variant else 0
        row["da3_vetoed_link_count"] = vetoed_link_count if "veto" in variant else 0
        row["da3_uncertain_link_count"] = len(link_rows) - confident_link_count
        row["link_geometry_consistency_mean"] = float(np.mean(consistency_vals)) if consistency_vals else 0.0
        row["cannot_link_count"] = vetoed_link_count
        row["fragmentation_rate"] = fragmentation_rate
        row["duplicate_tube_count"] = duplicate_tube_count
        row["uses_future"] = False

    phase2_window = float(phase2_summary["best_MV_AP_window"])
    phase2_scene = float(phase2_summary["best_MV_AP_scene"])
    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    base_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    real_rows = [row for row in aggregate_rows if row["variant_id"] in {"P4_B1_DA3_boost_only", "P4_B2_DA3_veto_only", "P4_B3_DA3_boost_plus_veto"}]
    best_real = max(real_rows, key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP50_window"]), float(row["MV_AP_scene"])))
    plan_local_success = bool(float(best_real["MV_AP_window"]) >= base_window + 0.005 and float(best_real["MV_AP50_window"]) >= base_ap50 + 0.010)
    plan_scene_success = bool(float(best_real["MV_AP_scene"]) >= base_scene + 0.010 and float(best_real["MV_AP_window"]) >= base_window - 0.003)
    improves_phase2_window = float(best_real["MV_AP_window"]) > phase2_window + 1e-12
    improves_phase2_scene = float(best_real["MV_AP_scene"]) > phase2_scene + 1e-12
    gate_rows = [
        {
            "gate_id": "DA3_plan_local_success_vs_F2_base",
            "pass": plan_local_success,
            "expected": f"MV_AP_window>={base_window + 0.005} and MV_AP50_window>={base_ap50 + 0.010}",
            "observed": f"MV_AP_window={best_real['MV_AP_window']}; MV_AP50_window={best_real['MV_AP50_window']}",
            "severity": "plan_success",
        },
        {
            "gate_id": "DA3_plan_scene_success_vs_F2_base",
            "pass": plan_scene_success,
            "expected": f"MV_AP_scene>={base_scene + 0.010} and MV_AP_window>={base_window - 0.003}",
            "observed": f"MV_AP_scene={best_real['MV_AP_scene']}; MV_AP_window={best_real['MV_AP_window']}",
            "severity": "plan_success_alternative",
        },
        {
            "gate_id": "DA3_increment_over_phase2_best_window_or_scene",
            "pass": improves_phase2_window or improves_phase2_scene,
            "expected": f"window>{phase2_window} or scene>{phase2_scene}",
            "observed": f"window={best_real['MV_AP_window']}; scene={best_real['MV_AP_scene']}",
            "severity": "contribution_required",
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
    blocking_failure_rows = []
    for row in gate_rows:
        if bool(row["pass"]):
            continue
        if row["gate_id"] == "DA3_plan_local_success_vs_F2_base" and plan_scene_success:
            continue
        if row["gate_id"] == "DA3_plan_scene_success_vs_F2_base" and plan_local_success:
            continue
        blocking_failure_rows.append(row)
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "continue with v99 DA3/D4RT verifier repair ladder; do not claim Phase4 contribution until this gate passes",
        }
        for row in blocking_failure_rows
    ]
    contribution_pass = bool((plan_local_success or plan_scene_success) and (improves_phase2_window or improves_phase2_scene))
    decision = "PASS_DA3_LINK_VERIFIER_CONTRIBUTION" if contribution_pass else "NO_GO_DA3_LINK_VERIFIER_DIAGNOSTIC_ONLY"
    casebook_rows = [
        {
            "schema_version": "stream4d_v99_phase4_casebook_v1",
            "phase_id": "v99_phase4_f2_da3_link_verifier",
            "rank": idx,
            "variant_id": row["variant_id"],
            "MV_AP_window": row["MV_AP_window"],
            "MV_AP50_window": row["MV_AP50_window"],
            "MV_AP_scene": row["MV_AP_scene"],
            "MV_AP50_scene": row["MV_AP50_scene"],
            "delta_vs_phase2_window": float(row["MV_AP_window"]) - phase2_window,
            "delta_vs_phase2_scene": float(row["MV_AP_scene"]) - phase2_scene,
            "link_geometry_consistency_mean": row["link_geometry_consistency_mean"],
            "da3_confident_link_count": row["da3_confident_link_count"],
        }
        for idx, row in enumerate(sorted(aggregate_rows, key=lambda r: float(r["MV_AP_window"]), reverse=True), start=1)
    ]
    summary = {
        "schema_version": "stream4d_v99_phase4_f2_da3_link_verifier_summary_v1",
        "phase_id": "v99_phase4_f2_da3_link_verifier",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "phase4_contribution_pass": contribution_pass,
        "parent_phase2_variant": parent_variant,
        "best_real_variant": best_real["variant_id"],
        "best_real_MV_AP_window": float(best_real["MV_AP_window"]),
        "best_real_MV_AP50_window": float(best_real["MV_AP50_window"]),
        "best_real_MV_AP_scene": float(best_real["MV_AP_scene"]),
        "best_real_MV_AP50_scene": float(best_real["MV_AP50_scene"]),
        "phase2_best_MV_AP_window": phase2_window,
        "phase2_best_MV_AP_scene": phase2_scene,
        "delta_best_real_vs_phase2_window": float(best_real["MV_AP_window"]) - phase2_window,
        "delta_best_real_vs_phase2_scene": float(best_real["MV_AP_scene"]) - phase2_scene,
        "link_count": len(link_rows),
        "da3_confident_link_count": confident_link_count,
        "da3_vetoed_link_count": vetoed_link_count,
        "link_geometry_consistency_mean": float(np.mean(consistency_vals)) if consistency_vals else 0.0,
        "blocking_failure_count": len(failure_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "link_metric_rows": _rel(OUT_DIR / "link_metric_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "link_metric_rows.csv", link_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if contribution_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
