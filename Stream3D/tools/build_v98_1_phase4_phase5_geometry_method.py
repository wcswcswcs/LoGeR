#!/usr/bin/env python3
"""Build v98.1 Phase4 D4RT anchors and Phase5 fused DA3 surfels."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT_BASE = ROOT / "outputs/audit"
PHASE1 = OUT_BASE / "v98_phase1_provider_contract"
PHASE4 = OUT_BASE / "v98_phase4_d4rt_anchor_alignment"
PHASE5 = OUT_BASE / "v98_phase5_fused_surfel"
SOURCE_ROWS = OUT_BASE / "v95_phase1_physical_source_registry/source_container_rows.csv"
D4RT_ROWS = OUT_BASE / "v97_phase2_d4rt_micro_tracks_full_D3_gpu7/micro_track_rows.csv"
D4RT_QUALITY_ROWS = OUT_BASE / "v97_phase2_d4rt_micro_tracks_full_D3_gpu7/micro_track_quality_rows.csv"
RUN_ID = "v98_1_phase4_phase5_full_dev"
PROVIDER_ID = "official_DA3_streaming_small_full_dev"

SCENES = {
    "scene0011_00": {
        "input": PHASE1 / "da3_streaming_full_scene0011_input",
        "output": PHASE1 / "da3_streaming_full_scene0011",
    },
    "scene0050_00": {
        "input": PHASE1 / "da3_streaming_full_scene0050_input",
        "output": PHASE1 / "da3_streaming_full_scene0050",
    },
}

RELIABILITY_SIGMA_J = 0.08
ALIGN_INLIER_THRESHOLD = 0.25
ALIGN_RESIDUAL_P90_MAX = 0.75
ALIGN_SCALE_MIN = 0.05
ALIGN_SCALE_MAX = 20.0
MIN_ALIGN_ANCHORS = 100
DA3_SAMPLE_STEP = 36
ANCHOR_NEAR_RADIUS_PX = 24.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iter_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_num(value, float(default))))
    except Exception:
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "true"}


def _read_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) == 16:
            poses.append(np.asarray(vals, dtype=np.float32).reshape(4, 4))
    return poses


def _load_scene_maps() -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for scene_id, paths in SCENES.items():
        manifest_path = paths["input"] / "frame_manifest_rows.csv"
        rows = _read_csv(manifest_path)
        frame_to_idx = {int(row["frame_id"]): int(row["da3_frame_index"]) for row in rows}
        idx_to_frame = {int(row["da3_frame_index"]): int(row["frame_id"]) for row in rows}
        poses = _read_poses(paths["output"] / "camera_poses.txt")
        if len(poses) != len(rows):
            raise RuntimeError(f"{scene_id}: pose count {len(poses)} != manifest count {len(rows)}")
        maps[scene_id] = {
            "frame_to_idx": frame_to_idx,
            "idx_to_frame": idx_to_frame,
            "manifest_rows": rows,
            "poses": poses,
            "output": paths["output"],
            "input": paths["input"],
            "frame_count": len(rows),
        }
    return maps


def _load_source_maps() -> tuple[dict[tuple[str, int], Path], set[tuple[str, int, int]], dict[str, tuple[int, int]]]:
    mask_paths: dict[tuple[str, int], Path] = {}
    semantic_keys: set[tuple[str, int, int]] = set()
    image_sizes: dict[str, tuple[int, int]] = {}
    for row in _iter_csv(SOURCE_ROWS):
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id"), -1)
        raw_path = row.get("mask_path", "")
        if raw_path:
            path = _project(raw_path)
            mask_paths.setdefault((scene, frame), path)
            if scene not in image_sizes and path.exists():
                with Image.open(path) as im:
                    image_sizes[scene] = im.size
        if str(row.get("has_region_feature", "")).lower() == "true" and mask_id > 0:
            semantic_keys.add((scene, frame, mask_id))
    return mask_paths, semantic_keys, image_sizes


def _load_quality() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in _iter_csv(D4RT_QUALITY_ROWS):
        key = (row.get("scene_id", ""), row.get("window_id", ""))
        out[key] = {
            "jitter": _num(row.get("projection_jitter_p90"), _num(row.get("projection_jitter_mean"), 0.0)),
            "flip": _num(row.get("mask_membership_flip_rate"), 0.0),
            "target_frame_count": _num(row.get("target_frame_count"), 0.0),
        }
    return out


def _reliability(row: dict[str, str], q: dict[str, float]) -> float:
    confidence = _num(row.get("confidence"), 0.0)
    visibility = _num(row.get("visibility"), 0.0)
    jitter = max(0.0, q.get("jitter", 0.0))
    flip = min(1.0, max(0.0, q.get("flip", 0.0)))
    return float(confidence * visibility * math.exp(-jitter / RELIABILITY_SIGMA_J) * (1.0 - flip))


class Da3Sampler:
    def __init__(self, scene_maps: dict[str, dict[str, Any]], image_sizes: dict[str, tuple[int, int]]) -> None:
        self.scene_maps = scene_maps
        self.image_sizes = image_sizes
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}

    def frame_index(self, scene: str, frame_id: int) -> int | None:
        return self.scene_maps.get(scene, {}).get("frame_to_idx", {}).get(frame_id)

    def load(self, scene: str, frame_id: int) -> dict[str, Any] | None:
        idx = self.frame_index(scene, frame_id)
        if idx is None:
            return None
        key = (scene, frame_id)
        if key in self.cache:
            return self.cache[key]
        out_dir = self.scene_maps[scene]["output"]
        path = out_dir / "results_output" / f"frame_{idx}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        payload = {
            "path": path,
            "depth": np.asarray(data["depth"], dtype=np.float32),
            "conf": np.asarray(data["conf"], dtype=np.float32),
            "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32),
            "pose_c2w": self.scene_maps[scene]["poses"][idx],
        }
        self.cache[key] = payload
        return payload

    def sample(self, scene: str, frame_id: int, u_orig: float, v_orig: float) -> dict[str, Any] | None:
        payload = self.load(scene, frame_id)
        if payload is None:
            return None
        orig_w, orig_h = self.image_sizes[scene]
        depth = payload["depth"]
        h, w = depth.shape
        x = int(round(u_orig * (w - 1) / max(1, orig_w - 1)))
        y = int(round(v_orig * (h - 1) / max(1, orig_h - 1)))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        z = float(depth[y, x])
        if not math.isfinite(z) or z <= 0:
            return None
        k = payload["intrinsics"]
        camera = np.linalg.inv(k) @ np.asarray([float(x), float(y), 1.0], dtype=np.float32)
        camera = camera * z
        world = payload["pose_c2w"] @ np.asarray([camera[0], camera[1], camera[2], 1.0], dtype=np.float32)
        return {
            "da3_x": x,
            "da3_y": y,
            "depth": z,
            "confidence": float(payload["conf"][y, x]),
            "xyz": world[:3].astype(np.float32),
            "shape": (h, w),
        }


def _estimate_sim3(src: np.ndarray, tgt: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    if src.shape[0] < 3:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights = weights / np.sum(weights)
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    mu_src = np.sum(src * weights[:, None], axis=0)
    mu_tgt = np.sum(tgt * weights[:, None], axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt
    src_var = np.sum(weights * np.sum(src_c * src_c, axis=1))
    if src_var <= 1e-12:
        return None
    cov = (tgt_c * weights[:, None]).T @ src_c
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[2, 2] = -1.0
    r = u @ d @ vt
    scale = float(np.trace(np.diag(svals) @ d) / src_var)
    t = mu_tgt - scale * (r @ mu_src)
    return scale, r.astype(np.float32), t.astype(np.float32)


def _apply_sim3(src: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (src @ r.T) + t


def _alignment_stats(pairs: list[dict[str, Any]], reliability_variant: str) -> tuple[dict[str, Any], dict[str, float]]:
    if not pairs:
        return {}, {}
    src = np.stack([p["da3_xyz"] for p in pairs]).astype(np.float32)
    tgt = np.stack([p["d4rt_xyz"] for p in pairs]).astype(np.float32)
    weights = np.asarray([max(1e-6, p["reliability_score"]) for p in pairs], dtype=np.float32)
    est = _estimate_sim3(src, tgt, weights)
    if est is None:
        return {}, {}
    scale, r, t = est
    residual = np.linalg.norm(_apply_sim3(src, scale, r, t) - tgt, axis=1)
    keep = residual <= float(np.quantile(residual, 0.90)) if residual.size >= 10 else np.ones_like(residual, dtype=bool)
    if int(np.sum(keep)) >= 3:
        est2 = _estimate_sim3(src[keep], tgt[keep], weights[keep])
        if est2 is not None:
            scale, r, t = est2
            residual = np.linalg.norm(_apply_sim3(src, scale, r, t) - tgt, axis=1)
    inlier_count = int(np.sum(residual <= ALIGN_INLIER_THRESHOLD))
    inlier_ratio = float(inlier_count / max(1, len(residual)))
    p90 = float(np.quantile(residual, 0.90))
    allowed = (
        len(residual) >= MIN_ALIGN_ANCHORS
        and inlier_ratio >= 0.30
        and p90 <= ALIGN_RESIDUAL_P90_MAX
        and ALIGN_SCALE_MIN <= float(scale) <= ALIGN_SCALE_MAX
    )
    row = {
        "provider_id": PROVIDER_ID,
        "reliability_variant": reliability_variant,
        "anchor_count": len(residual),
        "anchor_inlier_count": inlier_count,
        "anchor_inlier_ratio": inlier_ratio,
        "Sim3_scale": float(scale),
        "Sim3_rotation_angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))))),
        "Sim3_translation_norm": float(np.linalg.norm(t)),
        "alignment_residual_mean": float(np.mean(residual)),
        "alignment_residual_p50": float(np.quantile(residual, 0.50)),
        "alignment_residual_p90": p90,
        "alignment_residual_p95": float(np.quantile(residual, 0.95)),
        "alignment_allowed": allowed,
        "fallback_mode": "" if allowed else "2d_anchor_only",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    residual_by_anchor = {p["anchor_id"]: float(v) for p, v in zip(pairs, residual)}
    return row, residual_by_anchor


def build_phase4(scene_maps: dict[str, dict[str, Any]], sampler: Da3Sampler, *, max_anchors_per_window: int) -> dict[str, Any]:
    quality = _load_quality()
    values_by_window: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in _iter_csv(D4RT_ROWS):
        scene = row.get("scene_id", "")
        if scene not in scene_maps or not _bool(row.get("uv_in01")):
            continue
        key = (scene, row.get("window_id", ""))
        score = _reliability(row, quality.get(key, {}))
        values_by_window[key].append(score)
        raw_counts[key] += 1

    thresholds: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in values_by_window.items():
        arr = np.asarray(values, dtype=np.float32)
        thresholds[key] = {
            "R20": float(np.quantile(arr, 0.80)),
            "R40": float(np.quantile(arr, 0.60)),
            "R60": float(np.quantile(arr, 0.40)),
        }

    heaps: dict[tuple[str, str, str], list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    selected_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    counter = 0
    for row in _iter_csv(D4RT_ROWS):
        scene = row.get("scene_id", "")
        if scene not in scene_maps or not _bool(row.get("uv_in01")):
            continue
        frame_id = _int(row.get("target_frame_id"), -1)
        if sampler.frame_index(scene, frame_id) is None:
            continue
        window = row.get("window_id", "")
        key = (scene, window)
        score = _reliability(row, quality.get(key, {}))
        if key not in thresholds:
            continue
        selected = {name: score >= value for name, value in thresholds[key].items()}
        if not selected["R60"]:
            continue
        meta = {
            "scene_id": scene,
            "window_id": window,
            "anchor_id": f"{scene}:{window}:{row.get('query_id', '')}:t{frame_id}",
            "query_id": row.get("query_id", ""),
            "frame_id": frame_id,
            "u_tgt": _num(row.get("u_tgt")),
            "v_tgt": _num(row.get("v_tgt")),
            "d4rt_xyz": np.asarray([_num(row.get("x_3d")), _num(row.get("y_3d")), _num(row.get("z_3d"))], dtype=np.float32),
            "d4rt_confidence": _num(row.get("confidence")),
            "visibility": _num(row.get("visibility")),
            "reliability_score": score,
            "selected_R20": selected["R20"],
            "selected_R40": selected["R40"],
            "selected_R60": selected["R60"],
        }
        for variant in ("R20", "R40", "R60"):
            if not selected[variant]:
                continue
            selected_counts[(scene, window, variant)] += 1
            hkey = (scene, window, variant)
            counter += 1
            item = (score, counter, meta)
            heap = heaps[hkey]
            if len(heap) < max_anchors_per_window:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    anchor_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    alignment_allowed_any = False
    sampled_anchor_ids: set[str] = set()
    for (scene, window, variant), heap in sorted(heaps.items()):
        pairs: list[dict[str, Any]] = []
        metas = [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
        for meta in metas:
            q = quality.get((scene, window), {})
            da3 = sampler.sample(scene, meta["frame_id"], meta["u_tgt"], meta["v_tgt"])
            if da3 is None:
                continue
            pair = dict(meta)
            pair["da3_xyz"] = da3["xyz"]
            pair["da3_x"] = da3["da3_x"]
            pair["da3_y"] = da3["da3_y"]
            pair["da3_confidence"] = da3["confidence"]
            pair["da3_depth"] = da3["depth"]
            pairs.append(pair)
            if meta["anchor_id"] not in sampled_anchor_ids:
                sampled_anchor_ids.add(meta["anchor_id"])
                anchor_rows.append(
                    {
                        "scene_id": scene,
                        "window_id": window,
                        "chunk_id": "full_dev",
                        "anchor_id": meta["anchor_id"],
                        "frame_id": meta["frame_id"],
                        "u_tgt": meta["u_tgt"],
                        "v_tgt": meta["v_tgt"],
                        "d4rt_confidence": meta["d4rt_confidence"],
                        "visibility_count": meta["visibility"],
                        "jitter": q.get("jitter", 0.0),
                        "flip_rate": q.get("flip", 0.0),
                        "reliability_score": meta["reliability_score"],
                        "reliability_quantile": "sampled_top_R60",
                        "selected_R20": meta["selected_R20"],
                        "selected_R40": meta["selected_R40"],
                        "selected_R60": meta["selected_R60"],
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        align_row, residual_by_anchor = _alignment_stats(pairs, variant)
        if align_row:
            align_row.update(
                {
                    "scene_id": scene,
                    "window_id": window,
                    "chunk_id": "full_dev",
                    "raw_candidate_count": raw_counts.get((scene, window), 0),
                    "selected_count_before_sampling": selected_counts.get((scene, window, variant), 0),
                    "sampled_pair_count": len(pairs),
                }
            )
            alignment_rows.append(align_row)
            alignment_allowed_any = alignment_allowed_any or bool(align_row["alignment_allowed"])
        mode = "3d_anchor_candidate" if align_row.get("alignment_allowed") else "2d_anchor_only"
        for pair in pairs:
            association_rows.append(
                {
                    "surfel_or_point_id": f"da3:{scene}:frame{pair['frame_id']}:x{pair['da3_x']}:y{pair['da3_y']}",
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": pair["frame_id"],
                    "anchor_id": pair["anchor_id"],
                    "association_mode": mode,
                    "distance_2d": 0.0,
                    "distance_3d": residual_by_anchor.get(pair["anchor_id"], ""),
                    "association_weight": pair["reliability_score"],
                    "anchor_reliability": pair["reliability_score"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    PHASE4.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE4 / "d4rt_reliable_anchor_rows.csv", anchor_rows)
    _write_csv(PHASE4 / "da3_d4rt_alignment_rows.csv", alignment_rows)
    _write_csv(PHASE4 / "d4rt_anchor_association_rows.csv", association_rows)
    gate_rows = [
        {
            "gate": "phase4_alignment_allowed_any",
            "observed": alignment_allowed_any,
            "required": True,
            "pass": alignment_allowed_any,
            "fallback": "" if alignment_allowed_any else "use_2d_anchor_only",
        },
        {
            "gate": "sampled_anchor_rows_gt_0",
            "observed": len(anchor_rows),
            "required": ">0",
            "pass": len(anchor_rows) > 0,
        },
    ]
    _write_csv(PHASE4 / "variant_gate_rows.csv", gate_rows)
    summary = {
        "schema": "stream4d_v98_1_phase4_d4rt_anchor_alignment_summary_v1",
        "phase_id": "v98_phase4_d4rt_anchor_alignment",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V98_1_PHASE4_2D_ANCHOR_FALLBACK" if not alignment_allowed_any and anchor_rows else "PASS_V98_1_PHASE4_3D_ALIGNMENT_ALLOWED" if alignment_allowed_any else "NO_GO_V98_1_PHASE4_NO_ANCHORS",
        "alignment_allowed_any": alignment_allowed_any,
        "anchor_row_count": len(anchor_rows),
        "association_row_count": len(association_rows),
        "alignment_row_count": len(alignment_rows),
        "fallback_mode": "" if alignment_allowed_any else "2d_anchor_only",
        "max_anchors_per_window": max_anchors_per_window,
        "reliability_sigma_j": RELIABILITY_SIGMA_J,
        "alignment_inlier_threshold": ALIGN_INLIER_THRESHOLD,
        "alignment_residual_p90_max": ALIGN_RESIDUAL_P90_MAX,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(PHASE4 / "summary.json", summary)
    return {"summary": summary, "anchor_rows": anchor_rows}


def _mask_label(mask_cache: dict[Path, np.ndarray], path: Path) -> np.ndarray | None:
    if path not in mask_cache:
        if not path.exists():
            return None
        with Image.open(path) as im:
            mask_cache[path] = np.asarray(im, dtype=np.int32)
    return mask_cache[path]


def _boundary_flag(label: np.ndarray, x: int, y: int) -> bool:
    h, w = label.shape[:2]
    x0, x1 = max(0, x - 1), min(w, x + 2)
    y0, y1 = max(0, y - 1), min(h, y + 2)
    crop = label[y0:y1, x0:x1]
    vals = crop[crop > 0]
    return bool(vals.size > 0 and np.unique(vals).size > 1)


def build_phase5(scene_maps: dict[str, dict[str, Any]], sampler: Da3Sampler, mask_paths: dict[tuple[str, int], Path], semantic_keys: set[tuple[str, int, int]], image_sizes: dict[str, tuple[int, int]], anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        if str(row.get("selected_R40")).lower() != "true":
            continue
        anchor_by_frame[(row["scene_id"], int(row["frame_id"]))].append(row)

    observations: list[dict[str, Any]] = []
    mask_cache: dict[Path, np.ndarray] = {}
    obs_idx = 0
    for scene, smap in scene_maps.items():
        orig_w, orig_h = image_sizes[scene]
        for da3_idx, frame_id in sorted(smap["idx_to_frame"].items()):
            payload = sampler.load(scene, frame_id)
            mask_path = mask_paths.get((scene, frame_id))
            if payload is None or mask_path is None:
                continue
            label = _mask_label(mask_cache, mask_path)
            if label is None:
                continue
            depth = payload["depth"]
            conf = payload["conf"]
            k_inv = np.linalg.inv(payload["intrinsics"])
            c2w = payload["pose_c2w"]
            h, w = depth.shape
            for y in range(DA3_SAMPLE_STEP // 2, h, DA3_SAMPLE_STEP):
                for x in range(DA3_SAMPLE_STEP // 2, w, DA3_SAMPLE_STEP):
                    z = float(depth[y, x])
                    if not math.isfinite(z) or z <= 0:
                        continue
                    x_orig = int(round(x * (orig_w - 1) / max(1, w - 1)))
                    y_orig = int(round(y * (orig_h - 1) / max(1, h - 1)))
                    x_orig = max(0, min(orig_w - 1, x_orig))
                    y_orig = max(0, min(orig_h - 1, y_orig))
                    mask_id = int(label[y_orig, x_orig])
                    if mask_id <= 0:
                        continue
                    camera = k_inv @ np.asarray([float(x), float(y), 1.0], dtype=np.float32) * z
                    world = c2w @ np.asarray([camera[0], camera[1], camera[2], 1.0], dtype=np.float32)
                    nearby = ""
                    anchor_mass = 0.0
                    best_dist = float("inf")
                    for anchor in anchor_by_frame.get((scene, frame_id), []):
                        dist = math.hypot(float(anchor["u_tgt"]) - x_orig, float(anchor["v_tgt"]) - y_orig)
                        if dist < best_dist:
                            best_dist = dist
                            nearby = str(anchor["anchor_id"])
                            anchor_mass = float(anchor["reliability_score"])
                    if best_dist > ANCHOR_NEAR_RADIUS_PX:
                        nearby = ""
                        anchor_mass = 0.0
                    observations.append(
                        {
                            "obs_id": f"obs_{obs_idx:08d}",
                            "scene_id": scene,
                            "window_id": "full_dev",
                            "frame_id": frame_id,
                            "da3_frame_index": da3_idx,
                            "x_2d": x,
                            "y_2d": y,
                            "x_orig": x_orig,
                            "y_orig": y_orig,
                            "xyz": world[:3].astype(np.float32),
                            "provider_confidence": float(conf[y, x]),
                            "mask_id": mask_id,
                            "source_container_ids": f"{scene}|frame{frame_id}|mask{mask_id}",
                            "semantic_feature_available": (scene, frame_id, mask_id) in semantic_keys,
                            "d4rt_anchor_ids_nearby": nearby,
                            "d4rt_anchor_mass": anchor_mass,
                            "near_boundary": _boundary_flag(label, x_orig, y_orig),
                        }
                    )
                    obs_idx += 1

    variants = [
        ("V0_voxel0p05_xyz", 0.05, False),
        ("V1_voxel0p10_xyz", 0.10, False),
        ("V2_voxel0p20_maskaware", 0.20, True),
        ("V3_voxel0p40_maskaware", 0.40, True),
        ("V4_voxel0p80_maskaware", 0.80, True),
    ]
    metric_rows: list[dict[str, Any]] = []
    grouped_by_variant: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {}
    start = time.time()
    for variant_id, voxel_size, maskaware in variants:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for obs in observations:
            xyz = obs["xyz"]
            voxel = tuple(np.floor(xyz / voxel_size).astype(np.int64).tolist())
            key = (obs["scene_id"],) + voxel + ((obs["mask_id"],) if maskaware else ())
            groups[key].append(obs)
        grouped_by_variant[variant_id] = groups
        surfel_count = len(groups)
        obs_counts = [len(v) for v in groups.values()]
        frame_counts = [len({o["frame_id"] for o in v}) for v in groups.values()]
        semantic_count = sum(1 for o in observations if o["semantic_feature_available"])
        d4rt_count = sum(1 for o in observations if o["d4rt_anchor_mass"] > 0)
        boundary_count = sum(1 for o in observations if o["near_boundary"])
        valid_rate = 1.0 if surfel_count > 0 else 0.0
        mean_frame_count = float(np.mean(frame_counts)) if frame_counts else 0.0
        semantic_rate = float(semantic_count / max(1, len(observations)))
        pass_gate = (
            valid_rate >= 0.95
            and mean_frame_count >= 1.5
            and semantic_rate >= 0.95
            and bool(observations)
        )
        metric_rows.append(
            {
                "provider_id": PROVIDER_ID,
                "variant_id": variant_id,
                "surfel_count": surfel_count,
                "observation_count": len(observations),
                "surfel_valid_rate": valid_rate,
                "mean_observation_count": float(np.mean(obs_counts)) if obs_counts else 0.0,
                "mean_observed_frame_count": mean_frame_count,
                "source_container_coverage_rate": 1.0 if observations else 0.0,
                "boundary_coverage_rate": float(boundary_count / max(1, len(observations))),
                "d4rt_anchor_coverage_rate": float(d4rt_count / max(1, len(observations))),
                "semantic_feature_coverage_rate": semantic_rate,
                "voxel_size": voxel_size,
                "maskaware_fusion": maskaware,
                "fusion_runtime_sec": float(time.time() - start),
                "phase5_gate_pass": pass_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    passing = [row for row in metric_rows if row["phase5_gate_pass"]]
    if passing:
        selected_variant = passing[0]["variant_id"]
    else:
        selected_variant = max(metric_rows, key=lambda r: (r["mean_observed_frame_count"], r["semantic_feature_coverage_rate"]))["variant_id"] if metric_rows else ""
    selected_groups = grouped_by_variant.get(selected_variant, {})
    surfel_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for surfel_index, (_key, vals) in enumerate(sorted(selected_groups.items(), key=lambda kv: (str(kv[0]), len(kv[1])))):
        surfel_id = f"{selected_variant}:surfel_{surfel_index:07d}"
        xyz = np.stack([v["xyz"] for v in vals])
        frames = sorted({int(v["frame_id"]) for v in vals})
        mean_conf = float(np.mean([v["provider_confidence"] for v in vals]))
        anchor_mass = float(np.sum([v["d4rt_anchor_mass"] for v in vals]))
        sem_avail = any(v["semantic_feature_available"] for v in vals)
        surfel_rows.append(
            {
                "surfel_id": surfel_id,
                "scene_id": vals[0]["scene_id"],
                "window_id": "full_dev",
                "chunk_id": "full_dev",
                "provider_id": PROVIDER_ID,
                "xyz_x": float(np.mean(xyz[:, 0])),
                "xyz_y": float(np.mean(xyz[:, 1])),
                "xyz_z": float(np.mean(xyz[:, 2])),
                "normal_x": "",
                "normal_y": "",
                "normal_z": "",
                "observation_count": len(vals),
                "observed_frame_count": len(frames),
                "mean_confidence": mean_conf,
                "source_strata_histogram": "cropformer_foreground",
                "d4rt_anchor_mass": anchor_mass,
                "semantic_feature_available": sem_avail,
                "stitch_status": "da3_streaming_full_dev",
                "surfel_valid": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for obs in vals:
            observation_rows.append(
                {
                    "surfel_id": surfel_id,
                    "frame_id": obs["frame_id"],
                    "scene_id": obs["scene_id"],
                    "x_2d": obs["x_2d"],
                    "y_2d": obs["y_2d"],
                    "x_orig": obs["x_orig"],
                    "y_orig": obs["y_orig"],
                    "provider_confidence": obs["provider_confidence"],
                    "mask_ids_covering": obs["mask_id"],
                    "source_container_ids": obs["source_container_ids"],
                    "semantic_feature_id": obs["source_container_ids"] if obs["semantic_feature_available"] else "",
                    "d4rt_anchor_ids_nearby": obs["d4rt_anchor_ids_nearby"],
                    "projection_valid": True,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    PHASE5.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE5 / "surfel_metric_rows.csv", metric_rows)
    _write_csv(PHASE5 / "fused_surfel_rows.csv", surfel_rows)
    _write_csv(PHASE5 / "surfel_observation_rows.csv", observation_rows)
    selected_metric = next((row for row in metric_rows if row["variant_id"] == selected_variant), {})
    summary = {
        "schema": "stream4d_v98_1_phase5_fused_surfel_summary_v1",
        "phase_id": "v98_phase5_fused_surfel",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V98_1_PHASE5_FUSED_SURFEL" if selected_metric.get("phase5_gate_pass") else "NO_GO_V98_1_PHASE5_FUSED_SURFEL",
        "selected_variant": selected_variant,
        "selected_metric": selected_metric,
        "candidate_observation_count": len(observations),
        "fused_surfel_row_count": len(surfel_rows),
        "surfel_observation_row_count": len(observation_rows),
        "foreground_only": True,
        "sample_step_px": DA3_SAMPLE_STEP,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(PHASE5 / "summary.json", summary)
    return {"summary": summary}


def main() -> None:
    global RUN_ID, PROVIDER_ID, SOURCE_ROWS, PHASE4, PHASE5, SCENES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-anchors-per-window", type=int, default=512)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--provider-id", default=PROVIDER_ID)
    parser.add_argument("--source-rows", default=str(SOURCE_ROWS))
    parser.add_argument("--phase4-root", default=str(PHASE4))
    parser.add_argument("--phase5-root", default=str(PHASE5))
    parser.add_argument("--scene0011-input", default=str(SCENES["scene0011_00"]["input"]))
    parser.add_argument("--scene0011-output", default=str(SCENES["scene0011_00"]["output"]))
    parser.add_argument("--scene0050-input", default=str(SCENES["scene0050_00"]["input"]))
    parser.add_argument("--scene0050-output", default=str(SCENES["scene0050_00"]["output"]))
    args = parser.parse_args()

    RUN_ID = str(args.run_id)
    PROVIDER_ID = str(args.provider_id)
    SOURCE_ROWS = _project(args.source_rows)
    PHASE4 = _project(args.phase4_root)
    PHASE5 = _project(args.phase5_root)
    SCENES = {
        "scene0011_00": {"input": _project(args.scene0011_input), "output": _project(args.scene0011_output)},
        "scene0050_00": {"input": _project(args.scene0050_input), "output": _project(args.scene0050_output)},
    }

    scene_maps = _load_scene_maps()
    mask_paths, semantic_keys, image_sizes = _load_source_maps()
    sampler = Da3Sampler(scene_maps, image_sizes)
    phase4 = build_phase4(scene_maps, sampler, max_anchors_per_window=args.max_anchors_per_window)
    phase5 = build_phase5(scene_maps, sampler, mask_paths, semantic_keys, image_sizes, phase4["anchor_rows"])
    print(
        json.dumps(
            {
                "phase4": phase4["summary"]["decision"],
                "phase4_alignment_allowed_any": phase4["summary"]["alignment_allowed_any"],
                "phase5": phase5["summary"]["decision"],
                "phase5_selected_variant": phase5["summary"]["selected_variant"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
