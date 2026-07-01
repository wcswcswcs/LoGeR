from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_radio_checkpoint  # noqa: E402
from stream4d_native.v91_mask_feature_store import save_mask_feature_store  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402


SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
CANDIDATE_ROWS = ROOT / "outputs/audit/v71_candidate_bank/candidate_mask_rows.csv"
AVAILABILITY_JSON = ROOT / "outputs/audit/v46_loger_env_radio_radseg_availability_recheck_20260619/radio_vipe_availability.json"

FIELDS = [
    "scene_id",
    "chunk_id",
    "frame_id",
    "mask_id",
    "mask_observation_id",
    "semantic_backend",
    "feature_layer",
    "feature_resolution",
    "feature_pooling_method",
    "feature_available",
    "feature_norm",
    "feature_nan_count",
    "feature_dim",
    "semantic_prototype_id",
    "semantic_prototype_margin",
    "semantic_entropy",
    "semantic_intra_variance",
    "semantic_boundary_variance",
    "semantic_texture_score",
    "semantic_background_score_proxy",
    "used_token_count",
    "used_pixel_count",
    "broad_background_risk",
    "uses_gt_for_prediction",
    "feature_sha256",
    "feature_head_json",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _read_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return image.astype(np.int64, copy=False)


def _resize_binary(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.shape != (height, width):
        mask_u8 = cv2.resize(mask_u8, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    out = mask_u8.astype(bool)
    if not np.any(out) and np.any(mask):
        ys, xs = np.nonzero(mask)
        cy = int(np.clip(round(float(ys.mean()) * float(height) / max(mask.shape[0], 1)), 0, int(height) - 1))
        cx = int(np.clip(round(float(xs.mean()) * float(width) / max(mask.shape[1], 1)), 0, int(width) - 1))
        out[cy, cx] = True
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[1:, :] != mask[:-1, :]
    out[:-1, :] |= mask[:-1, :] != mask[1:, :]
    out[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    out[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return out & mask


def _feature_sha256(feature: np.ndarray) -> str:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _prototype_from_feature(feature: np.ndarray) -> tuple[str, float]:
    vec = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return "", 0.0
    order = np.argsort(np.abs(vec))[-4:][::-1]
    pieces = []
    for idx in order[:3]:
        sign = "p" if vec[int(idx)] >= 0 else "n"
        pieces.append(f"{sign}{int(idx)}")
    margin = 0.0
    if len(order) >= 2:
        margin = float(abs(vec[int(order[0])]) - abs(vec[int(order[1])]))
    return "radio|" + "|".join(pieces), margin


def _candidate_meta() -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in _read_csv(CANDIDATE_ROWS):
        scene = str(row.get("scene_id", ""))
        frame_id = adaptive._int(row.get("frame_id"), -1)
        mask_id = adaptive._int(row.get("mask_id"), -1)
        if scene and frame_id >= 0 and mask_id > 0:
            out[(scene, frame_id, mask_id)] = row
    return out


def _support_mask_index(scenes: set[str], support_rows_path: Path = SUPPORT_ROWS) -> dict[tuple[str, int], dict[int, dict[str, Any]]]:
    by_frame: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    candidate_meta = _candidate_meta()
    for row in _read_csv(support_rows_path):
        if adaptive._bool(row.get("uses_gt_for_prediction")) or adaptive._bool(row.get("uses_future")):
            continue
        if not adaptive._bool(row.get("native_support_allowed", "True")):
            continue
        scene = str(row.get("scene_id", ""))
        if scene not in scenes:
            continue
        frame_id = adaptive._int(row.get("frame_id"), -1)
        mask_id = adaptive._int(row.get("mask_id"), -1)
        if frame_id < 0 or mask_id <= 0:
            continue
        meta = dict(candidate_meta.get((scene, frame_id, mask_id), {}))
        meta.setdefault("scene_id", scene)
        meta.setdefault("frame_id", frame_id)
        meta.setdefault("mask_id", mask_id)
        meta.setdefault("chunk_id", row.get("chunk_id", ""))
        meta.setdefault("mask_observation_id", f"{scene}:{frame_id}:{mask_id}")
        by_frame[(scene, frame_id)][mask_id] = meta
    return by_frame


def _pool_mask_rows(
    *,
    adapter: FrozenFeatureAdapter,
    mask_dirs: dict[str, Path],
    scene: str,
    frame_id: int,
    masks_by_id: dict[int, dict[str, Any]],
    entropy_variance_scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    missing: list[dict[str, Any]] = []
    try:
        rgb = stream.load_rgb(int(frame_id))
    except FileNotFoundError as exc:
        return [], [], [], [{"scene_id": scene, "frame_id": int(frame_id), "missing": f"rgb:{exc}"}]
    label = _read_label(mask_dirs[scene] / f"{int(frame_id)}.png", rgb.shape[:2])
    if label is None:
        return [], [], [], [{"scene_id": scene, "frame_id": int(frame_id), "missing": "mask_png"}]
    try:
        feature_map = adapter.extract_dense_features(rgb)
    except Exception as exc:
        return [], [], [], [{"scene_id": scene, "frame_id": int(frame_id), "missing": f"feature_extract_failed:{type(exc).__name__}:{exc}"}]
    features = np.asarray(feature_map.features, dtype=np.float32)
    feature_h, feature_w, feature_dim = int(features.shape[0]), int(features.shape[1]), int(features.shape[2])
    rows: list[dict[str, Any]] = []
    feature_store_rows: list[dict[str, Any]] = []
    feature_vectors: list[np.ndarray] = []
    for mask_id, source_row in sorted(masks_by_id.items()):
        binary = label == int(mask_id)
        used_pixel_count = int(np.count_nonzero(binary))
        small = _resize_binary(binary, feature_h, feature_w)
        token_values = features[small]
        feature_available = bool(token_values.size > 0 and used_pixel_count >= 1)
        pooled = np.zeros((feature_dim,), dtype=np.float32)
        feature_norm = 0.0
        feature_nan_count = 0
        entropy: float | str = ""
        intra: float | str = ""
        boundary_variance: float | str = ""
        prototype_id = ""
        prototype_margin: float | str = ""
        used_token_count = int(token_values.shape[0]) if token_values.ndim == 2 else 0
        if feature_available:
            pooled = token_values.mean(axis=0).astype(np.float32)
            feature_nan_count = int(np.isnan(pooled).sum())
            if feature_nan_count == 0:
                raw_norm = float(np.linalg.norm(pooled))
                feature_norm = raw_norm
                if raw_norm > 1e-8:
                    pooled = pooled / raw_norm
                intra_value = float(np.mean(np.var(token_values.astype(np.float32), axis=0))) if token_values.shape[0] > 1 else 0.0
                boundary_small = _boundary(small)
                if np.any(boundary_small):
                    boundary_tokens = features[boundary_small]
                    boundary_variance_value = float(np.mean(np.var(boundary_tokens.astype(np.float32), axis=0))) if boundary_tokens.shape[0] > 1 else 0.0
                else:
                    boundary_variance_value = 0.0
                entropy_value = float(1.0 - math.exp(-intra_value / max(1e-12, entropy_variance_scale)))
                prototype_id, prototype_margin_value = _prototype_from_feature(pooled)
                entropy = entropy_value
                intra = intra_value
                boundary_variance = boundary_variance_value
                prototype_margin = prototype_margin_value
        broad = adaptive._bool(source_row.get("broad_background_risk")) or adaptive._num(source_row.get("area_ratio"), 0.0) >= 0.35
        row = {
            "scene_id": scene,
            "chunk_id": source_row.get("chunk_id", ""),
            "frame_id": int(frame_id),
            "mask_id": int(mask_id),
            "mask_observation_id": source_row.get("mask_observation_id", f"{scene}:{frame_id}:{mask_id}"),
            "semantic_backend": "radio_radseg",
            "feature_layer": "radio_radseg_spatial_features",
            "feature_resolution": f"{feature_h}x{feature_w}",
            "feature_pooling_method": "mask_token_mean",
            "feature_available": bool(feature_available and feature_nan_count == 0),
            "feature_norm": feature_norm,
            "feature_nan_count": feature_nan_count,
            "feature_dim": int(feature_dim),
            "semantic_prototype_id": prototype_id,
            "semantic_prototype_margin": prototype_margin,
            "semantic_entropy": entropy,
            "semantic_intra_variance": intra,
            "semantic_boundary_variance": boundary_variance,
            "semantic_texture_score": intra,
            "semantic_background_score_proxy": bool(broad),
            "used_token_count": used_token_count,
            "used_pixel_count": used_pixel_count,
            "broad_background_risk": bool(broad),
            "uses_gt_for_prediction": False,
            "feature_sha256": _feature_sha256(pooled) if feature_available and feature_nan_count == 0 else "",
            "feature_head_json": json.dumps([float(v) for v in pooled[:8]], separators=(",", ":")) if feature_available and feature_nan_count == 0 else "",
        }
        rows.append(row)
        if feature_available and feature_nan_count == 0:
            feature_store_rows.append(row)
            feature_vectors.append(pooled.astype(np.float32, copy=True))
    return rows, feature_store_rows, feature_vectors, missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    scenes = set(_parse_csv_list(args.scenes))
    mask_dirs = phase4._mask_dir_by_scene()
    support_rows_path = Path(args.support_rows)
    support_rows_path = support_rows_path if support_rows_path.is_absolute() else ROOT / support_rows_path
    support_index = _support_mask_index(scenes, support_rows_path)
    availability = _read_json(AVAILABILITY_JSON)
    checkpoint = str(args.checkpoint).strip() or str(availability.get("radio_checkpoint") or locate_default_radio_checkpoint() or "")
    adapter = FrozenFeatureAdapter(
        backend="radio_radseg",
        device=str(args.device),
        checkpoint=checkpoint,
        radio_lang_model=str(args.radio_lang_model),
        radio_lang_align=bool(args.radio_lang_align),
        radio_slide_crop=int(args.radio_slide_crop),
        radio_slide_stride=int(args.radio_slide_stride),
    )
    rows: list[dict[str, Any]] = []
    feature_store_rows: list[dict[str, Any]] = []
    feature_vectors: list[np.ndarray] = []
    missing_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frame_items = sorted(support_index.items())
    min_frame_by_scene: dict[str, int] = {}
    for item in _parse_csv_list(args.min_frame_exclusive_by_scene):
        if ":" not in item:
            continue
        scene, value = item.split(":", 1)
        min_frame_by_scene[scene] = adaptive._int(value, -1)
    if min_frame_by_scene:
        frame_items = [
            ((scene, frame_id), masks_by_id)
            for (scene, frame_id), masks_by_id in frame_items
            if int(frame_id) > int(min_frame_by_scene.get(scene, -1))
        ]
    max_frames = int(args.max_frames)
    if max_frames > 0:
        frame_items = frame_items[:max_frames]
    for index, ((scene, frame_id), masks_by_id) in enumerate(frame_items, start=1):
        if index % int(args.progress_every) == 0:
            print(f"[v91-radio] frame={index}/{len(frame_items)} scene={scene} frame_id={frame_id} masks={len(masks_by_id)}", file=sys.stderr, flush=True)
        frame_started = time.time()
        frame_feature_rows, frame_store_rows, frame_vectors, frame_missing = _pool_mask_rows(
            adapter=adapter,
            mask_dirs=mask_dirs,
            scene=scene,
            frame_id=int(frame_id),
            masks_by_id=masks_by_id,
            entropy_variance_scale=float(args.entropy_variance_scale),
        )
        rows.extend(frame_feature_rows)
        feature_store_rows.extend(frame_store_rows)
        feature_vectors.extend(frame_vectors)
        missing_rows.extend(frame_missing)
        frame_rows.append(
            {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "requested_mask_count": len(masks_by_id),
                "written_mask_count": len(frame_feature_rows),
                "missing_count": len(frame_missing),
                "runtime_sec": time.time() - frame_started,
                "uses_gt_for_prediction": False,
            }
        )
    valid = [row for row in rows if adaptive._bool(row.get("feature_available"))]
    entropy_values = [adaptive._num(row.get("semantic_entropy")) for row in valid if str(row.get("semantic_entropy", "")) != ""]
    broad_entropy = [adaptive._num(row.get("semantic_entropy")) for row in valid if adaptive._bool(row.get("broad_background_risk")) and str(row.get("semantic_entropy", "")) != ""]
    clean_entropy = [adaptive._num(row.get("semantic_entropy")) for row in valid if not adaptive._bool(row.get("broad_background_risk")) and str(row.get("semantic_entropy", "")) != ""]
    proto_counter = Counter(str(row.get("semantic_prototype_id", "")) for row in valid if row.get("semantic_prototype_id"))
    summary = {
        "phase": "v91_radio_mask_features",
        "schema": "stream4d_v91_radio_mask_features_v1",
        "semantic_backend": "radio_radseg",
        "scenes": sorted(scenes),
        "requested_frame_count": len(frame_items),
        "requested_unique_mask_observation_count": sum(len(v) for _k, v in frame_items),
        "mask_feature_row_count": len(rows),
        "valid_feature_count": len(valid),
        "feature_store_row_count": len(feature_store_rows),
        "semantic_feature_success_rate": float(len(valid) / max(1, len(rows))),
        "semantic_nan_rate": float(sum(1 for row in rows if adaptive._int(row.get("feature_nan_count"), 0) > 0) / max(1, len(rows))),
        "semantic_entropy_mean": float(np.mean(entropy_values)) if entropy_values else None,
        "semantic_entropy_p90": float(np.quantile(np.asarray(entropy_values, dtype=np.float32), 0.90)) if entropy_values else None,
        "diagnostic_broad_mask_Hsem_mean": float(np.mean(broad_entropy)) if broad_entropy else None,
        "diagnostic_clean_mask_Hsem_mean": float(np.mean(clean_entropy)) if clean_entropy else None,
        "semantic_prototype_count": len(proto_counter),
        "radio_checkpoint": checkpoint,
        "radio_lang_model": str(args.radio_lang_model),
        "radio_lang_align": bool(args.radio_lang_align),
        "radio_slide_crop": int(args.radio_slide_crop),
        "radio_slide_stride": int(args.radio_slide_stride),
        "radio_availability_artifact": adaptive._rel(AVAILABILITY_JSON),
        "support_rows": adaptive._rel(support_rows_path),
        "min_frame_exclusive_by_scene": min_frame_by_scene,
        "radio_available": bool(availability.get("radio_available")),
        "device": str(args.device),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    broad = summary["diagnostic_broad_mask_Hsem_mean"]
    clean = summary["diagnostic_clean_mask_Hsem_mean"]
    summary["diagnostic_broad_minus_clean_Hsem"] = None if broad is None or clean is None else float(broad - clean)
    complete = bool(rows) and len(valid) == len(rows) and not missing_rows
    summary["decision"] = "PASS_V91_RADIO_MASK_FEATURES" if complete else "PARTIAL_V91_RADIO_MASK_FEATURES"
    if feature_store_rows:
        store_manifest = save_mask_feature_store(
            output_dir=out,
            rows=feature_store_rows,
            features=np.stack(feature_vectors, axis=0).astype(np.float32),
            backend="radio_radseg",
            layer="radio_radseg_spatial_features",
            metadata={
                "source": "build_v91_radio_mask_features.py",
                "scenes": sorted(scenes),
                "checkpoint": checkpoint,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            },
        )
        summary["feature_store_manifest"] = adaptive._rel(out / "feature_store_manifest.json")
        summary["feature_store_npz"] = adaptive._rel(out / "mask_features.npz")
        summary["feature_store_sha256"] = store_manifest.get("store_sha256", "")
    else:
        summary["feature_store_manifest"] = ""
        summary["feature_store_npz"] = ""
        summary["feature_store_sha256"] = ""

    _write_csv(out / "mask_feature_rows.csv", rows, FIELDS)
    _write_csv(out / "frame_feature_rows.csv", frame_rows)
    _write_csv(out / "missing_input_rows.csv", missing_rows)
    _write_json(out / "semantic_summary.json", summary)
    outputs = [
        out / "mask_feature_rows.csv",
        out / "mask_feature_index.csv",
        out / "mask_features.npz",
        out / "feature_store_manifest.json",
        out / "frame_feature_rows.csv",
        out / "missing_input_rows.csv",
        out / "semantic_summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v91 RADIO/RADSeg mask features for support-readout masks.")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--support-rows", default=str(SUPPORT_ROWS))
    parser.add_argument("--min-frame-exclusive-by-scene", default="")
    parser.add_argument("--output-root", default="outputs/audit/v91_radio_mask_features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--radio-slide-crop", type=int, default=0)
    parser.add_argument("--radio-slide-stride", type=int, default=224)
    parser.add_argument("--entropy-variance-scale", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
