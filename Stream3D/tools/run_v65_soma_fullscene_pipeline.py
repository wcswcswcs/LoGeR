from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from stream4d_native.v47_carrier_observation_table import build_observation_tables
from stream4d_native.v47_common import ROOT, read_json, resolve_mask_dir, utc_now, write_csv, write_json
from stream4d_native.v53_chunk_universe import build_chunk_universe, write_chunk_universe
from stream4d_native.v53_local_objectlets import build_local_objectlets, write_local_objectlets
from stream4d_native.v53_mask_component_support import build_mask_component_support, write_mask_component_support
from stream4d_native.v53_representative_observations import (
    build_representative_observations,
    write_representative_observations,
)
from stream4d_native.v53_reprojection_ledger import build_reprojection_ledger, write_reprojection_ledger
from stream4d_native.v54_local_reproduction import build_v54_local_reproduction, write_v54_local_reproduction


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return ROOT.parent / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(ROOT.parent))
        except ValueError:
            return str(path_obj)


def _sha256(path: str | Path) -> str:
    path_obj = _project(path)
    digest = hashlib.sha256()
    with path_obj.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: str | Path) -> int:
    path_obj = _project(path)
    if not path_obj.exists():
        return 0
    with path_obj.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _row in reader)


def _csv_unique_frames(path: str | Path) -> list[int]:
    path_obj = _project(path)
    if not path_obj.exists():
        return []
    frames: set[int] = set()
    with path_obj.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("frame_id")
            if value not in (None, ""):
                frames.add(int(float(value)))
    return sorted(frames)


def _stage_file(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": _rel(path), "kind": kind, "exists": False}
    payload: dict[str, Any] = {
        "path": _rel(path),
        "kind": kind,
        "exists": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if kind == "csv":
        payload["row_count"] = _csv_row_count(path)
    return payload


def _read_summary_gate(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    gate = payload.get("gate") if isinstance(payload, dict) else None
    return gate if isinstance(gate, dict) else None


def _discover_numeric_stems(path: Path, suffix: str) -> list[int]:
    if not path.exists():
        return []
    out: list[int] = []
    for item in path.glob(f"*{suffix}"):
        try:
            out.append(int(item.stem))
        except ValueError:
            continue
    return sorted(set(out))


def _scene_stride_frames(scene: str, stride: int) -> list[int]:
    color_dir = ROOT / "data/scannet/processed" / scene / "color"
    color_ids = _discover_numeric_stems(color_dir, ".jpg")
    if not color_ids:
        return []
    start = min(color_ids)
    stop = max(color_ids)
    return [frame for frame in range(start, stop + 1, int(stride)) if frame in set(color_ids)]


def _float_token(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p").replace("-", "m")


def _repo_project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return ROOT.parent / path_obj
    return ROOT.parent / path_obj


def _cropformer_config_path() -> Path:
    return ROOT / "third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"


def _cropformer_weights_path() -> Path:
    return ROOT / "third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth"


def _d4rt_chunk_frame_specs(chunk_dir: Path, max_chunks: int = 0) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for chunk_index, path in enumerate(sorted(chunk_dir.glob("chunk_*.npz"), key=_chunk_sort_key)):
        if int(max_chunks) > 0 and chunk_index >= int(max_chunks):
            break
        with np.load(path) as payload:
            frame_ids = [int(value) for value in np.asarray(payload["frame_ids"], dtype=np.int64).tolist()]
        specs.append(
            {
                "chunk_index": int(chunk_index),
                "chunk_npz": path,
                "chunk_npz_sha256": _sha256(path),
                "frame_ids": frame_ids,
            }
        )
    if not specs:
        raise FileNotFoundError(f"No D4RT chunk_*.npz files found under {chunk_dir}")
    return specs


def _d4rt_cache_base(
    *,
    d4rt_cache_root: str | Path,
    scene: str,
    d4rt_chunk_size: int,
    d4rt_overlap_frames: int,
    d4rt_grid_size: int,
    d4rt_min_confidence: float,
    d4rt_min_visibility: float,
) -> Path:
    return (
        _project(d4rt_cache_root)
        / scene
        / (
            f"chunk{int(d4rt_chunk_size):02d}_overlap{int(d4rt_overlap_frames):02d}"
            f"_grid{int(d4rt_grid_size):02d}"
            f"_minconf{_float_token(float(d4rt_min_confidence))}"
            f"_minvis{_float_token(float(d4rt_min_visibility))}"
        )
    )


def _d4rt_cache_manifest_path(cache_base: Path, stride: int) -> Path:
    return cache_base / "cache_manifests" / f"d4rt_stride{int(stride)}_manifest.json"


def _d4rt_expected_params(args: argparse.Namespace) -> dict[str, Any]:
    d4rt_config = _repo_project(args.d4rt_config)
    d4rt_ckpt = _repo_project(args.d4rt_ckpt)
    return {
        "scene": str(args.scene),
        "stride": int(args.input_stride),
        "d4rt_root": str(_repo_project(args.d4rt_root)),
        "d4rt_config": str(d4rt_config),
        "d4rt_config_sha256": _sha256(d4rt_config),
        "d4rt_ckpt": str(d4rt_ckpt),
        "d4rt_ckpt_sha256": _sha256(d4rt_ckpt),
        "d4rt_ckpt_size_bytes": int(d4rt_ckpt.stat().st_size),
        "d4rt_device": str(args.d4rt_device),
        "d4rt_chunk_size": int(args.d4rt_chunk_size),
        "d4rt_overlap_frames": int(args.d4rt_overlap_frames),
        "d4rt_max_frames": int(args.d4rt_max_frames),
        "d4rt_grid_size": int(args.d4rt_grid_size),
        "d4rt_grid_margin_ratio": float(args.d4rt_grid_margin_ratio),
        "d4rt_query_chunk_size": int(args.d4rt_query_chunk_size),
        "d4rt_min_visibility": float(args.d4rt_min_visibility),
        "d4rt_min_confidence": float(args.d4rt_min_confidence),
        "d4rt_uv_radius": float(args.d4rt_uv_radius),
        "d4rt_max_matches_per_frame": int(args.d4rt_max_matches_per_frame),
        "d4rt_fit_trim_percentile": float(args.d4rt_fit_trim_percentile),
        "d4rt_max_sim3_anchors": int(args.d4rt_max_sim3_anchors),
        "d4rt_max_metric_points": int(args.d4rt_max_metric_points),
        "d4rt_max_gt_metric_points": int(args.d4rt_max_gt_metric_points),
        "d4rt_max_visual_points_per_stride": int(args.d4rt_max_visual_points_per_stride),
    }


def _d4rt_chunk_hash_rows(chunk_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_index, path in enumerate(sorted(chunk_dir.glob("chunk_*.npz"), key=_chunk_sort_key)):
        with np.load(path) as payload:
            frame_ids = [int(value) for value in np.asarray(payload["frame_ids"], dtype=np.int64).tolist()]
            keys = set(payload.files)
        rows.append(
            {
                "chunk_index": int(chunk_index),
                "chunk_npz": _rel(path),
                "chunk_npz_sha256": _sha256(path),
                "frame_count": int(len(frame_ids)),
                "frame_min": min(frame_ids) if frame_ids else None,
                "frame_max": max(frame_ids) if frame_ids else None,
                "has_transform_to_scene": bool(
                    {"transform_scale_to_scene", "transform_rot_to_scene", "transform_trans_to_scene"}.issubset(keys)
                ),
            }
        )
    return rows


def _summary_has_final_gt_transform(summary: dict[str, Any]) -> bool:
    transform = summary.get("final_gt_sim3_transform")
    if isinstance(transform, dict) and all(key in transform for key in ("scale", "rot", "trans")):
        return True
    fit = summary.get("final_gt_sim3")
    return isinstance(fit, dict) and all(key in fit for key in ("scale_d4rt_to_gt", "rotation_d4rt_to_gt", "translation_d4rt_to_gt"))


def _validate_d4rt_geometry_cache(
    *,
    cache_base: Path,
    args: argparse.Namespace,
    expected_params: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    stride = int(args.input_stride)
    manifest_path = _d4rt_cache_manifest_path(cache_base, stride)
    if not manifest_path.exists():
        return False, "manifest_missing", None
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return False, f"manifest_unreadable:{exc!r}", None
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "scene": manifest.get("scene") == str(args.scene),
        "stride": int(manifest.get("stride", -1)) == stride,
    }
    manifest_params = manifest.get("expected_params") if isinstance(manifest.get("expected_params"), dict) else {}
    for key, value in expected_params.items():
        checks[f"param_{key}"] = manifest_params.get(key) == value
    if not all(checks.values()):
        failed = ",".join(key for key, ok in checks.items() if not ok)
        return False, f"manifest_field_mismatch:{failed}", manifest

    stride_root = cache_base / f"stride_{stride}"
    summary_path = stride_root / "stride_summary.json"
    chunk_dir = stride_root / "chunks"
    if not summary_path.exists():
        return False, "stride_summary_missing", manifest
    if not chunk_dir.exists():
        return False, "chunk_dir_missing", manifest
    summary_sha = _sha256(summary_path)
    if summary_sha != str(manifest.get("stride_summary_sha256")):
        return False, "stride_summary_hash_mismatch", manifest
    summary = read_json(summary_path)
    if not _summary_has_final_gt_transform(summary):
        return False, "final_gt_sim3_transform_missing", manifest
    expected_frames = _scene_stride_frames(str(args.scene), stride)
    if int(summary.get("frame_count") or -1) != len(expected_frames):
        return False, "frame_count_mismatch", manifest
    if int(summary.get("chunk_count") or -1) != int(manifest.get("chunk_count", -2)):
        return False, "chunk_count_mismatch", manifest
    rows = _d4rt_chunk_hash_rows(chunk_dir)
    manifest_rows = manifest.get("chunk_rows", [])
    if len(rows) != len(manifest_rows):
        return False, "chunk_row_count_mismatch", manifest
    by_index = {int(row["chunk_index"]): row for row in manifest_rows}
    for row in rows:
        old = by_index.get(int(row["chunk_index"]))
        if old is None:
            return False, f"missing_chunk_manifest_row:{row['chunk_index']}", manifest
        if str(old.get("chunk_npz_sha256")) != str(row["chunk_npz_sha256"]):
            return False, f"chunk_hash_mismatch:{row['chunk_index']}", manifest
        if not bool(row["has_transform_to_scene"]):
            return False, f"chunk_transform_missing:{row['chunk_index']}", manifest
    return True, "cache_valid", manifest


def _d4rt_geometry_command(args: argparse.Namespace, cache_base: Path) -> list[str]:
    return [
        sys.executable,
        "tools/run_v65_d4rt_stride_overlap_geometry.py",
        "--scene",
        str(args.scene),
        "--strides",
        str(int(args.input_stride)),
        "--output-root",
        _rel(cache_base),
        "--d4rt-root",
        str(args.d4rt_root),
        "--d4rt-config",
        str(args.d4rt_config),
        "--d4rt-ckpt",
        str(args.d4rt_ckpt),
        "--device",
        str(args.d4rt_device),
        "--scannet-root",
        str(args.d4rt_scannet_root),
        "--chunk-size",
        str(int(args.d4rt_chunk_size)),
        "--overlap-frames",
        str(int(args.d4rt_overlap_frames)),
        "--max-frames",
        str(int(args.d4rt_max_frames)),
        "--grid-size",
        str(int(args.d4rt_grid_size)),
        "--grid-margin-ratio",
        str(float(args.d4rt_grid_margin_ratio)),
        "--query-chunk-size",
        str(int(args.d4rt_query_chunk_size)),
        "--min-visibility",
        str(float(args.d4rt_min_visibility)),
        "--min-confidence",
        str(float(args.d4rt_min_confidence)),
        "--uv-radius",
        str(float(args.d4rt_uv_radius)),
        "--max-matches-per-frame",
        str(int(args.d4rt_max_matches_per_frame)),
        "--fit-trim-percentile",
        str(float(args.d4rt_fit_trim_percentile)),
        "--max-sim3-anchors",
        str(int(args.d4rt_max_sim3_anchors)),
        "--max-metric-points",
        str(int(args.d4rt_max_metric_points)),
        "--max-gt-metric-points",
        str(int(args.d4rt_max_gt_metric_points)),
        "--max-visual-points-per-stride",
        str(int(args.d4rt_max_visual_points_per_stride)),
        "--save-chunks",
        "1",
    ]


def _write_d4rt_geometry_cache_manifest(
    *,
    cache_base: Path,
    args: argparse.Namespace,
    expected_params: dict[str, Any],
    process_row: dict[str, Any],
) -> dict[str, Any]:
    stride = int(args.input_stride)
    stride_root = cache_base / f"stride_{stride}"
    summary_path = stride_root / "stride_summary.json"
    geometry_summary_path = cache_base / "geometry_summary.json"
    chunk_dir = stride_root / "chunks"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if not _summary_has_final_gt_transform(summary):
        raise RuntimeError(f"D4RT stride summary lacks reusable final GT Sim3 transform: {summary_path}")
    chunk_rows = _d4rt_chunk_hash_rows(chunk_dir)
    manifest = {
        "schema_version": 1,
        "phase": "v65_d4rt_geometry_chunk_cache",
        "created_at": utc_now(),
        "scene": str(args.scene),
        "stride": stride,
        "cache_base": _rel(cache_base),
        "stride_root": _rel(stride_root),
        "chunk_dir": _rel(chunk_dir),
        "stride_summary": _rel(summary_path),
        "stride_summary_sha256": _sha256(summary_path),
        "geometry_summary": _rel(geometry_summary_path) if geometry_summary_path.exists() else "",
        "geometry_summary_sha256": _sha256(geometry_summary_path) if geometry_summary_path.exists() else "",
        "expected_params": expected_params,
        "frame_count": int(summary.get("frame_count") or 0),
        "chunk_count": int(summary.get("chunk_count") or len(chunk_rows)),
        "chunk_rows": chunk_rows,
        "process_row": process_row,
        "coordinate_contract": {
            "raw_chunk_xyz": "source chunk xyz is D4RT raw chunk-local xyz_ref",
            "chunk_self_stitched": "apply each chunk transform_*_to_scene",
            "chunk_final_gt_sim3": "apply chunk self-stitch then final_gt_sim3_transform from stride_summary",
            "default_ap_visualization_mode": "chunk_final_gt_sim3",
            "final_gt_sim3_is_diagnostic_only": True,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
    }
    manifest_path = _d4rt_cache_manifest_path(cache_base, stride)
    write_json(manifest_path, manifest)
    return manifest


def materialize_d4rt_geometry_by_cache(
    *,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    cache_policy = str(args.d4rt_cache_policy)
    if cache_policy not in {"cache_or_generate", "cache_only", "force_generate"}:
        raise ValueError(f"unsupported D4RT cache policy: {cache_policy}")
    expected_params = _d4rt_expected_params(args)
    cache_base = _d4rt_cache_base(
        d4rt_cache_root=args.d4rt_cache_root,
        scene=str(args.scene),
        d4rt_chunk_size=int(args.d4rt_chunk_size),
        d4rt_overlap_frames=int(args.d4rt_overlap_frames),
        d4rt_grid_size=int(args.d4rt_grid_size),
        d4rt_min_confidence=float(args.d4rt_min_confidence),
        d4rt_min_visibility=float(args.d4rt_min_visibility),
    )
    cache_base.mkdir(parents=True, exist_ok=True)
    valid_before, reason_before, manifest_before = _validate_d4rt_geometry_cache(
        cache_base=cache_base,
        args=args,
        expected_params=expected_params,
    )
    process_rows: list[dict[str, Any]] = []
    if valid_before and cache_policy != "force_generate":
        action = "reuse_cache"
        manifest = manifest_before or read_json(_d4rt_cache_manifest_path(cache_base, int(args.input_stride)))
    elif cache_policy == "cache_only":
        raise RuntimeError(f"D4RT geometry cache invalid: {reason_before}")
    else:
        action = "generate"
        if cache_policy == "force_generate" and cache_base.exists():
            shutil.rmtree(cache_base)
            cache_base.mkdir(parents=True, exist_ok=True)
        command = _d4rt_geometry_command(args, cache_base)
        log_dir = output_root / "d4rt_geometry_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"d4rt_stride{int(args.input_stride)}.log"
        env = os.environ.copy()
        if str(args.d4rt_gpus).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(args.d4rt_gpus).strip()
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        process_row = {
            "action": action,
            "command": " ".join(command),
            "cwd": str(ROOT),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
            "log_path": _rel(log_path),
            "returncode": int(proc.returncode),
            "elapsed_sec": float(time.perf_counter() - start),
        }
        process_rows.append(process_row)
        write_csv(output_root / "d4rt_geometry_process_rows.csv", process_rows)
        if proc.returncode != 0:
            raise RuntimeError(f"D4RT geometry extraction failed: returncode={proc.returncode} log={log_path}")
        manifest = _write_d4rt_geometry_cache_manifest(
            cache_base=cache_base,
            args=args,
            expected_params=expected_params,
            process_row=process_row,
        )
        valid_after, reason_after, _manifest_after = _validate_d4rt_geometry_cache(
            cache_base=cache_base,
            args=args,
            expected_params=expected_params,
        )
        if not valid_after:
            raise RuntimeError(f"D4RT geometry cache failed post-generate validation: {reason_after}")

    stride = int(args.input_stride)
    chunk_dir = cache_base / f"stride_{stride}" / "chunks"
    stride_summary = cache_base / f"stride_{stride}" / "stride_summary.json"
    geometry_summary = cache_base / "geometry_summary.json"
    summary = {
        "phase": "v65_pipeline_d4rt_geometry_materialization",
        "created_at": utc_now(),
        "scene": str(args.scene),
        "stride": stride,
        "d4rt_source": "pipeline_cache",
        "cache_policy": cache_policy,
        "cache_valid_before": bool(valid_before),
        "cache_reason_before": reason_before,
        "action": action,
        "cache_base": _rel(cache_base),
        "cache_manifest": _rel(_d4rt_cache_manifest_path(cache_base, stride)),
        "cache_manifest_sha256": _sha256(_d4rt_cache_manifest_path(cache_base, stride)),
        "chunk_dir": _rel(chunk_dir),
        "stride_summary": _rel(stride_summary),
        "stride_summary_sha256": _sha256(stride_summary),
        "geometry_summary": _rel(geometry_summary) if geometry_summary.exists() else "",
        "geometry_summary_sha256": _sha256(geometry_summary) if geometry_summary.exists() else "",
        "expected_params": expected_params,
        "frame_count": int(manifest.get("frame_count") or 0),
        "chunk_count": int(manifest.get("chunk_count") or 0),
        "coordinate_contract": manifest.get("coordinate_contract", {}),
        "process_rows_csv": _rel(output_root / "d4rt_geometry_process_rows.csv") if (output_root / "d4rt_geometry_process_rows.csv").exists() else "",
        "gate": {
            "cache_valid_after": True,
            "chunk_dir_exists": chunk_dir.exists(),
            "stride_summary_exists": stride_summary.exists(),
            "final_gt_sim3_transform_reusable": _summary_has_final_gt_transform(read_json(stride_summary)),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    write_json(output_root / "d4rt_geometry_materialization_summary.json", summary)
    write_csv(output_root / "d4rt_geometry_chunk_hashes.csv", _d4rt_chunk_hash_rows(chunk_dir))
    return summary


def _rgb_rows(scene: str, frame_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        path = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame_id)}.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "frame_id": int(frame_id),
                "source_rgb": _rel(path),
                "source_rgb_sha256": _sha256(path),
                "source_rgb_bytes": path.stat().st_size,
            }
        )
    return rows


def _mask_cache_base(
    *,
    mask_cache_root: str | Path,
    scene: str,
    stride: int,
    cropformer_confidence_threshold: float,
) -> Path:
    return (
        _project(mask_cache_root)
        / scene
        / f"stride_{int(stride)}"
        / f"cropformer_conf_{_float_token(float(cropformer_confidence_threshold))}"
        / "mask2former_hornet_3x"
    )


def _chunk_scene_name(scene: str, chunk_index: int) -> str:
    return f"{scene}_maskchunk{int(chunk_index):03d}"


def _chunk_manifest_path(cache_base: Path, chunk_index: int) -> Path:
    return cache_base / "chunk_manifests" / f"mask_chunk{int(chunk_index):03d}_manifest.json"


def _chunk_mask_dir(cache_base: Path, scene: str, chunk_index: int) -> Path:
    return cache_base / "shadow_processed" / _chunk_scene_name(scene, chunk_index) / "output_Cropformer" / "mask"


def _prepare_chunk_shadow_inputs(
    *,
    cache_base: Path,
    scene: str,
    chunk_index: int,
    rgb_rows: list[dict[str, Any]],
) -> Path:
    chunk_scene_root = cache_base / "shadow_processed" / _chunk_scene_name(scene, chunk_index)
    color_dir = chunk_scene_root / "color"
    if color_dir.exists():
        shutil.rmtree(color_dir)
    color_dir.mkdir(parents=True, exist_ok=True)
    for row in rgb_rows:
        src = _project(str(row["source_rgb"]))
        dst = color_dir / f"{int(row['frame_id'])}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return chunk_scene_root


def _validate_chunk_cache(
    *,
    cache_base: Path,
    scene: str,
    stride: int,
    chunk_spec: dict[str, Any],
    rgb_rows: list[dict[str, Any]],
    cropformer_confidence_threshold: float,
    config_sha256: str,
    weights_sha256: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    manifest_path = _chunk_manifest_path(cache_base, int(chunk_spec["chunk_index"]))
    if not manifest_path.exists():
        return False, "manifest_missing", None
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return False, f"manifest_unreadable:{exc!r}", None
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "scene": manifest.get("scene") == scene,
        "stride": int(manifest.get("stride", -1)) == int(stride),
        "chunk_index": int(manifest.get("chunk_index", -1)) == int(chunk_spec["chunk_index"]),
        "frame_ids": [int(value) for value in manifest.get("frame_ids", [])]
        == [int(value) for value in chunk_spec["frame_ids"]],
        "d4rt_chunk_sha256": str(manifest.get("d4rt_chunk_sha256")) == str(chunk_spec["chunk_npz_sha256"]),
        "cropformer_confidence_threshold": abs(
            float(manifest.get("cropformer_confidence_threshold", -1.0)) - float(cropformer_confidence_threshold)
        )
        < 1e-9,
        "cropformer_config_sha256": str(manifest.get("cropformer_config_sha256")) == str(config_sha256),
        "cropformer_weights_sha256": str(manifest.get("cropformer_weights_sha256")) == str(weights_sha256),
    }
    if not all(checks.values()):
        failed = ",".join(key for key, ok in checks.items() if not ok)
        return False, f"manifest_field_mismatch:{failed}", manifest
    rgb_by_frame = {int(row["frame_id"]): row for row in rgb_rows}
    frame_rows = manifest.get("frame_rows", [])
    if len(frame_rows) != len(rgb_rows):
        return False, "frame_row_count_mismatch", manifest
    for row in frame_rows:
        frame_id = int(row.get("frame_id", -1))
        rgb_row = rgb_by_frame.get(frame_id)
        if rgb_row is None:
            return False, f"unexpected_frame:{frame_id}", manifest
        if str(row.get("source_rgb_sha256")) != str(rgb_row["source_rgb_sha256"]):
            return False, f"rgb_hash_mismatch:{frame_id}", manifest
        mask_path = _project(str(row.get("chunk_mask_path", "")))
        if not mask_path.exists():
            return False, f"mask_missing:{frame_id}", manifest
        if _sha256(mask_path) != str(row.get("mask_sha256")):
            return False, f"mask_hash_mismatch:{frame_id}", manifest
    return True, "cache_valid", manifest


def _cropformer_command(
    *,
    cache_base: Path,
    scene: str,
    chunk_index: int,
    confidence_threshold: float,
) -> list[str]:
    return [
        sys.executable,
        "third_party/detectron2/projects/CropFormer/demo_cropformer/Cropformer.py",
        "--config-file",
        "third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml",
        "--root",
        str(cache_base / "shadow_processed"),
        "--image_path_pattern",
        "color/*.jpg",
        "--dataset",
        "scannet",
        "--seq_name_list",
        _chunk_scene_name(scene, chunk_index),
        "--confidence-threshold",
        str(float(confidence_threshold)),
        "--opts",
        "MODEL.WEIGHTS",
        "third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth",
    ]


def _write_chunk_manifest(
    *,
    cache_base: Path,
    scene: str,
    stride: int,
    chunk_spec: dict[str, Any],
    rgb_rows: list[dict[str, Any]],
    cropformer_confidence_threshold: float,
    config_sha256: str,
    weights_sha256: str,
    process_row: dict[str, Any],
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    mask_dir = _chunk_mask_dir(cache_base, scene, int(chunk_spec["chunk_index"]))
    for row in rgb_rows:
        frame_id = int(row["frame_id"])
        mask_path = mask_dir / f"{frame_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask_sha = _sha256(mask_path)
        aggregate.update(f"{frame_id} {mask_sha}\n".encode("utf-8"))
        frame_rows.append(
            {
                **row,
                "chunk_mask_path": _rel(mask_path),
                "mask_sha256": mask_sha,
                "mask_bytes": mask_path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "phase": "v65_cropformer_mask_chunk_cache",
        "created_at": utc_now(),
        "scene": scene,
        "stride": int(stride),
        "chunk_index": int(chunk_spec["chunk_index"]),
        "d4rt_chunk_npz": _rel(chunk_spec["chunk_npz"]),
        "d4rt_chunk_sha256": str(chunk_spec["chunk_npz_sha256"]),
        "frame_ids": [int(value) for value in chunk_spec["frame_ids"]],
        "frame_count": int(len(frame_rows)),
        "cropformer_confidence_threshold": float(cropformer_confidence_threshold),
        "cropformer_config": _rel(_cropformer_config_path()),
        "cropformer_config_sha256": config_sha256,
        "cropformer_weights": _rel(_cropformer_weights_path()),
        "cropformer_weights_sha256": weights_sha256,
        "mask_hash_aggregate_sha256": aggregate.hexdigest(),
        "frame_rows": frame_rows,
        "process_row": process_row,
        "uses_gt_for_prediction": False,
    }
    write_json(_chunk_manifest_path(cache_base, int(chunk_spec["chunk_index"])), manifest)
    return manifest


def _launch_cropformer_mask_chunk(
    *,
    cache_base: Path,
    output_root: Path,
    scene: str,
    stride: int,
    chunk_spec: dict[str, Any],
    rgb_rows: list[dict[str, Any]],
    gpu: str,
    cropformer_confidence_threshold: float,
    config_sha256: str,
    weights_sha256: str,
) -> tuple[subprocess.Popen[Any], Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    chunk_index = int(chunk_spec["chunk_index"])
    _prepare_chunk_shadow_inputs(cache_base=cache_base, scene=scene, chunk_index=chunk_index, rgb_rows=rgb_rows)
    mask_dir = _chunk_mask_dir(cache_base, scene, chunk_index)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    command = _cropformer_command(
        cache_base=cache_base,
        scene=scene,
        chunk_index=chunk_index,
        confidence_threshold=cropformer_confidence_threshold,
    )
    log_dir = output_root / "mask_materialization_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"mask_chunk{chunk_index:03d}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    handle = log_path.open("w", encoding="utf-8")
    start = time.perf_counter()
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    process_row = {
        "chunk_index": chunk_index,
        "gpu": str(gpu),
        "frame_count": int(len(rgb_rows)),
        "frame_min": min((int(row["frame_id"]) for row in rgb_rows), default=None),
        "frame_max": max((int(row["frame_id"]) for row in rgb_rows), default=None),
        "command": " ".join(command),
        "log_path": _rel(log_path),
        "start_monotonic": float(start),
    }
    chunk_context = {
        "scene": scene,
        "stride": int(stride),
        "chunk_spec": chunk_spec,
        "cropformer_confidence_threshold": float(cropformer_confidence_threshold),
        "config_sha256": config_sha256,
        "weights_sha256": weights_sha256,
    }
    return proc, handle, process_row, rgb_rows, chunk_context


def _merge_mask_chunks(
    *,
    cache_base: Path,
    output_root: Path,
    scene: str,
    expected_frames: list[int],
    chunk_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    final_processed_root = cache_base / "final_processed"
    final_mask_dir = final_processed_root / scene / "output_Cropformer" / "mask"
    if final_mask_dir.exists():
        shutil.rmtree(final_mask_dir)
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    expected = set(int(frame) for frame in expected_frames)
    present_hash_by_frame: dict[int, str] = {}
    duplicate_same_hash_count = 0
    duplicate_conflicts: list[dict[str, Any]] = []
    copy_rows: list[dict[str, Any]] = []
    for manifest in sorted(chunk_manifests, key=lambda item: int(item.get("chunk_index", 0))):
        chunk_index = int(manifest["chunk_index"])
        for row in manifest.get("frame_rows", []):
            frame_id = int(row["frame_id"])
            src = _project(str(row["chunk_mask_path"]))
            mask_sha = str(row["mask_sha256"])
            old_sha = present_hash_by_frame.get(frame_id)
            if old_sha is not None:
                if old_sha != mask_sha:
                    duplicate_conflicts.append(
                        {"frame_id": frame_id, "old_sha256": old_sha, "new_sha256": mask_sha, "chunk_index": chunk_index}
                    )
                else:
                    duplicate_same_hash_count += 1
                continue
            dst = final_mask_dir / f"{frame_id}.png"
            shutil.copy2(src, dst)
            copied_sha = _sha256(dst)
            if copied_sha != mask_sha:
                duplicate_conflicts.append(
                    {
                        "frame_id": frame_id,
                        "old_sha256": mask_sha,
                        "new_sha256": copied_sha,
                        "chunk_index": chunk_index,
                        "reason": "copy_hash_mismatch",
                    }
                )
            present_hash_by_frame[frame_id] = copied_sha
            copy_rows.append(
                {
                    "frame_id": frame_id,
                    "chunk_index": chunk_index,
                    "final_mask_path": _rel(dst),
                    "sha256": copied_sha,
                    "bytes": dst.stat().st_size,
                }
            )
    copy_rows.sort(key=lambda row: int(row["frame_id"]))
    _write_copy_rows_path = output_root / "mask_materialization_final_mask_hashes.csv"
    write_csv(_write_copy_rows_path, copy_rows)
    missing = sorted(expected - set(present_hash_by_frame))
    non_stride = sorted(set(present_hash_by_frame) - expected)
    aggregate = hashlib.sha256()
    for row in copy_rows:
        aggregate.update(f"{int(row['frame_id'])} {row['sha256']}\n".encode("utf-8"))
    gate = {
        "all_expected_stride_frames_have_2d_masks": len(missing) == 0,
        "duplicate_mask_hash_conflict_count_eq_0": len(duplicate_conflicts) == 0,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "final_processed_root": _rel(final_processed_root),
        "final_mask_dir": _rel(final_mask_dir),
        "final_mask_hashes_csv": _rel(_write_copy_rows_path),
        "expected_stride_frame_count": int(len(expected_frames)),
        "available_stride_mask_frame_count": int(len(expected & set(present_hash_by_frame))),
        "missing_stride_mask_frame_count": int(len(missing)),
        "coverage_ratio": float(len(expected & set(present_hash_by_frame)) / max(len(expected_frames), 1)),
        "first_missing_stride_frames": missing[:50],
        "last_missing_stride_frames": missing[-20:],
        "non_stride_mask_frame_count": int(len(non_stride)),
        "duplicate_same_hash_count": int(duplicate_same_hash_count),
        "duplicate_conflict_count": int(len(duplicate_conflicts)),
        "duplicate_conflicts_first20": duplicate_conflicts[:20],
        "present_stride_mask_file_hash_aggregate_sha256": aggregate.hexdigest() if copy_rows else None,
        "gate": gate,
    }


def materialize_cropformer_masks_by_d4rt_chunk(
    *,
    scene: str,
    stride: int,
    chunk_dir: Path,
    output_root: Path,
    mask_cache_root: str | Path,
    cache_policy: str,
    gpus: list[str],
    cropformer_confidence_threshold: float,
    max_chunks: int,
    parallelism: int,
) -> dict[str, Any]:
    if cache_policy not in {"cache_or_generate", "cache_only", "force_generate"}:
        raise ValueError(f"unsupported cache policy: {cache_policy}")
    if not gpus:
        raise ValueError("mask generation requires at least one GPU id")
    cache_base = _mask_cache_base(
        mask_cache_root=mask_cache_root,
        scene=scene,
        stride=stride,
        cropformer_confidence_threshold=cropformer_confidence_threshold,
    )
    cache_base.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256(_cropformer_config_path())
    weights_sha256 = _sha256(_cropformer_weights_path())
    chunk_specs = _d4rt_chunk_frame_specs(chunk_dir, max_chunks=max_chunks)
    expected_frames = _scene_stride_frames(scene, stride)
    chunk_rows: list[dict[str, Any]] = []
    valid_manifests: list[dict[str, Any]] = []
    chunks_to_run: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []

    for spec in chunk_specs:
        rgb_rows = _rgb_rows(scene, [int(value) for value in spec["frame_ids"]])
        cache_valid, cache_reason, manifest = _validate_chunk_cache(
            cache_base=cache_base,
            scene=scene,
            stride=stride,
            chunk_spec=spec,
            rgb_rows=rgb_rows,
            cropformer_confidence_threshold=cropformer_confidence_threshold,
            config_sha256=config_sha256,
            weights_sha256=weights_sha256,
        )
        row = {
            "chunk_index": int(spec["chunk_index"]),
            "d4rt_chunk_npz": _rel(spec["chunk_npz"]),
            "d4rt_chunk_sha256": spec["chunk_npz_sha256"],
            "frame_count": len(spec["frame_ids"]),
            "frame_min": min(spec["frame_ids"]) if spec["frame_ids"] else None,
            "frame_max": max(spec["frame_ids"]) if spec["frame_ids"] else None,
            "cache_valid_before": bool(cache_valid),
            "cache_reason_before": cache_reason,
            "cache_policy": cache_policy,
        }
        if cache_valid and cache_policy != "force_generate":
            row["action"] = "reuse_cache"
            valid_manifests.append(manifest or {})
        elif cache_policy == "cache_only":
            row["action"] = "cache_missing_or_invalid"
            chunk_rows.append(row)
            write_csv(output_root / "mask_materialization_chunk_rows.csv", chunk_rows)
            raise RuntimeError(f"mask cache invalid for chunk {spec['chunk_index']}: {cache_reason}")
        else:
            row["action"] = "generate"
            chunks_to_run.append((spec, rgb_rows, cache_reason))
        chunk_rows.append(row)

    process_rows: list[dict[str, Any]] = []
    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    pending = list(chunks_to_run)
    gpu_index = 0
    max_parallel = int(parallelism) if int(parallelism) > 0 else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))
    while pending or running:
        while pending and len(running) < max_parallel:
            spec, rgb_rows, _reason = pending.pop(0)
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1
            running.append(
                _launch_cropformer_mask_chunk(
                    cache_base=cache_base,
                    output_root=output_root,
                    scene=scene,
                    stride=stride,
                    chunk_spec=spec,
                    rgb_rows=rgb_rows,
                    gpu=gpu,
                    cropformer_confidence_threshold=cropformer_confidence_threshold,
                    config_sha256=config_sha256,
                    weights_sha256=weights_sha256,
                )
            )
        proc, handle, process_row, rgb_rows, context = running.pop(0)
        returncode = proc.wait()
        handle.close()
        process_row["returncode"] = int(returncode)
        process_row["elapsed_sec"] = float(time.perf_counter() - float(process_row["start_monotonic"]))
        process_rows.append(process_row)
        if returncode != 0:
            write_csv(output_root / "mask_materialization_process_rows.csv", process_rows)
            raise RuntimeError(f"CropFormer mask chunk failed: chunk={process_row['chunk_index']} returncode={returncode}")
        manifest = _write_chunk_manifest(
            cache_base=cache_base,
            scene=scene,
            stride=stride,
            chunk_spec=context["chunk_spec"],
            rgb_rows=rgb_rows,
            cropformer_confidence_threshold=context["cropformer_confidence_threshold"],
            config_sha256=context["config_sha256"],
            weights_sha256=context["weights_sha256"],
            process_row=process_row,
        )
        valid_manifests.append(manifest)

    merge_summary = _merge_mask_chunks(
        cache_base=cache_base,
        output_root=output_root,
        scene=scene,
        expected_frames=expected_frames,
        chunk_manifests=valid_manifests,
    )
    write_csv(output_root / "mask_materialization_chunk_rows.csv", chunk_rows)
    write_csv(output_root / "mask_materialization_process_rows.csv", process_rows)
    summary = {
        "phase": "v65_pipeline_chunked_cropformer_mask_materialization",
        "created_at": utc_now(),
        "scene": scene,
        "stride": int(stride),
        "chunk_source": "d4rt_chunk_frame_ids",
        "chunk_count_requested": int(len(chunk_specs)),
        "chunk_count_reused_cache": int(sum(1 for row in chunk_rows if row.get("action") == "reuse_cache")),
        "chunk_count_generated": int(sum(1 for row in chunk_rows if row.get("action") == "generate")),
        "cache_policy": cache_policy,
        "cache_base": _rel(cache_base),
        "mask_cache_root": _rel(_project(mask_cache_root)),
        "gpus": gpus,
        "parallelism": int(max_parallel),
        "cropformer_confidence_threshold": float(cropformer_confidence_threshold),
        "cropformer_config": _rel(_cropformer_config_path()),
        "cropformer_config_sha256": config_sha256,
        "cropformer_weights": _rel(_cropformer_weights_path()),
        "cropformer_weights_sha256": weights_sha256,
        "max_chunks": int(max_chunks),
        "chunk_rows_csv": _rel(output_root / "mask_materialization_chunk_rows.csv"),
        "process_rows_csv": _rel(output_root / "mask_materialization_process_rows.csv"),
        "merge_summary": merge_summary,
        "gate": {
            "all_chunks_valid_or_generated": True,
            "final_merge_pass": bool(merge_summary["gate"]["pass"]),
            "pass": bool(merge_summary["gate"]["pass"]),
        },
        "uses_gt_for_prediction": False,
    }
    write_json(output_root / "mask_materialization_summary.json", summary)
    return summary


def _mask_frame_coverage(
    scene: str,
    expected_frames: list[int],
    mask_root: str | Path | None,
    output_root: Path,
) -> dict[str, Any]:
    mask_dir = resolve_mask_dir(mask_root, scene)
    mask_frames = set(_discover_numeric_stems(mask_dir, ".png"))
    expected = set(expected_frames)
    present = sorted(expected & mask_frames)
    missing = sorted(expected - mask_frames)
    hash_rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for frame in present:
        path = mask_dir / f"{int(frame)}.png"
        digest = _sha256(path)
        aggregate.update(f"{int(frame)} {digest}\n".encode("utf-8"))
        hash_rows.append(
            {
                "frame_id": int(frame),
                "mask_path": _rel(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    hash_csv = output_root / "mask_frame_hashes.csv"
    write_csv(hash_csv, hash_rows)
    return {
        "mask_root_arg": str(mask_root or ""),
        "mask_dir": _rel(mask_dir),
        "expected_stride_frame_count": len(expected_frames),
        "available_stride_mask_frame_count": len(present),
        "missing_stride_mask_frame_count": len(missing),
        "coverage_ratio": float(len(present) / max(len(expected_frames), 1)),
        "first_present_stride_frames": present[:20],
        "last_present_stride_frames": present[-20:],
        "first_missing_stride_frames": missing[:50],
        "last_missing_stride_frames": missing[-20:],
        "non_stride_mask_frame_count": len(mask_frames - expected),
        "present_stride_mask_file_hash_aggregate_sha256": aggregate.hexdigest() if hash_rows else None,
        "mask_frame_hashes_csv": _rel(hash_csv),
        "gate": {
            "all_expected_stride_frames_have_2d_masks": len(missing) == 0,
            "pass": len(missing) == 0,
        },
    }


def _chunk_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def convert_d4rt_chunks_to_carrier_cache(
    *,
    chunk_dir: Path,
    output_cache_root: Path,
    scene: str,
    input_stride: int,
    globalize_carrier_ids: bool,
) -> dict[str, Any]:
    paths = sorted(chunk_dir.glob("chunk_*.npz"), key=_chunk_sort_key)
    if not paths:
        raise FileNotFoundError(f"No D4RT chunk_*.npz files found under {chunk_dir}")
    scene_dir = output_cache_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    seen_original_ids: set[int] = set()
    duplicate_original_id_count = 0
    frame_set: set[int] = set()
    frame_diffs: list[int] = []
    chunk_rows: list[dict[str, Any]] = []
    carrier_count_total = 0
    valid_count_total = 0
    confidence_values: list[float] = []
    visibility_values: list[float] = []

    for output_index, path in enumerate(paths):
        with np.load(path) as data:
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            uv = np.asarray(data["uv"], dtype=np.float32)
            visibility = np.asarray(data["visibility"], dtype=np.float32)
            confidence = np.asarray(data["confidence"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            source_carrier_ids = np.asarray(data["carrier_id"], dtype=np.int64)
            xyz = np.asarray(data["xyz"], dtype=np.float32) if "xyz" in data.files else None
            src_frame_global = (
                np.asarray(data["src_frame_global"], dtype=np.int64)
                if "src_frame_global" in data.files
                else np.full((uv.shape[1],), -1, dtype=np.int64)
            )
            src_xy = (
                np.asarray(data["src_xy"], dtype=np.int64)
                if "src_xy" in data.files
                else np.full((uv.shape[1], 2), -1, dtype=np.int64)
            )
        if uv.ndim != 3 or uv.shape[:2] != visibility.shape or uv.shape[:2] != confidence.shape:
            raise ValueError(f"Unexpected D4RT chunk shapes in {path}: uv={uv.shape}, visibility={visibility.shape}, confidence={confidence.shape}")
        if len(frame_ids) != uv.shape[0]:
            raise ValueError(f"frame_ids length does not match uv time dimension in {path}")
        source_ids = [int(value) for value in source_carrier_ids.tolist()]
        duplicate_original_id_count += sum(1 for value in source_ids if value in seen_original_ids)
        seen_original_ids.update(source_ids)
        if globalize_carrier_ids:
            carrier_ids = np.arange(uv.shape[1], dtype=np.int64) + (output_index + 1) * 10_000_000
        else:
            carrier_ids = source_carrier_ids
        carrier_path = scene_dir / f"carriers_window{output_index:03d}.npz"
        source_path = scene_dir / f"carrier_sources_window{output_index:03d}.npz"
        save_payload: dict[str, Any] = {
            "uv_pred": uv,
            "visibility_prob": visibility,
            "confidence_prob": confidence,
            "valid": valid,
            "carrier_id": carrier_ids,
            "source_carrier_id": source_carrier_ids,
            "src_frame_global": src_frame_global,
            "src_xy": src_xy,
        }
        if xyz is not None:
            save_payload["xyz_ref"] = xyz
        np.savez_compressed(carrier_path, **save_payload)
        np.savez_compressed(
            source_path,
            carrier_id=carrier_ids,
            source_carrier_id=source_carrier_ids,
            src_frame_global=src_frame_global,
            src_xy=src_xy,
        )
        diffs = [int(b - a) for a, b in zip(frame_ids.tolist(), frame_ids.tolist()[1:])]
        frame_diffs.extend(diffs)
        frame_set.update(int(value) for value in frame_ids.tolist())
        carrier_count_total += int(uv.shape[1])
        valid_count_total += int(valid.sum())
        confidence_values.extend(float(value) for value in confidence[valid].ravel().tolist())
        visibility_values.extend(float(value) for value in visibility[valid].ravel().tolist())
        manifest = {
            "scene": scene,
            "variant": "v65_soma_fullscene_stride_pipeline_from_d4rt_chunks",
            "pipeline_stage": "d4rt_chunk_to_v47_carrier_cache",
            "source_chunk_npz": _rel(path),
            "source_chunk_sha256": _sha256(path),
            "carrier_npz": _rel(carrier_path),
            "carrier_source_npz": _rel(source_path),
            "frame_ids": [int(value) for value in frame_ids.tolist()],
            "frame_diffs": diffs,
            "input_stride": int(input_stride),
            "globalize_carrier_ids": bool(globalize_carrier_ids),
            "carrier_count": int(uv.shape[1]),
            "uses_gt_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
        }
        write_json(carrier_path.with_name(f"{carrier_path.stem}_manifest.json"), manifest)
        chunk_rows.append(
            {
                "window_index": output_index,
                "source_chunk_npz": _rel(path),
                "carrier_npz": _rel(carrier_path),
                "frame_count": int(len(frame_ids)),
                "frame_min": int(frame_ids.min()),
                "frame_max": int(frame_ids.max()),
                "carrier_count": int(uv.shape[1]),
                "valid_observation_count": int(valid.sum()),
                "source_chunk_sha256": manifest["source_chunk_sha256"],
            }
        )

    write_csv(output_cache_root / "converted_chunk_rows.csv", chunk_rows)
    summary = {
        "phase": "v65_d4rt_chunk_to_v47_carrier_cache",
        "created_at": utc_now(),
        "scene": scene,
        "input_stride": int(input_stride),
        "source_chunk_dir": _rel(chunk_dir),
        "output_cache_root": _rel(output_cache_root),
        "chunk_count": len(paths),
        "unique_frame_count": len(frame_set),
        "frame_min": min(frame_set) if frame_set else None,
        "frame_max": max(frame_set) if frame_set else None,
        "frame_diffs_unique": sorted(set(frame_diffs)),
        "frame_diff_all_eq_input_stride": bool(frame_diffs and all(diff == int(input_stride) for diff in frame_diffs)),
        "carrier_count_per_chunk_total": int(carrier_count_total),
        "valid_observation_count": int(valid_count_total),
        "source_unique_carrier_id_count": len(seen_original_ids),
        "source_duplicate_carrier_id_count_across_chunks": int(duplicate_original_id_count),
        "globalize_carrier_ids": bool(globalize_carrier_ids),
        "confidence_valid_mean": float(np.mean(confidence_values)) if confidence_values else None,
        "visibility_valid_mean": float(np.mean(visibility_values)) if visibility_values else None,
        "uses_gt_for_prediction": False,
    }
    write_json(output_cache_root / "conversion_summary.json", summary)
    return summary


def _run_stage(name: str, fn) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, float(time.perf_counter() - start)


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"output root already exists; pass --overwrite to replace only this run root: {output_root}")
    audit_root = ROOT / "outputs" / "audit"
    try:
        output_root.relative_to(audit_root)
    except ValueError as exc:
        raise ValueError(f"refusing to overwrite outside Stream3D/outputs/audit: {output_root}") from exc
    shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    input_stride = int(args.input_stride)
    feature_backend = "" if str(args.feature_backend).strip().lower() in {"", "disabled", "none", "off"} else str(args.feature_backend)
    output_root = _project(args.output_root)
    _prepare_output_root(output_root, bool(args.overwrite))
    stage_times: dict[str, float] = {}
    d4rt_source = str(args.d4rt_source)
    d4rt_geometry_summary: dict[str, Any] | None = None
    if d4rt_source == "pipeline_cache":
        d4rt_geometry_summary, stage_times["materialize_d4rt_geometry_by_cache_sec"] = _run_stage(
            "materialize_d4rt_geometry_by_cache",
            lambda: materialize_d4rt_geometry_by_cache(args=args, output_root=output_root),
        )
        chunk_dir = _project(d4rt_geometry_summary["chunk_dir"])
    elif d4rt_source == "external":
        chunk_dir = _project(args.d4rt_chunk_dir)
        d4rt_geometry_summary = {
            "phase": "v65_pipeline_d4rt_geometry_materialization",
            "scene": scene,
            "stride": input_stride,
            "d4rt_source": "external",
            "chunk_dir": _rel(chunk_dir),
            "cache_policy": "external_not_validated",
            "gate": {
                "chunk_dir_exists": chunk_dir.exists(),
                "pass": chunk_dir.exists(),
            },
            "warning": "external D4RT chunk dir is accepted for backward compatibility; pipeline_cache is required for full cache/hash/coordinate-contract validation",
            "uses_gt_for_prediction": False,
        }
    else:
        raise ValueError(f"unsupported --d4rt-source: {d4rt_source}")
    carrier_cache_root = output_root / "carrier_cache"
    observation_root = output_root / "observation_tables"
    support_root = output_root / "mask_component_support"
    chunk_universe_root = output_root / "chunk_universe"
    representative_root = output_root / "representative_observations"
    ledger_root = output_root / "reprojection_ledger"
    objectlet_root = output_root / "local_objectlets"
    reproduction_root = output_root / "local_reproduction"
    visualization_root = output_root / "visualizations"

    expected_frames = _scene_stride_frames(scene, input_stride)
    mask_materialization_summary: dict[str, Any] | None = None
    mask_source = str(args.mask_source)
    if mask_source == "pipeline_cache":
        mask_materialization_summary, stage_times["materialize_cropformer_masks_by_d4rt_chunk_sec"] = _run_stage(
            "materialize_cropformer_masks_by_d4rt_chunk",
            lambda: materialize_cropformer_masks_by_d4rt_chunk(
                scene=scene,
                stride=input_stride,
                chunk_dir=chunk_dir,
                output_root=output_root,
                mask_cache_root=args.mask_cache_root,
                cache_policy=str(args.mask_cache_policy),
                gpus=[item.strip() for item in str(args.mask_gpus).split(",") if item.strip()],
                cropformer_confidence_threshold=float(args.mask_cropformer_confidence_threshold),
                max_chunks=int(args.mask_max_chunks),
                parallelism=int(args.mask_chunk_parallelism),
            ),
        )
        mask_root_for_pipeline = str(mask_materialization_summary["merge_summary"]["final_processed_root"])
    elif mask_source == "external":
        if not str(args.mask_root).strip():
            raise ValueError("--mask-source external requires --mask-root")
        mask_root_for_pipeline = str(args.mask_root)
    else:
        raise ValueError(f"unsupported --mask-source: {mask_source}")

    mask_coverage = _mask_frame_coverage(scene, expected_frames, mask_root_for_pipeline, output_root)
    if bool(args.stop_after_mask_materialization):
        summary = {
            "phase": "v65_soma_fullscene_pipeline",
            "created_at": utc_now(),
            "terminal_status": "stopped_after_mask_materialization",
            "scene": scene,
            "input_stride": input_stride,
            "cwd": os.getcwd(),
            "python": sys.executable,
            "argv": sys.argv,
            "d4rt_source": d4rt_source,
            "d4rt_chunk_dir": _rel(chunk_dir),
            "d4rt_geometry_materialization_summary": d4rt_geometry_summary,
            "mask_source": mask_source,
            "mask_root_for_pipeline": mask_root_for_pipeline,
            "resolved_mask_dir": mask_coverage.get("mask_dir"),
            "mask_materialization_summary": mask_materialization_summary,
            "mask_frame_coverage": mask_coverage,
            "pipeline_gate": {
                "soma_2d_mask_stride_gate_pass": bool(mask_coverage["gate"]["pass"]),
                "ap_ready": False,
                "pass": bool(mask_coverage["gate"]["pass"]),
            },
            "stage_times_sec": stage_times,
            "uses_gt_for_prediction": False,
        }
        write_json(output_root / "pipeline_summary.json", summary)
        return summary

    conversion_summary, stage_times["convert_d4rt_chunks_to_carrier_cache_sec"] = _run_stage(
        "convert_d4rt_chunks_to_carrier_cache",
        lambda: convert_d4rt_chunks_to_carrier_cache(
            chunk_dir=chunk_dir,
            output_cache_root=carrier_cache_root,
            scene=scene,
            input_stride=input_stride,
            globalize_carrier_ids=bool(args.globalize_carrier_ids),
        ),
    )

    observation_payload, stage_times["build_observation_tables_sec"] = _run_stage(
        "build_observation_tables",
        lambda: build_observation_tables(
            carrier_cache_root=carrier_cache_root,
            scenes=[scene],
            mask_root=mask_root_for_pipeline,
            visibility_threshold=float(args.visibility_threshold),
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            feature_backend=feature_backend,
        ),
    )
    write_csv(observation_root / "carrier_observation_table.csv", observation_payload["carrier_rows"])
    write_csv(observation_root / "mask_observation_table.csv", observation_payload["mask_rows"])
    write_csv(observation_root / "observation_window_rows.csv", observation_payload["window_rows"])
    write_json(observation_root / "observation_table_summary.json", observation_payload["summary"])

    if bool(args.stop_after_observation_if_mask_incomplete) and not mask_coverage["gate"]["pass"]:
        summary = _pipeline_summary(
            args=args,
            feature_backend=feature_backend,
            output_root=output_root,
            chunk_dir=chunk_dir,
            d4rt_source=d4rt_source,
            d4rt_geometry_summary=d4rt_geometry_summary,
            conversion_summary=conversion_summary,
            observation_summary=observation_payload["summary"],
            mask_coverage=mask_coverage,
            stage_times=stage_times,
            stage_roots={
                "carrier_cache": carrier_cache_root,
                "observation_tables": observation_root,
            },
            terminal_status="stopped_after_observation_mask_coverage_blocker",
        )
        write_json(output_root / "pipeline_summary.json", summary)
        return summary

    support_payload, stage_times["build_mask_component_support_sec"] = _run_stage(
        "build_mask_component_support",
        lambda: build_mask_component_support(
            carrier_table_path=observation_root / "carrier_observation_table.csv",
            mask_table_path=observation_root / "mask_observation_table.csv",
            max_union_unique_carriers=int(args.max_union_unique_carriers),
            min_visibility_prob=float(args.visibility_threshold),
            min_confidence=float(args.confidence_threshold),
            extra_visible_taus=[float(args.support_visible_tau)],
            gate_variant=str(args.support_variant),
        ),
    )
    write_mask_component_support(support_root, support_payload, visualization_root / "mask_component_support")

    chunk_payload, stage_times["build_chunk_universe_sec"] = _run_stage(
        "build_chunk_universe",
        lambda: build_chunk_universe(
            carrier_table_path=observation_root / "carrier_observation_table.csv",
            mask_table_path=observation_root / "mask_observation_table.csv",
            max_union_unique_carriers=int(args.max_union_unique_carriers),
            min_visibility_prob=float(args.visibility_threshold),
            min_confidence=float(args.confidence_threshold),
            chunk_size=int(args.chunk_size_selected),
            chunk_stride=int(args.chunk_stride_selected),
        ),
    )
    write_chunk_universe(chunk_universe_root, chunk_payload, visualization_root / "chunk_universe")

    representative_payload, stage_times["build_representative_observations_sec"] = _run_stage(
        "build_representative_observations",
        lambda: build_representative_observations(
            support_rows_path=support_root / "mask_component_support_rows.csv",
            mask_summary_path=support_root / "mask_summary_rows.csv",
            chunk_component_rows_path=chunk_universe_root / "chunk_component_rows.csv",
            chunk_mask_rows_path=chunk_universe_root / "chunk_mask_rows.csv",
            support_variant=str(args.support_variant),
            max_selected_ratio=float(args.max_selected_ratio),
            gate_variant=str(args.representative_variant),
        ),
    )
    write_representative_observations(representative_root, representative_payload, visualization_root / "representative_observations")

    ledger_payload, stage_times["build_reprojection_ledger_sec"] = _run_stage(
        "build_reprojection_ledger",
        lambda: build_reprojection_ledger(
            carrier_table_path=observation_root / "carrier_observation_table.csv",
            mask_table_path=observation_root / "mask_observation_table.csv",
            support_rows_path=support_root / "mask_component_support_rows.csv",
            representative_rows_path=representative_root / "representative_mask_rows.csv",
            support_variant=str(args.support_variant),
            representative_variant=str(args.representative_variant),
            max_union_unique_carriers=int(args.max_union_unique_carriers),
            min_visibility_prob=float(args.visibility_threshold),
            min_confidence=float(args.confidence_threshold),
            min_visible_carriers=int(args.min_visible_carriers),
            max_candidates=int(args.max_candidates),
            skip_no_related_measurement=bool(args.skip_no_related_measurement),
            include_repeated_support_candidates=bool(args.include_repeated_support_candidates),
            repeated_support_min_frames=int(args.repeated_support_min_frames),
            repeated_support_min_components=int(args.repeated_support_min_components),
            repeated_support_min_w_visible=float(args.repeated_support_min_w_visible),
            repeated_support_max_components=int(args.repeated_support_max_components),
            repeated_support_max_groups_per_scene=int(args.repeated_support_max_groups_per_scene),
            max_candidate_conflict_rate=float(args.max_candidate_conflict_rate),
        ),
    )
    write_reprojection_ledger(ledger_root, ledger_payload, visualization_root / "reprojection_ledger", mask_root=mask_root_for_pipeline)

    objectlet_payload, stage_times["build_local_objectlets_sec"] = _run_stage(
        "build_local_objectlets",
        lambda: build_local_objectlets(
            support_rows_path=support_root / "mask_component_support_rows.csv",
            candidate_rows_path=ledger_root / "candidate_rows.csv",
            ledger_rows_path=ledger_root / "reprojection_ledger_rows.csv",
            representative_rows_path=representative_root / "representative_mask_rows.csv",
            support_variant=str(args.support_variant),
            representative_variant=str(args.representative_variant),
        ),
    )
    write_local_objectlets(objectlet_root, objectlet_payload)

    reproduction_payload, stage_times["build_v54_local_reproduction_sec"] = _run_stage(
        "build_v54_local_reproduction",
        lambda: build_v54_local_reproduction(
            support_rows_path=support_root / "mask_component_support_rows.csv",
            chunk_component_rows_path=chunk_universe_root / "chunk_component_rows.csv",
            chunk_mask_rows_path=chunk_universe_root / "chunk_mask_rows.csv",
            objectlet_summary_path=objectlet_root / "local_objectlet_summary.json",
            objectlet_rows_path=objectlet_root / "objectlet_rows.csv",
            support_variant=str(args.support_variant),
            enable_weak_support_restitution=bool(args.enable_weak_support_restitution),
        ),
    )
    write_v54_local_reproduction(reproduction_root, reproduction_payload)

    summary = _pipeline_summary(
        args=args,
        feature_backend=feature_backend,
        output_root=output_root,
        chunk_dir=chunk_dir,
        d4rt_source=d4rt_source,
        d4rt_geometry_summary=d4rt_geometry_summary,
        mask_source=mask_source,
        mask_root_for_pipeline=mask_root_for_pipeline,
        mask_materialization_summary=mask_materialization_summary,
        conversion_summary=conversion_summary,
        observation_summary=observation_payload["summary"],
        mask_coverage=mask_coverage,
        stage_times=stage_times,
        stage_roots={
            "carrier_cache": carrier_cache_root,
            "observation_tables": observation_root,
            "mask_component_support": support_root,
            "chunk_universe": chunk_universe_root,
            "representative_observations": representative_root,
            "reprojection_ledger": ledger_root,
            "local_objectlets": objectlet_root,
            "local_reproduction": reproduction_root,
        },
        terminal_status="completed_with_input_gates_recorded",
    )
    write_json(output_root / "pipeline_summary.json", summary)
    return summary


def _pipeline_summary(
    *,
    args: argparse.Namespace,
    feature_backend: str,
    output_root: Path,
    chunk_dir: Path,
    d4rt_source: str,
    d4rt_geometry_summary: dict[str, Any] | None,
    mask_source: str,
    mask_root_for_pipeline: str,
    mask_materialization_summary: dict[str, Any] | None,
    conversion_summary: dict[str, Any],
    observation_summary: dict[str, Any],
    mask_coverage: dict[str, Any],
    stage_times: dict[str, float],
    stage_roots: dict[str, Path],
    terminal_status: str,
) -> dict[str, Any]:
    scene = str(args.scene)
    input_stride = int(args.input_stride)
    expected_frames = _scene_stride_frames(scene, input_stride)
    d4rt_frame_gate = {
        "expected_stride_frame_count": len(expected_frames),
        "d4rt_unique_frame_count": int(conversion_summary.get("unique_frame_count") or 0),
        "d4rt_frame_count_eq_expected": int(conversion_summary.get("unique_frame_count") or 0) == len(expected_frames),
        "d4rt_frame_diff_all_eq_input_stride": bool(conversion_summary.get("frame_diff_all_eq_input_stride")),
        "observation_unique_frame_count": int(observation_summary.get("unique_frame_count") or 0),
        "observation_unique_frame_count_eq_expected": int(observation_summary.get("unique_frame_count") or 0) == len(expected_frames),
    }
    d4rt_frame_gate["pass"] = bool(
        d4rt_frame_gate["d4rt_frame_count_eq_expected"]
        and d4rt_frame_gate["d4rt_frame_diff_all_eq_input_stride"]
        and d4rt_frame_gate["observation_unique_frame_count_eq_expected"]
    )

    outputs: dict[str, dict[str, Any]] = {}
    if "carrier_cache" in stage_roots:
        outputs["carrier_cache_conversion_summary"] = _stage_file(stage_roots["carrier_cache"] / "conversion_summary.json", "json")
        outputs["converted_chunk_rows"] = _stage_file(stage_roots["carrier_cache"] / "converted_chunk_rows.csv", "csv")
    if "observation_tables" in stage_roots:
        observation_root = stage_roots["observation_tables"]
        outputs["observation_summary"] = _stage_file(observation_root / "observation_table_summary.json", "json")
        outputs["carrier_observation_table"] = _stage_file(observation_root / "carrier_observation_table.csv", "csv")
        outputs["mask_observation_table"] = _stage_file(observation_root / "mask_observation_table.csv", "csv")
        outputs["observation_window_rows"] = _stage_file(observation_root / "observation_window_rows.csv", "csv")
    stage_specs = {
        "mask_component_support": [
            ("support_summary", "support_summary.json", "json"),
            ("mask_component_support_rows", "mask_component_support_rows.csv", "csv"),
            ("mask_summary_rows", "mask_summary_rows.csv", "csv"),
        ],
        "chunk_universe": [
            ("chunk_summary", "chunk_summary.json", "json"),
            ("chunk_rows", "chunk_rows.csv", "csv"),
            ("chunk_component_rows", "chunk_component_rows.csv", "csv"),
            ("chunk_mask_rows", "chunk_mask_rows.csv", "csv"),
        ],
        "representative_observations": [
            ("representative_summary", "representative_summary.json", "json"),
            ("representative_mask_rows", "representative_mask_rows.csv", "csv"),
        ],
        "reprojection_ledger": [
            ("reprojection_summary", "reprojection_summary.json", "json"),
            ("candidate_rows", "candidate_rows.csv", "csv"),
            ("reprojection_ledger_rows", "reprojection_ledger_rows.csv", "csv"),
        ],
        "local_objectlets": [
            ("local_objectlet_summary", "local_objectlet_summary.json", "json"),
            ("objectlet_rows", "objectlet_rows.csv", "csv"),
            ("selection_metric_rows", "selection_metric_rows.csv", "csv"),
        ],
        "local_reproduction": [
            ("local_reproduction_summary", "local_reproduction_summary.json", "json"),
            ("local_metric_rows", "local_metric_rows.csv", "csv"),
            ("local_variant_summary_rows", "local_variant_summary_rows.csv", "csv"),
        ],
    }
    for stage_name, specs in stage_specs.items():
        if stage_name not in stage_roots:
            continue
        root = stage_roots[stage_name]
        for key, file_name, kind in specs:
            outputs[f"{stage_name}.{key}"] = _stage_file(root / file_name, kind)

    summary_gates = {
        stage_name: _read_summary_gate(root / file_name)
        for stage_name, root, file_name in [
            ("v47_observation_tables", stage_roots.get("observation_tables", output_root), "observation_table_summary.json"),
            ("v53_mask_component_support", stage_roots.get("mask_component_support", output_root), "support_summary.json"),
            ("v53_chunk_universe", stage_roots.get("chunk_universe", output_root), "chunk_summary.json"),
            ("v53_representative_observations", stage_roots.get("representative_observations", output_root), "representative_summary.json"),
            ("v53_reprojection_ledger", stage_roots.get("reprojection_ledger", output_root), "reprojection_summary.json"),
            ("v53_local_objectlets", stage_roots.get("local_objectlets", output_root), "local_objectlet_summary.json"),
            ("v54_local_reproduction", stage_roots.get("local_reproduction", output_root), "local_reproduction_summary.json"),
        ]
        if stage_name.split("_", 1)[-1] or True
    }

    mask_table_frames = _csv_unique_frames(stage_roots.get("observation_tables", output_root) / "mask_observation_table.csv")
    d4rt_geometry_gate = (
        d4rt_geometry_summary.get("gate", {}) if isinstance(d4rt_geometry_summary, dict) else {"pass": False}
    )
    pipeline_ap_ready = bool(d4rt_frame_gate["pass"] and mask_coverage["gate"]["pass"] and d4rt_geometry_gate.get("pass", False))
    return {
        "phase": "v65_soma_fullscene_pipeline",
        "created_at": utc_now(),
        "terminal_status": terminal_status,
        "scene": scene,
        "input_stride": input_stride,
        "confidence_threshold": float(args.confidence_threshold),
        "visibility_threshold": float(args.visibility_threshold),
        "min_mask_area": int(args.min_mask_area),
        "feature_backend": feature_backend or "disabled",
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "d4rt_source": d4rt_source,
        "d4rt_chunk_dir": _rel(chunk_dir),
        "d4rt_geometry_materialization_summary": d4rt_geometry_summary,
        "d4rt_coordinate_contract": (
            d4rt_geometry_summary.get("coordinate_contract", {}) if isinstance(d4rt_geometry_summary, dict) else {}
        ),
        "mask_source": mask_source,
        "mask_root_arg": str(args.mask_root or ""),
        "mask_cache_root": str(args.mask_cache_root or ""),
        "mask_root_for_pipeline": mask_root_for_pipeline,
        "resolved_mask_dir": mask_coverage.get("mask_dir"),
        "output_root": _rel(output_root),
        "carrier_id_policy": {
            "globalize_carrier_ids": bool(args.globalize_carrier_ids),
            "source_duplicate_carrier_id_count_across_chunks": int(
                conversion_summary.get("source_duplicate_carrier_id_count_across_chunks") or 0
            ),
            "reason": "avoid accidental cross-chunk identity merge when source D4RT chunk carrier ids collide",
        },
        "d4rt_fullscene_stride_gate": d4rt_frame_gate,
        "soma_2d_mask_stride_gate": mask_coverage["gate"],
        "mask_frame_coverage": mask_coverage,
        "mask_observation_table_unique_frame_count": len(mask_table_frames),
        "mask_observation_table_first_frames": mask_table_frames[:20],
        "mask_observation_table_last_frames": mask_table_frames[-20:],
        "pipeline_gate": {
            "d4rt_geometry_materialization_gate_pass": bool(d4rt_geometry_gate.get("pass", False)),
            "d4rt_fullscene_stride_gate_pass": bool(d4rt_frame_gate["pass"]),
            "soma_2d_mask_stride_gate_pass": bool(mask_coverage["gate"]["pass"]),
            "ap_ready": pipeline_ap_ready,
            "pass": pipeline_ap_ready,
        },
        "do_not_use_for_ap": not pipeline_ap_ready,
        "do_not_use_for_ap_reason": None
        if pipeline_ap_ready
        else "SOMA 2D mask support is not full stride-5 scene coverage; AP/materialization from this run is diagnostic only.",
        "old_ap_invalidated_reason": (
            "Previous v65 AP/visualization rows were produced from scattered historical artifacts whose SOMA object support "
            "covered only a small frame subset. This pipeline pins all stage inputs to one run root and exposes coverage/hash gates."
        ),
        "mask_materialization_summary": mask_materialization_summary,
        "conversion_summary": conversion_summary,
        "observation_summary": observation_summary,
        "stage_times_sec": stage_times,
        "stage_roots": {name: _rel(path) for name, path in stage_roots.items()},
        "stage_outputs": outputs,
        "stage_summary_gates": summary_gates,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one auditable full-scene SOMA pipeline from D4RT stride chunks.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--input-stride", type=int, default=5)
    parser.add_argument(
        "--d4rt-source",
        choices=["pipeline_cache", "external"],
        default="pipeline_cache",
        help="pipeline_cache generates/validates D4RT geometry chunks inside this pipeline; external uses --d4rt-chunk-dir.",
    )
    parser.add_argument(
        "--d4rt-chunk-dir",
        default="outputs/audit/v65_gt_loger_d4rt_stride5_allconf_rgb/stride_5/chunks",
    )
    parser.add_argument("--d4rt-cache-root", default="outputs/cache/v65_d4rt_stride_geometry")
    parser.add_argument(
        "--d4rt-cache-policy",
        choices=["cache_or_generate", "cache_only", "force_generate"],
        default="cache_or_generate",
    )
    parser.add_argument("--d4rt-gpus", default="6,7")
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--d4rt-device", default="cuda")
    parser.add_argument("--d4rt-scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--d4rt-chunk-size", type=int, default=32)
    parser.add_argument("--d4rt-overlap-frames", type=int, default=3)
    parser.add_argument("--d4rt-max-frames", type=int, default=0)
    parser.add_argument("--d4rt-grid-size", type=int, default=8)
    parser.add_argument("--d4rt-grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--d4rt-query-chunk-size", type=int, default=4096)
    parser.add_argument("--d4rt-min-visibility", type=float, default=0.0)
    parser.add_argument("--d4rt-min-confidence", type=float, default=0.0)
    parser.add_argument("--d4rt-uv-radius", type=float, default=0.002)
    parser.add_argument("--d4rt-max-matches-per-frame", type=int, default=4096)
    parser.add_argument("--d4rt-fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--d4rt-max-sim3-anchors", type=int, default=120000)
    parser.add_argument("--d4rt-max-metric-points", type=int, default=500000)
    parser.add_argument("--d4rt-max-gt-metric-points", type=int, default=250000)
    parser.add_argument("--d4rt-max-visual-points-per-stride", type=int, default=1000000)
    parser.add_argument("--output-root", default="outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02")
    parser.add_argument(
        "--mask-source",
        choices=["pipeline_cache", "external"],
        default="pipeline_cache",
        help="pipeline_cache generates/validates CropFormer masks by D4RT chunk; external uses --mask-root explicitly.",
    )
    parser.add_argument(
        "--mask-root",
        default="",
        help="Only used with --mask-source external. Accepts processed-root, scene-root, output_Cropformer/mask, or mask dir.",
    )
    parser.add_argument("--mask-cache-root", default="outputs/cache/v65_cropformer_chunk_masks")
    parser.add_argument(
        "--mask-cache-policy",
        choices=["cache_or_generate", "cache_only", "force_generate"],
        default="cache_or_generate",
    )
    parser.add_argument("--mask-gpus", default="6,7")
    parser.add_argument("--mask-cropformer-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mask-chunk-parallelism", type=int, default=0)
    parser.add_argument("--mask-max-chunks", type=int, default=0)
    parser.add_argument("--stop-after-mask-materialization", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--globalize-carrier-ids", type=int, default=1)
    parser.add_argument("--visibility-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--feature-backend", default="")
    parser.add_argument("--max-union-unique-carriers", type=int, default=32)
    parser.add_argument("--support-visible-tau", type=float, default=0.05)
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--chunk-size-selected", type=int, default=32)
    parser.add_argument("--chunk-stride-selected", type=int, default=16)
    parser.add_argument("--max-selected-ratio", type=float, default=0.60)
    parser.add_argument("--representative-variant", default="K8_underseg_capped_partial_repair")
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=4000)
    parser.add_argument("--skip-no-related-measurement", type=int, default=1)
    parser.add_argument("--include-repeated-support-candidates", type=int, default=1)
    parser.add_argument("--repeated-support-min-frames", type=int, default=4)
    parser.add_argument("--repeated-support-min-components", type=int, default=2)
    parser.add_argument("--repeated-support-min-w-visible", type=float, default=0.50)
    parser.add_argument("--repeated-support-max-components", type=int, default=128)
    parser.add_argument("--repeated-support-max-groups-per-scene", type=int, default=80)
    parser.add_argument("--max-candidate-conflict-rate", type=float, default=0.18)
    parser.add_argument("--enable-weak-support-restitution", type=int, default=0)
    parser.add_argument("--stop-after-observation-if-mask-incomplete", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_pipeline(args)
    print(
        {
            "pipeline_summary": _rel(_project(args.output_root) / "pipeline_summary.json"),
            "pipeline_gate": summary.get("pipeline_gate"),
            "mask_frame_coverage": summary.get("mask_frame_coverage"),
        }
    )


if __name__ == "__main__":
    main()
