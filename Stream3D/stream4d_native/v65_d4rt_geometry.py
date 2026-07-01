from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .sim3 import Sim3Transform, apply_sim3_to_xyz


D4RT_COORDINATE_MODES = ("carrier_raw", "chunk_self_stitched", "chunk_final_gt_sim3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_stream3d_path(path: str | Path, *, stream3d_root: Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == stream3d_root.name:
        return stream3d_root.parent / path_obj
    return stream3d_root / path_obj


def _chunk_transform_from_payload(payload: np.lib.npyio.NpzFile) -> Sim3Transform:
    return Sim3Transform(
        scale=float(np.asarray(payload["transform_scale_to_scene"], dtype=np.float64).reshape(-1)[0]),
        rot=np.asarray(payload["transform_rot_to_scene"], dtype=np.float64).reshape(3, 3),
        trans=np.asarray(payload["transform_trans_to_scene"], dtype=np.float64).reshape(3),
    )


def _final_gt_transform_from_summary(path: Path) -> Sim3Transform:
    summary = _read_json(path)
    transform = summary.get("final_gt_sim3_transform") or {}
    if transform:
        return Sim3Transform(
            scale=float(transform["scale"]),
            rot=np.asarray(transform["rot"], dtype=np.float64).reshape(3, 3),
            trans=np.asarray(transform["trans"], dtype=np.float64).reshape(3),
        )
    fit = summary.get("final_gt_sim3") or {}
    if not fit:
        raise ValueError(f"missing final_gt_sim3 in {path}")
    return Sim3Transform(
        scale=float(fit["scale_d4rt_to_gt"]),
        rot=np.asarray(fit["rotation_d4rt_to_gt"], dtype=np.float64).reshape(3, 3),
        trans=np.asarray(fit["translation_d4rt_to_gt"], dtype=np.float64).reshape(3),
    )


def _fallback_final_gt_transform_from_summary(path: Path) -> Sim3Transform:
    """Load old summaries that did not persist rotation/translation arrays.

    Older v65 geometry summaries kept final Sim3 scalar diagnostics but not the
    matrix/vector itself. In that case the exact transform is not recoverable
    from the summary alone, so callers must use the stored final visual points
    or regenerate the geometry run. This function exists to produce a clear
    error instead of silently inventing a transform.
    """

    return _final_gt_transform_from_summary(path)


def infer_stride_summary_from_pipeline(
    *,
    pipeline_root: Path,
    scene: str,
    stream3d_root: Path,
) -> Path | None:
    scene_dir = pipeline_root / "carrier_cache" / scene
    manifest_paths = sorted(scene_dir.glob("carriers_window*_manifest.json"))
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        source = str(manifest.get("source_chunk_npz") or "").strip()
        if not source:
            continue
        source_path = _resolve_stream3d_path(source, stream3d_root=stream3d_root)
        candidate = source_path.parent.parent / "stride_summary.json"
        if candidate.exists():
            return candidate
    return None


def load_d4rt_geometry_frames(
    *,
    pipeline_root: Path,
    scene: str,
    stream3d_root: Path,
    coordinate_mode: str,
    d4rt_stride_summary: str | Path | None = None,
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    if coordinate_mode not in D4RT_COORDINATE_MODES:
        raise ValueError(f"unknown D4RT coordinate mode {coordinate_mode!r}; expected one of {D4RT_COORDINATE_MODES}")

    scene_dir = pipeline_root / "carrier_cache" / scene
    carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"missing AP pipeline carrier cache: {scene_dir}")

    stride_summary_path: Path | None = None
    final_transform: Sim3Transform | None = None
    if coordinate_mode == "chunk_final_gt_sim3":
        if d4rt_stride_summary is not None and str(d4rt_stride_summary).strip():
            stride_summary_path = _resolve_stream3d_path(d4rt_stride_summary, stream3d_root=stream3d_root)
        else:
            stride_summary_path = infer_stride_summary_from_pipeline(
                pipeline_root=pipeline_root,
                scene=scene,
                stream3d_root=stream3d_root,
            )
        if stride_summary_path is None:
            raise FileNotFoundError("could not infer stride_summary.json for chunk_final_gt_sim3 geometry mode")
        final_transform = _fallback_final_gt_transform_from_summary(stride_summary_path)

    diag: dict[str, Any] = {
        "d4rt_coordinate_mode": coordinate_mode,
        "d4rt_geometry_source": "pipeline_carrier_cache_xyz_ref" if coordinate_mode == "carrier_raw" else "pipeline_manifest_source_chunk_npz",
        "d4rt_applies_chunk_self_stitch": bool(coordinate_mode in {"chunk_self_stitched", "chunk_final_gt_sim3"}),
        "d4rt_applies_final_gt_sim3": bool(coordinate_mode == "chunk_final_gt_sim3"),
        "d4rt_stride_summary": str(stride_summary_path) if stride_summary_path is not None else "",
        "d4rt_stride_summary_sha256": _sha256(stride_summary_path) if stride_summary_path is not None else "",
        "carrier_cache_window_count": int(len(carrier_paths)),
    }

    def iterator() -> Iterator[dict[str, Any]]:
        for carrier_path in carrier_paths:
            manifest_path = carrier_path.with_name(f"{carrier_path.stem}_manifest.json")
            manifest = _read_json(manifest_path)
            frame_ids = [int(value) for value in manifest.get("frame_ids", [])]
            with np.load(carrier_path) as carrier_payload:
                uv = np.asarray(carrier_payload["uv_pred"], dtype=np.float32)
                valid = np.asarray(carrier_payload["valid"], dtype=bool)
                confidence = np.asarray(carrier_payload["confidence_prob"], dtype=np.float32)
                visibility = np.asarray(carrier_payload["visibility_prob"], dtype=np.float32)
                carrier_xyz = np.asarray(carrier_payload["xyz_ref"], dtype=np.float32)

            xyz = carrier_xyz
            source_chunk_path: Path | None = None
            source_chunk_sha = str(manifest.get("source_chunk_sha256") or "")
            if coordinate_mode in {"chunk_self_stitched", "chunk_final_gt_sim3"}:
                source = str(manifest.get("source_chunk_npz") or "").strip()
                if not source:
                    raise ValueError(f"missing source_chunk_npz in {manifest_path}")
                source_chunk_path = _resolve_stream3d_path(source, stream3d_root=stream3d_root)
                with np.load(source_chunk_path) as chunk_payload:
                    chunk_frame_ids = np.asarray(chunk_payload["frame_ids"], dtype=np.int64).tolist()
                    if [int(value) for value in chunk_frame_ids] != frame_ids:
                        raise ValueError(f"frame ids differ between {carrier_path} and {source_chunk_path}")
                    xyz = np.asarray(chunk_payload["xyz"], dtype=np.float32)
                    chunk_transform = _chunk_transform_from_payload(chunk_payload)
                if xyz.shape != carrier_xyz.shape:
                    raise ValueError(f"xyz shape differs between {carrier_path} and {source_chunk_path}: {carrier_xyz.shape} vs {xyz.shape}")
                xyz = apply_sim3_to_xyz(xyz, transform=chunk_transform)
                if final_transform is not None:
                    xyz = apply_sim3_to_xyz(xyz, transform=final_transform)

            if len(frame_ids) != xyz.shape[0]:
                raise ValueError(f"frame manifest length mismatch in {carrier_path}")
            yield {
                "carrier_path": carrier_path,
                "manifest_path": manifest_path,
                "source_chunk_path": source_chunk_path,
                "source_chunk_sha256": source_chunk_sha,
                "frame_ids": frame_ids,
                "xyz": xyz,
                "uv": uv,
                "valid": valid,
                "confidence": confidence,
                "visibility": visibility,
            }

    return iterator(), diag
