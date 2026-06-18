#!/usr/bin/env python3
"""Build ACL2 v70 RADIO/RADSeg sidecar caches from real KITTI RGB frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from v70_radio_sidecar_common import (
    apply_pca,
    component_boundary,
    component_count,
    connected_components_from_features,
    extract_radio_feature,
    find_stage_chunk,
    finite_float,
    fit_pca,
    image_path_for_frame,
    l2_normalize,
    load_radseg_encoder,
    load_stage_semantic,
    locate_default_radio_checkpoint,
    neighbor_contrast,
    parse_chunks,
    pool_semantic_to_grid,
    robust01,
    sha256_file,
    temporal_stability,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--stage-c-cache", type=Path, required=True)
    parser.add_argument("--radio-vipe-root", type=Path, required=True)
    parser.add_argument("--target-chunks", required=True)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--lang-model", default="siglip2")
    parser.add_argument("--seed", type=int, default=7002)
    parser.add_argument("--max-fit-tokens", type=int, default=120000)
    parser.add_argument("--max-frames-per-chunk", type=int, default=0, help="Debug only. 0 means all manifest frames.")
    return parser.parse_args()


def _as_torch(value: np.ndarray, *, dtype: str = "float32") -> Any:
    import torch

    if dtype == "float16":
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).half()
    if dtype == "int32":
        return torch.from_numpy(np.asarray(value, dtype=np.int32))
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def _save_visual_audit(chunk_dir: Path, frame_ids: list[int], pca_feat: np.ndarray, components: np.ndarray, stability: np.ndarray) -> list[dict[str, Any]]:
    from PIL import Image

    out_dir = chunk_dir / "visual_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    if pca_feat.shape[0] == 0:
        return []
    picks = sorted({0, int(pca_feat.shape[0] // 2), int(pca_feat.shape[0] - 1)})
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(12345)
    max_comp = int(max(1, components.max()))
    palette = rng.integers(32, 255, size=(max_comp + 1, 3), dtype=np.uint8)
    palette[0] = 0
    for idx in picks:
        feat3 = pca_feat[idx, :, :, :3]
        lo = np.percentile(feat3.reshape(-1, 3), 1, axis=0)
        hi = np.percentile(feat3.reshape(-1, 3), 99, axis=0)
        rgb = np.clip((feat3 - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        pca_path = out_dir / f"frame_{frame_ids[idx]:06d}_radio_pca_rgb.png"
        comp_path = out_dir / f"frame_{frame_ids[idx]:06d}_radio_component.png"
        stab_path = out_dir / f"frame_{frame_ids[idx]:06d}_radio_stability.png"
        Image.fromarray((rgb * 255.0).round().astype(np.uint8)).resize((pca_feat.shape[2] * 8, pca_feat.shape[1] * 8)).save(pca_path)
        Image.fromarray(palette[components[idx].clip(0, max_comp)]).resize((pca_feat.shape[2] * 8, pca_feat.shape[1] * 8)).save(comp_path)
        gray = np.repeat((stability[idx].clip(0, 1)[:, :, None] * 255.0).round().astype(np.uint8), 3, axis=2)
        Image.fromarray(gray).resize((pca_feat.shape[2] * 8, pca_feat.shape[1] * 8)).save(stab_path)
        rows.append({"frame": frame_ids[idx], "radio_pca_rgb": str(pca_path), "component": str(comp_path), "stability": str(stab_path)})
    return rows


def _extract_chunk_features(model: Any, chunk: Any, image_dir: Path, device: str, max_frames: int) -> dict[str, Any]:
    frame_ids = list(range(chunk.start_frame, chunk.end_frame))
    if max_frames and max_frames > 0:
        frame_ids = frame_ids[: int(max_frames)]
    features: list[np.ndarray] = []
    image_shapes: list[tuple[int, int]] = []
    for frame_id in frame_ids:
        path = image_path_for_frame(image_dir, frame_id)
        feat = extract_radio_feature(model, path, device)
        features.append(feat)
        from PIL import Image

        with Image.open(path) as im:
            image_shapes.append((int(im.height), int(im.width)))
    if not features:
        raise ValueError(f"no frames extracted for chunk {chunk.chunk_id}")
    return {"frame_ids": frame_ids, "features": np.stack(features, axis=0).astype(np.float32), "image_shapes": image_shapes}


def _sample_features(features: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    flat = features.reshape(-1, int(features.shape[-1]))
    if flat.shape[0] <= int(max_rows):
        return flat
    idx = rng.choice(flat.shape[0], size=int(max_rows), replace=False)
    return flat[idx]


def main() -> None:
    args = parse_args()
    target_chunks = parse_chunks(args.target_chunks)
    if not target_chunks:
        raise ValueError("--target-chunks is empty")
    checkpoint = args.checkpoint or locate_default_radio_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("no local RADIO/RADSeg checkpoint found; R1 must be blocked, not faked")
    if not args.radio_vipe_root.exists():
        raise FileNotFoundError(f"RADIO-ViPE root missing: {args.radio_vipe_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_radseg_encoder(args.radio_vipe_root, checkpoint, args.device, args.lang_model, amp=False)
    chunk_payloads: list[dict[str, Any]] = []
    pca_samples: list[np.ndarray] = []
    per_chunk_sample_cap = max(4096, int(args.max_fit_tokens // max(len(target_chunks), 1)))
    for chunk_id in target_chunks:
        chunk = find_stage_chunk(args.stage_c_cache, chunk_id)
        extracted = _extract_chunk_features(model, chunk, args.image_dir, args.device, args.max_frames_per_chunk)
        features = extracted["features"]
        pca_samples.append(_sample_features(features, per_chunk_sample_cap, args.seed + int(chunk_id)))
        chunk_payloads.append({"chunk": chunk, **extracted})

    all_samples = np.concatenate(pca_samples, axis=0)
    pca = fit_pca(all_samples, args.pca_dim, args.max_fit_tokens, args.seed)
    import torch

    pca_path = args.out_dir / "pca_basis.pt"
    torch.save(
        {
            "format": "radio_sidecar_pca_v1",
            "source": "third_party/RADIO-ViPE/RADSeg",
            "seq": str(args.seq),
            "target_chunks": target_chunks,
            "pca_dim_requested": int(args.pca_dim),
            "pca_dim_actual": int(pca["basis"].shape[0]),
            "mean": torch.from_numpy(pca["mean"]),
            "basis": torch.from_numpy(pca["basis"]),
            "explained_variance_ratio": torch.from_numpy(pca["explained_variance_ratio"]),
            "provenance": {
                "created_at": utc_now(),
                "python": sys.executable,
                "checkpoint": checkpoint,
                "radio_vipe_root": str(args.radio_vipe_root),
                "max_fit_tokens": int(args.max_fit_tokens),
                "pca_basis_source": "v70_fit_on_kitti01_sampled_frames",
            },
        },
        pca_path,
    )
    pca_sha = sha256_file(pca_path)

    manifest_chunks: list[dict[str, Any]] = []
    for row in chunk_payloads:
        chunk = row["chunk"]
        features = row["features"]
        frame_ids = row["frame_ids"]
        image_shapes = row["image_shapes"]
        pca_feat = apply_pca(features, pca)
        pca_norm = l2_normalize(pca_feat)
        components: list[np.ndarray] = []
        boundaries: list[np.ndarray] = []
        contrast_rows: list[dict[str, Any]] = []
        for frame_idx in range(int(pca_norm.shape[0])):
            comp = connected_components_from_features(
                pca_norm[frame_idx],
                seed=args.seed + int(chunk.chunk_id) * 1000 + frame_idx,
                min_clusters=8,
                max_clusters=64,
                max_components=200,
            )
            boundary = component_boundary(comp)
            components.append(comp)
            boundaries.append(boundary)
            contrast = neighbor_contrast(pca_norm[frame_idx], comp)
            contrast_rows.append({"frame": frame_ids[frame_idx], "component_count": component_count(comp), **contrast})
        comp_arr = np.stack(components, axis=0).astype(np.int32)
        boundary_arr = np.stack(boundaries, axis=0).astype(np.float32)
        interior = (1.0 - boundary_arr).clip(0.0, 1.0).astype(np.float32)
        temp = temporal_stability(features)

        semantic = load_stage_semantic(chunk.masklet_path)
        label_names = [str(x) for x in semantic.get("label_names", [])]
        pooled = pool_semantic_to_grid(
            semantic["label_maps"][: len(frame_ids)],
            semantic.get("confidence_maps")[: len(frame_ids)] if semantic.get("confidence_maps") is not None else None,
            label_names,
            (int(pca_feat.shape[1]), int(pca_feat.shape[2])),
        )
        radio_confidence = np.clip(temp["temporal_stability"] * (0.5 + 0.5 * pooled["confidence"]), 0.0, 1.0).astype(np.float32)
        static_score = np.clip(radio_confidence * interior * (1.0 - pooled["dynamic"]) * (1.0 - pooled["sky"]), 0.0, 1.0).astype(np.float32)
        dynamic_score = np.maximum(1.0 - temp["temporal_stability"], pooled["dynamic"]).astype(np.float32)
        lowtrust = np.maximum.reduce([1.0 - pooled["confidence"], boundary_arr * 0.5, pooled["lowtrust_label"] * 0.5]).astype(np.float32)

        component_counts = [int(x["component_count"]) for x in contrast_rows]
        boundary_contrasts = [finite_float(x.get("boundary_contrast")) for x in contrast_rows]
        valid_contrasts = [x for x in boundary_contrasts if x is not None]
        chunk_dir_name = f"chunk_{chunk.chunk_id:03d}_{chunk.start_frame:06d}_{chunk.end_frame:06d}"
        chunk_out = args.out_dir / chunk_dir_name
        chunk_out.mkdir(parents=True, exist_ok=True)
        visual_rows = _save_visual_audit(chunk_out, frame_ids, pca_feat, comp_arr, temp["temporal_stability"])
        payload = {
            "format": "radio_sidecar_v1",
            "source": "third_party/RADIO-ViPE/RADSeg",
            "seq": str(args.seq),
            "chunk_id": int(chunk.chunk_id),
            "global_start_frame": int(chunk.start_frame),
            "global_end_frame": int(chunk.end_frame),
            "frame_height": int(image_shapes[0][0]),
            "frame_width": int(image_shapes[0][1]),
            "radio_feat_pca": _as_torch(pca_feat, dtype="float16"),
            "radio_feat_norm": _as_torch(np.linalg.norm(features, axis=-1), dtype="float32"),
            "radio_confidence": _as_torch(radio_confidence, dtype="float32"),
            "object_component_id": _as_torch(comp_arr, dtype="int32"),
            "object_boundary_score": _as_torch(boundary_arr, dtype="float32"),
            "object_interior_score": _as_torch(interior, dtype="float32"),
            "radio_static_score": _as_torch(static_score, dtype="float32"),
            "radio_dynamic_score": _as_torch(dynamic_score, dtype="float32"),
            "radio_sky_context_score": _as_torch(pooled["sky"], dtype="float32"),
            "radio_lowtrust_score": _as_torch(lowtrust, dtype="float32"),
            "temporal_stability": _as_torch(temp["temporal_stability"], dtype="float32"),
            "temporal_embedding_mean_sim": _as_torch(temp["temporal_embedding_mean_sim"], dtype="float32"),
            "temporal_embedding_var": _as_torch(temp["temporal_embedding_var"], dtype="float32"),
            "patch_grid": [int(pca_feat.shape[1]), int(pca_feat.shape[2])],
            "pca_dim": int(pca_feat.shape[-1]),
            "pca_basis_sha256": pca_sha,
            "provenance": {
                "created_at": utc_now(),
                "python": sys.executable,
                "cmd": " ".join(sys.argv),
                "checkpoint": checkpoint,
                "radio_vipe_root": str(args.radio_vipe_root),
                "image_dir": str(args.image_dir),
                "stage_c_cache": str(args.stage_c_cache),
                "stage_chunk_dir": str(chunk.chunk_dir),
                "semantic_source": semantic.get("source", ""),
                "pca_basis_source": "v70_fit_on_kitti01_sampled_frames",
                "component_method": "MiniBatchKMeans on RADIO PCA features; connected split capped at 200 components",
                "risk_fields_note": "dynamic/sky/lowtrust combine RADIO temporal stability with existing semantic label/confidence context; not GT.",
            },
            "debug": {
                "frame_ids": frame_ids,
                "raw_feature_shape": list(features.shape),
                "pca_feature_shape": list(pca_feat.shape),
                "pca_explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
                "component_count_min": min(component_counts) if component_counts else None,
                "component_count_max": max(component_counts) if component_counts else None,
                "component_count_mean": float(np.mean(component_counts)) if component_counts else None,
                "object_boundary_contrast_mean": float(np.mean(valid_contrasts)) if valid_contrasts else None,
                "feature_all_finite": bool(np.isfinite(features).all()),
                "pca_all_finite": bool(np.isfinite(pca_feat).all()),
                "temporal_stability_min": float(np.min(temp["temporal_stability"])),
                "temporal_stability_max": float(np.max(temp["temporal_stability"])),
                "semantic_label_names": label_names,
                "visual_audit": visual_rows,
                "per_frame_contrast": contrast_rows,
            },
        }
        sidecar_path = chunk_out / "radio_sidecar.pt"
        torch.save(payload, sidecar_path)
        manifest_chunks.append(
            {
                "chunk_id": int(chunk.chunk_id),
                "start_frame": int(chunk.start_frame),
                "end_frame": int(chunk.end_frame),
                "frames_saved": len(frame_ids),
                "sidecar_path": str(sidecar_path),
                "component_count_min": payload["debug"]["component_count_min"],
                "component_count_max": payload["debug"]["component_count_max"],
                "object_boundary_contrast_mean": payload["debug"]["object_boundary_contrast_mean"],
                "temporal_stability_min": payload["debug"]["temporal_stability_min"],
                "temporal_stability_max": payload["debug"]["temporal_stability_max"],
                "feature_all_finite": payload["debug"]["feature_all_finite"],
                "pca_all_finite": payload["debug"]["pca_all_finite"],
            }
        )

    manifest = {
        "format": "radio_sidecar_manifest_v1",
        "created_at": utc_now(),
        "seq": str(args.seq),
        "target_chunks": target_chunks,
        "out_dir": str(args.out_dir),
        "pca_basis": str(pca_path),
        "pca_basis_sha256": pca_sha,
        "pca_dim_requested": int(args.pca_dim),
        "pca_dim_actual": int(pca["basis"].shape[0]),
        "checkpoint": checkpoint,
        "radio_vipe_root": str(args.radio_vipe_root),
        "chunks": manifest_chunks,
    }
    write_json(args.out_dir / "sidecar_manifest.json", manifest)
    print(json.dumps({"out_dir": str(args.out_dir), "chunks": len(manifest_chunks), "pca_basis_sha256": pca_sha}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

