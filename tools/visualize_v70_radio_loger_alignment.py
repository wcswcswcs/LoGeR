#!/usr/bin/env python3
"""Visualize and quantify v70 RADIO sidecar alignment with semantic and LoGeR carriers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np

from v70_radio_sidecar_common import (
    component_count,
    find_stage_chunk,
    finite_float,
    load_stage_semantic,
    neighbor_contrast,
    pearson,
    pool_semantic_to_grid,
    resize_linear,
    resize_nearest,
    robust01,
    torch_load,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-sidecar-dir", type=Path, required=True)
    parser.add_argument("--stage-c-cache", type=Path, required=True)
    parser.add_argument("--v68-feature-dir", type=Path, default=Path(""))
    parser.add_argument("--overlap-pairs-dir", type=Path, default=Path(""))
    parser.add_argument("--image-dir", type=Path, default=Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-chunks", default="")
    return parser.parse_args()


def _load_sidecars(sidecar_dir: Path, target_chunks: list[int] | None) -> list[tuple[Path, Mapping[str, Any]]]:
    rows: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(sidecar_dir.glob("chunk_*/radio_sidecar.pt")):
        payload = torch_load(path)
        if not isinstance(payload, Mapping):
            continue
        chunk_id = int(payload.get("chunk_id", -1))
        if target_chunks is not None and chunk_id not in target_chunks:
            continue
        rows.append((path, payload))
    if not rows:
        raise FileNotFoundError(f"no sidecar radio_sidecar.pt files found in {sidecar_dir}")
    return rows


def _tensor_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _tensor_int_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().long().numpy()
    return np.asarray(value, dtype=np.int64)


def _nmi_ari(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        x = np.asarray(a).reshape(-1)
        y = np.asarray(b).reshape(-1)
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return None, None
        return float(normalized_mutual_info_score(x, y)), float(adjusted_rand_score(x, y))
    except Exception:
        return None, None


def _auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y_true).reshape(-1).astype(np.int32)
        s = np.asarray(score).reshape(-1).astype(np.float32)
        finite = np.isfinite(s)
        y = y[finite]
        s = s[finite]
        if len(np.unique(y)) != 2:
            return None
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def _gram_row_instability(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    t, h, w, d = x.shape
    if t < 2:
        return np.zeros((h, w), dtype=np.float32)
    flat = x.reshape(t, h * w, d)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-6)
    grams = []
    for idx in range(t):
        g = flat[idx] @ flat[idx].T
        grams.append(g.astype(np.float32))
    stack = np.stack(grams, axis=0)
    score = np.var(stack, axis=0).mean(axis=1).reshape(h, w)
    return robust01(score)


def _loger_alignment(v68_feature_dir: Path, chunk_id: int, radio_boundary: np.ndarray, radio_stability: np.ndarray) -> dict[str, Any]:
    if not v68_feature_dir or str(v68_feature_dir) == ".":
        return {"available": False, "reason": "v68_feature_dir_not_provided"}
    path = v68_feature_dir / f"chunk_{chunk_id:03d}.pt"
    if not path.exists():
        return {"available": False, "reason": f"missing {path}"}
    payload = torch_load(path)
    tensor = payload.get("tap::global_k_raw_patchvec_layers") if isinstance(payload, Mapping) else None
    if tensor is None or not hasattr(tensor, "detach"):
        return {"available": False, "reason": "missing tap::global_k_raw_patchvec_layers"}
    arr = tensor.detach().cpu().float().numpy()
    # [T, selected_layers, H, W, D]
    rows: list[dict[str, Any]] = []
    selected = list((payload.get("taps") or {}).get("global_k_raw_patchvec_layers", {}).get("selected_layers") or [])
    for pos in range(arr.shape[1]):
        layer = int(selected[pos]) if pos < len(selected) else int(pos)
        score = _gram_row_instability(arr[:, pos])
        rb = resize_linear(radio_boundary.mean(axis=0), score.shape)
        rs = resize_linear(radio_stability.mean(axis=0), score.shape)
        rows.append(
            {
                "layer": layer,
                "corr_boundary_to_global_k_gram": pearson(rb, score),
                "corr_inv_stability_to_global_k_gram": pearson(1.0 - rs, score),
                "gram_score_mean": float(np.mean(score)),
            }
        )
    vals = [abs(float(r["corr_boundary_to_global_k_gram"])) for r in rows if r.get("corr_boundary_to_global_k_gram") is not None]
    return {"available": True, "path": str(path), "rows": rows, "max_abs_corr_boundary_to_global_k_gram": max(vals) if vals else None}


def _palette(labels: np.ndarray) -> np.ndarray:
    n = int(max(1, labels.max(initial=0) + 1))
    rng = np.random.default_rng(7003)
    colours = rng.integers(32, 255, size=(n, 3), dtype=np.uint8)
    colours[0] = 0
    return colours


def _save_visual(out_dir: Path, sidecar: Mapping[str, Any], semantic: Mapping[str, Any], pooled: Mapping[str, np.ndarray]) -> list[dict[str, str]]:
    from PIL import Image

    out = out_dir / "visual_audit"
    out.mkdir(parents=True, exist_ok=True)
    pca = _tensor_np(sidecar["radio_feat_pca"])
    comp = _tensor_int_np(sidecar["object_component_id"])
    boundary = _tensor_np(sidecar["object_boundary_score"])
    stability = _tensor_np(sidecar["temporal_stability"])
    label = pooled["label"].astype(np.int32)
    conf = pooled["confidence"].astype(np.float32)
    start = int(sidecar["global_start_frame"])
    frame_ids = list(range(start, start + int(pca.shape[0])))
    picks = sorted({0, int(pca.shape[0] // 2), int(pca.shape[0] - 1)})
    comp_pal = _palette(comp)
    label_pal = _palette(label)
    rows: list[dict[str, str]] = []
    for idx in picks:
        feat3 = pca[idx, :, :, :3]
        lo = np.percentile(feat3.reshape(-1, 3), 1, axis=0)
        hi = np.percentile(feat3.reshape(-1, 3), 99, axis=0)
        pca_rgb = np.clip((feat3 - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        stem = f"chunk_{int(sidecar['chunk_id']):03d}_frame_{frame_ids[idx]:06d}"
        paths = {
            "pca": out / f"{stem}_radio_pca_rgb.png",
            "component": out / f"{stem}_radio_component.png",
            "boundary": out / f"{stem}_radio_boundary.png",
            "stability": out / f"{stem}_radio_stability.png",
            "semantic_label": out / f"{stem}_semantic_label.png",
            "semantic_confidence": out / f"{stem}_semantic_confidence.png",
        }
        scale_size = (pca.shape[2] * 8, pca.shape[1] * 8)
        Image.fromarray((pca_rgb * 255).round().astype(np.uint8)).resize(scale_size).save(paths["pca"])
        Image.fromarray(comp_pal[comp[idx].clip(0, len(comp_pal) - 1)]).resize(scale_size).save(paths["component"])
        Image.fromarray(np.repeat((boundary[idx][:, :, None] * 255).astype(np.uint8), 3, axis=2)).resize(scale_size).save(paths["boundary"])
        Image.fromarray(np.repeat((stability[idx][:, :, None].clip(0, 1) * 255).astype(np.uint8), 3, axis=2)).resize(scale_size).save(paths["stability"])
        Image.fromarray(label_pal[label[idx].clip(0, len(label_pal) - 1)]).resize(scale_size).save(paths["semantic_label"])
        Image.fromarray(np.repeat((conf[idx][:, :, None].clip(0, 1) * 255).astype(np.uint8), 3, axis=2)).resize(scale_size).save(paths["semantic_confidence"])
        rows.append({k: str(v) for k, v in paths.items()})
    return rows


def _mean(vals: list[float | None]) -> float | None:
    good = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(sum(good) / len(good)) if good else None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_chunks = [int(x) for x in args.target_chunks.split(",") if x.strip()] if args.target_chunks else None
    sidecars = _load_sidecars(args.radio_sidecar_dir, target_chunks)
    metric_rows: list[dict[str, Any]] = []
    visual_rows_all: list[dict[str, Any]] = []
    for sidecar_path, sidecar in sidecars:
        chunk_id = int(sidecar["chunk_id"])
        chunk = find_stage_chunk(args.stage_c_cache, chunk_id)
        semantic = load_stage_semantic(chunk.masklet_path)
        pca = _tensor_np(sidecar["radio_feat_pca"])
        pca_norm = pca / np.maximum(np.linalg.norm(pca, axis=-1, keepdims=True), 1e-6)
        comp = _tensor_int_np(sidecar["object_component_id"])
        boundary = _tensor_np(sidecar["object_boundary_score"])
        stability = _tensor_np(sidecar["temporal_stability"])
        pooled = pool_semantic_to_grid(
            semantic["label_maps"][: pca.shape[0]],
            semantic.get("confidence_maps")[: pca.shape[0]] if semantic.get("confidence_maps") is not None else None,
            [str(x) for x in semantic.get("label_names", [])],
            (int(pca.shape[1]), int(pca.shape[2])),
        )
        radio_contrast_vals: list[float | None] = []
        label_contrast_vals: list[float | None] = []
        nmi_vals: list[float | None] = []
        ari_vals: list[float | None] = []
        for frame_idx in range(int(pca.shape[0])):
            radio_contrast_vals.append(finite_float(neighbor_contrast(pca_norm[frame_idx], comp[frame_idx]).get("boundary_contrast")))
            label_contrast_vals.append(finite_float(neighbor_contrast(pca_norm[frame_idx], pooled["label"][frame_idx]).get("boundary_contrast")))
            nmi, ari = _nmi_ari(comp[frame_idx], pooled["label"][frame_idx])
            nmi_vals.append(nmi)
            ari_vals.append(ari)
        dynamic_auc = _auc(pooled["dynamic"], _tensor_np(sidecar["radio_dynamic_score"]))
        static_auc = _auc(1.0 - np.maximum(pooled["dynamic"], pooled["sky"]), _tensor_np(sidecar["radio_static_score"]))
        loger = _loger_alignment(args.v68_feature_dir, chunk_id, boundary, stability)
        visual_rows = _save_visual(args.out_dir, sidecar, semantic, pooled)
        for row in visual_rows:
            row["chunk_id"] = chunk_id
            visual_rows_all.append(row)
        comp_counts = [component_count(comp[i]) for i in range(comp.shape[0])]
        radio_bc = _mean(radio_contrast_vals)
        label_bc = _mean(label_contrast_vals)
        max_corr = loger.get("max_abs_corr_boundary_to_global_k_gram") if loger.get("available") else None
        metric_rows.append(
            {
                "chunk_id": chunk_id,
                "sidecar_path": str(sidecar_path),
                "frames": int(pca.shape[0]),
                "patch_grid_h": int(pca.shape[1]),
                "patch_grid_w": int(pca.shape[2]),
                "object_component_count_min": min(comp_counts) if comp_counts else None,
                "object_component_count_max": max(comp_counts) if comp_counts else None,
                "object_component_count_mean": float(np.mean(comp_counts)) if comp_counts else None,
                "radio_object_boundary_contrast_mean": radio_bc,
                "semantic_label_boundary_contrast_mean": label_bc,
                "radio_minus_label_boundary_contrast": None if radio_bc is None or label_bc is None else float(radio_bc - label_bc),
                "alignment_semantic_nmi_mean": _mean(nmi_vals),
                "alignment_semantic_ari_mean": _mean(ari_vals),
                "dynamic_static_radio_dynamic_auc": dynamic_auc,
                "static_radio_static_auc": static_auc,
                "temporal_stability_finite": bool(np.isfinite(stability).all()),
                "temporal_stability_mean": float(np.mean(stability)),
                "loger_global_k_available": bool(loger.get("available")),
                "loger_global_k_reason": loger.get("reason", ""),
                "max_abs_corr_boundary_to_global_k_gram": max_corr,
                "r2_gate_boundary_contrast_pass": bool(radio_bc is not None and radio_bc >= 0.20),
                "r2_gate_component_count_pass": bool(comp_counts and min(comp_counts) >= 5 and max(comp_counts) <= 200),
                "r2_gate_temporal_finite_pass": bool(np.isfinite(stability).all()),
                "r2_gate_carrier_alignment_available": bool(max_corr is not None),
                "r2_gate_radio_beats_label_contrast": bool(radio_bc is not None and label_bc is not None and radio_bc > label_bc),
                "loger_alignment_detail": loger,
            }
        )
    write_csv(args.out_dir / "radio_alignment_metrics.csv", metric_rows)
    write_json(args.out_dir / "radio_alignment_metrics.json", metric_rows)
    write_json(args.out_dir / "visual_audit_manifest.json", visual_rows_all)

    any_better = any(bool(r.get("r2_gate_radio_beats_label_contrast")) for r in metric_rows)
    any_carrier = any(bool(r.get("r2_gate_carrier_alignment_available")) for r in metric_rows)
    pass_chunks = [
        r["chunk_id"]
        for r in metric_rows
        if r.get("r2_gate_boundary_contrast_pass")
        and r.get("r2_gate_component_count_pass")
        and r.get("r2_gate_temporal_finite_pass")
        and r.get("r2_gate_carrier_alignment_available")
    ]
    gate_pass = bool(pass_chunks and any_better)
    summary = {
        "created_at": utc_now(),
        "phase": "R2_radio_loger_alignment",
        "radio_sidecar_dir": str(args.radio_sidecar_dir),
        "stage_c_cache": str(args.stage_c_cache),
        "v68_feature_dir": str(args.v68_feature_dir),
        "overlap_pairs_dir": str(args.overlap_pairs_dir),
        "chunks_evaluated": [int(r["chunk_id"]) for r in metric_rows],
        "gate_pass": gate_pass,
        "pass_chunks": pass_chunks,
        "radio_beats_label_contrast_any_chunk": any_better,
        "carrier_alignment_available_any_chunk": any_carrier,
        "median_radio_boundary_contrast": median([r["radio_object_boundary_contrast_mean"] for r in metric_rows if r["radio_object_boundary_contrast_mean"] is not None]) if metric_rows else None,
        "median_radio_minus_label_boundary_contrast": median([r["radio_minus_label_boundary_contrast"] for r in metric_rows if r["radio_minus_label_boundary_contrast"] is not None]) if metric_rows else None,
        "blocker": "" if gate_pass else ("blocked_missing_loger_carrier_alignment" if not any_carrier else "blocked_r2_object_separability_or_label_control"),
        "notes": "R2 requires object boundary contrast, component count, temporal finite maps, and real LoGeR carrier alignment. Semantic NMI alone does not promote RADIO action.",
    }
    write_json(args.out_dir / "radio_alignment_gate_summary.json", summary)
    report = [
        "# v70 RADIO-LoGeR Alignment Report",
        "",
        f"- gate_pass: `{gate_pass}`",
        f"- blocker: `{summary['blocker'] or 'none'}`",
        f"- chunks_evaluated: `{summary['chunks_evaluated']}`",
        f"- pass_chunks: `{pass_chunks}`",
        f"- median_radio_boundary_contrast: `{summary['median_radio_boundary_contrast']}`",
        f"- median_radio_minus_label_boundary_contrast: `{summary['median_radio_minus_label_boundary_contrast']}`",
        "",
        "## Per Chunk",
        "",
    ]
    for row in metric_rows:
        report.append(
            f"- chunk {row['chunk_id']}: radio_bc={row['radio_object_boundary_contrast_mean']} "
            f"label_bc={row['semantic_label_boundary_contrast_mean']} "
            f"comp=[{row['object_component_count_min']},{row['object_component_count_max']}] "
            f"loger_available={row['loger_global_k_available']} corr={row['max_abs_corr_boundary_to_global_k_gram']}"
        )
    write_text(args.out_dir / "radio_alignment_report.md", "\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "gate_pass": gate_pass, "blocker": summary["blocker"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

