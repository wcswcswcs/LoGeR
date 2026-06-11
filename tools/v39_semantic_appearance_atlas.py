#!/usr/bin/env python3
"""Build v39 semantic-appearance atlas from landed cache and rollout artifacts.

This tool intentionally separates evidence levels:

* RGB/Lab and masklet appearance statistics are computed from the KITTI images
  and Stage-C VideoMasklet cache.
* Path influence/action summaries are read from the landed Track 0 atlas.
* Per-label D_g / attention / SWA / TTT tensor maps are not reconstructed when
  the rollout did not land those tensors; the output files mark that boundary
  explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float array in [0, 1] to CIE Lab."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mask = rgb > 0.04045
    lin = np.empty_like(rgb)
    lin[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    lin[~mask] = rgb[~mask] / 12.92
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ mat.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / white
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def _load_image(image_dir: Path, frame: int) -> Tuple[np.ndarray, np.ndarray]:
    path = image_dir / f"{int(frame):06d}.png"
    img = Image.open(path).convert("RGB")
    rgb_u8 = np.asarray(img, dtype=np.uint8)
    rgb = rgb_u8.astype(np.float64) / 255.0
    return rgb_u8, _rgb_to_lab(rgb)


def _cache_masklet_path(cache_dir: Path, chunk: int) -> Path | None:
    matches = sorted(cache_dir.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    return matches[0] if matches else None


def _mean_safe(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _p90_safe(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.quantile(np.asarray(vals, dtype=np.float64), 0.90))


def _mad_z(values_by_key: Dict[str, List[float]], key: str, value: float) -> float:
    vals = np.asarray([v for v in values_by_key.get(key, []) if math.isfinite(float(v))], dtype=np.float64)
    if vals.size < 3 or not math.isfinite(value):
        return float("nan")
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    return float((value - median) / (mad + 1e-6))


def _write_strip(path: Path, images: Sequence[np.ndarray], labels: Sequence[str]) -> bool:
    if not images:
        return False
    thumbs: List[Image.Image] = []
    for image, label in zip(images, labels):
        pil = Image.fromarray(image)
        pil.thumbnail((220, 90))
        canvas = Image.new("RGB", (220, 110), "white")
        canvas.paste(pil, (0, 0))
        thumbs.append(canvas)
    out = Image.new("RGB", (220 * len(thumbs), 110), "white")
    for idx, thumb in enumerate(thumbs):
        out.paste(thumb, (220 * idx, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return True


def _overlay_image(rgb: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
        ],
        dtype=np.uint8,
    )
    colors = palette[np.asarray(label_map, dtype=np.int64) % len(palette)]
    alpha = (label_map > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + colors.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _write_heat(path: Path, values: np.ndarray, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(values, cmap="magma")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _candidate_target(candidate: str) -> str:
    text = candidate.lower()
    if "sky" in text:
        return "sky"
    if "veg" in text or "vegetation" in text:
        return "vegetation"
    if "dynamic" in text:
        return "dynamic"
    if "shadow" in text or "lowtrust" in text:
        return "lowtrust_or_shadow_proxy"
    if "structure" in text or "static" in text:
        return "structure"
    return "mixed_or_unspecified"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--stage-c-cache-dir", required=True, type=Path)
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: List[Dict[str, Any]] = []
    masklet_rows_raw: List[Dict[str, Any]] = []
    iou_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    label_delta_values: Dict[str, List[float]] = {}
    chunk_visuals: List[Dict[str, Any]] = []

    for chunk in chunks:
        masklet_path = _cache_masklet_path(args.stage_c_cache_dir, chunk)
        if masklet_path is None:
            chunk_visuals.append({"chunk": chunk, "status": "missing_stage_c_masklet_cache"})
            continue
        data = torch.load(masklet_path, map_location="cpu")
        manifest = data.get("manifest", {})
        start = int(manifest.get("start_frame"))
        num_frames = int(data.get("num_frames", 0))
        labels = list(data.get("L_sem", []))
        groups = data.get("G_sem")
        weights = data.get("W_sem")
        q_mask = data.get("Q_mask")
        area_ratio = data.get("A_ratio")
        masks = data.get("M_mask").bool()
        visible = data.get("V_mask").bool()

        images_u8: List[np.ndarray] = []
        images_lab: List[np.ndarray] = []
        for t in range(num_frames):
            frame = start + t
            rgb_u8, lab = _load_image(args.image_dir, frame)
            images_u8.append(rgb_u8)
            images_lab.append(lab)
            luma = 0.2126 * rgb_u8[..., 0] + 0.7152 * rgb_u8[..., 1] + 0.0722 * rgb_u8[..., 2]
            frame_rows.append({
                "chunk": chunk,
                "frame": frame,
                "rgb_mean_r": float(rgb_u8[..., 0].mean()),
                "rgb_mean_g": float(rgb_u8[..., 1].mean()),
                "rgb_mean_b": float(rgb_u8[..., 2].mean()),
                "luma_mean": float(luma.mean()),
                "luma_p90": float(np.quantile(luma, 0.90)),
                "lab_l_mean": float(lab[..., 0].mean()),
                "lab_a_mean": float(lab[..., 1].mean()),
                "lab_b_mean": float(lab[..., 2].mean()),
            })

        selected_t = sorted(set([0, max(0, num_frames // 4), max(0, num_frames // 2), max(0, 3 * num_frames // 4), max(0, num_frames - 1)]))
        _write_strip(
            out_dir / f"rgb_frame_strip_chunk{chunk:03d}.png",
            [images_u8[t] for t in selected_t],
            [str(start + t) for t in selected_t],
        )

        label_maps: List[np.ndarray] = []
        anomaly_maps: List[np.ndarray] = []
        for t in selected_t:
            label_map = np.zeros(images_u8[t].shape[:2], dtype=np.int32)
            anomaly_map = np.zeros(images_u8[t].shape[:2], dtype=np.float32)
            for m, label in enumerate(labels):
                if not bool(visible[m, t]):
                    continue
                mask = masks[m, t].numpy()
                label_map[mask] = int(m) + 1
            label_maps.append(label_map)
            anomaly_maps.append(anomaly_map)
        overlay_ok = False
        if label_maps:
            overlays = [_overlay_image(images_u8[t], lm) for t, lm in zip(selected_t, label_maps)]
            overlay_ok = _write_strip(
                out_dir / f"semantic_mask_overlay_chunk{chunk:03d}.png",
                overlays,
                [str(start + t) for t in selected_t],
            )

        for m, label in enumerate(labels):
            ema: np.ndarray | None = None
            deltas: List[float] = []
            lab_means: List[np.ndarray] = []
            mask_ious: List[float] = []
            prev_mask: np.ndarray | None = None
            area_vals: List[float] = []
            q_vals: List[float] = []
            for t in range(num_frames):
                if not bool(visible[m, t]):
                    continue
                mask = masks[m, t].numpy()
                count = int(mask.sum())
                if count <= 0:
                    continue
                lab_mean = images_lab[t][mask].mean(axis=0)
                lab_means.append(lab_mean)
                if ema is None:
                    delta = 0.0
                    ema = lab_mean
                else:
                    delta = float(np.linalg.norm(lab_mean - ema))
                    ema = 0.8 * ema + 0.2 * lab_mean
                deltas.append(delta)
                label_delta_values.setdefault(str(label), []).append(delta)
                if prev_mask is not None:
                    union = int(np.logical_or(prev_mask, mask).sum())
                    inter = int(np.logical_and(prev_mask, mask).sum())
                    if union > 0:
                        mask_ious.append(float(inter / union))
                prev_mask = mask
                if area_ratio is not None:
                    area_vals.append(float(area_ratio[m, t]))
                if q_mask is not None:
                    q_vals.append(float(q_mask[m, t]))
            row = {
                "chunk": chunk,
                "masklet_index": m,
                "label": str(label),
                "group": int(groups[m]) if groups is not None else None,
                "semantic_weight": float(weights[m]) if weights is not None else None,
                "visible_frames": int(visible[m].sum().item()),
                "lab_delta_mean": _mean_safe(deltas),
                "lab_delta_p90": _p90_safe(deltas),
                "lab_delta_max": max(deltas) if deltas else float("nan"),
                "area_ratio_mean": _mean_safe(area_vals),
                "mask_quality_mean": _mean_safe(q_vals),
                "evidence_status": "computed_from_stage_c_masklet_and_rgb",
            }
            masklet_rows_raw.append(row)
            iou_rows.append({
                "chunk": chunk,
                "masklet_index": m,
                "label": str(label),
                "temporal_iou_mean": _mean_safe(mask_ious),
                "temporal_iou_p10": float(np.quantile(np.asarray(mask_ious), 0.10)) if mask_ious else float("nan"),
                "evidence_status": "computed_from_stage_c_masklet_masks",
            })
            stability_rows.append({
                "chunk": chunk,
                "masklet_index": m,
                "label": str(label),
                "label_stability": 1.0,
                "evidence_status": "stage_c_cache_has_static_label_per_masklet",
            })

        mid = num_frames // 2
        heat = np.zeros(images_u8[mid].shape[:2], dtype=np.float32)
        for row in masklet_rows_raw:
            if int(row["chunk"]) != chunk:
                continue
            m = int(row["masklet_index"])
            if bool(visible[m, mid]):
                heat[masks[m, mid].numpy()] = max(heat[masks[m, mid].numpy()].max(initial=0.0), _float(row["lab_delta_p90"], 0.0))
        heat_ok = _write_heat(
            out_dir / f"appearance_anomaly_heatmap_chunk{chunk:03d}.png",
            heat,
            f"v39 chunk {chunk} masklet Lab-delta p90 map",
        )
        chunk_visuals.append({
            "chunk": chunk,
            "masklet_path": str(masklet_path),
            "rgb_frame_strip": True,
            "semantic_mask_overlay": overlay_ok,
            "appearance_anomaly_heatmap": heat_ok,
            "spatial_dg_attention_swa_ttt_maps": "not_landed_by_rollout_artifacts",
        })

    masklet_rows: List[Dict[str, Any]] = []
    for row in masklet_rows_raw:
        out = dict(row)
        out["lab_delta_sem_mad_z"] = _mad_z(label_delta_values, str(row["label"]), _float(row["lab_delta_p90"]))
        masklet_rows.append(out)

    label_rows: List[Dict[str, Any]] = []
    for label in sorted(label_delta_values):
        vals = label_delta_values[label]
        subset = [r for r in masklet_rows if str(r["label"]) == label]
        label_rows.append({
            "label": label,
            "masklet_rows": len(subset),
            "lab_delta_mean": _mean_safe(vals),
            "lab_delta_p90": _p90_safe(vals),
            "lab_delta_max": max(vals) if vals else float("nan"),
            "mean_mask_quality": _mean_safe([_float(r.get("mask_quality_mean")) for r in subset]),
            "mean_area_ratio": _mean_safe([_float(r.get("area_ratio_mean")) for r in subset]),
            "evidence_status": "computed_from_stage_c_masklet_and_rgb",
        })

    per_path = _read_csv(args.atlas_dir / "semantic_path_action_influence.csv")
    influence_rows: List[Dict[str, Any]] = []
    for row in per_path:
        candidate = str(row.get("candidate", ""))
        target = _candidate_target(candidate)
        influence_rows.append({
            "candidate": candidate,
            "semantic_target_from_policy_name": target,
            "parent": row.get("parent"),
            "chunk": row.get("chunk"),
            "path": row.get("path"),
            "attention_mass_available": row.get("attention_mass_available"),
            "attention_mass_removed_before": row.get("attention_mass_removed_before"),
            "attention_mass_removed_after": row.get("attention_mass_removed_after"),
            "influence_mass": row.get("influence_mass"),
            "evidence_status": "candidate_level_path_summary_not_per_label",
        })

    missing_semantic_tensor = [{
        "status": "explainability_missing",
        "reason": "rollout artifacts do not land per-label token tensors for this quantity; v39 records the absence instead of reconstructing it",
    }]
    _write_csv(out_dir / "per_frame_rgb_luma_stats.csv", frame_rows)
    _write_csv(out_dir / "per_semantic_label_lab_delta.csv", label_rows)
    _write_csv(out_dir / "per_masklet_lab_delta.csv", masklet_rows)
    _write_csv(out_dir / "per_masklet_temporal_iou.csv", iou_rows)
    _write_csv(out_dir / "per_masklet_label_stability.csv", stability_rows)
    _write_csv(out_dir / "per_semantic_Dg_stats.csv", missing_semantic_tensor)
    _write_csv(out_dir / "per_semantic_ttt_conflict_stats.csv", missing_semantic_tensor)
    _write_csv(out_dir / "per_semantic_scale_risk_stats.csv", missing_semantic_tensor)
    _write_csv(out_dir / "per_semantic_source_attention_mass.csv", influence_rows)
    _write_csv(out_dir / "per_semantic_swa_cache_mass.csv", influence_rows)
    _write_csv(out_dir / "per_semantic_ttt_update_contribution.csv", influence_rows)

    atlas_rows: List[Dict[str, Any]] = []
    for label in label_rows:
        label_name = str(label["label"])
        target_influence = [
            _float(r.get("influence_mass"))
            for r in influence_rows
            if str(r.get("semantic_target_from_policy_name")) == label_name
        ]
        atlas_rows.append({
            **label,
            "candidate_level_influence_mass_max": max([v for v in target_influence if math.isfinite(v)], default=float("nan")),
            "source_attention_evidence_level": "candidate_level_path_summary_not_per_label",
        })
    _write_csv(out_dir / "semantic_appearance_influence_atlas.csv", atlas_rows)

    sky = [r for r in atlas_rows if str(r.get("label")) == "sky"]
    sky_summary = sky[0] if sky else {}
    summary = {
        "chunks": chunks,
        "frame_rows": len(frame_rows),
        "masklet_rows": len(masklet_rows),
        "semantic_label_rows": len(label_rows),
        "visual_rows": chunk_visuals,
        "sky_lab_delta_p90": sky_summary.get("lab_delta_p90"),
        "sky_candidate_level_influence_mass_max": sky_summary.get("candidate_level_influence_mass_max"),
        "sky_causality_decision": (
            "not_proven_per_label_influence_missing"
            if sky_summary
            else "sky_label_not_present_in_checked_stage_c_chunks"
        ),
        "missing_spatial_maps": [
            "D_g_heatmap",
            "source_attention_mass_heatmap",
            "SWA_overlap_nonoverlap_source_mass_map",
            "TTT_update_contribution_map",
            "full_combined_risk_map_with_Dg_attention_swa_ttt",
        ],
        "boundary": (
            "Appearance and masklet stats are computed from landed RGB/cache. "
            "Per-label D_g/attention/SWA/TTT tensor maps were not landed and are not fabricated."
        ),
    }
    _write_json(out_dir / "v39_semantic_appearance_summary.json", summary)
    _write_json(out_dir / "v39_spatial_visualization_boundary.json", {
        "generated": chunk_visuals,
        "missing": summary["missing_spatial_maps"],
        "reason": summary["boundary"],
    })
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
