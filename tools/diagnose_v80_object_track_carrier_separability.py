#!/usr/bin/env python3
"""Diagnose whether v80 error/support maps have an object-track carrier.

This is an offline audit tool. It does not run a new method or claim a gate.
It aligns a low-resolution support/error map with Stage-C masklets and asks
whether high-risk patches are concentrated inside temporally tracked objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _clean(value: Any) -> Any:
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any, length: int) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return [None for _ in range(length)]
    return [value for _ in range(length)]


def _float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _long_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().long()
    return torch.as_tensor(value, dtype=torch.long)


def _load_track_index(path: Path | None) -> dict[str, dict[str, Any]]:
    raw = _read_json(path)
    index: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        rows: Iterable[Any] = raw.values()
    elif isinstance(raw, list):
        rows = raw
    else:
        return index
    for item in rows:
        if not isinstance(item, dict):
            continue
        for key in ("local_track_id", "global_id"):
            if item.get(key) is not None:
                index[str(item.get(key))] = item
    return index


def _labels_for_mask(label_patch: torch.Tensor, mask: torch.Tensor, label_names: list[str], limit: int = 8) -> list[dict[str, Any]]:
    weights = mask.detach().cpu().float().reshape(-1)
    labels = label_patch.detach().cpu().long().reshape(-1)
    if labels.numel() == 0 or float(weights.sum().item()) <= 0.0:
        return []
    out: list[dict[str, Any]] = []
    total = float(weights.sum().item())
    for label_id in torch.unique(labels).tolist():
        label_mask = labels == int(label_id)
        mass = float(weights[label_mask].sum().item())
        if mass <= 0.0:
            continue
        label = label_names[int(label_id)] if 0 <= int(label_id) < len(label_names) else f"id_{int(label_id)}"
        out.append({"label_id": int(label_id), "label": str(label), "mass": mass, "ratio": mass / total})
    return sorted(out, key=lambda row: float(row["mass"]), reverse=True)[:limit]


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> float | None:
    denom = float(weight.sum().item())
    if denom <= 1e-8:
        return None
    return float((value * weight).sum().item() / denom)


def _safe_ratio(num: float, denom: float) -> float | None:
    if denom <= 1e-8:
        return None
    return float(num / denom)


def _downsample_masks(mask: torch.Tensor, patch_grid: tuple[int, int]) -> torch.Tensor:
    n, t, h, w = mask.shape
    flat = mask.reshape(n * t, 1, h, w).float()
    pooled = F.adaptive_avg_pool2d(flat, output_size=patch_grid)
    return pooled.reshape(n, t, patch_grid[0], patch_grid[1]).clamp(0.0, 1.0)


def _downsample_labels(labels: torch.Tensor, patch_grid: tuple[int, int]) -> torch.Tensor:
    flat = labels[:, None].float()
    small = F.interpolate(flat, size=patch_grid, mode="nearest").squeeze(1)
    return small.round().long()


def _plot(path: Path, risk: torch.Tensor, fixed: torch.Tensor, top: torch.Tensor, union_all: torch.Tensor, union_tracked: torch.Tensor) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        return f"plot skipped: {exc}"

    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("risk mean", risk.mean(dim=0)),
        ("fixed high risk", fixed.mean(dim=0)),
        ("top-quantile risk", top.mean(dim=0)),
        ("all masklets", union_all.mean(dim=0)),
        ("tracked masklets", union_tracked.mean(dim=0)),
        ("tracked x top-risk", (union_tracked * top).mean(dim=0)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, (title, image) in zip(axes.reshape(-1), panels):
        im = ax.imshow(image.detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return None


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    support = torch.load(args.support_map, map_location="cpu", weights_only=False)
    masklet = torch.load(args.stage_c_masklet, map_location="cpu", weights_only=False)
    if not isinstance(support, dict):
        raise RuntimeError(f"Unsupported support payload: {args.support_map}")
    if not isinstance(masklet, dict) or "M_mask" not in masklet:
        raise RuntimeError(f"Unsupported Stage-C masklet payload: {args.stage_c_masklet}")

    score = _float_tensor(support["score_overlap"])
    if score.ndim == 3:
        score = score[0]
    if score.ndim != 2:
        raise RuntimeError(f"score_overlap must be [frames,tokens] or [1,frames,tokens], got {tuple(score.shape)}")
    overlap = min(int(args.overlap or support.get("overlap_frames_effective") or score.shape[0]), int(score.shape[0]))
    patch_grid = (int(args.patch_grid[0]), int(args.patch_grid[1]))
    if patch_grid[0] * patch_grid[1] != int(score.shape[1]):
        raise RuntimeError(f"patch grid {patch_grid} does not match tokens_per_frame={int(score.shape[1])}")
    score_map = score[:overlap].reshape(overlap, patch_grid[0], patch_grid[1]).clamp(0.0, 1.0)
    risk = (1.0 - score_map).clamp(0.0, 1.0)
    fixed_high = (risk >= float(args.high_risk_threshold)).float()
    top_threshold = float(torch.quantile(risk.reshape(-1), float(args.top_risk_quantile)).item())
    top_high = (risk >= top_threshold).float()

    masks = _float_tensor(masklet["M_mask"])
    masks = masks[:, :overlap]
    visible = masklet.get("V_mask")
    if visible is not None:
        vis = _float_tensor(visible)[:, :overlap].reshape(masks.shape[0], overlap, 1, 1)
        masks = masks * vis
    pooled = _downsample_masks(masks, patch_grid)
    binary = (pooled >= float(args.object_patch_threshold)).float()
    union_all = binary.max(dim=0).values

    n_obj = int(pooled.shape[0])
    source_types = [str(x) if x is not None else "" for x in _as_list(masklet.get("source_type"), n_obj)]
    labels = [str(x) if x is not None else "" for x in _as_list(masklet.get("L_sem"), n_obj)]
    seed_tracks = _as_list(masklet.get("seed_global_track_idx"), n_obj)
    weights = _as_list(masklet.get("W_sem"), n_obj)
    g_sem = _as_list(masklet.get("G_sem"), n_obj)
    track_index = _load_track_index(args.track_metadata)

    sem = masklet.get("semantic_segmentation")
    if isinstance(sem, dict) and "label_maps" in sem:
        label_maps = _long_tensor(sem["label_maps"])[:overlap]
        label_patch = _downsample_labels(label_maps, patch_grid)
        label_names = [str(x) for x in sem.get("label_names", [])]
        start_frame = int(sem.get("global_start_frame", masklet.get("manifest", {}).get("start_frame", -1)))
    else:
        label_patch = torch.zeros_like(score_map, dtype=torch.long)
        label_names = []
        start_frame = int(masklet.get("manifest", {}).get("start_frame", -1))

    fixed_total = float(fixed_high.sum().item())
    top_total = float(top_high.sum().item())
    tracked_binary_list: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    global_risk_mean = float(risk.mean().item())

    for idx in range(n_obj):
        soft = pooled[idx]
        hard = binary[idx]
        is_tracked = source_types[idx] == "thing_tracked" or labels[idx] in {"car", "truck", "person", "bicycle", "motorcycle"}
        if is_tracked:
            tracked_binary_list.append(hard)
        track_meta = track_index.get(str(seed_tracks[idx]), {}) if is_tracked else {}
        area_mass = float(soft.sum().item())
        patch_count = int(hard.sum().item())
        risk_inside = _weighted_mean(risk, soft)
        top_inside = _weighted_mean(top_high, soft)
        fixed_inside = _weighted_mean(fixed_high, soft)
        top_cov = _safe_ratio(float((top_high * hard).sum().item()), top_total)
        fixed_cov = _safe_ratio(float((fixed_high * hard).sum().item()), fixed_total)
        row = {
            "object_index": idx,
            "label": labels[idx],
            "source_type": source_types[idx],
            "is_tracked_candidate": bool(is_tracked),
            "seed_global_track_idx": seed_tracks[idx],
            "stage_c_sem_group": g_sem[idx],
            "stage_c_weight": weights[idx],
            "track_canonical_label": track_meta.get("canonical_label"),
            "track_state": track_meta.get("state"),
            "track_visible_frames_total": track_meta.get("visible_frames"),
            "visible_overlap_frames": int((hard.reshape(overlap, -1).sum(dim=1) > 0).sum().item()),
            "object_patch_mass": area_mass,
            "object_binary_patch_count": patch_count,
            "risk_inside_object_mean": risk_inside,
            "risk_lift_vs_global": None if risk_inside is None else float(risk_inside - global_risk_mean),
            "support_inside_object_mean": None if risk_inside is None else float(1.0 - risk_inside),
            "fixed_high_risk_inside_object_fraction": fixed_inside,
            "top_high_risk_inside_object_fraction": top_inside,
            "fixed_high_risk_coverage_by_object": fixed_cov,
            "top_high_risk_coverage_by_object": top_cov,
            "top_labels_inside_object": _labels_for_mask(label_patch, soft, label_names, limit=6),
        }
        rows.append(row)

    union_tracked = torch.stack(tracked_binary_list, dim=0).max(dim=0).values if tracked_binary_list else torch.zeros_like(union_all)
    background = (1.0 - union_all).clamp(0.0, 1.0)
    tracked_background = (1.0 - union_tracked).clamp(0.0, 1.0)
    union_top_cov = _safe_ratio(float((top_high * union_all).sum().item()), top_total)
    tracked_top_cov = _safe_ratio(float((top_high * union_tracked).sum().item()), top_total)
    union_fixed_cov = _safe_ratio(float((fixed_high * union_all).sum().item()), fixed_total)
    tracked_fixed_cov = _safe_ratio(float((fixed_high * union_tracked).sum().item()), fixed_total)

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float(row["top_high_risk_coverage_by_object"] or 0.0),
            float(row["risk_inside_object_mean"] or 0.0),
            float(row["object_patch_mass"] or 0.0),
        ),
        reverse=True,
    )
    tracked_rows_sorted = [row for row in rows_sorted if row["is_tracked_candidate"]]
    min_mass = float(args.min_object_patch_mass)
    actionable = [
        row
        for row in rows_sorted
        if row["is_tracked_candidate"]
        and float(row["object_patch_mass"] or 0.0) >= min_mass
        and float(row["top_high_risk_coverage_by_object"] or 0.0) >= float(args.min_top_risk_object_coverage)
        and float(row["risk_lift_vs_global"] or 0.0) > 0.0
    ]
    radio_sidecar_available = bool(args.radio_sidecar and Path(args.radio_sidecar).is_file())
    if actionable:
        recommendation = "object_track_smoke_candidate_available"
    elif radio_sidecar_available:
        recommendation = "object_track_weak_try_radio_object_interior_if_plan_requires"
    else:
        recommendation = "do_not_run_object_track_runtime_smoke_on_this_chunk"

    plot_error = _plot(args.out_png, risk, fixed_high, top_high, union_all, union_tracked) if args.out_png else None
    summary = {
        "schema": "acl2_v80_object_track_carrier_separability_v1",
        "support_map": str(args.support_map),
        "stage_c_masklet": str(args.stage_c_masklet),
        "track_metadata": str(args.track_metadata) if args.track_metadata else "",
        "radio_sidecar": str(args.radio_sidecar) if args.radio_sidecar else "",
        "radio_sidecar_available": radio_sidecar_available,
        "seq": str(args.seq),
        "chunk": int(args.chunk),
        "start_frame": start_frame,
        "overlap_frames": int(overlap),
        "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "risk_definition": "1 - support.score_overlap",
        "risk_mean": global_risk_mean,
        "risk_q50": float(torch.quantile(risk.reshape(-1), 0.50).item()),
        "risk_q90": float(torch.quantile(risk.reshape(-1), 0.90).item()),
        "risk_q95": float(torch.quantile(risk.reshape(-1), 0.95).item()),
        "fixed_high_risk_threshold": float(args.high_risk_threshold),
        "fixed_high_risk_patch_ratio": float(fixed_high.mean().item()),
        "top_risk_quantile": float(args.top_risk_quantile),
        "top_risk_threshold": top_threshold,
        "top_high_risk_patch_ratio": float(top_high.mean().item()),
        "num_masklets": n_obj,
        "num_tracked_candidates": int(sum(1 for row in rows if row["is_tracked_candidate"])),
        "union_masklet_patch_ratio": float(union_all.mean().item()),
        "tracked_masklet_patch_ratio": float(union_tracked.mean().item()),
        "union_fixed_high_risk_coverage": union_fixed_cov,
        "tracked_fixed_high_risk_coverage": tracked_fixed_cov,
        "union_top_high_risk_coverage": union_top_cov,
        "tracked_top_high_risk_coverage": tracked_top_cov,
        "background_risk_mean_outside_all_masklets": _weighted_mean(risk, background),
        "risk_mean_outside_tracked_masklets": _weighted_mean(risk, tracked_background),
        "top_risk_labels": _labels_for_mask(label_patch, top_high, label_names, limit=10),
        "fixed_high_risk_labels": _labels_for_mask(label_patch, fixed_high, label_names, limit=10),
        "top_objects_by_risk_coverage": rows_sorted[:8],
        "tracked_objects_by_risk_coverage": tracked_rows_sorted[:8],
        "actionable_object_count": len(actionable),
        "actionable_objects": actionable[:5],
        "object_track_carrier_available": bool(actionable),
        "recommendation": recommendation,
        "interpretation_guardrail": (
            "This diagnostic only tests spatial separability/coverage. "
            "It is not a v80 gate pass and must not be reported as runtime improvement."
        ),
        "out_csv": str(args.out_csv),
        "out_png": str(args.out_png) if args.out_png else "",
        "plot_error": plot_error,
    }
    _write_json(args.out_json, summary)
    _write_csv(args.out_csv, rows_sorted)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-map", type=Path, required=True)
    parser.add_argument("--stage-c-masklet", type=Path, required=True)
    parser.add_argument("--track-metadata", type=Path)
    parser.add_argument("--radio-sidecar", type=Path)
    parser.add_argument("--seq", default="00")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--patch-grid", type=int, nargs=2, default=(22, 57))
    parser.add_argument("--object-patch-threshold", type=float, default=0.10)
    parser.add_argument("--high-risk-threshold", type=float, default=0.50)
    parser.add_argument("--top-risk-quantile", type=float, default=0.90)
    parser.add_argument("--min-object-patch-mass", type=float, default=2.0)
    parser.add_argument("--min-top-risk-object-coverage", type=float, default=0.05)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path)
    return parser.parse_args()


def main() -> None:
    summary = diagnose(parse_args())
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
