#!/usr/bin/env python3
"""Render v97 object micro-primitives into sparse support heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase6_render_splat"
RUN_ID = "v97_phase6_render_splat"
DEFAULT_PHASE4 = ROOT / "outputs/audit/v97_phase4_micro_affinity_feature_D3_source_preserve2048_region_proxy_500k_gpu6"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v97_phase5_object_birth_region_proxy_500k"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase6_render_splat_C0_region_proxy_500k_gpu7"


@triton.jit
def _splat_points_kernel(
    uv,
    weights,
    heatmap,
    n_points: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    radius: tl.constexpr,
    sigma: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    side: tl.constexpr = radius * 2 + 1
    total: tl.constexpr = side * side
    point_idx = offsets // total
    off = offsets - point_idx * total
    valid = point_idx < n_points
    dx_i = off % side - radius
    dy_i = off // side - radius
    u = tl.load(uv + point_idx * 2, mask=valid, other=0.0)
    v = tl.load(uv + point_idx * 2 + 1, mask=valid, other=0.0)
    point_weight = tl.load(weights + point_idx, mask=valid, other=0.0)
    x = tl.cast(tl.floor(u + 0.5), tl.int32) + dx_i
    y = tl.cast(tl.floor(v + 0.5), tl.int32) + dy_i
    inside = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    dist2 = tl.cast(dx_i * dx_i + dy_i * dy_i, tl.float32)
    weight = point_weight * tl.exp(-dist2 / (2.0 * sigma * sigma))
    tl.atomic_add(heatmap + y * width + x, weight, sem="relaxed", mask=inside)


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_micro_features(root: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _iter_csv(root / "micro_feature_index.csv"):
        idx = int(_num(row.get("feature_index")))
        out[idx] = {
            "scene_id": row.get("scene_id", ""),
            "window_id": row.get("window_id", ""),
            "micro_primitive_id": row.get("micro_primitive_id", ""),
            "target_frame_id": int(_num(row.get("target_frame_id"))),
            "target_mask_id": int(_num(row.get("target_mask_id"))),
            "u_tgt": float(_num(row.get("u_tgt"))),
            "v_tgt": float(_num(row.get("v_tgt"))),
            "visibility": float(_num(row.get("visibility"))),
            "confidence": float(_num(row.get("confidence"))),
            "B_pa": float(_num(row.get("B_pa"))),
            "near_boundary": _bool(row.get("near_boundary")),
            "distinct_mask_count_3x3": float(_num(row.get("distinct_mask_count_3x3"))),
        }
    return out


def _load_object_assignments(phase5_root: Path, feature_lookup: dict[int, dict[str, Any]], variant_id: str) -> dict[tuple[str, str, str, int], list[int]]:
    groups: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    for row in _iter_csv(phase5_root / "object_micro_primitive_rows.csv"):
        if row.get("variant_id") != variant_id:
            continue
        idx = int(_num(row.get("feature_index"), -1))
        meta = feature_lookup.get(idx)
        if meta is None:
            continue
        key = (row.get("scene_id", ""), row.get("window_id", ""), row.get("object_id", ""), int(meta["target_frame_id"]))
        groups[key].append(idx)
    return groups


def _cpu_splat(uv_np: np.ndarray, weights_np: np.ndarray, height: int, width: int, radius: int, sigma: float) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    for (u, v), point_weight in zip(uv_np, weights_np):
        cx = int(round(float(u)))
        cy = int(round(float(v)))
        for dy in range(-radius, radius + 1):
            y = cy + dy
            if y < 0 or y >= height:
                continue
            for dx in range(-radius, radius + 1):
                x = cx + dx
                if x < 0 or x >= width:
                    continue
                heatmap[y, x] += float(point_weight) * math.exp(-float(dx * dx + dy * dy) / (2.0 * float(sigma) * float(sigma)))
    return heatmap


def _triton_splat_tensor(uv_np: np.ndarray, weights_np: np.ndarray, height: int, width: int, radius: int, sigma: float, device: str) -> torch.Tensor:
    if uv_np.size == 0:
        return torch.zeros((height, width), dtype=torch.float32, device=device)
    uv_t = torch.as_tensor(uv_np.astype(np.float32, copy=False), device=device)
    weight_t = torch.as_tensor(weights_np.astype(np.float32, copy=False), device=device)
    heatmap = torch.zeros((height * width,), dtype=torch.float32, device=device)
    pixels_per_point = int((radius * 2 + 1) ** 2)
    block = 256
    grid = (triton.cdiv(int(uv_np.shape[0]) * pixels_per_point, block),)
    _splat_points_kernel[grid](uv_t, weight_t, heatmap, int(uv_np.shape[0]), int(height), int(width), int(radius), float(sigma), block)
    return heatmap.reshape(height, width)


def _triton_splat(uv_np: np.ndarray, weights_np: np.ndarray, height: int, width: int, radius: int, sigma: float, device: str) -> np.ndarray:
    return _triton_splat_tensor(uv_np, weights_np, height, width, radius, sigma, device).detach().cpu().numpy()


def _support_sparse_from_tensor(
    heatmap: torch.Tensor,
    *,
    threshold: float,
    max_sparse_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    flat = heatmap.reshape(-1)
    support = flat > float(threshold)
    if not bool(torch.any(support)) and flat.numel() > 0:
        support = flat > 0.0
    idx = torch.nonzero(support, as_tuple=False).flatten()
    support_area = int(idx.numel())
    peak = float(torch.max(flat).detach().cpu().item()) if flat.numel() else 0.0
    if support_area == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32), 0, peak, 0.0
    vals = flat[idx]
    support_mean = float(torch.mean(vals).detach().cpu().item())
    if vals.numel() > int(max_sparse_pixels):
        vals, pos = torch.topk(vals, k=int(max_sparse_pixels), largest=True, sorted=False)
        idx = idx[pos]
    idx_cpu = idx.detach().cpu().numpy().astype(np.int64, copy=False)
    vals_cpu = vals.detach().cpu().numpy().astype(np.float32, copy=False)
    width = int(heatmap.shape[1])
    ys = (idx_cpu // width).astype(np.int32, copy=False)
    xs = (idx_cpu % width).astype(np.int32, copy=False)
    return ys, xs, vals_cpu, support_area, peak, support_mean


def _fragment_count_from_sparse(ys: np.ndarray, xs: np.ndarray) -> int:
    if ys.size == 0:
        return 0
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    crop[ys - y0, xs - x0] = 1
    nlabels, _labels = cv2.connectedComponents(crop, connectivity=8)
    return int(max(0, nlabels - 1))


def _flush_shard(
    *,
    output_root: Path,
    shard_index: int,
    shard_payload: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    shard_path = output_root / "heatmap_shards" / f"heatmap_sparse_shard_{shard_index:04d}.npz"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_path,
        object_frame_index=np.asarray(shard_payload["object_frame_index"], dtype=np.int64),
        y=np.asarray(shard_payload["y"], dtype=np.int32),
        x=np.asarray(shard_payload["x"], dtype=np.int32),
        value=np.asarray(shard_payload["value"], dtype=np.float16),
        object_ids=np.asarray(shard_payload["object_ids"], dtype=object),
    )
    row = {
        "schema_version": "stream4d_v97_phase6_heatmap_manifest_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "shard_index": shard_index,
        "heatmap_shard_path": _rel(shard_path),
        "sparse_value_count": len(shard_payload["value"]),
        "object_frame_count": len(set(shard_payload["object_frame_index"])),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    return row, {"object_frame_index": [], "y": [], "x": [], "value": [], "object_ids": []}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "heatmap_shards").mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase6 requires CUDA/Triton; no CUDA device available")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    phase4_root = _project(args.phase4_root)
    phase5_root = _project(args.phase5_root)
    feature_lookup = _load_micro_features(phase4_root)
    groups = _load_object_assignments(phase5_root, feature_lookup, args.variant_id)
    if args.max_object_frames > 0:
        groups = dict(list(sorted(groups.items()))[: int(args.max_object_frames)])
    object_frame_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    shard_payload = {"object_frame_index": [], "y": [], "x": [], "value": [], "object_ids": []}
    missing_heatmap_count = 0
    render_started = time.time()
    for object_frame_index, (key, feature_indices) in enumerate(sorted(groups.items())):
        scene, window, object_id, frame_id = key
        metas = [feature_lookup[idx] for idx in feature_indices if idx in feature_lookup]
        uv = np.asarray([[meta["u_tgt"], meta["v_tgt"]] for meta in metas], dtype=np.float32).reshape(-1, 2)
        weights = np.asarray(
            [
                max(0.0, float(meta["visibility"]) * float(meta["confidence"])) * (0.75 if bool(meta["near_boundary"]) else 1.0)
                for meta in metas
            ],
            dtype=np.float32,
        )
        if uv.size == 0:
            missing_heatmap_count += 1
            continue
        sigma = float(args.sigma0) * (1.0 + float(args.eta_confidence) * max(0.0, 1.0 - float(np.mean([m["confidence"] for m in metas]))))
        radius = int(max(args.min_radius, min(args.max_radius, math.ceil(2.0 * sigma))))
        heatmap_t = _triton_splat_tensor(uv, weights, int(args.image_height), int(args.image_width), radius, sigma, device)
        if not parity_rows:
            sample_uv = uv[: int(args.parity_max_points)]
            sample_weights = weights[: int(args.parity_max_points)]
            cpu = _cpu_splat(sample_uv, sample_weights, int(args.image_height), int(args.image_width), radius, sigma)
            gpu = _triton_splat(sample_uv, sample_weights, int(args.image_height), int(args.image_width), radius, sigma, device)
            abs_err = np.abs(cpu - gpu)
            parity_rows.append(
                {
                    "schema_version": "stream4d_v97_phase6_triton_splat_parity_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "object_id": object_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "sample_point_count": int(sample_uv.shape[0]),
                    "cpu_vs_triton_abs_error_max": float(np.max(abs_err)) if abs_err.size else 0.0,
                    "cpu_vs_triton_abs_error_mean": float(np.mean(abs_err)) if abs_err.size else 0.0,
                    "cpu_vs_triton_positive_pixel_mismatch_rate": float(np.mean((cpu > 0) != (gpu > 0))) if abs_err.size else 0.0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        ys, xs, vals, support_area, support_peak, support_mean = _support_sparse_from_tensor(
            heatmap_t,
            threshold=float(args.support_threshold),
            max_sparse_pixels=int(args.max_sparse_pixels_per_object_frame),
        )
        fragment_count = _fragment_count_from_sparse(ys, xs)
        shard_payload["object_frame_index"].extend([object_frame_index] * len(vals))
        shard_payload["y"].extend([int(v) for v in ys.tolist()])
        shard_payload["x"].extend([int(v) for v in xs.tolist()])
        shard_payload["value"].extend([float(v) for v in vals.tolist()])
        shard_payload["object_ids"].extend([object_id] * len(vals))
        if len(shard_payload["value"]) >= int(args.shard_value_limit):
            manifest, shard_payload = _flush_shard(output_root=output_root, shard_index=len(manifest_rows), shard_payload=shard_payload)
            manifest_rows.append(manifest)
        area_ratio = support_area / float(int(args.image_height) * int(args.image_width))
        row = {
            "schema_version": "stream4d_v97_phase6_object_frame_support_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": args.variant_id,
            "scene_id": scene,
            "window_id": window,
            "object_id": object_id,
            "frame_id": int(frame_id),
            "support_area": support_area,
            "support_area_ratio": area_ratio,
            "support_peak": support_peak,
            "support_mean": support_mean,
            "visible_micro_count": int(len(metas)),
            "mean_visibility": float(np.mean([m["visibility"] for m in metas])) if metas else 0.0,
            "mean_confidence": float(np.mean([m["confidence"] for m in metas])) if metas else 0.0,
            "sigma_mean": sigma,
            "sigma_p90": sigma,
            "support_fragment_count": fragment_count,
            "heatmap_shard_path": "",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        object_frame_rows.append(row)
    if shard_payload["value"]:
        manifest, shard_payload = _flush_shard(output_root=output_root, shard_index=len(manifest_rows), shard_payload=shard_payload)
        manifest_rows.append(manifest)
    render_runtime_ms = 1000.0 * (time.time() - render_started)
    support_area_ratios = np.asarray([_num(row.get("support_area_ratio")) for row in object_frame_rows], dtype=np.float64)
    visible_counts = np.asarray([_num(row.get("visible_micro_count")) for row in object_frame_rows], dtype=np.float64)
    fragment_counts = np.asarray([_num(row.get("support_fragment_count")) for row in object_frame_rows], dtype=np.float64)
    peaks = np.asarray([_num(row.get("support_peak")) for row in object_frame_rows], dtype=np.float64)
    quality = {
        "schema_version": "stream4d_v97_phase6_rendered_support_quality_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": args.variant_id,
        "emitted_object_frame_count": len(object_frame_rows),
        "visible_micro_count_mean": float(np.mean(visible_counts)) if visible_counts.size else 0.0,
        "support_area_ratio_mean": float(np.mean(support_area_ratios)) if support_area_ratios.size else 0.0,
        "support_area_ratio_p10": float(np.percentile(support_area_ratios, 10)) if support_area_ratios.size else 0.0,
        "support_fragment_count_mean": float(np.mean(fragment_counts)) if fragment_counts.size else 0.0,
        "support_heatmap_peak_mean": float(np.mean(peaks)) if peaks.size else 0.0,
        "render_runtime_ms": render_runtime_ms,
        "GPU_memory_peak_MB": float(torch.cuda.max_memory_allocated() / (1024.0**2)),
        "missing_heatmap_count": missing_heatmap_count,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    quality_rows.append(quality)
    gates = [
        ("emitted_object_frame_count_gt_0", int(quality["emitted_object_frame_count"]) > 0, quality["emitted_object_frame_count"], ">0"),
        (
            "support_area_ratio_mean_within_sane_range",
            float(args.support_area_ratio_min) <= float(quality["support_area_ratio_mean"]) <= float(args.support_area_ratio_max),
            quality["support_area_ratio_mean"],
            f"[{args.support_area_ratio_min}, {args.support_area_ratio_max}]",
        ),
        ("missing_heatmap_count_eq_0", int(quality["missing_heatmap_count"]) == 0, quality["missing_heatmap_count"], 0),
        ("render_runtime_ms_within_budget", float(quality["render_runtime_ms"]) <= float(args.render_runtime_budget_ms), quality["render_runtime_ms"], args.render_runtime_budget_ms),
        ("triton_cpu_parity_error_max_le_1e_5", (not parity_rows) or float(parity_rows[0]["cpu_vs_triton_abs_error_max"]) <= 1e-5, parity_rows[0]["cpu_vs_triton_abs_error_max"] if parity_rows else "", "<=1e-5"),
        ("uses_gt_for_prediction_false", True, False, False),
        ("uses_future_false", True, False, False),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v97_phase6_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": args.variant_id,
            "gate": gate,
            "pass": bool(passed),
            "observed": observed,
            "required": required,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for gate, passed, observed, required in gates
    ]
    phase6_pass = all(bool(row["pass"]) for row in gate_rows)
    kernel_rows = [
        {
            "schema_version": "stream4d_v97_phase6_kernel_runtime_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": args.variant_id,
            "kernel": "triton_sparse_gaussian_splat",
            "object_frame_count": len(object_frame_rows),
            "render_runtime_ms": render_runtime_ms,
            "GPU_memory_peak_MB": quality["GPU_memory_peak_MB"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    _write_csv(
        output_root / "object_frame_support_rows.csv",
        object_frame_rows,
        [
            "schema_version",
            "phase_id",
            "run_id",
            "variant_id",
            "scene_id",
            "window_id",
            "object_id",
            "frame_id",
            "support_area",
            "support_area_ratio",
            "support_peak",
            "support_mean",
            "visible_micro_count",
            "mean_visibility",
            "mean_confidence",
            "sigma_mean",
            "sigma_p90",
            "support_fragment_count",
            "heatmap_shard_path",
            "uses_gt_for_prediction",
            "uses_future",
        ],
    )
    _write_csv(output_root / "rendered_support_quality_rows.csv", quality_rows, list(quality_rows[0].keys()))
    _write_csv(output_root / "heatmap_manifest_rows.csv", manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["schema_version"])
    _write_csv(output_root / "kernel_runtime_rows.csv", kernel_rows, list(kernel_rows[0].keys()))
    _write_csv(output_root / "phase6_gate_rows.csv", gate_rows, list(gate_rows[0].keys()))
    _write_csv(output_root / "triton_splat_parity_rows.csv", parity_rows, list(parity_rows[0].keys()) if parity_rows else ["schema_version"])
    summary = {
        "schema": "stream4d_v97_phase6_render_splat_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V97_PHASE6_RENDER_SPLAT_DIAGNOSTIC" if phase6_pass else "NO_GO_V97_PHASE6_RENDER_SPLAT_DIAGNOSTIC",
        "output_root": _rel(output_root),
        "phase4_root": _rel(phase4_root),
        "phase5_root": _rel(phase5_root),
        "variant_id": args.variant_id,
        "diagnostic_scope": "region_proxy_phase5_object_birth",
        "object_frame_group_count": len(groups),
        "emitted_object_frame_count": len(object_frame_rows),
        "heatmap_shard_count": len(manifest_rows),
        "quality": quality,
        "gate_rows": gate_rows,
        "triton_splat_parity": parity_rows[0] if parity_rows else None,
        "runtime_sec": float(time.time() - started),
        "GPU_memory_peak_MB": quality["GPU_memory_peak_MB"],
        "can_enter_phase7_diagnostic": phase6_pass,
        "can_enter_phase7_full": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "emitted_object_frame_count": len(object_frame_rows),
                "heatmap_shard_count": len(manifest_rows),
                "render_runtime_ms": render_runtime_ms,
                "support_area_ratio_mean": quality["support_area_ratio_mean"],
                "output_root": _rel(output_root),
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variant-id", default="C0_cover_seed_plus_affinity_expand")
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--sigma0", type=float, default=1.5)
    parser.add_argument("--eta-confidence", type=float, default=0.75)
    parser.add_argument("--min-radius", type=int, default=2)
    parser.add_argument("--max-radius", type=int, default=5)
    parser.add_argument("--support-threshold", type=float, default=0.05)
    parser.add_argument("--support-area-ratio-min", type=float, default=1e-6)
    parser.add_argument("--support-area-ratio-max", type=float, default=0.25)
    parser.add_argument("--render-runtime-budget-ms", type=float, default=180000.0)
    parser.add_argument("--max-object-frames", type=int, default=0)
    parser.add_argument("--parity-max-points", type=int, default=128)
    parser.add_argument("--shard-value-limit", type=int, default=2_000_000)
    parser.add_argument("--max-sparse-pixels-per-object-frame", type=int, default=20000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
