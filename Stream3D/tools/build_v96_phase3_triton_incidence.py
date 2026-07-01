#!/usr/bin/env python3
"""Build v96 Phase3 point-to-mask incidence with Triton lookup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402


PHASE_ID = "v96_phase3_triton_incidence"
RUN_ID = "v96_phase3_triton_incidence"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase3_triton_incidence"
EVENT_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "decode_variant",
    "query_variant",
    "scene_id",
    "window_id",
    "query_id",
    "source_frame_id",
    "target_frame_id",
    "query_stratum",
    "u_tgt",
    "v_tgt",
    "center_mask_id",
    "distinct_mask_count_3x3",
    "boundary_distance_px",
    "query_has_positive_mask",
    "query_has_multiple_masks_3x3",
    "uses_gt_for_prediction",
    "uses_future",
]


@triton.jit
def _incidence_lookup_kernel(
    x_ptr,
    y_ptr,
    frame_ptr,
    label_ptr,
    dist_ptr,
    center_out,
    distinct_out,
    dist_out,
    n: tl.constexpr,
    h: tl.constexpr,
    w: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    valid = offsets < n
    x = tl.load(x_ptr + offsets, mask=valid, other=-1.0)
    y = tl.load(y_ptr + offsets, mask=valid, other=-1.0)
    f = tl.load(frame_ptr + offsets, mask=valid, other=0)
    xi = tl.floor(x + 0.5).to(tl.int32)
    yi = tl.floor(y + 0.5).to(tl.int32)
    inside = valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    base = f * h * w + yi * w + xi
    center = tl.load(label_ptr + base, mask=inside, other=0)
    bdist = tl.load(dist_ptr + base, mask=inside, other=-1.0)

    l0 = tl.full((block,), 0, tl.int32)
    l1 = tl.full((block,), 0, tl.int32)
    l2 = tl.full((block,), 0, tl.int32)
    l3 = tl.full((block,), 0, tl.int32)
    count = tl.full((block,), 0, tl.int32)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            nx = xi + dx
            ny = yi + dy
            ninside = inside & (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            lab = tl.load(label_ptr + f * h * w + ny * w + nx, mask=ninside, other=0)
            is_new = (lab > 0) & (lab != l0) & (lab != l1) & (lab != l2) & (lab != l3)
            put0 = is_new & (count == 0)
            put1 = is_new & (count == 1)
            put2 = is_new & (count == 2)
            put3 = is_new & (count == 3)
            l0 = tl.where(put0, lab, l0)
            l1 = tl.where(put1, lab, l1)
            l2 = tl.where(put2, lab, l2)
            l3 = tl.where(put3, lab, l3)
            count = tl.where(is_new & (count < 4), count + 1, count)

    tl.store(center_out + offsets, center, mask=valid)
    tl.store(distinct_out + offsets, count, mask=valid)
    tl.store(dist_out + offsets, bdist, mask=valid)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_include(raw: str) -> tuple[Path, set[str]]:
    parts = raw.split("::")
    root = _project(parts[0])
    scenes: set[str] = set()
    for part in parts[1:]:
        if part.startswith("scene="):
            scenes.add(part.split("=", 1)[1])
        elif part:
            raise ValueError(f"unknown include filter {part!r}; use ::scene=<scene_id>")
    return root, scenes


def _mask_path_lookup(source_rows: Path) -> dict[tuple[str, str, int], Path]:
    out: dict[tuple[str, str, int], Path] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        raw = row.get("mask_path", "")
        if raw and key not in out:
            out[key] = _project(raw)
    return out


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _label_boundary(label: np.ndarray, band_px: int = 1) -> np.ndarray:
    positive = label > 0
    edge = np.zeros(label.shape, dtype=np.uint8)
    diff_x = label[:, 1:] != label[:, :-1]
    diff_y = label[1:, :] != label[:-1, :]
    edge[:, 1:] |= diff_x
    edge[:, :-1] |= diff_x
    edge[1:, :] |= diff_y
    edge[:-1, :] |= diff_y
    edge &= positive.astype(np.uint8)
    if band_px > 0:
        kernel = np.ones((band_px * 2 + 1, band_px * 2 + 1), dtype=np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
    return edge.astype(bool) & positive


def _boundary_distance(label: np.ndarray) -> np.ndarray:
    boundary = _label_boundary(label, band_px=1)
    if not np.any(boundary):
        return np.full(label.shape, 1e6, dtype=np.float32)
    not_boundary = (~boundary).astype(np.uint8)
    return cv2.distanceTransform(not_boundary, cv2.DIST_L2, 3).astype(np.float32)


def _load_track_rows(include_roots: list[str], decode_variants: set[str], max_track_rows: int) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    manifest: list[dict[str, Any]] = []
    for raw in include_roots:
        root, scenes = _parse_include(raw)
        track_path = root / "micro_track_rows.csv"
        if not track_path.exists():
            raise FileNotFoundError(f"missing micro_track_rows.csv under {root}")
        source_count = 0
        kept = 0
        with track_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source_count += 1
                if decode_variants and row.get("decode_variant", "") not in decode_variants:
                    continue
                if scenes and row.get("scene_id", "") not in scenes:
                    continue
                rows.append(row)
                kept += 1
                if max_track_rows > 0 and len(rows) >= max_track_rows:
                    break
        manifest.append(
            {
                "include_arg": raw,
                "source_root": _rel(root),
                "scene_filter": ",".join(sorted(scenes)) if scenes else "*",
                "source_track_rows_seen": source_count,
                "kept_track_rows": kept,
            }
        )
        if max_track_rows > 0 and len(rows) >= max_track_rows:
            break
    return rows, manifest


def _load_frame_tensors(
    rows: list[dict[str, str]],
    mask_lookup: dict[tuple[str, str, int], Path],
    scannet_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str, int], int], list[dict[str, Any]]]:
    frame_keys = sorted({(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("target_frame_id")))) for row in rows})
    labels: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    shape_hw: tuple[int, int] | None = None
    for frame_index, key in enumerate(frame_keys):
        scene, window, frame_id = key
        mask_path = mask_lookup.get(key)
        if mask_path is not None and mask_path.exists():
            label = _load_label(mask_path)
            mask_source = _rel(mask_path)
        else:
            stream = ScanNetStream(scene, root=scannet_root)
            label = np.asarray(stream.load_mask(frame_id), dtype=np.int32)
            mask_source = _rel(stream.mask_dir / f"{frame_id}.png")
        if shape_hw is None:
            shape_hw = (int(label.shape[0]), int(label.shape[1]))
        elif shape_hw != (int(label.shape[0]), int(label.shape[1])):
            raise ValueError(f"mixed label shapes are not supported: saw {shape_hw} and {label.shape}")
        labels.append(label.astype(np.int32, copy=False))
        distances.append(_boundary_distance(label))
        manifest.append(
            {
                "frame_tensor_index": frame_index,
                "scene_id": scene,
                "window_id": window,
                "target_frame_id": frame_id,
                "mask_source": mask_source,
                "height": int(label.shape[0]),
                "width": int(label.shape[1]),
            }
        )
    return np.stack(labels, axis=0), np.stack(distances, axis=0), {key: idx for idx, key in enumerate(frame_keys)}, manifest


def _cpu_lookup_one(label: np.ndarray, dist: np.ndarray, x: float, y: float) -> tuple[int, int, float]:
    h, w = label.shape
    xi = int(math.floor(float(x) + 0.5))
    yi = int(math.floor(float(y) + 0.5))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return 0, 0, -1.0
    center = int(label[yi, xi])
    labels: set[int] = set()
    for yy in range(max(0, yi - 1), min(h, yi + 2)):
        for xx in range(max(0, xi - 1), min(w, xi + 2)):
            value = int(label[yy, xx])
            if value > 0:
                labels.add(value)
    return center, min(len(labels), 4), float(dist[yi, xi])


def _summarize_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["decode_variant"]].append(row)
    out: list[dict[str, Any]] = []
    for variant, vals in sorted(by_variant.items()):
        total = len(vals)
        positive = sum(1 for row in vals if row["query_has_positive_mask"])
        boundary_rows = [row for row in vals if row.get("query_stratum") == "boundary"]
        conflict_rows = [row for row in vals if row.get("query_stratum") == "conflict"]
        out.append(
            {
                "decode_variant": variant,
                "query_variant": vals[0].get("query_variant", ""),
                "incidence_event_count": total,
                "query_with_positive_mask_rate": positive / max(1, total),
                "boundary_query_with_mask_rate": sum(1 for row in boundary_rows if row["query_has_positive_mask"]) / max(1, len(boundary_rows)),
                "conflict_query_with_multiple_masks_rate": sum(1 for row in conflict_rows if row["query_has_multiple_masks_3x3"]) / max(1, len(conflict_rows)),
                "mean_masks_per_query": float(np.mean([int(row["distinct_mask_count_3x3"]) for row in vals])) if vals else 0.0,
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase3 Triton incidence requires CUDA.")
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    decode_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    track_rows, include_manifest = _load_track_rows(args.include_root, decode_variants, int(args.max_track_rows))
    if not track_rows:
        raise RuntimeError("No micro_track rows selected for Phase3.")
    mask_lookup = _mask_path_lookup(_project(args.source_rows))
    labels_np, dist_np, frame_lookup, frame_manifest = _load_frame_tensors(track_rows, mask_lookup, _project(args.scannet_root))
    h, w = int(labels_np.shape[1]), int(labels_np.shape[2])

    x_np = np.asarray([_num(row.get("u_tgt"), -1.0) for row in track_rows], dtype=np.float32)
    y_np = np.asarray([_num(row.get("v_tgt"), -1.0) for row in track_rows], dtype=np.float32)
    frame_np = np.asarray(
        [frame_lookup[(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("target_frame_id"))))] for row in track_rows],
        dtype=np.int32,
    )
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    labels_t = torch.from_numpy(labels_np.reshape(-1)).to(device=device, dtype=torch.int32)
    dist_t = torch.from_numpy(dist_np.reshape(-1)).to(device=device, dtype=torch.float32)
    x_t = torch.from_numpy(x_np).to(device=device)
    y_t = torch.from_numpy(y_np).to(device=device)
    frame_t = torch.from_numpy(frame_np).to(device=device, dtype=torch.int32)
    n = int(x_np.shape[0])
    center_t = torch.empty((n,), device=device, dtype=torch.int32)
    distinct_t = torch.empty((n,), device=device, dtype=torch.int32)
    bdist_t = torch.empty((n,), device=device, dtype=torch.float32)
    block = int(args.triton_block_size)
    grid = (triton.cdiv(n, block),)
    kernel_start = time.time()
    _incidence_lookup_kernel[grid](
        x_t,
        y_t,
        frame_t,
        labels_t,
        dist_t,
        center_t,
        distinct_t,
        bdist_t,
        n,
        h,
        w,
        block,
    )
    torch.cuda.synchronize(device)
    runtime_incidence_sec = float(time.time() - kernel_start)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    center_np = center_t.cpu().numpy()
    distinct_np = distinct_t.cpu().numpy()
    bdist_np = bdist_t.cpu().numpy()

    parity_count = min(int(args.parity_sample_rows), n)
    sample_indices = np.linspace(0, n - 1, parity_count, dtype=np.int64) if parity_count > 0 else np.zeros((0,), dtype=np.int64)
    label_mismatch = 0
    distinct_mismatch = 0
    distance_errors: list[float] = []
    for idx in sample_indices.tolist():
        fidx = int(frame_np[idx])
        cpu_center, cpu_distinct, cpu_dist = _cpu_lookup_one(labels_np[fidx], dist_np[fidx], float(x_np[idx]), float(y_np[idx]))
        label_mismatch += int(cpu_center != int(center_np[idx]))
        distinct_mismatch += int(cpu_distinct != int(distinct_np[idx]))
        distance_errors.append(abs(cpu_dist - float(bdist_np[idx])))
    parity_row = {
        "schema_version": "stream4d_v96_phase3_cpu_triton_parity_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "parity_sample_rows": int(parity_count),
        "cpu_vs_triton_membership_mismatch_rate": float(label_mismatch / max(1, parity_count)),
        "cpu_vs_triton_distinct_count_mismatch_rate": float(distinct_mismatch / max(1, parity_count)),
        "cpu_vs_triton_distance_error_mean": float(np.mean(distance_errors)) if distance_errors else 0.0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    event_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(track_rows):
        event_rows.append(
            {
                "schema_version": "stream4d_v96_incidence_event_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "decode_variant": row.get("decode_variant", ""),
                "query_variant": row.get("query_variant", ""),
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "query_id": row.get("query_id", ""),
                "source_frame_id": int(_num(row.get("source_frame_id"))),
                "target_frame_id": int(_num(row.get("target_frame_id"))),
                "query_stratum": row.get("query_stratum", ""),
                "u_tgt": float(_num(row.get("u_tgt"))),
                "v_tgt": float(_num(row.get("v_tgt"))),
                "center_mask_id": int(center_np[idx]),
                "distinct_mask_count_3x3": int(distinct_np[idx]),
                "boundary_distance_px": float(bdist_np[idx]),
                "query_has_positive_mask": bool(int(center_np[idx]) > 0),
                "query_has_multiple_masks_3x3": bool(int(distinct_np[idx]) >= 2),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    variant_rows = _summarize_variant(event_rows)
    positive_rates = [row["query_with_positive_mask_rate"] for row in variant_rows]
    gate_rows = [
        {
            "schema_version": "stream4d_v96_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "query_with_positive_mask_rate_ge_0p60",
            "pass": bool(positive_rates and min(positive_rates) >= 0.60),
            "observed": min(positive_rates) if positive_rates else "",
            "required": 0.60,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v96_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "cpu_vs_triton_membership_mismatch_rate_le_1e_minus_4",
            "pass": bool(parity_row["cpu_vs_triton_membership_mismatch_rate"] <= 1e-4),
            "observed": parity_row["cpu_vs_triton_membership_mismatch_rate"],
            "required": 1e-4,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v96_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "runtime_incidence_recorded_no_oom",
            "pass": bool(runtime_incidence_sec > 0),
            "observed": runtime_incidence_sec,
            "required": "runtime>0",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    decision = "PASS_V96_PHASE3_TRITON_INCIDENCE" if all(bool(row.get("pass")) for row in gate_rows) else "NO_GO_V96_PHASE3_TRITON_INCIDENCE"
    _write_csv(output_root / "include_manifest_rows.csv", include_manifest)
    _write_csv(output_root / "frame_tensor_manifest_rows.csv", frame_manifest)
    _write_csv(output_root / "incidence_event_rows.csv", event_rows)
    _write_csv(output_root / "variant_metric_rows.csv", variant_rows)
    _write_csv(output_root / "cpu_triton_parity_rows.csv", [parity_row])
    _write_csv(output_root / "phase3_gate_rows.csv", gate_rows)
    tensor_manifest = {
        "schema": "stream4d_v96_phase3_incidence_tensor_manifest_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "label_tensor_shape": list(labels_np.shape),
        "track_row_count": n,
        "height": h,
        "width": w,
        "device": str(device),
        "triton_block_size": block,
        "runtime_incidence_sec": runtime_incidence_sec,
        "GPU_memory_peak_MB": peak_mb,
    }
    _write_json(output_root / "incidence_tensor_manifest.json", tensor_manifest)
    summary = {
        "schema": "stream4d_v96_phase3_triton_incidence_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "output_root": _rel(output_root),
        "selected_track_rows": n,
        "unique_frame_tensors": int(labels_np.shape[0]),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "cpu_triton_parity_rows": _rel(output_root / "cpu_triton_parity_rows.csv"),
        "phase3_gate_rows": _rel(output_root / "phase3_gate_rows.csv"),
        "incidence_event_rows": _rel(output_root / "incidence_event_rows.csv"),
        "incidence_tensor_manifest": _rel(output_root / "incidence_tensor_manifest.json"),
        "runtime_incidence_sec": runtime_incidence_sec,
        "runtime_total_sec": float(time.time() - started),
        "GPU_memory_peak_MB": peak_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "variant_summaries": variant_rows,
        "parity": parity_row,
        "gate_rows": gate_rows,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": decision, "selected_track_rows": n, "runtime_incidence_sec": runtime_incidence_sec, "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 Phase3 Triton incidence rows.")
    parser.add_argument("--include-root", action="append", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--decode-variants", default="D3_adaptive1024")
    parser.add_argument("--max-track-rows", type=int, default=0)
    parser.add_argument("--parity-sample-rows", type=int, default=4096)
    parser.add_argument("--triton-block-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
