#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.carrier_store import CarrierSources  # noqa: E402
from stream4d.d4rt_adapter import D4RTAdapter  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.sim3 import Sim3Transform  # noqa: E402
from tools.export_d4rt_grid_surfel_field_v8 import _base_grid_xy, _stable_surfel_id  # noqa: E402
from tools.run_v65_d4rt_stride_overlap_geometry import (  # noqa: E402
    ChunkRecord,
    load_window_without_masks,
    raw_valid_mask,
    resolve_repo,
    save_chunk_npz,
    write_csv,
    write_json,
)
from tools.run_v92_d4rt_window_highres_recompute import _load_windows, _rel, _sha256, _int, project  # noqa: E402


PHASE_ID = "v93_phase7_adaptive_d4rt_sampling"
RUN_ID = "v93_phase7_A512_adaptive_edge_conflict_d4rt_recompute"
DEFAULT_OUTPUT_ROOT = "outputs/audit/v93_phase7_adaptive_d4rt_recompute/A512_adaptive_edge_conflict"
DEFAULT_STRATUM_ROWS = "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic/sampling_stratum_rows.csv"
DEFAULT_EDGE_ROWS = "outputs/audit/v93_phase1_source_edge_registry/mask_edge_hypothesis_rows.csv"
DEFAULT_SOURCE_SUPPORT_ROWS = "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic/d4rt_source_support_rows.csv"
DEFAULT_WINDOW_ROWS = "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        return str(value)
    return value


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_window_id(window_id: str) -> str:
    return str(window_id).replace("/", "_")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _load_budgets(stratum_rows: Path, sampling_plan_id: str) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for row in _read_csv(stratum_rows):
        if row.get("sampling_plan_id") != sampling_plan_id:
            continue
        budget = _int(row.get("query_budget"), 0)
        if budget > 0:
            budgets[str(row.get("stratum_name", ""))] = budget
    if not budgets:
        raise RuntimeError(f"No positive stratum budgets for sampling_plan_id={sampling_plan_id} in {stratum_rows}")
    return budgets


def _load_edge_label_sets(edge_rows: Path) -> dict[tuple[str, int], dict[str, set[int]]]:
    out: dict[tuple[str, int], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in _read_csv(edge_rows):
        if str(row.get("split", "dev")) != "dev":
            continue
        scene = str(row.get("scene_id", ""))
        frame_id = _int(row.get("frame_id"), -1)
        edge_type = str(row.get("edge_type", ""))
        if frame_id < 0 or not edge_type:
            continue
        for name in ("source_mask_id", "edge_mask_id_a", "edge_mask_id_b"):
            value = _int(row.get(name), 0)
            if value > 0:
                out[(scene, frame_id)][edge_type].add(value)
                out[(scene, frame_id)]["any_edge"].add(value)
    return {key: {etype: set(vals) for etype, vals in value.items()} for key, value in out.items()}


def _load_low_support_label_sets(source_support_rows: Path) -> dict[tuple[str, int], set[int]]:
    values: list[float] = []
    rows = []
    for row in _read_csv(source_support_rows):
        if str(row.get("variant_id", "")) != "B0_local_only":
            continue
        ratio = _num(row.get("carrier_support_area_ratio"), 0.0)
        values.append(ratio)
        rows.append(row)
    threshold = float(np.percentile(np.asarray(values, dtype=np.float64), 25)) if values else 0.05
    out: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        ratio = _num(row.get("carrier_support_area_ratio"), 0.0)
        risk = _num(row.get("underseg_risk_score"), 0.0)
        if ratio <= threshold or risk > 0.0:
            out[(str(row.get("scene_id", "")), _int(row.get("frame_id"), -1))].add(_int(row.get("source_mask_id"), 0))
    return {key: value for key, value in out.items()}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(mask_u8, dtype=bool)
    k = max(1, int(radius))
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel)
    eroded = cv2.erode(mask_u8, kernel)
    return (dilated != eroded) & (dilated > 0)


def _inter_label_boundary(labels: np.ndarray) -> np.ndarray:
    labels_i = np.asarray(labels)
    fg = labels_i > 0
    change = np.zeros_like(fg, dtype=bool)
    change[:, 1:] |= labels_i[:, 1:] != labels_i[:, :-1]
    change[:, :-1] |= labels_i[:, 1:] != labels_i[:, :-1]
    change[1:, :] |= labels_i[1:, :] != labels_i[:-1, :]
    change[:-1, :] |= labels_i[1:, :] != labels_i[:-1, :]
    return change & fg


def _label_mask(labels: np.ndarray, label_set: set[int]) -> np.ndarray:
    if not label_set:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray(sorted(label_set), dtype=labels.dtype))


def _select_candidates(
    candidate: np.ndarray,
    *,
    budget: int,
    rng: np.random.Generator,
    used: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if budget <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ys, xs = np.where(candidate)
    if xs.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    keep = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist()) if (int(x), int(y)) not in used]
    if not keep:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if len(keep) > budget:
        order = rng.permutation(len(keep))[:budget]
        keep = [keep[int(i)] for i in order.tolist()]
    out_x = np.asarray([x for x, _y in keep], dtype=np.int64)
    out_y = np.asarray([y for _x, y in keep], dtype=np.int64)
    return out_x, out_y


def _append_points(
    *,
    frame_id: int,
    local_idx: int,
    width: int,
    height: int,
    labels: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    stratum_name: str,
    routing_signal: str,
    used: set[tuple[int, int]],
    arrays: dict[str, list[np.ndarray]],
    query_rows: list[dict[str, Any]],
    sampling_plan_id: str,
    scene: str,
    window_id: str,
) -> None:
    if xs.size == 0:
        return
    keep_x: list[int] = []
    keep_y: list[int] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        key = (int(x), int(y))
        if key in used:
            continue
        used.add(key)
        keep_x.append(int(x))
        keep_y.append(int(y))
    if not keep_x:
        return
    x_arr = np.asarray(keep_x, dtype=np.int64)
    y_arr = np.asarray(keep_y, dtype=np.int64)
    mask_values = labels[y_arr, x_arr].astype(np.int64, copy=False)
    carrier_ids = _stable_surfel_id(int(frame_id), x_arr, y_arr, int(width))
    arrays["carrier_id"].append(carrier_ids)
    arrays["src_frame"].append(np.full(x_arr.shape[0], int(local_idx), dtype=np.int64))
    arrays["src_frame_global"].append(np.full(x_arr.shape[0], int(frame_id), dtype=np.int64))
    arrays["src_xy"].append(np.stack([x_arr, y_arr], axis=1).astype(np.int64, copy=False))
    arrays["src_uv"].append(
        np.stack(
            [
                x_arr.astype(np.float32) / float(max(width - 1, 1)),
                y_arr.astype(np.float32) / float(max(height - 1, 1)),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    )
    arrays["src_mask_id"].append(mask_values)
    for idx, (x, y, mask_id) in enumerate(zip(x_arr.tolist(), y_arr.tolist(), mask_values.tolist())):
        query_rows.append(
            {
                "schema_version": "stream4d_v93_phase7_adaptive_query_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "sampling_plan_id": sampling_plan_id,
                "scene_id": scene,
                "split": "dev",
                "window_id": window_id,
                "frame_id": int(frame_id),
                "local_frame_index": int(local_idx),
                "query_index_in_window": "",
                "stratum_name": stratum_name,
                "routing_signal": routing_signal,
                "carrier_id": int(carrier_ids[idx]),
                "src_x_px": int(x),
                "src_y_px": int(y),
                "src_uv_x": float(x_arr[idx] / float(max(width - 1, 1))),
                "src_uv_y": float(y_arr[idx] / float(max(height - 1, 1))),
                "src_mask_id": int(mask_id),
                "uses_gt_for_routing": False,
                "uses_future": False,
            }
        )


def _adaptive_sources(
    *,
    masks: np.ndarray,
    frame_ids: list[int],
    scene: str,
    window_id: str,
    sampling_plan_id: str,
    budgets: dict[str, int],
    edge_label_sets: dict[tuple[str, int], dict[str, set[int]]],
    low_support_label_sets: dict[tuple[str, int], set[int]],
) -> tuple[CarrierSources, list[dict[str, Any]], list[dict[str, Any]]]:
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [T,H,W], got {masks.shape}")
    _, height, width = masks.shape
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    query_rows: list[dict[str, Any]] = []
    stratum_counter: Counter[str] = Counter()

    for local_idx, frame_id in enumerate(frame_ids):
        labels = np.asarray(masks[local_idx])
        fg = labels > 0
        edge_sets = edge_label_sets.get((scene, int(frame_id)), {})
        low_support = low_support_label_sets.get((scene, int(frame_id)), set())
        used: set[tuple[int, int]] = set()
        rng = np.random.default_rng(_stable_seed(f"{sampling_plan_id}:{scene}:{window_id}:{frame_id}"))

        for stratum_name, budget in budgets.items():
            candidate = np.zeros(labels.shape, dtype=bool)
            routing_signal = "unknown"
            if stratum_name == "uniform_base":
                grid = max(1, int(np.ceil(np.sqrt(float(max(1, budget))))))
                xs, ys = _base_grid_xy(height, width, grid, 0.02)
                if xs.size > budget:
                    order = rng.permutation(xs.size)[:budget]
                    xs = xs[order]
                    ys = ys[order]
                routing_signal = "uniform_image_grid"
                _append_points(
                    frame_id=int(frame_id),
                    local_idx=local_idx,
                    width=width,
                    height=height,
                    labels=labels,
                    xs=xs,
                    ys=ys,
                    stratum_name=stratum_name,
                    routing_signal=routing_signal,
                    used=used,
                    arrays=arrays,
                    query_rows=query_rows,
                    sampling_plan_id=sampling_plan_id,
                    scene=scene,
                    window_id=window_id,
                )
                stratum_counter[stratum_name] += int(xs.size)
                continue
            if stratum_name == "object_interior":
                dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
                candidate = fg & (dist >= 3.0)
                if not np.any(candidate):
                    candidate = fg
                routing_signal = "foreground_distance_transform"
            elif stratum_name == "source_outer_boundary":
                candidate = _boundary(fg, radius=2)
                routing_signal = "source_mask_outer_boundary"
            elif stratum_name == "nested_overlap_boundary":
                candidate = _boundary(_label_mask(labels, edge_sets.get("nested_overlap", set())), radius=2)
                routing_signal = "phase1_nested_overlap_edge_labels"
            elif stratum_name == "competing_mask_boundary":
                candidate = _boundary(_label_mask(labels, edge_sets.get("competing", set())), radius=2)
                routing_signal = "phase1_competing_edge_labels"
            elif stratum_name == "semantic_gradient":
                candidate = _inter_label_boundary(labels)
                routing_signal = "mask_label_gradient_proxy_no_gt"
            elif stratum_name == "conflict_underseg":
                labels_for_conflict = set(edge_sets.get("competing", set())) | set(low_support)
                candidate = _boundary(_label_mask(labels, labels_for_conflict), radius=3)
                routing_signal = "competing_or_low_support_underseg_proxy"
            elif stratum_name == "uncertainty_flip_jitter":
                candidate = _boundary(_label_mask(labels, low_support), radius=3)
                routing_signal = "low_support_underseg_proxy"
            elif stratum_name == "partwhole_overlap":
                candidate = _boundary(_label_mask(labels, edge_sets.get("nested_overlap", set())), radius=4)
                routing_signal = "phase1_nested_overlap_partwhole_proxy"
            xs, ys = _select_candidates(candidate, budget=int(budget), rng=rng, used=used)
            if xs.size == 0 and stratum_name not in {"nested_overlap_boundary", "uncertainty_flip_jitter", "partwhole_overlap"}:
                fallback = fg if np.any(fg) else np.ones(labels.shape, dtype=bool)
                xs, ys = _select_candidates(fallback, budget=int(budget), rng=rng, used=used)
                routing_signal += "_fallback_foreground"
            _append_points(
                frame_id=int(frame_id),
                local_idx=local_idx,
                width=width,
                height=height,
                labels=labels,
                xs=xs,
                ys=ys,
                stratum_name=stratum_name,
                routing_signal=routing_signal,
                used=used,
                arrays=arrays,
                query_rows=query_rows,
                sampling_plan_id=sampling_plan_id,
                scene=scene,
                window_id=window_id,
            )
            stratum_counter[stratum_name] += int(xs.size)

    if not arrays["carrier_id"]:
        empty = CarrierSources(
            carrier_id=np.empty((0,), dtype=np.int64),
            src_frame=np.empty((0,), dtype=np.int64),
            src_frame_global=np.empty((0,), dtype=np.int64),
            src_xy=np.empty((0, 2), dtype=np.int64),
            src_uv=np.empty((0, 2), dtype=np.float32),
            src_mask_id=np.empty((0,), dtype=np.int64),
        )
        return empty, query_rows, []
    for idx, row in enumerate(query_rows):
        row["query_index_in_window"] = idx
    sources = CarrierSources(
        carrier_id=np.concatenate(arrays["carrier_id"], axis=0),
        src_frame=np.concatenate(arrays["src_frame"], axis=0),
        src_frame_global=np.concatenate(arrays["src_frame_global"], axis=0),
        src_xy=np.concatenate(arrays["src_xy"], axis=0),
        src_uv=np.concatenate(arrays["src_uv"], axis=0),
        src_mask_id=np.concatenate(arrays["src_mask_id"], axis=0),
    )
    stratum_rows = [
        {
            "schema_version": "stream4d_v93_phase7_sampling_stratum_actual_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": sampling_plan_id,
            "scene_id": scene,
            "window_id": window_id,
            "stratum_name": name,
            "query_budget_per_frame": int(budgets.get(name, 0)),
            "frame_count": len(frame_ids),
            "expected_query_budget": int(budgets.get(name, 0)) * len(frame_ids),
            "actual_query_count": int(stratum_counter.get(name, 0)),
            "normalization_weight": 1.0 / float(max(1, int(budgets.get(name, 0)))),
            "uses_gt_for_routing": False,
            "uses_future": False,
        }
        for name in budgets
    ]
    return sources, query_rows, stratum_rows


def _run_window(
    *,
    args: argparse.Namespace,
    adapter: D4RTAdapter,
    stream: ScanNetStream,
    window: dict[str, Any],
    window_index: int,
    out_root: Path,
    budgets: dict[str, int],
    edge_label_sets: dict[tuple[str, int], dict[str, set[int]]],
    low_support_label_sets: dict[tuple[str, int], set[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.time()
    frame_ids = [int(v) for v in window["frame_ids"]]
    model_frame_ids = frame_ids if len(frame_ids) >= 2 else [frame_ids[0], frame_ids[0]]
    data = load_window_without_masks(stream, model_frame_ids)
    sources, query_rows, stratum_rows = _adaptive_sources(
        masks=np.asarray(data["mask"]),
        frame_ids=model_frame_ids,
        scene=str(args.scene),
        window_id=str(window["window_id"]),
        sampling_plan_id=str(args.sampling_plan_id),
        budgets=budgets,
        edge_label_sets=edge_label_sets,
        low_support_label_sets=low_support_label_sets,
    )
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch = adapter.infer_carriers(
        video_rgb_uint8=np.asarray(data["rgb"]),
        src_uv_norm=sources.src_uv,
        src_frame_local=sources.src_frame,
        carrier_id=sources.carrier_id,
        src_frame_global=sources.src_frame_global,
        src_xy=sources.src_xy,
        src_mask_id=sources.src_mask_id,
        query_chunk_size=int(args.query_chunk_size),
    )
    peak_memory_gb = ""
    if torch is not None and torch.cuda.is_available():
        peak_memory_gb = float(torch.cuda.max_memory_allocated() / (1024.0**3))
    chunk = ChunkRecord(
        chunk_index=window_index,
        frame_ids=frame_ids,
        xyz=np.asarray(batch.xyz_ref, dtype=np.float32)[: len(frame_ids)],
        uv=np.asarray(batch.uv_pred, dtype=np.float32)[: len(frame_ids)],
        valid=np.asarray(batch.valid, dtype=bool)[: len(frame_ids)],
        visibility=np.asarray(batch.visibility_prob, dtype=np.float32)[: len(frame_ids)],
        confidence=np.asarray(batch.confidence_prob, dtype=np.float32)[: len(frame_ids)],
        carrier_id=np.asarray(batch.carrier_id, dtype=np.int64),
        src_frame_global=np.asarray(batch.src_frame_global, dtype=np.int64),
        src_xy=np.asarray(batch.src_xy, dtype=np.int64),
        transform_to_scene=Sim3Transform(scale=1.0, rot=np.eye(3, dtype=np.float64), trans=np.zeros(3, dtype=np.float64)),
    )
    window_dir = out_root / "windows"
    npz_path = window_dir / f"window_{_safe_window_id(str(window['window_id']))}.npz"
    save_chunk_npz(npz_path, chunk)
    valid_observation_count = int(np.count_nonzero(raw_valid_mask(chunk, args)))
    seconds = float(time.time() - started)
    row = {
        "scene": args.scene,
        "split": args.split,
        "stride": int(args.stride),
        "window_id": window["window_id"],
        "window_index": window["window_index"],
        "frame_start": int(frame_ids[0]),
        "frame_end": int(frame_ids[-1]),
        "frame_ids": ",".join(str(v) for v in frame_ids),
        "model_frame_ids": ",".join(str(v) for v in model_frame_ids),
        "num_frames": int(len(frame_ids)),
        "num_model_frames": int(len(model_frame_ids)),
        "sampling_plan_id": str(args.sampling_plan_id),
        "query_budget_per_frame": int(sum(budgets.values())),
        "source_count": int(sources.carrier_id.shape[0]),
        "valid_observation_count": valid_observation_count,
        "runtime_sec_per_window": seconds,
        "peak_memory_gb": peak_memory_gb,
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "window_npz": _rel(npz_path),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "protocol_note": "D4RT forward input is restricted to local-window frames; source UVs are generated from non-GT mask/edge diagnostics.",
        "single_frame_window_padding": len(frame_ids) == 1,
    }
    return row, query_rows, stratum_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_root = project(args.output_root) / args.scene / f"stride_{int(args.stride)}"
    if out_root.exists() and bool(args.clean):
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8"
    )
    budgets = _load_budgets(project(args.stratum_rows), str(args.sampling_plan_id))
    edge_label_sets = _load_edge_label_sets(project(args.edge_rows))
    low_support_label_sets = _load_low_support_label_sets(project(args.source_support_rows))
    windows = _load_windows(args)
    if not windows:
        raise RuntimeError(f"No windows found for scene={args.scene} split={args.split}")
    stream = ScanNetStream(seq_name=args.scene, root=resolve_repo(args.scannet_root))
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    adapter = D4RTAdapter(
        d4rt_root=resolve_repo(args.d4rt_root),
        model_config=resolve_repo(args.d4rt_config),
        ckpt_path=resolve_repo(args.d4rt_ckpt),
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    all_stratum_rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        print(
            f"[v93-adaptive-d4rt] scene={args.scene} window={window['window_id']} "
            f"frames={window['frame_id_start']}..{window['frame_id_end']} "
            f"n={len(window['frame_ids'])} plan={args.sampling_plan_id} qpf={sum(budgets.values())}",
            flush=True,
        )
        row, query_rows, stratum_rows = _run_window(
            args=args,
            adapter=adapter,
            stream=stream,
            window=window,
            window_index=idx,
            out_root=out_root,
            budgets=budgets,
            edge_label_sets=edge_label_sets,
            low_support_label_sets=low_support_label_sets,
        )
        rows.append(row)
        all_query_rows.extend(query_rows)
        all_stratum_rows.extend(stratum_rows)
    write_csv(out_root / "window_rows.csv", rows)
    write_csv(out_root / "sampling_query_rows.csv", all_query_rows)
    write_csv(out_root / "sampling_stratum_rows.csv", all_stratum_rows)
    runtime_rows = [
        {
            "schema_version": "stream4d_v93_phase7_runtime_memory_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": str(args.sampling_plan_id),
            "scene_id": args.scene,
            "window_id": row.get("window_id", ""),
            "runtime_sec_per_window": row.get("runtime_sec_per_window", ""),
            "peak_memory_gb": row.get("peak_memory_gb", ""),
            "source_count": row.get("source_count", ""),
            "valid_observation_count": row.get("valid_observation_count", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": args.device,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in rows
    ]
    write_csv(out_root / "runtime_memory_rows.csv", runtime_rows)
    summary = {
        "schema": "stream4d_v93_phase7_adaptive_d4rt_recompute_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "sampling_plan_id": str(args.sampling_plan_id),
        "scene": args.scene,
        "split": args.split,
        "stride": int(args.stride),
        "query_budget_per_frame": int(sum(budgets.values())),
        "stratum_budgets": budgets,
        "window_count": len(rows),
        "frame_count": len({frame for row in windows for frame in row["frame_ids"]}),
        "source_count": int(sum(_int(row.get("source_count"), 0) for row in rows)),
        "valid_observation_count": int(sum(_int(row.get("valid_observation_count"), 0) for row in rows)),
        "runtime_sec_total": float(sum(_num(row.get("runtime_sec_per_window"), 0.0) for row in rows)),
        "duration_sec": float(time.time() - started),
        "peak_memory_gb_max": max([_num(row.get("peak_memory_gb"), 0.0) for row in rows], default=0.0),
        "sampling_query_rows": _rel(out_root / "sampling_query_rows.csv"),
        "sampling_stratum_rows": _rel(out_root / "sampling_stratum_rows.csv"),
        "window_rows_csv": _rel(out_root / "window_rows.csv"),
        "windows_dir": _rel(out_root / "windows"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "query_chunk_size": int(args.query_chunk_size),
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "created_at": _created_at(),
    }
    write_json(out_root / "summary.json", summary)
    sha_paths = [
        out_root / "summary.json",
        out_root / "window_rows.csv",
        out_root / "sampling_query_rows.csv",
        out_root / "sampling_stratum_rows.csv",
        out_root / "runtime_memory_rows.csv",
        out_root / "last_command.txt",
    ]
    _write_json(out_root / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run window-safe D4RT recompute with v93 adaptive source UV sampling.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--sampling-plan-id", default="A512_adaptive_edge_conflict")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-support-rows", default=DEFAULT_WINDOW_ROWS)
    parser.add_argument("--stratum-rows", default=DEFAULT_STRATUM_ROWS)
    parser.add_argument("--edge-rows", default=DEFAULT_EDGE_ROWS)
    parser.add_argument("--source-support-rows", default=DEFAULT_SOURCE_SUPPORT_ROWS)
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--uv-radius", type=float, default=0.002)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
