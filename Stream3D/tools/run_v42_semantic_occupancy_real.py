from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import stable_source_carrier_id
from stream4d_native.semantic_occupancy import MaterialQuery
from stream4d_native.semantic_part_tokens import stack_to_masks
from tools.export_v21_3_occupancy_carrier_cache import (
    _json_safe,
    _load_rgb_sparse_mask_window,
    _save_batch,
    _source_arrays_from_points,
)
from tools.run_v21_3_native_occupancy_ablation import _frame_ids


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(row.get(key)), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "") for key in keys})


def _load_external_query_masks(
    *,
    root: Path,
    scene: str,
    source: str,
    sample_dir: str,
    frame_ids: list[int],
    image_shape: tuple[int, int],
    min_area: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    out = np.zeros((len(frame_ids), height, width), dtype=np.int32)
    scene_source_dir = root / str(scene) / str(source) / str(sample_dir)
    flat_source_dir = root / str(source) / str(sample_dir)
    source_dir = scene_source_dir if scene_source_dir.exists() else flat_source_dir
    loaded: list[int] = []
    missing: list[int] = []
    total_masks = 0
    for frame_rank, frame_id in enumerate(frame_ids):
        path = source_dir / f"{source}_frame{int(frame_id):06d}_masks.npz"
        if not path.exists():
            missing.append(int(frame_id))
            continue
        with np.load(path) as data:
            masks = data["masks"]
        label_id = 1
        for _mask_id, mask in stack_to_masks(masks, min_area=int(min_area)):
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != (height, width):
                raise ValueError(
                    f"external query mask shape mismatch for {path}: got {binary.shape}, expected {(height, width)}"
                )
            out[frame_rank][binary] = int(label_id)
            label_id += 1
            total_masks += 1
        loaded.append(int(frame_id))
    return out, {
        "semantic_query_mask_source": str(source),
        "semantic_query_mask_root": str(root),
        "semantic_query_sample_dir": str(sample_dir),
        "semantic_query_loaded_frame_ids": loaded,
        "semantic_query_missing_frame_ids": missing,
        "semantic_query_mask_count": int(total_masks),
    }


def _split_interior_boundary(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    foreground = labels > 0
    boundary = np.zeros_like(foreground, dtype=bool)
    boundary[:, 1:, :] |= foreground[:, 1:, :] & (labels[:, 1:, :] != labels[:, :-1, :])
    boundary[:, :-1, :] |= foreground[:, :-1, :] & (labels[:, :-1, :] != labels[:, 1:, :])
    boundary[:, :, 1:] |= foreground[:, :, 1:] & (labels[:, :, 1:] != labels[:, :, :-1])
    boundary[:, :, :-1] |= foreground[:, :, :-1] & (labels[:, :, :-1] != labels[:, :, 1:])
    return foreground & ~boundary, boundary


def _sample_fixed_grid(num_frames: int, height: int, width: int, count: int, *, score: float) -> list[MaterialQuery]:
    count = max(0, int(count))
    if count == 0:
        return []
    frame_count = max(1, int(num_frames))
    per_frame = max(1, int(np.ceil(float(count) / frame_count)))
    side = max(1, int(np.ceil(np.sqrt(per_frame))))
    ys = np.linspace(0, max(height - 1, 0), num=side, dtype=np.int64)
    xs = np.linspace(0, max(width - 1, 0), num=side, dtype=np.int64)
    out: list[MaterialQuery] = []
    for frame_rank in range(frame_count):
        for y in ys.tolist():
            for x in xs.tolist():
                out.append(MaterialQuery(frame_rank=int(frame_rank), y=int(y), x=int(x), reason="fixed_grid", score=float(score)))
                if len(out) >= count:
                    return out
    return out[:count]


def _sample_region(
    region: np.ndarray,
    count: int,
    *,
    reason: str,
    score: float,
    used: set[tuple[int, int, int]],
) -> list[MaterialQuery]:
    count = max(0, int(count))
    if count == 0:
        return []
    region = np.asarray(region, dtype=bool)
    frame_ranks = [int(frame) for frame in np.flatnonzero(np.any(region, axis=(1, 2))).tolist()]
    if not frame_ranks:
        return []
    per_frame = max(1, int(np.ceil(float(count) / len(frame_ranks))))
    out: list[MaterialQuery] = []
    for frame_rank in frame_ranks:
        ys, xs = np.nonzero(region[frame_rank])
        if ys.size == 0:
            continue
        order = np.linspace(0, ys.size - 1, num=min(per_frame, ys.size), dtype=np.int64)
        for idx in order.tolist():
            key = (int(frame_rank), int(ys[idx]), int(xs[idx]))
            if key in used:
                continue
            used.add(key)
            out.append(MaterialQuery(frame_rank=key[0], y=key[1], x=key[2], reason=reason, score=float(score)))
            if len(out) >= count:
                return out
    return out


def _semantic_disagreement_proxy(foreground: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    # Training-free proxy for real source disagreement: boundary-adjacent foreground where semantic masks are least stable.
    dilated = boundary.copy()
    dilated[:, 1:, :] |= boundary[:, :-1, :]
    dilated[:, :-1, :] |= boundary[:, 1:, :]
    dilated[:, :, 1:] |= boundary[:, :, :-1]
    dilated[:, :, :-1] |= boundary[:, :, 1:]
    return np.asarray(foreground & dilated, dtype=bool)


def _schedule_queries(masks: np.ndarray, *, variant: str, budget: int) -> tuple[list[MaterialQuery], dict[str, Any]]:
    masks = np.asarray(masks, dtype=np.int32)
    foreground = masks > 0
    interior, boundary = _split_interior_boundary(masks)
    mask_frame_ranks = [int(frame) for frame in np.flatnonzero(np.any(foreground, axis=(1, 2))).tolist()]
    overlap = np.zeros_like(foreground, dtype=bool)
    for frame_rank in mask_frame_ranks:
        overlap[frame_rank] = foreground[frame_rank]
    disagreement = _semantic_disagreement_proxy(foreground, boundary)
    exploration = ~foreground

    if variant == "Q0":
        queries = _sample_fixed_grid(masks.shape[0], masks.shape[1], masks.shape[2], int(budget), score=0.20)
    elif variant == "Q5":
        quotas = [
            ("mask_boundary", boundary, 0.30, 0.90),
            ("overlap_anchor", overlap, 0.20, 0.82),
            ("disagreement_proxy", disagreement, 0.15, 0.78),
            ("mask_interior", interior, 0.20, 0.60),
            ("exploration", exploration, 0.15, 0.35),
        ]
        used: set[tuple[int, int, int]] = set()
        queries = []
        for idx, (reason, region, fraction, score) in enumerate(quotas):
            remaining = int(budget) - len(queries)
            if remaining <= 0:
                break
            count = remaining if idx == len(quotas) - 1 else int(round(float(budget) * float(fraction)))
            queries.extend(_sample_region(region, min(count, remaining), reason=reason, score=score, used=used))
        if len(queries) < int(budget):
            queries.extend(
                _sample_region(
                    exploration,
                    int(budget) - len(queries),
                    reason="exploration_fill",
                    score=0.10,
                    used=used,
                )
            )
    else:
        raise ValueError(f"unsupported semantic occupancy real variant: {variant}")
    return queries[: int(budget)], _coverage_metrics(
        masks=masks,
        queries=queries[: int(budget)],
        interior=interior,
        boundary=boundary,
        overlap=overlap,
        disagreement=disagreement,
        exploration=exploration,
    )


def _coverage_metrics(
    *,
    masks: np.ndarray,
    queries: list[MaterialQuery],
    interior: np.ndarray,
    boundary: np.ndarray,
    overlap: np.ndarray,
    disagreement: np.ndarray,
    exploration: np.ndarray,
) -> dict[str, Any]:
    selected = np.zeros_like(np.asarray(masks, dtype=np.int32) > 0, dtype=bool)
    for query in queries:
        selected[int(query.frame_rank), int(query.y), int(query.x)] = True

    def coverage(region: np.ndarray) -> float:
        denom = int(np.count_nonzero(region))
        if denom == 0:
            return 0.0
        return float(np.count_nonzero(selected & region) / denom)

    reason_counts: dict[str, int] = {}
    for query in queries:
        reason_counts[query.reason] = reason_counts.get(query.reason, 0) + 1
    accepted = [query for query in queries if float(query.score) >= 0.5]
    return {
        "query_count": int(len(queries)),
        "scheduled_acceptance_prior_count": int(len(accepted)),
        "scheduled_acceptance_prior_ratio": float(len(accepted) / max(len(queries), 1)),
        "part_interior_coverage": coverage(interior),
        "part_boundary_coverage": coverage(boundary),
        "overlap_anchor_coverage": coverage(overlap),
        "disagreement_coverage": coverage(disagreement),
        "exploration_outside_mask_ratio": float(np.count_nonzero(selected & exploration) / max(len(queries), 1)),
        "queries_per_scheduled_acceptance_prior": float(len(queries) / max(len(accepted), 1)),
        "reason_counts": reason_counts,
    }


def _queries_to_source_points(queries: list[MaterialQuery], *, height: int, width: int) -> np.ndarray:
    out = np.zeros((len(queries), 3), dtype=np.float32)
    for idx, query in enumerate(queries):
        out[idx, 0] = float(query.frame_rank)
        out[idx, 1] = float(query.x) / float(max(width - 1, 1))
        out[idx, 2] = float(query.y) / float(max(height - 1, 1))
    return out


def _d4rt_quality(batch: Any, *, min_visibility: float, min_confidence: float) -> dict[str, Any]:
    valid = np.asarray(batch.valid, dtype=bool)
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    visibility = np.asarray(batch.visibility_prob, dtype=np.float32)
    confidence = np.asarray(batch.confidence_prob, dtype=np.float32)
    in_bounds = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    accepted = valid & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
    track_lengths = np.count_nonzero(accepted, axis=0).astype(np.float64) if accepted.ndim == 2 else np.asarray([], dtype=np.float64)
    accepted_tubes = int(np.count_nonzero(track_lengths > 0))
    return {
        "uv_in01_rate": float(np.mean(in_bounds)) if in_bounds.size else 0.0,
        "valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "accepted_tube_count": accepted_tubes,
        "accepted_tube_ratio": float(accepted_tubes / max(valid.shape[1], 1)) if valid.ndim == 2 else 0.0,
        "visible_track_length_mean": float(np.mean(track_lengths[track_lengths > 0])) if np.count_nonzero(track_lengths > 0) else 0.0,
        "visible_track_length_p90": float(np.quantile(track_lengths[track_lengths > 0], 0.90)) if np.count_nonzero(track_lengths > 0) else 0.0,
        "queries_per_accepted_tube": float(valid.shape[1] / max(accepted_tubes, 1)) if valid.ndim == 2 else 0.0,
    }


def _run_variant(
    *,
    args: argparse.Namespace,
    adapter: D4RTAdapter,
    scene: str,
    frame_ids: list[int],
    frames: np.ndarray,
    masks: np.ndarray,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries, schedule_metrics = _schedule_queries(masks, variant=variant, budget=int(args.query_budget))
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    source_points = _queries_to_source_points(queries, height=height, width=width)
    src_frame_global, src_xy, src_mask_id = _source_arrays_from_points(
        source_points=source_points,
        frame_ids=frame_ids,
        image_width=width,
        image_height=height,
        masks=masks,
    )
    carrier_ids = np.asarray(
        [
            stable_source_carrier_id(int(src_frame_global[idx]), int(src_xy[idx, 0]), int(src_xy[idx, 1]), width)
            for idx in range(int(source_points.shape[0]))
        ],
        dtype=np.int64,
    )
    t0 = time.time()
    batch = adapter.infer_carriers(
        video_rgb_uint8=frames,
        src_uv_norm=source_points[:, 1:3],
        src_frame_local=source_points[:, 0].astype(np.int64),
        query_chunk_size=int(args.query_chunk_size),
        carrier_id=carrier_ids,
        src_frame_global=src_frame_global,
        src_xy=src_xy,
        src_mask_id=src_mask_id,
    )
    elapsed = float(time.time() - t0)
    if batch.persistent_tube_id is None:
        batch.persistent_tube_id = np.asarray(batch.carrier_id, dtype=np.int64).copy()
    if batch.parent_tube_id is None:
        batch.parent_tube_id = np.full_like(batch.persistent_tube_id, -1)
    if batch.warmstart_source_chunk is None:
        batch.warmstart_source_chunk = np.full_like(batch.persistent_tube_id, -1)
    if batch.warmstart_source_frame is None:
        batch.warmstart_source_frame = np.full_like(batch.persistent_tube_id, -1)
    if batch.is_warmstarted is None:
        batch.is_warmstarted = np.zeros_like(batch.persistent_tube_id, dtype=bool)

    scene_dir = Path(args.output_cache_root) / variant / scene
    saved = _save_batch(
        batch=batch,
        out_dir=scene_dir,
        frame_ids=frame_ids,
        variant=f"{variant}_semantic_occupancy_real",
        scene=scene,
        window_index=0,
    )
    quality = _d4rt_quality(batch, min_visibility=float(args.min_visibility), min_confidence=float(args.min_confidence))
    summary = {
        "scene": scene,
        "variant": variant,
        "status": "ok",
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "window_size": int(args.window_size),
        "query_budget": int(args.query_budget),
        "d4rt_time_sec": elapsed,
        "output_scene_dir": str(scene_dir),
        **saved,
        **schedule_metrics,
        **quality,
        "downstream_part_purity": None,
        "downstream_object_ARI": None,
        "uses_gt_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
    }
    query_rows = []
    for idx, query in enumerate(queries):
        query_rows.append(
            {
                "scene": scene,
                "variant": variant,
                "query_index": int(idx),
                "frame_rank": int(query.frame_rank),
                "frame_id": int(frame_ids[int(query.frame_rank)]),
                "y": int(query.y),
                "x": int(query.x),
                "u": float(source_points[idx, 1]),
                "v": float(source_points[idx, 2]),
                "reason": query.reason,
                "score": float(query.score),
                "src_mask_id": int(src_mask_id[idx]),
            }
        )
    return query_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v42 real semantic occupancy D4RT query variants.")
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--scene", default="scene0081_01")
    parser.add_argument("--variants", default="Q0,Q5")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--query-budget", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--semantic-query-source-root", default="")
    parser.add_argument("--semantic-query-source", default="")
    parser.add_argument("--semantic-query-sample-dir", default="sample8")
    parser.add_argument("--semantic-query-min-area", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=str(args.scene))
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = _frame_ids(stream, stride=int(args.frame_stride), max_frames=int(args.max_frames))
    frame_ids = frame_ids[: int(args.window_size)]
    data = _load_rgb_sparse_mask_window(stream, frame_ids)
    frames = np.asarray(data["rgb"])
    masks = np.asarray(data["mask"], dtype=np.int32)
    query_mask_diag: dict[str, Any] = {
        "semantic_query_mask_source": "stream_sparse_mask",
        "semantic_query_mask_root": "",
        "semantic_query_sample_dir": "",
        "semantic_query_loaded_frame_ids": data.get("sparse_mask_present_frame_ids", []),
        "semantic_query_missing_frame_ids": data.get("sparse_mask_missing_frame_ids", []),
        "semantic_query_mask_count": int(np.max(masks)) if masks.size else 0,
    }
    if str(args.semantic_query_source_root).strip() and str(args.semantic_query_source).strip():
        masks, query_mask_diag = _load_external_query_masks(
            root=Path(args.semantic_query_source_root),
            scene=str(args.scene),
            source=str(args.semantic_query_source),
            sample_dir=str(args.semantic_query_sample_dir),
            frame_ids=frame_ids,
            image_shape=(int(frames.shape[1]), int(frames.shape[2])),
            min_area=int(args.semantic_query_min_area),
        )
    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=str(args.device),
    )

    all_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in [item.strip() for item in str(args.variants).split(",") if item.strip()]:
        query_rows, summary = _run_variant(
            args=args,
            adapter=adapter,
            scene=str(args.scene),
            frame_ids=frame_ids,
            frames=frames,
            masks=masks,
            variant=variant,
        )
        all_query_rows.extend(query_rows)
        summary_rows.append(summary)

    output_root = Path(args.output_root) / str(args.scene)
    _write_csv(output_root / "query_rows.csv", all_query_rows)
    _write_csv(output_root / "coverage_summary.csv", summary_rows)
    _write_json(
        output_root / "semantic_occupancy_real_manifest.json",
        {
            "scene": str(args.scene),
            "frame_ids": frame_ids,
            "sparse_mask_present_frame_ids": data.get("sparse_mask_present_frame_ids", []),
            "sparse_mask_missing_frame_ids": data.get("sparse_mask_missing_frame_ids", []),
            "query_mask_diagnostics": query_mask_diag,
            "rows": summary_rows,
            "note": "Real D4RT query smoke. downstream_object_ARI is not computed by this runner.",
        },
    )
    print(json.dumps(_json_safe({"output_root": str(output_root), "rows": summary_rows}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
