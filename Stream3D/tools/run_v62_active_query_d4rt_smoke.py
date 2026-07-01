from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.d4rt_adapter import D4RTAdapter


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


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
            writer.writerow(
                {
                    key: json.dumps(_json_safe(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in keys
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool_str(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _parse_observation_id(token: str) -> tuple[str, int, int]:
    parts = str(token).split(":")
    if len(parts) != 4 or parts[0] != "m":
        raise ValueError(f"unsupported support observation id: {token}")
    return parts[1], int(parts[2]), int(parts[3])


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read RGB frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask frame: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image)


def _mask_centroid(mask: np.ndarray, mask_id: int) -> tuple[int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.int64) == int(mask_id))
    if ys.size == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean()))), int(ys.size)


def _frame_ids(scene_root: Path, start: int, size: int) -> list[int]:
    out: list[int] = []
    for frame_id in range(int(start), int(start) + int(size)):
        if (scene_root / "color" / f"{frame_id}.jpg").exists():
            out.append(int(frame_id))
    return out


def _candidate_source_priority(source: str) -> int:
    order = {
        "bridge_low_support": 0,
        "update_new_low_support": 1,
        "shared_shortcut_boundary": 2,
        "state_tentative": 3,
    }
    return order.get(str(source), 99)


def _select_queries(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[int]]:
    candidate_rows = _read_csv(Path(args.candidate_csv))
    novelty_rows = _read_csv(Path(args.novelty_csv))
    novelty_by_material = {row["material_node_id"]: row for row in novelty_rows}
    scene_root = Path(args.scannet_root) / str(args.scene)
    frame_ids = _frame_ids(scene_root, int(args.window_start), int(args.window_size))
    frame_id_set = set(frame_ids)
    allowed_sources = {source for source in str(args.candidate_sources).split(",") if source}
    sorted_candidates = sorted(
        [row for row in candidate_rows if row.get("scene") == str(args.scene)],
        key=lambda row: (_candidate_source_priority(row.get("candidate_source", "")), row.get("query_candidate_id", "")),
    )

    selected: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    used_observations: set[str] = set()

    def skip(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    for candidate in sorted_candidates:
        if len(selected) >= int(args.query_budget):
            break
        source = candidate.get("candidate_source", "")
        if allowed_sources and source not in allowed_sources:
            skip("candidate_source_not_allowed")
            continue
        if _bool_str(candidate.get("has_existing_query_outcome", "")):
            skip("already_has_query_outcome")
            continue
        novelty = novelty_by_material.get(candidate.get("material_node_id", ""))
        if novelty is None:
            skip("missing_novelty_row")
            continue
        try:
            observations = json.loads(novelty.get("support_observation_ids_json", "[]") or "[]")
        except json.JSONDecodeError:
            skip("invalid_support_observation_json")
            continue
        if not observations:
            skip("empty_support_observations")
            continue

        chosen: dict[str, Any] | None = None
        for token in observations:
            if str(token) in used_observations:
                continue
            try:
                obs_scene, frame_id, mask_id = _parse_observation_id(str(token))
            except ValueError:
                skip("invalid_support_observation_id")
                continue
            if obs_scene != str(args.scene):
                skip("support_scene_mismatch")
                continue
            if frame_id not in frame_id_set:
                skip("support_frame_outside_window")
                continue
            mask_path = scene_root / "output_Cropformer" / "mask" / f"{frame_id}.png"
            if not mask_path.exists():
                skip("missing_mask_frame")
                continue
            centroid = _mask_centroid(_load_mask(mask_path), mask_id)
            if centroid is None:
                skip("empty_mask_id")
                continue
            x, y, area = centroid
            used_observations.add(str(token))
            chosen = {
                **candidate,
                "support_observation_id": str(token),
                "support_frame_id": int(frame_id),
                "support_mask_id": int(mask_id),
                "source_x": int(x),
                "source_y": int(y),
                "source_mask_area_px": int(area),
                "novelty_state": novelty.get("state", ""),
                "novelty_type_from_increment": novelty.get("novelty_type", ""),
                "has_K_mat": _bool_str(novelty.get("has_K_mat", "")),
                "has_K_mask": _bool_str(novelty.get("has_K_mask", "")),
                "has_K_sem": _bool_str(novelty.get("has_K_sem", "")),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
            break
        if chosen is None:
            skip("no_usable_support_observation")
            continue
        selected.append(chosen)

    diagnostics = {
        "candidate_row_count_for_scene": int(sum(1 for row in candidate_rows if row.get("scene") == str(args.scene))),
        "selected_query_count": int(len(selected)),
        "requested_query_budget": int(args.query_budget),
        "candidate_sources_allowed": sorted(allowed_sources),
        "skip_counts": skip_counts,
    }
    return selected, diagnostics, frame_ids


def _load_window(scene_root: Path, frame_ids: list[int]) -> np.ndarray:
    if not frame_ids:
        raise ValueError("no RGB frames selected for D4RT window")
    return np.stack([_load_rgb(scene_root / "color" / f"{frame_id}.jpg") for frame_id in frame_ids], axis=0)


def _run_d4rt(args: argparse.Namespace, selected: list[dict[str, Any]], frame_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scene_root = Path(args.scannet_root) / str(args.scene)
    frames = _load_window(scene_root, frame_ids)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    frame_to_local = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    src_uv = np.asarray(
        [[float(row["source_x"]) / max(width - 1, 1), float(row["source_y"]) / max(height - 1, 1)] for row in selected],
        dtype=np.float32,
    )
    src_frame_local = np.asarray([frame_to_local[int(row["support_frame_id"])] for row in selected], dtype=np.int64)
    src_xy = np.asarray([[int(row["source_x"]), int(row["source_y"])] for row in selected], dtype=np.int32)
    src_mask_id = np.asarray([int(row["support_mask_id"]) for row in selected], dtype=np.int32)
    carrier_id = np.arange(len(selected), dtype=np.int64)
    src_frame_global = np.asarray([int(row["support_frame_id"]) for row in selected], dtype=np.int64)

    adapter = D4RTAdapter(
        d4rt_root=Path(args.d4rt_root),
        model_config=Path(args.d4rt_config),
        ckpt_path=Path(args.d4rt_ckpt),
        device=str(args.device),
    )
    start = time.time()
    batch = adapter.infer_carriers(
        video_rgb_uint8=frames,
        src_uv_norm=src_uv,
        src_frame_local=src_frame_local,
        query_chunk_size=int(args.query_chunk_size),
        carrier_id=carrier_id,
        src_frame_global=src_frame_global,
        src_xy=src_xy,
        src_mask_id=src_mask_id,
    )
    elapsed = float(time.time() - start)

    valid = np.asarray(batch.valid, dtype=bool)
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    visibility = np.asarray(batch.visibility_prob, dtype=np.float32)
    confidence = np.asarray(batch.confidence_prob, dtype=np.float32)
    accepted = valid & (visibility >= float(args.min_visibility)) & (confidence >= float(args.min_confidence))
    in_bounds = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    accepted_lengths = np.count_nonzero(accepted, axis=0) if accepted.ndim == 2 else np.asarray([], dtype=np.int64)

    query_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        accepted_count = int(accepted_lengths[idx]) if idx < accepted_lengths.shape[0] else 0
        valid_count = int(np.count_nonzero(valid[:, idx])) if valid.ndim == 2 else 0
        query_rows.append(
            {
                **row,
                "d4rt_query_index": int(idx),
                "source_u": float(src_uv[idx, 0]),
                "source_v": float(src_uv[idx, 1]),
                "support_frame_local": int(src_frame_local[idx]),
                "d4rt_valid_track_count": int(valid_count),
                "d4rt_accepted_track_count": int(accepted_count),
                "d4rt_has_valid_new_evidence": bool(accepted_count >= int(args.min_accepted_frames)),
                "confirm_or_quarantine_outcome": "not_computed_in_smoke",
            }
        )

    summary = {
        "status": "ok",
        "scene": str(args.scene),
        "query_budget": int(args.query_budget),
        "selected_query_count": int(len(selected)),
        "window_start": int(args.window_start),
        "window_size_requested": int(args.window_size),
        "window_frame_count": int(len(frame_ids)),
        "frame_ids": [int(frame_id) for frame_id in frame_ids],
        "image_shape_hw": [height, width],
        "d4rt_root": str(Path(args.d4rt_root)),
        "d4rt_config": str(Path(args.d4rt_config)),
        "d4rt_ckpt": str(Path(args.d4rt_ckpt)),
        "d4rt_time_sec": elapsed,
        "adapter_diagnostics": dict(adapter.last_infer_diagnostics),
        "valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "uv_in01_rate": float(np.mean(in_bounds)) if in_bounds.size else 0.0,
        "accepted_tube_count": int(np.count_nonzero(accepted_lengths >= int(args.min_accepted_frames))),
        "accepted_tube_ratio": float(np.mean(accepted_lengths >= int(args.min_accepted_frames))) if accepted_lengths.size else 0.0,
        "accepted_track_length_mean": float(np.mean(accepted_lengths)) if accepted_lengths.size else 0.0,
        "accepted_track_length_p90": float(np.quantile(accepted_lengths, 0.90)) if accepted_lengths.size else 0.0,
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "min_accepted_frames": int(args.min_accepted_frames),
        "active_query_confirm_or_quarantine_status": "not_computed_in_smoke",
        "ap_status": "not_run",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }

    npz_path = Path(args.output_root) / "carrier_batch_smoke.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        carrier_id=np.asarray(batch.carrier_id),
        src_frame=np.asarray(batch.src_frame),
        src_uv=np.asarray(batch.src_uv),
        src_xy=src_xy,
        src_mask_id=src_mask_id,
        uv_pred=uv,
        visibility_prob=visibility,
        confidence_prob=confidence,
        valid=valid,
    )
    summary["carrier_batch_npz"] = str(npz_path)
    return summary, query_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a v62 active-query D4RT smoke from de-circularized candidate rows.")
    parser.add_argument("--candidate-csv", default="Stream3D/outputs/audit/v62_active_query_refresh/query_candidate_rows.csv")
    parser.add_argument("--novelty-csv", default="Stream3D/outputs/audit/v62_increment_attribution/novelty_material_rows.csv")
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--scene", default="scene0011_00")
    parser.add_argument("--candidate-sources", default="bridge_low_support,update_new_low_support,shared_shortcut_boundary,state_tentative")
    parser.add_argument("--query-budget", type=int, default=16)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-accepted-frames", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    selected, selection_diagnostics, frame_ids = _select_queries(args)
    if len(selected) < int(args.query_budget):
        raise RuntimeError(
            f"only selected {len(selected)} usable queries from budget={args.query_budget}; "
            f"diagnostics={selection_diagnostics}"
        )
    scene_root = Path(args.scannet_root) / str(args.scene)
    preflight = {
        "status": "dry_run" if args.dry_run else "selected",
        "scene": str(args.scene),
        "scene_root": str(scene_root),
        "frame_ids": [int(frame_id) for frame_id in frame_ids],
        **selection_diagnostics,
        "selected_preview": selected[: min(5, len(selected))],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    output_root = Path(args.output_root)
    _write_json(output_root / "v62_d4rt_smoke_preflight.json", preflight)
    _write_csv(output_root / "v62_d4rt_smoke_selected_queries.csv", selected)
    if args.dry_run:
        print(json.dumps(_json_safe(preflight), indent=2, sort_keys=True))
        return

    summary, query_rows = _run_d4rt(args, selected, frame_ids)
    summary = {**summary, "selection_diagnostics": selection_diagnostics}
    _write_json(output_root / "v62_d4rt_smoke_summary.json", summary)
    _write_csv(output_root / "v62_d4rt_smoke_query_rows.csv", query_rows)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
