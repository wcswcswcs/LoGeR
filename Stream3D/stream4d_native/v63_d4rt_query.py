from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .v47_common import ROOT, parse_float, parse_int, read_csv, utc_now, write_csv, write_json


DEFAULT_SELECTED_QUERIES = "outputs/audit/v63_query_policy/selected_query_rows.csv"
DEFAULT_NOVELTY_MATERIAL = "outputs/audit/v62_increment_attribution/novelty_material_rows.csv"


@dataclass(frozen=True)
class V63D4RTQueryConfig:
    selected_query_rows: str | Path = DEFAULT_SELECTED_QUERIES
    novelty_material_rows: str | Path = DEFAULT_NOVELTY_MATERIAL
    output_root: str | Path = "outputs/audit/v63_d4rt_query"
    scannet_root: str | Path = "data/scannet/processed"
    d4rt_root: str | Path = "../Open-d4rt"
    d4rt_config: str | Path = "../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
    d4rt_ckpt: str | Path = "../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
    device: str = "cuda"
    policy_ids: tuple[str, ...] = ("R0_real_policy",)
    max_queries_per_policy: int | None = 16
    window_size: int = 32
    query_chunk_size: int = 128
    min_visibility: float = 0.5
    min_confidence: float = 0.5
    min_accepted_frames: int = 2
    dry_run: bool = True


def build_v63_d4rt_query(config: V63D4RTQueryConfig | None = None) -> dict[str, Any]:
    cfg = config or V63D4RTQueryConfig()
    selected_rows = read_csv(_project(cfg.selected_query_rows))
    novelty_rows = read_csv(_project(cfg.novelty_material_rows))
    novelty_by_material = {row.get("material_node_id", ""): row for row in novelty_rows}
    filtered = _filter_policy_rows(selected_rows, cfg.policy_ids, cfg.max_queries_per_policy)
    preflight_rows, skip_rows = _preflight_rows(filtered, novelty_by_material, cfg)
    query_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    if cfg.dry_run:
        query_rows = [
            {
                **row,
                "d4rt_status": "dry_run_preflight_only",
                "valid_track_count": "",
                "accepted_track_count": "",
                "accepted_track_ratio": "",
                "carrier_batch_npz": "",
            }
            for row in preflight_rows
        ]
    else:
        query_rows, group_summaries = _run_real_d4rt(preflight_rows, cfg)

    control_rows = _control_rows(query_rows, cfg)
    policy_counts = Counter(row.get("policy_id", "") for row in query_rows)
    preflight_counts = Counter(row.get("preflight_status", "") for row in preflight_rows)
    skipped_counts = Counter(row.get("skip_reason", "") for row in skip_rows)
    real_executed = not cfg.dry_run
    no_group_errors = all(row.get("status") == "ok" for row in group_summaries) if real_executed else False
    summary = {
        "phase": "v63_d4rt_query",
        "created_at": utc_now(),
        "method_status": "dry_run_preflight_only" if cfg.dry_run else "real_D4RT_query_executed",
        "selected_query_rows": _rel(cfg.selected_query_rows),
        "novelty_material_rows": _rel(cfg.novelty_material_rows),
        "policy_ids": list(cfg.policy_ids),
        "max_queries_per_policy": cfg.max_queries_per_policy,
        "window_size": int(cfg.window_size),
        "query_chunk_size": int(cfg.query_chunk_size),
        "preflight_query_count": len(preflight_rows),
        "query_result_count": len(query_rows),
        "skip_count": len(skip_rows),
        "preflight_status_counts": dict(preflight_counts),
        "skip_reason_counts": dict(skipped_counts),
        "policy_query_counts": dict(policy_counts),
        "group_count": len(group_summaries),
        "group_summaries": group_summaries,
        "dry_run": bool(cfg.dry_run),
        "d4rt_root": str(_project(cfg.d4rt_root)),
        "d4rt_config": str(_project(cfg.d4rt_config)),
        "d4rt_ckpt": str(_project(cfg.d4rt_ckpt)),
        "device": cfg.device,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "gate": {
            "preflight_has_usable_queries": len(preflight_rows) > 0,
            "real_D4RT_executed": real_executed,
            "all_real_D4RT_groups_ok": no_group_errors,
            "pass": bool(real_executed and no_group_errors and len(preflight_rows) > 0),
        },
    }
    return {
        "summary": summary,
        "query_result_rows": query_rows,
        "query_control_rows": control_rows,
        "preflight_skip_rows": skip_rows,
    }


def write_v63_d4rt_query(result: dict[str, Any], config: V63D4RTQueryConfig | None = None) -> dict[str, str]:
    cfg = config or V63D4RTQueryConfig()
    output_root = _project(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "query_execution_summary": output_root / "query_execution_summary.json",
        "query_result_rows": output_root / "query_result_rows.csv",
        "query_control_rows": output_root / "query_control_rows.csv",
        "preflight_skip_rows": output_root / "preflight_skip_rows.csv",
    }
    write_json(paths["query_execution_summary"], result["summary"])
    write_csv(paths["query_result_rows"], result["query_result_rows"])
    write_csv(paths["query_control_rows"], result["query_control_rows"])
    write_csv(paths["preflight_skip_rows"], result["preflight_skip_rows"])
    return {key: _rel(path) for key, path in paths.items()}


def _filter_policy_rows(rows: list[dict[str, str]], policy_ids: tuple[str, ...], max_queries_per_policy: int | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for policy_id in policy_ids:
        policy_rows = [row for row in rows if row.get("policy_id") == policy_id]
        limit = len(policy_rows) if max_queries_per_policy is None else int(max_queries_per_policy)
        out.extend(policy_rows[:limit])
    return out


def _preflight_rows(
    rows: list[dict[str, str]],
    novelty_by_material: dict[str, dict[str, str]],
    cfg: V63D4RTQueryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scannet_root = _project(cfg.scannet_root)
    preflight: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        novelty = novelty_by_material.get(row.get("material_node_id", ""), {})
        obs_tokens = _observation_tokens(row, novelty)
        if not obs_tokens:
            skipped.append(_skip(row, "missing_support_observation"))
            continue
        chosen: dict[str, Any] | None = None
        skip_reasons: Counter[str] = Counter()
        for token in obs_tokens:
            parsed = _parse_observation_id(token)
            if parsed is None:
                skip_reasons["invalid_support_observation_id"] += 1
                continue
            obs_scene, frame_id, mask_id = parsed
            if obs_scene != row.get("scene", ""):
                skip_reasons["support_scene_mismatch"] += 1
                continue
            scene_root = scannet_root / obs_scene
            mask_path = scene_root / "output_Cropformer" / "mask" / f"{frame_id}.png"
            if not mask_path.exists():
                skip_reasons["missing_mask_frame"] += 1
                continue
            centroid = _mask_centroid(mask_path, mask_id)
            if centroid is None:
                skip_reasons["empty_mask_id"] += 1
                continue
            window_start = int(frame_id // int(cfg.window_size) * int(cfg.window_size))
            frame_ids = _frame_ids(scene_root, window_start, int(cfg.window_size))
            if frame_id not in frame_ids:
                skip_reasons["source_frame_not_in_window"] += 1
                continue
            x, y, area = centroid
            chosen = {
                **row,
                "support_observation_id": token,
                "support_frame_id": frame_id,
                "support_mask_id": mask_id,
                "source_x": x,
                "source_y": y,
                "source_mask_area_px": area,
                "window_start": window_start,
                "window_frame_count": len(frame_ids),
                "frame_ids_json": json.dumps(frame_ids),
                "preflight_status": "usable",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
            break
        if chosen is None:
            skipped.append(_skip(row, ";".join(f"{key}:{value}" for key, value in sorted(skip_reasons.items())) or "no_usable_support"))
            continue
        preflight.append(chosen)
    return preflight, skipped


def _run_real_d4rt(preflight_rows: list[dict[str, Any]], cfg: V63D4RTQueryConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from stream4d.d4rt_adapter import D4RTAdapter

    adapter = D4RTAdapter(
        d4rt_root=_project(cfg.d4rt_root),
        model_config=_project(cfg.d4rt_config),
        ckpt_path=_project(cfg.d4rt_ckpt),
        device=cfg.device,
    )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in preflight_rows:
        grouped[(row.get("policy_id", ""), row.get("scene", ""), int(row.get("window_start", 0)))].append(row)
    query_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group_idx, ((policy_id, scene, window_start), rows) in enumerate(sorted(grouped.items())):
        try:
            summary, result_rows = _run_group(adapter, rows, cfg, group_idx, policy_id, scene, window_start)
        except Exception as exc:  # pragma: no cover - runtime path depends on D4RT/GPU
            summary = {
                "group_id": _group_id(group_idx, policy_id, scene, window_start),
                "policy_id": policy_id,
                "scene": scene,
                "window_start": window_start,
                "query_count": len(rows),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            result_rows = [{**row, "d4rt_status": "group_error", "error": str(exc)} for row in rows]
        summaries.append(summary)
        query_rows.extend(result_rows)
    return query_rows, summaries


def _run_group(
    adapter: Any,
    rows: list[dict[str, Any]],
    cfg: V63D4RTQueryConfig,
    group_idx: int,
    policy_id: str,
    scene: str,
    window_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scannet_root = _project(cfg.scannet_root)
    scene_root = scannet_root / scene
    frame_ids = _frame_ids(scene_root, window_start, int(cfg.window_size))
    frames = _load_window(scene_root, frame_ids)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    frame_to_local = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    src_uv = np.asarray(
        [[float(row["source_x"]) / max(width - 1, 1), float(row["source_y"]) / max(height - 1, 1)] for row in rows],
        dtype=np.float32,
    )
    src_frame_local = np.asarray([frame_to_local[int(row["support_frame_id"])] for row in rows], dtype=np.int64)
    src_xy = np.asarray([[int(row["source_x"]), int(row["source_y"])] for row in rows], dtype=np.int32)
    src_mask_id = np.asarray([int(row["support_mask_id"]) for row in rows], dtype=np.int32)
    carrier_id = np.arange(len(rows), dtype=np.int64)
    src_frame_global = np.asarray([int(row["support_frame_id"]) for row in rows], dtype=np.int64)
    start = time.time()
    batch = adapter.infer_carriers(
        video_rgb_uint8=frames,
        src_uv_norm=src_uv,
        src_frame_local=src_frame_local,
        query_chunk_size=int(cfg.query_chunk_size),
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
    accepted = valid & (visibility >= float(cfg.min_visibility)) & (confidence >= float(cfg.min_confidence))
    accepted_lengths = np.count_nonzero(accepted, axis=0) if accepted.ndim == 2 else np.asarray([], dtype=np.int64)
    in_bounds = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    output_root = _project(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    carrier_path = output_root / f"carrier_batch_{_group_id(group_idx, policy_id, scene, window_start)}.npz"
    np.savez_compressed(
        carrier_path,
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
    result_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        accepted_count = int(accepted_lengths[idx]) if idx < accepted_lengths.shape[0] else 0
        valid_count = int(np.count_nonzero(valid[:, idx])) if valid.ndim == 2 else 0
        result_rows.append(
            {
                **row,
                "d4rt_status": "ok",
                "d4rt_query_index": idx,
                "valid_track_count": valid_count,
                "accepted_track_count": accepted_count,
                "accepted_track_ratio": accepted_count / float(max(len(frame_ids), 1)),
                "in_bounds_track_count": int(np.count_nonzero(in_bounds[:, idx])) if in_bounds.ndim == 2 else 0,
                "carrier_batch_npz": _rel(carrier_path),
            }
        )
    summary = {
        "group_id": _group_id(group_idx, policy_id, scene, window_start),
        "policy_id": policy_id,
        "scene": scene,
        "window_start": int(window_start),
        "query_count": len(rows),
        "frame_ids": frame_ids,
        "image_shape_hw": [height, width],
        "status": "ok",
        "d4rt_time_sec": elapsed,
        "adapter_diagnostics": dict(adapter.last_infer_diagnostics),
        "carrier_batch_npz": _rel(carrier_path),
        "valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "uv_in01_rate": float(np.mean(in_bounds)) if in_bounds.size else 0.0,
        "accepted_tube_count": int(np.count_nonzero(accepted_lengths >= int(cfg.min_accepted_frames))),
        "accepted_tube_ratio": float(np.mean(accepted_lengths >= int(cfg.min_accepted_frames))) if accepted_lengths.size else 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    return summary, result_rows


def _control_rows(query_rows: list[dict[str, Any]], cfg: V63D4RTQueryConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_policy[row.get("policy_id", "")].append(row)
    for policy_id, policy_rows in sorted(by_policy.items()):
        accepted = [parse_float(row.get("accepted_track_ratio")) for row in policy_rows if row.get("accepted_track_ratio") not in {"", None}]
        rows.append(
            {
                "policy_id": policy_id,
                "query_count": len(policy_rows),
                "d4rt_status_counts": dict(Counter(row.get("d4rt_status", "") for row in policy_rows)),
                "mean_accepted_track_ratio": float(np.mean(accepted)) if accepted else None,
                "dry_run": bool(cfg.dry_run),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return rows


def _observation_tokens(row: dict[str, str], novelty: dict[str, str]) -> list[str]:
    for raw in [row.get("support_observation_ids_json"), novelty.get("support_observation_ids_json")]:
        try:
            tokens = json.loads(raw or "[]")
        except json.JSONDecodeError:
            tokens = []
        if tokens:
            return [str(token) for token in tokens]
    return []


def _parse_observation_id(token: str) -> tuple[str, int, int] | None:
    parts = str(token).split(":")
    if len(parts) != 4 or parts[0] != "m":
        return None
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _mask_centroid(path: Path, mask_id: int) -> tuple[int, int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    ys, xs = np.nonzero(np.asarray(image) == int(mask_id))
    if ys.size == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean()))), int(ys.size)


def _frame_ids(scene_root: Path, start: int, size: int) -> list[int]:
    return [frame_id for frame_id in range(int(start), int(start) + int(size)) if (scene_root / "color" / f"{frame_id}.jpg").exists()]


def _load_window(scene_root: Path, frame_ids: list[int]) -> np.ndarray:
    frames = []
    for frame_id in frame_ids:
        image = cv2.imread(str(scene_root / "color" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to load RGB frame {scene_root / 'color' / f'{frame_id}.jpg'}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not frames:
        raise ValueError(f"no frames available under {scene_root}")
    return np.stack(frames, axis=0)


def _skip(row: dict[str, str], reason: str) -> dict[str, Any]:
    return {
        "policy_id": row.get("policy_id", ""),
        "v63_candidate_id": row.get("v63_candidate_id", ""),
        "material_node_id": row.get("material_node_id", ""),
        "scene": row.get("scene", ""),
        "candidate_type": row.get("candidate_type", ""),
        "planned_action": row.get("planned_action", ""),
        "skip_reason": reason,
        "uses_gt_for_prediction": False,
    }


def _group_id(group_idx: int, policy_id: str, scene: str, window_start: int) -> str:
    safe_policy = "".join(ch if ch.isalnum() else "_" for ch in policy_id)
    safe_scene = "".join(ch if ch.isalnum() else "_" for ch in scene)
    return f"{group_idx:03d}_{safe_policy}_{safe_scene}_w{int(window_start):04d}"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
