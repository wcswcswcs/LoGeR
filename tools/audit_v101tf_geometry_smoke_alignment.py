#!/usr/bin/env python3
"""Audit v101 lifecycle-aligned geometry sidecar smoke runs.

The smoke root is produced by run_v101tf_stage_c_seed_bridge_target_traces.py
with --enable-per-chunk-geometry-sidecar 1.  This audit verifies that lifecycle
anchor ids in the trace payload can be joined to geometry-sidecar top-k hit
anchors from the same no-action payload.  It is diagnostic-only and does not
authorize runtime action.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
TARGET_ROWS = ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"
DEFAULT_SMOKE_ROOT = ROOT / "stage_c_seed_bridge_geometry_smoke_clean6"
DEFAULT_PREFIX = "stage_c_seed_geometry_smoke_clean6"

POS_TAX = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_TAX = "SAFE_GOOD"
TOKEN_TYPE_PATCH = 2


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def norm_id(value: Any) -> str:
    number = f(value)
    if math.isfinite(number):
        return str(int(number))
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def mean(values: list[Any]) -> float:
    finite = [f(value) for value in values if math.isfinite(f(value))]
    return sum(finite) / len(finite) if finite else math.nan


def std(values: list[Any]) -> float:
    finite = [f(value) for value in values if math.isfinite(f(value))]
    if not finite:
        return math.nan
    mu = sum(finite) / len(finite)
    return math.sqrt(sum((value - mu) ** 2 for value in finite) / len(finite))


def entropy_from_values(values: list[Any], *, bins: int = 5) -> tuple[int, float]:
    finite = [f(value) for value in values if math.isfinite(f(value))]
    if not finite:
        return 0, math.nan
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        return 1, 0.0
    counts = [0 for _ in range(bins)]
    for value in finite:
        idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
        counts[idx] += 1
    total = sum(counts)
    probs = [count / total for count in counts if count > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    active = sum(1 for prob in probs if prob >= 0.10)
    return active, entropy


def svd_spread_ratio(rows: list[dict[str, Any]], prefix: str) -> float:
    points: list[list[float]] = []
    for row in rows:
        point = [f(row.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")]
        if all(math.isfinite(value) for value in point):
            points.append(point)
    if len(points) < 3:
        return math.nan
    tensor = torch.tensor(points, dtype=torch.float32)
    tensor = tensor - tensor.mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(tensor)
    except Exception:
        return math.nan
    if int(singular.numel()) == 0:
        return math.nan
    largest = float(singular.max().item())
    smallest = float(singular.min().item())
    return smallest / largest if largest > 1.0e-8 else math.nan


def frac(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else math.nan


def parse_case(case_id: str) -> tuple[str, int | None, int | None]:
    parts = str(case_id).split("_")
    if len(parts) != 3:
        return parts[0] if parts else "", None, None
    try:
        return f"{int(parts[0]):02d}", int(parts[1]), int(parts[2])
    except ValueError:
        return parts[0], None, None


def case_from_payload_path(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if parts else ""


def target_by_case() -> dict[str, dict[str, str]]:
    return {row.get("case_id", ""): row for row in read_rows(TARGET_ROWS)}


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def token_patch_coords(token_idx: int, tokens_per_frame: int, patch_h: int, patch_w: int) -> tuple[int, int, int] | None:
    if tokens_per_frame <= 0:
        return None
    frame = int(token_idx) // int(tokens_per_frame)
    local = int(token_idx) % int(tokens_per_frame)
    if local >= patch_h * patch_w:
        return None
    return frame, local // patch_w, local % patch_w


def point_at_patch(points: Any, frame_idx: int, patch_row: int, patch_col: int, patch_h: int, patch_w: int) -> torch.Tensor | None:
    if not torch.is_tensor(points) or points.ndim < 4:
        return None
    if frame_idx < 0 or frame_idx >= int(points.shape[0]):
        return None
    height = int(points.shape[1])
    width = int(points.shape[2])
    y = min(height - 1, max(0, int((float(patch_row) + 0.5) * float(height) / max(float(patch_h), 1.0))))
    x = min(width - 1, max(0, int((float(patch_col) + 0.5) * float(width) / max(float(patch_w), 1.0))))
    return points[int(frame_idx), y, x].detach().float()


def scalar_at_patch(values: Any, frame_idx: int, patch_row: int, patch_col: int, patch_h: int, patch_w: int) -> float:
    if not torch.is_tensor(values) or values.ndim < 3:
        return math.nan
    if frame_idx < 0 or frame_idx >= int(values.shape[0]):
        return math.nan
    height = int(values.shape[1])
    width = int(values.shape[2])
    y = min(height - 1, max(0, int((float(patch_row) + 0.5) * float(height) / max(float(patch_h), 1.0))))
    x = min(width - 1, max(0, int((float(patch_col) + 0.5) * float(width) / max(float(patch_w), 1.0))))
    return f(values[int(frame_idx), y, x].detach().float().item())


def sidecar_paths_for_trace(payload_path: Path, case_id: str) -> tuple[Path | None, Path | None, str]:
    run_dir = payload_path.parent.parent
    sidecar_dir = run_dir / "per_chunk_geometry"
    if not sidecar_dir.is_dir():
        return None, None, "missing_per_chunk_geometry_dir"
    _, prev_chunk, curr_chunk = parse_case(case_id)
    delta = int(curr_chunk - prev_chunk) if prev_chunk is not None and curr_chunk is not None else 1
    prev_path = sidecar_dir / "chunk_000.pt"
    curr_path = sidecar_dir / f"chunk_{max(delta, 1):03d}.pt"
    if prev_path.is_file() and curr_path.is_file():
        return prev_path, curr_path, "matched_case_delta"
    sidecars = sorted(sidecar_dir.glob("chunk_*.pt"))
    if len(sidecars) >= 2:
        return sidecars[0], sidecars[-1], "fallback_sorted_sidecars"
    return None, None, "missing_expected_prev_current_sidecars"


def infer_patch_grid(curr_geo: dict[str, Any], payload: dict[str, Any]) -> tuple[int, int]:
    curr_local = curr_geo.get("local_points")
    patch_h = int(curr_local.shape[1]) // 14 if torch.is_tensor(curr_local) and int(curr_local.shape[1]) % 14 == 0 else 19
    patch_w = int(curr_local.shape[2]) // 14 if torch.is_tensor(curr_local) and int(curr_local.shape[2]) % 14 == 0 else 66
    token_type = curr_geo.get("token_type")
    tokens_per_frame = int(payload.get("tokens_per_frame", 0) or 0)
    if torch.is_tensor(token_type) and tokens_per_frame > 0:
        per_frame = token_type[:tokens_per_frame]
        patch_count = int((per_frame == TOKEN_TYPE_PATCH).sum().item())
        if patch_count == 19 * 66:
            patch_h, patch_w = 19, 66
    return patch_h, patch_w


def lifecycle_anchor_ids(payload: dict[str, Any], case_id: str) -> tuple[set[tuple[str, str]], int, int]:
    rows = payload.get("ttt_prev_stable_anchor_lifecycle_rows") or []
    anchors: set[tuple[str, str]] = set()
    seed_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        anchor_id = norm_id(row.get("anchor_id"))
        if anchor_id:
            anchors.add((case_id, anchor_id))
        if row.get("source_stage_c_seed_global_track_idx_mode") not in (None, ""):
            seed_rows += 1
    return anchors, len(rows), seed_rows


def collect_edge_rows(root: Path, target: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    edge_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []

    for payload_path in sorted(root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")):
        case_id = case_from_payload_path(root, payload_path)
        target_row = target.get(case_id, {})
        try:
            payload = torch_load(payload_path)
        except Exception as exc:  # noqa: BLE001
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(payload, dict):
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": "payload_not_dict"})
            continue
        lifecycle_ids, lifecycle_row_count, lifecycle_seed_rows = lifecycle_anchor_ids(payload, case_id)
        prev_path, curr_path, sidecar_match = sidecar_paths_for_trace(payload_path, case_id)
        if prev_path is None or curr_path is None:
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": sidecar_match})
            continue
        try:
            prev_geo = torch_load(prev_path)
            curr_geo = torch_load(curr_path)
        except Exception as exc:  # noqa: BLE001
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": f"sidecar:{type(exc).__name__}:{exc}"})
            continue
        if not isinstance(prev_geo, dict) or not isinstance(curr_geo, dict):
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": "sidecar_not_dict"})
            continue
        hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids")
        cache_indices = payload.get("current_Q_to_cache_K_topk_cache_indices")
        sampled_query_indices = payload.get("sampled_query_indices")
        if not (
            torch.is_tensor(hit)
            and torch.is_tensor(anchor_ids)
            and torch.is_tensor(cache_indices)
            and torch.is_tensor(sampled_query_indices)
        ):
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": "missing_hit_anchor_cache_or_query_indices"})
            continue
        hit = hit.detach().cpu().bool()
        anchor_ids = anchor_ids.detach().cpu().long()
        cache_indices = cache_indices.detach().cpu().long()
        sampled_query_indices = sampled_query_indices.detach().cpu().long()
        prev_local = prev_geo.get("local_points")
        curr_local = curr_geo.get("local_points")
        prev_world = prev_geo.get("points")
        curr_world = curr_geo.get("points")
        prev_conf = prev_geo.get("conf")
        curr_conf = curr_geo.get("conf")
        prev_pose = prev_geo.get("camera_poses")
        curr_pose = curr_geo.get("camera_poses")
        if not (torch.is_tensor(prev_local) and torch.is_tensor(curr_local) and torch.is_tensor(prev_pose) and torch.is_tensor(curr_pose)):
            read_errors.append({"case_id": case_id, "path": str(payload_path), "error": "missing_geometry_tensors_in_sidecar"})
            continue
        patch_h, patch_w = infer_patch_grid(curr_geo, payload)
        tokens_per_frame = int(payload.get("tokens_per_frame", 0) or 0)
        batch, head_count, query_count, topk_count = [int(v) for v in hit.shape]
        geometry_ids: set[tuple[str, str]] = set()
        nonnegative_count = 0
        for batch_idx in range(batch):
            for head_idx in range(head_count):
                for query_pos in range(query_count):
                    q_token = int(sampled_query_indices[min(query_pos, int(sampled_query_indices.numel()) - 1)].item())
                    q_coords = token_patch_coords(q_token, tokens_per_frame, patch_h, patch_w)
                    if q_coords is None:
                        continue
                    q_frame, q_pr, q_pc = q_coords
                    q_local = point_at_patch(curr_local, q_frame, q_pr, q_pc, patch_h, patch_w)
                    q_world = point_at_patch(curr_world, q_frame, q_pr, q_pc, patch_h, patch_w) if torch.is_tensor(curr_world) else None
                    if q_local is None:
                        continue
                    for topk_idx in range(topk_count):
                        if not bool(hit[batch_idx, head_idx, query_pos, topk_idx].item()):
                            continue
                        anchor_id = int(anchor_ids[batch_idx, head_idx, query_pos, topk_idx].item())
                        if anchor_id < 0:
                            continue
                        nonnegative_count += 1
                        anchor_text = str(anchor_id)
                        geometry_ids.add((case_id, anchor_text))
                        cache_token = int(cache_indices[batch_idx, head_idx, query_pos, topk_idx].item())
                        c_coords = token_patch_coords(cache_token, tokens_per_frame, patch_h, patch_w)
                        if c_coords is None:
                            continue
                        c_frame, c_pr, c_pc = c_coords
                        c_local = point_at_patch(prev_local, c_frame, c_pr, c_pc, patch_h, patch_w)
                        c_world = point_at_patch(prev_world, c_frame, c_pr, c_pc, patch_h, patch_w) if torch.is_tensor(prev_world) else None
                        if c_local is None:
                            continue
                        q_depth = f(q_local[2].item())
                        c_depth = f(c_local[2].item())
                        baseline = math.nan
                        if 0 <= q_frame < int(curr_pose.shape[0]) and 0 <= c_frame < int(prev_pose.shape[0]):
                            baseline = f((curr_pose[q_frame, :3, 3].float() - prev_pose[c_frame, :3, 3].float()).norm().item())
                        world_distance = (
                            f((q_world.float() - c_world.float()).norm().item())
                            if q_world is not None and c_world is not None
                            else math.nan
                        )
                        edge_rows.append(
                            {
                                "case_id": case_id,
                                "target_taxonomy": target_row.get("target_taxonomy", ""),
                                "case_label": target_row.get("case_label", ""),
                                "L3_handoff_transfer_penalty_proxy": target_row.get("L3_handoff_transfer_penalty_proxy", ""),
                                "trace_payload": str(payload_path),
                                "geometry_prev_sidecar": str(prev_path),
                                "geometry_curr_sidecar": str(curr_path),
                                "geometry_sidecar_match": sidecar_match,
                                "head_idx": head_idx,
                                "anchor_id": anchor_text,
                                "query_token": q_token,
                                "cache_token": cache_token,
                                "query_frame": q_frame,
                                "cache_frame": c_frame,
                                "query_depth": q_depth,
                                "cache_depth": c_depth,
                                "query_world_x": f(q_world[0].item()) if q_world is not None else math.nan,
                                "query_world_y": f(q_world[1].item()) if q_world is not None else math.nan,
                                "query_world_z": f(q_world[2].item()) if q_world is not None else math.nan,
                                "cache_world_x": f(c_world[0].item()) if c_world is not None else math.nan,
                                "cache_world_y": f(c_world[1].item()) if c_world is not None else math.nan,
                                "cache_world_z": f(c_world[2].item()) if c_world is not None else math.nan,
                                "query_conf": scalar_at_patch(curr_conf, q_frame, q_pr, q_pc, patch_h, patch_w),
                                "cache_conf": scalar_at_patch(prev_conf, c_frame, c_pr, c_pc, patch_h, patch_w),
                                "camera_translation_baseline": baseline,
                                "world_pair_distance": world_distance,
                                "baseline_over_query_depth": baseline / max(q_depth, 1.0e-6) if math.isfinite(baseline) and q_depth > 0 else math.nan,
                                "abs_log_depth_ratio": abs(math.log(max(q_depth, 1.0e-6) / max(c_depth, 1.0e-6)))
                                if q_depth > 0.0 and c_depth > 0.0 else math.nan,
                                "abs_depth_diff": abs(q_depth - c_depth) if math.isfinite(q_depth) and math.isfinite(c_depth) else math.nan,
                                "claim_level": "geometry_smoke_edge_diagnostic_no_action",
                            }
                        )
        joined = lifecycle_ids & geometry_ids
        parts = [row for row in edge_rows if row.get("case_id") == case_id]
        scale_mode_count, scale_mode_entropy = entropy_from_values([row.get("abs_log_depth_ratio") for row in parts])
        case_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": target_row.get("target_taxonomy", ""),
                "case_label": target_row.get("case_label", ""),
                "L3_handoff_transfer_penalty_proxy": target_row.get("L3_handoff_transfer_penalty_proxy", ""),
                "trace_payload": str(payload_path),
                "lifecycle_row_count": lifecycle_row_count,
                "lifecycle_rows_with_seed_mode": lifecycle_seed_rows,
                "lifecycle_unique_anchor_count": len(lifecycle_ids),
                "geometry_hit_nonnegative_count": nonnegative_count,
                "geometry_hit_unique_anchor_count": len(geometry_ids),
                "lifecycle_geometry_same_payload_join_count": len(joined),
                "lifecycle_geometry_same_payload_join_coverage": frac(len(joined), len(lifecycle_ids)),
                "geometry_edge_row_count": len(parts),
                "query_depth_mean": mean([row.get("query_depth") for row in parts]),
                "query_depth_std": std([row.get("query_depth") for row in parts]),
                "query_inverse_depth_std": std([1.0 / max(f(row.get("query_depth")), 1.0e-6) for row in parts if f(row.get("query_depth")) > 0.0]),
                "cache_depth_mean": mean([row.get("cache_depth") for row in parts]),
                "cache_depth_std": std([row.get("cache_depth") for row in parts]),
                "abs_log_depth_ratio_mean": mean([row.get("abs_log_depth_ratio") for row in parts]),
                "abs_log_depth_ratio_std": std([row.get("abs_log_depth_ratio") for row in parts]),
                "world_pair_distance_mean": mean([row.get("world_pair_distance") for row in parts]),
                "world_pair_distance_std": std([row.get("world_pair_distance") for row in parts]),
                "baseline_over_query_depth_mean": mean([row.get("baseline_over_query_depth") for row in parts]),
                "baseline_over_query_depth_std": std([row.get("baseline_over_query_depth") for row in parts]),
                "query_world_spread_svd_ratio": svd_spread_ratio(parts, "query_world"),
                "cache_world_spread_svd_ratio": svd_spread_ratio(parts, "cache_world"),
                "local_scale_mode_count": scale_mode_count,
                "local_scale_mode_entropy": scale_mode_entropy,
                "sidecar_count": len(list((payload_path.parent.parent / "per_chunk_geometry").glob("chunk_*.pt"))),
                "claim_level": "geometry_smoke_case_diagnostic_no_action",
            }
        )
    return case_rows, edge_rows, read_errors


def scoped_eval_rows(case_rows: list[dict[str, Any]], scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if scope == "clean_safe":
        eval_rows = [row for row in case_rows if row.get("target_taxonomy") in {POS_TAX, SAFE_TAX}]
        positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
        controls = [row for row in eval_rows if row.get("target_taxonomy") == SAFE_TAX]
        return eval_rows, positives, controls
    if scope == "all_non_handoff":
        eval_rows = [row for row in case_rows if row.get("target_taxonomy")]
        positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
        controls = [row for row in eval_rows if row.get("target_taxonomy") != POS_TAX]
        return eval_rows, positives, controls
    return [], [], []


def policy_rows(case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    metrics = [
        ("high_abs_log_depth_ratio_mean", "abs_log_depth_ratio_mean", "higher_bad"),
        ("high_abs_log_depth_ratio_std", "abs_log_depth_ratio_std", "higher_bad"),
        ("high_world_pair_distance_mean", "world_pair_distance_mean", "higher_bad"),
        ("high_world_pair_distance_std", "world_pair_distance_std", "higher_bad"),
        ("high_baseline_over_query_depth_mean", "baseline_over_query_depth_mean", "higher_bad"),
        ("high_baseline_over_query_depth_std", "baseline_over_query_depth_std", "higher_bad"),
        ("high_query_inverse_depth_std", "query_inverse_depth_std", "higher_bad"),
        ("low_query_world_spread_svd_ratio", "query_world_spread_svd_ratio", "lower_bad"),
        ("low_cache_world_spread_svd_ratio", "cache_world_spread_svd_ratio", "lower_bad"),
        ("high_local_scale_mode_count", "local_scale_mode_count", "higher_bad"),
        ("high_local_scale_mode_entropy", "local_scale_mode_entropy", "higher_bad"),
        ("high_geometry_hit_unique_anchor_count", "geometry_hit_unique_anchor_count", "higher_bad"),
        ("low_geometry_hit_unique_anchor_count", "geometry_hit_unique_anchor_count", "lower_bad"),
    ]
    rows: list[dict[str, Any]] = []
    best_by_scope: dict[str, dict[str, Any]] = {}
    for scope in ["clean_safe", "all_non_handoff"]:
        eval_rows, positives, controls = scoped_eval_rows(case_rows, scope)
        scoped_rows: list[dict[str, Any]] = []
        for policy_name, field, direction in metrics:
            usable = [row for row in eval_rows if math.isfinite(f(row.get(field)))]
            selected_count = len(positives)
            if not usable or selected_count <= 0:
                continue
            ranked = sorted(usable, key=lambda row: f(row.get(field)), reverse=(direction == "higher_bad"))
            selected = ranked[:selected_count]
            selected_cases = {row["case_id"] for row in selected}
            tp = sum(1 for row in positives if row["case_id"] in selected_cases)
            fp = sum(1 for row in controls if row["case_id"] in selected_cases)
            bad_recall = tp / len(positives) if positives else math.nan
            control_fpr = fp / len(controls) if controls else math.nan
            balanced_accuracy = 0.5 * (bad_recall + (1.0 - control_fpr)) if math.isfinite(bad_recall) and math.isfinite(control_fpr) else math.nan
            row = {
                "eval_scope": scope,
                "policy_name": policy_name,
                "score_field": field,
                "direction": direction,
                "selected_count": selected_count,
                "selected_cases": ";".join(sorted(selected_cases)),
                "true_positive_cases": ";".join(sorted(row["case_id"] for row in positives if row["case_id"] in selected_cases)),
                "false_positive_cases": ";".join(sorted(row["case_id"] for row in controls if row["case_id"] in selected_cases)),
                "bad_recall": bad_recall,
                "control_FPR": control_fpr,
                "balanced_accuracy": balanced_accuracy,
                "claim_level": "geometry_smoke_policy_sanity_no_action",
            }
            rows.append(row)
            scoped_rows.append(row)
        best_by_scope[scope] = max(scoped_rows, key=lambda row: f(row.get("balanced_accuracy"), -1.0), default={})
    return rows, best_by_scope.get("clean_safe", {}), best_by_scope.get("all_non_handoff", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    target = target_by_case()
    smoke_summary = read_json(args.smoke_root / "summary.json")
    case_rows, edge_rows, read_errors = collect_edge_rows(args.smoke_root, target)
    policy, best, best_all = policy_rows(case_rows)
    lifecycle_total = sum(int(row.get("lifecycle_unique_anchor_count") or 0) for row in case_rows)
    joined_total = sum(int(row.get("lifecycle_geometry_same_payload_join_count") or 0) for row in case_rows)
    positive_count = sum(1 for row in case_rows if row.get("target_taxonomy") == POS_TAX)
    safe_count = sum(1 for row in case_rows if row.get("target_taxonomy") == SAFE_TAX)
    geometry_sidecar_cases = len({path.parents[2].name for path in args.smoke_root.glob("*/READ_NO_ACTION/per_chunk_geometry/chunk_*.pt")})
    summary = {
        "schema": "acl2_v101_geometry_smoke_alignment_v1",
        "diagnostic_only": True,
        "smoke_root": str(args.smoke_root),
        "smoke_status": smoke_summary.get("status", ""),
        "selected_case_count": len(case_rows),
        "positive_case_count": positive_count,
        "safe_good_count": safe_count,
        "trace_payload_file_count": len(list(args.smoke_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))),
        "per_chunk_geometry_sidecar_file_count": len(list(args.smoke_root.glob("*/READ_NO_ACTION/per_chunk_geometry/chunk_*.pt"))),
        "case_with_per_chunk_geometry_sidecar_count": geometry_sidecar_cases,
        "pose_trace_file_count": len(list(args.smoke_root.glob("*/READ_NO_ACTION/per_chunk_pose_trace.jsonl"))),
        "read_error_count": len(read_errors),
        "geometry_edge_row_count": len(edge_rows),
        "lifecycle_unique_anchor_count": lifecycle_total,
        "lifecycle_geometry_same_payload_join_count": joined_total,
        "lifecycle_geometry_same_payload_join_coverage": frac(joined_total, lifecycle_total),
        "all_cases_geometry_aligned": bool(case_rows) and all(f(row.get("lifecycle_geometry_same_payload_join_coverage")) == 1.0 for row in case_rows),
        "geometry_smoke_alignment_pass": (
            smoke_summary.get("status") == "complete"
            and int(smoke_summary.get("failed_job_count", 0) or 0) == 0
            and len(read_errors) == 0
            and bool(case_rows)
            and all(f(row.get("lifecycle_geometry_same_payload_join_coverage")) == 1.0 for row in case_rows)
        ),
        "best_geometry_policy": best.get("policy_name", ""),
        "best_geometry_policy_balanced_accuracy": best.get("balanced_accuracy", ""),
        "best_geometry_policy_selected_cases": best.get("selected_cases", ""),
        "best_geometry_policy_eval_scope": best.get("eval_scope", ""),
        "best_geometry_policy_all_non_handoff": best_all.get("policy_name", ""),
        "best_geometry_policy_all_non_handoff_balanced_accuracy": best_all.get("balanced_accuracy", ""),
        "best_geometry_policy_all_non_handoff_selected_cases": best_all.get("selected_cases", ""),
        "q2_true_stage_pass": False,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "claim": "Lifecycle-aligned geometry smoke validates instrumentation on clean6 only; it is not full true-stage admission.",
    }

    prefix = args.prefix
    summary_path = FINAL / f"{prefix}_summary.json"
    case_path = FINAL / f"{prefix}_case_rows.csv"
    edge_path = FINAL / f"{prefix}_edge_rows.csv"
    error_path = FINAL / f"{prefix}_read_errors.csv"
    policy_path = FINAL / f"{prefix}_policy_rows.csv"
    report_path = FINAL / f"{prefix}_report.md"
    write_json(summary_path, summary)
    write_rows(case_path, case_rows)
    write_rows(edge_path, edge_rows)
    write_rows(error_path, read_errors)
    write_rows(policy_path, policy)
    report_path.write_text(
        "\n".join(
            [
                "# ACL2 v101 Geometry Smoke Alignment",
                "",
                "This report audits lifecycle-aligned geometry sidecars from a no-action clean6 smoke run.",
                "",
                "## Summary",
                "",
                f"- selected_case_count: {summary['selected_case_count']}",
                f"- positive_case_count: {summary['positive_case_count']}",
                f"- safe_good_count: {summary['safe_good_count']}",
                f"- trace_payload_file_count: {summary['trace_payload_file_count']}",
                f"- per_chunk_geometry_sidecar_file_count: {summary['per_chunk_geometry_sidecar_file_count']}",
                f"- geometry_edge_row_count: {summary['geometry_edge_row_count']}",
                f"- lifecycle_unique_anchor_count: {summary['lifecycle_unique_anchor_count']}",
                f"- lifecycle_geometry_same_payload_join_coverage: {summary['lifecycle_geometry_same_payload_join_coverage']}",
                f"- best_geometry_policy: {summary['best_geometry_policy']}",
                f"- best_geometry_policy_balanced_accuracy: {summary['best_geometry_policy_balanced_accuracy']}",
                f"- best_geometry_policy_all_non_handoff: {summary['best_geometry_policy_all_non_handoff']}",
                f"- best_geometry_policy_all_non_handoff_balanced_accuracy: {summary['best_geometry_policy_all_non_handoff_balanced_accuracy']}",
                f"- q2_true_stage_pass: {summary['q2_true_stage_pass']}",
                "",
                "## Blocker",
                "",
                "The smoke proves the instrumentation path on clean6, but it is sampled and not a full target-universe true-stage admission. Runtime remains disabled.",
                "",
                "## Artifacts",
                "",
                f"- `{summary_path}`",
                f"- `{case_path}`",
                f"- `{edge_path}`",
                f"- `{policy_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "selected_case_count": summary["selected_case_count"],
                "geometry_smoke_alignment_pass": summary["geometry_smoke_alignment_pass"],
                "lifecycle_geometry_same_payload_join_coverage": summary["lifecycle_geometry_same_payload_join_coverage"],
                "best_geometry_policy": summary["best_geometry_policy"],
                "best_geometry_policy_balanced_accuracy": summary["best_geometry_policy_balanced_accuracy"],
                "best_geometry_policy_all_non_handoff": summary["best_geometry_policy_all_non_handoff"],
                "best_geometry_policy_all_non_handoff_balanced_accuracy": summary[
                    "best_geometry_policy_all_non_handoff_balanced_accuracy"
                ],
                "q2_true_stage_pass": summary["q2_true_stage_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
