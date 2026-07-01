#!/usr/bin/env python3
"""Build trajectory-error visual panels for ACL2 v102 base cases.

This is an offline diagnostic/fail-forward helper. It reuses existing legacy
TUM trajectory artifacts and KITTI GT poses to materialize per-frame trajectory
error panels aligned with the v102 RGB/semantic overlays. It does not run a new
model and does not claim a strict local point-level residual map.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _umeyama_sim3,
)


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
BASE_ROWS = ROOT / "stage2_base_case_selection/base_case_rows.csv"
RGB_MANIFEST = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_manifest.csv"
OUT = ROOT / "stage2_base_case_selection/trajectory_error_overlay_panels"
MANIFEST = ROOT / "stage2_base_case_selection/trajectory_error_overlay_manifest.csv"
SUMMARY = ROOT / "stage2_base_case_selection/trajectory_error_overlay_summary.json"
KITTI_POSE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")

TRAJECTORY_SOURCES = [
    (
        "v98_stage1_k_hygiene_repair_read_no_action",
        Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/stage1_k_swa_trace_extension_hygiene_repair"),
        "READ_NO_ACTION",
    ),
    (
        "v98_stage1_k_trace_extension_read_no_action",
        Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/stage1_k_swa_trace_extension"),
        "READ_NO_ACTION",
    ),
    (
        "v97_trackE2_key_stability_read_no_action",
        Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control/trackE2_swa_key_stability_fallback_probe"),
        "READ_NO_ACTION",
    ),
    (
        "v97_trackE2_topk_identity_read_no_action",
        Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control/trackE2_swa_topk_identity_trace_probe"),
        "READ_NO_ACTION",
    ),
    (
        "v96_trackJ_raw_qk_trace_baseline_noop",
        Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control/trackJ_raw_qk_trace_smoke"),
        "baseline_noop",
    ),
    (
        "v95_trackE_alpha04_native_actual",
        Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control/trackE_alpha04_runtime_probe"),
        "native_actual",
    ),
    (
        "v101_outcomeD_merge_gauge_rich_selector_native_actual",
        Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/outcomeD_merge_gauge_rich_selector_replay_probe"),
        "native_actual",
    ),
    (
        "v94_phase5_remaining_labelled_carrier_native_actual",
        Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control/phase5_remaining_labelled_carrier_probe"),
        "native_actual",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def find_trajectory(case_id: str, seq: str) -> tuple[str, Path] | None:
    for source_id, root, run_name in TRAJECTORY_SOURCES:
        path = root / case_id / run_name / f"{seq}.txt"
        if path.is_file():
            return source_id, path
    return None


def load_aligned_errors(traj_path: Path, gt_path: Path) -> dict[str, Any]:
    gt_frames, _gt_poses, gt_pos = _load_kitti_gt(gt_path)
    pred_frames, pred_poses, pred_pos = _load_tum_prediction(traj_path, n_gt=len(gt_frames))
    if len(pred_frames) < 3:
        raise ValueError(f"need at least 3 prediction frames for alignment: {traj_path}")
    gt_for_pred = gt_pos[pred_frames]
    scale, rot, trans = _umeyama_sim3(pred_pos, gt_for_pred, with_scale=True)
    aligned_poses = _apply_alignment(pred_poses, scale, rot, trans)
    aligned_pos = aligned_poses[:, :3, 3]
    err_vec = aligned_pos - gt_for_pred
    err_m = np.linalg.norm(err_vec, axis=1)
    return {
        "frames": pred_frames,
        "gt_pos": gt_for_pred,
        "aligned_pos": aligned_pos,
        "err_m": err_m,
        "scale": scale,
    }


def chunk_bounds(rgb_row: dict[str, str], row: dict[str, str]) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for key in ("prev_chunk_dir", "curr_chunk_dir"):
        value = rgb_row.get(key, "")
        if not value:
            continue
        manifest_path = Path(value) / "manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        starts.append(int(payload["start_frame"]))
        ends.append(int(payload["end_frame"]))
    if starts and ends:
        return min(starts), max(ends)
    stride = 29
    prev = int(row["prev_chunk"])
    curr = int(row["curr_chunk"])
    return prev * stride, curr * stride + 32


def nearest_error(aligned: dict[str, Any], frame: int) -> float | None:
    frames = aligned["frames"]
    idxs = np.where(frames == int(frame))[0]
    if len(idxs) == 0:
        return None
    return float(aligned["err_m"][int(idxs[0])])


def panel_for_case(row: dict[str, str], rgb_row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    seq = row["seq"]
    found = find_trajectory(case_id, seq)
    if found is None:
        return {
            "case_id": case_id,
            "seq": seq,
            "status": "missing_trajectory_source",
            "trajectory_error_map_available": False,
            "local_point_error_map_available": False,
            "strict_visual_panel": False,
        }
    source_id, traj_path = found
    gt_path = KITTI_POSE_ROOT / f"{seq}.txt"
    if not gt_path.is_file():
        return {
            "case_id": case_id,
            "seq": seq,
            "status": "missing_gt_pose",
            "trajectory_source_id": source_id,
            "trajectory_path": traj_path.as_posix(),
            "gt_path": gt_path.as_posix(),
            "trajectory_error_map_available": False,
            "local_point_error_map_available": False,
            "strict_visual_panel": False,
        }
    aligned = load_aligned_errors(traj_path, gt_path)
    start, end = chunk_bounds(rgb_row, row)
    frame_ids = aligned["frames"].astype(int)
    mask = (frame_ids >= start) & (frame_ids < end)
    if not bool(mask.any()):
        mask = np.ones_like(frame_ids, dtype=bool)
    frames = frame_ids[mask]
    errors = aligned["err_m"][mask]
    gt_pos = aligned["gt_pos"][mask]
    aligned_pos = aligned["aligned_pos"][mask]
    prev_frame = int(float(rgb_row.get("prev_frame_id") or start))
    curr_frame = int(float(rgb_row.get("curr_frame_id") or prev_frame))
    focus_frame = curr_frame
    focus_error = nearest_error(aligned, focus_frame)
    if focus_error is None:
        idx = int(np.argmin(np.abs(frames - focus_frame)))
        focus_frame = int(frames[idx])
        focus_error = float(errors[idx])

    rgb_panel_path = Path(rgb_row.get("panel_path", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{case_id}_trajectory_error_overlay.png"

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_img = fig.add_subplot(grid[0, 0])
    if rgb_panel_path.is_file():
        ax_img.imshow(Image.open(rgb_panel_path).convert("RGB"))
        ax_img.set_title("RGB/semantic/risk overlay panel")
    else:
        ax_img.text(0.05, 0.5, "RGB/semantic overlay missing", fontsize=12)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    ax_err = fig.add_subplot(grid[0, 1])
    ax_err.plot(frames, errors, color="#2F6FBB", linewidth=1.4, label="aligned position error")
    ax_err.axvspan(start, end, color="#F2C94C", alpha=0.18, label="boundary window")
    ax_err.axvline(focus_frame, color="#D7191C", linewidth=1.2, label=f"focus frame {focus_frame}")
    ax_err.scatter([focus_frame], [focus_error], color="#D7191C", s=28, zorder=5)
    ax_err.set_title(f"{case_id} trajectory error over frame")
    ax_err.set_xlabel("frame")
    ax_err.set_ylabel("aligned position error (m)")
    ax_err.grid(alpha=0.3)
    ax_err.legend(loc="best", fontsize=8)

    ax_xz = fig.add_subplot(grid[1, 0])
    ax_xz.plot(gt_pos[:, 0], gt_pos[:, 2], color="black", linewidth=1.0, label="GT")
    sc = ax_xz.scatter(aligned_pos[:, 0], aligned_pos[:, 2], c=errors, cmap="magma", s=18, label="aligned pred")
    fig.colorbar(sc, ax=ax_xz, label="error (m)")
    ax_xz.set_title("boundary trajectory XZ colored by error")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.axis("equal")
    ax_xz.legend(loc="best", fontsize=8)

    ax_text = fig.add_subplot(grid[1, 1])
    ax_text.axis("off")
    text = "\n".join(
        [
            f"case_id={case_id}",
            f"trajectory_source={source_id}",
            f"trajectory_path={traj_path.as_posix()}",
            f"gt_path={gt_path.as_posix()}",
            "alignment=Sim3 Umeyama, prediction -> KITTI GT",
            f"boundary_window=[{start},{end})",
            f"focus_frame={focus_frame}",
            f"focus_aligned_error_m={focus_error}",
            f"boundary_mean_error_m={float(np.mean(errors))}",
            f"boundary_max_error_m={float(np.max(errors))}",
            f"L3={row.get('L3_handoff_transfer_penalty_proxy', '')}",
            f"primary_source={row.get('primary_drift_source', '')}",
            "trajectory_error_map_available=True",
            "local_point_error_map_available=False",
            "strict_visual_panel=False",
        ]
    )
    ax_text.text(0.01, 0.98, text, va="top", ha="left", fontsize=9)
    fig.suptitle(f"{case_id} trajectory-error + semantic visual autopsy", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close(fig)

    return {
        "case_id": case_id,
        "seq": seq,
        "prev_chunk": row.get("prev_chunk", ""),
        "curr_chunk": row.get("curr_chunk", ""),
        "status": "trajectory_error_overlay_built_not_strict",
        "trajectory_source_id": source_id,
        "trajectory_path": traj_path.as_posix(),
        "gt_path": gt_path.as_posix(),
        "alignment_protocol": "Sim3_Umeyama_prediction_to_KITTI_GT",
        "boundary_start_frame": start,
        "boundary_end_frame": end,
        "focus_frame": focus_frame,
        "focus_aligned_error_m": focus_error,
        "boundary_mean_error_m": float(np.mean(errors)),
        "boundary_max_error_m": float(np.max(errors)),
        "sim3_scale_to_gt": float(aligned["scale"]),
        "panel_path": out_path.as_posix(),
        "rgb_semantic_overlay_panel_path": rgb_row.get("panel_path", ""),
        "trajectory_error_map_available": True,
        "local_point_error_map_available": False,
        "strict_visual_panel": False,
        "strict_blocker": "trajectory error map is materialized, but local point-level residual map is not materialized",
    }


def main() -> int:
    rows = read_rows(BASE_ROWS)
    rgb_rows = {r.get("case_id", ""): r for r in read_rows(RGB_MANIFEST)}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            out_rows.append(panel_for_case(row, rgb_rows.get(row.get("case_id", ""), {})))
        except Exception as exc:  # noqa: BLE001
            out_rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "seq": row.get("seq", ""),
                    "prev_chunk": row.get("prev_chunk", ""),
                    "curr_chunk": row.get("curr_chunk", ""),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "trajectory_error_map_available": False,
                    "local_point_error_map_available": False,
                    "strict_visual_panel": False,
                }
            )
    write_rows(MANIFEST, out_rows)
    source_counts = Counter(r.get("trajectory_source_id", "missing") for r in out_rows)
    summary = {
        "case_count": len(out_rows),
        "trajectory_error_map_built_count": sum(1 for r in out_rows if r.get("trajectory_error_map_available")),
        "local_point_error_map_count": sum(1 for r in out_rows if r.get("local_point_error_map_available")),
        "strict_visual_count": sum(1 for r in out_rows if r.get("strict_visual_panel")),
        "source_counts": dict(source_counts),
        "manifest": MANIFEST.as_posix(),
        "out_dir": OUT.as_posix(),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0 if summary["trajectory_error_map_built_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
