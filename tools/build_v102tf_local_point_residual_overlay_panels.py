#!/usr/bin/env python3
"""Build local point residual visual panels for ACL2 v102 base cases.

The panels use existing v100/v101 per-chunk geometry sidecars. For each case,
they compare prev/curr overlap-frame world point predictions and visualize the
per-pixel point disagreement. Missing sidecars are recorded explicitly.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
BASE_ROWS = ROOT / "stage2_base_case_selection/base_case_rows.csv"
RGB_MANIFEST = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_manifest.csv"
TRAJ_MANIFEST = ROOT / "stage2_base_case_selection/trajectory_error_overlay_manifest.csv"
OUT = ROOT / "stage2_base_case_selection/local_point_residual_overlay_panels"
MANIFEST = ROOT / "stage2_base_case_selection/local_point_residual_overlay_manifest.csv"
SUMMARY = ROOT / "stage2_base_case_selection/local_point_residual_overlay_summary.json"

GEOMETRY_SOURCES = [
    (
        "v102_stage2_local_point_sidecar_repair_read_no_action",
        ROOT / "stage2_base_case_selection/local_point_sidecar_repair_traces",
        "READ_NO_ACTION",
    ),
    (
        "v101_stage_c_seed_bridge_geometry_smoke_target28_read_no_action",
        Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/stage_c_seed_bridge_geometry_smoke_target28"),
        "READ_NO_ACTION",
    ),
    (
        "v101_stage_c_seed_bridge_geometry_smoke_clean6_read_no_action",
        Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/stage_c_seed_bridge_geometry_smoke_clean6"),
        "READ_NO_ACTION",
    ),
    (
        "v101_stage_c_seed_bridge_geometry_smoke_2case_read_no_action",
        Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/stage_c_seed_bridge_geometry_smoke_2case"),
        "READ_NO_ACTION",
    ),
    (
        "v100_probe28_geometry_sidecar_ttt_swa_same_run",
        Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/trackS_same_space_latent_state/probe28_geometry_sidecar"),
        "TTT_SWA_SAME_RUN",
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


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def find_geometry_dir(case_id: str) -> tuple[str, Path] | None:
    for source_id, root, run_name in GEOMETRY_SOURCES:
        sidecar_dir = root / case_id / run_name / "per_chunk_geometry"
        if (sidecar_dir / "chunk_000.pt").is_file() and (sidecar_dir / "chunk_001.pt").is_file():
            return source_id, sidecar_dir
    return None


def choose_local_indices(prev_geo: dict[str, Any], curr_geo: dict[str, Any], rgb_row: dict[str, str]) -> tuple[int, int, int]:
    prev_start = int(prev_geo.get("start_frame", 0))
    curr_start = int(curr_geo.get("start_frame", 0))
    focus = int(float(rgb_row.get("curr_frame_id") or rgb_row.get("prev_frame_id") or curr_start))
    prev_i = min(max(focus - prev_start, 0), int(prev_geo["points"].shape[0]) - 1)
    curr_i = min(max(focus - curr_start, 0), int(curr_geo["points"].shape[0]) - 1)
    return focus, prev_i, curr_i


def finite_stats(values: np.ndarray) -> dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(vals)),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "max": float(np.max(vals)),
    }


def panel_for_case(row: dict[str, str], rgb_row: dict[str, str], traj_row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    found = find_geometry_dir(case_id)
    if found is None:
        return {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "prev_chunk": row.get("prev_chunk", ""),
            "curr_chunk": row.get("curr_chunk", ""),
            "status": "missing_per_chunk_geometry_sidecar",
            "local_point_error_map_available": False,
            "strict_visual_panel": False,
        }
    source_id, sidecar_dir = found
    prev_path = sidecar_dir / "chunk_000.pt"
    curr_path = sidecar_dir / "chunk_001.pt"
    prev_geo = torch_load(prev_path)
    curr_geo = torch_load(curr_path)
    if not isinstance(prev_geo, dict) or not isinstance(curr_geo, dict):
        return {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "status": "geometry_sidecar_not_dict",
            "geometry_source_id": source_id,
            "geometry_sidecar_dir": sidecar_dir.as_posix(),
            "local_point_error_map_available": False,
            "strict_visual_panel": False,
        }
    required = ["points", "conf", "start_frame", "end_frame"]
    missing = [key for key in required if key not in prev_geo or key not in curr_geo]
    if missing or not torch.is_tensor(prev_geo.get("points")) or not torch.is_tensor(curr_geo.get("points")):
        return {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "status": "missing_point_tensors",
            "missing_keys": ";".join(missing),
            "geometry_source_id": source_id,
            "geometry_sidecar_dir": sidecar_dir.as_posix(),
            "local_point_error_map_available": False,
            "strict_visual_panel": False,
        }

    focus_frame, prev_i, curr_i = choose_local_indices(prev_geo, curr_geo, rgb_row)
    prev_points = prev_geo["points"][prev_i].detach().cpu().float()
    curr_points = curr_geo["points"][curr_i].detach().cpu().float()
    prev_conf = prev_geo["conf"][prev_i].detach().cpu().float() if torch.is_tensor(prev_geo.get("conf")) else torch.ones(prev_points.shape[:2])
    curr_conf = curr_geo["conf"][curr_i].detach().cpu().float() if torch.is_tensor(curr_geo.get("conf")) else torch.ones(curr_points.shape[:2])
    h = min(int(prev_points.shape[0]), int(curr_points.shape[0]))
    w = min(int(prev_points.shape[1]), int(curr_points.shape[1]))
    prev_points = prev_points[:h, :w]
    curr_points = curr_points[:h, :w]
    conf = torch.minimum(prev_conf[:h, :w], curr_conf[:h, :w])
    residual = torch.linalg.norm(prev_points - curr_points, dim=-1)
    valid = torch.isfinite(residual) & torch.isfinite(conf) & (conf > 0.05)
    residual_np = residual.numpy()
    residual_np[~valid.numpy()] = np.nan
    stats = finite_stats(residual_np)

    rgb_panel = Path(rgb_row.get("panel_path", ""))
    traj_panel = Path(traj_row.get("panel_path", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{case_id}_local_point_residual_overlay.png"
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2)

    ax_rgb = fig.add_subplot(grid[0, 0])
    if rgb_panel.is_file():
        ax_rgb.imshow(Image.open(rgb_panel).convert("RGB"))
        ax_rgb.set_title("RGB/semantic/risk overlay")
    else:
        ax_rgb.text(0.05, 0.5, "RGB/semantic overlay missing")
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    ax_res = fig.add_subplot(grid[0, 1])
    vmax = stats["p90"] if math.isfinite(stats["p90"]) and stats["p90"] > 0 else None
    img = ax_res.imshow(residual_np, cmap="inferno", vmin=0.0, vmax=vmax)
    fig.colorbar(img, ax=ax_res, label="prev/curr world-point residual")
    ax_res.set_title(f"local point residual map frame {focus_frame}")
    ax_res.set_xticks([])
    ax_res.set_yticks([])

    ax_traj = fig.add_subplot(grid[1, 0])
    if traj_panel.is_file():
        ax_traj.imshow(Image.open(traj_panel).convert("RGB"))
        ax_traj.set_title("trajectory error panel")
    else:
        ax_traj.text(0.05, 0.5, "trajectory panel missing")
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])

    ax_text = fig.add_subplot(grid[1, 1])
    ax_text.axis("off")
    text = "\n".join(
        [
            f"case_id={case_id}",
            f"geometry_source={source_id}",
            f"sidecar_dir={sidecar_dir.as_posix()}",
            f"prev_sidecar={prev_path.as_posix()}",
            f"curr_sidecar={curr_path.as_posix()}",
            f"focus_frame={focus_frame}",
            f"prev_local_idx={prev_i}",
            f"curr_local_idx={curr_i}",
            f"valid_pixel_count={int(valid.sum().item())}",
            f"residual_mean={stats['mean']}",
            f"residual_p90={stats['p90']}",
            f"residual_max={stats['max']}",
            f"L3={row.get('L3_handoff_transfer_penalty_proxy', '')}",
            f"primary_source={row.get('primary_drift_source', '')}",
            "local_point_error_map_available=True",
            "strict_visual_panel=True for this case only; corpus gate still requires all selected base cases",
        ]
    )
    ax_text.text(0.01, 0.98, text, va="top", ha="left", fontsize=8.5)
    fig.suptitle(f"{case_id} local point residual + semantic/trajectory autopsy", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close(fig)

    return {
        "case_id": case_id,
        "seq": row.get("seq", ""),
        "prev_chunk": row.get("prev_chunk", ""),
        "curr_chunk": row.get("curr_chunk", ""),
        "status": "local_point_residual_overlay_built",
        "geometry_source_id": source_id,
        "geometry_sidecar_dir": sidecar_dir.as_posix(),
        "prev_geometry_sidecar": prev_path.as_posix(),
        "curr_geometry_sidecar": curr_path.as_posix(),
        "focus_frame": focus_frame,
        "prev_local_idx": prev_i,
        "curr_local_idx": curr_i,
        "valid_pixel_count": int(valid.sum().item()),
        "local_point_residual_mean": stats["mean"],
        "local_point_residual_p50": stats["p50"],
        "local_point_residual_p90": stats["p90"],
        "local_point_residual_max": stats["max"],
        "panel_path": out_path.as_posix(),
        "rgb_semantic_overlay_panel_path": rgb_row.get("panel_path", ""),
        "trajectory_error_overlay_panel_path": traj_row.get("panel_path", ""),
        "local_point_error_map_available": True,
        "strict_visual_panel": True,
        "strict_blocker": "",
    }


def main() -> int:
    rows = read_rows(BASE_ROWS)
    rgb_by_case = {r.get("case_id", ""): r for r in read_rows(RGB_MANIFEST)}
    traj_by_case = {r.get("case_id", ""): r for r in read_rows(TRAJ_MANIFEST)}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            out_rows.append(panel_for_case(row, rgb_by_case.get(row.get("case_id", ""), {}), traj_by_case.get(row.get("case_id", ""), {})))
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
                    "local_point_error_map_available": False,
                    "strict_visual_panel": False,
                }
            )
    write_rows(MANIFEST, out_rows)
    source_counts = Counter(r.get("geometry_source_id", "missing") for r in out_rows)
    summary = {
        "case_count": len(out_rows),
        "local_point_residual_map_built_count": sum(1 for r in out_rows if r.get("local_point_error_map_available")),
        "strict_visual_count": sum(1 for r in out_rows if r.get("strict_visual_panel")),
        "source_counts": dict(source_counts),
        "manifest": MANIFEST.as_posix(),
        "out_dir": OUT.as_posix(),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0 if summary["local_point_residual_map_built_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
