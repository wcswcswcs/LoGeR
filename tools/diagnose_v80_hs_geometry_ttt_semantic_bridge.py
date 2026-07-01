#!/usr/bin/env python3
"""Lightweight geometry-error to TTT/semantic bridge for v80 HS runs.

This is diagnostic-only. It reuses completed trajectory and dump artifacts:

- KITTI trajectory files for baseline/candidate/control frame-error deltas.
- TTT spatial post-delta maps for role/action/world-delta masks.
- READ cue patch dumps for READ-active masks.
- Stage-C semantic label maps for label-level explanations.

No model forward pass is run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional diagnostic plotting.
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_v80_ttt_geometry_error_visual_bridge import _load_aligned_run, _write_csv, _write_json  # noqa: E402


ROLE_NAMES = {0: "void", 1: "positive", 2: "neutral", 3: "negative", 4: "swa_protect"}
STATIC_CARRIER_KEYWORDS = (
    "road",
    "sidewalk",
    "building",
    "house",
    "wall",
    "fence",
    "pole",
    "traffic",
    "sign",
    "bridge",
    "tunnel",
    "rail",
    "curb",
    "ground",
    "parking",
    "lamp",
    "light",
)
RISKY_CARRIER_KEYWORDS = (
    "sky",
    "tree",
    "vegetation",
    "grass",
    "plant",
    "car",
    "truck",
    "bus",
    "van",
    "person",
    "rider",
    "bicycle",
    "motorcycle",
    "train",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, default=None)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--control-name", default="control")
    parser.add_argument(
        "--ttt-map-run-dir",
        type=Path,
        default=None,
        help="Optional run directory containing ttt_spatial_post_delta_maps; defaults to candidate-run-dir.",
    )
    parser.add_argument(
        "--read-cue-run-dir",
        type=Path,
        default=None,
        help="Optional run directory containing read_cue_patch_dumps; defaults to candidate-run-dir.",
    )
    parser.add_argument("--seq", default="00")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--stage-c-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k-frames", type=int, default=8)
    parser.add_argument(
        "--focus-frames",
        default="",
        help="Optional comma-separated global frames to include even if they are not among top-k deltas.",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument(
        "--skip-frame-semantic",
        action="store_true",
        help="Write trajectory/error CSVs even when TTT/READ spatial dump artifacts are unavailable.",
    )
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text or "").replace(";", ",").split(",") if x.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _jsonable(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_pt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data if isinstance(data, dict) else {}


def _primary_chunk_for_frame(frame: int, chunk_size: int, overlap: int) -> int:
    stride = int(chunk_size) - int(overlap)
    return int(math.floor(int(frame) / max(1, stride)))


def _local_frame(frame: int, chunk: int, chunk_size: int, overlap: int) -> int:
    stride = int(chunk_size) - int(overlap)
    return int(frame) - int(chunk) * stride


def _trajectory_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seq = str(args.seq).zfill(2)
    def _traj(run_dir: Path) -> Path:
        primary = run_dir / f"{seq}.txt"
        if primary.exists():
            return primary
        legacy = run_dir / "01.txt"
        if legacy.exists():
            return legacy
        return primary

    base = _load_aligned_run(_traj(args.baseline_run_dir), args.gt)
    cand = _load_aligned_run(_traj(args.candidate_run_dir), args.gt)
    ctrl = _load_aligned_run(_traj(args.control_run_dir), args.gt) if args.control_run_dir else None
    base_by_frame = {int(f): i for i, f in enumerate(base["frames"])}
    ctrl_by_frame = {int(f): i for i, f in enumerate(ctrl["frames"])} if ctrl else {}
    rows: list[dict[str, Any]] = []
    for ci, frame_raw in enumerate(cand["frames"]):
        frame = int(frame_raw)
        if frame not in base_by_frame:
            continue
        bi = base_by_frame[frame]
        chunk = _primary_chunk_for_frame(frame, int(args.chunk_size), int(args.chunk_overlap))
        row = {
            "frame": frame,
            "primary_chunk_id": int(chunk),
            "local_frame": _local_frame(frame, chunk, int(args.chunk_size), int(args.chunk_overlap)),
            "baseline_error_m": float(base["err_m"][bi]),
            "candidate_error_m": float(cand["err_m"][ci]),
            "delta_error_vs_baseline_m": float(cand["err_m"][ci] - base["err_m"][bi]),
            "candidate_x": float(cand["aligned_pos"][ci, 0]),
            "candidate_y": float(cand["aligned_pos"][ci, 1]),
            "candidate_z": float(cand["aligned_pos"][ci, 2]),
            "baseline_x": float(base["aligned_pos"][bi, 0]),
            "baseline_y": float(base["aligned_pos"][bi, 1]),
            "baseline_z": float(base["aligned_pos"][bi, 2]),
            "gt_x": float(cand["gt_pos"][ci, 0]),
            "gt_y": float(cand["gt_pos"][ci, 1]),
            "gt_z": float(cand["gt_pos"][ci, 2]),
        }
        if ctrl and frame in ctrl_by_frame:
            ui = ctrl_by_frame[frame]
            row["control_error_m"] = float(ctrl["err_m"][ui])
            row["delta_error_vs_control_m"] = float(cand["err_m"][ci] - ctrl["err_m"][ui])
        rows.append(row)
    return rows


def _top_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    values = x.detach().cpu().float()
    if values.ndim != 2:
        raise ValueError(f"expected H,W map, got {tuple(values.shape)}")
    flat = values.flatten()
    k = max(1, int(math.ceil(float(frac) * int(flat.numel()))))
    idx = torch.topk(flat, k=min(k, int(flat.numel())), largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape_as(values)


def _mean_first_dim(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().cpu().float()
    while y.ndim > 3:
        y = y.mean(dim=0)
    return y


def _stage_c_payload(stage_c_cache_dir: Path, chunk: int) -> dict[str, Any]:
    matches = sorted(stage_c_cache_dir.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        return {"path": "", "payload": {}}
    return {"path": str(matches[0]), "payload": _load_pt(matches[0])}


def _label_grid(stage_c: dict[str, Any], local_frame: int, h: int, w: int) -> tuple[torch.Tensor | None, list[str]]:
    payload = stage_c.get("payload") if isinstance(stage_c, dict) else {}
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict):
        return None, []
    labels = sem.get("label_maps")
    names = [str(x) for x in sem.get("label_names", [])]
    if not torch.is_tensor(labels):
        return None, names
    maps = labels.detach().cpu().long()
    if maps.ndim != 3 or not (0 <= int(local_frame) < int(maps.shape[0])):
        return None, names
    label_map = maps[int(local_frame)]
    img_h, img_w = int(label_map.shape[0]), int(label_map.shape[1])
    ys = torch.clamp(((torch.arange(h).float() + 0.5) * img_h / h).long(), 0, img_h - 1)
    xs = torch.clamp(((torch.arange(w).float() + 0.5) * img_w / w).long(), 0, img_w - 1)
    return label_map[ys[:, None], xs[None, :]], names


def _label_counts(label_grid: torch.Tensor | None, mask: torch.Tensor, names: list[str], top_k: int = 10) -> list[dict[str, Any]]:
    if label_grid is None:
        return []
    selected = label_grid[mask.detach().cpu().bool()]
    if int(selected.numel()) <= 0:
        return []
    vals, counts = torch.unique(selected.long(), return_counts=True)
    rows = []
    total = int(counts.sum().item())
    for value, count in zip(vals.tolist(), counts.tolist()):
        name = names[value] if 0 <= int(value) < len(names) else str(value)
        rows.append({"label_id": int(value), "label_name": name, "count": int(count), "fraction": float(count / max(1, total))})
    return sorted(rows, key=lambda r: r["count"], reverse=True)[:top_k]


def _label_keyword_mask(label_grid: torch.Tensor | None, names: list[str], keywords: tuple[str, ...]) -> torch.Tensor | None:
    if label_grid is None:
        return None
    selected_ids = []
    lowered = [str(x).lower() for x in names]
    for idx, name in enumerate(lowered):
        if any(keyword in name for keyword in keywords):
            selected_ids.append(int(idx))
    out = torch.zeros_like(label_grid.detach().cpu().long(), dtype=torch.bool)
    if selected_ids:
        ids = torch.tensor(selected_ids, dtype=torch.long)
        out = torch.isin(label_grid.detach().cpu().long(), ids)
    return out


def _mask_mass(mask: torch.Tensor | None) -> float | None:
    if mask is None:
        return None
    return float(mask.detach().cpu().bool().float().mean().item())


def _role_fractions(role: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = torch.round(role.detach().cpu().float())[mask.detach().cpu().bool()].long()
    denom = int(selected.numel())
    out: dict[str, float] = {}
    if denom <= 0:
        return out
    for role_id, name in ROLE_NAMES.items():
        out[f"role_{role_id}_{name}_fraction"] = float((selected == int(role_id)).float().sum().item() / denom)
    return out


def _mask_overlap(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.detach().cpu().bool()
    bb = b.detach().cpu().bool()
    denom = int(bb.sum().item())
    if denom <= 0:
        return None
    return float((aa & bb).float().sum().item() / denom)


def _plot_trajectory_diagnostics(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    if plt is None or not rows:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray([int(r["frame"]) for r in rows], dtype=np.int64)
    baseline_error = np.asarray([float(r["baseline_error_m"]) for r in rows], dtype=np.float64)
    candidate_error = np.asarray([float(r["candidate_error_m"]) for r in rows], dtype=np.float64)
    delta = np.asarray([float(r["delta_error_vs_baseline_m"]) for r in rows], dtype=np.float64)
    has_control = all(r.get("control_error_m") is not None for r in rows)
    control_error = np.asarray([float(r["control_error_m"]) for r in rows], dtype=np.float64) if has_control else None
    gt_x = np.asarray([float(r["gt_x"]) for r in rows], dtype=np.float64)
    gt_z = np.asarray([float(r["gt_z"]) for r in rows], dtype=np.float64)
    baseline_x = np.asarray([float(r["baseline_x"]) for r in rows], dtype=np.float64)
    baseline_z = np.asarray([float(r["baseline_z"]) for r in rows], dtype=np.float64)
    candidate_x = np.asarray([float(r["candidate_x"]) for r in rows], dtype=np.float64)
    candidate_z = np.asarray([float(r["candidate_z"]) for r in rows], dtype=np.float64)

    paths: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, baseline_error, label="baseline", linewidth=1.2)
    ax.plot(frames, candidate_error, label="candidate", linewidth=1.2)
    if control_error is not None:
        ax.plot(frames, control_error, label="control", linewidth=1.0, alpha=0.75)
    ax.bar(frames, delta, label="candidate-baseline", alpha=0.28)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("aligned error / delta (m)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "error_over_frame.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["error_over_frame_png"] = str(path)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(gt_x, gt_z, color="black", label="GT", linewidth=1.2)
    ax.plot(baseline_x, baseline_z, color="#377eb8", label="baseline", linewidth=1.0, alpha=0.75)
    sc = ax.scatter(candidate_x, candidate_z, c=delta, cmap="coolwarm", s=14, label="candidate delta")
    fig.colorbar(sc, ax=ax, label="candidate-baseline error (m)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "trajectory_error_map_xz.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["trajectory_error_map_xz_png"] = str(path)
    return paths


def _plot_frame_panel(
    *,
    out_dir: Path,
    row: dict[str, Any],
    world_f: torch.Tensor,
    action_f: torch.Tensor,
    role_f: torch.Tensor,
    read_q90_f: torch.Tensor,
    proposed_static_carrier: torch.Tensor | None,
    risky_read_world: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> str | None:
    if plt is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    role_np = torch.round(role_f.detach().cpu().float()).numpy()
    panels: list[tuple[str, np.ndarray, str, dict[str, Any]]] = [
        ("world delta", world_f.detach().cpu().float().numpy(), "magma", {}),
        ("action delta", action_f.detach().cpu().float().numpy(), "viridis", {}),
        ("TTT role", role_np, "tab10", {"vmin": 0, "vmax": 4}),
        ("READ q90", read_q90_f.detach().cpu().bool().numpy().astype(np.float32), "gray_r", {"vmin": 0, "vmax": 1}),
        (
            "static carrier",
            (proposed_static_carrier.detach().cpu().bool().numpy().astype(np.float32) if proposed_static_carrier is not None else np.zeros_like(role_np)),
            "gray_r",
            {"vmin": 0, "vmax": 1},
        ),
        (
            "risky read-world",
            (risky_read_world.detach().cpu().bool().numpy().astype(np.float32) if risky_read_world is not None else np.zeros_like(role_np)),
            "gray_r",
            {"vmin": 0, "vmax": 1},
        ),
    ]
    if labels is not None:
        panels.append(("semantic label id", labels.detach().cpu().long().numpy(), "tab20", {}))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes_flat = list(axes.flatten())
    for ax, (title, image, cmap, kwargs) in zip(axes_flat, panels):
        im = ax.imshow(image, cmap=cmap, interpolation="nearest", **kwargs)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if title in {"world delta", "action delta"}:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    frame = int(row["frame"])
    chunk = int(row["primary_chunk_id"])
    fig.suptitle(
        f"frame {frame} chunk {chunk} delta_base={float(row['delta_error_vs_baseline_m']):.3f}m",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"frame_{frame:06d}_chunk_{chunk:03d}_ttt_semantic_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _frame_semantic_row(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    chunk = int(row["primary_chunk_id"])
    local = int(row["local_frame"])
    ttt_root = args.ttt_map_run_dir or args.candidate_run_dir
    read_root = args.read_cue_run_dir or args.candidate_run_dir
    ttt = _load_pt(ttt_root / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt")
    read = _load_pt(read_root / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt")
    tensors = read.get("tensors") if isinstance(read.get("tensors"), dict) else {}
    read_patch = tensors.get("read_patch_final")
    read_q90 = tensors.get("read_active_q90_patch")
    role = ttt.get("R_ttt_tok_patch")
    action = ttt.get("action_delta_norm_projection_patch")
    world = ttt.get("pass1_pass2_world_points_l2_patch")
    local_points = ttt.get("pass1_pass2_local_points_l2_patch")
    out = dict(row)
    out.update(
        {
            "candidate_run_dir": str(args.candidate_run_dir),
            "ttt_map_run_dir": str(ttt_root),
            "read_cue_run_dir": str(read_root),
            "ttt_map_path": str(ttt_root / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt"),
            "read_cue_path": str(read_root / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt"),
            "has_ttt_map": bool(ttt),
            "has_read_cue": bool(read),
        }
    )
    stage_c = _stage_c_payload(args.stage_c_cache_dir, chunk)
    out["stage_c_masklet"] = stage_c.get("path", "")
    payload = stage_c.get("payload") if isinstance(stage_c, dict) else {}
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    label_maps = sem.get("label_maps") if isinstance(sem, dict) else None
    label_names = [str(x) for x in sem.get("label_names", [])] if isinstance(sem, dict) else []
    if torch.is_tensor(label_maps) and label_maps.ndim == 3 and 0 <= local < int(label_maps.shape[0]):
        frame_labels = label_maps.detach().cpu().long()[local]
        full_mask = torch.ones_like(frame_labels, dtype=torch.bool)
        static_mask = _label_keyword_mask(frame_labels, label_names, STATIC_CARRIER_KEYWORDS)
        risky_mask = _label_keyword_mask(frame_labels, label_names, RISKY_CARRIER_KEYWORDS)
        out.update(
            {
                "frame_semantic_available": True,
                "frame_semantic_label_shape": [int(frame_labels.shape[0]), int(frame_labels.shape[1])],
                "frame_semantic_top_labels": _label_counts(frame_labels, full_mask, label_names),
                "frame_static_semantic_mass": _mask_mass(static_mask),
                "frame_risky_semantic_mass": _mask_mass(risky_mask),
            }
        )
    else:
        out["frame_semantic_available"] = False
    if not (torch.is_tensor(role) and torch.is_tensor(action) and torch.is_tensor(world)):
        out["diagnostic_status"] = "missing_required_maps"
        return out
    role_t = role.detach().cpu().float()
    action_t = _mean_first_dim(action)
    world_t = world.detach().cpu().float()
    local_t = local_points.detach().cpu().float() if torch.is_tensor(local_points) else None
    if not (0 <= local < int(role_t.shape[0])):
        out["diagnostic_status"] = "local_frame_out_of_range"
        return out
    role_f = role_t[local]
    action_f = action_t[:, :, :] if action_t.ndim == 3 else action_t
    action_f = action_f[local]
    world_f = world_t[local]
    local_f = local_t[local] if torch.is_tensor(local_t) else None
    h, w = int(role_f.shape[0]), int(role_f.shape[1])
    read_f = read_patch.detach().cpu().float()[local] if torch.is_tensor(read_patch) else torch.zeros((h, w))
    read_q90_f = read_q90.detach().cpu().bool()[local] if torch.is_tensor(read_q90) else _top_mask(read_f, float(args.top_frac))
    action_top = _top_mask(action_f, float(args.top_frac))
    world_top = _top_mask(world_f, float(args.top_frac))
    local_top = _top_mask(local_f, float(args.top_frac)) if torch.is_tensor(local_f) else torch.zeros_like(world_top)
    negative = torch.round(role_f).long() == 3
    positive = torch.round(role_f).long() == 1
    labels, names = _label_grid(stage_c, local, h, w)
    static_semantic = _label_keyword_mask(labels, names, STATIC_CARRIER_KEYWORDS)
    risky_semantic = _label_keyword_mask(labels, names, RISKY_CARRIER_KEYWORDS)
    read_world = read_q90_f & world_top
    proposed_static_carrier = read_world & static_semantic if static_semantic is not None else None
    risky_read_world = read_world & risky_semantic if risky_semantic is not None else None
    out.update(
        {
            "diagnostic_status": "ok",
            "stage_c_masklet": stage_c.get("path", ""),
            "read_q90_mass": float(read_q90_f.float().mean().item()),
            "action_top_mass": float(action_top.float().mean().item()),
            "world_top_mass": float(world_top.float().mean().item()),
            "read_q90_given_world_top": _mask_overlap(read_q90_f, world_top),
            "read_q90_given_action_top": _mask_overlap(read_q90_f, action_top),
            "negative_given_world_top": _mask_overlap(negative, world_top),
            "negative_given_action_top": _mask_overlap(negative, action_top),
            "negative_given_read_q90": _mask_overlap(negative, read_q90_f),
            "positive_given_world_top": _mask_overlap(positive, world_top),
            "positive_given_action_top": _mask_overlap(positive, action_top),
            "positive_given_read_q90": _mask_overlap(positive, read_q90_f),
            "read_world_overlap_mass": _mask_mass(read_world),
            "static_semantic_mass": _mask_mass(static_semantic),
            "risky_semantic_mass": _mask_mass(risky_semantic),
            "proposed_static_carrier_mass": _mask_mass(proposed_static_carrier),
            "risky_read_world_mass": _mask_mass(risky_read_world),
            "proposed_static_carrier_given_world_top": _mask_overlap(proposed_static_carrier, world_top) if proposed_static_carrier is not None else None,
            "proposed_static_carrier_given_read_q90": _mask_overlap(proposed_static_carrier, read_q90_f) if proposed_static_carrier is not None else None,
            "proposed_static_carrier_given_read_world": _mask_overlap(proposed_static_carrier, read_world) if proposed_static_carrier is not None else None,
            "risky_semantic_given_read_world": _mask_overlap(risky_semantic, read_world) if risky_semantic is not None else None,
            "action_top_given_proposed_static_carrier": _mask_overlap(action_top, proposed_static_carrier) if proposed_static_carrier is not None else None,
            "proposed_static_carrier_given_action_top": _mask_overlap(proposed_static_carrier, action_top) if proposed_static_carrier is not None else None,
            "negative_given_proposed_static_carrier": _mask_overlap(negative, proposed_static_carrier) if proposed_static_carrier is not None else None,
            "positive_given_proposed_static_carrier": _mask_overlap(positive, proposed_static_carrier) if proposed_static_carrier is not None else None,
            "negative_given_risky_read_world": _mask_overlap(negative, risky_read_world) if risky_read_world is not None else None,
            "positive_given_risky_read_world": _mask_overlap(positive, risky_read_world) if risky_read_world is not None else None,
            "action_top_role_fractions": _role_fractions(role_f, action_top),
            "world_top_role_fractions": _role_fractions(role_f, world_top),
            "read_q90_role_fractions": _role_fractions(role_f, read_q90_f),
            "proposed_static_carrier_role_fractions": _role_fractions(role_f, proposed_static_carrier) if proposed_static_carrier is not None else {},
            "risky_read_world_role_fractions": _role_fractions(role_f, risky_read_world) if risky_read_world is not None else {},
            "action_top_labels": _label_counts(labels, action_top, names),
            "world_top_labels": _label_counts(labels, world_top, names),
            "read_q90_labels": _label_counts(labels, read_q90_f, names),
            "read_world_labels": _label_counts(labels, read_world, names),
            "proposed_static_carrier_labels": _label_counts(labels, proposed_static_carrier, names) if proposed_static_carrier is not None else [],
            "risky_read_world_labels": _label_counts(labels, risky_read_world, names) if risky_read_world is not None else [],
            "negative_labels": _label_counts(labels, negative, names),
            "positive_labels": _label_counts(labels, positive, names),
            "action_mean": float(action_f.mean().item()),
            "action_q90": float(torch.quantile(action_f.flatten(), 0.9).item()),
            "world_delta_mean": float(world_f.mean().item()),
            "world_delta_q90": float(torch.quantile(world_f.flatten(), 0.9).item()),
            "local_delta_mean": float(local_f.mean().item()) if torch.is_tensor(local_f) else None,
            "local_delta_q90": float(torch.quantile(local_f.flatten(), 0.9).item()) if torch.is_tensor(local_f) else None,
        }
    )
    panel = _plot_frame_panel(
        out_dir=args.out_dir / "visual_panels",
        row=row,
        world_f=world_f,
        action_f=action_f,
        role_f=role_f,
        read_q90_f=read_q90_f,
        proposed_static_carrier=proposed_static_carrier,
        risky_read_world=risky_read_world,
        labels=labels,
    )
    out["visual_panel_png"] = panel
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "delta_error_vs_baseline_m",
        "delta_error_vs_control_m",
        "read_q90_given_world_top",
        "read_q90_given_action_top",
        "negative_given_world_top",
        "negative_given_action_top",
        "negative_given_read_q90",
        "positive_given_world_top",
        "positive_given_action_top",
        "positive_given_read_q90",
        "read_world_overlap_mass",
        "static_semantic_mass",
        "risky_semantic_mass",
        "proposed_static_carrier_mass",
        "risky_read_world_mass",
        "proposed_static_carrier_given_world_top",
        "proposed_static_carrier_given_read_q90",
        "proposed_static_carrier_given_read_world",
        "risky_semantic_given_read_world",
        "action_top_given_proposed_static_carrier",
        "proposed_static_carrier_given_action_top",
        "negative_given_proposed_static_carrier",
        "positive_given_proposed_static_carrier",
        "negative_given_risky_read_world",
        "positive_given_risky_read_world",
        "action_mean",
        "world_delta_mean",
        "local_delta_mean",
        "frame_static_semantic_mass",
        "frame_risky_semantic_mass",
    ]
    out = {"frames": len(rows), "ok_frames": sum(1 for r in rows if r.get("diagnostic_status") == "ok")}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and str(r.get(key)) != ""]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_max"] = float(np.max(vals)) if vals else None
    return out


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_rows = _trajectory_rows(args)
    plot_paths = _plot_trajectory_diagnostics(args.out_dir / "plots", traj_rows)
    selected = sorted(traj_rows, key=lambda r: float(r["delta_error_vs_baseline_m"]), reverse=True)[: int(args.top_k_frames)]
    selected_frames = {int(r["frame"]) for r in selected}
    focus = set(_parse_ints(args.focus_frames))
    if focus:
        by_frame = {int(r["frame"]): r for r in traj_rows}
        for frame in sorted(focus):
            row = by_frame.get(int(frame))
            if row is not None and int(frame) not in selected_frames:
                row = dict(row)
                row["selection_reason"] = "focus_frame"
                selected.append(row)
                selected_frames.add(int(frame))
    _write_csv(args.out_dir / "per_frame_error_delta.csv", traj_rows)
    diag_rows = [] if bool(args.skip_frame_semantic) else [_frame_semantic_row(args, row) for row in selected]
    _write_csv(args.out_dir / "selected_geometry_ttt_semantic_rows.csv", diag_rows)
    summary = {
        "schema": "acl2_v80_hs_geometry_ttt_semantic_bridge_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "control": args.control_name if args.control_run_dir else None,
        "args": vars(args),
        "trajectory_frame_rows": len(traj_rows),
        "selected_frames": selected,
        "aggregate": _aggregate(diag_rows),
        "rows": diag_rows,
        "plot_paths": plot_paths,
        "outputs": {
            "per_frame_error_delta_csv": str(args.out_dir / "per_frame_error_delta.csv"),
            "selected_geometry_ttt_semantic_rows_csv": str(args.out_dir / "selected_geometry_ttt_semantic_rows.csv"),
            "summary_json": str(args.out_dir / "geometry_ttt_semantic_bridge_summary.json"),
            "plots_dir": str(args.out_dir / "plots"),
            "visual_panels_dir": str(args.out_dir / "visual_panels"),
        },
    }
    _write_json(args.out_dir / "geometry_ttt_semantic_bridge_summary.json", summary)
    print(json.dumps(_jsonable({"aggregate": summary["aggregate"], "summary": summary["outputs"]["summary_json"]}), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
