#!/usr/bin/env python3
"""Offline short Q/K-to-TTT carrier alignment diagnostic for v80 HS artifacts.

This script is diagnostic-only. It compares PCA-energy top regions from short
frame/global Q/K taps against READ q90, TTT action/world-delta maps, TTT roles,
and Stage-C semantic labels. PCA Q/K top-overlap is a proxy visualization, not
a true attention dot-product measurement.
"""

from __future__ import annotations

import argparse
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
except Exception:  # pragma: no cover - plotting is optional.
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_v80_hs_geometry_ttt_semantic_bridge import (  # noqa: E402
    _jsonable,
    _label_counts,
    _label_grid,
    _load_pt,
    _role_fractions,
    _stage_c_payload,
    _write_csv,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qk-root", type=Path, required=True)
    parser.add_argument("--ttt-run-dir", type=Path, required=True)
    parser.add_argument("--stage-c-cache-dir", type=Path, required=True)
    parser.add_argument("--bridge-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks", default="138,139")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--seq", default="00")
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text or "").replace(";", ",").split(",") if x.strip()]


def _top_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    values = x.detach().cpu().float()
    flat = values.reshape(values.shape[0], -1) if values.ndim == 3 else values.flatten()[None, :]
    k = max(1, int(math.ceil(float(frac) * int(flat.shape[-1]))))
    idx = torch.topk(flat, k=min(k, int(flat.shape[-1])), dim=-1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape_as(values)


def _mean_first_dim(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().cpu().float()
    while y.ndim > 3:
        y = y.mean(dim=0)
    return y


def _pca_energy(payload: dict[str, Any], key: str) -> tuple[torch.Tensor | None, list[int]]:
    tensor = payload.get(f"tap::{key}")
    layer_ids = payload.get(f"layer_ids::{key}")
    ids = [int(v) for v in layer_ids.detach().cpu().tolist()] if torch.is_tensor(layer_ids) else []
    if not torch.is_tensor(tensor):
        return None, ids
    x = tensor.detach().cpu().float()
    if x.ndim != 5:
        raise ValueError(f"expected T,L,H,W,C for {key}, got {tuple(x.shape)}")
    return torch.linalg.vector_norm(x, dim=-1).mean(dim=1), ids


def _mask_overlap(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.detach().cpu().bool()
    bb = b.detach().cpu().bool()
    denom = int(bb.sum().item())
    if denom <= 0:
        return None
    return float((aa & bb).float().sum().item() / denom)


def _mask_mass(mask: torch.Tensor) -> float:
    return float(mask.detach().cpu().bool().float().mean().item())


def _selected_frame_set(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("selected_frames", data.get("rows", []))
    return {int(row["frame"]) for row in rows if isinstance(row, dict) and "frame" in row}


def _global_frame(chunk: int, local: int, chunk_size: int, overlap: int) -> int:
    return int(chunk) * (int(chunk_size) - int(overlap)) + int(local)


def _read_q90_from(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    read = _load_pt(path)
    tensors = read.get("tensors") if isinstance(read.get("tensors"), dict) else {}
    q90 = tensors.get("read_active_q90_patch")
    return q90.detach().cpu().bool() if torch.is_tensor(q90) else None


def _plot_panel(
    out_dir: Path,
    *,
    frame: int,
    chunk: int,
    local: int,
    maps: dict[str, torch.Tensor],
) -> str | None:
    if plt is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("frame Q", maps["frame_q_energy"]),
        ("frame K", maps["frame_k_energy"]),
        ("global Q", maps["global_q_energy"]),
        ("global K", maps["global_k_energy"]),
        ("READ q90", maps["read_q90"].float()),
        ("TTT action", maps["action"]),
        ("world delta", maps["world"]),
        ("TTT role", torch.round(maps["role"].float())),
        ("frame Q top", maps["frame_q_top"].float()),
        ("frame K top", maps["frame_k_top"].float()),
        ("global Q top", maps["global_q_top"].float()),
        ("global K top", maps["global_k_top"].float()),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for ax, (title, image) in zip(axes.flatten(), panels):
        cmap = "tab10" if title == "TTT role" else ("gray_r" if "top" in title or "READ" in title else "viridis")
        kwargs = {"vmin": 0, "vmax": 4} if title == "TTT role" else {}
        im = ax.imshow(image.detach().cpu().numpy(), cmap=cmap, interpolation="nearest", **kwargs)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if title not in {"READ q90", "TTT role"} and "top" not in title:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"frame {frame} chunk {chunk} local {local}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"frame_{frame:06d}_chunk_{chunk:03d}_short_qk_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _analyze_chunk(args: argparse.Namespace, chunk: int, selected_frames: set[int]) -> list[dict[str, Any]]:
    pca_path = args.qk_root / f"chunk{chunk:03d}" / "pca_features" / f"chunk_{chunk:03d}.pt"
    read_path = args.qk_root / f"chunk{chunk:03d}" / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt"
    read_fallback = args.ttt_run_dir / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt"
    ttt_path = args.ttt_run_dir / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt"
    pca = _load_pt(pca_path)
    ttt = _load_pt(ttt_path)
    read_q90 = _read_q90_from(read_path)
    if read_q90 is None:
        read_q90 = _read_q90_from(read_fallback)
    if not torch.is_tensor(read_q90):
        raise RuntimeError(f"missing read_active_q90_patch for chunk {chunk}: {read_path} or {read_fallback}")

    frame_q, frame_q_layers = _pca_energy(pca, "pca_attn_frame_q_layers")
    frame_k, frame_k_layers = _pca_energy(pca, "pca_attn_frame_k_layers")
    global_q, global_q_layers = _pca_energy(pca, "pca_attn_global_q_layers")
    global_k, global_k_layers = _pca_energy(pca, "pca_attn_global_k_layers")
    required = {
        "frame_q": frame_q,
        "frame_k": frame_k,
        "global_q": global_q,
        "global_k": global_k,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"missing Q/K PCA tensors for chunk {chunk}: {missing}")

    role = ttt.get("R_ttt_tok_patch")
    action = ttt.get("action_delta_norm_projection_patch")
    world = ttt.get("pass1_pass2_world_points_l2_patch")
    if not (torch.is_tensor(role) and torch.is_tensor(action) and torch.is_tensor(world)):
        raise RuntimeError(f"missing TTT maps for chunk {chunk}: {ttt_path}")
    role = role.detach().cpu().float()
    action = _mean_first_dim(action)
    world = world.detach().cpu().float()
    t = min(
        int(frame_q.shape[0]),
        int(frame_k.shape[0]),
        int(global_q.shape[0]),
        int(global_k.shape[0]),
        int(read_q90.shape[0]),
        int(role.shape[0]),
        int(action.shape[0]),
        int(world.shape[0]),
    )
    frame_q = frame_q[:t]
    frame_k = frame_k[:t]
    global_q = global_q[:t]
    global_k = global_k[:t]
    read_q90 = read_q90[:t]
    role = role[:t]
    action = action[:t]
    world = world[:t]

    masks = {
        "frame_q_top": _top_mask(frame_q, float(args.top_frac)),
        "frame_k_top": _top_mask(frame_k, float(args.top_frac)),
        "global_q_top": _top_mask(global_q, float(args.top_frac)),
        "global_k_top": _top_mask(global_k, float(args.top_frac)),
        "action_top": _top_mask(action, float(args.top_frac)),
        "world_top": _top_mask(world, float(args.top_frac)),
    }
    positive = torch.round(role).long() == 1
    negative = torch.round(role).long() == 3
    stage_c = _stage_c_payload(args.stage_c_cache_dir, chunk)

    rows: list[dict[str, Any]] = []
    for local in range(t):
        frame = _global_frame(chunk, local, int(args.chunk_size), int(args.chunk_overlap))
        is_selected = frame in selected_frames
        labels, names = _label_grid(stage_c, local, int(role.shape[1]), int(role.shape[2]))
        local_masks = {name: tensor[local] for name, tensor in masks.items()}
        local_masks["read_q90"] = read_q90[local]
        local_masks["positive"] = positive[local]
        local_masks["negative"] = negative[local]
        row: dict[str, Any] = {
            "chunk": int(chunk),
            "local_frame": int(local),
            "frame": int(frame),
            "selected_bad_frame": bool(is_selected),
            "pca_path": str(pca_path),
            "ttt_path": str(ttt_path),
            "read_path": str(read_path if read_path.exists() else read_fallback),
            "frame_q_layers": frame_q_layers,
            "frame_k_layers": frame_k_layers,
            "global_q_layers": global_q_layers,
            "global_k_layers": global_k_layers,
            "read_q90_mass": _mask_mass(local_masks["read_q90"]),
            "frame_q_top_mass": _mask_mass(local_masks["frame_q_top"]),
            "frame_k_top_mass": _mask_mass(local_masks["frame_k_top"]),
            "global_q_top_mass": _mask_mass(local_masks["global_q_top"]),
            "global_k_top_mass": _mask_mass(local_masks["global_k_top"]),
        }
        for prefix in ["frame_q", "frame_k", "global_q", "global_k"]:
            top = local_masks[f"{prefix}_top"]
            row[f"read_q90_given_{prefix}_top"] = _mask_overlap(local_masks["read_q90"], top)
            row[f"action_top_given_{prefix}_top"] = _mask_overlap(local_masks["action_top"], top)
            row[f"world_top_given_{prefix}_top"] = _mask_overlap(local_masks["world_top"], top)
            row[f"positive_given_{prefix}_top"] = _mask_overlap(local_masks["positive"], top)
            row[f"negative_given_{prefix}_top"] = _mask_overlap(local_masks["negative"], top)
            row[f"{prefix}_top_given_read_q90"] = _mask_overlap(top, local_masks["read_q90"])
            row[f"{prefix}_top_given_action_top"] = _mask_overlap(top, local_masks["action_top"])
            row[f"{prefix}_top_given_world_top"] = _mask_overlap(top, local_masks["world_top"])
            row[f"{prefix}_top_role_fractions"] = _role_fractions(role[local], top)
            row[f"{prefix}_top_labels"] = _label_counts(labels, top, names)

        row["frame_q_top_given_frame_k_top"] = _mask_overlap(local_masks["frame_q_top"], local_masks["frame_k_top"])
        row["frame_k_top_given_frame_q_top"] = _mask_overlap(local_masks["frame_k_top"], local_masks["frame_q_top"])
        row["global_q_top_given_global_k_top"] = _mask_overlap(local_masks["global_q_top"], local_masks["global_k_top"])
        row["global_k_top_given_global_q_top"] = _mask_overlap(local_masks["global_k_top"], local_masks["global_q_top"])
        row["frame_q_top_given_global_q_top"] = _mask_overlap(local_masks["frame_q_top"], local_masks["global_q_top"])
        row["frame_k_top_given_global_k_top"] = _mask_overlap(local_masks["frame_k_top"], local_masks["global_k_top"])

        if is_selected:
            panel = _plot_panel(
                args.out_dir / "visual_panels",
                frame=frame,
                chunk=chunk,
                local=local,
                maps={
                    "frame_q_energy": frame_q[local],
                    "frame_k_energy": frame_k[local],
                    "global_q_energy": global_q[local],
                    "global_k_energy": global_k[local],
                    "read_q90": read_q90[local],
                    "action": action[local],
                    "world": world[local],
                    "role": role[local],
                    "frame_q_top": local_masks["frame_q_top"],
                    "frame_k_top": local_masks["frame_k_top"],
                    "global_q_top": local_masks["global_q_top"],
                    "global_k_top": local_masks["global_k_top"],
                },
            )
            row["visual_panel_png"] = panel
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "frame_q_top_given_frame_k_top",
        "frame_k_top_given_frame_q_top",
        "global_q_top_given_global_k_top",
        "global_k_top_given_global_q_top",
        "frame_q_top_given_global_q_top",
        "frame_k_top_given_global_k_top",
    ]
    for prefix in ["frame_q", "frame_k", "global_q", "global_k"]:
        keys.extend(
            [
                f"read_q90_given_{prefix}_top",
                f"action_top_given_{prefix}_top",
                f"world_top_given_{prefix}_top",
                f"positive_given_{prefix}_top",
                f"negative_given_{prefix}_top",
                f"{prefix}_top_given_read_q90",
                f"{prefix}_top_given_action_top",
                f"{prefix}_top_given_world_top",
            ]
        )
    out: dict[str, Any] = {
        "rows": len(rows),
        "selected_bad_rows": sum(1 for row in rows if row.get("selected_bad_frame")),
        "qk_overlap_is_proxy": True,
    }
    for scope_name, scope_rows in [("all", rows), ("selected_bad", [r for r in rows if r.get("selected_bad_frame")])]:
        for key in keys:
            vals = [float(r[key]) for r in scope_rows if r.get(key) is not None]
            out[f"{scope_name}_{key}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{scope_name}_{key}_max"] = float(np.max(vals)) if vals else None
    return out


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_frames = _selected_frame_set(args.bridge_summary)
    rows: list[dict[str, Any]] = []
    for chunk in _parse_ints(args.chunks):
        rows.extend(_analyze_chunk(args, chunk, selected_frames))
    _write_csv(args.out_dir / "short_qk_carrier_alignment_rows.csv", rows)
    selected_rows = [row for row in rows if row.get("selected_bad_frame")]
    _write_csv(args.out_dir / "short_qk_carrier_alignment_selected_bad_rows.csv", selected_rows)
    summary = {
        "schema": "acl2_v80_short_qk_carrier_alignment_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "qk_overlap_is_proxy": True,
        "args": vars(args),
        "aggregate": _aggregate(rows),
        "selected_frames": sorted(int(v) for v in selected_frames),
        "outputs": {
            "rows_csv": str(args.out_dir / "short_qk_carrier_alignment_rows.csv"),
            "selected_bad_rows_csv": str(args.out_dir / "short_qk_carrier_alignment_selected_bad_rows.csv"),
            "summary_json": str(args.out_dir / "short_qk_carrier_alignment_summary.json"),
            "visual_panels_dir": str(args.out_dir / "visual_panels"),
        },
    }
    _write_json(args.out_dir / "short_qk_carrier_alignment_summary.json", summary)
    print(
        json.dumps(
            _jsonable({"aggregate": summary["aggregate"], "summary": summary["outputs"]["summary_json"]}),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
