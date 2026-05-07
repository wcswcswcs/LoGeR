#!/usr/bin/env python3
"""Phase-B HMC probe-trace dashboards.

This tool runs LoGeR chunks in the same committed-state order as the v2
pipeline, collects probe-pass cues, and writes representative visual
dashboards. It intentionally does not report ATE; Phase B is about whether
the control signals are interpretable before read-path experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from loger.pipeline.dynamic_cue_extractor import (
    CUE_ANCHOR,
    CUE_DYN,
    CUE_OCC,
    CUE_STAT,
    CUE_UNC,
    DynamicCueExtractor,
)
from loger.pipeline.geometry_backbone import (
    GeometryOutput,
    LoGeRGeometryBackbone,
    TTTLayerCache,
    TOKEN_TYPE_PATCH,
    load_images as loger_load_images,
)
from loger.pipeline.hybrid_memory_controller import (
    HybridMemoryController,
    HybridMemoryState,
    hybrid_state_fingerprint,
)
from loger.pipeline.semantic_prior_generator import SemanticPriorGenerator
from run_geometry_backbone_inference import collect_image_paths as collect_image_paths_geo
from run_pipeline_abc import (
    _apply_prior_policy,
    _empty_masklet_output,
    _move_tensor_tree_to_device,
    split_into_chunks,
)


def _parse_segments(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad segment spec: {part}")
        a, b = part.split(":", 1)
        start, end = int(a), int(b)
        if end <= start:
            raise ValueError(f"Bad segment range: {part}")
        out.append((start, end))
    if not out:
        raise ValueError("No segments requested.")
    return out


def _sample_frames(segments: Sequence[Tuple[int, int]], count: int) -> Dict[int, Tuple[int, int]]:
    samples: Dict[int, Tuple[int, int]] = {}
    for seg_idx, (start, end) in enumerate(segments):
        if count <= 1:
            frames = [start + (end - start) // 2]
        else:
            frames = np.linspace(start, end - 1, count, dtype=int).tolist()
        for frame_idx in frames:
            samples[int(frame_idx)] = (seg_idx, int(frame_idx))
    return samples


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    if x.numel() == 0:
        return x
    lo = torch.quantile(x.reshape(-1), 0.02)
    hi = torch.quantile(x.reshape(-1), 0.98)
    if float(hi - lo) < 1e-6:
        lo = x.min()
        hi = x.max()
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _patch_score_from_token_prior(geo: GeometryOutput, values: torch.Tensor) -> torch.Tensor:
    T = int(geo.num_frames)
    H, W = geo.patch_grid
    flat = values.detach().cpu().float().reshape(-1)
    n = T * H * W
    if flat.numel() < n:
        padded = torch.ones(n, dtype=torch.float32)
        padded[: flat.numel()] = flat
        flat = padded
    return flat[:n].reshape(T, H, W)


def _ttt_update_proxy(geo: GeometryOutput, layer_caches: Sequence[TTTLayerCache], branch: int = 0) -> torch.Tensor:
    """Cheap token-level update-needed proxy from TTT write cache.

    We do not re-run the exact fast-weight residual here. The proxy follows
    the plan's acceptable first substitute: eta-weighted update magnitude,
    aggregated over heads and selected TTT layers.
    """

    T = int(geo.num_frames)
    H, W = geo.patch_grid
    patch_mask = (geo.token_type.detach().cpu().long() == TOKEN_TYPE_PATCH)
    n_patch = T * H * W
    accum = torch.zeros(n_patch, dtype=torch.float32)
    used = 0
    for layer in layer_caches:
        lr = {0: layer.lr0, 1: layer.lr1, 2: layer.lr2}.get(branch, layer.lr0)
        lr = lr.detach().cpu().float().squeeze(-1)  # [bh, L]
        k = layer.k.detach().cpu().float()
        v = layer.v.detach().cpu().float()
        L = min(lr.shape[1], k.shape[1], v.shape[1])
        if L <= 0:
            continue
        score = lr[:, :L].abs() * torch.linalg.norm(k[:, :L], dim=-1) * torch.linalg.norm(v[:, :L], dim=-1)
        token_score = score.mean(dim=0)
        if token_score.numel() >= patch_mask.numel():
            patch_score = token_score[: patch_mask.numel()][patch_mask]
        else:
            patch_score = token_score
        if patch_score.numel() < n_patch:
            patch_score = torch.nn.functional.pad(patch_score, (0, n_patch - patch_score.numel()))
        accum += patch_score[:n_patch]
        used += 1
    if used > 0:
        accum /= float(used)
    return _safe_norm(accum.reshape(T, H, W))


def _ttt_apply_residual_proxy(geo: GeometryOutput, layer_caches: Sequence[TTTLayerCache]) -> Optional[torch.Tensor]:
    """Token-level approximation of the plan's ``||f_W(k)-v|| / ||v||`` signal."""

    T = int(geo.num_frames)
    H, W = geo.patch_grid
    n_patch = T * H * W
    patch_mask = (geo.token_type.detach().cpu().long() == TOKEN_TYPE_PATCH)
    accum = torch.zeros(n_patch, dtype=torch.float32)
    used = 0
    for layer in layer_caches:
        raw = layer.apply_output_raw
        if raw is None:
            continue
        raw = raw.detach().cpu().float()
        v = layer.v.detach().cpu().float()
        L = min(raw.shape[1], v.shape[1])
        if L <= 0:
            continue
        residual = torch.linalg.norm(raw[:, :L] - v[:, :L], dim=-1) / (
            torch.linalg.norm(v[:, :L], dim=-1) + 1e-6
        )
        token_score = residual.mean(dim=0)
        if token_score.numel() >= patch_mask.numel():
            patch_score = token_score[: patch_mask.numel()][patch_mask]
        else:
            patch_score = token_score
        if patch_score.numel() < n_patch:
            patch_score = torch.nn.functional.pad(patch_score, (0, n_patch - patch_score.numel()))
        accum += patch_score[:n_patch]
        used += 1
    if used == 0:
        return None
    return _safe_norm((accum / float(used)).reshape(T, H, W))


def _patch_summary(arr: Optional[torch.Tensor]) -> Dict[str, float]:
    if arr is None:
        return {"mean": float("nan"), "std": float("nan"), "p90": float("nan")}
    x = arr.detach().cpu().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "p90": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
    }


def _save_map_panel(ax: Any, title: str, data: Optional[torch.Tensor], *, cmap: str = "magma") -> None:
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if data is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        return
    arr = data.detach().cpu().float().numpy()
    ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)


def _save_dashboard(
    *,
    out_path: Path,
    image_path: str,
    title: str,
    maps: Dict[str, Optional[torch.Tensor]],
) -> None:
    rgb = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(4, 4, figsize=(17, 13), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=9)
    axes[0].axis("off")

    panel_order = [
        ("C_dyn", "magma"),
        ("C_occ", "magma"),
        ("C_unc", "magma"),
        ("C_anchor", "viridis"),
        ("G_write", "viridis"),
        ("attn_dyn", "magma"),
        ("dyn4d", "magma"),
        ("key_cos_l0", "magma"),
        ("key_cos_l4", "magma"),
        ("ttt_update_proxy", "magma"),
        ("ttt_apply_resid", "magma"),
        ("A_prior", "viridis"),
    ]
    for ax, (key, cmap) in zip(axes[1:], panel_order):
        _save_map_panel(ax, key, maps.get(key), cmap=cmap)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Phase-B HMC probe trace dashboards.")
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--segments", default="0:200,300:500,400:600,800:1000")
    p.add_argument("--samples_per_segment", type=int, default=6)
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--chunk_overlap", type=int, default=3)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=5)
    p.add_argument("--geometry_edge_rtol", type=float, default=0.0)
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--max_end_frame", type=int, default=1000)
    p.add_argument("--k_intra", type=int, default=5)
    p.add_argument("--sigma_pt", type=float, default=0.25)
    p.add_argument("--dyn_fusion_mode", default="calibrated_soft_or")
    p.add_argument("--implicit_weight", type=float, default=0.50)
    p.add_argument("--implicit_gate_floor", type=float, default=0.25)
    p.add_argument("--lambda_s", type=float, default=1.2)
    p.add_argument("--lambda_a", type=float, default=0.8)
    p.add_argument("--lambda_d", type=float, default=1.2)
    p.add_argument("--lambda_o", type=float, default=0.3)
    p.add_argument("--lambda_u", type=float, default=0.3)
    p.add_argument("--mp_alpha", type=float, default=0.1)
    p.add_argument("--mp_min", type=float, default=0.8)
    p.add_argument("--mp_max", type=float, default=1.2)
    p.add_argument("--mp_score_source", default="dyn")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = _parse_segments(args.segments)
    max_end = min(max(e for _, e in segments), int(args.max_end_frame))
    image_paths, _ = collect_image_paths_geo(args.input, 0, max_end, 1)
    if not image_paths:
        raise SystemExit("No images found.")
    total_frames = len(image_paths)
    chunks = split_into_chunks(total_frames, args.chunk_size, args.chunk_overlap)
    sample_map = _sample_frames(segments, args.samples_per_segment)

    with open(out_dir / "phase_b_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    target_w, target_h = args.resolution if args.resolution else (None, None)
    images_loger = loger_load_images(image_paths, target_w=target_w, target_h=target_h)

    backbone = LoGeRGeometryBackbone.from_config(
        args.checkpoint,
        args.config,
        device=args.device,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
        edge_rtol=args.geometry_edge_rtol,
        update_ttt_weights=False,
    )
    hmc = HybridMemoryController(device=args.device, lambda_min=1.0, lambda_max=1.0)
    extractor = DynamicCueExtractor(
        k_intra=args.k_intra,
        sigma_pt=args.sigma_pt,
        dyn_fusion_mode=args.dyn_fusion_mode,
        implicit_weight=args.implicit_weight,
        implicit_gate_floor=args.implicit_gate_floor,
        lambda_s=args.lambda_s,
        lambda_a=args.lambda_a,
        lambda_d=args.lambda_d,
        lambda_o=args.lambda_o,
        lambda_u=args.lambda_u,
    )
    prior_gen = SemanticPriorGenerator(
        use_g_write_geo=True,
        rho_sem=0.6,
        a_min_special=1.0,
        a_token_floor=0.0,
    )

    state = HybridMemoryState()
    trace_jsonl = out_dir / "phase_b_trace_summary.jsonl"
    chunk_csv = out_dir / "phase_b_chunk_summary.csv"
    trace_jsonl.write_text("", encoding="utf-8")
    rows: List[Dict[str, Any]] = []

    print(f"Phase-B dashboard: {total_frames} frames, {len(chunks)} chunks, out={out_dir}")
    for ci, (start, end) in enumerate(chunks):
        if args.reset_every > 0 and ci > 0 and ci % args.reset_every == 0:
            ttt = state.ttt_state or {}
            state = HybridMemoryState(ttt_state={
                "w0": [None] * len(ttt.get("w0", [])),
                "w1": [None] * len(ttt.get("w1", [])),
                "w2": [None] * len(ttt.get("w2", [])),
                "history": ttt.get("history"),
            })

        t0 = time.time()
        chunk = images_loger[start:end]
        probe = hmc.run_probe(backbone, chunk, state)
        cue = extractor.run(probe.geometry)
        mo = _empty_masklet_output(
            probe.geometry.num_frames,
            int(probe.geometry.local_points.shape[1]),
            int(probe.geometry.local_points.shape[2]),
        )
        prior = prior_gen.run(cue, mo, probe.geometry)
        prior = _apply_prior_policy(
            prior,
            cue,
            probe.geometry,
            policy="eta_mean_preserving",
            alpha=args.mp_alpha,
            p_min=args.mp_min,
            p_max=args.mp_max,
            score_source=args.mp_score_source,
        )
        ttt_proxy = _ttt_update_proxy(probe.geometry, probe.hybrid_cache.ttt_cache.layer_caches, branch=0)
        ttt_apply_resid = _ttt_apply_residual_proxy(probe.geometry, probe.hybrid_cache.ttt_cache.layer_caches)
        result = hmc.run_controlled(backbone, chunk, state, None, mode="native", token_type=probe.geometry.token_type)
        state = result.state_next
        if state.ttt_state is not None:
            state.ttt_state = _move_tensor_tree_to_device(state.ttt_state, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cue_patch = cue.E_cue_patch.detach().cpu().float()
        a_prior = _patch_score_from_token_prior(probe.geometry, prior.A_patch_flat)
        maps_for_summary = {
            "C_stat": cue_patch[..., CUE_STAT],
            "C_dyn": cue_patch[..., CUE_DYN],
            "C_occ": cue_patch[..., CUE_OCC],
            "C_unc": cue_patch[..., CUE_UNC],
            "C_anchor": cue_patch[..., CUE_ANCHOR],
            "G_write": cue.G_write_geo_patch.detach().cpu().float(),
            "attn_dyn": probe.geometry.attn_dynamic_patch.detach().cpu().float() if probe.geometry.attn_dynamic_patch is not None else None,
            "dyn4d": probe.geometry.dyn4d_patch.detach().cpu().float() if probe.geometry.dyn4d_patch is not None else None,
            "key_cos_l0": probe.geometry.frame_attn_key_cosine_l0.detach().cpu().float() if probe.geometry.frame_attn_key_cosine_l0 is not None else None,
            "key_cos_l4": probe.geometry.frame_attn_key_cosine_l4.detach().cpu().float() if probe.geometry.frame_attn_key_cosine_l4 is not None else None,
            "ttt_update_proxy": ttt_proxy,
            "ttt_apply_resid": ttt_apply_resid,
            "A_prior": a_prior,
        }

        row: Dict[str, Any] = {
            "chunk_idx": ci,
            "start": start,
            "end": end,
            "state_before_probe_hash": probe.debug.get("committed_state_hash"),
            "state_after_commit_hash": hybrid_state_fingerprint(state),
            "swa_read_importance_available": False,
            "chunk_attn_dynamic_mass_available": False,
            "wall_s": time.time() - t0,
        }
        for key, value in maps_for_summary.items():
            stats = _patch_summary(value)
            for stat_name, stat_value in stats.items():
                row[f"{key}_{stat_name}"] = stat_value
        rows.append(row)
        with trace_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

        sample_frames = [g for g in sample_map if start <= g < end]
        for global_idx in sample_frames:
            local_idx = global_idx - start
            seg_idx, _ = sample_map[global_idx]
            panel_maps = {
                key: (_safe_norm(value[local_idx]) if value is not None else None)
                for key, value in maps_for_summary.items()
            }
            dash_path = out_dir / f"segment_{seg_idx}_{segments[seg_idx][0]}_{segments[seg_idx][1]}" / f"frame_{global_idx:04d}_chunk_{ci:03d}.png"
            _save_dashboard(
                out_path=dash_path,
                image_path=image_paths[global_idx],
                title=f"KITTI01 frame {global_idx}, chunk {ci} [{start},{end})",
                maps=panel_maps,
            )

        print(f"chunk {ci:03d} [{start},{end}) done in {row['wall_s']:.1f}s, samples={len(sample_frames)}")

    if rows:
        with chunk_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    availability = {
        "available": [
            "C_stat", "C_dyn", "C_occ", "C_unc", "C_anchor", "G_write_geo",
            "attn_dynamic_patch", "dyn4d_patch", "frame_attn_key_cosine_l0",
            "frame_attn_key_cosine_l4", "ttt_update_proxy", "ttt_apply_resid", "A_prior",
        ],
        "not_available_yet": [
            "SWA previous-token read importance",
            "chunk-wise bidirectional attention dynamic mass",
        ],
        "reason": "Dense SWA/chunk attention importance maps are not exported yet; TTT apply residual is available as a token-level proxy.",
    }
    with open(out_dir / "phase_b_trace_availability.json", "w", encoding="utf-8") as f:
        json.dump(availability, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
