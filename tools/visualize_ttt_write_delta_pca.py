#!/usr/bin/env python3
"""Visualize projected TTT write-delta maps with raw frame references.

The v68 dump stores layer/branch scalar write-delta norms projected by the
spatial write prior.  This script PCA-colors that row vector per patch.  It is
not a raw per-token fast-weight-gradient visualization, and the report says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps


def _parse_frames(text: str) -> List[int]:
    frames: List[int] = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        frames.append(int(item))
    return frames


def _load_payloads(dump_dir: Path) -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    for path in sorted(dump_dir.glob("chunk_*_ttt_spatial_post_delta_map.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _find_payload(payloads: Iterable[Dict[str, object]], frame: int) -> Tuple[Dict[str, object], int]:
    for payload in payloads:
        start = int(payload.get("start_frame", 0))
        end = int(payload.get("end_frame", start))
        if start <= frame < end:
            return payload, frame - start
    raise FileNotFoundError(f"No TTT spatial payload covers frame {frame}")


def _robust01(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(x[finite], lo_q))
    hi = float(np.percentile(x[finite], hi_q))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _pca_rgb(row_hw: torch.Tensor) -> Image.Image:
    arr = row_hw.detach().cpu().float().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected [row,H,W], got {arr.shape}")
    rows, height, width = arr.shape
    x = arr.reshape(rows, height * width).T
    x = x - x.mean(axis=0, keepdims=True)
    if min(x.shape) < 3:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        comps = x @ vt[:3].T
        rgb = np.stack([_robust01(comps[:, i]) for i in range(3)], axis=-1).reshape(height, width, 3)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _heat_image(hw: torch.Tensor, cmap_name: str = "magma") -> Image.Image:
    arr = hw.detach().cpu().float().numpy()
    arr = _robust01(arr)
    cmap = colormaps.get_cmap(cmap_name)
    rgba = cmap(arr)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _resize_panel(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pad = 6
    bbox = draw.textbbox((pad, pad), text)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return out


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    vals = x.detach().cpu().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"mean": float("nan"), "q90": float("nan"), "max": float("nan")}
    return {
        "mean": float(vals.mean().item()),
        "q90": float(torch.quantile(vals, 0.90).item()),
        "max": float(vals.max().item()),
    }


def _select_tensor(payload: Dict[str, object], key: str, local_frame: int) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Missing tensor {key}")
    if value.ndim == 4:
        return value[:, local_frame].detach().cpu().float()
    if value.ndim == 3:
        return value[local_frame].detach().cpu().float()
    raise ValueError(f"Unsupported tensor shape for {key}: {tuple(value.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", default="0,31,60,89,95")
    parser.add_argument("--delta-key", default="action_delta_norm_projection_patch")
    parser.add_argument("--panel-width", type=int, default=420)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = _load_payloads(dump_dir)
    if not payloads:
        raise FileNotFoundError(f"No TTT spatial payloads found in {dump_dir}")

    frames = _parse_frames(args.frames)
    records: List[Dict[str, object]] = []
    panel_w = int(args.panel_width)
    panel_h = max(1, int(round(panel_w * 266 / 924)))
    panel_size = (panel_w, panel_h)

    for frame in frames:
        payload, local = _find_payload(payloads, frame)
        raw_path = image_dir / f"{frame:06d}.png"
        raw = Image.open(raw_path).convert("RGB")
        delta_rows = _select_tensor(payload, args.delta_key, local)
        delta_pca = _pca_rgb(delta_rows)
        delta_mag = _heat_image(delta_rows.mean(dim=0))
        d_tok = _heat_image(_select_tensor(payload, "D_tok_patch", local), "viridis")
        p_ttt = _heat_image(_select_tensor(payload, "ttt_write_prior_patch", local), "plasma")

        panels = [
            _label(_resize_panel(raw, panel_size), f"raw KITTI01 f{frame:06d}"),
            _label(_resize_panel(delta_pca, panel_size), "TTT write-delta PCA"),
            _label(_resize_panel(delta_mag, panel_size), "action delta mean"),
            _label(_resize_panel(d_tok, panel_size), "D_tok risk"),
            _label(_resize_panel(p_ttt, panel_size), "P_ttt_write"),
        ]
        canvas = Image.new("RGB", (panel_w * len(panels), panel_h), (255, 255, 255))
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * panel_w, 0))
        out_path = out_dir / f"ttt_write_delta_pca_f{frame:06d}.png"
        canvas.save(out_path)
        records.append({
            "frame": int(frame),
            "chunk": int(payload.get("chunk_idx", -1)),
            "local_frame": int(local),
            "payload": str(payload.get("_path", "")),
            "image": str(out_path),
            "delta_key": str(args.delta_key),
            "delta_stats": _tensor_stats(delta_rows),
            "D_tok_stats": _tensor_stats(_select_tensor(payload, "D_tok_patch", local)),
            "P_ttt_write_stats": _tensor_stats(_select_tensor(payload, "ttt_write_prior_patch", local)),
        })

    summary_path = out_dir / "ttt_write_delta_visual_summary.json"
    summary_path.write_text(json.dumps({
        "schema": "acl2_v76_ttt_write_delta_visual_summary_v1",
        "dump_dir": str(dump_dir),
        "image_dir": str(image_dir),
        "delta_key": str(args.delta_key),
        "projection_not_raw_per_token_fast_weight_delta": True,
        "records": records,
    }, indent=2), encoding="utf-8")

    md = [
        "# TTT Write-Delta PCA Visual Confirmation",
        "",
        "This diagnostic visualizes `action_delta_norm_projection_patch` from the v68 TTT spatial post-delta dump.",
        "It is a layer/branch write-delta projection over `P_ttt_write`, not a raw per-token fast-weight gradient.",
        "",
        "| frame | chunk | local | image | delta mean | delta q90 | D_tok mean | P_ttt mean |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for rec in records:
        delta = rec["delta_stats"]
        d_tok_stats = rec["D_tok_stats"]
        p_stats = rec["P_ttt_write_stats"]
        rel = Path(str(rec["image"])).name
        md.append(
            f"| {rec['frame']} | {rec['chunk']} | {rec['local_frame']} | [{rel}]({rel}) | "
            f"{delta['mean']:.6f} | {delta['q90']:.6f} | "
            f"{d_tok_stats['mean']:.6f} | {p_stats['mean']:.6f} |"
        )
    (out_dir / "ttt_write_delta_visual_confirmation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "frames": frames, "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
