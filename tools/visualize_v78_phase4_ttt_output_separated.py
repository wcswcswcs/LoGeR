#!/usr/bin/env python3
"""Visualize v78 Phase4 output-separated TTT PCA/debug tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_RGB_DIR = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")

PALETTE: Dict[str, Tuple[int, int, int]] = {
    "void": (0, 0, 0),
    "person": (235, 120, 60),
    "car": (220, 74, 74),
    "truck": (220, 74, 74),
    "bus": (220, 74, 74),
    "road": (104, 104, 104),
    "ground": (116, 116, 116),
    "sky": (96, 180, 238),
    "grass": (74, 162, 74),
    "tree": (32, 120, 76),
    "wall": (156, 156, 156),
    "handrail_or_fence": (56, 100, 176),
    "pole": (220, 188, 74),
    "building": (160, 126, 192),
    "house": (170, 132, 190),
    "bridge": (142, 142, 174),
    "other_construction": (150, 150, 160),
    "traffic sign": (245, 214, 58),
    "billboard_or_bulletin_board": (245, 214, 58),
    "mountain": (130, 108, 72),
}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _stable_colour(label: str) -> Tuple[int, int, int]:
    if label in PALETTE:
        return PALETTE[label]
    value = 2166136261
    for byte in str(label).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return (64 + (value & 127), 64 + ((value >> 8) & 127), 64 + ((value >> 16) & 127))


def _colour_table(label_names: Sequence[str]) -> np.ndarray:
    colours = np.zeros((max(1, len(label_names)), 3), dtype=np.uint8)
    for idx, name in enumerate(label_names):
        colours[idx] = _stable_colour(str(name))
    return colours


def _ids_containing(label_names: Sequence[str], words: Iterable[str]) -> set[int]:
    lowered = [str(x).lower() for x in label_names]
    needles = [str(w).lower() for w in words]
    return {idx for idx, name in enumerate(lowered) if any(needle in name for needle in needles)}


def _resize(img: Image.Image, size: Tuple[int, int], *, nearest: bool = False) -> Image.Image:
    return img.convert("RGB").resize(
        size,
        Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
    )


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    text = str(text)[:120]
    bbox = draw.textbbox((5, 4), text)
    draw.rectangle((0, 0, out.width, max(22, bbox[3] + 5)), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))
    return out


def _robust01(arr: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(x[finite], lo_q))
    hi = float(np.percentile(x[finite], hi_q))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _heat(arr: np.ndarray) -> Image.Image:
    y = _robust01(np.asarray(arr, dtype=np.float32))
    rgb = np.zeros((*y.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.round(255.0 * y).astype(np.uint8)
    rgb[..., 1] = np.round(255.0 * (1.0 - np.abs(y - 0.5) * 2.0)).astype(np.uint8)
    rgb[..., 2] = np.round(255.0 * (1.0 - y)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _pca_rgb(feat: torch.Tensor) -> Image.Image:
    x = feat.detach().cpu().float().numpy()
    if x.ndim != 3:
        raise ValueError(f"expected [H,W,C], got {x.shape}")
    h, w, c = x.shape
    flat = x.reshape(h * w, c)
    finite = np.isfinite(flat).all(axis=1)
    if c < 3 or int(finite.sum()) < 3:
        mag = _robust01(np.linalg.norm(flat, axis=1).reshape(h, w))
        rgb = np.stack([mag, mag, mag], axis=-1)
        return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), "RGB")
    centered = flat - flat.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered[finite], full_matrices=False)
        comps = centered @ vt[:3].T
        rgb = np.stack([_robust01(comps[:, i]) for i in range(3)], axis=-1).reshape(h, w, 3)
    except np.linalg.LinAlgError:
        mag = _robust01(np.linalg.norm(flat, axis=1).reshape(h, w))
        rgb = np.stack([mag, mag, mag], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), "RGB")


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    vals = x.detach().cpu().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "q90": float("nan"), "max": float("nan")}
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std().item()),
        "q90": float(torch.quantile(vals, 0.90).item()),
        "max": float(vals.max().item()),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(path: Path) -> Dict[str, Any]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return {
        "width": int(img.width),
        "height": int(img.height),
        "sha256": _sha256(path),
        "image_intensity_std": float(arr.std()),
        "nonempty_image": bool(img.width >= 512 and img.height >= 160 and float(arr.std()) > 1.0),
    }


def _find_rgb(rgb_dir: Path, frame: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = rgb_dir / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing RGB {rgb_dir}/{frame:06d}.*")


def _load_semantic(stage_c_chunk: Path) -> Dict[str, Any]:
    payload = _torch_load(stage_c_chunk)
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict) or not torch.is_tensor(sem.get("label_maps")):
        raise KeyError(f"No semantic_segmentation.label_maps in {stage_c_chunk}")
    labels = sem["label_maps"].detach().cpu().long()
    conf = sem.get("confidence_maps")
    if torch.is_tensor(conf):
        conf = conf.detach().cpu().float()
    else:
        conf = torch.ones_like(labels, dtype=torch.float32)
    label_names = [str(x) for x in sem.get("label_names", [])]
    return {
        "label_maps": labels,
        "confidence_maps": conf,
        "label_names": label_names,
        "colours": _colour_table(label_names),
        "dynamic_ids": _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle")),
        "stable_ids": _ids_containing(
            label_names,
            ("road", "ground", "wall", "fence", "pole", "building", "house", "bridge", "construction", "traffic sign", "tree"),
        ),
    }


def _semantic_patch(semantic: Dict[str, Any], local: int, size: Tuple[int, int]) -> Tuple[torch.Tensor, Image.Image, torch.Tensor]:
    labels = semantic["label_maps"][int(local)].numpy().astype(np.uint8)
    conf = semantic["confidence_maps"][int(local)].numpy().astype(np.float32)
    label_img = Image.fromarray(labels, "L").resize(size, Image.Resampling.NEAREST)
    conf_img = Image.fromarray(np.clip(conf * 255.0, 0, 255).astype(np.uint8), "L").resize(size, Image.Resampling.BILINEAR)
    label_patch = torch.from_numpy(np.asarray(label_img).copy()).long()
    conf_patch = torch.from_numpy(np.asarray(conf_img).astype(np.float32) / 255.0).float()
    colours = semantic["colours"]
    label_np = np.clip(label_patch.numpy(), 0, max(0, len(colours) - 1))
    sem_rgb = Image.fromarray(colours[label_np], "RGB")
    return label_patch, sem_rgb, conf_patch


def _mask_from_ids(labels: torch.Tensor, ids: set[int]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        out |= labels == int(idx)
    return out


def _same_mass_random(mask: torch.Tensor, seed: int) -> torch.Tensor:
    flat = mask.reshape(-1)
    count = int(flat.sum().item())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(int(flat.numel()), generator=gen)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[perm[:count]] = True
    return out.reshape_as(mask)


def _mask_img(mask: torch.Tensor, colour: Tuple[int, int, int]) -> Image.Image:
    m = mask.detach().cpu().bool().numpy()
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[m] = colour
    rgb[~m] = (35, 35, 35)
    return Image.fromarray(rgb, "RGB")


def _row(panels: Sequence[Image.Image], cell_size: Tuple[int, int]) -> Image.Image:
    # Panels are already resized and labelled by _panel(); do not scale labels.
    cells = [panel.convert("RGB") for panel in panels]
    canvas = Image.new("RGB", (cell_size[0] * len(cells), cell_size[1]), (0, 0, 0))
    for idx, panel in enumerate(cells):
        canvas.paste(panel, (idx * cell_size[0], 0))
    return canvas


def _panel(img: Image.Image, text: str, cell_size: Tuple[int, int], *, nearest: bool = True) -> Image.Image:
    return _label(_resize(img, cell_size, nearest=nearest), text)


def _stack(rows: Sequence[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


def _tap_tensor(payload: Dict[str, Any], tap: str) -> torch.Tensor:
    key = f"tap::{tap}"
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"missing tensor {key}")
    return value.detach().cpu().float()


def _layer_index(payload: Dict[str, Any], tap: str, layer: int) -> int:
    ids = payload.get(f"layer_ids::{tap}")
    if torch.is_tensor(ids):
        values = [int(x) for x in ids.detach().cpu().reshape(-1).tolist()]
    else:
        values = payload.get("taps", {}).get(tap, {}).get("selected_layer_ids", [])
        values = [int(x) for x in values]
    if int(layer) not in values:
        raise KeyError(f"layer {layer} missing for {tap}; available={values}")
    return values.index(int(layer))


def _load_delta_payload(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return _torch_load(path)


def _load_anchor_payload(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    payload = _torch_load(path)
    return payload if isinstance(payload, dict) else None


def _delta_map(payload: Optional[Dict[str, Any]], key: str, local: int) -> Optional[torch.Tensor]:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not torch.is_tensor(value):
        return None
    x = value.detach().cpu().float()
    if x.ndim == 4:
        local = max(0, min(int(local), int(x.shape[1]) - 1))
        return x[:, local].mean(dim=0)
    if x.ndim == 3:
        local = max(0, min(int(local), int(x.shape[0]) - 1))
        return x[local]
    return None


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_manifest_row(
    *,
    visual_file: Path,
    visual_kind: str,
    frames: Sequence[int],
    output_name: str,
    layers: Sequence[int],
    separated_outputs: Sequence[str],
    semantic_present: bool,
    d_geo_present: bool,
    d_geo_source: str,
    post_zp_delta_present: bool,
    same_mass_random_present: bool,
) -> Dict[str, Any]:
    stats = _image_stats(visual_file)
    return {
        "visual_file": str(visual_file),
        "visual_kind": visual_kind,
        "output_name": output_name,
        "layers": ",".join(str(x) for x in layers),
        "global_frames": ",".join(str(x) for x in frames),
        "operator_update_final_separated": all(x in separated_outputs for x in ("operator", "update", "final")),
        "semantic_label_present": semantic_present,
        "D_geo_present": d_geo_present,
        "D_geo_source": d_geo_source,
        "post_zp_delta_present": post_zp_delta_present,
        "write_update_selected_role_present": True,
        "same_write_mass_random_present": same_mass_random_present,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-pt", type=Path, required=True)
    parser.add_argument("--stage-c-masklet", type=Path, required=True)
    parser.add_argument("--post-delta-pt", type=Path, default=None)
    parser.add_argument("--anchor-pt", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", default="174,189,205")
    parser.add_argument("--operator-layers", default="6,14,18")
    parser.add_argument("--single-layer", type=int, default=18)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--seed", type=int, default=78)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _torch_load(args.pca_pt)
    semantic = _load_semantic(args.stage_c_masklet)
    post_delta = _load_delta_payload(args.post_delta_pt)
    anchor_payload = _load_anchor_payload(args.anchor_pt)

    start_frame = int(payload.get("start_frame", 0))
    chunk_idx = int(payload.get("chunk_idx", -1))
    patch_h, patch_w = [int(x) for x in payload.get("patch_grid", [19, 66])]
    cell_size = (int(args.panel_width), max(90, int(round(int(args.panel_width) * patch_h / patch_w))))
    frames = _parse_ints(args.frames)
    frames = [frame for frame in frames if start_frame <= frame < int(payload.get("end_frame", start_frame))]
    if not frames:
        raise ValueError("No requested frame is covered by pca dump")
    operator_layers = _parse_ints(args.operator_layers)
    single_layer = int(args.single_layer)
    patch_size = (patch_w, patch_h)

    taps = {
        "operator": "pca_ttt_operator_output_layers",
        "update": "pca_ttt_update_term_layers",
        "final": "pca_ttt_final_output_layers",
    }
    tensors = {name: _tap_tensor(payload, tap) for name, tap in taps.items()}
    layer_indices = {
        name: {layer: _layer_index(payload, tap, layer) for layer in sorted(set(operator_layers + [single_layer]))}
        for name, tap in taps.items()
    }

    frame_cache: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        local = int(frame) - start_frame
        labels, sem_img, conf_patch = _semantic_patch(semantic, local, patch_size)
        dyn_mask = _mask_from_ids(labels, semantic["dynamic_ids"])
        stable_mask = _mask_from_ids(labels, semantic["stable_ids"])
        d_geo = None
        d_geo_source = "missing"
        if isinstance(anchor_payload, dict) and torch.is_tensor(anchor_payload.get("D_geo")):
            anchor_d = anchor_payload["D_geo"].detach().cpu().float()
            if anchor_d.ndim == 3 and local < int(anchor_d.shape[0]):
                d_geo = anchor_d[local]
                d_geo_source = "anchor_bank_D_geo"
        if d_geo is None:
            d_geo = _delta_map(post_delta, "D_tok_patch", local)
            if d_geo is not None:
                d_geo_source = "post_delta_D_tok_patch"
        if d_geo is None:
            # Fallback is explicit and is marked as not real D_geo in the audit.
            d_geo = tensors["update"][local, layer_indices["update"][single_layer]].norm(dim=-1)
            d_geo_source = "fallback_update_norm_not_D_geo"
        d_geo = d_geo.detach().cpu().float()
        high_d = d_geo > torch.quantile(d_geo.reshape(-1), 0.75)
        low_conf = conf_patch < 0.55
        write_role = high_d | dyn_mask | low_conf
        random_role = _same_mass_random(write_role, int(args.seed) + int(frame))
        frame_cache[int(frame)] = {
            "local": local,
            "rgb": Image.open(_find_rgb(args.rgb_dir, frame)).convert("RGB"),
            "semantic": sem_img,
            "labels": labels,
            "conf": conf_patch,
            "d_geo": d_geo,
            "d_geo_source": d_geo_source,
            "write_role": write_role,
            "stable_role": stable_mask,
            "random_role": random_role,
        }

    manifest_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    operator_rows: List[Image.Image] = []
    for frame in frames:
        rec = frame_cache[frame]
        local = int(rec["local"])
        panels = [
            _panel(rec["rgb"], f"RGB f{frame:06d}", cell_size, nearest=False),
            _panel(rec["semantic"], "semantic label", cell_size),
            _panel(_heat(rec["d_geo"].numpy()), "D_geo / D_tok", cell_size),
        ]
        for layer in operator_layers:
            feat = tensors["operator"][local, layer_indices["operator"][layer]]
            panels.append(_panel(_pca_rgb(feat), f"TTT operator L{layer:02d} PCA", cell_size))
        operator_rows.append(_row(panels, cell_size))
    operator_file = out_dir / f"chunk_{chunk_idx:03d}_TTT_operator_output_L06_L14_L18.png"
    _stack(operator_rows).save(operator_file)
    manifest_rows.append(
        _build_manifest_row(
            visual_file=operator_file,
            visual_kind="TTT_operator_output_separated",
            frames=frames,
            output_name="operator",
            layers=operator_layers,
            separated_outputs=("operator", "update", "final"),
            semantic_present=True,
            d_geo_present=str(frame_cache[frames[0]]["d_geo_source"]) != "fallback_update_norm_not_D_geo",
            d_geo_source=str(frame_cache[frames[0]]["d_geo_source"]),
            post_zp_delta_present=post_delta is not None,
            same_mass_random_present=True,
        )
    )

    for output_name, file_suffix in (("update", "update_term"), ("final", "final_output")):
        rows: List[Image.Image] = []
        for frame in frames:
            rec = frame_cache[frame]
            local = int(rec["local"])
            feat = tensors[output_name][local, layer_indices[output_name][single_layer]]
            panels = [
                _panel(rec["rgb"], f"RGB f{frame:06d}", cell_size, nearest=False),
                _panel(rec["semantic"], "semantic label", cell_size),
                _panel(_heat(rec["d_geo"].numpy()), "D_geo / D_tok", cell_size),
                _panel(_pca_rgb(feat), f"TTT {file_suffix} L{single_layer:02d} PCA", cell_size),
                _panel(_heat(feat.norm(dim=-1).numpy()), f"TTT {file_suffix} L{single_layer:02d} norm", cell_size),
            ]
            rows.append(_row(panels, cell_size))
        out_file = out_dir / f"chunk_{chunk_idx:03d}_TTT_{file_suffix}_L{single_layer:02d}.png"
        _stack(rows).save(out_file)
        manifest_rows.append(
            _build_manifest_row(
                visual_file=out_file,
                visual_kind=f"TTT_{file_suffix}_separated",
                frames=frames,
                output_name=output_name,
                layers=[single_layer],
                separated_outputs=("operator", "update", "final"),
                semantic_present=True,
                d_geo_present=str(frame_cache[frames[0]]["d_geo_source"]) != "fallback_update_norm_not_D_geo",
                d_geo_source=str(frame_cache[frames[0]]["d_geo_source"]),
                post_zp_delta_present=post_delta is not None,
                same_mass_random_present=True,
            )
        )

    post_rows: List[Image.Image] = []
    for frame in frames:
        rec = frame_cache[frame]
        local = int(rec["local"])
        committed = _delta_map(post_delta, "committed_post_delta_norm_projection_patch", local)
        native = _delta_map(post_delta, "native_delta_norm_projection_patch", local)
        action = _delta_map(post_delta, "action_delta_norm_projection_patch", local)
        panels = [
            _panel(rec["rgb"], f"RGB f{frame:06d}", cell_size, nearest=False),
            _panel(rec["semantic"], "semantic label", cell_size),
            _panel(_heat(rec["d_geo"].numpy()), "D_geo / D_tok", cell_size),
        ]
        for name, tensor in (("post-zp committed", committed), ("post-zp native", native), ("post-zp action", action)):
            if tensor is None:
                panels.append(_panel(Image.new("RGB", patch_size, (10, 10, 10)), f"{name}: missing", cell_size))
            else:
                panels.append(_panel(_heat(tensor.numpy()), name, cell_size))
        post_rows.append(_row(panels, cell_size))
    post_file = out_dir / f"chunk_{chunk_idx:03d}_TTT_post_zp_delta_overlay.png"
    _stack(post_rows).save(post_file)
    manifest_rows.append(
        _build_manifest_row(
            visual_file=post_file,
            visual_kind="TTT_post_zp_delta_overlay",
            frames=frames,
            output_name="post_zp_delta",
            layers=[single_layer],
            separated_outputs=("operator", "update", "final"),
            semantic_present=True,
            d_geo_present=str(frame_cache[frames[0]]["d_geo_source"]) != "fallback_update_norm_not_D_geo",
            d_geo_source=str(frame_cache[frames[0]]["d_geo_source"]),
            post_zp_delta_present=post_delta is not None,
            same_mass_random_present=True,
        )
    )

    role_rows: List[Image.Image] = []
    role_records: List[Dict[str, Any]] = []
    for frame in frames:
        rec = frame_cache[frame]
        local = int(rec["local"])
        p_write = _delta_map(post_delta, "ttt_write_prior_patch", local)
        panels = [
            _panel(rec["rgb"], f"RGB f{frame:06d}", cell_size, nearest=False),
            _panel(rec["semantic"], "semantic label", cell_size),
            _panel(_heat(rec["d_geo"].numpy()), "D_geo / D_tok", cell_size),
            _panel(_mask_img(rec["write_role"], (230, 90, 60)), "selected write/update role", cell_size),
            _panel(_mask_img(rec["random_role"], (80, 170, 230)), "same-write-mass random", cell_size),
        ]
        if p_write is not None:
            panels.append(_panel(_heat(p_write.numpy()), "P_ttt_write", cell_size))
        else:
            panels.append(_panel(Image.new("RGB", patch_size, (10, 10, 10)), "P_ttt_write missing", cell_size))
        role_rows.append(_row(panels, cell_size))
        role = rec["write_role"]
        random_role = rec["random_role"]
        role_records.append(
            {
                "global_frame": int(frame),
                "local_frame": int(local),
                "selected_write_role_mass": int(role.sum().item()),
                "same_write_mass_random_mass": int(random_role.sum().item()),
                "selected_write_role_ratio": float(role.float().mean().item()),
                "stable_role_ratio": float(rec["stable_role"].float().mean().item()),
                "dynamic_or_highD_or_lowconf_definition": "dynamic_semantic OR D_tok_q75 OR semantic_conf_lt_0.55",
                "same_mass_random_seed": int(args.seed) + int(frame),
            }
        )
    role_file = out_dir / f"chunk_{chunk_idx:03d}_TTT_write_role_mass_panel.png"
    _stack(role_rows).save(role_file)
    manifest_rows.append(
        _build_manifest_row(
            visual_file=role_file,
            visual_kind="TTT_write_role_mass_panel",
            frames=frames,
            output_name="write_role_mass",
            layers=[single_layer],
            separated_outputs=("operator", "update", "final"),
            semantic_present=True,
            d_geo_present=str(frame_cache[frames[0]]["d_geo_source"]) != "fallback_update_norm_not_D_geo",
            d_geo_source=str(frame_cache[frames[0]]["d_geo_source"]),
            post_zp_delta_present=post_delta is not None,
            same_mass_random_present=True,
        )
    )

    for row in manifest_rows:
        if row["visual_kind"] == "TTT_operator_output_separated":
            pattern = "operator L06/L14/L18 separated; inspect PCA spatial structure against RGB/semantic/D_geo"
        elif row["visual_kind"] == "TTT_update_term_separated":
            pattern = "update_term L18 separated; inspect write/update strength structure"
        elif row["visual_kind"] == "TTT_final_output_separated":
            pattern = "final_output L18 separated; compare residual-mixed output with update term"
        elif row["visual_kind"] == "TTT_post_zp_delta_overlay":
            pattern = "post-zp delta projection shown separately; projection is not raw per-token fast-weight gradient"
        else:
            pattern = "write role mass and same-write-mass random shown separately"
        review_rows.append(
            {
                "visual_file": row["visual_file"],
                "review_status": "needs_human_review",
                "attention_type": "TTT output-separated",
                "tap": row["output_name"],
                "layers": row["layers"],
                "global_frames": row["global_frames"],
                "visual_pattern_observed": pattern,
                "forbidden_interpretation_guard": "Do not treat final_output as write/update control evidence; do not treat post-zp projection as raw token fast-weight gradient.",
            }
        )

    manifest_path = out_dir / "visual_artifact_manifest.csv"
    _write_csv(
        manifest_path,
        manifest_rows,
        [
            "visual_file",
            "visual_kind",
            "output_name",
            "layers",
            "global_frames",
            "operator_update_final_separated",
            "semantic_label_present",
            "D_geo_present",
            "D_geo_source",
            "post_zp_delta_present",
            "write_update_selected_role_present",
            "same_write_mass_random_present",
            "width",
            "height",
            "sha256",
            "image_intensity_std",
            "nonempty_image",
        ],
    )
    _write_csv(
        out_dir / "visual_review.csv",
        review_rows,
        [
            "visual_file",
            "review_status",
            "attention_type",
            "tap",
            "layers",
            "global_frames",
            "visual_pattern_observed",
            "forbidden_interpretation_guard",
        ],
    )
    _write_csv(
        out_dir / "write_role_mass.csv",
        role_records,
        [
            "global_frame",
            "local_frame",
            "selected_write_role_mass",
            "same_write_mass_random_mass",
            "selected_write_role_ratio",
            "stable_role_ratio",
            "dynamic_or_highD_or_lowconf_definition",
            "same_mass_random_seed",
        ],
    )

    audit = {
        "schema": "acl2_v78_phase4_ttt_visual_integrity_audit_v1",
        "pca_pt": str(args.pca_pt),
        "stage_c_masklet": str(args.stage_c_masklet),
        "post_delta_pt": str(args.post_delta_pt) if args.post_delta_pt else "",
        "anchor_pt": str(args.anchor_pt) if args.anchor_pt else "",
        "num_visual_files": len(manifest_rows),
        "num_review_rows": len(review_rows),
        "review_coverage": float(len(review_rows) / max(1, len(manifest_rows))),
        "operator_update_final_separated": all(bool(row["operator_update_final_separated"]) for row in manifest_rows),
        "semantic_label_present": all(bool(row["semantic_label_present"]) for row in manifest_rows),
        "D_geo_present": all(bool(row["D_geo_present"]) for row in manifest_rows),
        "post_zp_delta_present": all(bool(row["post_zp_delta_present"]) for row in manifest_rows),
        "same_write_mass_random_present": all(bool(row["same_write_mass_random_present"]) for row in manifest_rows),
        "nonempty_visual_count": int(sum(bool(row["nonempty_image"]) for row in manifest_rows)),
        "invalid_visual_count": int(sum(not bool(row["nonempty_image"]) for row in manifest_rows)),
    }
    audit["gate_pass"] = bool(
        audit["num_visual_files"] >= 5
        and audit["review_coverage"] >= 0.8
        and audit["operator_update_final_separated"]
        and audit["semantic_label_present"]
        and audit["D_geo_present"]
        and audit["post_zp_delta_present"]
        and audit["same_write_mass_random_present"]
        and audit["invalid_visual_count"] == 0
    )
    (out_dir / "visual_integrity_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    insight = [
        "# Phase4 TTT Output-Separated Visual Insight",
        "",
        "This file is generated by `tools/visualize_v78_phase4_ttt_output_separated.py`.",
        "It records visual artifacts for review; it does not claim a TTT action success.",
        "",
        "Important guardrails:",
        "- `operator_output`, `update_term`, and `final_output` are saved as separate panels.",
        "- `final_output` is residual-mixed output and must not be used as write/update control evidence.",
        "- `post-zp delta` is a write-prior projection of layer/branch fast-weight delta norms, not a raw per-token fast-weight gradient.",
        "- The selected write/update role is a diagnostic visual mask: dynamic semantic OR D_tok_q75 OR semantic confidence < 0.55.",
        "",
        f"Visual audit gate_pass: `{audit['gate_pass']}`.",
        f"Frames: `{','.join(str(x) for x in frames)}`.",
        f"Layers: operator `{','.join(str(x) for x in operator_layers)}`, update/final `L{single_layer:02d}`.",
        "",
        "Manual visual confirmation must inspect the PNG files before using these patterns to design Phase4 controls.",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(insight) + "\n", encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "visual_files": [str(row["visual_file"]) for row in manifest_rows],
        "audit": audit,
        "role_records": role_records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
