#!/usr/bin/env python3
"""Layer-wise PCA visualization for ACL2 v68 feature dumps.

This tool is deliberately conservative: every PCA basis is fit for exactly
one ``(tap, layer)`` pair. The generated JSON files may include auto-ranked
candidates, but they do not mark layers as method-ready unless the required
control group is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.semantic_prior_generator import (  # noqa: E402
    _mode_pool_dense_semantic_patches,
    _normalize_dense_semantic_confidence,
)


RGBPalette = Dict[str, Tuple[int, int, int]]

PALETTE: RGBPalette = {
    "void": (0, 0, 0),
    "person": (235, 120, 60),
    "car": (220, 74, 74),
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

PATH_TAPS: Dict[str, Tuple[str, ...]] = {
    "read": ("global_q_raw_patchvec_layers", "global_k_raw_patchvec_layers"),
    "swa": ("global_q_raw_patchvec_layers", "global_k_raw_patchvec_layers"),
    "ttt": ("global_q_raw_patchvec_layers", "global_k_raw_patchvec_layers"),
    "merge": ("global_k_raw_patchvec_layers", "global_q_raw_patchvec_layers"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--semantic-pt", default="", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-fit-tokens", type=int, default=30000)
    parser.add_argument("--visual-frames-per-chunk", type=int, default=3)
    parser.add_argument("--cell-scale", type=int, default=8)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument(
        "--require-control-group",
        type=int,
        default=1,
        help="Keep selected_layers empty unless enough layer controls are present.",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _stable_colour(label: str) -> Tuple[int, int, int]:
    if label in PALETTE:
        return PALETTE[label]
    value = 2166136261
    for byte in str(label).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return (
        int(64 + (value & 127)),
        int(64 + ((value >> 8) & 127)),
        int(64 + ((value >> 16) & 127)),
    )


def _colour_table(label_names: Sequence[str]) -> np.ndarray:
    colours = np.zeros((max(len(label_names), 1), 3), dtype=np.uint8)
    for idx, name in enumerate(label_names):
        colours[idx] = np.asarray(_stable_colour(str(name)), dtype=np.uint8)
    return colours


def _ids_containing(label_names: Sequence[str], words: Iterable[str]) -> List[int]:
    lowered = [str(x).lower() for x in label_names]
    keys = [str(w).lower() for w in words]
    return [idx for idx, name in enumerate(lowered) if any(k in name for k in keys)]


def _as_feature_tensor(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach().cpu().float()
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        return x.unsqueeze(-1)
    raise ValueError(f"Unsupported feature tensor shape {tuple(x.shape)}")


def _load_feature_chunks(feature_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(feature_dir.glob("chunk_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.pt files found under {feature_dir}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = _torch_load(path)
        rows.append({"path": path, "payload": payload})
    return rows


def _layer_ids_for_tap(payload: Mapping[str, Any], tap: str, tensor: torch.Tensor) -> List[int]:
    meta = dict(dict(payload.get("taps") or {}).get(tap) or {})
    selected = meta.get("selected_layers") or []
    if selected and len(selected) == int(tensor.shape[1]):
        return [int(x) for x in selected]
    return list(range(int(tensor.shape[1])))


def _sample_rows(x: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    flat = x.reshape(-1, int(x.shape[-1]))
    if flat.shape[0] <= max_rows:
        return flat
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(int(flat.shape[0]), generator=gen)[: int(max_rows)]
    return flat[idx]


def _fit_pca(samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    if samples.ndim != 2 or samples.shape[0] < 3:
        raise ValueError(f"Need a [N,D] tensor with N>=3, got {tuple(samples.shape)}")
    med = samples.median(dim=0).values
    mad = (samples - med).abs().median(dim=0).values.clamp_min(1e-6)
    z = (samples - med) / mad
    mean = z.mean(dim=0)
    z = z - mean
    _, svals, vh = torch.linalg.svd(z, full_matrices=False)
    ncomp = min(3, int(vh.shape[0]))
    comps = vh[:ncomp].T.contiguous()
    var = svals.square()
    if float(var.sum()) <= 0.0:
        ratio = torch.zeros(3, dtype=torch.float32)
    else:
        ratio = (var / var.sum())[:ncomp].float()
        if ncomp < 3:
            ratio = torch.nn.functional.pad(ratio, (0, 3 - ncomp))
    return {"median": med, "mad": mad, "mean": mean, "components": comps, "explained": ratio}


def _project_pca(x: torch.Tensor, basis: Mapping[str, torch.Tensor]) -> torch.Tensor:
    original_shape = x.shape[:-1]
    flat = x.reshape(-1, int(x.shape[-1])).float()
    z = (flat - basis["median"]) / basis["mad"] - basis["mean"]
    proj = z @ basis["components"]
    if proj.shape[1] < 3:
        proj = torch.nn.functional.pad(proj, (0, 3 - int(proj.shape[1])))
    return proj.reshape(*original_shape, 3)


def _percentile_bounds(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = z.reshape(-1, 3)
    lo = torch.quantile(flat, 0.01, dim=0)
    hi = torch.quantile(flat, 0.99, dim=0)
    close = (hi - lo).abs() < 1e-6
    if bool(close.any()):
        lo2 = flat.min(dim=0).values
        hi2 = flat.max(dim=0).values
        lo = torch.where(close, lo2, lo)
        hi = torch.where(close, hi2, hi)
    return lo, hi


def _project_to_rgb(z: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> np.ndarray:
    rgb = ((z - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return (rgb.detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def _load_semantic(semantic_pt: Path) -> Optional[Dict[str, Any]]:
    if not semantic_pt or not semantic_pt.exists():
        return None
    payload = _torch_load(semantic_pt)
    sem = payload.get("semantic_segmentation")
    if not isinstance(sem, dict):
        return None
    if not torch.is_tensor(sem.get("label_maps")):
        return None
    label_names = [str(x) for x in (sem.get("label_names") or [])]
    return {
        "label_maps": sem["label_maps"].detach().cpu().long(),
        "confidence_maps": sem.get("confidence_maps"),
        "label_names": label_names,
        "colours": _colour_table(label_names),
        "dynamic_ids": _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle")),
        "sky_ids": _ids_containing(label_names, ("sky",)),
        "road_ids": _ids_containing(label_names, ("road", "ground")),
        "lowstuff_ids": _ids_containing(label_names, ("grass", "tree", "vegetation", "mountain")),
        "static_ids": _ids_containing(
            label_names,
            ("wall", "fence", "pole", "building", "house", "bridge", "construction", "traffic sign", "billboard"),
        ),
    }


def _patch_semantic(
    semantic: Optional[Mapping[str, Any]],
    *,
    start: int,
    end: int,
    patch_grid: Tuple[int, int],
) -> Optional[Dict[str, torch.Tensor]]:
    if semantic is None:
        return None
    labels_all = semantic["label_maps"]
    if end > int(labels_all.shape[0]):
        return None
    labels = labels_all[int(start) : int(end)]
    conf_raw = semantic.get("confidence_maps")
    conf = None
    if torch.is_tensor(conf_raw):
        conf = conf_raw.detach().cpu()[int(start) : int(end)]
    conf_norm, _ = _normalize_dense_semantic_confidence(conf, target_shape=tuple(labels.shape))
    if conf_norm is None:
        conf_norm = torch.ones_like(labels, dtype=torch.float32)
    patch_label, purity, patch_conf = _mode_pool_dense_semantic_patches(
        labels.long(),
        conf_norm,
        patch_grid=patch_grid,
    )
    trust = (patch_conf * purity.square()).clamp(0.0, 1.0)
    return {"label": patch_label.long(), "purity": purity.float(), "confidence": patch_conf.float(), "trust": trust.float()}


def _mask_from_ids(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros_like(labels, dtype=torch.bool)
    out = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        out |= labels == int(idx)
    return out


def _weighted_mean(z: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, min_count: int = 64) -> Optional[torch.Tensor]:
    flat_z = z.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    if int(flat_mask.sum().item()) < int(min_count):
        return None
    flat_w = weight.reshape(-1)[flat_mask].float().clamp_min(1e-4)
    vals = flat_z[flat_mask].float()
    return (vals * flat_w[:, None]).sum(dim=0) / flat_w.sum().clamp_min(1e-6)


def _dist(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(torch.linalg.norm(a - b).item())


def _semantic_metrics(
    z_by_chunk: Sequence[Tuple[Dict[str, Any], torch.Tensor]],
    semantic: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if semantic is None:
        return {
            "semantic_available": False,
            "support_risk_z_dist": None,
            "dynamic_static_z_dist": None,
            "sky_static_z_dist": None,
            "road_static_z_dist": None,
            "semantic_trust_mean": None,
            "support_token_count": 0,
            "risk_token_count": 0,
        }
    zs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    trusts: List[torch.Tensor] = []
    purities: List[torch.Tensor] = []
    for row, z in z_by_chunk:
        sem_patch = row.get("semantic_patch")
        if sem_patch is None:
            continue
        labels.append(sem_patch["label"].reshape(-1))
        trusts.append(sem_patch["trust"].reshape(-1))
        purities.append(sem_patch["purity"].reshape(-1))
        zs.append(z.reshape(-1, 3).float())
    if not zs:
        return {
            "semantic_available": False,
            "support_risk_z_dist": None,
            "dynamic_static_z_dist": None,
            "sky_static_z_dist": None,
            "road_static_z_dist": None,
            "semantic_trust_mean": None,
            "support_token_count": 0,
            "risk_token_count": 0,
        }
    zcat = torch.cat(zs, dim=0)
    lcat = torch.cat(labels, dim=0)
    tcat = torch.cat(trusts, dim=0)
    pcat = torch.cat(purities, dim=0)
    dynamic = _mask_from_ids(lcat, semantic.get("dynamic_ids", []))
    sky = _mask_from_ids(lcat, semantic.get("sky_ids", []))
    road = _mask_from_ids(lcat, semantic.get("road_ids", []))
    lowstuff = _mask_from_ids(lcat, semantic.get("lowstuff_ids", []))
    static = _mask_from_ids(lcat, semantic.get("static_ids", []))
    support = static & (tcat >= 0.50)
    risk = dynamic | sky | lowstuff | (tcat < 0.50)
    static_mean = _weighted_mean(zcat, static, tcat)
    support_mean = _weighted_mean(zcat, support, tcat)
    risk_mean = _weighted_mean(zcat, risk, 1.0 - tcat + 0.1)
    dynamic_mean = _weighted_mean(zcat, dynamic, tcat)
    sky_mean = _weighted_mean(zcat, sky, tcat)
    road_mean = _weighted_mean(zcat, road, tcat)
    return {
        "semantic_available": True,
        "support_risk_z_dist": _dist(support_mean, risk_mean),
        "dynamic_static_z_dist": _dist(dynamic_mean, static_mean),
        "sky_static_z_dist": _dist(sky_mean, static_mean),
        "road_static_z_dist": _dist(road_mean, static_mean),
        "semantic_trust_mean": float(tcat.mean().item()),
        "patch_purity_mean": float(pcat.mean().item()),
        "support_token_count": int(support.sum().item()),
        "risk_token_count": int(risk.sum().item()),
        "dynamic_token_count": int(dynamic.sum().item()),
        "static_token_count": int(static.sum().item()),
        "sky_token_count": int(sky.sum().item()),
        "road_token_count": int(road.sum().item()),
    }


def _select_visual_indices(num_frames: int, count: int) -> List[int]:
    if num_frames <= 0:
        return []
    if count <= 1:
        return [num_frames // 2]
    raw = np.linspace(0, num_frames - 1, int(count), dtype=int).tolist()
    return sorted(set(int(x) for x in raw))


def _resize_panel(arr: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if scale > 1:
        img = img.resize((img.width * int(scale), img.height * int(scale)), Image.Resampling.NEAREST)
    return img


def _label_panel(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 420), 17), fill=(0, 0, 0))
    draw.text((4, 3), text[:82], fill=(255, 255, 255))
    return out


def _trust_rgb(trust: torch.Tensor) -> np.ndarray:
    arr = trust.detach().cpu().clamp(0.0, 1.0).numpy()
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * (1.0 - arr)).round().astype(np.uint8)
    rgb[..., 1] = (255.0 * arr).round().astype(np.uint8)
    rgb[..., 2] = 64
    return rgb


def _semantic_rgb(labels: torch.Tensor, colours: np.ndarray) -> np.ndarray:
    arr = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    safe = np.clip(arr, 0, max(0, len(colours) - 1))
    return colours[safe]


def _write_visuals(
    *,
    out_dir: Path,
    tap: str,
    layer_id: int,
    chunks: Sequence[Dict[str, Any]],
    rgb_by_chunk: Sequence[np.ndarray],
    semantic: Optional[Mapping[str, Any]],
    cell_scale: int,
    visual_frames_per_chunk: int,
) -> List[Dict[str, Any]]:
    contact_dir = out_dir / "contact_sheets"
    film_dir = out_dir / "filmstrips"
    heat_dir = out_dir / "pc_heatmaps"
    contact_dir.mkdir(parents=True, exist_ok=True)
    film_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Image.Image] = []
    film_panels: List[Image.Image] = []
    visual_rows: List[Dict[str, Any]] = []
    colours = semantic.get("colours") if semantic is not None else np.zeros((1, 3), dtype=np.uint8)
    for row, rgb_frames in zip(chunks, rgb_by_chunk):
        payload = row["payload"]
        start = int(payload["start_frame"])
        sem_patch = row.get("semantic_patch")
        frame_indices = _select_visual_indices(int(rgb_frames.shape[0]), visual_frames_per_chunk)
        for local_frame in frame_indices:
            global_frame = start + int(local_frame)
            pca_img = _resize_panel(rgb_frames[int(local_frame)], cell_scale)
            pca_img = _label_panel(pca_img, f"pca {tap} L{layer_id} c{payload['chunk_idx']:03d} f{global_frame:06d}")
            heat_path = heat_dir / f"{_slug(tap)}_L{int(layer_id):02d}_chunk{int(payload['chunk_idx']):03d}_f{global_frame:06d}.png"
            pca_img.save(heat_path)
            film_panels.append(pca_img)
            if sem_patch is None:
                sem_img = _resize_panel(np.zeros_like(rgb_frames[int(local_frame)]), cell_scale)
                trust_img = _resize_panel(np.zeros_like(rgb_frames[int(local_frame)]), cell_scale)
            else:
                sem_img = _resize_panel(_semantic_rgb(sem_patch["label"][int(local_frame)], colours), cell_scale)
                trust_img = _resize_panel(_trust_rgb(sem_patch["trust"][int(local_frame)]), cell_scale)
            sem_img = _label_panel(sem_img, f"semantic c{payload['chunk_idx']:03d} f{global_frame:06d}")
            trust_img = _label_panel(trust_img, "semantic trust")
            row_img = Image.new("RGB", (sem_img.width + trust_img.width + pca_img.width, sem_img.height), (0, 0, 0))
            row_img.paste(sem_img, (0, 0))
            row_img.paste(trust_img, (sem_img.width, 0))
            row_img.paste(pca_img, (sem_img.width + trust_img.width, 0))
            rows.append(row_img)
            visual_rows.append(
                {
                    "tap": tap,
                    "layer": int(layer_id),
                    "chunk_idx": int(payload["chunk_idx"]),
                    "global_frame": int(global_frame),
                    "pc_heatmap": str(heat_path),
                }
            )
    if rows:
        contact = Image.new("RGB", (max(x.width for x in rows), sum(x.height for x in rows)), (0, 0, 0))
        y = 0
        for img in rows:
            contact.paste(img, (0, y))
            y += img.height
        contact.save(contact_dir / f"{_slug(tap)}_L{int(layer_id):02d}_contact.png")
    if film_panels:
        per_row = min(6, len(film_panels))
        width = max(x.width for x in film_panels) * per_row
        height = max(x.height for x in film_panels) * int(math.ceil(len(film_panels) / per_row))
        film = Image.new("RGB", (width, height), (0, 0, 0))
        for idx, img in enumerate(film_panels):
            film.paste(img, ((idx % per_row) * img.width, (idx // per_row) * img.height))
        film.save(film_dir / f"{_slug(tap)}_L{int(layer_id):02d}_filmstrip.png")
    return visual_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _control_group_available(layer_ids: Sequence[int]) -> bool:
    unique = sorted(set(int(x) for x in layer_ids))
    if len(unique) < 6:
        return False
    adjacent_pairs = sum(1 for a, b in zip(unique, unique[1:]) if b == a + 1)
    return adjacent_pairs >= 2


def _rank_candidates(rows: Sequence[Mapping[str, Any]], taps: Sequence[str]) -> List[Dict[str, Any]]:
    eligible = [r for r in rows if str(r.get("tap")) in set(taps)]
    ranked = sorted(
        eligible,
        key=lambda r: (
            _safe_float(r.get("auto_path_score")) is not None,
            _safe_float(r.get("auto_path_score")) or -1.0,
        ),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for row in ranked[:4]:
        out.append(
            {
                "tap": str(row.get("tap")),
                "layer": int(row.get("layer")),
                "auto_path_score": _safe_float(row.get("auto_path_score")),
                "support_risk_z_dist": _safe_float(row.get("support_risk_z_dist")),
                "explained_top3": _safe_float(row.get("explained_top3")),
            }
        )
    return out


def _score_controls(candidate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tap = str(candidate.get("tap"))
    layer = int(candidate.get("layer"))
    score = _safe_float(candidate.get("auto_path_score")) or 0.0
    same_tap = [r for r in rows if str(r.get("tap")) == tap]
    adjacent_scores = [
        _safe_float(r.get("auto_path_score")) or 0.0
        for r in same_tap
        if abs(int(r.get("layer")) - layer) == 1
    ]
    other_scores = [
        _safe_float(r.get("auto_path_score")) or 0.0
        for r in same_tap
        if int(r.get("layer")) != layer and abs(int(r.get("layer")) - layer) != 1
    ]
    adjacent_max = max(adjacent_scores) if adjacent_scores else None
    random_median = float(np.median(other_scores)) if other_scores else None
    pass_adjacent = adjacent_max is not None and score > float(adjacent_max)
    pass_random = random_median is not None and score > float(random_median)
    return {
        "adjacent_max_score": adjacent_max,
        "random_median_score": random_median,
        "pass_adjacent_control": bool(pass_adjacent),
        "pass_random_control": bool(pass_random),
        "control_pass": bool(pass_adjacent and pass_random),
    }


def _write_selection_jsons(
    *,
    out_dir: Path,
    feature_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    layer_ids: Sequence[int],
    require_control_group: bool,
    unavailable_taps: Mapping[str, str],
) -> Dict[str, Any]:
    control_ok = _control_group_available(layer_ids)
    result: Dict[str, Any] = {}
    for path_name, taps in PATH_TAPS.items():
        candidates = _rank_candidates(rows, taps)
        candidates_with_controls: List[Dict[str, Any]] = []
        for candidate in candidates:
            enriched = dict(candidate)
            enriched.update(_score_controls(candidate, rows))
            candidates_with_controls.append(enriched)
        selected = [x for x in candidates_with_controls if bool(x.get("control_pass"))][:2]
        gate_pass = bool(selected) and ((not require_control_group) or control_ok)
        if gate_pass:
            status = "auto_selection_ready_control_checked"
        elif not control_ok and require_control_group:
            status = "blocked_no_adjacent_random_layer_control"
        else:
            status = "failed_candidate_does_not_beat_controls"
        payload = {
            "schema": "acl2_v68_layer_pca_selection_v1",
            "path": path_name.upper(),
            "feature_dir": str(feature_dir),
            "gate_pass": bool(gate_pass),
            "selection_status": status,
            "selected_layers": selected,
            "candidate_layers": candidates_with_controls,
            "control_group_available": bool(control_ok),
            "require_control_group": bool(require_control_group),
            "pca_only_not_cue_action_evidence": True,
            "unavailable_taps": dict(unavailable_taps),
        }
        out_path = out_dir / f"{path_name}_layers_selected.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        result[path_name] = payload
    return result


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    chunks = _load_feature_chunks(args.feature_dir)
    semantic = _load_semantic(args.semantic_pt) if str(args.semantic_pt) else None
    for row in chunks:
        payload = row["payload"]
        patch_grid = tuple(int(x) for x in payload.get("patch_grid", [0, 0]))
        row["semantic_patch"] = _patch_semantic(
            semantic,
            start=int(payload["start_frame"]),
            end=int(payload["end_frame"]),
            patch_grid=(int(patch_grid[0]), int(patch_grid[1])),
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    available_taps: Dict[str, Dict[str, Any]] = {}
    unavailable_taps: Dict[str, str] = {}
    first_payload = chunks[0]["payload"]
    requested_taps = list(first_payload.get("requested_taps") or [])
    all_layer_ids: List[int] = []
    metric_rows: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []

    for tap in requested_taps:
        tensor_key = f"tap::{tap}"
        if tensor_key not in first_payload:
            reason = str(dict(first_payload.get("taps") or {}).get(tap, {}).get("reason") or "tap_unavailable")
            unavailable_taps[tap] = reason
            continue
        tensors: List[torch.Tensor] = []
        valid_chunks: List[Dict[str, Any]] = []
        for row in chunks:
            payload = row["payload"]
            value = payload.get(tensor_key)
            if torch.is_tensor(value):
                tensors.append(_as_feature_tensor(value))
                valid_chunks.append(row)
        if not tensors:
            unavailable_taps[tap] = "no_tensor_values"
            continue
        layer_ids = _layer_ids_for_tap(valid_chunks[0]["payload"], tap, tensors[0])
        all_layer_ids.extend(layer_ids)
        available_taps[tap] = {"layer_ids": layer_ids, "num_chunks": len(valid_chunks)}
        for layer_pos, layer_id in enumerate(layer_ids):
            sample_parts = []
            per_chunk_budget = max(256, int(args.max_fit_tokens) // max(1, len(tensors)))
            for chunk_offset, tensor in enumerate(tensors):
                sample_parts.append(
                    _sample_rows(tensor[:, layer_pos], per_chunk_budget, seed=int(args.seed) + layer_pos * 17 + chunk_offset)
                )
            samples = torch.cat(sample_parts, dim=0)
            if samples.shape[0] > int(args.max_fit_tokens):
                samples = _sample_rows(samples, int(args.max_fit_tokens), seed=int(args.seed) + layer_pos * 101)
            basis = _fit_pca(samples)
            z_sample = _project_pca(samples.reshape(samples.shape[0], 1, 1, samples.shape[1]), basis).reshape(-1, 3)
            lo, hi = _percentile_bounds(z_sample)
            z_chunks: List[Tuple[Dict[str, Any], torch.Tensor]] = []
            rgb_chunks: List[np.ndarray] = []
            for row, tensor in zip(valid_chunks, tensors):
                z = _project_pca(tensor[:, layer_pos], basis)
                z_chunks.append((row, z))
                rgb_chunks.append(_project_to_rgb(z, lo, hi))
            sem_metrics = _semantic_metrics(z_chunks, semantic)
            explained = basis["explained"]
            support_risk = _safe_float(sem_metrics.get("support_risk_z_dist")) or 0.0
            dyn_static = _safe_float(sem_metrics.get("dynamic_static_z_dist")) or 0.0
            trust_mean = _safe_float(sem_metrics.get("semantic_trust_mean")) or 0.0
            z_std = float(torch.cat([z.reshape(-1, 3) for _, z in z_chunks], dim=0).std(dim=0).mean().item())
            explained_top3 = float(explained.sum().item())
            auto_score = (0.55 * support_risk + 0.25 * dyn_static + 0.20 * z_std) * (0.25 + 0.75 * trust_mean)
            row = {
                "tap": tap,
                "layer": int(layer_id),
                "layer_pos_in_dump": int(layer_pos),
                "num_chunks": int(len(valid_chunks)),
                "fit_tokens": int(samples.shape[0]),
                "feature_dim": int(samples.shape[1]),
                "explained_pc1": float(explained[0].item()),
                "explained_pc2": float(explained[1].item()),
                "explained_pc3": float(explained[2].item()),
                "explained_top3": explained_top3,
                "z_std_mean": z_std,
                "auto_path_score": float(auto_score),
                **sem_metrics,
            }
            metric_rows.append(row)
            visual_rows.extend(
                _write_visuals(
                    out_dir=args.out_dir,
                    tap=tap,
                    layer_id=int(layer_id),
                    chunks=valid_chunks,
                    rgb_by_chunk=rgb_chunks,
                    semantic=semantic,
                    cell_scale=int(args.cell_scale),
                    visual_frames_per_chunk=int(args.visual_frames_per_chunk),
                )
            )

    metric_fields = [
        "tap",
        "layer",
        "layer_pos_in_dump",
        "num_chunks",
        "fit_tokens",
        "feature_dim",
        "explained_pc1",
        "explained_pc2",
        "explained_pc3",
        "explained_top3",
        "z_std_mean",
        "semantic_available",
        "semantic_trust_mean",
        "patch_purity_mean",
        "support_risk_z_dist",
        "dynamic_static_z_dist",
        "sky_static_z_dist",
        "road_static_z_dist",
        "support_token_count",
        "risk_token_count",
        "dynamic_token_count",
        "static_token_count",
        "sky_token_count",
        "road_token_count",
        "auto_path_score",
    ]
    _write_csv(args.out_dir / "auto_layer_metrics.csv", metric_rows, metric_fields)
    review_rows = [
        {
            "tap": row["tap"],
            "layer": row["layer"],
            "review_source": "auto_initial_not_human_review",
            "visual_decision": "needs_human_or_control_review",
            "can_select_for_method": "false",
            "auto_path_score": row.get("auto_path_score"),
            "note": "PCA visualization only; not cue/action evidence.",
        }
        for row in metric_rows
    ]
    _write_csv(
        args.out_dir / "visual_layer_review.csv",
        review_rows,
        ["tap", "layer", "review_source", "visual_decision", "can_select_for_method", "auto_path_score", "note"],
    )
    _write_csv(
        args.out_dir / "pc_heatmaps_manifest.csv",
        visual_rows,
        ["tap", "layer", "chunk_idx", "global_frame", "pc_heatmap"],
    )

    selections = _write_selection_jsons(
        out_dir=args.out_dir,
        feature_dir=args.feature_dir,
        rows=metric_rows,
        layer_ids=all_layer_ids,
        require_control_group=bool(int(args.require_control_group)),
        unavailable_taps=unavailable_taps,
    )
    summary = {
        "schema": "acl2_v68_layerwise_pca_summary_v1",
        "feature_dir": str(args.feature_dir),
        "out_dir": str(args.out_dir),
        "num_feature_chunks": int(len(chunks)),
        "feature_chunk_files": [str(x["path"]) for x in chunks],
        "semantic_pt": str(args.semantic_pt) if str(args.semantic_pt) else "",
        "semantic_available": bool(semantic is not None),
        "available_taps": available_taps,
        "unavailable_taps": unavailable_taps,
        "num_pca_units": int(len(metric_rows)),
        "pca_unit": "per_tap_per_layer",
        "control_group_available": bool(_control_group_available(all_layer_ids)),
        "require_control_group": bool(int(args.require_control_group)),
        "selection_status_by_path": {k: v["selection_status"] for k, v in selections.items()},
        "gate_pass_by_path": {k: bool(v["gate_pass"]) for k, v in selections.items()},
        "outputs": {
            "auto_layer_metrics": str(args.out_dir / "auto_layer_metrics.csv"),
            "visual_layer_review": str(args.out_dir / "visual_layer_review.csv"),
            "pc_heatmaps_manifest": str(args.out_dir / "pc_heatmaps_manifest.csv"),
            "contact_sheets": str(args.out_dir / "contact_sheets"),
            "filmstrips": str(args.out_dir / "filmstrips"),
            "pc_heatmaps": str(args.out_dir / "pc_heatmaps"),
        },
    }
    (args.out_dir / "pca_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
