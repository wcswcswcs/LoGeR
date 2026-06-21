#!/usr/bin/env python3
"""Generate v78 PCA-grounded overlay panels.

The visualizer can be used for Phase 0 repair or Phase 8 rediscovery. It builds
auditable panels with RGB, semantic label, confidence, PCA, spatial D_geo,
chunk failure metrics, actual candidate mask, same-mass random, and
group-stratified random. The masks are deterministic training-free rules used
for visual/action audit, not learned selectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_PCA_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v76tf_c9_informed_semantic_tri_replay_memory_control/"
    "report_final/phase8_layer_feature_pca_visual_audit/full_qkv_smoke96_pca_rgb4views"
)
DEFAULT_SEMANTIC_PT = Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt")
DEFAULT_RGB_DIR = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_ANCHOR_BANK = Path(
    "results/kitti01_hmc_v2/acl2_v69_semantic_anchor_scale/report_final/phaseA_anchor_bank/anchor_bank"
)
DEFAULT_SCALE_LEDGER = Path(
    "results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase1_scale_drift_ledger/scale_drift_ledger.json"
)
DEFAULT_GEOM_LEDGER = Path(
    "results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase2_geometry_memory_cue_ledger/geometry_cue_by_chunk.json"
)

PALETTE: dict[str, tuple[int, int, int]] = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-csv", type=Path, default=None)
    parser.add_argument("--questions-csv", type=Path, default=None)
    parser.add_argument("--pca-root", type=Path, default=DEFAULT_PCA_ROOT)
    parser.add_argument("--semantic-pt", type=Path, default=DEFAULT_SEMANTIC_PT)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--stage-c-cache", type=Path, default=None, help="Accepted for Phase 8 CLI compatibility.")
    parser.add_argument("--radio-sidecar", type=Path, default=None, help="Recorded in provenance when provided.")
    parser.add_argument("--anchor-bank-dir", type=Path, default=DEFAULT_ANCHOR_BANK)
    parser.add_argument("--scale-ledger", type=Path, default=DEFAULT_SCALE_LEDGER)
    parser.add_argument("--geometry-ledger", type=Path, default=DEFAULT_GEOM_LEDGER)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", default="0,31,60,89,95")
    parser.add_argument("--max-clues", type=int, default=0)
    parser.add_argument("--seed", type=int, default=78)
    parser.add_argument("--panel-scale", type=int, default=6)
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _stable_colour(label: str) -> tuple[int, int, int]:
    if label in PALETTE:
        return PALETTE[label]
    value = 2166136261
    for byte in str(label).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return (64 + (value & 127), 64 + ((value >> 8) & 127), 64 + ((value >> 16) & 127))


def _colour_table(label_names: list[str]) -> np.ndarray:
    colours = np.zeros((max(1, len(label_names)), 3), dtype=np.uint8)
    for idx, name in enumerate(label_names):
        colours[idx] = _stable_colour(str(name))
    return colours


def _ids_containing(label_names: list[str], words: Iterable[str]) -> set[int]:
    lowered = [str(x).lower() for x in label_names]
    keys = [str(w).lower() for w in words]
    return {idx for idx, name in enumerate(lowered) if any(k in name for k in keys)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _load_registry(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.registry_csv and args.registry_csv.exists():
        for row in _read_csv(args.registry_csv):
            if str(row.get("review_status")) not in {"confirmed", "ambiguous"}:
                continue
            rows.append(
                {
                    "clue_id": row["clue_id"],
                    "memory_body": row["memory_body"],
                    "attention_type": row["attention_type"],
                    "tap": row["tap"],
                    "layer": int(row["layer"]),
                    "component": row.get("component", ""),
                    "review_status": row.get("review_status", ""),
                    "visual_pattern_observed": row.get("visual_pattern_observed", ""),
                }
            )
    elif args.questions_csv and args.questions_csv.exists():
        for idx, row in enumerate(_read_csv(args.questions_csv)):
            tap = row.get("new_tap_or_layer_to_dump") or row.get("old_tap") or row.get("tap") or ""
            layer_text = row.get("old_layer") or row.get("layer") or "0"
            found = re.search(r"(\d+)", str(layer_text))
            if not tap or not found:
                continue
            rows.append(
                {
                    "clue_id": row.get("new_candidate_hypothesis") or f"V78-REDISC-{idx:03d}",
                    "memory_body": row.get("old_action") or row.get("memory_body") or "unknown",
                    "attention_type": row.get("old_action") or "unknown",
                    "tap": tap,
                    "layer": int(found.group(1)),
                    "component": row.get("old_tap", ""),
                    "review_status": "needs_new_tap",
                    "visual_pattern_observed": row.get("new_visual_question", ""),
                }
            )
    else:
        raise FileNotFoundError("Provide --registry-csv or --questions-csv")
    if args.max_clues > 0:
        rows = rows[: int(args.max_clues)]
    return rows


def _load_semantic(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict) or not torch.is_tensor(sem.get("label_maps")):
        raise KeyError(f"No semantic_segmentation.label_maps in {path}")
    label_names = [str(x) for x in sem.get("label_names", [])]
    return {
        "label_maps": sem["label_maps"].detach().cpu().long(),
        "confidence_maps": sem.get("confidence_maps").detach().cpu().float()
        if torch.is_tensor(sem.get("confidence_maps"))
        else torch.ones_like(sem["label_maps"], dtype=torch.float32),
        "label_names": label_names,
        "colours": _colour_table(label_names),
        "road_ids": _ids_containing(label_names, ("road", "ground")),
        "sky_ids": _ids_containing(label_names, ("sky",)),
        "vegetation_ids": _ids_containing(label_names, ("grass", "tree", "vegetation", "mountain")),
        "dynamic_ids": _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle")),
        "static_ids": _ids_containing(
            label_names,
            ("wall", "fence", "pole", "building", "house", "bridge", "construction", "traffic sign", "billboard"),
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("rows", []) if isinstance(payload, dict) else []:
        try:
            out[int(row["chunk_id"])] = dict(row)
        except Exception:
            continue
    return out


def _read_heatmap_manifest(pca_root: Path) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    path = pca_root / "pc_heatmaps_manifest.csv"
    rows: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in _read_csv(path):
        key = (row["tap"], int(row["layer"]), int(row["global_frame"]))
        rows.setdefault(key, []).append(row)
    return rows


def _nearest_available_frame(
    heatmaps: dict[tuple[str, int, int], list[dict[str, Any]]],
    tap: str,
    layer: int,
    target: int,
) -> int | None:
    candidates = [f for (t, l, f) in heatmaps if t == tap and l == int(layer)]
    if not candidates:
        return None
    return min(candidates, key=lambda x: (abs(x - int(target)), x))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(path: Path) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return {
        "sha256": _sha256(path),
        "width": int(img.width),
        "height": int(img.height),
        "image_intensity_std": float(arr.std()),
        "nonempty_image": bool(img.width >= 512 and img.height >= 256 and float(arr.std()) > 1.0),
    }


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text[:110], fill=(255, 255, 255))
    return out


def _resize(img: Image.Image, scale: int) -> Image.Image:
    w, h = 66 * scale, 19 * scale
    return img.convert("RGB").resize((w, h), Image.Resampling.NEAREST)


def _heat(arr: np.ndarray, *, invert: bool = False) -> Image.Image:
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        x = np.zeros((19, 66), dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        y = np.zeros_like(x, dtype=np.float32)
    else:
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
        y = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    if invert:
        y = 1.0 - y
    rgb = np.zeros((*y.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * y).round().astype(np.uint8)
    rgb[..., 1] = (255 * (1.0 - np.abs(y - 0.5) * 2.0)).round().astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - y)).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _mask_img(mask: np.ndarray, colour: tuple[int, int, int]) -> Image.Image:
    m = np.asarray(mask, dtype=bool)
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[m] = colour
    rgb[~m] = (35, 35, 35)
    return Image.fromarray(rgb, "RGB")


def _semantic_panel(labels: torch.Tensor, colours: np.ndarray) -> Image.Image:
    arr = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    arr = np.clip(arr, 0, max(0, len(colours) - 1))
    return Image.fromarray(colours[arr], "RGB")


def _downsample_label(labels: torch.Tensor, frame: int, size: tuple[int, int] = (66, 19)) -> torch.Tensor:
    img = Image.fromarray(labels[int(frame)].detach().cpu().numpy().astype(np.uint8), "L")
    return torch.from_numpy(np.asarray(img.resize(size, Image.Resampling.NEAREST)).copy()).long()


def _downsample_float(x: torch.Tensor, frame: int, size: tuple[int, int] = (66, 19)) -> torch.Tensor:
    arr = x[int(frame)].detach().cpu().numpy().astype(np.float32)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    out = np.asarray(img.resize(size, Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    return torch.from_numpy(out).float()


def _find_rgb_path(rgb_dir: Path, frame: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = rgb_dir / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"RGB frame not found: {rgb_dir}/{int(frame):06d}.*")


def _load_anchor(anchor_dir: Path, chunk_id: int) -> dict[str, torch.Tensor]:
    path = anchor_dir / f"chunk_{int(chunk_id):03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _torch_load(path)
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("D_geo")):
        raise KeyError(f"No D_geo tensor in {path}")
    return {
        "D_geo": payload["D_geo"].detach().cpu().float(),
        "risk_score": payload.get("risk_score").detach().cpu().float()
        if torch.is_tensor(payload.get("risk_score"))
        else payload["D_geo"].detach().cpu().float(),
        "semantic_trust": payload.get("semantic_trust").detach().cpu().float()
        if torch.is_tensor(payload.get("semantic_trust"))
        else torch.ones_like(payload["D_geo"], dtype=torch.float32),
        "gram_motion": payload.get("gram_motion").detach().cpu().float()
        if torch.is_tensor(payload.get("gram_motion"))
        else torch.zeros_like(payload["D_geo"], dtype=torch.float32),
    }


def _mask_from_ids(labels: torch.Tensor, ids: set[int]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        out |= labels == int(idx)
    return out


def _action_mask(clue: dict[str, Any], labels: torch.Tensor, conf: torch.Tensor, d_geo: torch.Tensor, sem: dict[str, Any]) -> torch.Tensor:
    road = _mask_from_ids(labels, sem["road_ids"])
    sky = _mask_from_ids(labels, sem["sky_ids"])
    vegetation = _mask_from_ids(labels, sem["vegetation_ids"])
    dynamic = _mask_from_ids(labels, sem["dynamic_ids"])
    static = _mask_from_ids(labels, sem["static_ids"])
    low_conf = conf < 0.55
    high_d = d_geo > torch.quantile(d_geo.flatten(), 0.75)
    clue_id = str(clue["clue_id"])
    memory = str(clue.get("memory_body", ""))
    if "GLOBAL-K-L07" in clue_id or "GLOBAL-K-L17" in clue_id:
        mask = (road | static | vegetation) & (~dynamic) & (conf >= 0.45)
    elif "GLOBAL-V-L13" in clue_id:
        mask = sky | vegetation | dynamic | low_conf | high_d
    elif "FRAME-V-L18" in clue_id:
        mask = (road | static) & (conf >= 0.55) & (d_geo <= 0.65)
    elif "SWA" in clue_id or "swa" in memory:
        mask = (road | static) & (conf >= 0.50) & (d_geo <= 0.70)
    elif "TTT" in clue_id or "ttt" in memory:
        mask = high_d | dynamic | low_conf
    else:
        mask = high_d | low_conf
    if int(mask.sum().item()) == 0:
        mask = high_d
    return mask.bool()


def _same_mass_random(mask: torch.Tensor, seed: int) -> torch.Tensor:
    flat = mask.reshape(-1)
    count = int(flat.sum().item())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(int(flat.numel()), generator=gen)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[perm[:count]] = True
    return out.reshape_as(mask)


def _group_stratified_random(mask: torch.Tensor, groups: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.zeros_like(mask, dtype=torch.bool).reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_groups = groups.reshape(-1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    for gid in torch.unique(flat_groups).tolist():
        idx = torch.nonzero(flat_groups == int(gid), as_tuple=False).reshape(-1)
        count = int(flat_mask[idx].sum().item())
        if count <= 0 or int(idx.numel()) == 0:
            continue
        perm = idx[torch.randperm(int(idx.numel()), generator=gen)[: min(count, int(idx.numel()))]]
        out[perm] = True
    return out.reshape_as(mask)


def _groups(labels: torch.Tensor, sem: dict[str, Any]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.long)
    for gid, key in enumerate(("road_ids", "sky_ids", "vegetation_ids", "dynamic_ids", "static_ids"), start=1):
        out[_mask_from_ids(labels, sem[key])] = int(gid)
    return out


def _failure_panel(
    scale_row: dict[str, Any],
    geom_row: dict[str, Any],
    *,
    size: tuple[int, int] = (66, 19),
) -> Image.Image:
    vals = [
        float(scale_row.get("future_after_overlap") or 0.0),
        float(scale_row.get("head_to_tail") or 0.0),
        float(scale_row.get("scale_cv") or 0.0),
        float(scale_row.get("boundary_jump") or 0.0),
        float(geom_row.get("raw_overlap_residual_rmse") or 0.0),
        float(geom_row.get("D_geo_q90_patch") or 0.0),
    ]
    arr = np.zeros((size[1], size[0]), dtype=np.float32)
    stripes = np.array_split(np.arange(size[0]), len(vals))
    for s, val in zip(stripes, vals):
        arr[:, s] = val
    img = _heat(arr)
    return img


def _metric_text(scale_row: dict[str, Any], geom_row: dict[str, Any]) -> str:
    def f(key: str, row: dict[str, Any]) -> str:
        try:
            value = float(row.get(key))
        except Exception:
            return "NA"
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3g}"

    return (
        f"future {f('future_after_overlap', scale_row)} | h2t {f('head_to_tail', scale_row)} | "
        f"scale_cv {f('scale_cv', scale_row)} | Dq90 {f('D_geo_q90_patch', geom_row)}"
    )


def _chunk_for_frame(frame: int) -> tuple[int, int]:
    if frame <= 31:
        return 0, frame
    if frame <= 60:
        return 1, frame - 29
    if frame <= 89:
        return 2, frame - 58
    return 3, frame - 87


def _compose_grid(panels: list[Image.Image]) -> Image.Image:
    cols = 4
    rows = int(math.ceil(len(panels) / cols))
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    return canvas


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase8_mode = bool(args.questions_csv)
    visual_dir = args.out_dir / ("new_pca_overlay_panels" if phase8_mode else "overlay_panels")
    film_dir = args.out_dir / ("new_temporal_filmstrips" if phase8_mode else "temporal_filmstrips")
    action_dir = args.out_dir / ("new_action_vs_random_panels" if phase8_mode else "action_vs_random_panels")
    contact_dir = args.out_dir / ("new_pca_contact_sheets" if phase8_mode else "contact_sheets")
    visual_dir.mkdir(parents=True, exist_ok=True)
    film_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    registry = _load_registry(args)
    semantic = _load_semantic(args.semantic_pt)
    heatmaps = _read_heatmap_manifest(args.pca_root)
    scale_by_chunk = _ledger_by_chunk(args.scale_ledger)
    geom_by_chunk = _ledger_by_chunk(args.geometry_ledger)
    requested_frames = [int(x) for x in str(args.frames).split(",") if str(x).strip()]

    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    clue_visuals: dict[str, list[Image.Image]] = {}
    failures: list[str] = []

    for clue_idx, clue in enumerate(registry):
        for requested_frame in requested_frames:
            frame = _nearest_available_frame(heatmaps, clue["tap"], int(clue["layer"]), requested_frame)
            if frame is None:
                failures.append(f"missing_pca_heatmap:{clue['clue_id']}:{requested_frame}")
                continue
            chunk_id, local = _chunk_for_frame(frame)
            key = (clue["tap"], int(clue["layer"]), int(frame))
            heat_rows = heatmaps.get(key) or []
            if not heat_rows:
                failures.append(f"missing_pca_heatmap_row:{clue['clue_id']}:{frame}")
                continue
            pca_path = Path(heat_rows[0]["pc_heatmap"])
            anchor = _load_anchor(args.anchor_bank_dir, chunk_id)
            if local >= int(anchor["D_geo"].shape[0]):
                local = int(anchor["D_geo"].shape[0]) - 1

            label_patch = _downsample_label(semantic["label_maps"], frame)
            conf_patch = _downsample_float(semantic["confidence_maps"], frame)
            d_geo = anchor["D_geo"][local].float().clamp(0.0, 1.0)
            actual_mask = _action_mask(clue, label_patch, conf_patch, d_geo, semantic)
            same_random = _same_mass_random(actual_mask, int(args.seed) + clue_idx * 101 + frame)
            group_random = _group_stratified_random(actual_mask, _groups(label_patch, semantic), int(args.seed) + clue_idx * 197 + frame)

            rgb = Image.open(_find_rgb_path(args.rgb_dir, frame)).convert("RGB")
            rgb = rgb.resize((66, 19), Image.Resampling.BILINEAR)
            semantic_img = _semantic_panel(label_patch, semantic["colours"])
            conf_img = _heat(conf_patch.numpy())
            pca_img = Image.open(pca_path).convert("RGB").resize((66, 19), Image.Resampling.BILINEAR)
            d_geo_img = _heat(d_geo.numpy())
            scale_row = scale_by_chunk.get(chunk_id, {})
            geom_row = geom_by_chunk.get(chunk_id, {})
            failure_img = _failure_panel(scale_row, geom_row)
            metric_text = _metric_text(scale_row, geom_row)
            actual_img = _mask_img(actual_mask.numpy(), (255, 110, 50))
            same_img = _mask_img(same_random.numpy(), (120, 180, 255))
            group_img = _mask_img(group_random.numpy(), (180, 120, 255))

            panels = [
                _label(_resize(rgb, args.panel_scale), f"RGB f{frame:06d} c{chunk_id}"),
                _label(_resize(semantic_img, args.panel_scale), "semantic label"),
                _label(_resize(conf_img, args.panel_scale), "semantic confidence"),
                _label(_resize(pca_img, args.panel_scale), f"PCA {clue['tap']} L{int(clue['layer']):02d}"),
                _label(_resize(d_geo_img, args.panel_scale), "D_geo spatial from v69 anchor bank"),
                _label(_resize(failure_img, args.panel_scale), metric_text),
                _label(_resize(actual_img, args.panel_scale), "actual candidate mask"),
                _label(_resize(same_img, args.panel_scale), "same-mass random"),
                _label(_resize(group_img, args.panel_scale), "group-stratified random"),
            ]
            overlay = _compose_grid(panels)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(clue["clue_id"]))
            out_path = visual_dir / f"{safe}_f{int(frame):06d}.png"
            overlay.save(out_path)

            action_panel = _compose_grid([panels[0], panels[1], panels[3], panels[4], panels[5], panels[6], panels[7], panels[8]])
            action_path = action_dir / f"{safe}_action_vs_random_f{int(frame):06d}.png"
            action_panel.save(action_path)
            clue_visuals.setdefault(str(clue["clue_id"]), []).append(overlay)

            for visual_type, path in (("overlay_panel", out_path), ("action_vs_random_panel", action_path)):
                stats = _image_stats(path)
                row = {
                    "visual_file": str(path),
                    "visual_type": visual_type,
                    "clue_id": clue["clue_id"],
                    "chunk_id": chunk_id,
                    "frame_id": frame,
                    "tap": clue["tap"],
                    "layer": int(clue["layer"]),
                    "memory_body": clue.get("memory_body", ""),
                    "sha256": stats["sha256"],
                    "width": stats["width"],
                    "height": stats["height"],
                    "image_intensity_std": f"{stats['image_intensity_std']:.6f}",
                    "nonempty_image": stats["nonempty_image"],
                    "RGB_overlay_present": True,
                    "semantic_overlay_present": True,
                    "confidence_overlay_present": True,
                    "PCA_overlay_present": True,
                    "D_geo_overlay_present": True,
                    "future_overlay_present": True,
                    "head_tail_scale_overlay_present": True,
                    "action_mask_overlay_present": True,
                    "same_mass_random_overlay_present": True,
                    "group_stratified_random_overlay_present": True,
                    "source_pca_heatmap": str(pca_path),
                    "source_anchor_bank": str(args.anchor_bank_dir / f"chunk_{int(chunk_id):03d}.pt"),
                    "source_scale_ledger": str(args.scale_ledger),
                    "source_geometry_ledger": str(args.geometry_ledger),
                    "radio_sidecar": str(args.radio_sidecar) if args.radio_sidecar else "",
                    "actual_mask_mass": int(actual_mask.sum().item()),
                    "same_mass_random_mass": int(same_random.sum().item()),
                    "group_stratified_random_mass": int(group_random.sum().item()),
                }
                manifest_rows.append(row)
                review_rows.append(
                    {
                        "visual_file": str(path),
                        "chunk_id": chunk_id,
                        "frame_id": frame,
                        "tap": clue["tap"],
                        "layer": int(clue["layer"]),
                        "memory_body": clue.get("memory_body", ""),
                        "overlay_types": "RGB;semantic;confidence;PCA;D_geo;failure_metrics;actual;same_mass_random;group_stratified_random",
                        "review_status": "needs_new_tap",
                        "visual_pattern_observed": clue.get("visual_pattern_observed", ""),
                        "semantic_alignment": "generated panel; requires visual review",
                        "geometry_alignment": "D_geo spatial overlay plus chunk-level failure metrics present",
                        "failure_alignment": "future/head_tail/scale/boundary metrics rendered as chunk-level overlay",
                        "action_mask_alignment": "actual candidate mask rendered for review; not yet a method success claim",
                        "random_mask_difference": "same-mass and group-stratified random controls rendered",
                        "reviewer_note": "Panel generated by tools/visualize_v78_pca_rediscovery_panels.py; review must be filled after visual inspection.",
                        "new_hypothesis_id": "",
                    }
                )

    for clue_id, visuals in clue_visuals.items():
        if not visuals:
            continue
        w = max(img.width for img in visuals)
        h = max(img.height for img in visuals)
        film = Image.new("RGB", (w, h * len(visuals)), (0, 0, 0))
        y = 0
        for img in visuals:
            film.paste(img, (0, y))
            y += h
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(clue_id))
        contact_cols = min(2, len(visuals))
        contact_rows = int(math.ceil(len(visuals) / max(contact_cols, 1)))
        contact = Image.new("RGB", (contact_cols * w, contact_rows * h), (0, 0, 0))
        for idx, img in enumerate(visuals):
            contact.paste(img, ((idx % contact_cols) * w, (idx // contact_cols) * h))
        contact_path = contact_dir / f"{safe}_contact_sheet.png"
        contact.save(contact_path)
        stats = _image_stats(contact_path)
        manifest_rows.append(
            {
                "visual_file": str(contact_path),
                "visual_type": "pca_contact_sheet",
                "clue_id": clue_id,
                "chunk_id": "multi",
                "frame_id": "multi",
                "tap": "",
                "layer": "",
                "memory_body": "",
                "sha256": stats["sha256"],
                "width": stats["width"],
                "height": stats["height"],
                "image_intensity_std": f"{stats['image_intensity_std']:.6f}",
                "nonempty_image": stats["nonempty_image"],
                "RGB_overlay_present": True,
                "semantic_overlay_present": True,
                "confidence_overlay_present": True,
                "PCA_overlay_present": True,
                "D_geo_overlay_present": True,
                "future_overlay_present": True,
                "head_tail_scale_overlay_present": True,
                "action_mask_overlay_present": True,
                "same_mass_random_overlay_present": True,
                "group_stratified_random_overlay_present": True,
                "source_pca_heatmap": "see per-frame panels",
                "source_anchor_bank": str(args.anchor_bank_dir),
                "source_scale_ledger": str(args.scale_ledger),
                "source_geometry_ledger": str(args.geometry_ledger),
                "radio_sidecar": str(args.radio_sidecar) if args.radio_sidecar else "",
                "actual_mask_mass": "",
                "same_mass_random_mass": "",
                "group_stratified_random_mass": "",
            }
        )

        film_path = film_dir / f"{safe}_filmstrip.png"
        film.save(film_path)
        stats = _image_stats(film_path)
        manifest_rows.append(
            {
                "visual_file": str(film_path),
                "visual_type": "temporal_filmstrip",
                "clue_id": clue_id,
                "chunk_id": "multi",
                "frame_id": "multi",
                "tap": "",
                "layer": "",
                "memory_body": "",
                "sha256": stats["sha256"],
                "width": stats["width"],
                "height": stats["height"],
                "image_intensity_std": f"{stats['image_intensity_std']:.6f}",
                "nonempty_image": stats["nonempty_image"],
                "RGB_overlay_present": True,
                "semantic_overlay_present": True,
                "confidence_overlay_present": True,
                "PCA_overlay_present": True,
                "D_geo_overlay_present": True,
                "future_overlay_present": True,
                "head_tail_scale_overlay_present": True,
                "action_mask_overlay_present": True,
                "same_mass_random_overlay_present": True,
                "group_stratified_random_overlay_present": True,
                "source_pca_heatmap": "see per-frame panels",
                "source_anchor_bank": str(args.anchor_bank_dir),
                "source_scale_ledger": str(args.scale_ledger),
                "source_geometry_ledger": str(args.geometry_ledger),
                "radio_sidecar": str(args.radio_sidecar) if args.radio_sidecar else "",
                "actual_mask_mass": "",
                "same_mass_random_mass": "",
                "group_stratified_random_mass": "",
            }
        )

    manifest_fields = [
        "visual_file",
        "visual_type",
        "clue_id",
        "chunk_id",
        "frame_id",
        "tap",
        "layer",
        "memory_body",
        "sha256",
        "width",
        "height",
        "image_intensity_std",
        "nonempty_image",
        "RGB_overlay_present",
        "semantic_overlay_present",
        "confidence_overlay_present",
        "PCA_overlay_present",
        "D_geo_overlay_present",
        "future_overlay_present",
        "head_tail_scale_overlay_present",
        "action_mask_overlay_present",
        "same_mass_random_overlay_present",
        "group_stratified_random_overlay_present",
        "source_pca_heatmap",
        "source_anchor_bank",
        "source_scale_ledger",
        "source_geometry_ledger",
        "radio_sidecar",
        "actual_mask_mass",
        "same_mass_random_mass",
        "group_stratified_random_mass",
    ]
    review_fields = [
        "visual_file",
        "chunk_id",
        "frame_id",
        "tap",
        "layer",
        "memory_body",
        "overlay_types",
        "review_status",
        "visual_pattern_observed",
        "semantic_alignment",
        "geometry_alignment",
        "failure_alignment",
        "action_mask_alignment",
        "random_mask_difference",
        "reviewer_note",
        "new_hypothesis_id",
    ]
    _write_csv(args.out_dir / "visual_artifact_manifest.csv", manifest_rows, manifest_fields)
    _write_csv(args.out_dir / "visual_review.csv", review_rows, review_fields)
    integrity = {
        "schema": "acl2_v78_visual_panel_generation_v1",
        "num_registry_rows": len(registry),
        "num_visual_files": len(manifest_rows),
        "num_review_rows": len(review_rows),
        "all_files_exist": all(Path(r["visual_file"]).exists() for r in manifest_rows),
        "all_sha256_present": all(bool(r["sha256"]) for r in manifest_rows),
        "all_dimensions_present": all(int(r["width"]) >= 512 and int(r["height"]) >= 256 for r in manifest_rows),
        "all_nonempty": all(bool(r["nonempty_image"]) for r in manifest_rows),
        "missing_overlay_count": sum(
            1
            for r in manifest_rows
            if not (
                bool(r["RGB_overlay_present"])
                and bool(r["semantic_overlay_present"])
                and bool(r["confidence_overlay_present"])
                and bool(r["PCA_overlay_present"])
                and bool(r["D_geo_overlay_present"])
                and bool(r["future_overlay_present"])
                and bool(r["action_mask_overlay_present"])
                and bool(r["same_mass_random_overlay_present"])
                and bool(r["group_stratified_random_overlay_present"])
            )
        ),
        "generation_failures": failures,
        "gate_pass": False,
        "gate_note": "visual files generated; gate remains false until visual_review.csv is manually/semiautomatically reviewed",
        "provenance": {
            "registry_csv": str(args.registry_csv) if args.registry_csv else "",
            "questions_csv": str(args.questions_csv) if args.questions_csv else "",
            "pca_root": str(args.pca_root),
            "semantic_pt": str(args.semantic_pt),
            "rgb_dir": str(args.rgb_dir),
            "anchor_bank_dir": str(args.anchor_bank_dir),
            "scale_ledger": str(args.scale_ledger),
            "geometry_ledger": str(args.geometry_ledger),
            "stage_c_cache": str(args.stage_c_cache) if args.stage_c_cache else "",
            "radio_sidecar": str(args.radio_sidecar) if args.radio_sidecar else "",
        },
    }
    (args.out_dir / "visual_integrity_audit.json").write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "visual_insight.md").write_text(
        "\n".join(
            [
                "# v78 Generated Visual Panels",
                "",
                f"- Visual files: {len(manifest_rows)}",
                f"- Registry rows: {len(registry)}",
                f"- Generation failures: {len(failures)}",
                "",
                "These panels include real spatial D_geo from the v69 anchor bank and chunk-level failure metrics from",
                "the v73 scale/geometry ledgers. They are generated visual evidence only; review rows are intentionally",
                "`needs_human_review` until visual inspection confirms/rejects each pattern.",
                "",
                "Next step: open representative panels, update/replace `visual_review.csv` with confirmed/rejected/ambiguous",
                "statuses, then rerun `tools/audit_v78_visual_artifacts.py`.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
