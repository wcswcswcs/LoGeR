#!/usr/bin/env python3
"""Render v81S SWA QKV/overlap visual panels from repaired artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BANK = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS2_swa_good_bad_pair_bank/swa_good_bad_pair_bank.csv"
)
DEFAULT_QKV_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_visual_confirmation/qkv_prefix_runs"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_visual_confirmation"
)
DEFAULT_KITTI_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _heat(arr: np.ndarray, size: tuple[int, int], *, title: str = "") -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((19, 66), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(arr.min())
    amax = float(arr.max())
    norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    rgb = np.stack([norm * 255.0, (1.0 - np.abs(norm - 0.5) * 2.0) * 180.0, (1.0 - norm) * 255.0], axis=-1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return _title(img, title)


def _mask(mask: np.ndarray, size: tuple[int, int], *, title: str = "") -> Image.Image:
    arr = np.asarray(mask, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.stack([arr * 255.0, arr * 255.0, np.zeros_like(arr)], axis=-1)
    img = Image.fromarray(rgb.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return _title(img, title)


def _title(img: Image.Image, title: str) -> Image.Image:
    if not title:
        return img.convert("RGB")
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 4), title[:80], fill=(255, 255, 255), font=font)
    return img


def _text(lines: list[str], size: tuple[int, int], title: str) -> Image.Image:
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0], 20), fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    y = 28
    for line in lines:
        draw.text((6, y), str(line)[:92], fill=(25, 25, 25), font=font)
        y += 14
        if y > size[1] - 14:
            break
    return img


def _rgb(kitti_root: Path, seq: str, frame: int, size: tuple[int, int]) -> Image.Image:
    path = kitti_root / seq / "image_2" / f"{int(frame):06d}.png"
    if not path.is_file():
        return _text([f"missing RGB: {path}"], size, "RGB")
    return _title(Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR), f"RGB f{frame}")


def _semantic(preprocess_root: Path, seq: str, frame: int, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    path = preprocess_root / seq / "sparse_masklets_with_semantic.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    labels = sem.get("label_maps") if isinstance(sem, dict) else None
    conf = sem.get("confidence_maps") if isinstance(sem, dict) else None
    if not torch.is_tensor(labels) or int(frame) >= int(labels.shape[0]):
        missing = _text([f"missing semantic frame {frame}", str(path)], size, "Semantic")
        return missing, missing.copy()
    lab = labels[int(frame)].detach().cpu().long().numpy()
    color = np.zeros((*lab.shape, 3), dtype=np.uint8)
    color[..., 0] = (lab * 37 % 255).astype(np.uint8)
    color[..., 1] = (lab * 71 % 255).astype(np.uint8)
    color[..., 2] = (lab * 113 % 255).astype(np.uint8)
    sem_img = _title(Image.fromarray(color).resize(size, Image.Resampling.NEAREST), "Semantic label")
    if torch.is_tensor(conf):
        conf_img = _heat(conf[int(frame)].detach().cpu().float().numpy(), size, title="Semantic confidence")
    else:
        conf_img = _text(["missing confidence_maps"], size, "Semantic confidence")
    return sem_img, conf_img


def _qkv_map(path: Path, tap: str, local_frame: int, size: tuple[int, int], title: str) -> tuple[Image.Image, bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get(f"tap::{tap}") if isinstance(payload, dict) else None
    if not torch.is_tensor(tensor):
        return _text([f"missing {tap}", str(path)], size, title), False
    idx = max(0, min(int(local_frame), int(tensor.shape[0]) - 1))
    x = tensor[idx].float()
    while x.ndim > 3:
        x = x[0]
    if x.ndim == 3:
        x = torch.linalg.norm(x, dim=-1)
    return _heat(x.detach().cpu().numpy(), size, title=title), True


def _pair_maps(pair_path: Path, frame: int, size: tuple[int, int]) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    payload = torch.load(pair_path, map_location="cpu", weights_only=False)
    coords = payload.get("prev_pixel_coords")
    frame_ids = payload.get("prev_frame_ids")
    prev = payload.get("prev_overlap_points")
    curr = payload.get("curr_overlap_points")
    labels = payload.get("prev_semantic_labels")
    if not all(torch.is_tensor(x) for x in (coords, frame_ids, prev, curr, labels)):
        missing = _text(["missing pair tensors", str(pair_path)], size, "Overlap")
        return missing, missing.copy(), missing.copy(), {"route_proxy_available": False}
    selector = frame_ids.long() == int(frame)
    if int(selector.sum().item()) <= 0:
        selector = torch.ones_like(frame_ids, dtype=torch.bool)
    coords = coords[selector].long()
    residual = torch.linalg.norm(prev[selector].float() - curr[selector].float(), dim=1)
    labels = labels[selector].long()
    height, width = 266, 924
    residual_map = torch.zeros((height, width), dtype=torch.float32)
    route = torch.zeros((height, width), dtype=torch.float32)
    y = coords[:, 0].clamp(0, height - 1)
    x = coords[:, 1].clamp(0, width - 1)
    residual_map[y, x] = torch.maximum(residual_map[y, x], residual)
    route[y, x] = 1.0
    # Deterministic same-count random control.
    seed = int(hashlib.sha256(str(pair_path).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    count = int(route.sum().item())
    random_mask = np.zeros((height * width,), dtype=np.float32)
    if count > 0:
        random_mask[rng.choice(height * width, size=min(count, height * width), replace=False)] = 1.0
    random_mask = random_mask.reshape(height, width)
    meta = {
        "route_proxy_available": True,
        "route_proxy_source": "overlap_pair_sample_pixels",
        "route_proxy_pair_count": count,
        "route_proxy_labels": int(torch.unique(labels).numel()) if labels.numel() else 0,
    }
    return (
        _heat(residual_map.numpy(), size, title="Overlap residual"),
        _mask(route.numpy(), size, title="Actual overlap sample route proxy"),
        _mask(random_mask, size, title="Same-count random route"),
        meta,
    )


def _make_panel(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    prev_chunk = int(row["prev_chunk"])
    curr_chunk = int(row["curr_chunk"])
    pair_path = Path(row["overlap_pair_file"])
    pair_payload = torch.load(pair_path, map_location="cpu", weights_only=False)
    frame = int(pair_payload.get("overlap_start_frame", row.get("frame_start", 0)))
    prev_start = int(pair_payload.get("prev_start_frame", prev_chunk * 29))
    curr_start = int(pair_payload.get("curr_start_frame", curr_chunk * 29))
    prev_local = max(0, frame - prev_start)
    curr_local = max(0, frame - curr_start)
    qkv_dir = args.qkv_root / f"seq{seq}_native_prefix641" / "v68_layer_pca_features"
    prev_qkv = qkv_dir / f"chunk_{prev_chunk:03d}.pt"
    curr_qkv = qkv_dir / f"chunk_{curr_chunk:03d}.pt"
    tile = (360, 220)
    rgb = _rgb(args.kitti_root, seq, frame, tile)
    sem, sem_conf = _semantic(args.preprocess_root, seq, frame, tile)
    residual, route, random_route, route_meta = _pair_maps(pair_path, frame, tile)
    q_img, q_ok = _qkv_map(curr_qkv, "global_q_raw_patchvec_layers", curr_local, tile, "Current Q norm")
    k_img, k_ok = _qkv_map(prev_qkv, "global_k_raw_patchvec_layers", prev_local, tile, "Cache K norm")
    v_img, v_ok = _qkv_map(prev_qkv, "global_v_raw_patchvec_layers", prev_local, tile, "Cache V norm")
    text = _text(
        [
            f"seq={seq} pair={prev_chunk}->{curr_chunk} frame={frame}",
            f"case={row.get('case_type')} J_mid={row.get('J_mid')}",
            f"future={row.get('future_after_overlap')} boundary={row.get('boundary_jump')}",
            f"raw_overlap={row.get('raw_overlap_residual')} qrisk={row.get('artifact_quality_risk')}",
            "true SWA carried/rejected route: unavailable in current artifacts",
            "route panel is overlap sample proxy, not runtime SWA action",
        ],
        tile,
        "Audit notes",
    )
    panel = Image.new("RGB", (tile[0] * 3, tile[1] * 3), (255, 255, 255))
    for img, xy in [
        (rgb, (0, 0)),
        (sem, (tile[0], 0)),
        (sem_conf, (tile[0] * 2, 0)),
        (residual, (0, tile[1])),
        (route, (tile[0], tile[1])),
        (random_route, (tile[0] * 2, tile[1])),
        (q_img, (0, tile[1] * 2)),
        (k_img, (tile[0], tile[1] * 2)),
        (v_img if v_ok else text, (tile[0] * 2, tile[1] * 2)),
    ]:
        panel.paste(img, xy)
    case_dir = "bad_pair_panels" if row.get("case_type") == "bad" else "good_pair_panels"
    panel_dir = args.out_dir / case_dir
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / f"seq{seq}_chunk{prev_chunk:03d}_{curr_chunk:03d}_{row.get('case_type')}.png"
    panel.save(panel_path)
    # Plan-named duplicate views keep provenance simple without regenerating images.
    for sub in ("qkv_alignment_panels", "route_vs_random_panels"):
        target_dir = args.out_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / panel_path.name
        if not target.exists():
            panel.save(target)
    out = {
        "seq": seq,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "case_type": row.get("case_type", ""),
        "visual_file": str(panel_path),
        "sha256": _sha256(panel_path),
        "has_rgb_overlap_frame": True,
        "has_dense_semantic_label": True,
        "has_semantic_confidence": True,
        "has_overlap_residual_map": True,
        "has_qkv_maps": bool(q_ok and k_ok and v_ok),
        "has_actual_route_mask": False,
        "has_route_proxy_mask": bool(route_meta.get("route_proxy_available")),
        "has_same_count_random_mask": True,
        "route_source": route_meta.get("route_proxy_source", ""),
        "qkv_prev_file": str(prev_qkv),
        "qkv_curr_file": str(curr_qkv),
        "overlap_pair_file": str(pair_path),
        "artifact_quality_risk": row.get("artifact_quality_risk", ""),
    }
    out.update(route_meta)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--qkv-root", type=Path, default=DEFAULT_QKV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _read_csv(args.bank)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    manifest = [_make_panel(row, args) for row in rows]
    _write_csv(args.out_dir / "visual_manifest.csv", manifest)
    print(json.dumps({"rows": len(manifest), "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
