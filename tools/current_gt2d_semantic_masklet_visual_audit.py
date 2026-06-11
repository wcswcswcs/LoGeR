#!/usr/bin/env python3
"""Visual audit for current KITTI projected GT semantic and VideoMasklet inputs.

The audit intentionally keeps the evidence human-inspectable:

    RGB | Stage-C VideoMasklet frontend overlay | projected SemanticKITTI GT

The projected semantic panel is sparse LiDAR-to-image GT, not dense 2D semantic
segmentation.  The script also reloads the same frame through
``GTSemanticProvider`` and compares it with the landed v29C projection cache so
the visualization is tied to the code path that can load GT-style semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.gt_semantic_provider import (  # noqa: E402
    GTSemanticLayout,
    GTSemanticProvider,
    SEMANTIC_KITTI_ID_TO_NAME,
)


SEMANTIC_KITTI_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (30, 30, 30),
    1: (255, 255, 255),
    10: (0, 0, 142),
    11: (119, 11, 32),
    13: (0, 60, 100),
    15: (0, 0, 230),
    16: (0, 80, 100),
    18: (0, 0, 70),
    20: (0, 0, 90),
    30: (220, 20, 60),
    31: (255, 0, 0),
    32: (255, 0, 120),
    40: (128, 64, 128),
    44: (250, 170, 160),
    48: (244, 35, 232),
    49: (81, 0, 81),
    50: (70, 70, 70),
    51: (190, 153, 153),
    52: (102, 102, 156),
    60: (230, 150, 140),
    70: (107, 142, 35),
    71: (152, 251, 152),
    72: (152, 251, 152),
    80: (153, 153, 153),
    81: (220, 220, 0),
    99: (255, 170, 0),
    252: (0, 0, 142),
    253: (255, 0, 0),
    254: (220, 20, 60),
    255: (255, 0, 120),
    256: (0, 80, 100),
    257: (0, 60, 100),
    258: (0, 0, 70),
    259: (0, 0, 90),
}


MASKLET_GROUP_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (37, 171, 70),
    1: (55, 126, 184),
    2: (255, 127, 0),
    3: (152, 78, 163),
    4: (180, 180, 180),
}


MASKLET_LABEL_TO_SEMANTIC_KITTI_ID: Dict[str, int] = {
    "road": 40,
    "sidewalk": 48,
    "building": 50,
    "wall": 50,
    "fence": 51,
    "pole": 80,
    "traffic sign": 81,
    "traffic_sign": 81,
    "vegetation": 70,
    "tree": 70,
    "terrain": 72,
    "grass": 72,
    "car": 10,
    "moving_car": 252,
    "truck": 18,
    "bus": 13,
    "person": 30,
    "bicycle": 11,
    "motorcycle": 15,
    # SemanticKITTI has no sky class because LiDAR projection cannot label sky.
    # Keep sky visually distinct and document it as not directly comparable.
    "sky": -1000,
}


def _parse_frames(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _color_for_semantic(label: int) -> Tuple[int, int, int]:
    if int(label) in SEMANTIC_KITTI_COLOR:
        return SEMANTIC_KITTI_COLOR[int(label)]
    rng = np.random.default_rng(int(label) + 991)
    return tuple(int(x) for x in rng.integers(40, 230, size=3))


def _overlay_points(
    image: Image.Image,
    sem: np.ndarray,
    valid: np.ndarray,
    *,
    radius: int,
    alpha: float,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(rgb, dtype=np.float32)
    yy, xx = np.nonzero(valid)
    for y, x in zip(yy.tolist(), xx.tolist()):
        c = _color_for_semantic(int(sem[y, x]))
        y0, y1 = max(0, y - radius), min(rgb.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(rgb.shape[1], x + radius + 1)
        color[y0:y1, x0:x1] = c
    mask = np.any(color > 0, axis=2)
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _masklet_color(label: str, group_id: int, *, mode: str) -> Tuple[int, int, int]:
    if mode == "semantic_kitti":
        mapped = MASKLET_LABEL_TO_SEMANTIC_KITTI_ID.get(label.strip().lower())
        if mapped is not None:
            if mapped == -1000:
                return (70, 130, 180)
            return _color_for_semantic(mapped)
    return MASKLET_GROUP_COLOR.get(group_id, (255, 255, 0))


def _overlay_masklets(
    image: Image.Image,
    payload: Dict[str, object],
    local_idx: int,
    *,
    color_mode: str,
) -> Tuple[Image.Image, Dict[str, object]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    out = rgb.copy()
    masks = payload["M_mask"]
    v_mask = payload["V_mask"]
    labels = list(payload.get("L_sem", []))
    groups = payload["G_sem"].detach().cpu().long().reshape(-1)
    q_mask = payload.get("Q_mask")
    visible: List[str] = []

    for j in range(int(payload.get("num_masklets", masks.shape[0]))):
        if local_idx < 0 or local_idx >= masks.shape[1]:
            continue
        if not bool(v_mask[j, local_idx]):
            continue
        mask = masks[j, local_idx].detach().cpu().numpy().astype(bool)
        if not mask.any():
            continue
        group_id = int(groups[j].item()) if j < int(groups.numel()) else 4
        label = str(labels[j]) if j < len(labels) else f"id{j}"
        color = np.array(_masklet_color(label, group_id, mode=color_mode), dtype=np.float32)
        alpha = 0.34
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
        mapped = MASKLET_LABEL_TO_SEMANTIC_KITTI_ID.get(label.strip().lower())
        mapped_text = "sky(no-gt)" if mapped == -1000 else (str(mapped) if mapped is not None else "unmapped")
        q_val = None
        if torch.is_tensor(q_mask) and j < q_mask.shape[0] and local_idx < q_mask.shape[1]:
            q_val = float(q_mask[j, local_idx].item())
        visible.append(
            f"{j}:{label}:SK{mapped_text}:G{group_id}:Q{q_val:.2f}"
            if q_val is not None else
            f"{j}:{label}:SK{mapped_text}:G{group_id}"
        )

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font = _font(14)
    y = 8
    for text in visible[:12]:
        draw.text((8, y), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += 18
    return img, {
        "visible_masklets": len(visible),
        "visible_masklet_labels": ";".join(visible),
    }


def _title_panel(img: Image.Image, title: str) -> Image.Image:
    font = _font(18)
    out = Image.new("RGB", (img.width, img.height + 32), (20, 20, 20))
    out.paste(img.convert("RGB"), (0, 32))
    draw = ImageDraw.Draw(out)
    draw.text((10, 6), title, fill=(255, 255, 255), font=font)
    return out


def _make_three_panel(rgb: Image.Image, masklet: Image.Image, gt: Image.Image, out_path: Path) -> None:
    panels = [
        _title_panel(rgb, "RGB original"),
        _title_panel(masklet, "VideoMasklet frontend overlay"),
        _title_panel(gt, "Sparse projected SemanticKITTI GT"),
    ]
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (3 * w, h), (0, 0, 0))
    for i, panel in enumerate(panels):
        padded = Image.new("RGB", (w, h), (0, 0, 0))
        padded.paste(panel, (0, 0))
        canvas.paste(padded, (i * w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _find_chunk(stage_c_cache_dir: Path, frame: int) -> Optional[Path]:
    for manifest_path in sorted(stage_c_cache_dir.glob("chunk_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        start = int(manifest.get("start_frame", -1))
        end = int(manifest.get("end_frame", -1))
        if start <= frame < end:
            return manifest_path.parent
    return None


def _label_summary(sem: np.ndarray, valid: np.ndarray) -> str:
    labels, counts = np.unique(sem[valid].astype(np.int32), return_counts=True)
    order = np.argsort(-counts)
    chunks = []
    for label, count in zip(labels[order][:8], counts[order][:8]):
        chunks.append(f"{int(label)}:{SEMANTIC_KITTI_ID_TO_NAME.get(int(label), f'id{int(label)}')}:{int(count)}")
    return " ".join(chunks)


def _provider_layout(sequence_root: Path) -> GTSemanticLayout:
    return GTSemanticLayout(
        name="current_audit:semantickitti_sequence01_point_projection",
        kind="semantic_kitti_point_projection",
        label_dir=sequence_root / "labels",
        image_dir=sequence_root / "image_2",
        calib_path=sequence_root / "calib.txt",
        velodyne_dir=sequence_root / "velodyne",
        point_label_dir=sequence_root / "labels",
        frame_digits=6,
        suffix=".label",
        semantic_id_encoding="semantic_kitti_uint32_lower16",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", default="/mnt/data/users/chengshun.wang/data/semantickitti_odometry/dataset/sequences/01")
    parser.add_argument("--stage-c-cache-dir", default="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full")
    parser.add_argument("--projection-cache-dir", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/projection_cache/seq01")
    parser.add_argument("--out-dir", default="results/kitti01_hmc_v2/current_code_semantic_vggt4d_audit/gt2d_semantic_visual")
    parser.add_argument("--frames", default="290,464,650")
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument(
        "--masklet-color-mode",
        choices=("semantic_kitti", "group"),
        default="semantic_kitti",
        help="Use the SemanticKITTI palette for comparable masklet labels, or coarse group colors.",
    )
    args = parser.parse_args()

    sequence_root = Path(args.sequence_root)
    stage_c_cache_dir = Path(args.stage_c_cache_dir)
    projection_cache_dir = Path(args.projection_cache_dir)
    out_dir = Path(args.out_dir)
    provider = GTSemanticProvider(_provider_layout(sequence_root))
    rows: List[Dict[str, object]] = []

    for frame in _parse_frames(args.frames):
        row: Dict[str, object] = {"frame": frame}
        image_path = sequence_root / "image_2" / f"{frame:06d}.png"
        sem_path = projection_cache_dir / f"{frame:06d}_sem_sparse.npy"
        valid_path = projection_cache_dir / f"{frame:06d}_valid_mask.npy"
        chunk_dir = _find_chunk(stage_c_cache_dir, frame)
        if not image_path.exists() or not sem_path.exists() or not valid_path.exists() or chunk_dir is None:
            row.update(
                {
                    "status": "missing_inputs",
                    "image_exists": image_path.exists(),
                    "sem_exists": sem_path.exists(),
                    "valid_exists": valid_path.exists(),
                    "chunk_dir": str(chunk_dir) if chunk_dir else "",
                }
            )
            rows.append(row)
            continue

        rgb = Image.open(image_path).convert("RGB")
        sem = np.load(sem_path).astype(np.int32)
        valid = np.load(valid_path).astype(bool)
        provider_frame = provider.load_frame(frame)
        provider_match = bool(
            provider_frame.semantic.shape == sem.shape
            and provider_frame.valid_mask.shape == valid.shape
            and np.array_equal(provider_frame.semantic, sem)
            and np.array_equal(provider_frame.valid_mask, valid)
        )

        payload = torch.load(chunk_dir / "masklet.pt", map_location="cpu")
        manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
        local_idx = int(frame) - int(manifest["start_frame"])
        masklet_overlay, masklet_stats = _overlay_masklets(
            rgb,
            payload,
            local_idx,
            color_mode=str(args.masklet_color_mode),
        )
        gt_overlay = _overlay_points(rgb, sem, valid, radius=max(0, int(args.point_radius)), alpha=0.82)
        out_path = out_dir / f"frame_{frame:06d}_rgb_masklet_gtsem.png"
        _make_three_panel(rgb, masklet_overlay, gt_overlay, out_path)
        row.update(
            {
                "status": "ok",
                "image_path": str(image_path),
                "stage_c_masklet_path": str(chunk_dir / "masklet.pt"),
                "projection_sem_path": str(sem_path),
                "projection_valid_path": str(valid_path),
                "provider_source_path": provider_frame.source_path,
                "provider_cache_exact_match": provider_match,
                "valid_projected_pixels": int(valid.sum()),
                "projected_coverage": float(valid.mean()),
                "top_projected_labels": _label_summary(sem, valid),
                "visual_path": str(out_path),
                "masklet_color_mode": str(args.masklet_color_mode),
                **masklet_stats,
            }
        )
        rows.append(row)

    _write_csv(out_dir / "gt2d_semantic_visual_audit.csv", rows)
    summary = {
        "frames_requested": _parse_frames(args.frames),
        "frames_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "provider_cache_exact_match_all_ok": all(
            bool(row.get("provider_cache_exact_match")) for row in rows if row.get("status") == "ok"
        ),
        "masklet_color_mode": str(args.masklet_color_mode),
        "masklet_color_note": (
            "VideoMasklet labels are mapped to SemanticKITTI colors when possible; "
            "sky is not directly comparable to sparse LiDAR GT."
        ),
        "out_dir": str(out_dir),
        "note": "Projected GT is sparse SemanticKITTI LiDAR-to-image semantic, not dense 2D semantic.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt2d_semantic_visual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["frames_ok"] == len(summary["frames_requested"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
