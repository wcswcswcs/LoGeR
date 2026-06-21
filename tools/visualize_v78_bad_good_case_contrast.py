#!/usr/bin/env python3
"""Build bad-vs-reference visual contrasts for v78 targeting.

The input tables are diagnostic rankings from build_v78_bad_window_tables.py.
This tool pairs high-error cases with low-error reference cases from the same
run when possible, then renders RGB/semantic/confidence strips. "Reference" is
not an HMC gate success claim; it only means lower error under the table's
ranking metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_RGB_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")


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


FAMILIES = {
    "single_chunk": {
        "table": "bad_single_chunk_table.csv",
        "metric": "local_sim3_rmse_m",
        "start": "chunk_start_frame",
        "end": "chunk_end_frame",
        "id": "chunk_id",
        "sort_desc": True,
    },
    "adjacent_pair": {
        "table": "bad_adjacent_chunk_pair_table.csv",
        "metric": "tail3_to_future_from_boundary_sim3_rmse_m",
        "start": "pair_start_frame",
        "end": "pair_end_frame",
        "id": "chunk_pair",
        "boundary": "boundary_frame",
        "sort_desc": True,
    },
    "five_chunk": {
        "table": "bad_5chunk_window_table.csv",
        "metric": "window5_joint_sim3_rmse_m",
        "start": "window_start_frame",
        "end": "window_end_frame",
        "id": "window_chunks",
        "sort_desc": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-window-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--unique-scene", action="store_true", help="Skip duplicate sequence/window scenes across different runs.")
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--panel-height", type=int, default=96)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _case_key(row: dict[str, str], family: str) -> str:
    spec = FAMILIES[family]
    return f"{row.get('run')}:{row.get('sequence')}:{row.get(spec['id'], '')}"


def _scene_key(row: dict[str, str], family: str) -> str:
    spec = FAMILIES[family]
    return f"{row.get('sequence')}:{row.get(spec['id'], '')}:{row.get(spec['start'], '')}:{row.get(spec['end'], '')}"


def _sort_rows(rows: list[dict[str, str]], metric: str, *, reverse: bool) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: _finite(row.get(metric)) if _finite(row.get(metric)) is not None else math.inf,
        reverse=reverse,
    )


def _pick_reference(
    rows: list[dict[str, str]],
    bad: dict[str, str],
    family: str,
) -> tuple[dict[str, str], str]:
    metric = FAMILIES[family]["metric"]
    bad_key = _case_key(bad, family)
    bad_value = _finite(bad.get(metric))

    def lower_error(row: dict[str, str]) -> bool:
        value = _finite(row.get(metric))
        return value is not None and (bad_value is None or value < bad_value)

    candidates = [
        row
        for row in rows
        if _case_key(row, family) != bad_key
        and row.get("run") == bad.get("run")
        and row.get("sequence") == bad.get("sequence")
        and lower_error(row)
    ]
    strategy = "same_run_low_error"
    if not candidates:
        candidates = [
            row
            for row in rows
            if _case_key(row, family) != bad_key
            and row.get("sequence") == bad.get("sequence")
            and lower_error(row)
        ]
        strategy = "same_sequence_low_error"
    if not candidates:
        candidates = [
            row
            for row in rows
            if _case_key(row, family) != bad_key and lower_error(row)
        ]
        strategy = "global_low_error"
    if not candidates:
        fallback = [
            row
            for row in rows
            if _case_key(row, family) != bad_key and _finite(row.get(metric)) is not None
        ]
        if not fallback:
            return bad, "self_no_reference_available"
        ref = min(fallback, key=lambda row: float(row[metric]))
        return ref, "best_available_reference_not_lower"
    ref = min(candidates, key=lambda row: float(row[metric]))
    return ref, strategy


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


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ChunkSemanticLoader:
    def __init__(self, preprocess_root: Path) -> None:
        self.preprocess_root = preprocess_root
        self.index_cache: dict[str, list[dict[str, Any]]] = {}
        self.chunk_cache: dict[Path, dict[str, Any]] = {}

    def _index(self, seq: str) -> list[dict[str, Any]]:
        seq = str(seq).zfill(2)
        if seq in self.index_cache:
            return self.index_cache[seq]
        path = self.preprocess_root / seq / "stage_c_cache_semantic_chunks" / "cache_index.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        self.index_cache[seq] = rows
        return rows

    def _chunk_path(self, seq: str, frame: int) -> Path | None:
        seq = str(seq).zfill(2)
        for row in self._index(seq):
            if int(row["start_frame"]) <= int(frame) < int(row["end_frame"]):
                return self.preprocess_root / seq / "stage_c_cache_semantic_chunks" / row["chunk"] / "masklet.pt"
        return None

    def frame(self, seq: str, frame: int) -> tuple[Image.Image | None, Image.Image | None, dict[str, Any]]:
        path = self._chunk_path(seq, frame)
        if path is None or not path.exists():
            return None, None, {"semantic_available": False, "reason": "missing_chunk_cache"}
        if path not in self.chunk_cache:
            self.chunk_cache[path] = _torch_load(path)
        payload = self.chunk_cache[path]
        sem = payload.get("semantic_segmentation", {}) if isinstance(payload, dict) else {}
        label_maps = sem.get("label_maps")
        conf_maps = sem.get("confidence_maps")
        if not torch.is_tensor(label_maps):
            return None, None, {"semantic_available": False, "reason": "missing_label_maps", "path": str(path)}
        global_start = int(sem.get("global_start_frame", payload.get("manifest", {}).get("start_frame", 0)))
        local = max(0, min(int(frame) - global_start, int(label_maps.shape[0]) - 1))
        label = label_maps[local].detach().cpu().long().numpy()
        label_names = [str(x) for x in sem.get("label_names", [])]
        colours = _colour_table(label_names)
        label = np.clip(label, 0, max(0, len(colours) - 1))
        sem_img = Image.fromarray(colours[label], "RGB")
        if torch.is_tensor(conf_maps):
            conf = conf_maps[local].detach().cpu().float().numpy()
            conf_img = _heat(conf)
        else:
            conf_img = None
        return sem_img, conf_img, {"semantic_available": True, "path": str(path), "local_frame": local}


def _heat(arr: np.ndarray) -> Image.Image:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.any():
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
        y = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    else:
        y = np.zeros_like(x, dtype=np.float32)
    rgb = np.zeros((*y.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * y).round().astype(np.uint8)
    rgb[..., 1] = (255 * (1.0 - np.abs(y - 0.5) * 2.0)).round().astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - y)).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _find_rgb(rgb_root: Path, seq: str, frame: int) -> Path | None:
    base = rgb_root / str(seq).zfill(2) / "image_2"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = base / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    return None


def _resize(img: Image.Image | None, size: tuple[int, int], fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    if img is None:
        return Image.new("RGB", size, fill)
    return img.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text[:90], fill=(255, 255, 255))
    return out


def _metric_text(row: dict[str, str], family: str, role: str, strategy: str = "") -> str:
    spec = FAMILIES[family]
    metric = spec["metric"]
    value = _finite(row.get(metric))
    metric_s = "NA" if value is None else f"{value:.6g}"
    start = row.get(spec["start"], "")
    end = row.get(spec["end"], "")
    case_id = row.get(spec["id"], "")
    extra = f" | strategy={strategy}" if strategy else ""
    return (
        f"{role} {family} | seq={row.get('sequence')} run={row.get('run')} "
        f"id={case_id} frames={start}-{end} {metric}={metric_s}{extra}"
    )


def _case_frames(row: dict[str, str], family: str) -> list[int]:
    spec = FAMILIES[family]
    start = int(float(row[spec["start"]]))
    end = int(float(row[spec["end"]]))
    if family == "single_chunk":
        frames = [start, (start + end - 1) // 2, end - 1]
    elif family == "adjacent_pair":
        boundary = int(float(row.get("boundary_frame") or ((start + end) // 2)))
        frames = [start, max(start, boundary - 1), min(end - 1, boundary), end - 1]
    else:
        total = max(1, end - start - 1)
        frames = [start + round(total * frac) for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    out: list[int] = []
    for frame in frames:
        frame = int(frame)
        if frame not in out:
            out.append(frame)
    return out


def _stack_vertical(images: list[Image.Image]) -> Image.Image:
    w = max(img.width for img in images)
    h = sum(img.height for img in images)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


def _stack_horizontal(images: list[Image.Image]) -> Image.Image:
    w = sum(img.width for img in images)
    h = max(img.height for img in images)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def _text_panel(text: str, width: int, height: int = 42) -> Image.Image:
    img = Image.new("RGB", (width, height), (22, 22, 22))
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), text[:180], fill=(245, 245, 245))
    return img


def _case_strip(
    *,
    row: dict[str, str],
    role: str,
    family: str,
    strategy: str,
    rgb_root: Path,
    loader: ChunkSemanticLoader,
    frame_size: tuple[int, int],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    frames = _case_frames(row, family)
    cells: list[Image.Image] = []
    provenance: list[dict[str, Any]] = []
    for frame in frames:
        rgb_path = _find_rgb(rgb_root, row["sequence"], frame)
        rgb_img = Image.open(rgb_path).convert("RGB") if rgb_path else None
        sem_img, conf_img, sem_info = loader.frame(row["sequence"], frame)
        rgb = _label(_resize(rgb_img, frame_size), f"{role} RGB f{frame:06d}")
        sem = _label(_resize(sem_img, frame_size, fill=(35, 35, 35)), "semantic")
        conf = _label(_resize(conf_img, frame_size, fill=(35, 35, 35)), "confidence")
        cells.append(_stack_vertical([rgb, sem, conf]))
        provenance.append(
            {
                "frame": frame,
                "rgb_path": str(rgb_path) if rgb_path else "",
                **sem_info,
            }
        )
    header = _text_panel(_metric_text(row, family, role, strategy), max(1, len(cells)) * frame_size[0])
    return _stack_vertical([header, _stack_horizontal(cells)]), provenance


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


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.out_dir / "bad_good_case_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    loader = ChunkSemanticLoader(args.preprocess_root)
    frame_size = (int(args.panel_width), int(args.panel_height))

    contrast_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    panels_by_family: dict[str, int] = {}

    for family, spec in FAMILIES.items():
        rows = _read_csv(args.bad_window_dir / spec["table"])
        metric = spec["metric"]
        valid_rows = [row for row in rows if _finite(row.get(metric)) is not None]
        ranked_bad_rows = _sort_rows(valid_rows, metric, reverse=True)
        bad_rows: list[dict[str, str]] = []
        seen_scenes: set[str] = set()
        for row in ranked_bad_rows:
            scene_key = _scene_key(row, family)
            if args.unique_scene and scene_key in seen_scenes:
                continue
            seen_scenes.add(scene_key)
            bad_rows.append(row)
            if len(bad_rows) >= int(args.top_k):
                break
        panels_by_family[family] = len(bad_rows)
        for idx, bad in enumerate(bad_rows, start=1):
            ref, strategy = _pick_reference(valid_rows, bad, family)
            bad_strip, bad_prov = _case_strip(
                row=bad,
                role="bad",
                family=family,
                strategy="ranked_high_error",
                rgb_root=args.rgb_root,
                loader=loader,
                frame_size=frame_size,
            )
            ref_strip, ref_prov = _case_strip(
                row=ref,
                role="reference_low_error",
                family=family,
                strategy=strategy,
                rgb_root=args.rgb_root,
                loader=loader,
                frame_size=frame_size,
            )
            width = max(bad_strip.width, ref_strip.width)
            title = _text_panel(
                f"{family} contrast {idx}: bad metric {bad.get(metric)} vs reference {ref.get(metric)}",
                width,
                height=38,
            )
            panel = _stack_vertical([title, bad_strip, ref_strip])
            case_name = _safe(f"{family}_{idx}_{bad.get('sequence')}_{bad.get('run')}_{bad.get(spec['id'], '')}")
            out_path = panel_dir / f"{case_name}.png"
            panel.save(out_path)
            stats = _image_stats(out_path)
            bad_value = _finite(bad.get(metric))
            ref_value = _finite(ref.get(metric))
            contrast_ratio = None
            if bad_value is not None and ref_value is not None and abs(ref_value) > 1e-12:
                contrast_ratio = bad_value / ref_value
            common = {
                "family": family,
                "contrast_rank": idx,
                "metric": metric,
                "bad_metric_value": bad_value,
                "reference_metric_value": ref_value,
                "bad_to_reference_ratio": contrast_ratio,
                "reference_strategy": strategy,
                "visual_file": str(out_path),
            }
            for role, row, prov in (("bad", bad, bad_prov), ("reference", ref, ref_prov)):
                contrast_row = dict(common)
                contrast_row.update(
                    {
                        "role": role,
                        "run": row.get("run", ""),
                        "sequence": row.get("sequence", ""),
                        "case_id": row.get(spec["id"], ""),
                        "frame_start": row.get(spec["start"], ""),
                        "frame_end": row.get(spec["end"], ""),
                        "trajectory": row.get("trajectory", ""),
                        "sample_frames": json.dumps([p["frame"] for p in prov]),
                        "source_assets": json.dumps(prov, ensure_ascii=False),
                    }
                )
                contrast_rows.append(contrast_row)
            manifest = {
                "visual_file": str(out_path),
                "family": family,
                "contrast_rank": idx,
                "metric": metric,
                "bad_case": _case_key(bad, family),
                "reference_case": _case_key(ref, family),
                "reference_strategy": strategy,
                "rgb_overlay_present": True,
                "semantic_overlay_present": True,
                "confidence_overlay_present": True,
                "trajectory_metric_present": True,
                **stats,
            }
            manifest_rows.append(manifest)

    _write_csv(args.out_dir / "bad_good_case_contrast.csv", contrast_rows)
    _write_csv(args.out_dir / "visual_artifact_manifest.csv", manifest_rows)
    summary = {
        "schema": "acl2_v78_bad_good_case_contrast_v1",
        "diagnostic_only": True,
        "bad_window_dir": str(args.bad_window_dir),
        "out_dir": str(args.out_dir),
        "top_k": int(args.top_k),
        "unique_scene": bool(args.unique_scene),
        "definition_of_reference": "low-error row under the same ranking metric; not an HMC gate success claim",
        "panels_by_family": panels_by_family,
        "num_contrast_rows": len(contrast_rows),
        "num_visual_files": len(manifest_rows),
        "all_visual_files_exist": all(Path(row["visual_file"]).exists() for row in manifest_rows),
        "all_nonempty": all(bool(row["nonempty_image"]) for row in manifest_rows),
    }
    _json_write(args.out_dir / "bad_good_case_contrast_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
