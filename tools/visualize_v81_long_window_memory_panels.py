#!/usr/bin/env python3
"""Generate v81 long-window visual confirmation panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from visualize_v80_case_pca_qkv_ttt_panels import (  # noqa: E402
    load_rgb,
    radio_boundary_tile,
    semantic_payload,
    semantic_tiles,
    tile_text,
)


DEFAULT_BANK = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase2_long_window_visual_confirmation"
)
DEFAULT_KITTI_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def heat_image(arr: np.ndarray, size: tuple[int, int], high_is_hot: bool = True) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.size == 0:
        return tile_text(["empty map"], size=size, title="Map")
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if amax > amin:
        norm = (arr - amin) / (amax - amin)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
    if not high_is_hot:
        norm = 1.0 - norm
    rgb = np.stack([norm * 255, (1.0 - np.abs(norm - 0.5) * 2.0) * 160, (1.0 - norm) * 255], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize(size, resample=Image.Resampling.NEAREST)


def render_support_map(summary_path: Path, size: tuple[int, int]) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    if not summary_path.is_file():
        missing = tile_text([f"missing selected summary: {summary_path}"], size=size, title="Selected/Support")
        return missing, missing.copy(), {"support_map_rendered": False, "support_map_note": "missing_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        missing = tile_text([f"summary parse failed: {type(exc).__name__}"], size=size, title="Selected/Support")
        return missing, missing.copy(), {"support_map_rendered": False, "support_map_note": type(exc).__name__}
    pt_path = Path(str(summary.get("support_path") or summary.get("source_support_map") or ""))
    if not pt_path.is_file():
        txt = [
            "selected-write summary available",
            f"chunk={summary.get('chunk')}",
            f"selected_low_support={summary.get('selected_low_support_given_selected_runtime')}",
            f"selected_runtime_mass={summary.get('selected_runtime_mass')}",
            f"support pt missing: {pt_path}",
        ]
        tile = tile_text(txt, size=size, title="Selected Summary")
        return tile, tile.copy(), {"support_map_rendered": False, "support_map_note": "support_pt_missing"}
    try:
        payload = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        tile = tile_text([f"support pt load failed: {type(exc).__name__}", str(pt_path)], size=size, title="Selected/Support")
        return tile, tile.copy(), {"support_map_rendered": False, "support_map_note": type(exc).__name__}
    if not isinstance(payload, dict) or not hasattr(payload.get("score_overlap"), "detach"):
        tile = tile_text(["support pt structure unsupported", str(pt_path)], size=size, title="Selected/Support")
        return tile, tile.copy(), {"support_map_rendered": False, "support_map_note": "unsupported_structure"}
    score = payload["score_overlap"].detach().float().cpu().numpy()
    control = payload.get("control_overlap")
    control_arr = control.detach().float().cpu().numpy() if hasattr(control, "detach") else score * 0.0
    tokens = int(payload.get("tokens_per_frame") or score.shape[-1])
    grid_h = 19
    grid_w = max(tokens // grid_h, 1)
    score2 = score.reshape(-1, tokens).mean(axis=0)[: grid_h * grid_w].reshape(grid_h, grid_w)
    control2 = control_arr.reshape(-1, tokens).mean(axis=0)[: grid_h * grid_w].reshape(grid_h, grid_w)
    meta = {
        "support_map_rendered": True,
        "support_map_note": "score_overlap/control_overlap",
        "support_pt": str(pt_path),
        "support_artifact": payload.get("artifact"),
    }
    return heat_image(score2, size, high_is_hot=False), heat_image(control2, size, high_is_hot=False), meta


def filmstrip(kitti_root: Path, seq: str, chunks: Sequence[int], size: tuple[int, int]) -> Image.Image:
    tile_w = size[0] // max(len(chunks), 1)
    out = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for i, chunk in enumerate(chunks):
        frame = int(chunk) * 29 + 16
        img = load_rgb(kitti_root, seq, frame, (tile_w, size[1]))
        x = i * tile_w
        out.paste(img, (x, 0))
        draw.rectangle((x, 0, x + 70, 18), fill=(0, 0, 0))
        draw.text((x + 3, 3), f"c{chunk} f{frame}", fill=(255, 255, 255), font=font)
    return out


def make_panel(row: dict[str, str], out_dir: Path, kitti_root: Path, preprocess_root: Path) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    start = int(row["chunk_start"])
    end = int(row["chunk_end"])
    center = int(row["center_chunk"])
    chunks = list(range(start, end + 1))
    frame = center * 29 + 16
    tile = (420, 250)
    rgb = filmstrip(kitti_root, seq, chunks, tile)
    sem_payload = semantic_payload(preprocess_root, seq, center)
    sem_img, conf_img, role_img, sem_meta = semantic_tiles(sem_payload, frame, tile)
    radio = radio_boundary_tile(preprocess_root, seq, center, frame, tile)
    sources = parse_json_list(row.get("selected_chunk_sources", ""))
    support_summary = Path(sources[0]) if sources else Path("")
    support_img, random_img, support_meta = render_support_map(support_summary, tile)
    metrics = [
        f"window={row.get('window_id')}",
        f"case={row.get('case_type')} reason={row.get('target_reason')}",
        f"J_long={row.get('J_long')}",
        f"rmse={row.get('window5_joint_sim3_rmse')}",
        f"scale_cv={row.get('window5_subchunk_scale_cv')}",
        f"future={row.get('downstream_future_consistency')}",
        f"sel_low_ratio={row.get('selected_low_support_ratio')}",
        f"cluster_len={row.get('continuous_low_support_cluster_len')}",
        f"direction={row.get('selected_minus_control_downstream_direction')}",
        f"stable/harm/context={row.get('stable_mass')}/{row.get('harm_mass')}/{row.get('context_mass')}",
        f"has_radio={row.get('has_radio')} has_ttt_post_delta={row.get('has_ttt_post_delta')}",
        "TTT operator/update/final: source post-delta availability only; no synthetic map.",
        "READ/SWA confirmed maps: Phase4 not built yet.",
    ]
    metric_tile = tile_text(metrics, size=tile, title="Long-Window Evidence")

    panel = Image.new("RGB", (tile[0] * 3, tile[1] * 3), (255, 255, 255))
    placements = [
        (rgb, (0, 0)),
        (sem_img, (tile[0], 0)),
        (conf_img, (tile[0] * 2, 0)),
        (role_img, (0, tile[1])),
        (radio, (tile[0], tile[1])),
        (support_img, (tile[0] * 2, tile[1])),
        (random_img, (0, tile[1] * 2)),
        (metric_tile, (tile[0], tile[1] * 2)),
        (tile_text(["Selected summary sources:", *sources[:8]], size=tile, title="Artifact Provenance"), (tile[0] * 2, tile[1] * 2)),
    ]
    for img, xy in placements:
        panel.paste(img, xy)

    subdir = out_dir / "long_window_panels"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"{row['window_id']}_{row['case_type']}.png"
    panel.save(path)
    # Also create plan-required subdirectory aliases with the real rendered maps.
    sw_dir = out_dir / "selected_write_vs_random_panels"
    ds_dir = out_dir / "downstream_direction_panels"
    sw_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    support_img.save(sw_dir / f"{row['window_id']}_selected_support.png")
    metric_tile.save(ds_dir / f"{row['window_id']}_downstream_direction.png")

    return {
        "seq": seq,
        "window_id": row["window_id"],
        "case_type": row["case_type"],
        "chunk_start": start,
        "chunk_end": end,
        "center_chunk": center,
        "visual_file": str(path),
        "width": panel.width,
        "height": panel.height,
        "sha256": sha256_file(path),
        "semantic_panel_available": sem_meta.get("semantic_panel_available"),
        "support_map_rendered": support_meta.get("support_map_rendered"),
        "support_map_note": support_meta.get("support_map_note"),
        "support_artifact": support_meta.get("support_artifact"),
        "has_radio": row.get("has_radio"),
        "has_ttt_post_delta": row.get("has_ttt_post_delta"),
        "selected_low_support_ratio": row.get("selected_low_support_ratio"),
        "continuous_low_support_cluster_len": row.get("continuous_low_support_cluster_len"),
        "selected_minus_control_downstream_direction": row.get("selected_minus_control_downstream_direction"),
        "selected_chunk_sources": row.get("selected_chunk_sources"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    args = parser.parse_args()
    rows = read_rows(args.bank)
    manifest = [make_panel(row, args.out_dir, args.kitti_root, args.preprocess_root) for row in rows]
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    print(json.dumps({"out_dir": str(args.out_dir), "panel_count": len(manifest)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
