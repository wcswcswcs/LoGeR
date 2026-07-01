#!/usr/bin/env python3
"""Build RGB/semantic/risk overlay panels for ACL2 v102 base cases.

This script uses real KITTI RGB frames and Stage-C semantic chunk caches.  It is
still conservative: the generated panels do not include a local point/trajectory
error map, so they are not marked as strict Stage-1 visual passes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
BASE_ROWS = ROOT / "stage2_base_case_selection/base_case_rows.csv"
OUT = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_panels"
MANIFEST = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_manifest.csv"
SUMMARY = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_summary.json"
KITTI_RGB_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
STAGE_C_ROOT = Path("results/kitti_preprocess")

DYNAMIC_LABELS = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def find_chunk_dir(seq: str, chunk_idx: int) -> Path | None:
    root = STAGE_C_ROOT / seq / "stage_c_cache_semantic_chunks"
    matches = sorted(root.glob(f"chunk_{chunk_idx:03d}_*"))
    return matches[0] if matches else None


def load_chunk(seq: str, chunk_idx: int) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    chunk_dir = find_chunk_dir(seq, chunk_idx)
    if chunk_dir is None:
        return None
    manifest_path = chunk_dir / "manifest.json"
    masklet_path = chunk_dir / "masklet.pt"
    if not manifest_path.is_file() or not masklet_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(masklet_path, map_location="cpu")
    return chunk_dir, manifest, payload


def semantic_colors(max_label: int) -> np.ndarray:
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for label in range(max_label + 1):
        colors[label] = [
            (37 * label + 17) % 255,
            (67 * label + 43) % 255,
            (97 * label + 89) % 255,
        ]
    colors[0] = [0, 0, 0]
    return colors


def resize_rgb(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def edge_mask(label_map: np.ndarray) -> np.ndarray:
    edge = np.zeros(label_map.shape, dtype=bool)
    edge[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    return edge


def dynamic_mask(label_map: np.ndarray, label_names: list[str]) -> np.ndarray:
    dynamic_ids = {idx for idx, name in enumerate(label_names) if str(name).lower() in DYNAMIC_LABELS}
    if not dynamic_ids:
        return np.zeros(label_map.shape, dtype=bool)
    out = np.zeros(label_map.shape, dtype=bool)
    for idx in dynamic_ids:
        out |= label_map == idx
    return out


def stable_anchor_masks(prev: dict[str, Any], curr: dict[str, Any], prev_i: int, curr_i: int) -> tuple[np.ndarray, np.ndarray, int]:
    prev_ids = as_list(prev.get("seed_global_track_idx"))
    curr_ids = as_list(curr.get("seed_global_track_idx"))
    prev_visible = prev["V_mask"][:, prev_i].bool().cpu().numpy()
    curr_visible = curr["V_mask"][:, curr_i].bool().cpu().numpy()
    curr_index = {int(seed): idx for idx, seed in enumerate(curr_ids)}
    common: list[tuple[int, int]] = []
    for pidx, seed in enumerate(prev_ids):
        cidx = curr_index.get(int(seed))
        if cidx is None:
            continue
        if prev_visible[pidx] and curr_visible[cidx]:
            common.append((pidx, cidx))
    h, w = prev["M_mask"].shape[-2:]
    prev_mask = np.zeros((h, w), dtype=bool)
    curr_mask = np.zeros((h, w), dtype=bool)
    for pidx, cidx in common:
        prev_mask |= prev["M_mask"][pidx, prev_i].bool().cpu().numpy()
        curr_mask |= curr["M_mask"][cidx, curr_i].bool().cpu().numpy()
    return prev_mask, curr_mask, len(common)


def overlay_rgb(rgb: np.ndarray, label_map: np.ndarray, stable: np.ndarray, risk: np.ndarray) -> np.ndarray:
    colors = semantic_colors(int(label_map.max()))
    sem_rgb = colors[label_map]
    out = (0.58 * rgb.astype(np.float32) + 0.42 * sem_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    out[stable] = (0.45 * out[stable] + 0.55 * np.array([0, 255, 80])).astype(np.uint8)
    out[risk] = (0.45 * out[risk] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def panel_for_case(row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    seq = row["seq"]
    prev_idx = int(row["prev_chunk"])
    curr_idx = int(row["curr_chunk"])
    prev_loaded = load_chunk(seq, prev_idx)
    curr_loaded = load_chunk(seq, curr_idx)
    if prev_loaded is None or curr_loaded is None:
        return {"case_id": case_id, "status": "missing_stage_c_chunk", "strict_visual_panel": False}
    prev_dir, prev_manifest, prev = prev_loaded
    curr_dir, curr_manifest, curr = curr_loaded
    prev_start, prev_end = int(prev_manifest["start_frame"]), int(prev_manifest["end_frame"])
    curr_start, curr_end = int(curr_manifest["start_frame"]), int(curr_manifest["end_frame"])
    overlap_start = max(prev_start, curr_start)
    overlap_end = min(prev_end, curr_end)
    if overlap_start < overlap_end:
        prev_frame_id = overlap_start
        curr_frame_id = overlap_start
        overlap_frame_available = True
    else:
        prev_frame_id = max(prev_start, prev_end - 1)
        curr_frame_id = curr_start
        overlap_frame_available = False
    prev_i = max(0, min(prev_frame_id - prev_start, int(prev_manifest["chunk_shape"][0]) - 1))
    curr_i = max(0, min(curr_frame_id - curr_start, int(curr_manifest["chunk_shape"][0]) - 1))

    prev_sem = prev["semantic_segmentation"]
    curr_sem = curr["semantic_segmentation"]
    prev_label = prev_sem["label_maps"][prev_i].cpu().numpy()
    curr_label = curr_sem["label_maps"][curr_i].cpu().numpy()
    prev_conf = prev_sem["confidence_maps"][prev_i].cpu().numpy()
    curr_conf = curr_sem["confidence_maps"][curr_i].cpu().numpy()
    prev_stable, curr_stable, stable_count = stable_anchor_masks(prev, curr, prev_i, curr_i)

    prev_risk = dynamic_mask(prev_label, as_list(prev_sem.get("label_names"))) | edge_mask(prev_label) | (prev_conf < 0.45)
    curr_risk = dynamic_mask(curr_label, as_list(curr_sem.get("label_names"))) | edge_mask(curr_label) | (curr_conf < 0.45)

    prev_rgb_path = KITTI_RGB_ROOT / seq / "image_2" / f"{prev_frame_id:06d}.png"
    curr_rgb_path = KITTI_RGB_ROOT / seq / "image_2" / f"{curr_frame_id:06d}.png"
    missing_rgb = [p.as_posix() for p in (prev_rgb_path, curr_rgb_path) if not p.is_file()]
    if missing_rgb:
        return {
            "case_id": case_id,
            "status": "missing_rgb_frame",
            "missing_rgb_paths": ";".join(missing_rgb),
            "strict_visual_panel": False,
        }
    prev_rgb = resize_rgb(prev_rgb_path, prev_label.shape)
    curr_rgb = resize_rgb(curr_rgb_path, curr_label.shape)
    prev_overlay = overlay_rgb(prev_rgb, prev_label, prev_stable, prev_risk)
    curr_overlay = overlay_rgb(curr_rgb, curr_label, curr_stable, curr_risk)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{case_id}_rgb_semantic_overlay.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes[0, 0].imshow(prev_rgb)
    axes[0, 0].set_title(f"{case_id} prev chunk {prev_idx} RGB frame {prev_frame_id}")
    axes[0, 1].imshow(prev_overlay)
    axes[0, 1].set_title("prev semantic + stable(green) + risk(red)")
    axes[1, 0].imshow(curr_overlay)
    axes[1, 0].set_title(f"curr semantic + stable(green) + risk(red), frame {curr_frame_id}")
    metric_text = "\n".join(
        [
            f"L3={row.get('L3_handoff_transfer_penalty_proxy', '')}",
            f"adj_log_scale={row.get('L3_adjacent_log_scale_jump', '')}",
            f"source={row.get('primary_drift_source', '')}",
            f"overlap_frame_available={overlap_frame_available}",
            f"stable_common_seed_count={stable_count}",
            "missing: local point/trajectory error map",
        ]
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "case_id": case_id,
        "seq": seq,
        "prev_chunk": prev_idx,
        "curr_chunk": curr_idx,
        "prev_frame_id": prev_frame_id,
        "curr_frame_id": curr_frame_id,
        "overlap_frame_available": overlap_frame_available,
        "prev_rgb_path": prev_rgb_path.as_posix(),
        "curr_rgb_path": curr_rgb_path.as_posix(),
        "prev_chunk_dir": prev_dir.as_posix(),
        "curr_chunk_dir": curr_dir.as_posix(),
        "panel_path": out_path.as_posix(),
        "stable_common_seed_count": stable_count,
        "prev_dynamic_boundary_lowconf_frac": float(prev_risk.mean()),
        "curr_dynamic_boundary_lowconf_frac": float(curr_risk.mean()),
        "rgb_semantic_overlay_available": True,
        "strict_visual_panel": False,
        "strict_blocker": "local point/trajectory error map is not materialized in the inspected artifacts",
        "status": "rgb_semantic_risk_overlay_built_not_strict",
    }


def main() -> int:
    rows = read_rows(BASE_ROWS)
    manifest_rows = []
    for row in rows:
        try:
            manifest_rows.append(panel_for_case(row))
        except Exception as exc:  # noqa: BLE001
            manifest_rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "seq": row.get("seq", ""),
                    "prev_chunk": row.get("prev_chunk", ""),
                    "curr_chunk": row.get("curr_chunk", ""),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "strict_visual_panel": False,
                }
            )
    write_rows(MANIFEST, manifest_rows)
    summary = {
        "case_count": len(manifest_rows),
        "built_count": sum(1 for r in manifest_rows if r.get("rgb_semantic_overlay_available")),
        "strict_visual_count": sum(1 for r in manifest_rows if r.get("strict_visual_panel")),
        "manifest": MANIFEST.as_posix(),
        "out_dir": OUT.as_posix(),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["built_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
