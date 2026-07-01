#!/usr/bin/env python3
"""Generate ACL2 v80 Phase2 case visual panels.

Panels use real KITTI RGB frames and real stage-c semantic/confidence tensors.
Missing direct QKV/SWA/TTT hook dumps are rendered as explicit availability
notes rather than synthetic visuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CASE_BANK_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase2_case_visual_confirmation"
)
DEFAULT_KITTI_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_ARTIFACT_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)

STABLE_WORDS = ("building", "house", "wall", "fence", "handrail", "pole", "traffic sign", "bridge", "construction", "billboard", "pillar")
DYNAMIC_WORDS = ("car", "person", "rider", "bicycle", "motorcycle", "bus", "truck", "train", "dog")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "terrain", "void", "unknown", "plant")
CONTEXT_WORDS = ("sky", "road", "ground", "sidewalk", "path", "crosswalk")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel_or_str(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def tile_text(lines: Sequence[str], size: tuple[int, int] = (420, 280), title: str | None = None) -> Image.Image:
    img = Image.new("RGB", size, (248, 248, 246))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 10
    if title:
        draw.text((10, y), title, fill=(20, 20, 20), font=font)
        y += 20
    for line in lines:
        for part in [line[i : i + 64] for i in range(0, len(line), 64)]:
            draw.text((10, y), part, fill=(30, 30, 30), font=font)
            y += 15
            if y > size[1] - 18:
                return img
    return img


def load_rgb(kitti_root: Path, seq: str, frame: int, size: tuple[int, int]) -> Image.Image:
    path = kitti_root / "sequences" / seq / "image_2" / f"{int(frame):06d}.png"
    if not path.is_file():
        return tile_text([f"missing RGB: {path}"], size=size, title="RGB")
    return Image.open(path).convert("RGB").resize(size)


def find_chunk_dir(root: Path, chunk_id: int) -> Path | None:
    matches = sorted(root.glob(f"chunk_{int(chunk_id):03d}_*"))
    return matches[0] if matches else None


def label_ids(names: Sequence[Any], words: Sequence[str]) -> set[int]:
    lowered = [str(name).lower() for name in names]
    return {idx for idx, name in enumerate(lowered) if any(word in name for word in words)}


def semantic_payload(preprocess_root: Path, seq: str, chunk: int) -> dict[str, Any]:
    root = preprocess_root / seq / "stage_c_cache_semantic_chunks"
    chunk_dir = find_chunk_dir(root, chunk)
    if chunk_dir is None:
        return {"available": False, "error": "missing_stage_c_chunk_dir"}
    path = chunk_dir / "masklet.pt"
    if not path.is_file():
        return {"available": False, "error": "missing_masklet_pt"}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"available": False, "error": type(exc).__name__}
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict) or not hasattr(sem.get("label_maps"), "detach"):
        return {"available": False, "error": "missing_label_maps"}
    return {"available": True, "sem": sem}


def palette(ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((int(ids.max()) + 1 if ids.size else 1, 3), dtype=np.uint8)
    for idx in range(colors.shape[0]):
        colors[idx] = ((37 * idx + 31) % 255, (73 * idx + 59) % 255, (109 * idx + 83) % 255)
    colors[0] = (0, 0, 0)
    return colors


def semantic_tiles(payload: dict[str, Any], frame: int, size: tuple[int, int]) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    if not payload.get("available"):
        missing = tile_text([f"semantic unavailable: {payload.get('error')}"], size=size, title="Semantic")
        return missing, missing.copy(), missing.copy(), {"semantic_panel_available": False}
    sem = payload["sem"]
    labels = sem["label_maps"].detach().cpu().numpy()
    start = int(sem.get("global_start_frame", 0))
    idx = int(np.clip(int(frame) - start, 0, labels.shape[0] - 1))
    lab = labels[idx].astype(np.int64)
    pal = palette(lab)
    sem_img = Image.fromarray(pal[lab]).resize(size, resample=Image.Resampling.NEAREST)
    conf = sem.get("confidence_maps")
    if hasattr(conf, "detach"):
        arr = conf.detach().cpu().numpy()[idx].astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        conf_rgb = np.stack([arr * 255, arr * 255, arr * 255], axis=-1).astype(np.uint8)
        conf_img = Image.fromarray(conf_rgb).resize(size)
    else:
        conf_img = tile_text(["missing confidence_maps"], size=size, title="Confidence")
    names = sem.get("label_names", [])
    stable = label_ids(names, STABLE_WORDS)
    dynamic = label_ids(names, DYNAMIC_WORDS)
    lowtrust = label_ids(names, LOWTRUST_WORDS)
    context = label_ids(names, CONTEXT_WORDS)
    role = np.zeros((*lab.shape, 3), dtype=np.uint8)
    role[np.isin(lab, list(stable))] = (40, 180, 80)
    role[np.isin(lab, list(dynamic))] = (220, 70, 60)
    role[np.isin(lab, list(lowtrust))] = (170, 120, 30)
    role[np.isin(lab, list(context))] = (70, 120, 220)
    role_img = Image.fromarray(role).resize(size, resample=Image.Resampling.NEAREST)
    return sem_img, conf_img, role_img, {"semantic_panel_available": True}


def radio_boundary_tile(preprocess_root: Path, seq: str, chunk: int, frame: int, size: tuple[int, int]) -> Image.Image:
    for pattern in ("radseg_sidecar_chunks*", "radio_sidecar_chunks*"):
        for radio_root in sorted((preprocess_root / seq).glob(pattern)):
            chunk_dir = find_chunk_dir(radio_root, chunk)
            if chunk_dir is None:
                continue
            path = chunk_dir / "radio_sidecar.pt"
            if not path.is_file():
                continue
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                continue
            score = payload.get("object_boundary_score") if isinstance(payload, dict) else None
            if not hasattr(score, "detach"):
                continue
            start = int(payload.get("global_start_frame", chunk * 29))
            idx = int(np.clip(int(frame) - start, 0, score.shape[0] - 1))
            arr = score.detach().float().cpu().numpy()[idx]
            arr = np.clip(arr / max(float(arr.max()), 1e-6), 0.0, 1.0)
            rgb = np.stack([arr * 255, arr * 96, arr * 32], axis=-1).astype(np.uint8)
            return Image.fromarray(rgb).resize(size)
    return tile_text(["RADIO boundary unavailable for this seq/chunk"], size=size, title="RADIO Boundary")


def classify_direct_artifact(path: Path) -> str:
    text = str(path).lower()
    name = path.name.lower()
    if "read_cue_patch_dumps" in text:
        return "read_cue_patch_dumps"
    if "ttt_spatial_post_delta_maps" in text:
        return "ttt_spatial_post_delta_maps"
    if "pca_features" in text:
        return "pca_features"
    if name == "hmc_state_hash.jsonl":
        return "hmc_state_hash"
    if name == "merge_state_trace.jsonl":
        return "merge_state_trace"
    if "ttt" in name and name.endswith(".png"):
        return "ttt_visual_png"
    return "other"


def build_direct_artifact_index(roots: Sequence[Path], max_artifacts: int = 20000) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    dir_names = {"read_cue_patch_dumps", "pca_features", "ttt_spatial_post_delta_maps"}
    file_names = {"hmc_state_hash.jsonl", "merge_state_trace.jsonl"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(entries) >= max_artifacts:
                return entries
            if path.is_dir() and path.name in dir_names:
                entries.append({"path": path, "text": str(path).lower(), "kind": classify_direct_artifact(path)})
            elif path.is_file() and (path.name in file_names or ("TTT" in path.name and path.suffix.lower() == ".png")):
                entries.append({"path": path, "text": str(path).lower(), "kind": classify_direct_artifact(path)})
    return entries


def case_chunk_tokens(chunks: Sequence[int]) -> list[str]:
    tokens: list[str] = []
    for chunk in chunks:
        c = int(chunk)
        tokens.extend([f"chunk{c:03d}", f"chunk{c:02d}"])
    if len(chunks) >= 2:
        start, end = int(min(chunks)), int(max(chunks))
        tokens.extend([f"chunks{start:03d}_{end:03d}", f"chunks{start:02d}_{end:02d}"])
    return sorted(dict.fromkeys(tokens))


def direct_artifacts_for_case(
    memory: str,
    seq: str,
    chunks: Sequence[int],
    index: Sequence[dict[str, Any]],
    base: Path,
    limit: int = 8,
) -> dict[str, Any]:
    seq_token = f"seq{str(seq).zfill(2)}"
    chunk_tokens = case_chunk_tokens(chunks)
    kinds: set[str] = set()
    paths: list[str] = []
    for entry in index:
        text = str(entry["text"])
        if seq_token not in text:
            continue
        if not any(token in text for token in chunk_tokens):
            continue
        kind = str(entry["kind"])
        if memory == "short" and kind not in {"read_cue_patch_dumps", "pca_features", "hmc_state_hash"}:
            continue
        if memory == "mid" and kind not in {"read_cue_patch_dumps", "pca_features", "merge_state_trace", "hmc_state_hash"}:
            continue
        if memory == "long" and kind not in {"ttt_spatial_post_delta_maps", "pca_features", "ttt_visual_png", "hmc_state_hash"}:
            continue
        kinds.add(kind)
        if len(paths) < limit:
            paths.append(rel_or_str(Path(entry["path"]), base))
    return {
        "direct_qkv_ttt_artifact_available": bool(paths),
        "direct_artifact_type_count": len(kinds),
        "direct_artifact_types": ";".join(sorted(kinds)),
        "direct_artifact_paths_sample": ";".join(paths),
    }


def case_id(memory: str, row: dict[str, Any]) -> str:
    if memory == "short":
        return f"{row['seq']}_chunk{int(row['chunk_id']):03d}_{row['case_type']}"
    if memory == "mid":
        return f"{row['seq']}_pair{int(row['prev_chunk']):03d}_{int(row['curr_chunk']):03d}_{row['case_type']}"
    return f"{row['seq']}_win{int(row['chunk_start']):03d}_{int(row['chunk_end']):03d}_{row['case_type']}"


def select_frame_and_chunk(memory: str, row: dict[str, Any]) -> tuple[int, int]:
    frame = int(round((int(float(row["frame_start"])) + int(float(row["frame_end"]))) / 2))
    if memory == "short":
        chunk = int(float(row["chunk_id"]))
    elif memory == "mid":
        frame = int(float(row.get("boundary_frame") or frame))
        chunk = int(float(row["curr_chunk"]))
    else:
        chunk = int(round((int(float(row["chunk_start"])) + int(float(row["chunk_end"]))) / 2))
    return frame, chunk


def filmstrip(kitti_root: Path, seq: str, chunks: Sequence[int], size: tuple[int, int]) -> Image.Image:
    tile_w = size[0] // max(len(chunks), 1)
    out = Image.new("RGB", size, (245, 245, 245))
    for i, chunk in enumerate(chunks):
        frame = int(chunk) * 29 + 16
        img = load_rgb(kitti_root, seq, frame, (tile_w, size[1]))
        out.paste(img, (i * tile_w, 0))
    return out


def make_panel(
    memory: str,
    row: dict[str, Any],
    out_dir: Path,
    kitti_root: Path,
    preprocess_root: Path,
    artifact_index: Sequence[dict[str, Any]],
    artifact_base: Path,
) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    frame, chunk = select_frame_and_chunk(memory, row)
    tile_size = (420, 280)
    if memory == "long":
        chunks = list(range(int(float(row["chunk_start"])), int(float(row["chunk_end"])) + 1))
        rgb = filmstrip(kitti_root, seq, chunks, tile_size)
    else:
        rgb = load_rgb(kitti_root, seq, frame, tile_size)
    sem_payload = semantic_payload(preprocess_root, seq, chunk)
    sem_img, conf_img, role_img, sem_meta = semantic_tiles(sem_payload, frame, tile_size)
    radio_img = radio_boundary_tile(preprocess_root, seq, chunk, frame, tile_size)
    metrics = [f"memory={memory}", f"case={row.get('case_type')}", f"seq={seq}", f"frame={frame}", f"chunk={chunk}"]
    for key in (
        "J_short",
        "J_mid",
        "J_long",
        "local_sim3_ate",
        "future_after_overlap",
        "window5_joint_sim3_rmse",
        "scale_cv",
        "window5_subchunk_scale_cv",
        "TTT_update_conflict",
    ):
        value = row.get(key)
        if value not in (None, ""):
            metrics.append(f"{key}={value}")
    if memory == "long":
        direct_chunks = list(range(int(float(row["chunk_start"])), int(float(row["chunk_end"])) + 1))
    elif memory == "mid":
        direct_chunks = [int(float(row["prev_chunk"])), int(float(row["curr_chunk"]))]
    else:
        direct_chunks = [int(float(row["chunk_id"]))]
    direct_meta = direct_artifacts_for_case(memory, seq, direct_chunks, artifact_index, artifact_base)
    if direct_meta["direct_qkv_ttt_artifact_available"]:
        direct = [
            "Direct artifact availability:",
            f"types={direct_meta['direct_artifact_types']}",
            "samples:",
            *str(direct_meta["direct_artifact_paths_sample"]).split(";")[:6],
            "These are existing artifacts only; no hook maps are synthesized.",
        ]
    else:
        direct = [
            "Direct hook availability:",
            "Q/K/V PCA dump: missing for this case",
            "READ selected/random masks: missing for this case",
            "SWA cache/current K/V: missing for this case",
            "TTT operator/update/final maps: missing for this case",
            "This tile is an availability note, not a fake hook visualization.",
        ]
    metric_tile = tile_text(metrics, size=tile_size, title="Case Metrics")
    direct_tile = tile_text(direct, size=tile_size, title="QKV/TTT Availability")
    panel = Image.new("RGB", (tile_size[0] * 3, tile_size[1] * 2), (255, 255, 255))
    for img, xy in (
        (rgb, (0, 0)),
        (sem_img, (tile_size[0], 0)),
        (conf_img, (tile_size[0] * 2, 0)),
        (role_img, (0, tile_size[1])),
        (radio_img, (tile_size[0], tile_size[1])),
        (metric_tile if memory != "long" else direct_tile, (tile_size[0] * 2, tile_size[1])),
    ):
        panel.paste(img, xy)
    if memory != "long":
        # Replace a small strip at the bottom of the metrics tile with direct-hook notes.
        draw = ImageDraw.Draw(panel)
        draw.rectangle((tile_size[0] * 2, tile_size[1] * 2 - 60, tile_size[0] * 3, tile_size[1] * 2), fill=(248, 248, 246))
        note = (
            "Direct artifacts found; see manifest sample paths."
            if direct_meta["direct_qkv_ttt_artifact_available"]
            else "Direct QKV/SWA/TTT dumps missing; no fake hook maps."
        )
        draw.text((tile_size[0] * 2 + 8, tile_size[1] * 2 - 54), note, fill=(30, 30, 30), font=ImageFont.load_default())
    panel_dir = out_dir / f"{memory}_case_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rel_name = f"{case_id(memory, row)}.png"
    path = panel_dir / rel_name
    panel.save(path)
    return {
        "memory_body": memory,
        "case_type": row.get("case_type"),
        "seq": seq,
        "case_id": case_id(memory, row),
        "frame": frame,
        "chunk": chunk,
        "visual_file": str(path),
        "width": panel.width,
        "height": panel.height,
        "sha256": sha256_file(path),
        "semantic_panel_available": sem_meta["semantic_panel_available"],
        **direct_meta,
        "availability_note": (
            "matched existing direct memory-hook artifacts by seq/chunk; panel still shows real RGB/semantic/confidence/role/geometry evidence"
            if direct_meta["direct_qkv_ttt_artifact_available"]
            else "direct QKV/SWA/TTT hook dumps unavailable; panel shows real RGB/semantic/confidence/role/geometry evidence plus missing-artifact note"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank-dir", type=Path, default=DEFAULT_CASE_BANK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--artifact-root", action="append", type=Path, default=[DEFAULT_ARTIFACT_ROOT])
    parser.add_argument("--max-direct-artifacts", type=int, default=20000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_index = build_direct_artifact_index(args.artifact_root, max_artifacts=int(args.max_direct_artifacts))
    rows_by_memory = {
        "short": read_rows(args.case_bank_dir / "short_single_chunk_cases.csv"),
        "mid": read_rows(args.case_bank_dir / "mid_adjacent_pair_cases.csv"),
        "long": read_rows(args.case_bank_dir / "long_five_chunk_cases.csv"),
    }
    manifest: list[dict[str, Any]] = []
    for memory, rows in rows_by_memory.items():
        for row in rows:
            manifest.append(make_panel(memory, row, args.out_dir, args.kitti_root, args.preprocess_root, artifact_index, Path.cwd()))
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    print(json.dumps({"out_dir": str(args.out_dir), "panel_count": len(manifest), "direct_artifact_index_count": len(artifact_index)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
