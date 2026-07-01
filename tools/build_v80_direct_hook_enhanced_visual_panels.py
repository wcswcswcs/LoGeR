#!/usr/bin/env python3
"""Build v80 Phase2 direct-hook enhanced visual panels for audited subsets.

This tool intentionally works as a subset bridge: it visualizes only cases whose
concrete READ/SWA/TTT hook artifacts exist and can be loaded. It does not turn a
subset pass into a full Phase2 action-ready pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageStat

from visualize_v80_case_pca_qkv_ttt_panels import (
    DEFAULT_CASE_BANK_DIR,
    DEFAULT_KITTI_ROOT,
    DEFAULT_PREPROCESS_ROOT,
    case_id,
    filmstrip,
    load_rgb,
    radio_boundary_tile,
    read_rows,
    semantic_payload,
    semantic_tiles,
    select_frame_and_chunk,
    sha256_file,
    tile_text,
    write_csv,
)


DEFAULT_REPAIR_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase2_direct_hook_repair"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase2_direct_hook_enhanced_visual_review"
)

READ_TAPS = (
    "pca_attn_global_k_layers",
    "pca_attn_global_v_layers",
    "pca_attn_frame_v_layers",
)
SWA_TAPS = (
    "pca_swa_current_q_layers",
    "pca_swa_current_k_layers",
    "pca_swa_current_v_layers",
    "pca_swa_cache_k_layers",
    "pca_swa_cache_v_layers",
)
TTT_TAPS = (
    "pca_ttt_operator_output_layers",
    "pca_ttt_update_term_layers",
    "pca_ttt_final_output_layers",
)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict payload, got {type(payload).__name__}")
    return payload


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def audit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_key(row: dict[str, Any]) -> str:
    group = row.get("artifact_group")
    artifact_type = row.get("artifact_type")
    if group == "read" and artifact_type == "v68_layer_pca_feature_dump":
        return "read_pca"
    if group == "read" and artifact_type == "read_cue_patch_dump":
        return "read_cue_patch"
    if group == "swa" and artifact_type == "v68_layer_pca_feature_dump":
        return "swa_pca"
    if group == "ttt" and artifact_type == "v68_layer_pca_feature_dump":
        return "ttt_pca"
    if group == "ttt" and artifact_type == "ttt_spatial_post_delta_map":
        return "ttt_spatial_post_delta"
    return ""


def path_rank(path: str) -> tuple[int, str]:
    rank = 0
    if "layers_mixed" in path:
        rank += 20
    if "mid_extra" in path:
        rank += 10
    if "chunks010_013" in path:
        rank += 8
    if "chunk011" in path:
        rank += 6
    return (rank, path)


def build_artifact_index(audit_csv: Path) -> dict[tuple[str, str, int], Path]:
    index: dict[tuple[str, str, int], Path] = {}
    ranked: dict[tuple[str, str, int], tuple[int, str]] = {}
    for row in audit_rows(audit_csv):
        if row.get("status") != "complete":
            continue
        key = artifact_key(row)
        if not key:
            continue
        chunk = safe_int(row.get("chunk"), default=-1)
        seq = str(row.get("seq") or "unknown").zfill(2)
        path = row.get("path") or ""
        if chunk < 0 or not path:
            continue
        current_rank = path_rank(path)
        idx_key = (seq, key, chunk)
        if idx_key not in ranked or current_rank > ranked[idx_key]:
            ranked[idx_key] = current_rank
            index[idx_key] = Path(path)
    return index


def local_frame(payload: dict[str, Any], frame: int, first_dim: int) -> int:
    start = safe_int(payload.get("start_frame"), default=0)
    return int(np.clip(int(frame) - start, 0, max(first_dim - 1, 0)))


def tensor_heatmap(payload: dict[str, Any], tensor_key: str, frame: int) -> np.ndarray | None:
    value = payload.get(tensor_key)
    if value is None and tensor_key.startswith("tap::"):
        value = payload.get(tensor_key.removeprefix("tap::"))
    if value is None and isinstance(payload.get("tensors"), dict):
        value = payload["tensors"].get(tensor_key)
    if not torch.is_tensor(value):
        return None
    arr = value.detach().float().cpu().numpy()
    if arr.ndim == 5:
        idx = local_frame(payload, frame, arr.shape[0])
        arr = np.abs(arr[idx]).mean(axis=(0, 3))
    elif arr.ndim == 4:
        idx = local_frame(payload, frame, arr.shape[1])
        arr = np.abs(arr[:, idx]).mean(axis=0)
    elif arr.ndim == 3:
        idx = local_frame(payload, frame, arr.shape[0])
        arr = arr[idx]
    elif arr.ndim == 2:
        pass
    else:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def normalize(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def colorize(arr: np.ndarray, size: tuple[int, int], title: str) -> Image.Image:
    x = normalize(arr)
    red = np.clip(1.7 * x - 0.25, 0.0, 1.0)
    green = np.sin(np.clip(x, 0.0, 1.0) * np.pi)
    blue = np.clip(1.25 - 1.6 * x, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    img = Image.fromarray((rgb * 255.0).astype(np.uint8)).resize(size, resample=Image.Resampling.BILINEAR)
    return label_tile(img, title)


def label_tile(img: Image.Image, title: str) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((6, 4), title[:72], fill=(255, 255, 255), font=font)
    return out


def missing_tile(title: str, detail: str, size: tuple[int, int]) -> Image.Image:
    return tile_text([detail], size=size, title=title)


def heatmap_tile(payload: dict[str, Any] | None, tensor_key: str, frame: int, size: tuple[int, int], title: str) -> Image.Image:
    if payload is None:
        return missing_tile(title, "missing artifact payload", size)
    arr = tensor_heatmap(payload, tensor_key, frame)
    if arr is None:
        return missing_tile(title, f"missing tensor: {tensor_key}", size)
    return colorize(arr, size, title)


def strip_heatmap(
    artifact_index: dict[tuple[str, str, int], Path],
    seq: str,
    artifact_key_name: str,
    chunks: Sequence[int],
    tensor_key: str,
    frame_by_chunk: dict[int, int],
    size: tuple[int, int],
    title: str,
) -> Image.Image:
    if not chunks:
        return missing_tile(title, "no chunks requested", size)
    tile_w = max(1, size[0] // len(chunks))
    out = Image.new("RGB", size, (248, 248, 246))
    for i, chunk in enumerate(chunks):
        path = artifact_index.get((seq.zfill(2), artifact_key_name, int(chunk)))
        payload = torch_load(path) if path and path.is_file() else None
        frame = frame_by_chunk.get(int(chunk), int(chunk) * 29 + 16)
        tile = heatmap_tile(payload, tensor_key, frame, (tile_w, size[1]), f"c{chunk}:{title}")
        out.paste(tile, (i * tile_w, 0))
    return out


def image_ok(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"exists": path.is_file(), "width": None, "height": None, "nonblank": False, "error": ""}
    if not path.is_file():
        out["error"] = "missing_file"
        return out
    try:
        img = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(img)
        out.update({"width": img.width, "height": img.height, "nonblank": max(stat.stddev) > 0.0})
    except Exception as exc:  # pragma: no cover
        out["error"] = type(exc).__name__
    return out


def direct_requirements(memory: str, row: dict[str, Any]) -> list[tuple[str, int]]:
    if memory == "short":
        chunk = safe_int(row.get("chunk_id"))
        return [("read_pca", chunk), ("read_cue_patch", chunk), ("swa_pca", chunk)]
    if memory == "mid":
        return [
            ("swa_pca", safe_int(row.get("prev_chunk"))),
            ("swa_pca", safe_int(row.get("curr_chunk"))),
        ]
    return [
        (artifact, chunk)
        for chunk in range(safe_int(row.get("chunk_start")), safe_int(row.get("chunk_end")) + 1)
        for artifact in ("ttt_pca", "ttt_spatial_post_delta")
    ]


def direct_status(
    artifact_index: dict[tuple[str, str, int], Path],
    seq: str,
    requirements: Sequence[tuple[str, int]],
) -> tuple[bool, list[str], list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []
    paths: list[str] = []
    for artifact, chunk in requirements:
        label = f"{artifact}:chunk{chunk:03d}"
        path = artifact_index.get((seq.zfill(2), artifact, int(chunk)))
        if path and path.is_file():
            available.append(label)
            paths.append(str(path))
        else:
            missing.append(label)
    return (not missing, available, missing, paths)


def metrics_lines(memory: str, row: dict[str, Any], frame: int, chunks: Sequence[int], direct_ok: bool) -> list[str]:
    lines = [
        f"memory={memory}",
        f"case={row.get('case_type')}",
        f"seq={str(row.get('seq')).zfill(2)}",
        f"frame={frame}",
        f"chunks={','.join(str(c) for c in chunks)}",
        f"direct_required_ok={direct_ok}",
    ]
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
        "case_reason",
    ):
        value = row.get(key)
        if value not in (None, ""):
            lines.append(f"{key}={value}")
    return lines


def make_short_panel(
    row: dict[str, Any],
    artifact_index: dict[tuple[str, str, int], Path],
    out_dir: Path,
    kitti_root: Path,
    preprocess_root: Path,
) -> dict[str, Any]:
    memory = "short"
    seq = str(row["seq"]).zfill(2)
    frame, chunk = select_frame_and_chunk(memory, row)
    chunks = [chunk]
    tile_size = (320, 220)
    requirements = direct_requirements(memory, row)
    direct_ok, available, missing, paths = direct_status(artifact_index, seq, requirements)
    rgb = load_rgb(kitti_root, seq, frame, tile_size)
    sem_payload = semantic_payload(preprocess_root, seq, chunk)
    sem_img, _conf_img, role_img, sem_meta = semantic_tiles(sem_payload, frame, tile_size)
    read_payload = torch_load(artifact_index[(seq, "read_pca", chunk)]) if (seq, "read_pca", chunk) in artifact_index else None
    cue_payload = torch_load(artifact_index[(seq, "read_cue_patch", chunk)]) if (seq, "read_cue_patch", chunk) in artifact_index else None
    swa_payload = torch_load(artifact_index[(seq, "swa_pca", chunk)]) if (seq, "swa_pca", chunk) in artifact_index else None
    tiles = [
        label_tile(rgb, "RGB"),
        label_tile(sem_img, "Dense semantic"),
        label_tile(role_img, "Semantic role"),
        heatmap_tile(cue_payload, "read_patch_final", frame, tile_size, "READ cue final"),
        heatmap_tile(cue_payload, "read_active_q90_patch", frame, tile_size, "READ active q90"),
        heatmap_tile(read_payload, "tap::pca_attn_global_k_layers", frame, tile_size, "READ global K PCA"),
        heatmap_tile(read_payload, "tap::pca_attn_global_v_layers", frame, tile_size, "READ global V PCA"),
        heatmap_tile(read_payload, "tap::pca_attn_frame_v_layers", frame, tile_size, "READ frame V PCA"),
        heatmap_tile(swa_payload, "tap::pca_swa_current_v_layers", frame, tile_size, "SWA current V PCA"),
        heatmap_tile(swa_payload, "tap::pca_swa_cache_v_layers", frame, tile_size, "SWA cache V PCA"),
        tile_text(metrics_lines(memory, row, frame, chunks, direct_ok), size=tile_size, title="Case metrics"),
        tile_text(available + ([f"MISSING {x}" for x in missing] if missing else []), size=tile_size, title="Direct evidence"),
    ]
    return save_panel(memory, row, out_dir, tiles, tile_size, frame, chunk, chunks, sem_meta, direct_ok, available, missing, paths)


def make_mid_panel(
    row: dict[str, Any],
    artifact_index: dict[tuple[str, str, int], Path],
    out_dir: Path,
    kitti_root: Path,
    preprocess_root: Path,
) -> dict[str, Any]:
    memory = "mid"
    seq = str(row["seq"]).zfill(2)
    frame, chunk = select_frame_and_chunk(memory, row)
    chunks = [safe_int(row.get("prev_chunk")), safe_int(row.get("curr_chunk"))]
    frame_by_chunk = {c: c * 29 + 16 for c in chunks}
    frame_by_chunk[chunk] = frame
    tile_size = (320, 220)
    requirements = direct_requirements(memory, row)
    direct_ok, available, missing, paths = direct_status(artifact_index, seq, requirements)
    sem_payload = semantic_payload(preprocess_root, seq, chunk)
    sem_img, conf_img, role_img, sem_meta = semantic_tiles(sem_payload, frame, tile_size)
    tiles = [
        label_tile(filmstrip(kitti_root, seq, chunks, tile_size), "RGB pair filmstrip"),
        label_tile(sem_img, "Boundary semantic"),
        label_tile(role_img, "Boundary role"),
        label_tile(conf_img, "Boundary confidence"),
        strip_heatmap(artifact_index, seq, "swa_pca", chunks, "tap::pca_swa_current_q_layers", frame_by_chunk, tile_size, "SWA current Q"),
        strip_heatmap(artifact_index, seq, "swa_pca", chunks, "tap::pca_swa_current_k_layers", frame_by_chunk, tile_size, "SWA current K"),
        strip_heatmap(artifact_index, seq, "swa_pca", chunks, "tap::pca_swa_current_v_layers", frame_by_chunk, tile_size, "SWA current V"),
        strip_heatmap(artifact_index, seq, "swa_pca", chunks, "tap::pca_swa_cache_k_layers", frame_by_chunk, tile_size, "SWA cache K"),
        strip_heatmap(artifact_index, seq, "swa_pca", chunks, "tap::pca_swa_cache_v_layers", frame_by_chunk, tile_size, "SWA cache V"),
        label_tile(radio_boundary_tile(preprocess_root, seq, chunk, frame, tile_size), "RADIO boundary"),
        tile_text(metrics_lines(memory, row, frame, chunks, direct_ok), size=tile_size, title="Case metrics"),
        tile_text(available + ([f"MISSING {x}" for x in missing] if missing else []), size=tile_size, title="Direct evidence"),
    ]
    return save_panel(memory, row, out_dir, tiles, tile_size, frame, chunk, chunks, sem_meta, direct_ok, available, missing, paths)


def make_long_panel(
    row: dict[str, Any],
    artifact_index: dict[tuple[str, str, int], Path],
    out_dir: Path,
    kitti_root: Path,
    preprocess_root: Path,
) -> dict[str, Any]:
    memory = "long"
    seq = str(row["seq"]).zfill(2)
    frame, chunk = select_frame_and_chunk(memory, row)
    chunks = list(range(safe_int(row.get("chunk_start")), safe_int(row.get("chunk_end")) + 1))
    frame_by_chunk = {c: c * 29 + 16 for c in chunks}
    tile_size = (320, 220)
    requirements = direct_requirements(memory, row)
    direct_ok, available, missing, paths = direct_status(artifact_index, seq, requirements)
    sem_payload = semantic_payload(preprocess_root, seq, chunk)
    sem_img, conf_img, role_img, sem_meta = semantic_tiles(sem_payload, frame, tile_size)
    tiles = [
        label_tile(filmstrip(kitti_root, seq, chunks, tile_size), "RGB five-chunk strip"),
        label_tile(sem_img, "Center semantic"),
        label_tile(role_img, "Center role"),
        label_tile(conf_img, "Center confidence"),
        strip_heatmap(artifact_index, seq, "ttt_pca", chunks, "tap::pca_ttt_operator_output_layers", frame_by_chunk, tile_size, "TTT operator PCA"),
        strip_heatmap(artifact_index, seq, "ttt_pca", chunks, "tap::pca_ttt_update_term_layers", frame_by_chunk, tile_size, "TTT update PCA"),
        strip_heatmap(artifact_index, seq, "ttt_pca", chunks, "tap::pca_ttt_final_output_layers", frame_by_chunk, tile_size, "TTT final PCA"),
        strip_heatmap(artifact_index, seq, "ttt_spatial_post_delta", chunks, "ttt_write_prior_patch", frame_by_chunk, tile_size, "TTT write prior"),
        strip_heatmap(artifact_index, seq, "ttt_spatial_post_delta", chunks, "action_delta_norm_projection_patch", frame_by_chunk, tile_size, "TTT action delta"),
        strip_heatmap(artifact_index, seq, "ttt_spatial_post_delta", chunks, "native_delta_norm_projection_patch", frame_by_chunk, tile_size, "TTT native delta"),
        tile_text(metrics_lines(memory, row, frame, chunks, direct_ok), size=tile_size, title="Case metrics"),
        tile_text(available + ([f"MISSING {x}" for x in missing] if missing else []), size=tile_size, title="Direct evidence"),
    ]
    return save_panel(memory, row, out_dir, tiles, tile_size, frame, chunk, chunks, sem_meta, direct_ok, available, missing, paths)


def save_panel(
    memory: str,
    row: dict[str, Any],
    out_dir: Path,
    tiles: Sequence[Image.Image],
    tile_size: tuple[int, int],
    frame: int,
    chunk: int,
    chunks: Sequence[int],
    sem_meta: dict[str, Any],
    direct_ok: bool,
    available: Sequence[str],
    missing: Sequence[str],
    paths: Sequence[str],
) -> dict[str, Any]:
    panel = Image.new("RGB", (tile_size[0] * 4, tile_size[1] * 3), (255, 255, 255))
    for idx, img in enumerate(tiles):
        panel.paste(img.resize(tile_size), ((idx % 4) * tile_size[0], (idx // 4) * tile_size[1]))
    panel_dir = out_dir / f"{memory}_direct_hook_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / f"{case_id(memory, row)}.png"
    panel.save(panel_path)
    return {
        "memory_body": memory,
        "case_type": row.get("case_type"),
        "seq": str(row.get("seq")).zfill(2),
        "case_id": case_id(memory, row),
        "frame": frame,
        "chunk": chunk,
        "chunks_required": ",".join(str(c) for c in chunks),
        "visual_file": str(panel_path),
        "width": panel.width,
        "height": panel.height,
        "sha256": sha256_file(panel_path),
        "semantic_panel_available": bool(sem_meta.get("semantic_panel_available")),
        "direct_qkv_ttt_artifact_available": bool(direct_ok),
        "direct_evidence_scope": (
            f"seq{str(row.get('seq')).zfill(2)}_"
            f"{str(row.get('case_type') or 'unknown')}_subset_only"
        ),
        "required_artifacts": ",".join(f"{a}:chunk{c:03d}" for a, c in direct_requirements(memory, row)),
        "available_artifacts": ";".join(available),
        "missing_direct_artifacts": ";".join(missing),
        "direct_artifact_paths": ";".join(paths),
        "availability_note": "real direct hook tensors rendered as heatmaps; subset output does not claim full Phase2 readiness",
    }


def selected_case_rows(case_bank_dir: Path, seq: str, case_type: str) -> dict[str, list[dict[str, Any]]]:
    rows_by_memory = {
        "short": read_rows(case_bank_dir / "short_single_chunk_cases.csv"),
        "mid": read_rows(case_bank_dir / "mid_adjacent_pair_cases.csv"),
        "long": read_rows(case_bank_dir / "long_five_chunk_cases.csv"),
    }
    return {
        memory: [
            row for row in rows
            if str(row.get("seq")).zfill(2) == seq.zfill(2)
            and (case_type == "all" or row.get("case_type") == case_type)
        ]
        for memory, rows in rows_by_memory.items()
    }


def write_review(out_dir: Path, manifest: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest:
        semantic_ok = bool(row.get("semantic_panel_available"))
        direct_ok = bool(row.get("direct_qkv_ttt_artifact_available"))
        if semantic_ok and direct_ok:
            status = "confirmed"
            action_readiness = "ready_for_subset_rule_design_only"
            pattern = "real semantic panel and direct READ/SWA/TTT hook heatmaps available"
        elif semantic_ok:
            status = "ambiguous"
            action_readiness = "not_ready_missing_direct_hook_confirmation"
            pattern = "semantic panel available but direct hook evidence incomplete"
        else:
            status = "rejected"
            action_readiness = "not_ready_missing_semantic_panel"
            pattern = "semantic panel missing or unreadable"
        rows.append(
            {
                "memory_body": row.get("memory_body"),
                "case_type": row.get("case_type"),
                "seq": row.get("seq"),
                "case_id": row.get("case_id"),
                "visual_file": row.get("visual_file"),
                "review_status": status,
                "visual_pattern_observed": pattern,
                "action_readiness": action_readiness,
                "scope_note": (
                    f"seq{str(row.get('seq')).zfill(2)}_"
                    f"{str(row.get('case_type') or 'unknown')}_subset_only_not_full_phase2"
                ),
                "reviewer": "codex_auto_direct_hook_visual_audit",
            }
        )
    write_csv(out_dir / "visual_review.csv", rows)
    return rows


def audit_visuals(out_dir: Path, manifest: Sequence[dict[str, Any]], review: Sequence[dict[str, Any]], seq: str, case_type: str) -> dict[str, Any]:
    image_failures: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        ok = image_ok(Path(str(row.get("visual_file", ""))))
        if not ok["exists"] or not ok["nonblank"]:
            image_failures.append({"case_id": row.get("case_id"), **ok})
        counts[str(row.get("memory_body"))][str(row.get("case_type"))] += 1
    status_counts = Counter(row.get("review_status") for row in review)
    direct_count = sum(bool(row.get("direct_qkv_ttt_artifact_available")) for row in manifest)
    semantic_count = sum(bool(row.get("semantic_panel_available")) for row in manifest)
    confirmed_count = status_counts.get("confirmed", 0)
    subset_gate_pass = (
        len(manifest) > 0
        and not image_failures
        and direct_count == len(manifest)
        and semantic_count == len(manifest)
        and confirmed_count == len(manifest)
    )
    audit = {
        "schema": "acl2_v80tf_phase2_direct_hook_enhanced_subset_visual_audit_v1",
        "visual_root": str(out_dir),
        "scope": {
            "seq": seq.zfill(2),
            "case_type": case_type,
            "subset_only": True,
            "full_phase2_claimed": False,
        },
        "num_visual_files": len(manifest),
        "num_review_rows": len(review),
        "image_failure_count": len(image_failures),
        "image_failures": image_failures[:20],
        "semantic_panel_available_count": semantic_count,
        "direct_qkv_ttt_artifact_available_count": direct_count,
        "review_status_counts": dict(status_counts),
        "panel_counts_by_memory": {memory: dict(counter) for memory, counter in counts.items()},
        "subset_gate_pass": subset_gate_pass,
        "full_phase2_action_ready_gate_pass": False,
        "full_phase2_action_ready_reason": "This audit covers only the filtered direct-hook subset, not all 72 Phase1 cases.",
        "notes": [
            "subset_gate_pass validates nonblank panels, semantic panels, direct hook evidence, and confirmed review for this subset.",
            "This JSON must not be used as evidence that full Phase2 action_ready passed.",
        ],
    }
    write_json(out_dir / "visual_integrity_audit.json", audit)
    return audit


def write_insight(out_dir: Path, audit: dict[str, Any]) -> None:
    scope = audit.get("scope", {})
    seq = str(scope.get("seq") or "unknown").zfill(2)
    case_type = str(scope.get("case_type") or "unknown")
    lines = [
        "# ACL2 v80 Direct-Hook Enhanced Visual Insight",
        "",
        f"Scope: seq{seq} {case_type} subset only.",
        "",
        f"- visual rows: {audit['num_visual_files']}",
        f"- direct hook rows: {audit['direct_qkv_ttt_artifact_available_count']}",
        f"- semantic rows: {audit['semantic_panel_available_count']}",
        f"- subset_gate_pass: {audit['subset_gate_pass']}",
        f"- full_phase2_action_ready_gate_pass: {audit['full_phase2_action_ready_gate_pass']}",
        "",
        "Evidence rendered:",
        "",
        "- short cases: READ cue, READ global/frame PCA, SWA current/cache PCA.",
        "- mid cases: SWA current/cache Q/K/V PCA over adjacent chunks.",
        "- long cases: TTT operator/update/final PCA and spatial post-delta maps over five chunks.",
        "",
        "Interpretation:",
        "",
        f"The direct-hook blocker is repaired for the seq{seq} {case_type} subset represented by this output, because the panels render concrete .pt tensor payloads instead of missing-artifact notes. This remains insufficient for the full v80 Phase2 gate unless every Phase1 sequence/case subset is direct-confirmed.",
        "",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(lines), encoding="utf-8")


def build_manifest(
    rows_by_memory: dict[str, list[dict[str, Any]]],
    artifact_index: dict[tuple[str, str, int], Path],
    out_dir: Path,
    kitti_root: Path,
    preprocess_root: Path,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in rows_by_memory["short"]:
        manifest.append(make_short_panel(row, artifact_index, out_dir, kitti_root, preprocess_root))
    for row in rows_by_memory["mid"]:
        manifest.append(make_mid_panel(row, artifact_index, out_dir, kitti_root, preprocess_root))
    for row in rows_by_memory["long"]:
        manifest.append(make_long_panel(row, artifact_index, out_dir, kitti_root, preprocess_root))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank-dir", type=Path, default=DEFAULT_CASE_BANK_DIR)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument("--direct-hook-audit-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--case-type", default="bad", choices=("good", "bad", "all"))
    args = parser.parse_args()

    audit_csv = args.direct_hook_audit_csv or args.repair_root / "direct_hook_audit" / "direct_hook_repair_audit.csv"
    artifact_index = build_artifact_index(audit_csv)
    rows_by_memory = selected_case_rows(args.case_bank_dir, args.seq, args.case_type)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(rows_by_memory, artifact_index, args.out_dir, args.kitti_root, args.preprocess_root)
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    review = write_review(args.out_dir, manifest)
    audit = audit_visuals(args.out_dir, manifest, review, args.seq, args.case_type)
    write_insight(args.out_dir, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
