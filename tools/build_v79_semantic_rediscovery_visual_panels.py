#!/usr/bin/env python3
"""Build ACL2 v79 Phase7 rediscovery visual evidence panels.

This tool is deliberately conservative.  It visualizes real artifacts that
already exist in the v79 report tree and marks missing direct hooks explicitly
instead of drawing synthetic Q/K/V or READ/TTT alignment masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final"
)
DEFAULT_PHASE7_DIR = DEFAULT_REPORT_ROOT / "phase7_semantic_pca_qkv_ttt_rediscovery"
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_IMAGE_DIR = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")

PALETTE: Dict[str, Tuple[int, int, int]] = {
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

REVIEW_FIELDS = ["artifact_group", "path", "status", "review_note"]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _stats(values: torch.Tensor | np.ndarray) -> Dict[str, Optional[float]]:
    if torch.is_tensor(values):
        arr = values.detach().cpu().float().reshape(-1).numpy()
    else:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "q90": None, "max": None, "finite_count": 0}
    return {
        "mean": float(np.mean(arr)),
        "q90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "finite_count": int(arr.size),
    }


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


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    lines = [line for line in str(text).split("\n") if line]
    if not lines:
        return out
    pad = 6
    widths: List[int] = []
    heights: List[int] = []
    for line in lines:
        bbox = draw.textbbox((pad, pad), line)
        widths.append(int(bbox[2] - bbox[0]))
        heights.append(int(bbox[3] - bbox[1]))
    box_w = max(widths) + 10
    line_h = max(max(heights), 10) + 4
    box_h = line_h * len(lines) + 4
    draw.rectangle((pad - 4, pad - 3, pad - 4 + box_w, pad - 3 + box_h), fill=(0, 0, 0))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255))
        y += line_h
    return out


def _text_panel(lines: Sequence[str], size: Tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (18, 18, 18))
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((8, y), str(line), fill=(240, 240, 240))
        y += 16
        if y > size[1] - 14:
            break
    return img


def _resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _concat_row(panels: Sequence[Image.Image]) -> Image.Image:
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    out = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def _concat_col(rows: Sequence[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    out = Image.new("RGB", (width, height), (0, 0, 0))
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def _robust01(array: torch.Tensor | np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    if torch.is_tensor(array):
        arr = array.detach().cpu().float().numpy()
    else:
        arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(arr[finite], lo_q))
    hi = float(np.percentile(arr[finite], hi_q))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _heat_image(array: torch.Tensor | np.ndarray, cmap_name: str = "magma") -> Image.Image:
    arr = _robust01(array)
    rgba = colormaps.get_cmap(cmap_name)(arr)
    return Image.fromarray((rgba[..., :3] * 255.0).astype(np.uint8), mode="RGB")


def _semantic_image(label_map: torch.Tensor, label_names: Sequence[str]) -> Image.Image:
    labels = label_map.detach().cpu().long().numpy()
    colours = _colour_table(label_names)
    labels = np.clip(labels, 0, colours.shape[0] - 1)
    return Image.fromarray(colours[labels], mode="RGB")


def _role_image(role_map: torch.Tensor) -> Image.Image:
    roles = role_map.detach().cpu().float().round().long().numpy()
    colours = np.asarray(
        [
            (0, 0, 0),
            (70, 180, 90),
            (190, 190, 70),
            (220, 70, 70),
            (75, 140, 220),
        ],
        dtype=np.uint8,
    )
    roles = np.clip(roles, 0, colours.shape[0] - 1)
    return Image.fromarray(colours[roles], mode="RGB")


def _image_path(image_dir: Path, frame: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB image found for frame {frame} in {image_dir}")


def _load_stage_chunk(stage_c_cache: Path, chunk: int) -> Dict[str, Any]:
    matches = sorted(stage_c_cache.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"No Stage-C masklet.pt found for chunk {chunk} under {stage_c_cache}")
    payload = _torch_load(matches[0])
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {matches[0]}")
    sem = payload.get("semantic_segmentation")
    if not isinstance(sem, dict) or not torch.is_tensor(sem.get("label_maps")):
        raise KeyError(f"Missing semantic_segmentation.label_maps in {matches[0]}")
    payload["_path"] = str(matches[0])
    return payload


def _semantic_panels_for_frame(
    *,
    stage_chunk: Mapping[str, Any],
    image_dir: Path,
    global_frame: int,
    panel_size: Tuple[int, int],
    title_prefix: str,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    sem = stage_chunk["semantic_segmentation"]
    start = int(sem.get("global_start_frame", stage_chunk.get("manifest", {}).get("start_frame", 0)))
    local = int(global_frame) - start
    label_maps = sem["label_maps"]
    if local < 0 or local >= int(label_maps.shape[0]):
        raise IndexError(f"Frame {global_frame} outside chunk semantic range starting at {start}")
    label_names = [str(x) for x in (sem.get("label_names") or [])]
    conf = sem.get("confidence_maps")
    rgb = Image.open(_image_path(image_dir, int(global_frame))).convert("RGB")
    label_img = _semantic_image(label_maps[local], label_names)
    if torch.is_tensor(conf):
        conf_img = _heat_image(conf[local], "viridis")
        conf_stats = _stats(conf[local])
    else:
        conf_img = Image.new("RGB", tuple(reversed(label_maps.shape[-2:])), (0, 0, 0))
        conf_stats = {"mean": None, "q90": None, "max": None, "finite_count": 0}
    panels = [
        _label(_resize(rgb, panel_size), f"{title_prefix} RGB f{global_frame:06d}"),
        _label(_resize(label_img, panel_size), f"{title_prefix} semantic"),
        _label(_resize(conf_img, panel_size), f"{title_prefix} confidence"),
    ]
    return panels, {
        "frame": int(global_frame),
        "local_frame": int(local),
        "semantic_source": str(sem.get("source", "")),
        "labels": label_names,
        "confidence_stats": conf_stats,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                clean[field] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(clean)


def _mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [_safe_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _build_qkv_panels(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, Any]]:
    panel_dir = out_dir / "new_qkv_visual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    read_dir = args.report_root / "phase2_semantic_read_global_control/rollouts/chunk10/READ1_L07_SEMANTIC_LAYOUT_SELECT"
    rows = _read_jsonl(read_dir / "hmc_state_hash.jsonl")
    stage = _load_stage_chunk(args.stage_c_cache, 10)
    frames = [290, 306, 321]
    visual_rows: List[Dict[str, Any]] = []
    panel_size = (360, 109)
    mean_lines = [
        "direct Q/K/V tensor dump: missing",
        "evidence: hmc JSONL aggregate only",
        f"rows={len(rows)}",
        f"read output mean={_mean_present(rows, 'prior_v78_l07_l13_output_mean')}",
        f"read output q90={_mean_present(rows, 'prior_v78_l07_l13_output_q90')}",
        f"l07 layout mean={_mean_present(rows, 'prior_v78_l07_l13_l07_layout_mean')}",
        f"l13 neg mean={_mean_present(rows, 'prior_v78_l07_l13_l13_neg_mean')}",
        "status: proxy panel; no alignment score claimed",
    ]
    for frame in frames:
        panels, meta = _semantic_panels_for_frame(
            stage_chunk=stage,
            image_dir=args.image_dir,
            global_frame=frame,
            panel_size=panel_size,
            title_prefix="READ1 chunk10",
        )
        panels.append(_text_panel(mean_lines, panel_size))
        out_path = panel_dir / f"read1_qkv_missing_direct_dump_proxy_f{frame:06d}.png"
        _concat_row(panels).save(out_path)
        visual_rows.append(
            {
                "artifact_group": "new_qkv_visual_panels",
                "panel": str(out_path),
                "frame": int(frame),
                "chunk": 10,
                "evidence_type": "proxy_missing_direct_qkv_dump",
                "direct_qkv_tensor_available": False,
                "direct_alignment_score_claimed": False,
                "source_hmc_jsonl": str(read_dir / "hmc_state_hash.jsonl"),
                "semantic_source": meta["semantic_source"],
                "notes": "RGB/semantic/confidence are real; QKV map is not drawn because per-token dump is missing.",
            }
        )
    return visual_rows


def _select_payload_tensor(payload: Mapping[str, Any], key: str, local: int) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Missing tensor {key}")
    if value.ndim == 4:
        return value[:, int(local)].detach().cpu().float()
    if value.ndim == 3:
        return value[int(local)].detach().cpu().float()
    raise ValueError(f"Unsupported tensor shape for {key}: {tuple(value.shape)}")


def _load_ttt_payloads(dump_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in sorted(dump_dir.glob("chunk_*_ttt_spatial_post_delta_map.pt")):
        payload = _torch_load(path)
        if not isinstance(payload, dict):
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _build_ttt_panels(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, Any]]:
    panel_dir = out_dir / "new_ttt_branch_visual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = (
        args.report_root
        / "phase5_cross_memory_semantic_handshake/read_to_ttt_fivechunk_7_11/chunk11"
        / "HS5_READ_TO_TTT_SEM/ttt_spatial_post_delta_maps"
    )
    payloads = _load_ttt_payloads(dump_dir)
    if not payloads:
        raise FileNotFoundError(f"No TTT spatial payloads found in {dump_dir}")
    panel_size = (300, 91)
    rows: List[Image.Image] = []
    visual_rows: List[Dict[str, Any]] = []
    for payload in payloads:
        chunk = int(payload.get("chunk_idx", -1))
        start = int(payload.get("start_frame", 0))
        end = int(payload.get("end_frame", start))
        frame = min(end - 1, start + max(0, (end - start) // 2))
        local = int(frame - start)
        stage = _load_stage_chunk(args.stage_c_cache, chunk)
        sem_panels, meta = _semantic_panels_for_frame(
            stage_chunk=stage,
            image_dir=args.image_dir,
            global_frame=frame,
            panel_size=panel_size,
            title_prefix=f"HS5 chunk{chunk:03d}",
        )
        delta_rows = _select_payload_tensor(payload, "action_delta_norm_projection_patch", local)
        delta_mag = delta_rows.mean(dim=0)
        d_tok = _select_payload_tensor(payload, "D_tok_patch", local)
        p_ttt = _select_payload_tensor(payload, "ttt_write_prior_patch", local)
        r_ttt = _select_payload_tensor(payload, "R_ttt_tok_patch", local)
        world_delta = _select_payload_tensor(payload, "pass1_pass2_world_points_l2_patch", local)
        panels = [
            sem_panels[0],
            sem_panels[1],
            _label(_resize(_heat_image(delta_mag, "magma"), panel_size), "action delta mean"),
            _label(_resize(_heat_image(d_tok, "viridis"), panel_size), "D_tok risk"),
            _label(_resize(_role_image(r_ttt), panel_size), "R_ttt role 1/2/3"),
            _label(_resize(_heat_image(p_ttt, "plasma"), panel_size), "P_ttt_write"),
            _label(_resize(_heat_image(world_delta, "inferno"), panel_size), "pass1-pass2 world L2"),
        ]
        row_img = _concat_row(panels)
        rows.append(row_img)
        out_path = panel_dir / f"hs5_ttt_carrier_chunk{chunk:03d}_f{frame:06d}.png"
        row_img.save(out_path)
        visual_rows.append(
            {
                "artifact_group": "new_ttt_branch_visual_panels",
                "panel": str(out_path),
                "frame": int(frame),
                "chunk": int(chunk),
                "evidence_type": "actual_ttt_spatial_post_delta_tensor",
                "payload": str(payload.get("_path", "")),
                "semantic_source": meta["semantic_source"],
                "action_delta_stats": _stats(delta_mag),
                "d_tok_stats": _stats(d_tok),
                "p_ttt_stats": _stats(p_ttt),
                "world_delta_stats": _stats(world_delta),
                "projection_not_raw_per_token_fast_weight_delta": bool(payload.get("projection_not_raw_per_token_fast_weight_delta", True)),
                "notes": "TTT map is real patch-aligned projected post-delta evidence, not a raw fast-weight gradient.",
            }
        )
    if rows:
        contact = panel_dir / "hs5_ttt_carrier_contact_sheet.png"
        _concat_col(rows).save(contact)
        visual_rows.append(
            {
                "artifact_group": "new_ttt_branch_visual_panels",
                "panel": str(contact),
                "frame": "",
                "chunk": "7-11",
                "evidence_type": "contact_sheet",
                "payload": str(dump_dir),
                "notes": "Contact sheet over all available HS5 READ->TTT five-window TTT spatial payloads.",
            }
        )
    return visual_rows


def _ledger_row(path: Path, chunk_or_pair: str) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair = row.get("chunk_pair") or row.get("chunks") or row.get("chunk") or ""
            if str(pair) == str(chunk_or_pair):
                return dict(row)
    return {}


def _phase3_metric_lines(report_root: Path) -> List[str]:
    decision_paths = [
        report_root / "phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09/phase9_swa_cache_value_decision.json",
        report_root / "phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09_alllayers_strong035/phase9_swa_cache_value_decision.json",
    ]
    lines = ["SWA gate summary:"]
    for path in decision_paths:
        payload = _load_json(path) or {}
        lines.append(f"{path.parent.name}: pass={payload.get('phase9_any_gate_pass')}")
        decisions = payload.get("decisions") or {}
        for key in ("P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST", "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST"):
            node = decisions.get(key)
            if not isinstance(node, dict):
                continue
            passes = node.get("mechanism_metric_passes") or node.get("metric_passes") or []
            blockers = node.get("blockers") or []
            lines.append(f"{key}: passes={passes} blockers={blockers[:2]}")
    return lines[:10]


def _build_merge_panels(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, Any]]:
    panel_dir = out_dir / "new_merge_boundary_visual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_size = (300, 91)
    visual_rows: List[Dict[str, Any]] = []
    chunks = {8: _load_stage_chunk(args.stage_c_cache, 8), 9: _load_stage_chunk(args.stage_c_cache, 9)}
    frames = [258, 261, 262, 263, 266, 290]
    frame_rows: List[Image.Image] = []
    for frame in frames:
        chunk = 8 if frame <= 264 else 9
        panels, meta = _semantic_panels_for_frame(
            stage_chunk=chunks[chunk],
            image_dir=args.image_dir,
            global_frame=frame,
            panel_size=panel_size,
            title_prefix=f"boundary c8-9",
        )
        frame_rows.append(_concat_row(panels))
        visual_rows.append(
            {
                "artifact_group": "new_merge_boundary_visual_panels",
                "panel": "boundary_frame_strip_contact.png",
                "frame": int(frame),
                "chunk": int(chunk),
                "evidence_type": "actual_rgb_semantic_confidence_boundary_frame",
                "semantic_source": meta["semantic_source"],
                "notes": "Frame belongs to chunk8 tail/chunk9 head overlap neighborhood.",
            }
        )
    ledger_path = args.report_root / "phase1_current_bad_target_mining_with_semantic_diagnosis/adjacent_semantic_handoff_targets.csv"
    ledger = _ledger_row(ledger_path, "8-9")
    metric_lines = [
        "Phase1 adjacent target chunks8-9",
        f"future_after_overlap={ledger.get('future_after_overlap', '')}",
        f"boundary_jump={ledger.get('boundary_jump', '')}",
        f"raw_overlap_residual={ledger.get('raw_overlap_residual', '')}",
        f"ledger={ledger_path.name}",
        "",
    ] + _phase3_metric_lines(args.report_root)
    frame_rows.append(_text_panel(metric_lines, (panel_size[0] * 3, panel_size[1])))
    contact = panel_dir / "boundary_frame_strip_contact.png"
    _concat_col(frame_rows).save(contact)
    visual_rows.append(
        {
            "artifact_group": "new_merge_boundary_visual_panels",
            "panel": str(contact),
            "frame": "258,261,262,263,266,290",
            "chunk": "8-9",
            "evidence_type": "merge_boundary_contact_sheet_with_real_metrics",
            "phase1_ledger": str(ledger_path),
            "phase1_metrics": ledger,
            "notes": "Actual RGB/semantic/confidence strips plus recorded Phase1/Phase3 boundary metrics; no raw residual heatmap was available.",
        }
    )
    return visual_rows


def _write_visual_insight_append(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = out_dir / "visual_insight.md"
    old = path.read_text(encoding="utf-8") if path.exists() else "# v79 Phase7 Visual Insight\n"
    generated = [row for row in rows if row.get("panel")]
    lines = [
        "",
        "## Generated Visual Evidence Pass",
        "",
        "This pass generated Phase7 visual evidence panels from existing real artifacts.",
        "It does not claim direct Q/K/V or READ_TTT alignment where those hooks are absent.",
        "",
        f"- Panel records: `{len(generated)}`.",
        "- `new_qkv_visual_panels`: RGB/semantic/confidence plus aggregate READ cue evidence; direct per-token Q/K/V dump is still missing.",
        "- `new_ttt_branch_visual_panels`: actual HS5 READ->TTT TTT spatial post-delta tensors overlaid with semantic/R_ttt/D_tok/P_ttt/world-delta maps.",
        "- `new_merge_boundary_visual_panels`: actual chunk8-9 boundary RGB/semantic/confidence strips with Phase1/Phase3 recorded metrics.",
        "",
        "Interpretation:",
        "- The TTT panels show a real post-delta carrier exists, but this remains a projected post-zp/output-delta carrier, not proof that semantic READ evidence reached the correct long-window geometry carrier.",
        "- The QKV panels confirm the current blocker: only aggregate L07/L13 READ stats are present; direct token-level K/V or READ role maps must be dumped before an alignment score can be claimed.",
        "- The merge-boundary panels keep chunks8-9 as a boundary/gauge candidate rather than treating it as a pure semantic handoff failure.",
        "",
    ]
    marker = "## Generated Visual Evidence Pass"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip() + "\n"
    path.write_text(old.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rows.extend(_build_qkv_panels(args, out_dir))
    rows.extend(_build_ttt_panels(args, out_dir))
    rows.extend(_build_merge_panels(args, out_dir))

    generated_dirs = {
        "new_qkv_visual_panels": len(list((out_dir / "new_qkv_visual_panels").glob("*.png"))),
        "new_ttt_branch_visual_panels": len(list((out_dir / "new_ttt_branch_visual_panels").glob("*.png"))),
        "new_merge_boundary_visual_panels": len(list((out_dir / "new_merge_boundary_visual_panels").glob("*.png"))),
    }
    direct_hooks_present = {
        "READ_TTT_ROLE_ALIGNMENT_LOG": False,
        "READ_SWA_ROLE_ALIGNMENT_LOG": False,
        "SWA_TTT_ROLE_ALIGNMENT_LOG": False,
        "direct_qkv_tensor_dump": False,
    }
    visual_audit = {
        "schema": "acl2_v79_phase7_visual_integrity_audit_v2",
        "panel_dirs_populated": all(count > 0 for count in generated_dirs.values()),
        "generated_dirs": generated_dirs,
        "panel_record_count": len(rows),
        "no_fake_visuals": True,
        "direct_hooks_present": direct_hooks_present,
        "strict_alignment_gate_pass": False,
        "gate_pass": False,
        "reason": "required_visual_panel_dirs_populated_but_direct_cross_memory_alignment_hooks_are_still_missing",
    }
    review_rows = [
        {
            "artifact_group": name,
            "path": str(out_dir / name),
            "status": "generated" if count > 0 else "missing",
            "review_note": f"{count} PNG panel(s); see rediscovery_visual_panel_summary.csv.",
        }
        for name, count in generated_dirs.items()
    ]

    fields = [
        "artifact_group",
        "panel",
        "frame",
        "chunk",
        "evidence_type",
        "direct_qkv_tensor_available",
        "direct_alignment_score_claimed",
        "source_hmc_jsonl",
        "payload",
        "phase1_ledger",
        "phase1_metrics",
        "semantic_source",
        "action_delta_stats",
        "d_tok_stats",
        "p_ttt_stats",
        "world_delta_stats",
        "projection_not_raw_per_token_fast_weight_delta",
        "notes",
    ]
    _write_csv(out_dir / "rediscovery_visual_panel_summary.csv", rows, fields)
    _write_csv(out_dir / "visual_review.csv", review_rows, REVIEW_FIELDS)
    (out_dir / "visual_integrity_audit.json").write_text(
        json.dumps(visual_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rediscovery_visual_summary.json").write_text(
        json.dumps(
            {
                "schema": "acl2_v79_phase7_rediscovery_visual_summary_v1",
                "out_dir": str(out_dir),
                "report_root": str(args.report_root),
                "stage_c_cache": str(args.stage_c_cache),
                "image_dir": str(args.image_dir),
                "visual_audit": visual_audit,
                "panel_records": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_visual_insight_append(out_dir, rows)
    return {
        "out_dir": str(out_dir),
        "generated_dirs": generated_dirs,
        "panel_record_count": len(rows),
        "strict_alignment_gate_pass": False,
        "gate_pass": False,
        "reason": visual_audit["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PHASE7_DIR)
    parser.add_argument("--stage-c-cache", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
