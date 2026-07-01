#!/usr/bin/env python3
"""Build posthoc READ-vs-TTT alignment diagnostics from real v79 dumps.

This tool does not create a strict Phase5 success signal.  It only compares
READ cue patch masks dumped by HybridMemoryController against same-run TTT
role/post-delta maps so the failure mode can be inspected without inventing a
missing in-controller READ_TTT_ROLE_ALIGNMENT_LOG.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _jsonable(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_pt(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _payload_tensor(payload: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    value = payload.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    tensors = payload.get("tensors")
    if isinstance(tensors, dict):
        value = tensors.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    return None


def _as_t_h_w(value: torch.Tensor) -> torch.Tensor:
    x = value.detach().cpu().float()
    if x.ndim == 4:
        x = x.mean(dim=0)
    if x.ndim != 3:
        raise ValueError(f"expected T,H,W or B,T,H,W tensor, got shape={tuple(x.shape)}")
    return x


def _align_shape(a: torch.Tensor, b: torch.Tensor, *, name_a: str, name_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"shape mismatch {name_a}={tuple(a.shape)} {name_b}={tuple(b.shape)}")
    return a, b


def _finite_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Optional[float]:
    x = values.detach().cpu().float()
    if mask is not None:
        m = mask.detach().cpu().bool()
        if tuple(m.shape) != tuple(x.shape):
            raise ValueError(f"mask shape {tuple(m.shape)} does not match values {tuple(x.shape)}")
        x = x[m]
    x = x[torch.isfinite(x)]
    if int(x.numel()) == 0:
        return None
    return float(x.mean().item())


def _mass(mask: torch.Tensor) -> float:
    m = mask.detach().cpu().bool()
    if int(m.numel()) == 0:
        return 0.0
    return float(m.float().mean().item())


def _conditional_mass(mask: torch.Tensor, condition: torch.Tensor) -> Optional[float]:
    m = mask.detach().cpu().bool()
    c = condition.detach().cpu().bool()
    denom = int(c.sum().item())
    if denom <= 0:
        return None
    return float((m & c).float().sum().item() / float(denom))


def _role_fraction(role: torch.Tensor, role_id: int, mask: Optional[torch.Tensor] = None) -> Optional[float]:
    r = torch.round(role.detach().cpu().float()).long()
    selected = r == int(role_id)
    if mask is None:
        return _mass(selected)
    return _conditional_mass(selected, mask)


def _top_fraction_mask(values: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    x = values.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"expected T,H,W tensor for top mask, got {tuple(x.shape)}")
    t, h, w = x.shape
    flat = x.reshape(t, h * w)
    k = max(1, int(np.ceil(float(fraction) * h * w)))
    idx = torch.topk(flat, k=min(k, h * w), dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(t, h, w)


def _robust01(array: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = array.detach().cpu().float().numpy() if isinstance(array, torch.Tensor) else np.asarray(array, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(arr[finite], 2.0))
    hi = float(np.nanpercentile(arr[finite], 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _heat_image(array: torch.Tensor | np.ndarray, cmap_name: str = "magma") -> Image.Image:
    arr = _robust01(array)
    rgba = colormaps.get_cmap(cmap_name)(arr)
    return Image.fromarray((rgba[..., :3] * 255.0).astype(np.uint8), mode="RGB")


def _mask_image(mask: torch.Tensor) -> Image.Image:
    arr = np.clip(mask.detach().cpu().float().numpy(), 0.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 40).astype(np.uint8)
    rgb[..., 1] = (arr * 220).astype(np.uint8)
    rgb[..., 2] = (arr * 240).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _role_mode_image(role_t_h_w: torch.Tensor) -> Image.Image:
    role = torch.round(role_t_h_w.detach().cpu().float()).long()
    if role.ndim != 3:
        raise ValueError(f"expected T,H,W role map, got {tuple(role.shape)}")
    values = []
    for role_id in range(5):
        values.append((role == role_id).sum(dim=0, keepdim=True))
    mode = torch.cat(values, dim=0).argmax(dim=0).numpy()
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
    mode = np.clip(mode, 0, colours.shape[0] - 1)
    return Image.fromarray(colours[mode], mode="RGB")


def _label(image: Image.Image, text: str, *, width: int = 300) -> Image.Image:
    image = image.resize((width, int(round(width * image.height / max(image.width, 1)))), Image.Resampling.BILINEAR)
    header_h = 24
    out = Image.new("RGB", (image.width, image.height + header_h), (20, 20, 20))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), text, fill=(245, 245, 245))
    return out


def _write_contact_sheet(rows: Sequence[Dict[str, Any]], out_path: Path) -> Optional[str]:
    panels: List[Image.Image] = []
    for row in rows:
        if not row.get("_panel_arrays"):
            continue
        arrays = row["_panel_arrays"]
        chunk = int(row["chunk_idx"])
        row_imgs = [
            _label(_heat_image(arrays["read_mean"], "magma"), f"c{chunk} READ mean"),
            _label(_mask_image(arrays["read_q90_occupancy"]), "READ top10 occ"),
            _label(_role_mode_image(arrays["role"]), "TTT role mode"),
            _label(_heat_image(arrays["world_mean"], "viridis"), "world delta"),
            _label(_heat_image(arrays["action_mean"], "inferno"), "action delta"),
        ]
        total_w = sum(img.width for img in row_imgs)
        max_h = max(img.height for img in row_imgs)
        canvas = Image.new("RGB", (total_w, max_h), (0, 0, 0))
        x = 0
        for img in row_imgs:
            canvas.paste(img, (x, 0))
            x += img.width
        panels.append(canvas)
    if not panels:
        return None
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def _find_pairs(run_dir: Path) -> List[Tuple[int, Path, Path]]:
    read_dir = run_dir / "read_cue_patch_dumps"
    ttt_dir = run_dir / "ttt_spatial_post_delta_maps"
    pairs: List[Tuple[int, Path, Path]] = []
    for read_path in sorted(read_dir.glob("chunk_*_read_cue_patch.pt")):
        stem = read_path.name
        chunk_s = stem.split("_")[1]
        chunk_idx = int(chunk_s)
        ttt_path = ttt_dir / f"chunk_{chunk_idx:03d}_ttt_spatial_post_delta_map.pt"
        if ttt_path.exists():
            pairs.append((chunk_idx, read_path, ttt_path))
    return pairs


def _analyze_pair(
    *,
    chunk_idx: int,
    read_path: Path,
    ttt_path: Path,
    positive_role: int,
    neutral_role: int,
    negative_role: int,
) -> Dict[str, Any]:
    read_payload = _load_pt(read_path)
    ttt_payload = _load_pt(ttt_path)
    read_patch = _as_t_h_w(_payload_tensor(read_payload, "read_patch_final"))
    read_gt050 = _as_t_h_w(_payload_tensor(read_payload, "read_active_gt050_patch")).bool()
    read_q90 = _as_t_h_w(_payload_tensor(read_payload, "read_active_q90_patch")).bool()
    role = _as_t_h_w(_payload_tensor(ttt_payload, "R_ttt_tok_patch"))
    ttt_prior = _as_t_h_w(_payload_tensor(ttt_payload, "ttt_write_prior_patch"))
    world = _as_t_h_w(_payload_tensor(ttt_payload, "pass1_pass2_world_points_l2_patch"))
    local = _as_t_h_w(_payload_tensor(ttt_payload, "pass1_pass2_local_points_l2_patch"))
    action = _as_t_h_w(_payload_tensor(ttt_payload, "action_delta_norm_projection_patch"))
    read_patch, role = _align_shape(read_patch, role, name_a="read_patch_final", name_b="R_ttt_tok_patch")
    for name, value in {
        "read_gt050": read_gt050,
        "read_q90": read_q90,
        "ttt_prior": ttt_prior,
        "world": world,
        "local": local,
        "action": action,
    }.items():
        _align_shape(read_patch, value, name_a="read_patch_final", name_b=name)
    inactive_q90 = ~read_q90
    high_world = _top_fraction_mask(world, 0.10)
    high_action = _top_fraction_mask(action, 0.10)
    role_rounded = torch.round(role).long()
    row: Dict[str, Any] = {
        "chunk_idx": int(chunk_idx),
        "read_dump": str(read_path),
        "ttt_dump": str(ttt_path),
        "read_schema": read_payload.get("schema"),
        "ttt_schema": ttt_payload.get("schema"),
        "num_frames": int(read_patch.shape[0]),
        "patch_h": int(read_patch.shape[1]),
        "patch_w": int(read_patch.shape[2]),
        "read_mean": _finite_mean(read_patch),
        "read_gt050_mass": _mass(read_gt050),
        "read_q90_mass": _mass(read_q90),
        "ttt_prior_mean": _finite_mean(ttt_prior),
        "world_delta_mean": _finite_mean(world),
        "local_delta_mean": _finite_mean(local),
        "action_delta_mean": _finite_mean(action),
        "world_delta_read_q90_mean": _finite_mean(world, read_q90),
        "world_delta_inactive_mean": _finite_mean(world, inactive_q90),
        "local_delta_read_q90_mean": _finite_mean(local, read_q90),
        "local_delta_inactive_mean": _finite_mean(local, inactive_q90),
        "action_delta_read_q90_mean": _finite_mean(action, read_q90),
        "action_delta_inactive_mean": _finite_mean(action, inactive_q90),
        "ttt_prior_read_q90_mean": _finite_mean(ttt_prior, read_q90),
        "ttt_prior_inactive_mean": _finite_mean(ttt_prior, inactive_q90),
        "read_q90_high_world_intersection_mass": _mass(read_q90 & high_world),
        "read_q90_given_high_world": _conditional_mass(read_q90, high_world),
        "high_world_given_read_q90": _conditional_mass(high_world, read_q90),
        "read_q90_high_action_intersection_mass": _mass(read_q90 & high_action),
        "read_q90_given_high_action": _conditional_mass(read_q90, high_action),
        "high_action_given_read_q90": _conditional_mass(high_action, read_q90),
    }
    row["world_delta_active_minus_inactive"] = (
        row["world_delta_read_q90_mean"] - row["world_delta_inactive_mean"]
        if row["world_delta_read_q90_mean"] is not None and row["world_delta_inactive_mean"] is not None
        else None
    )
    row["action_delta_active_minus_inactive"] = (
        row["action_delta_read_q90_mean"] - row["action_delta_inactive_mean"]
        if row["action_delta_read_q90_mean"] is not None and row["action_delta_inactive_mean"] is not None
        else None
    )
    row["ttt_prior_active_minus_inactive"] = (
        row["ttt_prior_read_q90_mean"] - row["ttt_prior_inactive_mean"]
        if row["ttt_prior_read_q90_mean"] is not None and row["ttt_prior_inactive_mean"] is not None
        else None
    )
    for role_id, label in [
        (0, "unassigned"),
        (positive_role, "positive"),
        (neutral_role, "neutral"),
        (negative_role, "negative"),
    ]:
        role_mask = role_rounded == int(role_id)
        row[f"role_{label}_mass"] = _mass(role_mask)
        row[f"role_{label}_given_read_q90"] = _role_fraction(role, int(role_id), read_q90)
        row[f"read_q90_given_role_{label}"] = _conditional_mass(read_q90, role_mask)
        row[f"read_q90_role_{label}_intersection_mass"] = _mass(read_q90 & role_mask)
        row[f"high_world_given_role_{label}"] = _conditional_mass(high_world, role_mask)
        row[f"high_action_given_role_{label}"] = _conditional_mass(high_action, role_mask)
        row[f"high_world_given_read_q90_role_{label}"] = _conditional_mass(high_world, read_q90 & role_mask)
        row[f"high_action_given_read_q90_role_{label}"] = _conditional_mass(high_action, read_q90 & role_mask)
    row["_panel_arrays"] = {
        "read_mean": read_patch.mean(dim=0),
        "read_q90_occupancy": read_q90.float().mean(dim=0),
        "role": role,
        "world_mean": world.mean(dim=0),
        "action_mean": action.mean(dim=0),
    }
    return row


def _mean_rows(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    visible_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    keys: List[str] = []
    for row in visible_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in visible_rows:
            writer.writerow({k: row.get(k) for k in keys})


def _load_in_controller_alignment(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "hmc_state_hash.jsonl"
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "rows": 0,
            "scores": [],
            "sample": [],
        }
    scores: List[float] = []
    sample: List[Any] = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            rec = json.loads(line)
            log = (
                rec.get("prior_read_ttt_role_alignment")
                or rec.get("read_ttt_role_alignment")
                or rec.get("READ_TTT_ROLE_ALIGNMENT_LOG")
            )
            if log is not None and len(sample) < 3:
                sample.append(log)
            score = rec.get("prior_read_ttt_alignment_score", rec.get("read_ttt_alignment_score"))
            if isinstance(score, (int, float)) and np.isfinite(float(score)):
                scores.append(float(score))
    return {
        "available": bool(sample),
        "path": str(path),
        "rows": int(rows),
        "score_avg": float(np.mean(scores)) if scores else None,
        "scores": scores[:16],
        "sample": sample,
    }


def _write_observations(path: Path, summary: Dict[str, Any]) -> None:
    agg = summary["aggregate"]
    direct = summary.get("in_controller_alignment_log") or {}
    direct_available = bool(direct.get("available"))
    lines = [
        "# ACL2 v79 READ-TTT alignment probe observations",
        "",
        "This is a same-run posthoc dump comparison, not a strict Phase5 gate pass.",
        (
            "An in-controller READ/TTT role intersection log is available for this run; "
            "this file cross-checks it with dumped masks and TTT maps."
            if direct_available
            else "It uses dumped READ active masks and TTT role/post-delta maps, but it still lacks an in-controller READ stable/harm role intersection log."
        ),
        "",
        "## Aggregate",
        "",
        f"- chunks_analyzed={agg.get('chunks_analyzed')}",
        f"- read_q90_mass_avg={agg.get('read_q90_mass_avg')}",
        f"- role_positive_given_read_q90_avg={agg.get('role_positive_given_read_q90_avg')}",
        f"- role_neutral_given_read_q90_avg={agg.get('role_neutral_given_read_q90_avg')}",
        f"- role_negative_given_read_q90_avg={agg.get('role_negative_given_read_q90_avg')}",
        f"- high_world_given_read_q90_avg={agg.get('high_world_given_read_q90_avg')}",
        f"- high_world_given_role_positive_avg={agg.get('high_world_given_role_positive_avg')}",
        f"- high_world_given_read_q90_role_positive_avg={agg.get('high_world_given_read_q90_role_positive_avg')}",
        f"- world_delta_active_minus_inactive_avg={agg.get('world_delta_active_minus_inactive_avg')}",
        f"- action_delta_active_minus_inactive_avg={agg.get('action_delta_active_minus_inactive_avg')}",
        f"- in_controller_alignment_log_available={direct_available}",
        f"- in_controller_alignment_score_avg={direct.get('score_avg')}",
        "",
        "## Interpretation boundary",
        "",
        "- If READ active mass mainly overlaps TTT negative/unassigned roles or high world/action delta zones, the weak READ signal is likely not being converted into a stable persistent TTT carrier.",
        "- If READ active mass overlaps TTT positive role but action/world delta remains worse than inactive regions, the actuator may be applying the right role to a geometrically unstable regime.",
        "- This probe is still posthoc evidence; the Phase5 gate must be read from the runner decision JSON, not from this visualization/probe alone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--positive-role", type=int, default=1)
    parser.add_argument("--neutral-role", type=int, default=2)
    parser.add_argument("--negative-role", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "read_ttt_alignment_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _find_pairs(args.run_dir)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for chunk_idx, read_path, ttt_path in pairs:
        try:
            rows.append(
                _analyze_pair(
                    chunk_idx=chunk_idx,
                    read_path=read_path,
                    ttt_path=ttt_path,
                    positive_role=int(args.positive_role),
                    neutral_role=int(args.neutral_role),
                    negative_role=int(args.negative_role),
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "chunk_idx": int(chunk_idx),
                    "read_dump": str(read_path),
                    "ttt_dump": str(ttt_path),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
    contact_sheet = _write_contact_sheet(rows, out_dir / "read_ttt_alignment_contact_sheet.png")
    aggregate = {
        "chunks_analyzed": int(len(rows)),
        "chunk_indices": [int(row["chunk_idx"]) for row in rows],
        "read_q90_mass_avg": _mean_rows(rows, "read_q90_mass"),
        "role_positive_given_read_q90_avg": _mean_rows(rows, "role_positive_given_read_q90"),
        "role_neutral_given_read_q90_avg": _mean_rows(rows, "role_neutral_given_read_q90"),
        "role_negative_given_read_q90_avg": _mean_rows(rows, "role_negative_given_read_q90"),
        "read_q90_given_role_positive_avg": _mean_rows(rows, "read_q90_given_role_positive"),
        "read_q90_given_role_negative_avg": _mean_rows(rows, "read_q90_given_role_negative"),
        "high_world_given_read_q90_avg": _mean_rows(rows, "high_world_given_read_q90"),
        "read_q90_given_high_world_avg": _mean_rows(rows, "read_q90_given_high_world"),
        "high_world_given_role_positive_avg": _mean_rows(rows, "high_world_given_role_positive"),
        "high_action_given_role_positive_avg": _mean_rows(rows, "high_action_given_role_positive"),
        "high_world_given_read_q90_role_positive_avg": _mean_rows(rows, "high_world_given_read_q90_role_positive"),
        "high_action_given_read_q90_role_positive_avg": _mean_rows(rows, "high_action_given_read_q90_role_positive"),
        "world_delta_active_minus_inactive_avg": _mean_rows(rows, "world_delta_active_minus_inactive"),
        "action_delta_active_minus_inactive_avg": _mean_rows(rows, "action_delta_active_minus_inactive"),
        "ttt_prior_active_minus_inactive_avg": _mean_rows(rows, "ttt_prior_active_minus_inactive"),
    }
    in_controller_alignment = _load_in_controller_alignment(args.run_dir)
    strict_blocker = (
        "posthoc_same_run_dump_comparison_only; in-controller READ_TTT role log is available, but Phase5 success still depends on runner metric/control gates"
        if bool(in_controller_alignment.get("available"))
        else "posthoc_same_run_dump_comparison_only; missing in-controller READ stable/harm to TTT positive/negative alignment log"
    )
    summary = {
        "schema": "acl2_v79_read_ttt_alignment_probe_v1",
        "run_dir": str(args.run_dir),
        "out_dir": str(out_dir),
        "strict_phase5_alignment_gate_pass": False,
        "strict_gate_blocker": strict_blocker,
        "in_controller_alignment_log": in_controller_alignment,
        "role_mapping": {
            "positive": int(args.positive_role),
            "neutral": int(args.neutral_role),
            "negative": int(args.negative_role),
            "unassigned": 0,
        },
        "pairs_found": int(len(pairs)),
        "errors": errors,
        "aggregate": aggregate,
        "contact_sheet": contact_sheet,
        "rows": [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
    }
    _write_csv(out_dir / "read_ttt_alignment_per_chunk.csv", rows)
    (out_dir / "read_ttt_alignment_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_observations(out_dir / "read_ttt_alignment_observations.md", summary)
    print(json.dumps(_jsonable({"pairs_found": len(pairs), "chunks_analyzed": len(rows), "errors": errors, "aggregate": aggregate}), indent=2, sort_keys=True))
    print(f"wrote_summary={out_dir / 'read_ttt_alignment_summary.json'}")
    print(f"wrote_csv={out_dir / 'read_ttt_alignment_per_chunk.csv'}")
    if contact_sheet:
        print(f"wrote_contact_sheet={contact_sheet}")


if __name__ == "__main__":
    main()
