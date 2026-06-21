#!/usr/bin/env python3
"""ACL2 v70 RADIO/TTT diagnostic over v68 spatial post-zp delta maps.

This is an R5 diagnostic, not an online HMC success claim. It checks whether
existing spatial/token-aligned TTT post-zp delta projection maps can support the
v70 TTT path, then applies RADIO sidecar group masks to the write-prior and
post-zp projection maps with matched shuffle/random controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from diagnose_v70_radio_merge_oracle import _index_sidecars, _load_sidecar
    from v70_radio_sidecar_common import parse_chunks, utc_now
except ImportError:  # pragma: no cover
    from tools.diagnose_v70_radio_merge_oracle import _index_sidecars, _load_sidecar
    from tools.v70_radio_sidecar_common import parse_chunks, utc_now


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs: List[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            xs.append(val)
    return float(np.mean(xs)) if xs else None


def _finite_median(values: Iterable[Any]) -> Optional[float]:
    xs: List[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            xs.append(val)
    return float(np.median(xs)) if xs else None


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return float(num / den)


def _rel_change(after: float, before: float) -> float:
    if not math.isfinite(after) or not math.isfinite(before) or abs(before) <= 1e-12:
        return float("nan")
    return float((after - before) / abs(before))


def _stats_tensor(x: torch.Tensor) -> Dict[str, Any]:
    vals = x.detach().cpu().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"mean": None, "max": None, "nonzero_frac": None}
    return {
        "mean": float(vals.mean().item()),
        "max": float(vals.max().item()),
        "nonzero_frac": float((vals.abs() > 0).float().mean().item()),
    }


def _find_spatial_maps(root: Path, target_chunks: Sequence[int]) -> Dict[int, Path]:
    chunks = {int(c) for c in target_chunks}
    out: Dict[int, Path] = {}
    for path in sorted(root.glob("**/ttt_spatial_post_delta_maps/chunk_*_ttt_spatial_post_delta_map.pt")):
        try:
            chunk = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if chunk in chunks and chunk not in out:
            out[chunk] = path
    return out


def _load_spatial_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected dict payload")
    if payload.get("schema") != "acl2_v68_ttt_spatial_post_delta_map_v1":
        raise ValueError(f"{path}: unsupported schema {payload.get('schema')!r}")
    return payload


def _tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"spatial payload missing tensor {key}")
    return value.detach().cpu().float()


def _crop_sidecar_field(
    sidecar: Mapping[str, Any],
    key: str,
    *,
    start_frame: int,
    frames: int,
) -> torch.Tensor:
    value = sidecar.get(key)
    if value is None:
        raise ValueError(f"sidecar missing field {key}")
    arr = value.detach().cpu().float() if torch.is_tensor(value) else torch.as_tensor(value).float()
    local_start = int(start_frame) - int(sidecar["global_start_frame"])
    local_end = local_start + int(frames)
    if local_start < 0 or local_end > int(arr.shape[0]):
        raise ValueError(
            f"{key}: requested frames [{local_start},{local_end}) outside sidecar shape {tuple(arr.shape)}"
        )
    return arr[local_start:local_end]


def _resize_to_patch_grid(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    # x: F,H,W. Bilinear keeps soft RADIO masks stable after the 24x78 -> 19x66 projection.
    y = F.interpolate(x[:, None, :, :], size=out_hw, mode="bilinear", align_corners=False)[:, 0]
    return torch.clamp(y.float(), 0.0, 1.0)


def _radio_masks_for_payload(
    *,
    sidecar: Mapping[str, Any],
    start_frame: int,
    frames: int,
    out_hw: Tuple[int, int],
    min_confidence: float,
    min_stability: float,
    min_interior: float,
    max_activity_risk: float,
    boundary_threshold: float,
) -> Dict[str, torch.Tensor]:
    conf = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "radio_confidence", start_frame=start_frame, frames=frames),
        out_hw,
    )
    stability = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "temporal_stability", start_frame=start_frame, frames=frames),
        out_hw,
    )
    interior = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "object_interior_score", start_frame=start_frame, frames=frames),
        out_hw,
    )
    dynamic = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "radio_dynamic_score", start_frame=start_frame, frames=frames),
        out_hw,
    )
    sky = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "radio_sky_context_score", start_frame=start_frame, frames=frames),
        out_hw,
    )
    lowtrust = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "radio_lowtrust_score", start_frame=start_frame, frames=frames),
        out_hw,
    )
    boundary = _resize_to_patch_grid(
        _crop_sidecar_field(sidecar, "object_boundary_score", start_frame=start_frame, frames=frames),
        out_hw,
    )
    activity = torch.maximum(torch.maximum(dynamic, sky), lowtrust)
    static = (
        (conf >= float(min_confidence))
        & (stability >= float(min_stability))
        & (interior >= float(min_interior))
        & (activity <= float(max_activity_risk))
    ).float()
    dynamic_lowstable = (
        (activity > float(max_activity_risk))
        | (stability < float(min_stability))
        | (conf < float(min_confidence))
    ).float()
    cross_object = ((boundary >= float(boundary_threshold)) & (interior < float(min_interior))).float()
    reliable = (
        (conf >= float(min_confidence))
        & (stability >= float(min_stability))
        & (activity <= float(max_activity_risk))
    ).float()
    return {
        "static": static,
        "dynamic_lowstable": dynamic_lowstable,
        "cross_object": cross_object,
        "reliable": reliable,
        "activity_risk": activity,
        "confidence": conf,
        "stability": stability,
        "interior": interior,
        "boundary": boundary,
    }


def _permute_like(mask: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    flat = mask.detach().cpu().float().reshape(-1).numpy()
    perm = rng.permutation(flat.shape[0])
    return torch.from_numpy(flat[perm].reshape(tuple(mask.shape))).float()


def _candidate_scale(
    name: str,
    masks: Mapping[str, torch.Tensor],
    *,
    static_boost: float,
    suppress: float,
) -> torch.Tensor:
    base = torch.ones_like(masks["static"])
    if name == "radio_static_object_write_floor":
        return base + float(static_boost) * masks["static"]
    if name == "radio_dynamic_lowstable_no_persistent":
        return torch.clamp(base - float(suppress) * masks["dynamic_lowstable"], min=0.0)
    if name == "radio_cross_object_conflict_veto":
        return torch.clamp(base - float(suppress) * masks["cross_object"], min=0.0)
    if name == "radio_combined_static_dynamic_cross":
        scale = base + float(static_boost) * masks["static"]
        scale = scale * torch.clamp(base - float(suppress) * masks["dynamic_lowstable"], min=0.0)
        scale = scale * torch.clamp(base - float(suppress) * masks["cross_object"], min=0.0)
        return torch.clamp(scale, min=0.0)
    raise KeyError(name)


def _renormalize(candidate: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    c_mean = float(candidate.float().mean().item())
    n_mean = float(native.float().mean().item())
    if not math.isfinite(c_mean) or abs(c_mean) <= 1e-12:
        return candidate
    return candidate * (n_mean / c_mean)


def _mass(tensor: torch.Tensor, weight: torch.Tensor) -> float:
    return float((tensor.float() * weight.float()).sum().item())


def _mass_frac(tensor: torch.Tensor, weight: torch.Tensor) -> float:
    return _safe_ratio(_mass(tensor, weight), float(tensor.float().sum().item()))


def _make_row(
    *,
    payload: Mapping[str, Any],
    path: Path,
    sidecar_path: Path,
    candidate_type: str,
    candidate_family: str,
    group_name: str,
    scale: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    renormalize_total: bool,
    min_group_change: float,
    min_post_delta_change: float,
) -> Dict[str, Any]:
    prior = _tensor(payload, "ttt_write_prior_patch")
    committed = _tensor(payload, "committed_post_delta_norm_projection_patch")
    native = _tensor(payload, "native_delta_norm_projection_patch")
    action = _tensor(payload, "action_delta_norm_projection_patch")
    group = masks[group_name].float()
    candidate_prior = prior * scale.float()
    if renormalize_total:
        candidate_prior = _renormalize(candidate_prior, prior)
    ratio = candidate_prior / torch.clamp(prior, min=1e-6)
    candidate_committed = committed * ratio[None, :, :, :]
    candidate_native = native * ratio[None, :, :, :]
    candidate_action = action * ratio[None, :, :, :]

    prior_before_group = _mass_frac(prior, group)
    prior_after_group = _mass_frac(candidate_prior, group)
    committed_before_group = _mass_frac(committed.mean(dim=0), group)
    committed_after_group = _mass_frac(candidate_committed.mean(dim=0), group)
    total_prior_change = float(torch.mean(torch.abs(candidate_prior - prior)).item() / max(float(prior.mean().item()), 1e-12))
    total_committed_change = float(
        torch.mean(torch.abs(candidate_committed - committed)).item()
        / max(float(committed.mean().item()), 1e-12)
    )
    layer_rows = payload.get("layer_branch_rows") or []
    native_cos = [
        row.get("candidate_native_cosine")
        for row in layer_rows
        if isinstance(row, dict) and isinstance(row.get("candidate_native_cosine"), (int, float))
    ]
    action_native_cos = [
        row.get("candidate_action_native_cosine")
        for row in layer_rows
        if isinstance(row, dict) and isinstance(row.get("candidate_action_native_cosine"), (int, float))
    ]
    native_cos_mean = _finite_mean(native_cos)
    action_native_cos_mean = _finite_mean(action_native_cos)
    native_action_cosine_projection = (
        prior[None, :, :, :] * float(action_native_cos_mean)
        if action_native_cos_mean is not None and math.isfinite(float(action_native_cos_mean))
        else torch.empty((0,), dtype=torch.float32)
    )

    intended_change = abs(_rel_change(prior_after_group, prior_before_group))
    post_delta_change = abs(_rel_change(committed_after_group, committed_before_group))
    write_group_change_pass = bool(math.isfinite(intended_change) and intended_change >= float(min_group_change))
    post_delta_change_pass = bool(math.isfinite(post_delta_change) and post_delta_change >= float(min_post_delta_change))
    return {
        "source_spatial_map": str(path),
        "source_radio_sidecar": str(sidecar_path),
        "chunk": int(payload.get("chunk_idx")),
        "candidate_type": candidate_type,
        "candidate_family": candidate_family,
        "group_name": group_name,
        "schema": payload.get("schema"),
        "spatial_token_aligned": bool(payload.get("spatial_token_aligned", False)),
        "projection_not_raw_per_token_fast_weight_delta": bool(
            payload.get("projection_not_raw_per_token_fast_weight_delta", False)
        ),
        "full_resolution_output_delta_maps_stored": bool(payload.get("full_resolution_output_delta_maps_stored", False)),
        "native_action_cosine_map_is_projection": True,
        "native_action_cosine_raw_spatial_map_available": False,
        "num_frames": int(payload.get("num_frames")),
        "patch_grid": list(payload.get("patch_grid") or []),
        "layer_branch_rows": int(len(layer_rows)) if isinstance(layer_rows, list) else 0,
        "renormalize_total": bool(renormalize_total),
        "radio_static_mask_mean": float(masks["static"].mean().item()),
        "radio_dynamic_lowstable_mask_mean": float(masks["dynamic_lowstable"].mean().item()),
        "radio_cross_object_mask_mean": float(masks["cross_object"].mean().item()),
        "native_prior_group_mass_frac": prior_before_group,
        "candidate_prior_group_mass_frac": prior_after_group,
        "prior_group_mass_relative_change": _rel_change(prior_after_group, prior_before_group),
        "committed_delta_group_mass_frac_before": committed_before_group,
        "committed_delta_group_mass_frac_after": committed_after_group,
        "committed_delta_group_mass_relative_change": _rel_change(committed_after_group, committed_before_group),
        "total_prior_abs_change_ratio": total_prior_change,
        "total_committed_delta_abs_change_ratio": total_committed_change,
        "native_delta_total_abs_change_ratio": float(
            torch.mean(torch.abs(candidate_native - native)).item()
            / max(float(native.mean().item()), 1e-12)
        ),
        "action_delta_total_abs_change_ratio": float(
            torch.mean(torch.abs(candidate_action - action)).item()
            / max(float(action.mean().item()), 1e-12)
        ),
        "candidate_native_cosine_mean": native_cos_mean,
        "candidate_action_native_cosine_mean": action_native_cos_mean,
        "native_action_cosine_projection_mean": (
            float(native_action_cosine_projection.float().mean().item())
            if native_action_cosine_projection.numel()
            else None
        ),
        "write_group_change_pass": write_group_change_pass,
        "post_delta_change_pass": post_delta_change_pass,
        "diagnostic_action_pass": bool(write_group_change_pass and post_delta_change_pass),
        "prior_stats": _stats_tensor(prior),
        "candidate_prior_stats": _stats_tensor(candidate_prior),
        "committed_delta_stats": _stats_tensor(committed),
        "candidate_committed_delta_stats": _stats_tensor(candidate_committed),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-map-root", type=Path, required=True)
    parser.add_argument("--radio-sidecar-dir", type=Path, action="append", required=True)
    parser.add_argument("--target-chunks", default="6,7,8,10,12,19,20,29,30,31,32")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--radio-min-confidence", type=float, default=0.45)
    parser.add_argument("--radio-min-stability", type=float, default=0.35)
    parser.add_argument("--radio-min-interior", type=float, default=0.25)
    parser.add_argument("--radio-max-activity-risk", type=float, default=0.85)
    parser.add_argument("--radio-boundary-threshold", type=float, default=0.85)
    parser.add_argument("--static-boost", type=float, default=0.35)
    parser.add_argument("--suppress", type=float, default=0.50)
    parser.add_argument("--min-group-change", type=float, default=0.20)
    parser.add_argument("--min-post-delta-change", type=float, default=0.05)
    parser.add_argument("--renormalize-total", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7070)
    args = parser.parse_args()

    target_chunks = parse_chunks(args.target_chunks)
    spatial_index = _find_spatial_maps(args.spatial_map_root, target_chunks)
    sidecar_index = _index_sidecars(args.radio_sidecar_dir)
    sidecar_cache: Dict[int, Dict[str, Any]] = {}
    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    missing_spatial = sorted(set(target_chunks) - set(spatial_index))
    missing_sidecar = sorted(set(target_chunks) - set(sidecar_index))
    candidates = [
        ("radio_static_object_write_floor", "static"),
        ("radio_dynamic_lowstable_no_persistent", "dynamic_lowstable"),
        ("radio_cross_object_conflict_veto", "cross_object"),
        ("radio_combined_static_dynamic_cross", "dynamic_lowstable"),
    ]

    for chunk in target_chunks:
        if chunk not in spatial_index or chunk not in sidecar_index:
            continue
        spatial_path = spatial_index[chunk]
        sidecar_path = sidecar_index[chunk]
        try:
            payload = _load_spatial_payload(spatial_path)
            sidecar = _load_sidecar(sidecar_index, chunk, sidecar_cache)
            prior = _tensor(payload, "ttt_write_prior_patch")
            masks = _radio_masks_for_payload(
                sidecar=sidecar,
                start_frame=int(payload["start_frame"]),
                frames=int(prior.shape[0]),
                out_hw=(int(prior.shape[1]), int(prior.shape[2])),
                min_confidence=float(args.radio_min_confidence),
                min_stability=float(args.radio_min_stability),
                min_interior=float(args.radio_min_interior),
                max_activity_risk=float(args.radio_max_activity_risk),
                boundary_threshold=float(args.radio_boundary_threshold),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic should record bad chunks.
            failures.append({
                "chunk": int(chunk),
                "spatial_path": str(spatial_path),
                "sidecar_path": str(sidecar_path),
                "failure": f"{type(exc).__name__}:{exc}",
            })
            continue

        rows.append(
            _make_row(
                payload=payload,
                path=spatial_path,
                sidecar_path=sidecar_path,
                candidate_type="native_ttt_write_prior",
                candidate_family="baseline",
                group_name="reliable",
                scale=torch.ones_like(prior),
                masks=masks,
                renormalize_total=bool(args.renormalize_total),
                min_group_change=float(args.min_group_change),
                min_post_delta_change=float(args.min_post_delta_change),
            )
        )
        for cand_name, group_name in candidates:
            scale = _candidate_scale(cand_name, masks, static_boost=float(args.static_boost), suppress=float(args.suppress))
            rows.append(
                _make_row(
                    payload=payload,
                    path=spatial_path,
                    sidecar_path=sidecar_path,
                    candidate_type=cand_name,
                    candidate_family="radio",
                    group_name=group_name,
                    scale=scale,
                    masks=masks,
                    renormalize_total=bool(args.renormalize_total),
                    min_group_change=float(args.min_group_change),
                    min_post_delta_change=float(args.min_post_delta_change),
                )
            )
            shuffled = dict(masks)
            shuffled[group_name] = _permute_like(masks[group_name], rng)
            scale_shuf = _candidate_scale(
                cand_name,
                shuffled,
                static_boost=float(args.static_boost),
                suppress=float(args.suppress),
            )
            rows.append(
                _make_row(
                    payload=payload,
                    path=spatial_path,
                    sidecar_path=sidecar_path,
                    candidate_type=f"{cand_name}_spatial_shuffle",
                    candidate_family="control",
                    group_name=group_name,
                    scale=scale_shuf,
                    masks=masks,
                    renormalize_total=bool(args.renormalize_total),
                    min_group_change=float(args.min_group_change),
                    min_post_delta_change=float(args.min_post_delta_change),
                )
            )

    # Candidate beats controls only within the same chunk and logical family prefix.
    for row in rows:
        if row.get("candidate_family") != "radio":
            row["beats_matched_control_by_post_delta_change"] = False
            continue
        prefix = str(row["candidate_type"])
        control_name = f"{prefix}_spatial_shuffle"
        controls = [
            other for other in rows
            if int(other.get("chunk", -1)) == int(row["chunk"])
            and other.get("candidate_type") == control_name
        ]
        best_control = max(
            [abs(float(other.get("committed_delta_group_mass_relative_change") or 0.0)) for other in controls],
            default=float("-inf"),
        )
        value = abs(float(row.get("committed_delta_group_mass_relative_change") or 0.0))
        row["matched_control_abs_post_delta_change"] = best_control if math.isfinite(best_control) else None
        row["beats_matched_control_by_post_delta_change"] = bool(value > best_control)
        row["diagnostic_radio_gate_pass"] = bool(
            row.get("diagnostic_action_pass")
            and row.get("beats_matched_control_by_post_delta_change")
        )

    radio_gate_rows = [row for row in rows if row.get("candidate_family") == "radio" and row.get("diagnostic_radio_gate_pass")]
    diagnostic_gate_chunks = sorted({int(row["chunk"]) for row in radio_gate_rows})
    summary = {
        "schema": "acl2_v70_radio_ttt_spatial_delta_diagnostic_summary_v1",
        "created_at": utc_now(),
        "spatial_map_root": str(args.spatial_map_root),
        "radio_sidecar_dirs": [str(x) for x in args.radio_sidecar_dir],
        "target_chunks": list(target_chunks),
        "spatial_map_chunks": sorted(spatial_index),
        "sidecar_chunks": sorted(sidecar_index),
        "missing_spatial_chunks": missing_spatial,
        "missing_sidecar_chunks": missing_sidecar,
        "failures": failures,
        "rows": len(rows),
        "counts": {
            "radio_rows": sum(row.get("candidate_family") == "radio" for row in rows),
            "control_rows": sum(row.get("candidate_family") == "control" for row in rows),
            "baseline_rows": sum(row.get("candidate_family") == "baseline" for row in rows),
            "write_group_change_pass": sum(bool(row.get("write_group_change_pass")) for row in rows),
            "post_delta_change_pass": sum(bool(row.get("post_delta_change_pass")) for row in rows),
            "diagnostic_action_pass": sum(bool(row.get("diagnostic_action_pass")) for row in rows),
            "diagnostic_radio_gate_rows": len(radio_gate_rows),
        },
        "diagnostic_gate_chunks": diagnostic_gate_chunks,
        "candidate_counts": {
            name: {
                "rows": sum(row.get("candidate_type") == name for row in rows),
                "diagnostic_radio_gate_pass": sum(
                    row.get("candidate_type") == name and bool(row.get("diagnostic_radio_gate_pass"))
                    for row in rows
                ),
            }
            for name in sorted({str(row.get("candidate_type")) for row in rows})
        },
        "median_radio_prior_group_change_abs": _finite_median(
            abs(float(row.get("prior_group_mass_relative_change")))
            for row in rows
            if row.get("candidate_family") == "radio"
        ),
        "median_radio_post_delta_group_change_abs": _finite_median(
            abs(float(row.get("committed_delta_group_mass_relative_change")))
            for row in rows
            if row.get("candidate_family") == "radio"
        ),
        "native_action_cosine_raw_spatial_map_available": False,
        "native_action_cosine_map_status": "projection_only_from_layer_branch_scalar_cosines",
        "official_ttt_online_gate_evaluated": False,
        "r6_online_allowed_by_this_diagnostic": False,
        "decision": "diagnostic_only_no_online_ttt_promotion",
        "note": (
            "This diagnostic uses v68 spatial/token-aligned post-zp delta projection maps. "
            "It does not contain raw per-token fast-weight tensors, raw spatial native/action "
            "cosine maps, or future/local/scale trajectory metrics."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "radio_ttt_spatial_delta_results.csv", rows)
    (args.out_dir / "radio_ttt_spatial_delta_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
