#!/usr/bin/env python3
"""Build ACL2 v84 Phase1 Memory Ruler candidate universe.

The builder uses existing overlap pair tensors plus current-chunk direct READ
and SWA/PCA dumps when available. It does not backfill missing memory usage:
unavailable READ/SWA values stay empty and are counted in the gate summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PAIR_BANK = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase1_ruler_candidate_universe")
DEFAULT_DIRECT_ROOTS = [
    Path(
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase2_direct_hook_repair"
    ),
    Path("results/kitti01_hmc_v2/acl2_v84tf_memory_ruler_audit/phase1_direct_hook_repair"),
]

PATCH_GRID = (19, 66)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-bank", type=Path, default=DEFAULT_PAIR_BANK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--direct-root", type=Path, action="append", default=None)
    parser.add_argument("--max-token-rows-per-pair", type=int, default=240)
    parser.add_argument("--pairwise-max-anchors", type=int, default=80)
    parser.add_argument(
        "--usage-neighborhood-radius",
        type=int,
        default=0,
        help="Aggregate READ/SWA usage over a local patch neighborhood to repair nearest-grid mismatch.",
    )
    parser.add_argument("--torch-load-limit-pairs", type=int, default=0, help="0 means all pairs")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def clamp01(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.0, float(value)))


def seq_norm(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.zfill(2)


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_float(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return value


def finite_mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else None


def robust_scale(values: Sequence[float], q: float = 0.90) -> float:
    vals = sorted(abs(float(v)) for v in values if math.isfinite(float(v)))
    if not vals:
        return 1.0
    idx = max(0, min(len(vals) - 1, int(round((len(vals) - 1) * q))))
    return max(vals[idx], 1e-6)


def mode_and_purity(values: Sequence[int]) -> tuple[int | None, float | None]:
    vals = [int(v) for v in values]
    if not vals:
        return None, None
    counter = Counter(vals)
    label, count = counter.most_common(1)[0]
    return label, count / len(vals)


def parse_seq_chunk(path: Path) -> tuple[str, int | None]:
    text = str(path)
    seq_match = re.search(r"seq[_-]?(\d{2})", text)
    if not seq_match:
        seq_match = re.search(r"/(\d{2})/", text)
    chunk_match = re.search(r"chunk[_]?(\d{3})", text)
    return (seq_match.group(1) if seq_match else "", int(chunk_match.group(1)) if chunk_match else None)


def discover_direct_paths(roots: Sequence[Path]) -> tuple[dict[tuple[str, int], list[Path]], dict[tuple[str, int], list[Path]]]:
    read_paths: dict[tuple[str, int], list[Path]] = defaultdict(list)
    pca_paths: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/read_cue_patch_dumps/chunk_*_read_cue_patch.pt")):
            seq, chunk = parse_seq_chunk(path)
            if seq and chunk is not None:
                read_paths[(seq, chunk)].append(path)
        for path in sorted(root.glob("**/pca_features/chunk_*.pt")):
            seq, chunk = parse_seq_chunk(path)
            if seq and chunk is not None:
                pca_paths[(seq, chunk)].append(path)
    return dict(read_paths), dict(pca_paths)


def choose_read_payload(paths: Sequence[Path]) -> tuple[Path | None, dict[str, Any] | None, str]:
    for path in paths:
        try:
            payload = torch_load(path)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("tensors"), dict) and "read_patch_final" in payload["tensors"]:
            return path, payload, "read_cue_patch_dump"
    return None, None, "missing_read_cue_patch_dump"


def choose_swa_payload(paths: Sequence[Path]) -> tuple[Path | None, dict[str, Any] | None, str]:
    required = [
        "tap::pca_swa_current_q_layers",
        "tap::pca_swa_current_k_layers",
        "tap::pca_swa_cache_k_layers",
        "tap::pca_swa_current_v_layers",
        "tap::pca_swa_cache_v_layers",
    ]
    for path in paths:
        try:
            payload = torch_load(path)
        except Exception:
            continue
        if isinstance(payload, dict) and all(key in payload for key in required):
            return path, payload, "qk_compatibility_proxy_no_route_mass"
    return None, None, "missing_swa_pca_dump"


def read_patch_value(
    payload: dict[str, Any] | None,
    key: str,
    frame_idx: int,
    py: int,
    px: int,
    *,
    radius: int = 0,
) -> float | None:
    if payload is None:
        return None
    tensors = payload.get("tensors") if isinstance(payload.get("tensors"), dict) else {}
    value = tensors.get(key)
    try:
        import torch

        if not torch.is_tensor(value):
            return None
        if value.ndim < 3 or frame_idx < 0 or frame_idx >= value.shape[0]:
            return None
        y0 = max(0, py - int(radius))
        y1 = min(int(value.shape[1]), py + int(radius) + 1)
        x0 = max(0, px - int(radius))
        x1 = min(int(value.shape[2]), px + int(radius) + 1)
        window = value[frame_idx, y0:y1, x0:x1].float()
        if window.numel() == 0:
            return None
        # Local max is used only to tolerate nearest-grid mismatch between
        # overlap pixels and direct-hook patch tensors.
        return float(window.max().item())
    except Exception:
        return None


def swa_qk_value(
    payload: dict[str, Any] | None,
    frame_idx: int,
    py: int,
    px: int,
    *,
    radius: int = 0,
) -> tuple[float | None, float | None]:
    if payload is None:
        return None, None
    import torch

    try:
        q_tensor = payload["tap::pca_swa_current_q_layers"]
        y0 = max(0, py - int(radius))
        y1 = min(int(q_tensor.shape[2]), py + int(radius) + 1)
        x0 = max(0, px - int(radius))
        x1 = min(int(q_tensor.shape[3]), px + int(radius) + 1)
        q = payload["tap::pca_swa_current_q_layers"][frame_idx, 0, y0:y1, x0:x1].float().reshape(-1, q_tensor.shape[-1])
        k = payload["tap::pca_swa_cache_k_layers"][frame_idx, 0, y0:y1, x0:x1].float().reshape(-1, q_tensor.shape[-1])
        v = payload["tap::pca_swa_current_v_layers"][frame_idx, 0, y0:y1, x0:x1].float().reshape(-1, q_tensor.shape[-1])
        cv = payload["tap::pca_swa_cache_v_layers"][frame_idx, 0, y0:y1, x0:x1].float().reshape(-1, q_tensor.shape[-1])
        qk_denom = q.norm(dim=-1) * k.norm(dim=-1)
        vv_denom = v.norm(dim=-1) * cv.norm(dim=-1)
        qk_valid = qk_denom > 1e-8
        vv_valid = vv_denom > 1e-8
        qk = float(((q * k).sum(dim=-1)[qk_valid] / qk_denom[qk_valid]).mean().item()) if bool(qk_valid.any()) else None
        vv = float(((v * cv).sum(dim=-1)[vv_valid] / vv_denom[vv_valid]).mean().item()) if bool(vv_valid.any()) else None
        return clamp01((qk + 1.0) / 2.0) if qk is not None else None, clamp01((vv + 1.0) / 2.0) if vv is not None else None
    except Exception:
        return None, None


def make_patch_indices(coords: Any, grid: tuple[int, int]) -> tuple[Any, str]:
    import torch

    c = coords.long()
    y = c[:, 0]
    x = c[:, 1]
    h = int(y.max().item()) + 1 if y.numel() else 1
    w = int(x.max().item()) + 1 if x.numel() else 1
    py = torch.clamp((y.float() / max(h, 1) * grid[0]).long(), 0, grid[0] - 1)
    px = torch.clamp((x.float() / max(w, 1) * grid[1]).long(), 0, grid[1] - 1)
    return py * grid[1] + px, f"pixel_range_yx=({h},{w});grid={grid[0]}x{grid[1]};nearest_floor"


def pairwise_distance_stats(anchor_points: list[tuple[tuple[float, float, float], tuple[float, float, float]]], max_anchors: int) -> dict[str, Any]:
    if len(anchor_points) < 2:
        return {"median": None, "mad": None, "count": 0}
    import itertools
    import math as _math

    if len(anchor_points) > max_anchors:
        step = max(1, len(anchor_points) // max_anchors)
        anchor_points = anchor_points[::step][:max_anchors]
    deltas: list[float] = []
    for (p0, c0), (p1, c1) in itertools.combinations(anchor_points, 2):
        dp = _math.dist(p0, p1)
        dc = _math.dist(c0, c1)
        if dp > 1e-6 and dc > 1e-6:
            deltas.append(_math.log((dp + 1e-6) / (dc + 1e-6)))
    if not deltas:
        return {"median": None, "mad": None, "count": 0}
    med = median(deltas)
    mad = median([abs(v - med) for v in deltas])
    return {"median": med, "mad": mad, "count": len(deltas)}


def build_pair_candidates(
    pair_row: Mapping[str, str],
    read_paths: dict[tuple[str, int], list[Path]],
    pca_paths: dict[tuple[str, int], list[Path]],
    *,
    max_token_rows: int,
    pairwise_max_anchors: int,
    usage_neighborhood_radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    seq = seq_norm(pair_row.get("seq"))
    prev_chunk = safe_int(pair_row.get("prev_chunk"))
    curr_chunk = safe_int(pair_row.get("curr_chunk"))
    source_path = Path(str(pair_row.get("source_path") or ""))
    row_base = {
        "seq": seq,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "case_type": pair_row.get("case_type", ""),
        "base_case_type": pair_row.get("base_case_type", ""),
        "quality_source": pair_row.get("quality_source", ""),
        "quality_type": pair_row.get("quality_type", ""),
        "source_path": str(source_path),
    }

    summary: dict[str, Any] = dict(row_base)
    if not source_path.is_file():
        summary.update({"pair_available": False, "missing_reason": "missing_overlap_pair_pt"})
        return [], [], summary

    overlap = torch_load(source_path)
    if not isinstance(overlap, dict):
        summary.update({"pair_available": False, "missing_reason": "invalid_overlap_payload"})
        return [], [], summary

    read_path, read_payload, read_status = choose_read_payload(read_paths.get((seq, int(curr_chunk or -1)), []))
    swa_path, swa_payload, swa_status = choose_swa_payload(pca_paths.get((seq, int(curr_chunk or -1)), []))

    prev_pts = tensor_float(overlap.get("prev_overlap_local_points"))
    curr_pts = tensor_float(overlap.get("curr_overlap_local_points"))
    prev_global = tensor_float(overlap.get("prev_overlap_points"))
    curr_global = tensor_float(overlap.get("curr_overlap_points"))
    prev_conf = tensor_float(overlap.get("prev_conf"))
    curr_conf = tensor_float(overlap.get("curr_conf"))
    prev_sem = overlap.get("prev_semantic_labels")
    curr_sem = overlap.get("curr_semantic_labels")
    prev_sem_conf = tensor_float(overlap.get("prev_semantic_conf"))
    curr_sem_conf = tensor_float(overlap.get("curr_semantic_conf"))
    curr_frame_ids = overlap.get("curr_frame_ids")
    curr_coords = overlap.get("curr_pixel_coords")
    if any(value is None for value in [prev_pts, curr_pts, prev_conf, curr_conf, prev_sem, curr_sem, prev_sem_conf, curr_sem_conf, curr_frame_ids, curr_coords]):
        summary.update({"pair_available": False, "missing_reason": "missing_required_overlap_tensors"})
        return [], [], summary

    residual = torch.linalg.norm(prev_global.float() - curr_global.float(), dim=-1) if prev_global is not None and curr_global is not None else torch.linalg.norm(prev_pts - curr_pts, dim=-1)
    patch_flat, mapping_note = make_patch_indices(curr_coords, PATCH_GRID)
    patch_y = (patch_flat // PATCH_GRID[1]).long()
    patch_x = (patch_flat % PATCH_GRID[1]).long()
    curr_start = int(overlap.get("curr_start_frame", int(curr_frame_ids.min().item())))
    local_frame = torch.clamp(curr_frame_ids.long() - curr_start, 0, 31)

    res_scale = robust_scale([float(v) for v in residual.tolist()])
    parallax = torch.linalg.norm(prev_pts - curr_pts, dim=-1)
    par_scale = robust_scale([float(v) for v in parallax.tolist()])

    groups: dict[int, Any] = {}
    for pf in torch.unique(patch_flat).tolist():
        idx = torch.nonzero(patch_flat == int(pf), as_tuple=False).flatten()
        groups[int(pf)] = idx

    token_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    anchor_points: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    sorted_groups = sorted(groups.items(), key=lambda item: int(groups[item[0]].numel()), reverse=True)
    for pf, idx in sorted_groups[:max_token_rows]:
        py = int(pf // PATCH_GRID[1])
        px = int(pf % PATCH_GRID[1])
        count = int(idx.numel())
        labels = [int(v) for v in curr_sem[idx].detach().cpu().tolist()]
        label, purity = mode_and_purity(labels)
        same_region = float((prev_sem[idx].long() == curr_sem[idx].long()).float().mean().item())
        sem_conf = float(((prev_sem_conf[idx] + curr_sem_conf[idx]) * 0.5).mean().item())
        conf_mean = float(((prev_conf[idx] + curr_conf[idx]) * 0.5).mean().item())
        zero_ratio = float(((prev_conf[idx] <= 0) | (curr_conf[idx] <= 0)).float().mean().item())
        res_mean = float(residual[idx].mean().item())
        par_mean = float(parallax[idx].mean().item())
        pts_prev_patch = prev_pts[idx]
        pts_curr_patch = curr_pts[idx]
        spread_prev = float(pts_prev_patch.std(dim=0).norm().item()) if count > 1 else 0.0
        spread_curr = float(pts_curr_patch.std(dim=0).norm().item()) if count > 1 else 0.0
        spread = 0.5 * (spread_prev + spread_curr)
        spread_proxy = clamp01(spread / max(res_scale, 1e-6))
        parallax_proxy = clamp01(par_mean / max(par_scale, 1e-6))
        nondegenerate_proxy = clamp01((spread_proxy or 0.0) * 0.7 + (parallax_proxy or 0.0) * 0.3)
        far_context_proxy = clamp01((1.0 - (parallax_proxy or 0.0)) * 0.5 + (1.0 - (spread_proxy or 0.0)) * 0.5)
        geometry_leverage = clamp01((spread_proxy or 0.0) * (parallax_proxy or 0.0) * (nondegenerate_proxy or 0.0) * (1.0 - (far_context_proxy or 0.0)))
        residual_consistency = clamp01(1.0 / (1.0 + res_mean / max(res_scale, 1e-6)))
        overlap_consistency = clamp01(conf_mean * (residual_consistency or 0.0) * same_region * (1.0 - zero_ratio))

        read_values = [
            read_patch_value(
                read_payload,
                "read_patch_final",
                int(local_frame[i].item()),
                py,
                px,
                radius=usage_neighborhood_radius,
            )
            for i in idx[: min(count, 256)]
        ]
        read_usage = finite_mean([v for v in read_values if v is not None])
        qk_values: list[float] = []
        v_values: list[float] = []
        for i in idx[: min(count, 256)]:
            qk, vv = swa_qk_value(
                swa_payload,
                int(local_frame[i].item()),
                py,
                px,
                radius=usage_neighborhood_radius,
            )
            if qk is not None:
                qk_values.append(qk)
            if vv is not None:
                v_values.append(vv)
        qk_compat = finite_mean(qk_values)
        swa_usage = qk_compat
        v_protect = finite_mean(v_values)

        semantic_trust = clamp01(sem_conf * (purity if purity is not None else 0.0))
        cross_boundary = clamp01(1.0 - same_region)
        residual_risk = clamp01(res_mean / max(res_scale, 1e-6))
        # Far/low-parallax evidence is context or degeneracy, not necessarily
        # risk. Treat explicit risk as boundary, zero-confidence, or high
        # residual so that far context does not masquerade as dynamic/harmful.
        risk_score = clamp01(
            1.0
            - (1.0 - (cross_boundary or 0.0))
            * (1.0 - zero_ratio)
            * (1.0 - (residual_risk or 0.0))
        )
        memory_factor = math.sqrt(max(read_usage or 0.0, 0.0) * max(swa_usage or 0.0, 0.0)) if read_usage is not None and swa_usage is not None else None
        ruler_anchor_score = (
            clamp01((semantic_trust or 0.0) * (geometry_leverage or 0.0) * (overlap_consistency or 0.0) * (memory_factor or 0.0) * (1.0 - (risk_score or 0.0)))
            if memory_factor is not None
            else None
        )
        if zero_ratio > 0 or (risk_score or 0) >= 0.65:
            role = "RULER_RISK"
        elif (geometry_leverage or 0) < 0.05:
            role = "RULER_DEGENERATE"
        elif memory_factor is not None and (ruler_anchor_score or 0) > 0:
            role = "RULER_ANCHOR"
        else:
            role = "RULER_CONTEXT"

        srcs = [str(source_path)]
        if read_path:
            srcs.append(str(read_path))
        if swa_path:
            srcs.append(str(swa_path))
        token = {
            **row_base,
            "chunk_id": curr_chunk,
            "frame_id": int(round(float(curr_frame_ids[idx].float().mean().item()))),
            "patch_y": py,
            "patch_x": px,
            "overlap_sample_count": count,
            "semantic_label": label,
            "semantic_confidence": sem_conf,
            "patch_purity": purity,
            "track_available": False,
            "track_unavailable": True,
            "radio_available": str(pair_row.get("has_radio", "")).lower() == "true",
            "semantic_trust": semantic_trust,
            "geometry_spread": spread,
            "parallax_proxy": parallax_proxy,
            "nondegenerate_proxy": nondegenerate_proxy,
            "far_context_proxy": far_context_proxy,
            "geometry_leverage": geometry_leverage,
            "overlap_residual": res_mean,
            "confidence_weighted_residual": res_mean * conf_mean,
            "same_region_proxy": same_region,
            "cross_boundary_proxy": cross_boundary,
            "overlap_consistency": overlap_consistency,
            "READ_usage": read_usage,
            "SWA_usage": swa_usage,
            "SWA_usage_source": swa_status if swa_usage is not None else "",
            "QK_compatibility": qk_compat,
            "cache_V_compatibility": v_protect,
            "risk_score": risk_score,
            "risk_formula": "boundary_or_zero_conf_or_high_residual;far_context_not_risk",
            "ruler_anchor_score": ruler_anchor_score,
            "ruler_role": role,
            "source_artifact_paths": srcs,
            "patch_mapping_note": mapping_note,
            "usage_neighborhood_radius": usage_neighborhood_radius,
            "usage_aggregation_note": (
                "nearest_patch"
                if usage_neighborhood_radius <= 0
                else f"read_local_max_swa_qk_mean_radius{usage_neighborhood_radius}"
            ),
        }
        token_rows.append(token)
        pair_rows.append(
            {
                **row_base,
                "patch_y": py,
                "patch_x": px,
                "candidate_kind": "patch_overlap_candidate",
                "sample_count": count,
                "prev_centroid": [float(x) for x in pts_prev_patch.mean(dim=0).tolist()],
                "curr_centroid": [float(x) for x in pts_curr_patch.mean(dim=0).tolist()],
                "ruler_anchor_score": ruler_anchor_score,
                "ruler_role": role,
            }
        )
        if role == "RULER_ANCHOR":
            anchor_points.append((tuple(float(x) for x in pts_prev_patch.mean(dim=0).tolist()), tuple(float(x) for x in pts_curr_patch.mean(dim=0).tolist())))

    read_available = read_payload is not None
    swa_available = swa_payload is not None
    hq = str(pair_row.get("quality_type", "")).lower() == "high_quality"
    anchor_scores = [safe_float(row.get("ruler_anchor_score")) for row in token_rows]
    anchor_scores = [v for v in anchor_scores if v is not None]
    roles = Counter(row["ruler_role"] for row in token_rows)
    dist_stats = pairwise_distance_stats(anchor_points, pairwise_max_anchors)
    summary.update(
        {
            "pair_available": True,
            "token_rows": len(token_rows),
            "candidate_pair_rows": len(pair_rows),
            "semantic_projection_ratio": safe_float(pair_row.get("semantic_confidence_mean")) is not None or bool(overlap.get("semantic_label_projected_ratio")),
            "semantic_label_projected_ratio": overlap.get("semantic_label_projected_ratio", ""),
            "read_usage_available": read_available,
            "read_usage_source": read_status,
            "read_usage_path": str(read_path) if read_path else "",
            "swa_usage_available": swa_available,
            "swa_usage_source": swa_status,
            "swa_usage_path": str(swa_path) if swa_path else "",
            "geometry_leverage_available": bool(token_rows),
            "high_quality_row": hq,
            "zero_conf_ratio": safe_float(pair_row.get("either_zero_ratio")),
            "ruler_anchor_count": roles.get("RULER_ANCHOR", 0),
            "ruler_risk_count": roles.get("RULER_RISK", 0),
            "ruler_degenerate_count": roles.get("RULER_DEGENERATE", 0),
            "ruler_context_count": roles.get("RULER_CONTEXT", 0),
            "ruler_anchor_mass": sum(anchor_scores),
            "pairwise_log_distance_ratio_median": dist_stats["median"],
            "pairwise_log_distance_ratio_mad": dist_stats["mad"],
            "pairwise_distance_ratio_count": dist_stats["count"],
            "patch_mapping_note": mapping_note,
        }
    )
    return token_rows, pair_rows, summary


def build_missing_report(feature_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase1 Missing Artifact Report",
        "",
        "No missing value was backfilled. READ/SWA usage is counted only when current-chunk direct dumps were loadable.",
        "",
        "## Missing by Pair",
        "",
        "| seq | prev | curr | case_type | read | swa | geometry | note |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        note = row.get("missing_reason", "")
        if row.get("pair_available") and row.get("read_usage_available") and row.get("swa_usage_available") and row.get("geometry_leverage_available"):
            continue
        lines.append(
            f"| {row.get('seq')} | {row.get('prev_chunk')} | {row.get('curr_chunk')} | {row.get('case_type')} | "
            f"{row.get('read_usage_available')} | {row.get('swa_usage_available')} | {row.get('geometry_leverage_available')} | {note} |"
        )
    lines.extend(["", "## Feature Availability Summary", ""])
    for row in feature_rows:
        lines.append(f"- {row['feature_group']}: {row['available_rows']}/{row['total_rows']} ({row['available_ratio']})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    direct_roots = args.direct_root if args.direct_root is not None else DEFAULT_DIRECT_ROOTS
    pairs = read_csv(args.pair_bank)
    if args.torch_load_limit_pairs and args.torch_load_limit_pairs > 0:
        pairs = pairs[: args.torch_load_limit_pairs]
    read_paths, pca_paths = discover_direct_paths(direct_roots)

    token_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in pairs:
        tokens, candidate_pairs, summary = build_pair_candidates(
            row,
            read_paths,
            pca_paths,
            max_token_rows=args.max_token_rows_per_pair,
            pairwise_max_anchors=args.pairwise_max_anchors,
            usage_neighborhood_radius=max(0, int(args.usage_neighborhood_radius)),
        )
        token_rows.extend(tokens)
        pair_rows.extend(candidate_pairs)
        summaries.append(summary)

    total = len(summaries)
    high_quality = [row for row in summaries if row.get("high_quality_row")]
    feature_rows = [
        {
            "feature_group": "semantic_projection",
            "available_rows": sum(1 for row in summaries if row.get("semantic_label_projected_ratio") not in {"", None}),
            "total_rows": total,
        },
        {
            "feature_group": "READ_usage",
            "available_rows": sum(1 for row in summaries if row.get("read_usage_available")),
            "total_rows": total,
        },
        {
            "feature_group": "SWA_usage",
            "available_rows": sum(1 for row in summaries if row.get("swa_usage_available")),
            "total_rows": total,
        },
        {
            "feature_group": "geometry_leverage_high_quality",
            "available_rows": sum(1 for row in high_quality if row.get("geometry_leverage_available")),
            "total_rows": len(high_quality),
        },
    ]
    for row in feature_rows:
        denom = max(int(row["total_rows"]), 1)
        row["available_ratio"] = int(row["available_rows"]) / denom

    seqs = sorted({str(row.get("seq")) for row in summaries if row.get("seq")})
    bad_rows = [row for row in summaries if row.get("base_case_type") == "bad"]
    good_rows = [row for row in summaries if row.get("base_case_type") != "bad"]
    semantic_ratio_ok = all(
        (safe_float(row.get("semantic_label_projected_ratio")) or 0.0) >= 0.95
        for row in high_quality
        if row.get("semantic_label_projected_ratio") not in {"", None}
    )
    read_ratio = next(row["available_ratio"] for row in feature_rows if row["feature_group"] == "READ_usage")
    swa_ratio = next(row["available_ratio"] for row in feature_rows if row["feature_group"] == "SWA_usage")
    geo_hq_ratio = next(row["available_ratio"] for row in feature_rows if row["feature_group"] == "geometry_leverage_high_quality")
    phase1_gate_pass = (
        total >= 24
        and len(seqs) >= 3
        and len(summaries) >= 24
        and semantic_ratio_ok
        and read_ratio >= 0.70
        and swa_ratio >= 0.70
        and geo_hq_ratio >= 0.80
    )

    out_dir = args.out_dir
    write_csv(out_dir / "ruler_candidate_tokens.csv", token_rows)
    write_csv(out_dir / "ruler_candidate_pairs.csv", pair_rows)
    write_csv(out_dir / "ruler_candidate_pair_summary.csv", summaries)
    write_csv(out_dir / "feature_availability.csv", feature_rows)
    (out_dir / "missing_artifact_report.md").write_text(build_missing_report(feature_rows, summaries), encoding="utf-8")
    write_json(
        out_dir / "phase1_gate_summary.json",
        {
            "schema": "acl2_v84_phase1_gate_summary_v1",
            "phase1_gate_pass": phase1_gate_pass,
            "adjacent_pair_rows": total,
            "bad_rows": len(bad_rows),
            "good_or_false_positive_rows": len(good_rows),
            "sequence_coverage": seqs,
            "sequence_coverage_count": len(seqs),
            "semantic_projection_high_quality_ge_0_95": semantic_ratio_ok,
            "read_usage_available_rows": sum(1 for row in summaries if row.get("read_usage_available")),
            "read_usage_available_ratio": read_ratio,
            "swa_usage_available_rows": sum(1 for row in summaries if row.get("swa_usage_available")),
            "swa_usage_available_ratio": swa_ratio,
            "geometry_leverage_high_quality_ratio": geo_hq_ratio,
            "token_rows": len(token_rows),
            "candidate_pair_rows": len(pair_rows),
            "direct_roots": [str(path) for path in direct_roots],
            "usage_neighborhood_radius": max(0, int(args.usage_neighborhood_radius)),
            "notes": [
                "SWA_usage is QK compatibility proxy when PCA tensors exist; route_mass is not fabricated.",
                "track features unavailable: C_track treated as neutral and track_unavailable=true at token rows.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "phase1_gate_pass": phase1_gate_pass,
                "read_ratio": read_ratio,
                "swa_ratio": swa_ratio,
                "token_rows": len(token_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
