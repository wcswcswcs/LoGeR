#!/usr/bin/env python3
"""Audit v78 P9_20 route-bias feature maps against semantic group controls.

This is a diagnostic-only audit. It does not run the HMC method or promote any
candidate. It reads the saved P9_20/P9_21 SWA overlap feature dumps, aligns the
overlap tokens to KITTI semantic groups, and compares the actual high-route
tokens against random masks that preserve mass, source semantic group counts,
and source-to-query semantic transition counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


DEFAULT_ACTUAL_FEATURE = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/smoke_chunk06_context2_v10_attention_bias_beta070/"
    "chunk06/P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST/swa_overlap_feature_maps/"
    "chunk_006_swa_overlap_source_bias_geometric_layer_03.pt"
)
DEFAULT_RANDOM_FEATURE = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/smoke_chunk06_context2_v10_attention_bias_beta070/"
    "chunk06/P9_21_ATTENTION_BIAS_STABLE_AGREEMENT_RANDOM_SAME_MASS_LAST/swa_overlap_feature_maps/"
    "chunk_006_swa_overlap_source_bias_geometric_layer_03.pt"
)
DEFAULT_SEMANTIC_PT = Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt")
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/route_bias_group_audit_v1"
)

GROUP_NAMES = {
    0: "other_void",
    1: "road_ground",
    2: "sky",
    3: "vegetation_mountain",
    4: "dynamic",
    5: "static_built",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-feature", type=Path, default=DEFAULT_ACTUAL_FEATURE)
    parser.add_argument("--random-feature", type=Path, default=DEFAULT_RANDOM_FEATURE)
    parser.add_argument("--semantic-pt", type=Path, default=DEFAULT_SEMANTIC_PT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-frames", default="", help="Comma list. Default derives previous overlap frames.")
    parser.add_argument("--query-frames", default="", help="Comma list. Default derives current overlap frames.")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--grid-width", type=int, default=66)
    parser.add_argument("--grid-height", type=int, default=19)
    parser.add_argument("--top-quantiles", default="0.80,0.90")
    parser.add_argument("--seed", type=int, default=780920)
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return _jsonable(x.item())
        return [_jsonable(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k, "")) for k in fieldnames})


def _ids_containing(label_names: list[str], words: Iterable[str]) -> set[int]:
    lowered = [str(x).lower() for x in label_names]
    keys = [str(w).lower() for w in words]
    return {idx for idx, name in enumerate(lowered) if any(k in name for k in keys)}


def _load_semantic(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict) or not torch.is_tensor(sem.get("label_maps")):
        raise KeyError(f"No semantic_segmentation.label_maps in {path}")
    label_names = [str(x) for x in sem.get("label_names", [])]
    confidence = sem.get("confidence_maps")
    if not torch.is_tensor(confidence):
        confidence = torch.ones_like(sem["label_maps"], dtype=torch.float32)
    return {
        "label_maps": sem["label_maps"].detach().cpu().long(),
        "confidence_maps": confidence.detach().cpu().float(),
        "label_names": label_names,
        "road_ids": _ids_containing(label_names, ("road", "ground")),
        "sky_ids": _ids_containing(label_names, ("sky",)),
        "vegetation_ids": _ids_containing(label_names, ("grass", "tree", "vegetation", "mountain")),
        "dynamic_ids": _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle")),
        "static_ids": _ids_containing(
            label_names,
            ("wall", "fence", "pole", "building", "house", "bridge", "construction", "traffic sign", "billboard"),
        ),
    }


def _downsample_label(labels: torch.Tensor, frame: int, size: tuple[int, int]) -> torch.Tensor:
    img = Image.fromarray(labels[int(frame)].detach().cpu().numpy().astype(np.uint8), "L")
    return torch.from_numpy(np.asarray(img.resize(size, Image.Resampling.NEAREST)).copy()).long()


def _downsample_float(x: torch.Tensor, frame: int, size: tuple[int, int]) -> torch.Tensor:
    arr = x[int(frame)].detach().cpu().numpy().astype(np.float32)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    out = np.asarray(img.resize(size, Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    return torch.from_numpy(out.copy()).float()


def _mask_from_ids(labels: torch.Tensor, ids: set[int]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        out |= labels == int(idx)
    return out


def _groups(labels: torch.Tensor, sem: dict[str, Any]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.long)
    for gid, key in enumerate(("road_ids", "sky_ids", "vegetation_ids", "dynamic_ids", "static_ids"), start=1):
        out[_mask_from_ids(labels, sem[key])] = int(gid)
    return out


def _parse_int_list(text: str) -> list[int]:
    if not str(text).strip():
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _feature_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Feature payload missing tensor {key}")
    x = value.detach().cpu().float()
    if x.ndim == 3 and int(x.shape[0]) == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected {key} as [overlap,tokens], got {tuple(x.shape)}")
    return x


def _slice_patch_tokens(
    x: torch.Tensor,
    *,
    tokens_per_frame: int,
    patch_tokens: int,
) -> tuple[torch.Tensor, int]:
    patch_start = max(0, int(tokens_per_frame) - int(patch_tokens))
    patch_end = patch_start + int(patch_tokens)
    if patch_end > int(x.shape[-1]):
        raise ValueError(
            f"Cannot slice patch tokens: shape={tuple(x.shape)} tokens_per_frame={tokens_per_frame} "
            f"patch_tokens={patch_tokens} patch_start={patch_start}"
        )
    return x[:, patch_start:patch_end], patch_start


def _flatten_frames(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(values, dim=0).reshape(-1)


def _semantic_vectors(
    sem: dict[str, Any],
    frames: list[int],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels: list[torch.Tensor] = []
    confs: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    for frame in frames:
        label = _downsample_label(sem["label_maps"], frame, size)
        conf = _downsample_float(sem["confidence_maps"], frame, size)
        labels.append(label)
        confs.append(conf)
        groups.append(_groups(label, sem))
    return _flatten_frames(labels), _flatten_frames(confs), _flatten_frames(groups)


def _select_random_like(mask: torch.Tensor, seed: int) -> torch.Tensor:
    flat = mask.reshape(-1)
    count = int(flat.sum().item())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(int(flat.numel()), generator=gen)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[perm[:count]] = True
    return out.reshape_as(mask)


def _stratified_random_like(mask: torch.Tensor, strata: torch.Tensor, seed: int) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    flat_strata = strata.reshape(-1)
    out = torch.zeros_like(flat_mask, dtype=torch.bool)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    for sid in torch.unique(flat_strata).tolist():
        idx = torch.nonzero(flat_strata == int(sid), as_tuple=False).reshape(-1)
        count = int(flat_mask[idx].sum().item())
        if count <= 0 or int(idx.numel()) == 0:
            continue
        perm = idx[torch.randperm(int(idx.numel()), generator=gen)[: min(count, int(idx.numel()))]]
        out[perm] = True
    return out.reshape_as(mask)


def _ranked_within_strata(scores: torch.Tensor, mask: torch.Tensor, strata: torch.Tensor, *, largest: bool) -> torch.Tensor:
    flat_scores = scores.reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_strata = strata.reshape(-1)
    out = torch.zeros_like(flat_mask, dtype=torch.bool)
    for sid in torch.unique(flat_strata).tolist():
        idx = torch.nonzero(flat_strata == int(sid), as_tuple=False).reshape(-1)
        count = int(flat_mask[idx].sum().item())
        if count <= 0:
            continue
        order = torch.argsort(flat_scores[idx], descending=bool(largest))
        out[idx[order[:count]]] = True
    return out.reshape_as(mask)


def _top_quantile_mask(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.reshape(-1)
    threshold = torch.quantile(flat, float(q))
    mask = flat >= threshold
    if int(mask.sum().item()) == 0:
        mask[int(torch.argmax(flat).item())] = True
    return mask.reshape_as(x)


def _safe_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")
    return float(x.reshape(-1)[mask.reshape(-1)].mean().item())


def _group_distribution(mask: torch.Tensor, groups: torch.Tensor) -> dict[int, float]:
    out: dict[int, float] = {}
    flat_m = mask.reshape(-1)
    flat_g = groups.reshape(-1)
    denom = int(flat_m.sum().item())
    for gid in sorted(GROUP_NAMES):
        if denom <= 0:
            out[gid] = 0.0
        else:
            out[gid] = float(((flat_m & (flat_g == gid)).sum().item()) / denom)
    return out


def _entropy(dist: dict[int, float]) -> float:
    total = 0.0
    for value in dist.values():
        if value > 0:
            total -= value * math.log(value)
    return float(total)


def _l1(a: dict[int, float], b: dict[int, float]) -> float:
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in sorted(GROUP_NAMES)))


def _dominant(dist: dict[int, float]) -> tuple[str, float]:
    if not dist:
        return "none", 0.0
    gid = max(dist, key=lambda k: float(dist[k]))
    return GROUP_NAMES.get(gid, str(gid)), float(dist[gid])


def _mask_metrics(
    *,
    name: str,
    quantile: float,
    mask: torch.Tensor,
    actual_score: torch.Tensor,
    random_score: torch.Tensor,
    actual_control: torch.Tensor,
    random_control: torch.Tensor,
    dq: torch.Tensor,
    ds: torch.Tensor,
    max_d: torch.Tensor,
    source_conf: torch.Tensor,
    query_conf: torch.Tensor,
    source_groups: torch.Tensor,
    query_groups: torch.Tensor,
    all_source_dist: dict[int, float],
) -> dict[str, Any]:
    source_dist = _group_distribution(mask, source_groups)
    query_dist = _group_distribution(mask, query_groups)
    source_dom, source_dom_frac = _dominant(source_dist)
    query_dom, query_dom_frac = _dominant(query_dist)
    flat_mask = mask.reshape(-1)
    source_group_match = (source_groups.reshape(-1) == query_groups.reshape(-1))
    count = int(flat_mask.sum().item())
    total = int(flat_mask.numel())
    return {
        "quantile": float(quantile),
        "selector": name,
        "token_count": count,
        "token_fraction": float(count / max(1, total)),
        "actual_score_mean": _safe_mean(actual_score, mask),
        "random_score_mean": _safe_mean(random_score, mask),
        "actual_control_mean": _safe_mean(actual_control, mask),
        "random_control_mean": _safe_mean(random_control, mask),
        "Dq_mean": _safe_mean(dq, mask),
        "Ds_mean": _safe_mean(ds, mask),
        "maxD_mean": _safe_mean(max_d, mask),
        "source_confidence_mean": _safe_mean(source_conf, mask),
        "query_confidence_mean": _safe_mean(query_conf, mask),
        "source_query_group_match_fraction": _safe_mean(source_group_match.float(), mask),
        "source_group_entropy": _entropy(source_dist),
        "query_group_entropy": _entropy(query_dist),
        "source_group_l1_vs_all": _l1(source_dist, all_source_dist),
        "source_dominant_group": source_dom,
        "source_dominant_group_fraction": source_dom_frac,
        "query_dominant_group": query_dom,
        "query_dominant_group_fraction": query_dom_frac,
        "source_road_static_fraction": float(source_dist.get(1, 0.0) + source_dist.get(5, 0.0)),
        "source_dynamic_fraction": float(source_dist.get(4, 0.0)),
        "query_road_static_fraction": float(query_dist.get(1, 0.0) + query_dist.get(5, 0.0)),
        "query_dynamic_fraction": float(query_dist.get(4, 0.0)),
    }


def _by_group_rows(
    *,
    quantile: float,
    selector: str,
    mask: torch.Tensor,
    actual_score: torch.Tensor,
    random_score: torch.Tensor,
    dq: torch.Tensor,
    ds: torch.Tensor,
    max_d: torch.Tensor,
    source_conf: torch.Tensor,
    query_conf: torch.Tensor,
    source_groups: torch.Tensor,
    query_groups: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_selected = int(mask.sum().item())
    total_tokens = int(mask.numel())
    for gid, group_name in GROUP_NAMES.items():
        group_mask = source_groups == int(gid)
        selected = mask & group_mask
        group_count = int(group_mask.sum().item())
        selected_count = int(selected.sum().item())
        rows.append(
            {
                "quantile": float(quantile),
                "selector": selector,
                "source_group_id": int(gid),
                "source_group": group_name,
                "group_token_count": group_count,
                "group_token_fraction": float(group_count / max(1, total_tokens)),
                "selected_count": selected_count,
                "selected_fraction_of_selector": float(selected_count / max(1, total_selected)),
                "selected_fraction_within_group": float(selected_count / max(1, group_count)),
                "actual_score_mean": _safe_mean(actual_score, selected),
                "random_score_mean": _safe_mean(random_score, selected),
                "Dq_mean": _safe_mean(dq, selected),
                "Ds_mean": _safe_mean(ds, selected),
                "maxD_mean": _safe_mean(max_d, selected),
                "source_confidence_mean": _safe_mean(source_conf, selected),
                "query_confidence_mean": _safe_mean(query_conf, selected),
                "query_group_match_fraction": _safe_mean((query_groups == source_groups).float(), selected),
            }
        )
    return rows


def _transition_rows(
    *,
    quantile: float,
    selector: str,
    mask: torch.Tensor,
    source_groups: torch.Tensor,
    query_groups: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = int(mask.sum().item())
    for sgid, sname in GROUP_NAMES.items():
        for qgid, qname in GROUP_NAMES.items():
            pair = mask & (source_groups == int(sgid)) & (query_groups == int(qgid))
            count = int(pair.sum().item())
            if count <= 0:
                continue
            rows.append(
                {
                    "quantile": float(quantile),
                    "selector": selector,
                    "source_group": sname,
                    "query_group": qname,
                    "count": count,
                    "fraction_of_selector": float(count / max(1, total)),
                }
            )
    return rows


def _derive_frames(args: argparse.Namespace, actual: dict[str, Any]) -> tuple[list[int], list[int], dict[str, Any]]:
    overlap = int(actual.get("overlap_frames_effective") or args.chunk_overlap)
    if overlap != int(args.chunk_overlap):
        inferred_overlap = overlap
    else:
        inferred_overlap = int(args.chunk_overlap)
    chunk_idx = int(actual.get("chunk_idx"))
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    query_start = chunk_idx * stride
    source_frames = _parse_int_list(args.source_frames)
    query_frames = _parse_int_list(args.query_frames)
    if not source_frames:
        source_frames = list(range(query_start - inferred_overlap, query_start))
    if not query_frames:
        query_frames = list(range(query_start, query_start + inferred_overlap))
    if len(source_frames) != inferred_overlap or len(query_frames) != inferred_overlap:
        raise ValueError(
            f"Expected {inferred_overlap} source/query frames, got source={source_frames} query={query_frames}"
        )
    meta = {
        "chunk_idx": chunk_idx,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap_arg": int(args.chunk_overlap),
        "feature_overlap_frames_effective": inferred_overlap,
        "chunk_stride": stride,
        "derived_query_start_frame": query_start,
    }
    return source_frames, query_frames, meta


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual = _torch_load(args.actual_feature)
    random = _torch_load(args.random_feature)
    if not isinstance(actual, dict) or not isinstance(random, dict):
        raise TypeError("Feature files must contain dict payloads")

    actual_score_2d = _feature_tensor(actual, "score_overlap")
    random_score_2d = _feature_tensor(random, "score_overlap")
    actual_control_2d = _feature_tensor(actual, "control_overlap")
    random_control_2d = _feature_tensor(random, "control_overlap")
    dq_2d = _feature_tensor(actual, "Dq_overlap")
    ds_2d = _feature_tensor(actual, "Ds_overlap")

    grid_tokens = int(args.grid_width) * int(args.grid_height)
    if tuple(actual_score_2d.shape) != tuple(random_score_2d.shape):
        raise ValueError(f"actual/random score shape mismatch: {actual_score_2d.shape} vs {random_score_2d.shape}")
    tokens_per_frame = int(actual.get("tokens_per_frame", int(actual_score_2d.shape[-1])) or int(actual_score_2d.shape[-1]))
    actual_score_2d, patch_start = _slice_patch_tokens(
        actual_score_2d,
        tokens_per_frame=tokens_per_frame,
        patch_tokens=grid_tokens,
    )
    random_score_2d, random_patch_start = _slice_patch_tokens(
        random_score_2d,
        tokens_per_frame=tokens_per_frame,
        patch_tokens=grid_tokens,
    )
    actual_control_2d, _ = _slice_patch_tokens(
        actual_control_2d,
        tokens_per_frame=tokens_per_frame,
        patch_tokens=grid_tokens,
    )
    random_control_2d, _ = _slice_patch_tokens(
        random_control_2d,
        tokens_per_frame=tokens_per_frame,
        patch_tokens=grid_tokens,
    )
    dq_2d, _ = _slice_patch_tokens(dq_2d, tokens_per_frame=tokens_per_frame, patch_tokens=grid_tokens)
    ds_2d, _ = _slice_patch_tokens(ds_2d, tokens_per_frame=tokens_per_frame, patch_tokens=grid_tokens)
    if patch_start != random_patch_start:
        raise ValueError(f"actual/random patch_start mismatch: {patch_start} vs {random_patch_start}")

    source_frames, query_frames, frame_meta = _derive_frames(args, actual)
    if int(actual_score_2d.shape[0]) != len(source_frames):
        raise ValueError(f"Feature overlap axis {actual_score_2d.shape[0]} does not match frames {source_frames}")

    sem = _load_semantic(args.semantic_pt)
    size = (int(args.grid_width), int(args.grid_height))
    source_labels, source_conf, source_groups = _semantic_vectors(sem, source_frames, size)
    query_labels, query_conf, query_groups = _semantic_vectors(sem, query_frames, size)

    actual_score = actual_score_2d.reshape(-1)
    random_score = random_score_2d.reshape(-1)
    actual_control = actual_control_2d.reshape(-1)
    random_control = random_control_2d.reshape(-1)
    dq = dq_2d.reshape(-1)
    ds = ds_2d.reshape(-1)
    max_d = torch.maximum(dq, ds)
    stable_score_from_d = 1.0 - max_d
    pair_strata = source_groups * 10 + query_groups
    all_mask = torch.ones_like(actual_score, dtype=torch.bool)
    all_source_dist = _group_distribution(all_mask, source_groups)

    quantiles = _parse_float_list(args.top_quantiles)
    selector_rows: list[dict[str, Any]] = []
    by_group_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    quantile_summary: dict[str, Any] = {}

    all_metrics = _mask_metrics(
        name="all_tokens",
        quantile=-1.0,
        mask=all_mask,
        actual_score=actual_score,
        random_score=random_score,
        actual_control=actual_control,
        random_control=random_control,
        dq=dq,
        ds=ds,
        max_d=max_d,
        source_conf=source_conf,
        query_conf=query_conf,
        source_groups=source_groups,
        query_groups=query_groups,
        all_source_dist=all_source_dist,
    )

    for q in quantiles:
        actual_top = _top_quantile_mask(actual_score, q)
        random_feature_top = _top_quantile_mask(random_score, q)
        same_count_random = _select_random_like(actual_top, args.seed + int(round(q * 1000)))
        source_group_random = _stratified_random_like(actual_top, source_groups, args.seed + 11 + int(round(q * 1000)))
        pair_group_random = _stratified_random_like(actual_top, pair_strata, args.seed + 23 + int(round(q * 1000)))
        low_d_source_group = _ranked_within_strata(stable_score_from_d, actual_top, source_groups, largest=True)
        high_d_source_group = _ranked_within_strata(stable_score_from_d, actual_top, source_groups, largest=False)
        low_d_pair_group = _ranked_within_strata(stable_score_from_d, actual_top, pair_strata, largest=True)

        masks = {
            "actual_top_score": actual_top,
            "p9_21_same_mass_feature_top": random_feature_top,
            "deterministic_same_count_random": same_count_random,
            "source_group_stratified_random": source_group_random,
            "source_query_pair_stratified_random": pair_group_random,
            "lowD_within_source_group": low_d_source_group,
            "highD_within_source_group": high_d_source_group,
            "lowD_within_source_query_pair": low_d_pair_group,
        }

        metrics_by_selector: dict[str, dict[str, Any]] = {}
        for selector, mask in masks.items():
            metrics = _mask_metrics(
                name=selector,
                quantile=q,
                mask=mask,
                actual_score=actual_score,
                random_score=random_score,
                actual_control=actual_control,
                random_control=random_control,
                dq=dq,
                ds=ds,
                max_d=max_d,
                source_conf=source_conf,
                query_conf=query_conf,
                source_groups=source_groups,
                query_groups=query_groups,
                all_source_dist=all_source_dist,
            )
            selector_rows.append(metrics)
            metrics_by_selector[selector] = metrics
            by_group_rows.extend(
                _by_group_rows(
                    quantile=q,
                    selector=selector,
                    mask=mask,
                    actual_score=actual_score,
                    random_score=random_score,
                    dq=dq,
                    ds=ds,
                    max_d=max_d,
                    source_conf=source_conf,
                    query_conf=query_conf,
                    source_groups=source_groups,
                    query_groups=query_groups,
                )
            )
            transition_rows.extend(
                _transition_rows(
                    quantile=q,
                    selector=selector,
                    mask=mask,
                    source_groups=source_groups,
                    query_groups=query_groups,
                )
            )

        actual_m = metrics_by_selector["actual_top_score"]
        group_m = metrics_by_selector["source_group_stratified_random"]
        pair_m = metrics_by_selector["source_query_pair_stratified_random"]
        same_feature_m = metrics_by_selector["p9_21_same_mass_feature_top"]
        low_d_pair_m = metrics_by_selector["lowD_within_source_query_pair"]
        quantile_summary[str(q)] = {
            "actual_minus_p9_21_same_mass_feature_actual_score_mean": float(
                actual_m["actual_score_mean"] - same_feature_m["actual_score_mean"]
            ),
            "actual_minus_source_group_random_actual_score_mean": float(
                actual_m["actual_score_mean"] - group_m["actual_score_mean"]
            ),
            "actual_minus_source_query_pair_random_actual_score_mean": float(
                actual_m["actual_score_mean"] - pair_m["actual_score_mean"]
            ),
            "actual_minus_lowD_pair_actual_score_mean": float(
                actual_m["actual_score_mean"] - low_d_pair_m["actual_score_mean"]
            ),
            "actual_source_group_l1_vs_all": actual_m["source_group_l1_vs_all"],
            "actual_source_road_static_fraction": actual_m["source_road_static_fraction"],
            "actual_source_dynamic_fraction": actual_m["source_dynamic_fraction"],
            "actual_source_query_group_match_fraction": actual_m["source_query_group_match_fraction"],
            "same_feature_source_group_l1_vs_all": same_feature_m["source_group_l1_vs_all"],
            "same_feature_source_road_static_fraction": same_feature_m["source_road_static_fraction"],
            "source_group_random_score_mean": group_m["actual_score_mean"],
            "source_query_pair_random_score_mean": pair_m["actual_score_mean"],
            "lowD_pair_score_mean": low_d_pair_m["actual_score_mean"],
        }

    score_consistency = {
        "actual_score_minus_1_minus_maxD_mean_abs": float((actual_score - stable_score_from_d).abs().mean().item()),
        "actual_score_mean": float(actual_score.mean().item()),
        "actual_score_q90": float(torch.quantile(actual_score, 0.90).item()),
        "random_score_mean": float(random_score.mean().item()),
        "random_score_q90": float(torch.quantile(random_score, 0.90).item()),
    }

    assessment_notes: list[str] = [
        "diagnostic_only_no_phase9_promotion",
        "P9_20 already failed phase9_gate_pass in phase9_swa_cache_value_decision.json",
    ]
    q90 = quantile_summary.get("0.9") or quantile_summary.get("0.90")
    if isinstance(q90, dict):
        if float(q90["actual_source_group_l1_vs_all"]) < 0.12:
            assessment_notes.append("actual_top_q90_source_group_distribution_close_to_all_tokens")
        else:
            assessment_notes.append("actual_top_q90_has_source_group_shift")
        if abs(float(q90["actual_minus_lowD_pair_actual_score_mean"])) < 1e-4:
            assessment_notes.append("actual_top_q90_is_equivalent_to_lowD_within_source_query_pair_control")
        if float(q90["actual_minus_source_query_pair_random_actual_score_mean"]) > 0.05:
            assessment_notes.append("stable_score_signal_survives_source_query_pair_stratified_random")

    summary = {
        "schema": "acl2_v78_p9_route_bias_group_alignment_audit_v1",
        "actual_feature": str(args.actual_feature),
        "random_feature": str(args.random_feature),
        "semantic_pt": str(args.semantic_pt),
        "output_dir": str(args.output_dir),
        "feature_metadata": {
            "actual_schema": actual.get("schema"),
            "actual_kind": actual.get("kind"),
            "actual_mode": actual.get("mode"),
            "random_mode": random.get("mode"),
            "swa_layer_idx": actual.get("swa_layer_idx"),
            "tokens_per_frame": actual.get("tokens_per_frame"),
            "patch_start": patch_start,
            "patch_tokens": grid_tokens,
            "patch_grid": [int(args.grid_height), int(args.grid_width)],
            "source_tokens": actual.get("source_tokens"),
            "runtime_swa_overlap_feature_not_qk_proxy": actual.get("runtime_swa_overlap_feature_not_qk_proxy"),
        },
        "frame_alignment": {
            **frame_meta,
            "source_frames": source_frames,
            "query_frames": query_frames,
            "alignment_note": (
                "Default source frames use the previous tail overlap and query frames use the current head overlap: "
                "source=[query_start-overlap, query_start), query=[query_start, query_start+overlap)."
            ),
        },
        "semantic_groups": {
            "label_names": sem["label_names"],
            "group_names": GROUP_NAMES,
            "road_ids": sorted(int(x) for x in sem["road_ids"]),
            "sky_ids": sorted(int(x) for x in sem["sky_ids"]),
            "vegetation_ids": sorted(int(x) for x in sem["vegetation_ids"]),
            "dynamic_ids": sorted(int(x) for x in sem["dynamic_ids"]),
            "static_ids": sorted(int(x) for x in sem["static_ids"]),
        },
        "all_token_metrics": all_metrics,
        "score_consistency": score_consistency,
        "quantile_summary": quantile_summary,
        "assessment_notes": assessment_notes,
        "decision": {
            "method_goal_achieved": False,
            "phase9_promotion_allowed_by_this_audit": False,
            "reason": (
                "This audit can only attribute the saved P9_20 route map. The previously measured "
                "Phase9 gate remains failed unless a new runtime candidate is run and passes the gate."
            ),
        },
    }

    _write_csv(args.output_dir / "route_bias_selected_mask_summary.csv", selector_rows)
    _write_csv(args.output_dir / "route_bias_by_source_group.csv", by_group_rows)
    _write_csv(args.output_dir / "route_bias_source_query_transitions.csv", transition_rows)
    _write_json(args.output_dir / "route_bias_group_alignment_summary.json", summary)

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
