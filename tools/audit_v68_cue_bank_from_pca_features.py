#!/usr/bin/env python3
"""Audit ACL2 v68 cue-bank signals from layer PCA feature dumps.

The geometry carrier implemented here is the plan's Gram-row temporal
instability over selected feature layers. It is an artifact audit only: no
trajectory action is applied by this script.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.semantic_prior_generator import (  # noqa: E402
    _mode_pool_dense_semantic_patches,
    _normalize_dense_semantic_confidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--semantic-pt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=6803)
    return parser.parse_args()


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _parse_selected_layers(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_layers") or []
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in selected:
        tap = str(row.get("tap") or "")
        layer = int(row.get("layer"))
        key = (tap, layer)
        if not tap or key in seen:
            continue
        seen.add(key)
        out.append({"tap": tap, "layer": layer, "source": "selected_layers"})
    return out


def _load_semantic(path: Path) -> Dict[str, Any]:
    payload = _torch_load(path)
    sem = payload.get("semantic_segmentation")
    if not isinstance(sem, dict):
        raise KeyError(f"No semantic_segmentation dict in {path}")
    labels = sem.get("label_maps")
    if not torch.is_tensor(labels):
        raise KeyError(f"No semantic_segmentation.label_maps tensor in {path}")
    label_names = [str(x) for x in (sem.get("label_names") or [])]
    return {
        "label_maps": labels.detach().cpu().long(),
        "confidence_maps": sem.get("confidence_maps"),
        "label_names": label_names,
        "dynamic_ids": _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle")),
        "sky_ids": _ids_containing(label_names, ("sky",)),
        "road_ids": _ids_containing(label_names, ("road", "ground")),
        "lowstuff_ids": _ids_containing(label_names, ("grass", "tree", "vegetation", "mountain")),
        "vertical_ids": _ids_containing(
            label_names,
            ("wall", "fence", "pole", "building", "house", "bridge", "construction", "traffic sign", "billboard"),
        ),
    }


def _ids_containing(label_names: Sequence[str], words: Iterable[str]) -> List[int]:
    lowered = [str(x).lower() for x in label_names]
    keys = [str(w).lower() for w in words]
    return [idx for idx, name in enumerate(lowered) if any(k in name for k in keys)]


def _mask_from_ids(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        mask |= labels == int(idx)
    return mask


def _patch_semantic(
    semantic: Mapping[str, Any],
    *,
    start: int,
    end: int,
    patch_grid: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    labels = semantic["label_maps"][int(start) : int(end)]
    conf_raw = semantic.get("confidence_maps")
    conf = None
    if torch.is_tensor(conf_raw):
        conf = conf_raw.detach().cpu()[int(start) : int(end)]
    conf_norm, _ = _normalize_dense_semantic_confidence(conf, target_shape=tuple(labels.shape))
    if conf_norm is None:
        conf_norm = torch.ones_like(labels, dtype=torch.float32)
    patch_label, purity, patch_conf = _mode_pool_dense_semantic_patches(
        labels.long(),
        conf_norm,
        patch_grid=patch_grid,
    )
    trust = (patch_conf * purity.square()).clamp(0.0, 1.0)
    vertical = _mask_from_ids(patch_label, semantic.get("vertical_ids", []))
    road = _mask_from_ids(patch_label, semantic.get("road_ids", []))
    dynamic = _mask_from_ids(patch_label, semantic.get("dynamic_ids", []))
    sky = _mask_from_ids(patch_label, semantic.get("sky_ids", []))
    lowstuff = _mask_from_ids(patch_label, semantic.get("lowstuff_ids", []))
    support = trust * (vertical.float() + 0.25 * road.float()).clamp(0.0, 1.0)
    risk_label = torch.maximum(dynamic.float(), torch.maximum(sky.float() * 0.9, lowstuff.float() * 0.7))
    risk = (trust * risk_label + (1.0 - trust)).clamp(0.0, 1.0)
    lowobs = (
        0.5 * sky.float()
        + 0.4 * road.float()
        + 0.4 * lowstuff.float()
        - 0.8 * vertical.float()
    ).clamp(0.0, 1.0) * trust
    return {
        "label": patch_label.long(),
        "purity": purity.float(),
        "confidence": patch_conf.float(),
        "trust": trust.float(),
        "support": support.float(),
        "risk": risk.float(),
        "lowobs": lowobs.float(),
    }


def _feature_for_layer(payload: Mapping[str, Any], tap: str, layer: int) -> Optional[torch.Tensor]:
    key = f"tap::{tap}"
    tensor = payload.get(key)
    if not torch.is_tensor(tensor):
        return None
    meta = dict(dict(payload.get("taps") or {}).get(tap) or {})
    selected_layers = [int(x) for x in (meta.get("selected_layers") or [])]
    if selected_layers:
        if int(layer) not in selected_layers:
            return None
        pos = selected_layers.index(int(layer))
    else:
        pos = int(layer)
    if tensor.ndim != 5 or pos < 0 or pos >= int(tensor.shape[1]):
        return None
    return tensor[:, pos].detach().cpu().float()


def _robust01(x: torch.Tensor) -> torch.Tensor:
    vals = x.detach().cpu().float()
    finite = torch.isfinite(vals)
    if not bool(finite.any().item()):
        return torch.zeros_like(vals)
    good = vals[finite]
    lo = torch.quantile(good, 0.05)
    hi = torch.quantile(good, 0.95)
    if float((hi - lo).abs().item()) < 1e-8:
        lo = good.min()
        hi = good.max()
    return ((vals - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _gram_row_temporal_instability(features: torch.Tensor) -> torch.Tensor:
    """Return [T,H,W] score from Gram-row variance over time.

    The same token score is repeated over T so it can be combined with
    per-frame semantic maps without inventing a separate temporal model.
    """

    T, H, W, D = [int(x) for x in features.shape]
    if T < 2:
        return torch.zeros((T, H, W), dtype=torch.float32)
    flat = torch.nn.functional.normalize(features.reshape(T, H * W, D).float(), dim=-1, eps=1e-6)
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    count = 0
    for t in range(T):
        gram = flat[t] @ flat[t].T
        count += 1
        if mean is None:
            mean = gram.clone()
            m2 = torch.zeros_like(gram)
            continue
        delta = gram - mean
        mean = mean + delta / float(count)
        assert m2 is not None
        m2 = m2 + delta * (gram - mean)
    assert m2 is not None
    var_rows = (m2 / float(max(count - 1, 1))).mean(dim=1)
    score = _robust01(var_rows.reshape(H, W))
    return score[None].repeat(T, 1, 1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    x = a.reshape(-1).float()
    y = b.reshape(-1).float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) < 3:
        return None
    x = x[finite]
    y = y[finite]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) <= 1e-8:
        return None
    return float((x @ y / denom).item())


def _cue_row(
    *,
    chunk_idx: int,
    start: int,
    end: int,
    cue_name: str,
    cue: torch.Tensor,
    sem: Mapping[str, torch.Tensor],
    top_frac: float,
    seed: int,
) -> Dict[str, Any]:
    flat = cue.reshape(-1).float()
    finite = torch.isfinite(flat)
    vals = flat[finite]
    available = bool(vals.numel() > 0)
    if not available:
        return {
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start),
            "end_frame": int(end),
            "cue_name": cue_name,
            "available": False,
        }
    n = int(vals.numel())
    k = max(1, int(round(float(top_frac) * n)))
    top_vals, top_idx = torch.topk(vals, k=min(k, n), largest=True)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(chunk_idx) * 1009 + sum(ord(c) for c in cue_name))
    rand_idx = torch.randperm(n, generator=gen)[: min(k, n)]
    random_vals = vals[rand_idx]
    row = {
        "chunk_idx": int(chunk_idx),
        "start_frame": int(start),
        "end_frame": int(end),
        "cue_name": cue_name,
        "available": True,
        "mean": float(vals.mean().item()),
        "q90": float(torch.quantile(vals, 0.90).item()),
        "gt050_mass": float((vals > 0.5).float().mean().item()),
        "selected_token_count": int(top_vals.numel()),
        "selected_score_mean": float(top_vals.mean().item()),
        "random_score_mean": float(random_vals.mean().item()),
        "selected_minus_random": float(top_vals.mean().item() - random_vals.mean().item()),
        "corr_sem_trust": _pearson(cue, sem["trust"]),
        "corr_sem_support": _pearson(cue, sem["support"]),
        "corr_sem_risk": _pearson(cue, sem["risk"]),
        "corr_sem_lowobs": _pearson(cue, sem["lowobs"]),
    }
    for name in ("trust", "support", "risk", "lowobs"):
        sem_flat = sem[name].reshape(-1).float()[finite]
        row[f"selected_{name}_mean"] = float(sem_flat[top_idx].mean().item())
        row[f"random_{name}_mean"] = float(sem_flat[rand_idx].mean().item())
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    fields = list(fieldnames) if fieldnames is not None else [
        "chunk_idx",
        "start_frame",
        "end_frame",
        "cue_name",
        "available",
        "mean",
        "q90",
        "gt050_mass",
        "selected_token_count",
        "selected_score_mean",
        "random_score_mean",
        "selected_minus_random",
        "corr_sem_trust",
        "corr_sem_support",
        "corr_sem_risk",
        "corr_sem_lowobs",
        "selected_trust_mean",
        "random_trust_mean",
        "selected_support_mean",
        "random_support_mean",
        "selected_risk_mean",
        "random_risk_mean",
        "selected_lowobs_mean",
        "random_lowobs_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    names = sorted({str(r.get("cue_name")) for r in rows})
    for name in names:
        subset = [r for r in rows if str(r.get("cue_name")) == name and str(r.get("available")) in {"True", "true", "1", "True"}]
        if not subset:
            out[name] = {"available_chunks": 0}
            continue
        deltas = [_safe_float(r.get("selected_minus_random")) for r in subset]
        deltas = [x for x in deltas if x is not None]
        means = [_safe_float(r.get("mean")) for r in subset]
        means = [x for x in means if x is not None]
        out[name] = {
            "available_chunks": int(len(subset)),
            "mean_of_mean": float(np.mean(means)) if means else None,
            "median_selected_minus_random": float(np.median(deltas)) if deltas else None,
            "positive_random_delta_chunks": int(sum(1 for x in deltas if x > 0.0)),
        }
    return out


def main() -> None:
    args = parse_args()
    selected = _parse_selected_layers(args.selection_json)
    if not selected:
        raise SystemExit(f"No selected_layers in {args.selection_json}")
    semantic = _load_semantic(args.semantic_pt)
    feature_paths = sorted(args.feature_dir.glob("chunk_*.pt"))
    if not feature_paths:
        raise FileNotFoundError(f"No chunk_*.pt files under {args.feature_dir}")
    rows: List[Dict[str, Any]] = []
    layer_debug: List[Dict[str, Any]] = []
    for path in feature_paths:
        payload = _torch_load(path)
        chunk_idx = int(payload["chunk_idx"])
        start = int(payload["start_frame"])
        end = int(payload["end_frame"])
        patch_grid = tuple(int(x) for x in payload["patch_grid"])
        sem = _patch_semantic(semantic, start=start, end=end, patch_grid=(int(patch_grid[0]), int(patch_grid[1])))
        motion_parts: List[torch.Tensor] = []
        for item in selected:
            feat = _feature_for_layer(payload, str(item["tap"]), int(item["layer"]))
            available = feat is not None
            layer_row = {
                "chunk_idx": int(chunk_idx),
                "tap": str(item["tap"]),
                "layer": int(item["layer"]),
                "available": bool(available),
            }
            if feat is not None:
                score = _gram_row_temporal_instability(feat)
                motion_parts.append(score)
                layer_row.update(
                    {
                        "motion_mean": float(score.mean().item()),
                        "motion_q90": float(torch.quantile(score.reshape(-1), 0.90).item()),
                    }
                )
                rows.append(
                    _cue_row(
                        chunk_idx=chunk_idx,
                        start=start,
                        end=end,
                        cue_name=f"G3_grammotion::{item['tap']}::L{int(item['layer'])}",
                        cue=score,
                        sem=sem,
                        top_frac=float(args.top_frac),
                        seed=int(args.random_seed),
                    )
                )
            layer_debug.append(layer_row)
        if motion_parts:
            motion = torch.stack(motion_parts, dim=0).mean(dim=0)
            cue_maps = {
                "G3_selected_grammotion": motion,
                "S_support": sem["support"],
                "S_risk": sem["risk"],
                "S_lowobs": sem["lowobs"],
                "C_read_motion_semrisk": _robust01(motion * sem["risk"]),
                "C_merge_static_motion_inverse": _robust01((1.0 - motion) * sem["support"]),
                "C_ttt_motion_risk_proxy": _robust01(motion * sem["risk"] + (1.0 - sem["support"])),
            }
            for cue_name, cue in cue_maps.items():
                rows.append(
                    _cue_row(
                        chunk_idx=chunk_idx,
                        start=start,
                        end=end,
                        cue_name=cue_name,
                        cue=cue,
                        sem=sem,
                        top_frac=float(args.top_frac),
                        seed=int(args.random_seed),
                    )
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "cue_bank_audit.csv", rows)
    _write_csv(
        args.out_dir / "selected_layer_motion_debug.csv",
        layer_debug,
        ["chunk_idx", "tap", "layer", "available", "motion_mean", "motion_q90"],
    )
    summary = {
        "schema": "acl2_v68_cue_bank_audit_v1",
        "feature_dir": str(args.feature_dir),
        "selection_json": str(args.selection_json),
        "semantic_pt": str(args.semantic_pt),
        "selected_layers": selected,
        "num_feature_chunks": int(len(feature_paths)),
        "cue_summary": _summarize_rows(rows),
        "outputs": {
            "cue_bank_audit": str(args.out_dir / "cue_bank_audit.csv"),
            "selected_layer_motion_debug": str(args.out_dir / "selected_layer_motion_debug.csv"),
        },
        "notes": [
            "G3 uses full Gram-row temporal variance for each selected layer.",
            "C_ttt_motion_risk_proxy is a proxy because real TTT post-delta maps are not present in these feature dumps.",
            "This script applies no HMC action and reports no trajectory metric.",
        ],
    }
    (args.out_dir / "cue_bank_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
