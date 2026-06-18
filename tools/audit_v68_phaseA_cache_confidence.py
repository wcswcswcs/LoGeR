#!/usr/bin/env python3
"""Phase A cache/confidence guard for ACL2 v68.

This is a read-only artifact audit. It verifies that Stage-C chunk masklets
carry dense label maps and confidence maps, then computes the same
confidence-aware patch trust used by ``project_dense_semantic_label_maps``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from loger.pipeline.geometry_backbone import PATCH_SIZE
from loger.pipeline.semantic_prior_generator import (
    _mode_pool_dense_semantic_patches,
    _normalize_dense_semantic_confidence,
)


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _semantic(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload.get("semantic_segmentation", {}) or {})
    return dict(getattr(payload, "semantic_segmentation", {}) or {})


def _shape(x: Any) -> Optional[List[int]]:
    if torch.is_tensor(x):
        return [int(v) for v in x.shape]
    return None


def _finite_minmax(x: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if not torch.is_tensor(x) or x.numel() <= 0:
        return None, None
    vals = x.detach().cpu().float()
    finite = torch.isfinite(vals)
    if not bool(finite.any().item()):
        return None, None
    vals = vals[finite]
    return float(vals.min().item()), float(vals.max().item())


def _chunk_id(path: Path) -> int:
    name = path.parent.name
    if name.startswith("chunk_"):
        parts = name.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
    return -1


def _iter_chunk_paths(cache_dir: Path) -> Iterable[Path]:
    yield from sorted(cache_dir.glob("chunk_*/masklet.pt"), key=lambda p: (_chunk_id(p), str(p)))


def _slice_equal(
    chunk_tensor: torch.Tensor,
    full_tensor: Optional[torch.Tensor],
    *,
    start: int,
    end: int,
    atol: float = 1e-6,
) -> Optional[bool]:
    if full_tensor is None:
        return None
    if end <= start or start < 0 or end > int(full_tensor.shape[0]):
        return False
    ref = full_tensor[start:end].detach().cpu()
    cur = chunk_tensor.detach().cpu()
    if tuple(ref.shape) != tuple(cur.shape):
        return False
    if cur.dtype.is_floating_point or ref.dtype.is_floating_point:
        return bool(torch.allclose(cur.float(), ref.float(), atol=atol, rtol=0.0))
    return bool(torch.equal(cur, ref))


def _row_for_chunk(
    path: Path,
    *,
    full_labels: Optional[torch.Tensor],
    full_confidence: Optional[torch.Tensor],
    patch_size: int,
) -> Dict[str, Any]:
    payload = _load_pt(path)
    sem = _semantic(payload)
    labels = sem.get("label_maps")
    confidence = sem.get("confidence_maps")
    row: Dict[str, Any] = {
        "chunk_id": _chunk_id(path),
        "chunk_dir": str(path.parent),
        "has_label_maps": torch.is_tensor(labels),
        "has_confidence_maps": torch.is_tensor(confidence),
        "semantic_format": sem.get("format"),
        "semantic_source": sem.get("source"),
        "global_start_frame": sem.get("global_start_frame"),
        "global_end_frame": sem.get("global_end_frame"),
        "label_shape": json.dumps(_shape(labels)),
        "confidence_shape": json.dumps(_shape(confidence)),
        "label_dtype": str(labels.dtype) if torch.is_tensor(labels) else "",
        "confidence_dtype": str(confidence.dtype) if torch.is_tensor(confidence) else "",
        "label_equal_full_slice": None,
        "confidence_equal_full_slice": None,
        "confidence_raw_min": None,
        "confidence_raw_max": None,
        "confidence_normalized_min": None,
        "confidence_normalized_max": None,
        "confidence_normalization_applied": None,
        "patch_grid_h": None,
        "patch_grid_w": None,
        "patch_purity_mean": None,
        "patch_confidence_mean": None,
        "semantic_trust_mean": None,
        "semantic_projection_source": "unavailable",
        "phaseA_chunk_pass": False,
        "failure_reason": "",
    }
    if not torch.is_tensor(labels):
        row["failure_reason"] = "missing_label_maps"
        return row
    if labels.ndim != 3:
        row["failure_reason"] = f"label_maps_ndim_{int(labels.ndim)}"
        return row
    if not torch.is_tensor(confidence):
        row["failure_reason"] = "missing_confidence_maps"
        return row
    if confidence.ndim not in (3, 4):
        row["failure_reason"] = f"confidence_maps_ndim_{int(confidence.ndim)}"
        return row

    start = int(sem.get("global_start_frame", 0) or 0)
    end = int(sem.get("global_end_frame", start + int(labels.shape[0])) or (start + int(labels.shape[0])))
    row["label_equal_full_slice"] = _slice_equal(labels, full_labels, start=start, end=end)
    row["confidence_equal_full_slice"] = _slice_equal(confidence, full_confidence, start=start, end=end, atol=1e-5)

    raw_min, raw_max = _finite_minmax(confidence)
    row["confidence_raw_min"] = raw_min
    row["confidence_raw_max"] = raw_max

    H, W = int(labels.shape[-2]), int(labels.shape[-1])
    patch_h = max(1, H // max(int(patch_size), 1))
    patch_w = max(1, W // max(int(patch_size), 1))
    row["patch_grid_h"] = patch_h
    row["patch_grid_w"] = patch_w
    conf_norm, cdebug = _normalize_dense_semantic_confidence(
        confidence,
        target_shape=(int(labels.shape[0]), H, W),
    )
    _, purity, cpatch = _mode_pool_dense_semantic_patches(
        labels.detach().cpu().long(),
        conf_norm,
        patch_grid=(patch_h, patch_w),
    )
    trust = (cpatch * purity.square()).clamp(0.0, 1.0)
    row["confidence_normalized_min"] = cdebug.get("semantic_confidence_normalized_min")
    row["confidence_normalized_max"] = cdebug.get("semantic_confidence_normalized_max")
    row["confidence_normalization_applied"] = cdebug.get("semantic_confidence_normalization_applied")
    row["patch_purity_mean"] = float(purity.mean().item()) if purity.numel() else 0.0
    row["patch_confidence_mean"] = float(cpatch.mean().item()) if cpatch.numel() else 0.0
    row["semantic_trust_mean"] = float(trust.mean().item()) if trust.numel() else 0.0
    row["semantic_projection_source"] = "dense_label_maps_and_confidence_maps"
    normalized_ok = (
        row["confidence_normalized_min"] is not None
        and row["confidence_normalized_max"] is not None
        and float(row["confidence_normalized_min"]) >= 0.0
        and float(row["confidence_normalized_max"]) <= 1.0
    )
    shape_ok = tuple(labels.shape) == tuple(confidence.shape[:3])
    full_ok = row["label_equal_full_slice"] in (True, None) and row["confidence_equal_full_slice"] in (True, None)
    trust_ok = row["semantic_trust_mean"] is not None and float(row["semantic_trust_mean"]) >= 0.0
    row["phaseA_chunk_pass"] = bool(shape_ok and normalized_ok and full_ok and trust_ok)
    if not row["phaseA_chunk_pass"]:
        reasons: List[str] = []
        if not shape_ok:
            reasons.append("label_confidence_shape_mismatch")
        if not normalized_ok:
            reasons.append("confidence_not_normalized")
        if not full_ok:
            reasons.append("chunk_slice_mismatch_full_semantic")
        if not trust_ok:
            reasons.append("trust_unavailable")
        row["failure_reason"] = ";".join(reasons)
    return row


def _load_full_semantic(path: Optional[Path]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    if path is None:
        return None, None, {}
    payload = _load_pt(path)
    sem = _semantic(payload)
    labels = sem.get("label_maps")
    conf = sem.get("confidence_maps")
    return (
        labels.detach().cpu() if torch.is_tensor(labels) else None,
        conf.detach().cpu() if torch.is_tensor(conf) else None,
        {
            "path": str(path),
            "has_label_maps": torch.is_tensor(labels),
            "has_confidence_maps": torch.is_tensor(conf),
            "label_shape": _shape(labels),
            "confidence_shape": _shape(conf),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--full-semantic-pt", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-chunks", default=38, type=int)
    parser.add_argument("--patch-size", default=PATCH_SIZE, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_labels, full_confidence, full_debug = _load_full_semantic(args.full_semantic_pt)
    rows = [
        _row_for_chunk(
            path,
            full_labels=full_labels,
            full_confidence=full_confidence,
            patch_size=int(args.patch_size),
        )
        for path in _iter_chunk_paths(args.cache_dir)
    ]

    csv_path = args.out_dir / "semantic_confidence_audit.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "chunk_id",
        "chunk_dir",
        "phaseA_chunk_pass",
        "failure_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    pass_count = sum(1 for row in rows if bool(row.get("phaseA_chunk_pass")))
    confidence_available_count = sum(1 for row in rows if bool(row.get("has_confidence_maps")))
    label_available_count = sum(1 for row in rows if bool(row.get("has_label_maps")))
    summary: Dict[str, Any] = {
        "schema": "acl2_v68_phaseA_cache_confidence_guard_v1",
        "cache_dir": str(args.cache_dir),
        "expected_chunks": int(args.expected_chunks),
        "num_chunks": len(rows),
        "label_maps_chunks": label_available_count,
        "confidence_maps_chunks": confidence_available_count,
        "phaseA_chunk_pass_count": pass_count,
        "all_chunks_have_label_maps": bool(label_available_count == len(rows) == int(args.expected_chunks)),
        "all_chunks_have_confidence_maps": bool(confidence_available_count == len(rows) == int(args.expected_chunks)),
        "all_chunks_pass": bool(pass_count == len(rows) == int(args.expected_chunks)),
        "semantic_source": "dense_label_maps_and_confidence_maps" if confidence_available_count == len(rows) and rows else "incomplete",
        "confidence_normalized_to_0_1": bool(
            rows
            and all(
                row.get("confidence_normalized_min") is not None
                and row.get("confidence_normalized_max") is not None
                and float(row["confidence_normalized_min"]) >= 0.0
                and float(row["confidence_normalized_max"]) <= 1.0
                for row in rows
            )
        ),
        "semantic_confidence_audit_csv": str(csv_path),
        "full_semantic": full_debug,
        "patch_size": int(args.patch_size),
        "patch_purity_mean": (
            sum(float(row["patch_purity_mean"]) for row in rows if row.get("patch_purity_mean") is not None)
            / max(sum(1 for row in rows if row.get("patch_purity_mean") is not None), 1)
        ),
        "semantic_trust_mean": (
            sum(float(row["semantic_trust_mean"]) for row in rows if row.get("semantic_trust_mean") is not None)
            / max(sum(1 for row in rows if row.get("semantic_trust_mean") is not None), 1)
        ),
        "failures": [
            {
                "chunk_id": row.get("chunk_id"),
                "chunk_dir": row.get("chunk_dir"),
                "failure_reason": row.get("failure_reason"),
            }
            for row in rows
            if not bool(row.get("phaseA_chunk_pass"))
        ],
    }
    json_path = args.out_dir / "dense_semantic_source_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
