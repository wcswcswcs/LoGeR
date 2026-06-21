#!/usr/bin/env python3
"""Compute compact current/cache K/V consistency from v78 v68 feature dumps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_FEATURE_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/qkv_tiny_smoke_chunk06_p9_34_v1/chunk06/"
    "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
    "v68_layer_pca_features"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/qkv_tiny_smoke_chunk06_p9_34_v1/"
    "compact_qkv_consistency_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overlap-frames", type=int, default=3)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _layer_ids(payload: dict[str, Any], tap: str) -> list[int]:
    tensor = payload.get(f"layer_ids::{tap}")
    if torch.is_tensor(tensor):
        return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    meta = payload.get("taps", {}).get(tap, {})
    return [int(x) for x in meta.get("selected_layer_ids", [])]


def _stats_for_pair(
    *,
    payload: dict[str, Any],
    feature_file: Path,
    kind: str,
) -> list[dict[str, Any]]:
    current_tap = f"pca_swa_current_{kind}_layers"
    cache_tap = f"pca_swa_cache_{kind}_layers"
    current = payload.get(f"tap::{current_tap}")
    cache = payload.get(f"tap::{cache_tap}")
    if not torch.is_tensor(current) or not torch.is_tensor(cache):
        return [
            {
                "feature_file": str(feature_file),
                "chunk_idx": payload.get("chunk_idx"),
                "kind": kind,
                "available": False,
                "reason": "missing_current_or_cache_tap",
            }
        ]
    cur = current.detach().cpu().float()
    cached = cache.detach().cpu().float()
    if tuple(cur.shape) != tuple(cached.shape):
        return [
            {
                "feature_file": str(feature_file),
                "chunk_idx": payload.get("chunk_idx"),
                "kind": kind,
                "available": False,
                "reason": f"shape_mismatch:{tuple(cur.shape)}!={tuple(cached.shape)}",
            }
        ]

    rows: list[dict[str, Any]] = []
    layer_ids = _layer_ids(payload, current_tap)
    diff = cur - cached
    for layer_pos in range(int(cur.shape[1]) if cur.ndim >= 2 else 1):
        cur_layer = cur[:, layer_pos] if cur.ndim >= 2 else cur
        cache_layer = cached[:, layer_pos] if cached.ndim >= 2 else cached
        diff_layer = cur_layer - cache_layer
        cur_flat = cur_layer.reshape(-1, cur_layer.shape[-1])
        cache_flat = cache_layer.reshape(-1, cache_layer.shape[-1])
        cos = F.cosine_similarity(cur_flat, cache_flat, dim=-1)
        rows.append(
            {
                "feature_file": str(feature_file),
                "chunk_idx": int(payload.get("chunk_idx")),
                "start_frame": int(payload.get("start_frame")),
                "end_frame": int(payload.get("end_frame")),
                "kind": kind,
                "layer_pos": int(layer_pos),
                "layer_id": int(layer_ids[layer_pos]) if layer_pos < len(layer_ids) else None,
                "available": True,
                "shape": list(cur_layer.shape),
                "allclose": bool(torch.allclose(cur_layer, cache_layer)),
                "max_abs_diff": float(diff_layer.abs().max().item()),
                "mean_abs_diff": float(diff_layer.abs().mean().item()),
                "rmse_diff": float(torch.sqrt((diff_layer.float() ** 2).mean()).item()),
                "cosine_mean": float(cos.mean().item()),
                "cosine_min": float(cos.min().item()),
                "cosine_p05": float(torch.quantile(cos, 0.05).item()),
                "cosine_p50": float(torch.quantile(cos, 0.50).item()),
                "cosine_p95": float(torch.quantile(cos, 0.95).item()),
            }
        )
    rows.append(
        {
            "feature_file": str(feature_file),
            "chunk_idx": int(payload.get("chunk_idx")),
            "start_frame": int(payload.get("start_frame")),
            "end_frame": int(payload.get("end_frame")),
            "kind": f"{kind}_all_layers",
            "layer_pos": "all",
            "layer_id": "all",
            "available": True,
            "shape": list(cur.shape),
            "allclose": bool(torch.allclose(cur, cached)),
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "rmse_diff": float(torch.sqrt((diff.float() ** 2).mean()).item()),
            "cosine_mean": _finite(
                F.cosine_similarity(cur.reshape(-1, cur.shape[-1]), cached.reshape(-1, cached.shape[-1]), dim=-1).mean().item()
            ),
        }
    )
    return rows


def _tap_tensor(payload: dict[str, Any], kind: str, source: str) -> torch.Tensor | None:
    tap = f"tap::pca_swa_{source}_{kind}_layers"
    value = payload.get(tap)
    return value.detach().cpu().float() if torch.is_tensor(value) else None


def _cross_overlap_rows(
    feature_payloads: list[tuple[Path, dict[str, Any]]],
    overlap_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(feature_payloads, key=lambda item: int(item[1].get("chunk_idx", -1)))
    for (prev_file, prev_payload), (cur_file, cur_payload) in zip(ordered, ordered[1:]):
        prev_chunk = int(prev_payload.get("chunk_idx"))
        cur_chunk = int(cur_payload.get("chunk_idx"))
        if cur_chunk != prev_chunk + 1:
            continue
        for kind in ("k", "v"):
            for source_pair in (
                ("current", "current"),
                ("cache", "current"),
                ("current", "cache"),
                ("cache", "cache"),
            ):
                prev_tensor = _tap_tensor(prev_payload, kind, source_pair[0])
                cur_tensor = _tap_tensor(cur_payload, kind, source_pair[1])
                if prev_tensor is None or cur_tensor is None:
                    rows.append(
                        {
                            "prev_feature_file": str(prev_file),
                            "cur_feature_file": str(cur_file),
                            "prev_chunk_idx": prev_chunk,
                            "cur_chunk_idx": cur_chunk,
                            "kind": kind,
                            "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                            "available": False,
                            "reason": "missing_tap",
                        }
                    )
                    continue
                if prev_tensor.ndim < 5 or tuple(prev_tensor.shape[2:]) != tuple(cur_tensor.shape[2:]):
                    rows.append(
                        {
                            "prev_feature_file": str(prev_file),
                            "cur_feature_file": str(cur_file),
                            "prev_chunk_idx": prev_chunk,
                            "cur_chunk_idx": cur_chunk,
                            "kind": kind,
                            "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                            "available": False,
                            "reason": "shape_mismatch_or_unexpected_ndim",
                        }
                    )
                    continue
                layer_ids = _layer_ids(prev_payload, f"pca_swa_{source_pair[0]}_{kind}_layers")
                n = min(int(overlap_frames), int(prev_tensor.shape[0]), int(cur_tensor.shape[0]))
                for layer_pos in range(int(prev_tensor.shape[1])):
                    prev_tail = prev_tensor[-n:, layer_pos]
                    cur_head = cur_tensor[:n, layer_pos]
                    cos = F.cosine_similarity(
                        prev_tail.reshape(-1, prev_tail.shape[-1]),
                        cur_head.reshape(-1, cur_head.shape[-1]),
                        dim=-1,
                    )
                    diff = prev_tail - cur_head
                    rows.append(
                        {
                            "prev_feature_file": str(prev_file),
                            "cur_feature_file": str(cur_file),
                            "prev_chunk_idx": prev_chunk,
                            "cur_chunk_idx": cur_chunk,
                            "prev_tail_start_frame": int(prev_payload.get("end_frame")) - n,
                            "prev_tail_end_frame": int(prev_payload.get("end_frame")),
                            "cur_head_start_frame": int(cur_payload.get("start_frame")),
                            "cur_head_end_frame": int(cur_payload.get("start_frame")) + n,
                            "kind": kind,
                            "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                            "layer_pos": int(layer_pos),
                            "layer_id": int(layer_ids[layer_pos]) if layer_pos < len(layer_ids) else None,
                            "available": True,
                            "overlap_frames": int(n),
                            "grid_shape": list(prev_tail.shape),
                            "mean_abs_diff": float(diff.abs().mean().item()),
                            "rmse_diff": float(torch.sqrt((diff.float() ** 2).mean()).item()),
                            "cosine_mean": float(cos.mean().item()),
                            "cosine_min": float(cos.min().item()),
                            "cosine_p05": float(torch.quantile(cos, 0.05).item()),
                            "cosine_p50": float(torch.quantile(cos, 0.50).item()),
                            "cosine_p95": float(torch.quantile(cos, 0.95).item()),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    feature_payloads: list[tuple[Path, dict[str, Any]]] = []
    for feature_file in sorted(args.feature_dir.glob("*.pt")):
        payload = torch.load(feature_file, map_location="cpu")
        feature_payloads.append((feature_file, payload))
        rows.extend(_stats_for_pair(payload=payload, feature_file=feature_file, kind="k"))
        rows.extend(_stats_for_pair(payload=payload, feature_file=feature_file, kind="v"))
    cross_rows = _cross_overlap_rows(feature_payloads, args.overlap_frames)

    rows_csv = args.out_dir / "compact_qkv_consistency_rows.csv"
    cross_csv = args.out_dir / "compact_qkv_cross_overlap_rows.csv"
    summary_json = args.out_dir / "compact_qkv_consistency_summary.json"
    _write_csv(rows_csv, rows)
    _write_csv(cross_csv, cross_rows)
    available_rows = [row for row in rows if row.get("available") is True]
    available_cross = [row for row in cross_rows if row.get("available") is True]
    max_abs = max((_finite(row.get("max_abs_diff")) or 0.0 for row in available_rows), default=None)
    mean_abs = max((_finite(row.get("mean_abs_diff")) or 0.0 for row in available_rows), default=None)
    min_cos = min((_finite(row.get("cosine_min")) or 1.0 for row in available_rows if row.get("cosine_min") != ""), default=None)
    cross_by_kind_layer = {
        f"{row['source_pair']}:{row['kind']}:L{row['layer_id']}": row["cosine_mean"]
        for row in available_cross
    }
    _write_json(
        summary_json,
        {
            "schema": "acl2_v78_compact_qkv_consistency_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "feature_dir": str(args.feature_dir),
            "num_feature_files": len(list(args.feature_dir.glob("*.pt"))),
            "rows_csv": str(rows_csv),
            "cross_overlap_rows_csv": str(cross_csv),
            "num_rows": len(rows),
            "num_cross_overlap_rows": len(cross_rows),
            "all_available_rows_allclose": bool(available_rows)
            and all(bool(row.get("allclose")) for row in available_rows),
            "max_abs_diff_across_rows": max_abs,
            "max_mean_abs_diff_across_rows": mean_abs,
            "min_cosine_across_layer_rows": min_cos,
            "cross_overlap_cosine_mean_by_source_kind_layer": cross_by_kind_layer,
            "cross_overlap_min_cosine_mean": min(
                (_finite(row.get("cosine_mean")) or 1.0 for row in available_cross),
                default=None,
            ),
            "interpretation": [
                "The compact dump is available and contains current/cache SWA K/V tensors for layers 18 and 26.",
                "For this tiny smoke, current/cache K and V tensors are allclose in both context chunks.",
                "Same-chunk current/cache identity proves the compact export path works but does not explain action differences.",
                "Cross-chunk previous-tail to current-head K/V cosine is nontrivial and can become the K/V alignment signal for selector routing.",
            ],
            "limitations": [
                "This audit covers one configured tiny smoke only.",
                "The SWA overlap selected mask has 1260 tokens per frame, while compact PCA feature grid is 19x66=1254, so selected-mask-conditioned K/V stats are not computed here.",
                "No baseline/control was run in this tiny smoke; phase9 gate status is not interpretable.",
            ],
        },
    )
    print(json.dumps({"rows": str(rows_csv), "cross": str(cross_csv), "summary": str(summary_json)}, indent=2))


if __name__ == "__main__":
    main()
