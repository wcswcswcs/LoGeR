#!/usr/bin/env python3
"""Audit ACL2 past_plus_future_light_real support alias in qq/qk/kk paths."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.hybrid_memory_controller import (  # noqa: E402
    _acl2_support_indices,
    _global_acl2_centroid_metric,
)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _support_weight_summary(t: int, support_indices: List[int]) -> Dict[str, float]:
    past = [s for s in support_indices if s < t]
    future = [s for s in support_indices if s > t]
    weights: Dict[int, float] = {}
    if past:
        for s in past:
            weights[s] = 0.75 / float(len(past))
    if future:
        for s in future:
            weights[s] = 0.25 / float(len(future))
    total = sum(weights.values())
    if total > 0.0:
        weights = {s: w / total for s, w in weights.items()}
    return {
        "past_count": float(len(past)),
        "future_count": float(len(future)),
        "weight_sum": float(sum(weights.values())),
        "past_weight_sum": float(sum(w for s, w in weights.items() if s < t)),
        "future_weight_sum": float(sum(w for s, w in weights.items() if s > t)),
    }


def _weighted_centroid(centroids: torch.Tensor, t: int, support_indices: List[int]) -> torch.Tensor:
    past = [s for s in support_indices if s < t]
    future = [s for s in support_indices if s > t]
    pieces: List[torch.Tensor] = []
    weights: List[float] = []
    if past:
        pieces.append(centroids[past].mean(dim=0))
        weights.append(0.75)
    if future:
        pieces.append(centroids[future].mean(dim=0))
        weights.append(0.25)
    if not pieces:
        return centroids[support_indices].permute(1, 0, 2)
    w = torch.tensor(weights, dtype=centroids.dtype, device=centroids.device)
    w = w / w.sum().clamp_min(1e-12)
    return (torch.stack(pieces, dim=0) * w[:, None, None]).sum(dim=0).unsqueeze(1)


def _manual_metric(q_layers: torch.Tensor, k_layers: torch.Tensor, basis: str) -> torch.Tensor:
    q_raw = q_layers.detach().float()
    k_raw = k_layers.detach().float()
    t_count, layer_count, height, width, dim = q_raw.shape
    q = F.normalize(q_raw.reshape(t_count, layer_count, height * width, dim), dim=-1)
    k = F.normalize(k_raw.reshape(t_count, layer_count, height * width, dim), dim=-1)
    q_cent = q.mean(dim=2)
    k_cent = k.mean(dim=2)

    rows: List[torch.Tensor] = []
    for t in range(t_count):
        support_indices = _acl2_support_indices(t_count, t, "past_plus_future_light_real")
        if basis == "kk":
            target = k[t]
            cent = _weighted_centroid(k_cent, t, support_indices)
        elif basis == "qk":
            target = q[t]
            cent = _weighted_centroid(k_cent, t, support_indices)
        else:
            target = q[t]
            cent = _weighted_centroid(q_cent, t, support_indices)
        sim = torch.einsum("lpd,lsd->lps", target, cent)
        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        rows.append((1.0 - sim01.mean(dim=(0, 2))).reshape(-1))
    return torch.stack(rows, dim=0).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=52052)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(int(args.seed))
    shape = (int(args.num_frames), int(args.layers), int(args.height), int(args.width), int(args.dim))
    q_layers = torch.randn(shape, generator=generator)
    k_layers = torch.randn(shape, generator=generator)
    default = torch.zeros((shape[0] * shape[2] * shape[3],), dtype=torch.float32)

    frame_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for basis in ("qq", "qk", "kk"):
        actual_real = _global_acl2_centroid_metric(
            q_layers,
            k_layers,
            default,
            basis=basis,
            stat="low",
            layerwin=f"g0_{shape[1] - 1}",
            support="past_plus_future_light_real",
        )
        actual_alias = _global_acl2_centroid_metric(
            q_layers,
            k_layers,
            default,
            basis=basis,
            stat="low",
            layerwin=f"g0_{shape[1] - 1}",
            support="past_plus_future_light",
        )
        actual_full = _global_acl2_centroid_metric(
            q_layers,
            k_layers,
            default,
            basis=basis,
            stat="low",
            layerwin=f"g0_{shape[1] - 1}",
            support="full_chunk_true",
        )
        expected = _manual_metric(q_layers, k_layers, basis)
        max_abs_diff = float((actual_real - expected).abs().max().item())
        alias_max_abs_diff = float((actual_real - actual_alias).abs().max().item())
        full_mean_abs_diff = float((actual_real - actual_full).abs().mean().item())
        support_pass = max_abs_diff <= 1e-6 and alias_max_abs_diff <= 1e-6
        summary_rows.append({
            "basis": basis,
            "support": "past_plus_future_light_real",
            "max_abs_diff_vs_manual_weighted": max_abs_diff,
            "max_abs_diff_vs_non_real_alias": alias_max_abs_diff,
            "mean_abs_diff_vs_full_support": full_mean_abs_diff,
            "support_alias_pass": support_pass,
            "differs_from_full_support": full_mean_abs_diff > 1e-6,
        })
        per_frame = int(shape[2] * shape[3])
        for t in range(shape[0]):
            support_indices = _acl2_support_indices(shape[0], t, "past_plus_future_light_real")
            weights = _support_weight_summary(t, support_indices)
            start = t * per_frame
            end = start + per_frame
            frame_rows.append({
                "basis": basis,
                "local_frame": int(t),
                "support": "past_plus_future_light_real",
                "support_count": int(len(support_indices)),
                "past_count": int(weights["past_count"]),
                "future_count": int(weights["future_count"]),
                "weight_sum": weights["weight_sum"],
                "past_weight_sum": weights["past_weight_sum"],
                "future_weight_sum": weights["future_weight_sum"],
                "frame_max_abs_diff_vs_manual_weighted": float(
                    (actual_real[start:end] - expected[start:end]).abs().max().item()
                ),
            })

    _write_csv(out_dir / "support_alias_unit_audit.csv", frame_rows)
    _write_csv(out_dir / "support_alias_unit_audit_summary.csv", summary_rows)
    summary = {
        "all_support_alias_pass": all(bool(row["support_alias_pass"]) for row in summary_rows),
        "all_bases_differ_from_full_support": all(bool(row["differs_from_full_support"]) for row in summary_rows),
        "summary_rows": summary_rows,
    }
    (out_dir / "support_alias_unit_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["all_support_alias_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
