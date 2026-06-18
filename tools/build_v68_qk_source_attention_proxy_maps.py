#!/usr/bin/env python3
"""Build ACL2 v68 dense Q/K source-attention proxy maps.

This is a Task-4 artifact builder, not a trajectory evaluator.  It uses the
saved v68 Q/K feature dumps to create a dense per-layer source-token mass map:

    [layer, query_frame, source_frame, patch_y, patch_x]

The map is a pooled-Q proxy.  For each layer and query frame, patch Q vectors
are averaged into one query vector, then scored against every source patch K
token with scaled dot-product and softmax.  This is intentionally named a
proxy: it is not the full model SDPA pair matrix and not a performance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_FEATURE_DIR = (
    "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/"
    "phaseC_target_feature_dumps/features"
)
DEFAULT_OUT_DIR = (
    "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/"
    "task4_qk_source_attention_proxy_maps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--chunks", default="")
    parser.add_argument("--layers", default="")
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    return parser.parse_args()


def _parse_int_csv(text: str) -> Optional[List[int]]:
    values: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values if values else None


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def _select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _feature_paths(feature_dir: Path, chunks: Optional[Sequence[int]]) -> List[Path]:
    paths = sorted(feature_dir.glob("chunk_*.pt"))
    if chunks is None:
        return paths
    wanted = {int(v) for v in chunks}
    out: List[Path] = []
    for path in paths:
        try:
            chunk = int(path.stem.split("_")[-1])
        except Exception:
            continue
        if chunk in wanted:
            out.append(path)
    return out


def _layer_ids_for_tap(payload: Mapping[str, Any], tap: str, tensor: torch.Tensor) -> List[int]:
    meta = dict(dict(payload.get("taps") or {}).get(tap) or {})
    selected = [int(x) for x in (meta.get("selected_layers") or [])]
    if selected and len(selected) == int(tensor.shape[1]):
        return selected
    return list(range(int(tensor.shape[1])))


def _entropy_norm(prob: torch.Tensor) -> torch.Tensor:
    p = prob.float().clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    denom = math.log(max(int(prob.shape[-1]), 2))
    return ent / float(denom)


def _build_proxy_for_chunk(
    payload: Mapping[str, Any],
    *,
    layers_filter: Optional[Sequence[int]],
    overlap_frames: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    q = payload.get("tap::global_q_raw_patchvec_layers")
    k = payload.get("tap::global_k_raw_patchvec_layers")
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise KeyError("Both tap::global_q_raw_patchvec_layers and tap::global_k_raw_patchvec_layers are required")
    if q.ndim != 5 or k.ndim != 5:
        raise ValueError(f"Expected q/k [T,L,H,W,D], got q={tuple(q.shape)} k={tuple(k.shape)}")
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")

    layer_ids_all = _layer_ids_for_tap(payload, "global_q_raw_patchvec_layers", q)
    selected_positions: List[int] = []
    selected_layer_ids: List[int] = []
    layer_filter_set = {int(x) for x in layers_filter} if layers_filter is not None else None
    for pos, layer_id in enumerate(layer_ids_all):
        if layer_filter_set is None or int(layer_id) in layer_filter_set:
            selected_positions.append(int(pos))
            selected_layer_ids.append(int(layer_id))
    if not selected_positions:
        raise ValueError(f"No requested layers present. available={layer_ids_all} requested={layers_filter}")

    q = q[:, selected_positions].detach().float()
    k = k[:, selected_positions].detach().float()
    T, L, H, W, D = [int(x) for x in q.shape]
    source_tokens = T * H * W
    scale = 1.0 / math.sqrt(float(max(D, 1)))
    overlap_n = max(1, min(int(overlap_frames), T))
    head = torch.arange(0, overlap_n, dtype=torch.long)
    tail = torch.arange(T - overlap_n, T, dtype=torch.long)

    layer_maps: List[torch.Tensor] = []
    source_frame_mass_rows: List[torch.Tensor] = []
    entropy_rows: List[torch.Tensor] = []
    same_frame_rows: List[torch.Tensor] = []
    neighbor_rows: List[torch.Tensor] = []
    tail_to_head_rows: List[torch.Tensor] = []
    head_to_tail_rows: List[torch.Tensor] = []
    tail_head_maps: List[torch.Tensor] = []
    head_tail_maps: List[torch.Tensor] = []

    for li in range(L):
        q_l = q[:, li].to(device=device, non_blocking=True)
        k_l = k[:, li].to(device=device, non_blocking=True)
        q_pool = q_l.reshape(T, H * W, D).mean(dim=1)
        k_flat = k_l.reshape(source_tokens, D)
        scores = (q_pool @ k_flat.T) * scale
        attn = torch.softmax(scores.float(), dim=-1)
        src_map = attn.reshape(T, T, H, W)
        src_frame_mass = src_map.sum(dim=(-1, -2))
        entropy_rows.append(_entropy_norm(attn).detach().cpu())
        source_frame_mass_rows.append(src_frame_mass.detach().cpu())
        same_frame_rows.append(src_frame_mass.diagonal().detach().cpu())
        neigh_vals: List[torch.Tensor] = []
        if T > 1:
            neigh_vals.append(src_frame_mass[1:, :-1].diagonal())
            neigh_vals.append(src_frame_mass[:-1, 1:].diagonal())
        neighbor_rows.append(torch.cat(neigh_vals).detach().cpu() if neigh_vals else torch.empty(0))
        tail_to_head = src_frame_mass[tail.to(device), :][:, head.to(device)].sum(dim=1)
        head_to_tail = src_frame_mass[head.to(device), :][:, tail.to(device)].sum(dim=1)
        tail_to_head_rows.append(tail_to_head.detach().cpu())
        head_to_tail_rows.append(head_to_tail.detach().cpu())
        tail_head_maps.append(src_map[tail.to(device), :][:, head.to(device)].mean(dim=(0, 1)).detach().cpu())
        head_tail_maps.append(src_map[head.to(device), :][:, tail.to(device)].mean(dim=(0, 1)).detach().cpu())
        layer_maps.append(src_map.detach().to(device="cpu", dtype=out_dtype))

    source_attention_proxy = torch.stack(layer_maps, dim=0).contiguous()
    source_frame_mass = torch.stack(source_frame_mass_rows, dim=0).float().contiguous()
    tail_query_to_head_source_map = torch.stack(tail_head_maps, dim=0).to(dtype=out_dtype).contiguous()
    head_query_to_tail_source_map = torch.stack(head_tail_maps, dim=0).to(dtype=out_dtype).contiguous()

    def _mean_cat(rows: Sequence[torch.Tensor]) -> Optional[float]:
        vals = [r.reshape(-1).float() for r in rows if r.numel()]
        if not vals:
            return None
        return float(torch.cat(vals).mean().item())

    metrics = {
        "mean_entropy_norm": _mean_cat(entropy_rows),
        "mean_same_frame_mass": _mean_cat(same_frame_rows),
        "mean_neighbor_1frame_mass": _mean_cat(neighbor_rows),
        "mean_tail_query_to_head_source_mass": _mean_cat(tail_to_head_rows),
        "mean_head_query_to_tail_source_mass": _mean_cat(head_to_tail_rows),
    }
    if metrics["mean_tail_query_to_head_source_mass"] is not None and metrics["mean_head_query_to_tail_source_mass"] is not None:
        metrics["head_tail_overlap_balance_abs"] = abs(
            float(metrics["mean_tail_query_to_head_source_mass"])
            - float(metrics["mean_head_query_to_tail_source_mass"])
        )
    else:
        metrics["head_tail_overlap_balance_abs"] = None

    out_payload: Dict[str, Any] = {
        "schema": "acl2_v68_qk_pooled_source_attention_proxy_v1",
        "source": "offline_from_v68_layer_pca_feature_dump",
        "proxy_definition": (
            "mean-pool patch Q per query frame/layer, score against all source patch K tokens with scaled dot-product, "
            "softmax over source patch tokens"
        ),
        "not_raw_model_sdpa_attention": True,
        "chunk_idx": int(payload.get("chunk_idx")),
        "start_frame": int(payload.get("start_frame")),
        "end_frame": int(payload.get("end_frame")),
        "selected_layers": [int(x) for x in selected_layer_ids],
        "patch_grid": [H, W],
        "overlap_frames": int(overlap_n),
        "source_attention_proxy_shape": list(source_attention_proxy.shape),
        "source_attention_proxy_dtype": str(source_attention_proxy.dtype),
        "source_attention_proxy": source_attention_proxy,
        "source_frame_mass": source_frame_mass,
        "tail_query_to_head_source_map": tail_query_to_head_source_map,
        "head_query_to_tail_source_map": head_query_to_tail_source_map,
        "metrics": metrics,
    }
    row = {
        "chunk_idx": int(payload.get("chunk_idx")),
        "start_frame": int(payload.get("start_frame")),
        "end_frame": int(payload.get("end_frame")),
        "selected_layers": ",".join(str(x) for x in selected_layer_ids),
        "patch_grid": f"{H}x{W}",
        "overlap_frames": int(overlap_n),
        **metrics,
    }
    return out_payload, row


def _json_default(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


def main() -> int:
    args = parse_args()
    feature_dir = args.feature_dir
    out_dir = args.out_dir
    map_dir = out_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    chunks = _parse_int_csv(args.chunks)
    layers = _parse_int_csv(args.layers)
    device = _select_device(args.device)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    paths = _feature_paths(feature_dir, chunks)
    if not paths:
        raise FileNotFoundError(f"No chunk feature dumps found under {feature_dir} for chunks={chunks}")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = _torch_load(path)
        out_payload, row = _build_proxy_for_chunk(
            payload,
            layers_filter=layers,
            overlap_frames=args.overlap_frames,
            device=device,
            out_dtype=out_dtype,
        )
        chunk_idx = int(out_payload["chunk_idx"])
        out_path = map_dir / f"chunk_{chunk_idx:03d}_qk_source_attention_proxy.pt"
        torch.save(out_payload, out_path)
        row["path"] = str(out_path)
        row["bytes"] = int(out_path.stat().st_size)
        rows.append(row)
        print(f"wrote {out_path}")

    csv_path = out_dir / "qk_source_attention_proxy_rows.csv"
    fieldnames = [
        "chunk_idx",
        "start_frame",
        "end_frame",
        "selected_layers",
        "patch_grid",
        "overlap_frames",
        "mean_entropy_norm",
        "mean_same_frame_mass",
        "mean_neighbor_1frame_mass",
        "mean_tail_query_to_head_source_mass",
        "mean_head_query_to_tail_source_mass",
        "head_tail_overlap_balance_abs",
        "path",
        "bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "schema": "acl2_v68_qk_source_attention_proxy_summary_v1",
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "map_dir": str(map_dir),
        "num_chunks": len(rows),
        "chunks": [int(r["chunk_idx"]) for r in rows],
        "device": str(device),
        "dtype": args.dtype,
        "layers_requested": layers,
        "overlap_frames": int(args.overlap_frames),
        "proxy_not_raw_model_attention": True,
        "mean_metrics": {
            "mean_entropy_norm": _mean("mean_entropy_norm"),
            "mean_same_frame_mass": _mean("mean_same_frame_mass"),
            "mean_neighbor_1frame_mass": _mean("mean_neighbor_1frame_mass"),
            "mean_tail_query_to_head_source_mass": _mean("mean_tail_query_to_head_source_mass"),
            "mean_head_query_to_tail_source_mass": _mean("mean_head_query_to_tail_source_mass"),
            "mean_head_tail_overlap_balance_abs": _mean("head_tail_overlap_balance_abs"),
        },
        "rows_csv": str(csv_path),
    }
    summary_path = out_dir / "qk_source_attention_proxy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
