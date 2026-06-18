#!/usr/bin/env python3
"""Build controlled overlap-pair directories for v69 Phase C audits.

The controls are diagnostic-only artifacts. They copy selected materialized
overlap-pair tensors and alter only semantic labels/confidence so that the
existing overlap-pair action oracle can be rerun unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch


GROUP_LABELS: Mapping[str, Sequence[str]] = {
    "dynamic": ("person", "car", "truck", "bus", "van", "rider", "cyclist", "bicycle", "motorcycle", "animal"),
    "sky_context": ("sky", "cloud", "horizon"),
    "vegetation_farstuff": ("grass", "tree", "vegetation", "plant", "terrain", "mountain"),
    "vertical_static": (
        "building",
        "house",
        "wall",
        "handrail_or_fence",
        "fence",
        "pole",
        "traffic sign",
        "traffic light",
        "billboard_or_bulletin_board",
        "bridge",
    ),
    "ground_static": ("road", "ground", "sidewalk", "crosswalk", "floor"),
    "void_lowtrust": ("void", "unknown", "unlabeled"),
}


def _normalise_label_names(label_names: Any) -> Dict[int, str]:
    if isinstance(label_names, Mapping):
        return {int(k): str(v) for k, v in label_names.items()}
    return {int(i): str(v) for i, v in enumerate(label_names)}


def _load_label_to_id(path: Path) -> Dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    if not isinstance(sem, dict):
        return {}
    label_names = _normalise_label_names(sem.get("label_names", []))
    return {name: idx for idx, name in label_names.items()}


def _ids_for_group(group: str, label_to_id: Mapping[str, int]) -> Set[int]:
    names = GROUP_LABELS.get(group)
    if names is None:
        raise ValueError(f"unknown semantic group: {group}")
    return {int(label_to_id[name]) for name in names if name in label_to_id}


def _fallback_non_group_id(label_to_id: Mapping[str, int], group_ids: Set[int]) -> int:
    for name in ("void", "sky", "person", "car"):
        idx = label_to_id.get(name)
        if idx is not None and int(idx) not in group_ids:
            return int(idx)
    for idx in sorted(int(v) for v in label_to_id.values()):
        if idx not in group_ids:
            return int(idx)
    raise ValueError("could not find a non-group label id")


def _load_pair(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict payload")
    return obj


def _copy_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pair.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def _generator(seed: int, pair_index: int, control_index: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 1009 * int(pair_index) + 104729 * int(control_index))
    return gen


def _permute_tensor(value: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if value.numel() <= 1:
        return value.clone()
    perm = torch.randperm(value.numel(), generator=gen)
    return value.reshape(-1)[perm].reshape(value.shape).clone()


def _random_mask_same_count(mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    flat = mask.reshape(-1).bool()
    n = int(flat.numel())
    k = int(flat.sum().item())
    out = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return out.reshape(mask.shape)
    if k >= n:
        return torch.ones(n, dtype=torch.bool).reshape(mask.shape)
    idx = torch.randperm(n, generator=gen)[:k]
    out[idx] = True
    return out.reshape(mask.shape)


def _random_mask_preserve_bins(
    mask: torch.Tensor,
    bins: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    flat_mask = mask.reshape(-1).bool()
    flat_bins = bins.reshape(-1).long()
    out = torch.zeros_like(flat_mask)
    for bin_id in torch.unique(flat_bins).tolist():
        bin_mask = flat_bins == int(bin_id)
        n = int(bin_mask.sum().item())
        k = int((flat_mask & bin_mask).sum().item())
        if k <= 0:
            continue
        local = torch.nonzero(bin_mask, as_tuple=False).reshape(-1)
        if k >= n:
            out[local] = True
            continue
        pick = torch.randperm(n, generator=gen)[:k]
        out[local[pick]] = True
    return out.reshape(mask.shape)


def _spatial_bins(pair: Mapping[str, Any], *, grid_size: int) -> torch.Tensor:
    coords = pair.get("curr_pixel_coords")
    labels = pair.get("curr_semantic_labels")
    if not torch.is_tensor(labels):
        raise ValueError("pair is missing curr_semantic_labels")
    if not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[0] != labels.numel() or coords.shape[1] < 2:
        return torch.zeros_like(labels, dtype=torch.long)
    coords = coords.detach().cpu().long()
    x = coords[:, 0]
    y = coords[:, 1]
    gx = torch.div(x, max(int(grid_size), 1), rounding_mode="floor")
    gy = torch.div(y, max(int(grid_size), 1), rounding_mode="floor")
    return gy * 10000 + gx


def _score_bins(pair: Mapping[str, Any], *, bins: int) -> torch.Tensor:
    labels = pair.get("curr_semantic_labels")
    if not torch.is_tensor(labels):
        raise ValueError("pair is missing curr_semantic_labels")
    prev_conf = pair.get("prev_conf")
    curr_conf = pair.get("curr_conf")
    if torch.is_tensor(prev_conf) and torch.is_tensor(curr_conf) and prev_conf.numel() == labels.numel() and curr_conf.numel() == labels.numel():
        score = torch.minimum(prev_conf.detach().cpu().float(), curr_conf.detach().cpu().float())
    else:
        score = torch.arange(labels.numel(), dtype=torch.float32)
    n_bins = max(int(bins), 1)
    if labels.numel() == 0:
        return torch.empty_like(labels, dtype=torch.long)
    ranks = torch.argsort(torch.argsort(score))
    denom = max(int(labels.numel()), 1)
    return torch.clamp((ranks * n_bins) // denom, max=n_bins - 1).long()


def _apply_group_mask(
    out: MutableMapping[str, Any],
    mask: torch.Tensor,
    *,
    group_label_id: int,
    non_group_label_id: int,
) -> None:
    labels = out.get("curr_semantic_labels")
    if not torch.is_tensor(labels):
        raise ValueError("pair is missing curr_semantic_labels")
    new_labels = torch.full_like(labels.detach().cpu().long(), int(non_group_label_id))
    new_labels[mask.reshape(-1).bool()] = int(group_label_id)
    out["curr_semantic_labels"] = new_labels.reshape(labels.shape)
    if torch.is_tensor(out.get("prev_semantic_labels")) and out["prev_semantic_labels"].numel() == labels.numel():
        out["prev_semantic_labels"] = out["curr_semantic_labels"].clone()


def _apply_control(
    pair: Mapping[str, Any],
    *,
    control: str,
    group_ids: Set[int],
    group_label_id: int,
    non_group_label_id: int,
    gen: torch.Generator,
    spatial_grid: int,
    weight_bins: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = _copy_pair(pair)
    labels = out.get("curr_semantic_labels")
    if not torch.is_tensor(labels):
        raise ValueError("pair is missing curr_semantic_labels")
    labels = labels.detach().cpu().long()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in sorted(group_ids):
        mask |= labels == int(label_id)

    if control == "label_shuffled":
        out["curr_semantic_labels"] = _permute_tensor(labels, gen).long()
        if torch.is_tensor(out.get("prev_semantic_labels")) and out["prev_semantic_labels"].numel() == labels.numel():
            out["prev_semantic_labels"] = out["curr_semantic_labels"].clone()
    elif control == "confidence_shuffled":
        conf = out.get("curr_semantic_conf")
        if not torch.is_tensor(conf):
            raise ValueError("pair is missing curr_semantic_conf")
        out["curr_semantic_conf"] = _permute_tensor(conf.detach().cpu().float(), gen)
        if torch.is_tensor(out.get("prev_semantic_conf")) and out["prev_semantic_conf"].numel() == conf.numel():
            out["prev_semantic_conf"] = out["curr_semantic_conf"].clone()
    elif control == "same_anchor_count_random":
        _apply_group_mask(
            out,
            _random_mask_same_count(mask, gen),
            group_label_id=group_label_id,
            non_group_label_id=non_group_label_id,
        )
    elif control == "same_spatial_coverage_random":
        _apply_group_mask(
            out,
            _random_mask_preserve_bins(mask, _spatial_bins(pair, grid_size=spatial_grid), gen),
            group_label_id=group_label_id,
            non_group_label_id=non_group_label_id,
        )
    elif control == "same_weight_distribution_random":
        _apply_group_mask(
            out,
            _random_mask_preserve_bins(mask, _score_bins(pair, bins=weight_bins), gen),
            group_label_id=group_label_id,
            non_group_label_id=non_group_label_id,
        )
    else:
        raise ValueError(f"unknown control: {control}")

    new_labels = out.get("curr_semantic_labels")
    new_mask = torch.zeros_like(labels, dtype=torch.bool)
    if torch.is_tensor(new_labels):
        new_labels = new_labels.detach().cpu().long()
        for label_id in sorted(group_ids):
            new_mask |= new_labels == int(label_id)

    meta = {
        "control": control,
        "original_group_point_count": int(mask.sum().item()),
        "control_group_point_count": int(new_mask.sum().item()),
        "point_count": int(labels.numel()),
        "group_label_id": int(group_label_id),
        "non_group_label_id": int(non_group_label_id),
    }
    out["schema"] = "acl2_v69_overlap_pair_control_v1"
    out["v69_control"] = meta
    return out, meta


def _select_pair_files(input_dir: Path, curr_chunks: Optional[Set[int]]) -> List[Path]:
    files = sorted(input_dir.glob("chunk_*_*.pt"))
    if curr_chunks is None:
        return files
    selected = []
    for path in files:
        parts = path.stem.split("_")
        if len(parts) >= 3 and int(parts[-1]) in curr_chunks:
            selected.append(path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--semantic-full-pt", type=Path, required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--filter-group", default="ground_static")
    parser.add_argument("--curr-chunks", default="")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--spatial-grid", type=int, default=80)
    parser.add_argument("--weight-bins", type=int, default=10)
    args = parser.parse_args()

    label_to_id = _load_label_to_id(args.semantic_full_pt)
    group_ids = _ids_for_group(args.filter_group, label_to_id)
    if not group_ids:
        raise ValueError(f"group {args.filter_group!r} has no ids in {args.semantic_full_pt}")
    group_label_id = min(group_ids)
    non_group_label_id = _fallback_non_group_id(label_to_id, group_ids)
    curr_chunks = {int(x) for x in str(args.curr_chunks).split(",") if x.strip()} or None
    pair_files = _select_pair_files(args.input_dir, curr_chunks)
    if not pair_files:
        raise FileNotFoundError(f"no selected pair files in {args.input_dir}")

    controls = [str(x).strip() for x in args.control if str(x).strip()]
    valid_controls = {
        "label_shuffled",
        "confidence_shuffled",
        "same_anchor_count_random",
        "same_spatial_coverage_random",
        "same_weight_distribution_random",
    }
    unknown = sorted(set(controls) - valid_controls)
    if unknown:
        raise ValueError(f"unknown controls: {unknown}")

    summary_rows: List[Dict[str, Any]] = []
    for control_index, control in enumerate(controls):
        out_dir = args.out_root / control
        out_dir.mkdir(parents=True, exist_ok=True)
        for pair_index, src in enumerate(pair_files):
            pair = _load_pair(src)
            gen = _generator(args.random_seed, pair_index, control_index)
            controlled, meta = _apply_control(
                pair,
                control=control,
                group_ids=group_ids,
                group_label_id=group_label_id,
                non_group_label_id=non_group_label_id,
                gen=gen,
                spatial_grid=args.spatial_grid,
                weight_bins=args.weight_bins,
            )
            dst = out_dir / src.name
            torch.save(controlled, dst)
            summary_rows.append({
                "control": control,
                "source_pair": str(src),
                "output_pair": str(dst),
                "prev_chunk": int(pair.get("prev_chunk", -1)),
                "curr_chunk": int(pair.get("curr_chunk", -1)),
                **meta,
            })

    summary = {
        "schema": "acl2_v69_overlap_pair_controls_summary_v1",
        "input_dir": str(args.input_dir),
        "out_root": str(args.out_root),
        "semantic_full_pt": str(args.semantic_full_pt),
        "controls": controls,
        "filter_group": str(args.filter_group),
        "group_ids": sorted(group_ids),
        "group_label_id": int(group_label_id),
        "non_group_label_id": int(non_group_label_id),
        "curr_chunks": sorted(curr_chunks) if curr_chunks is not None else "all",
        "pair_files": len(pair_files),
        "random_seed": int(args.random_seed),
        "spatial_grid": int(args.spatial_grid),
        "weight_bins": int(args.weight_bins),
        "rows": summary_rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "overlap_pair_controls_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
