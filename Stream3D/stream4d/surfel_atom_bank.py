from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .measurement_bank import MeasurementBank, json_safe


@dataclass
class SurfelAtomBank:
    scene: str
    variant: str
    atom_ids: np.ndarray
    offsets: np.ndarray
    surfel_indices: np.ndarray
    source_frame: np.ndarray
    source_mask_id: np.ndarray
    atom_size: np.ndarray
    mean_rgb: np.ndarray
    trajectory_descriptor: np.ndarray
    boundary_safe_ratio: np.ndarray
    negative_visible_outside_ratio: np.ndarray
    mask_entropy: np.ndarray
    trajectory_variance: np.ndarray
    is_unknown: np.ndarray
    mask_membership_json: list[dict[str, int]]
    frame_visibility_json: list[dict[str, int]]
    neighbor_atom_ids_json: list[list[int]]
    meta: dict[str, Any]

    @property
    def num_atoms(self) -> int:
        return int(self.atom_ids.shape[0])

    def surfels_for_atom(self, atom_index: int) -> np.ndarray:
        start = int(self.offsets[atom_index])
        end = int(self.offsets[atom_index + 1])
        return self.surfel_indices[start:end]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray(self.scene),
            variant=np.asarray(self.variant),
            atom_ids=self.atom_ids.astype(np.int64, copy=False),
            offsets=self.offsets.astype(np.int64, copy=False),
            surfel_indices=self.surfel_indices.astype(np.int64, copy=False),
            source_frame=self.source_frame.astype(np.int64, copy=False),
            source_mask_id=self.source_mask_id.astype(np.int64, copy=False),
            atom_size=self.atom_size.astype(np.int64, copy=False),
            mean_rgb=self.mean_rgb.astype(np.float32, copy=False),
            trajectory_descriptor=self.trajectory_descriptor.astype(np.float32, copy=False),
            boundary_safe_ratio=self.boundary_safe_ratio.astype(np.float32, copy=False),
            negative_visible_outside_ratio=self.negative_visible_outside_ratio.astype(np.float32, copy=False),
            mask_entropy=self.mask_entropy.astype(np.float32, copy=False),
            trajectory_variance=self.trajectory_variance.astype(np.float32, copy=False),
            is_unknown=self.is_unknown.astype(bool, copy=False),
            mask_membership_json=np.asarray(json.dumps(json_safe(self.mask_membership_json), sort_keys=True)),
            frame_visibility_json=np.asarray(json.dumps(json_safe(self.frame_visibility_json), sort_keys=True)),
            neighbor_atom_ids_json=np.asarray(json.dumps(json_safe(self.neighbor_atom_ids_json), sort_keys=True)),
            meta_json=np.asarray(json.dumps(json_safe(self.meta), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "SurfelAtomBank":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                scene=str(data["scene"].item()),
                variant=str(data["variant"].item()),
                atom_ids=np.asarray(data["atom_ids"], dtype=np.int64),
                offsets=np.asarray(data["offsets"], dtype=np.int64),
                surfel_indices=np.asarray(data["surfel_indices"], dtype=np.int64),
                source_frame=np.asarray(data["source_frame"], dtype=np.int64),
                source_mask_id=np.asarray(data["source_mask_id"], dtype=np.int64),
                atom_size=np.asarray(data["atom_size"], dtype=np.int64),
                mean_rgb=np.asarray(data["mean_rgb"], dtype=np.float32),
                trajectory_descriptor=np.asarray(data["trajectory_descriptor"], dtype=np.float32),
                boundary_safe_ratio=np.asarray(data["boundary_safe_ratio"], dtype=np.float32),
                negative_visible_outside_ratio=np.asarray(data["negative_visible_outside_ratio"], dtype=np.float32),
                mask_entropy=np.asarray(data["mask_entropy"], dtype=np.float32),
                trajectory_variance=np.asarray(data["trajectory_variance"], dtype=np.float32),
                is_unknown=np.asarray(data["is_unknown"], dtype=bool),
                mask_membership_json=json.loads(str(data["mask_membership_json"].item())),
                frame_visibility_json=json.loads(str(data["frame_visibility_json"].item())),
                neighbor_atom_ids_json=json.loads(str(data["neighbor_atom_ids_json"].item())),
                meta=json.loads(str(data["meta_json"].item())),
            )


def _safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.int64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    entropy = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    return float(entropy / max(np.log2(max(counts.shape[0], 2)), 1e-12))


def _surfel_descriptors(bank: MeasurementBank, boundary_safe_px: float) -> dict[str, np.ndarray]:
    visible = np.asarray(bank.visible_ok, dtype=bool)
    uv = np.asarray(bank.uv_pred, dtype=np.float32)
    n_frames, n_surfels = visible.shape
    mean_uv = np.zeros((n_surfels, 2), dtype=np.float32)
    var_uv = np.zeros((n_surfels,), dtype=np.float32)
    visible_count = visible.sum(axis=0).astype(np.int64)
    for surfel_idx in range(n_surfels):
        ok = visible[:, surfel_idx]
        if np.any(ok):
            pts = uv[ok, surfel_idx]
            mean_uv[surfel_idx] = np.mean(pts, axis=0)
            var_uv[surfel_idx] = float(np.mean(np.var(pts, axis=0)))
    frame_frac = visible_count.astype(np.float32) / float(max(n_frames, 1))
    negative_ratio = np.asarray(bank.negative_observation, dtype=bool).sum(axis=0).astype(np.float32) / np.maximum(
        visible_count.astype(np.float32),
        1.0,
    )
    positive_ratio = np.asarray(bank.positive_observation, dtype=bool).sum(axis=0).astype(np.float32) / np.maximum(
        visible_count.astype(np.float32),
        1.0,
    )
    boundary_safe = np.asarray(bank.source_boundary_distance, dtype=np.float32) >= float(boundary_safe_px)
    entropy = np.zeros((n_surfels,), dtype=np.float32)
    target_mask_id = np.asarray(bank.target_mask_id, dtype=np.int64)
    for surfel_idx in range(n_surfels):
        entropy[surfel_idx] = _safe_entropy(target_mask_id[visible[:, surfel_idx], surfel_idx])
    return {
        "mean_uv": mean_uv,
        "var_uv": var_uv,
        "frame_frac": frame_frac,
        "visible_count": visible_count,
        "negative_ratio": negative_ratio,
        "positive_ratio": positive_ratio,
        "boundary_safe": boundary_safe,
        "mask_entropy": entropy,
    }


def _quantize(values: np.ndarray, bins: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    clipped = np.clip((values - low) / max(high - low, 1e-12), 0.0, 0.999999)
    return np.floor(clipped * int(bins)).astype(np.int64)


def _atom_key_for_surfel(
    bank: MeasurementBank,
    desc: dict[str, np.ndarray],
    base_frame: np.ndarray,
    base_mask: np.ndarray,
    idx: int,
    *,
    variant: str,
    trajectory_bins: int,
    rgb_bins: int,
) -> tuple[Any, ...]:
    src_frame = int(base_frame[idx])
    src_mask = int(base_mask[idx])
    if src_mask <= 0:
        base: tuple[Any, ...] = ("unknown", src_frame)
    else:
        base = ("mask", src_frame, src_mask)
    if variant == "A0":
        return base

    uv_bin = _quantize(desc["mean_uv"][idx], trajectory_bins)
    vis_bin = int(min(3, np.floor(float(desc["frame_frac"][idx]) * 4.0)))
    key = (*base, "traj", int(uv_bin[0]), int(uv_bin[1]), vis_bin)
    if variant == "A1":
        return key

    rgb_bin = _quantize(np.asarray(bank.src_rgb[idx], dtype=np.float32), rgb_bins)
    key = (*key, "rgb", int(rgb_bin[0]), int(rgb_bin[1]), int(rgb_bin[2]))
    if variant == "A2":
        return key

    boundary_bin = int(bool(desc["boundary_safe"][idx]))
    neg_bin = int(float(desc["negative_ratio"][idx]) >= 0.25)
    entropy_bin = int(float(desc["mask_entropy"][idx]) >= 0.45)
    return (*key, "risk", boundary_bin, neg_bin, entropy_bin)


def _coarse_a4_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(key) <= 10:
        return key
    # Drop the final boundary/negative/entropy subdivision and keep source,
    # trajectory and appearance bins for conservative small-atom merging.
    if "risk" in key:
        risk_idx = key.index("risk")
        return key[:risk_idx]
    return key


def _group_surfels(
    bank: MeasurementBank,
    desc: dict[str, np.ndarray],
    base_frame: np.ndarray,
    base_mask: np.ndarray,
    *,
    variant: str,
    trajectory_bins: int,
    rgb_bins: int,
    min_surfels: int,
    merge_small_surfels: int,
) -> list[np.ndarray]:
    raw: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx in range(bank.num_surfels):
        key_variant = "A3" if variant == "A4" else variant
        key = _atom_key_for_surfel(
            bank,
            desc,
            base_frame,
            base_mask,
            idx,
            variant=key_variant,
            trajectory_bins=trajectory_bins,
            rgb_bins=rgb_bins,
        )
        raw[key].append(idx)

    if variant == "A4":
        merged: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
        stable: list[np.ndarray] = []
        for key, values in raw.items():
            if len(values) < int(merge_small_surfels):
                merged[_coarse_a4_key(key)].extend(values)
            else:
                stable.append(np.asarray(values, dtype=np.int64))
        groups = stable + [np.asarray(values, dtype=np.int64) for values in merged.values()]
    else:
        groups = [np.asarray(values, dtype=np.int64) for values in raw.values()]

    return [group for group in groups if int(group.shape[0]) >= int(min_surfels)]


def _base_arrays(bank: MeasurementBank, base_mode: str) -> tuple[np.ndarray, np.ndarray]:
    base_frame = np.asarray(bank.src_frame_global, dtype=np.int64).copy()
    base_mask = np.asarray(bank.src_mask_id, dtype=np.int64).copy()
    if base_mode == "source":
        return base_frame, base_mask
    if base_mode not in {"source_or_target", "target_dominant"}:
        raise ValueError(f"Unsupported atom base_mode: {base_mode}")

    target_mask_id = np.asarray(bank.target_mask_id, dtype=np.int64)
    positive = np.asarray(bank.positive_observation, dtype=bool) & (target_mask_id > 0)
    frame_ids = np.asarray(bank.frame_ids, dtype=np.int64)
    for surfel_idx in range(bank.num_surfels):
        if base_mode == "source_or_target" and int(base_mask[surfel_idx]) > 0:
            continue
        local_frames = np.flatnonzero(positive[:, surfel_idx])
        if local_frames.size == 0:
            continue
        # Prefer the first positive frame for deterministic target-mask birth.
        # Mask ids are frame-local, so the frame id is part of the atom base.
        local = int(local_frames[0])
        base_frame[surfel_idx] = int(frame_ids[local])
        base_mask[surfel_idx] = int(target_mask_id[local, surfel_idx])
    return base_frame, base_mask


def _histogram(values: np.ndarray) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in Counter(int(x) for x in values.tolist() if int(x) > 0).items()}


def _mask_votes(bank: MeasurementBank, surfels: np.ndarray, max_votes: int) -> list[tuple[int, int, float]]:
    votes: dict[tuple[int, int], float] = {}
    frame_ids = np.asarray(bank.frame_ids, dtype=np.int64)
    target_mask_id = np.asarray(bank.target_mask_id, dtype=np.int64)
    positive = np.asarray(bank.positive_observation, dtype=bool)
    for frame_idx, frame_id in enumerate(frame_ids.tolist()):
        ids = target_mask_id[frame_idx, surfels]
        ids = ids[positive[frame_idx, surfels] & (ids > 0)]
        if ids.size == 0:
            continue
        for mask_id, count in Counter(int(v) for v in ids.tolist()).most_common(2):
            key = (int(frame_id), int(mask_id))
            votes[key] = max(votes.get(key, 0.0), float(count))
    src_frames = np.asarray(bank.src_frame_global[surfels], dtype=np.int64)
    src_masks = np.asarray(bank.src_mask_id[surfels], dtype=np.int64)
    for (frame_id, mask_id), count in Counter(zip(src_frames.tolist(), src_masks.tolist())).items():
        if int(mask_id) > 0:
            key = (int(frame_id), int(mask_id))
            votes[key] = max(votes.get(key, 0.0), float(count))
    out = [(frame, mask, score) for (frame, mask), score in votes.items()]
    out.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))
    return out[: int(max_votes)]


def _neighbors_for_atoms(source_frame: np.ndarray, source_mask: np.ndarray, mean_uv: np.ndarray, max_neighbors: int) -> list[list[int]]:
    by_key: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (frame, mask) in enumerate(zip(source_frame.tolist(), source_mask.tolist())):
        by_key[(int(frame), int(mask))].append(idx)
    neighbors: list[list[int]] = [[] for _ in range(source_frame.shape[0])]
    for indices in by_key.values():
        if len(indices) <= 1:
            continue
        coords = mean_uv[np.asarray(indices, dtype=np.int64)]
        for local_idx, atom_idx in enumerate(indices):
            dist = np.linalg.norm(coords - coords[local_idx], axis=1)
            order = np.argsort(dist, kind="mergesort").tolist()
            neighbors[atom_idx] = [int(indices[o]) for o in order if indices[o] != atom_idx][: int(max_neighbors)]
    return neighbors


def build_surfel_atom_bank(
    bank: MeasurementBank,
    *,
    variant: str,
    base_mode: str = "source",
    min_surfels: int = 4,
    merge_small_surfels: int = 12,
    boundary_safe_px: float = 3.0,
    trajectory_bins: int = 4,
    rgb_bins: int = 4,
    max_mask_votes: int = 8,
    max_neighbors: int = 8,
) -> tuple[SurfelAtomBank, dict[str, Any]]:
    if variant not in {"A0", "A1", "A2", "A3", "A4"}:
        raise ValueError(f"Unsupported atom variant: {variant}")
    desc = _surfel_descriptors(bank, boundary_safe_px=boundary_safe_px)
    base_frame, base_mask = _base_arrays(bank, base_mode=base_mode)
    groups = _group_surfels(
        bank,
        desc,
        base_frame,
        base_mask,
        variant=variant,
        trajectory_bins=int(trajectory_bins),
        rgb_bins=int(rgb_bins),
        min_surfels=int(min_surfels),
        merge_small_surfels=int(merge_small_surfels),
    )
    groups.sort(key=lambda arr: (int(base_frame[arr[0]]), int(base_mask[arr[0]]), int(arr[0])))

    atom_ids: list[int] = []
    offsets = [0]
    flat_surfels: list[int] = []
    source_frame: list[int] = []
    source_mask_id: list[int] = []
    atom_size: list[int] = []
    mean_rgb: list[np.ndarray] = []
    trajectory_descriptor: list[np.ndarray] = []
    boundary_safe_ratio: list[float] = []
    negative_visible_outside_ratio: list[float] = []
    mask_entropy: list[float] = []
    trajectory_variance: list[float] = []
    is_unknown: list[bool] = []
    mask_membership_json: list[dict[str, int]] = []
    frame_visibility_json: list[dict[str, int]] = []
    atom_mean_uv: list[np.ndarray] = []

    for atom_id, surfels in enumerate(groups):
        surfels = np.asarray(sorted(int(v) for v in surfels.tolist()), dtype=np.int64)
        src_frame_counts = Counter(int(v) for v in base_frame[surfels].tolist())
        src_mask_counts = Counter(int(v) for v in base_mask[surfels].tolist())
        frame = src_frame_counts.most_common(1)[0][0] if src_frame_counts else -1
        mask = src_mask_counts.most_common(1)[0][0] if src_mask_counts else -1
        atom_ids.append(int(atom_id))
        flat_surfels.extend(int(v) for v in surfels.tolist())
        offsets.append(len(flat_surfels))
        source_frame.append(int(frame))
        source_mask_id.append(int(mask))
        atom_size.append(int(surfels.shape[0]))
        mean_rgb.append(np.mean(np.asarray(bank.src_rgb[surfels], dtype=np.float32), axis=0))
        mean_uv = np.mean(desc["mean_uv"][surfels], axis=0)
        atom_mean_uv.append(mean_uv)
        trajectory_descriptor.append(
            np.asarray(
                [
                    float(mean_uv[0]),
                    float(mean_uv[1]),
                    float(np.mean(desc["frame_frac"][surfels])),
                    float(np.mean(desc["positive_ratio"][surfels])),
                ],
                dtype=np.float32,
            )
        )
        boundary_safe_ratio.append(float(np.mean(desc["boundary_safe"][surfels])))
        negative_visible_outside_ratio.append(float(np.mean(desc["negative_ratio"][surfels])))
        mask_entropy.append(float(np.mean(desc["mask_entropy"][surfels])))
        trajectory_variance.append(float(np.mean(desc["var_uv"][surfels])))
        is_unknown.append(int(mask) <= 0)
        mask_membership_json.append(_histogram(np.asarray(bank.src_mask_id[surfels], dtype=np.int64)))
        visible_frames = {
            str(int(bank.frame_ids[frame_idx])): int(count)
            for frame_idx, count in enumerate(np.asarray(bank.visible_ok[:, surfels], dtype=bool).sum(axis=1).tolist())
            if int(count) > 0
        }
        frame_visibility_json.append(visible_frames)

    neighbor_atom_ids_json = _neighbors_for_atoms(
        np.asarray(source_frame, dtype=np.int64),
        np.asarray(source_mask_id, dtype=np.int64),
        np.asarray(atom_mean_uv, dtype=np.float32) if atom_mean_uv else np.zeros((0, 2), dtype=np.float32),
        max_neighbors=int(max_neighbors),
    )
    bank_out = SurfelAtomBank(
        scene=bank.scene,
        variant=variant,
        atom_ids=np.asarray(atom_ids, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        surfel_indices=np.asarray(flat_surfels, dtype=np.int64),
        source_frame=np.asarray(source_frame, dtype=np.int64),
        source_mask_id=np.asarray(source_mask_id, dtype=np.int64),
        atom_size=np.asarray(atom_size, dtype=np.int64),
        mean_rgb=np.asarray(mean_rgb, dtype=np.float32) if mean_rgb else np.zeros((0, 3), dtype=np.float32),
        trajectory_descriptor=np.asarray(trajectory_descriptor, dtype=np.float32)
        if trajectory_descriptor
        else np.zeros((0, 4), dtype=np.float32),
        boundary_safe_ratio=np.asarray(boundary_safe_ratio, dtype=np.float32),
        negative_visible_outside_ratio=np.asarray(negative_visible_outside_ratio, dtype=np.float32),
        mask_entropy=np.asarray(mask_entropy, dtype=np.float32),
        trajectory_variance=np.asarray(trajectory_variance, dtype=np.float32),
        is_unknown=np.asarray(is_unknown, dtype=bool),
        mask_membership_json=mask_membership_json,
        frame_visibility_json=frame_visibility_json,
        neighbor_atom_ids_json=neighbor_atom_ids_json,
        meta={
            "algorithm": "v14_surfel_atom_bank",
            "variant": variant,
            "base_mode": base_mode,
            "min_surfels": int(min_surfels),
            "merge_small_surfels": int(merge_small_surfels),
            "boundary_safe_px": float(boundary_safe_px),
            "trajectory_bins": int(trajectory_bins),
            "rgb_bins": int(rgb_bins),
            "max_mask_votes": int(max_mask_votes),
            "max_neighbors": int(max_neighbors),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
        },
    )
    diag = atom_bank_summary(bank_out, bank)
    return bank_out, diag


def atom_bank_summary(atom_bank: SurfelAtomBank, measurement_bank: MeasurementBank | None = None) -> dict[str, Any]:
    sizes = atom_bank.atom_size.astype(np.float64)
    known = ~atom_bank.is_unknown.astype(bool)
    num_surfels = int(measurement_bank.num_surfels) if measurement_bank is not None else int(atom_bank.surfel_indices.shape[0])
    return {
        "scene": atom_bank.scene,
        "variant": atom_bank.variant,
        "num_atoms": int(atom_bank.num_atoms),
        "num_known_atoms": int(np.count_nonzero(known)),
        "num_unknown_atoms": int(np.count_nonzero(~known)),
        "mean_surfels_per_atom": float(np.mean(sizes)) if sizes.size else 0.0,
        "median_surfels_per_atom": float(np.median(sizes)) if sizes.size else 0.0,
        "atom_support_pre_ratio": float(atom_bank.surfel_indices.shape[0] / max(num_surfels, 1)),
        "known_atom_support_ratio": float(np.sum(atom_bank.atom_size[known]) / max(num_surfels, 1)) if sizes.size else 0.0,
        "mask_entropy_mean": float(np.mean(atom_bank.mask_entropy)) if atom_bank.mask_entropy.size else 0.0,
        "mask_entropy_p90": float(np.percentile(atom_bank.mask_entropy, 90)) if atom_bank.mask_entropy.size else 0.0,
        "trajectory_variance_mean": float(np.mean(atom_bank.trajectory_variance)) if atom_bank.trajectory_variance.size else 0.0,
        "trajectory_variance_p90": float(np.percentile(atom_bank.trajectory_variance, 90)) if atom_bank.trajectory_variance.size else 0.0,
        "boundary_safe_ratio_mean": float(np.mean(atom_bank.boundary_safe_ratio)) if atom_bank.boundary_safe_ratio.size else 0.0,
        "negative_visible_outside_ratio_mean": float(np.mean(atom_bank.negative_visible_outside_ratio))
        if atom_bank.negative_visible_outside_ratio.size
        else 0.0,
    }


def atom_to_object_record(
    atom_bank: SurfelAtomBank,
    bank: MeasurementBank,
    atom_index: int,
    *,
    max_mask_votes: int = 8,
    fringe_from_neighbors: bool = False,
) -> dict[str, Any]:
    surfels = atom_bank.surfels_for_atom(atom_index)
    fringe = np.empty((0,), dtype=np.int64)
    if fringe_from_neighbors:
        neighbor_ids = atom_bank.neighbor_atom_ids_json[atom_index][:2]
        parts = [atom_bank.surfels_for_atom(int(idx)) for idx in neighbor_ids]
        if parts:
            fringe = np.concatenate(parts).astype(np.int64, copy=False)
    return {
        "mask_list": _mask_votes(bank, surfels, max_votes=int(max_mask_votes)),
        "carrier_ids": surfels.astype(np.int64, copy=False),
        "core_surfels": surfels.astype(np.int64, copy=False),
        "fringe_surfels": fringe,
        "unknown_surfels": np.empty((0,), dtype=np.int64),
        "reject_surfels": np.empty((0,), dtype=np.int64),
        "v14_atom": {
            "atom_id": int(atom_bank.atom_ids[atom_index]),
            "variant": atom_bank.variant,
            "source_frame": int(atom_bank.source_frame[atom_index]),
            "source_mask_id": int(atom_bank.source_mask_id[atom_index]),
            "atom_size": int(atom_bank.atom_size[atom_index]),
            "mask_entropy": float(atom_bank.mask_entropy[atom_index]),
            "trajectory_variance": float(atom_bank.trajectory_variance[atom_index]),
            "boundary_safe_ratio": float(atom_bank.boundary_safe_ratio[atom_index]),
            "negative_visible_outside_ratio": float(atom_bank.negative_visible_outside_ratio[atom_index]),
        },
    }
