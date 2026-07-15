#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROLE_TO_ID = {
    "dynamic": 0,
    "boundary_lowpurity": 1,
    "weak_context": 2,
    "stable_landmark": 3,
    "vegetation_repetitive": 4,
    "sky_lowobs": 5,
    "unknown_lowtrust": 6,
}

LABEL_TO_ROLE = {
    "void": "unknown_lowtrust",
    "parasol_or_umbrella": "weak_context",
    "roadblock": "weak_context",
    "bus": "dynamic",
    "truck": "dynamic",
    "bicycle": "dynamic",
    "motorcycle": "dynamic",
    "person": "dynamic",
    "bench": "stable_landmark",
    "flower_pot_or_vase": "weak_context",
    "handrail_or_fence": "stable_landmark",
    "wall": "stable_landmark",
    "pillar": "stable_landmark",
    "pole": "stable_landmark",
    "ground": "weak_context",
    "grass": "vegetation_repetitive",
    "road": "weak_context",
    "path": "weak_context",
    "crosswalk": "weak_context",
    "building": "stable_landmark",
    "house": "stable_landmark",
    "bridge": "stable_landmark",
    "other_construction": "stable_landmark",
    "sky": "sky_lowobs",
    "mountain": "stable_landmark",
    "billboard_or_bulletin_board": "stable_landmark",
    "wheeled_machine": "dynamic",
    "other_machine": "dynamic",
    "tree": "vegetation_repetitive",
    "stone": "stable_landmark",
    "flower": "vegetation_repetitive",
    "other_plant": "vegetation_repetitive",
    "trash_can": "weak_context",
    "car": "dynamic",
    "traffic sign": "stable_landmark",
    "stair": "stable_landmark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project existing KITTI dense semantic maps to HorizonStream runtime patch tokens."
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--kitti-preprocess-root", default="results/kitti_preprocess")
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--boundary-purity-threshold", type=float, default=0.60)
    parser.add_argument("--trust-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--write-token-csv", action="store_true")
    return parser.parse_args()


def resize_crop_like_horizonstream(arr: np.ndarray, size: int, patch_size: int, *, is_label: bool) -> np.ndarray:
    """Mirror HorizonStream's load_images_for_eval crop path for already max-side-resized semantic maps."""
    image = Image.fromarray(arr)
    w1, h1 = image.size
    long_edge = size
    scale = float(long_edge) / float(max(w1, h1))
    new_size = tuple(int(round(x * scale)) for x in image.size)
    interp = Image.Resampling.NEAREST if is_label else Image.Resampling.BILINEAR
    image = image.resize(new_size, interp)

    w, h = image.size
    cx, cy = w // 2, h // 2
    halfw = ((2 * cx) // patch_size) * (patch_size // 2)
    halfh = ((2 * cy) // patch_size) * (patch_size // 2)
    if w == h:
        halfh = int(3 * halfw / 4)
    image = image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return np.asarray(image)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def iter_cache_entries(cache_dir: Path) -> list[dict[str, Any]]:
    entries = []
    with (cache_dir / "cache_index.jsonl").open("r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return sorted(entries, key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))


def role_masses(label_patch: np.ndarray, label_names: list[str], total: int) -> dict[str, float]:
    counts = Counter(int(v) for v in label_patch.reshape(-1).tolist())
    masses = {role: 0.0 for role in ROLE_TO_ID}
    for label_id, count in counts.items():
        name = label_names[label_id] if 0 <= label_id < len(label_names) else "void"
        role = LABEL_TO_ROLE.get(name, "unknown_lowtrust")
        masses[role] += float(count) / float(total)
    return masses


def project_frame(
    label_map: np.ndarray,
    confidence_map: np.ndarray,
    label_names: list[str],
    patch_size: int,
    boundary_purity_threshold: float,
    trust_confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    h, w = label_map.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    patch_count = patch_h * patch_w
    role_ids = np.zeros((patch_count,), dtype=np.uint8)
    label_ids = np.zeros((patch_count,), dtype=np.uint8)
    risk = np.zeros((patch_count,), dtype=np.float32)
    stable = np.zeros((patch_count,), dtype=np.float32)
    token_rows: list[dict[str, Any]] = []

    role_mass_sums = {role: 0.0 for role in ROLE_TO_ID}
    nonvoid_pixel_count = 0
    low_purity_count = 0
    lowtrust_count = 0
    purity_sum = 0.0
    conf_sum = 0.0

    token_idx = 1
    patch_area = patch_size * patch_size
    for py in range(patch_h):
        y0 = py * patch_size
        y1 = y0 + patch_size
        for px in range(patch_w):
            x0 = px * patch_size
            x1 = x0 + patch_size
            patch = label_map[y0:y1, x0:x1]
            conf_patch = confidence_map[y0:y1, x0:x1]
            flat = patch.reshape(-1)
            counts = np.bincount(flat.astype(np.int64), minlength=len(label_names))
            dominant_label_id = int(counts.argmax())
            dominant_count = int(counts[dominant_label_id])
            nonvoid_count = int(patch_area - counts[0]) if len(counts) else patch_area
            nonvoid_ratio = float(nonvoid_count) / float(patch_area)
            purity = float(dominant_count) / float(patch_area)
            conf_mean = float(np.mean(conf_patch))
            label_name = label_names[dominant_label_id] if dominant_label_id < len(label_names) else "void"
            role = LABEL_TO_ROLE.get(label_name, "unknown_lowtrust")
            masses = role_masses(patch, label_names, patch_area)
            if purity < boundary_purity_threshold and nonvoid_ratio > 0.0:
                role = "boundary_lowpurity"
                low_purity_count += 1
            if conf_mean < trust_confidence_threshold or nonvoid_ratio <= 0.0:
                role = "unknown_lowtrust"
                lowtrust_count += 1

            dynamic_mass = masses["dynamic"]
            boundary_mass = masses["boundary_lowpurity"] + (1.0 - purity if nonvoid_ratio > 0.0 else 0.0)
            lowtrust_mass = masses["unknown_lowtrust"] + (1.0 - nonvoid_ratio)
            weak_mass = masses["weak_context"]
            stable_mass = masses["stable_landmark"]
            risk_value = 1.5 * dynamic_mass + 1.0 * boundary_mass + 0.8 * lowtrust_mass + 0.3 * weak_mass
            stable_value = stable_mass

            idx = py * patch_w + px
            role_ids[idx] = ROLE_TO_ID[role]
            label_ids[idx] = dominant_label_id
            risk[idx] = np.float32(risk_value)
            stable[idx] = np.float32(stable_value)

            nonvoid_pixel_count += nonvoid_count
            purity_sum += purity
            conf_sum += conf_mean
            for role_name, mass in masses.items():
                role_mass_sums[role_name] += mass

            token_rows.append(
                {
                    "token_idx": token_idx,
                    "patch_y": py,
                    "patch_x": px,
                    "label_id": dominant_label_id,
                    "role_id": ROLE_TO_ID[role],
                    "nonvoid_ratio": nonvoid_ratio,
                    "purity": purity,
                    "confidence_mean": conf_mean,
                    "risk": risk_value,
                    "stable": stable_value,
                }
            )
            token_idx += 1

    denom = float(max(patch_count, 1))
    frame_summary = {
        "patch_h": patch_h,
        "patch_w": patch_w,
        "patch_count": patch_count,
        "pixel_nonvoid_ratio": float(nonvoid_pixel_count) / float(max(patch_count * patch_area, 1)),
        "patch_nonvoid_ratio_any": float(np.mean([row["nonvoid_ratio"] > 0.0 for row in token_rows])),
        "patch_nonvoid_ratio_majority": float(np.mean([row["nonvoid_ratio"] >= 0.5 for row in token_rows])),
        "patch_purity_mean": purity_sum / denom,
        "semantic_trust_mean": conf_sum / denom,
        "boundary_lowpurity_patch_ratio": float(low_purity_count) / denom,
        "unknown_lowtrust_patch_ratio": float(lowtrust_count) / denom,
        "dynamic_mass_mean": role_mass_sums["dynamic"] / denom,
        "stable_mass_mean": role_mass_sums["stable_landmark"] / denom,
        "weak_context_mass_mean": role_mass_sums["weak_context"] / denom,
        "vegetation_mass_mean": role_mass_sums["vegetation_repetitive"] / denom,
        "sky_mass_mean": role_mass_sums["sky_lowobs"] / denom,
    }
    return role_ids, label_ids, risk, stable, token_rows, frame_summary


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    diag_dir = results_root / "diagnostics"
    sem_dir = results_root / "semantic_projection"
    audit_dir = results_root / "audit"
    diag_dir.mkdir(parents=True, exist_ok=True)
    sem_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    seqs = [item.strip() for item in args.seqs.split(",") if item.strip()]
    frame_rows_path = diag_dir / "hs_frame_semantic_rows.csv"
    token_rows_path = diag_dir / "hs_token_semantic_rows.csv"
    legacy_rows_path = audit_dir / "hs_semantic_projection_to_37x37_rows.csv"

    frame_fieldnames = [
        "seq",
        "frame_idx",
        "patch_h",
        "patch_w",
        "patch_count",
        "pixel_nonvoid_ratio",
        "patch_nonvoid_ratio_any",
        "patch_nonvoid_ratio_majority",
        "patch_purity_mean",
        "semantic_trust_mean",
        "dynamic_mass_mean",
        "stable_mass_mean",
        "weak_context_mass_mean",
        "vegetation_mass_mean",
        "sky_mass_mean",
        "boundary_lowpurity_patch_ratio",
        "unknown_lowtrust_patch_ratio",
    ]
    token_fieldnames = [
        "seq",
        "frame_idx",
        "token_idx",
        "patch_y",
        "patch_x",
        "label_id",
        "role_id",
        "nonvoid_ratio",
        "purity",
        "confidence_mean",
        "risk",
        "stable",
    ]

    summary: dict[str, Any] = {
        "input_resize": {"semantic_max_side_source": 720, "horizonstream_size": args.size},
        "patch_size": args.patch_size,
        "patch_grid": None,
        "special_tokens": {"mrt_token_count": 1, "register_token_count": 0, "pose_tokens_per_window_slot": 32},
        "patch_start_idx": 1,
        "token_to_pixel_mapping": {
            "token_idx_0": "MRT special token, no spatial patch",
            "token_idx_1": "patch_y=0, patch_x=0, pixels [0:14,0:14] in 154x518 runtime image",
            "formula": "token_idx = 1 + patch_y * patch_w + patch_x",
        },
        "role_to_id": ROLE_TO_ID,
        "label_to_role": LABEL_TO_ROLE,
        "seqs": {},
        "stage2_gate": {},
    }

    provenance: dict[str, Any] = {"seqs": {}, "notes": []}
    all_frame_stats = []
    observed_labels: set[str] = set()
    unmapped_labels: set[str] = set()

    with frame_rows_path.open("w", newline="") as frame_f, token_rows_path.open("w", newline="") as token_f:
        frame_writer = csv.DictWriter(frame_f, fieldnames=frame_fieldnames)
        token_writer = csv.DictWriter(token_f, fieldnames=token_fieldnames)
        frame_writer.writeheader()
        token_writer.writeheader()

        for seq in seqs:
            cache_dir = Path(args.kitti_preprocess_root) / seq / "stage_c_cache_semantic_chunks"
            conversion_summary = load_json(cache_dir / "conversion_summary.json")
            entries = iter_cache_entries(cache_dir)
            expected_frames = int(conversion_summary["num_frames"])
            seen = np.zeros((expected_frames,), dtype=bool)
            role_matrix = np.zeros((expected_frames, 0), dtype=np.uint8)
            label_matrix = np.zeros((expected_frames, 0), dtype=np.uint8)
            risk_matrix = np.zeros((expected_frames, 0), dtype=np.float32)
            stable_matrix = np.zeros((expected_frames, 0), dtype=np.float32)
            confidence_matrix = np.zeros((expected_frames, 0), dtype=np.float32)
            seq_frame_stats = []
            seq_label_names: list[str] | None = None
            seq_sources = set()

            for entry in entries:
                chunk_dir = cache_dir / entry["chunk"]
                obj = torch.load(chunk_dir / "masklet.pt", map_location="cpu")
                seg = obj["semantic_segmentation"]
                label_maps = seg["label_maps"].cpu().numpy()
                confidence_maps = seg["confidence_maps"].cpu().numpy()
                label_names = list(seg["label_names"])
                seq_label_names = label_names
                seq_sources.add(str(seg.get("source", "")))
                observed_labels.update(label_names)
                for label in label_names:
                    if label not in LABEL_TO_ROLE:
                        unmapped_labels.add(label)

                start_frame = int(entry["start_frame"])
                for local_idx in range(label_maps.shape[0]):
                    frame_idx = start_frame + local_idx
                    if frame_idx >= expected_frames or seen[frame_idx]:
                        continue
                    label_runtime = resize_crop_like_horizonstream(label_maps[local_idx], args.size, args.patch_size, is_label=True)
                    conf_runtime = resize_crop_like_horizonstream(confidence_maps[local_idx].astype(np.float32), args.size, args.patch_size, is_label=False)
                    role_ids, label_ids, risk, stable, token_rows, frame_summary = project_frame(
                        label_runtime,
                        conf_runtime,
                        label_names,
                        args.patch_size,
                        args.boundary_purity_threshold,
                        args.trust_confidence_threshold,
                    )
                    if role_matrix.shape[1] == 0:
                        patch_count = int(role_ids.shape[0])
                        role_matrix = np.zeros((expected_frames, patch_count), dtype=np.uint8)
                        label_matrix = np.zeros((expected_frames, patch_count), dtype=np.uint8)
                        risk_matrix = np.zeros((expected_frames, patch_count), dtype=np.float32)
                        stable_matrix = np.zeros((expected_frames, patch_count), dtype=np.float32)
                        confidence_matrix = np.zeros((expected_frames, patch_count), dtype=np.float32)
                        summary["patch_grid"] = [frame_summary["patch_h"], frame_summary["patch_w"]]

                    role_matrix[frame_idx] = role_ids
                    label_matrix[frame_idx] = label_ids
                    risk_matrix[frame_idx] = risk
                    stable_matrix[frame_idx] = stable
                    confidence_matrix[frame_idx] = np.array([row["confidence_mean"] for row in token_rows], dtype=np.float32)
                    seen[frame_idx] = True

                    frame_row = {"seq": seq, "frame_idx": frame_idx, **frame_summary}
                    frame_writer.writerow(frame_row)
                    seq_frame_stats.append(frame_summary)
                    all_frame_stats.append(frame_summary)

                    for row in token_rows:
                        out_row = {"seq": seq, "frame_idx": frame_idx, **row}
                        token_writer.writerow(out_row)

            coverage = float(seen.mean()) if expected_frames else 0.0
            np.save(sem_dir / f"seq{seq}_role_ids.npy", role_matrix)
            np.save(sem_dir / f"seq{seq}_label_ids.npy", label_matrix)
            np.save(sem_dir / f"seq{seq}_risk.npy", risk_matrix)
            np.save(sem_dir / f"seq{seq}_stable.npy", stable_matrix)
            np.save(sem_dir / f"seq{seq}_confidence.npy", confidence_matrix)
            seq_stats = {
                "expected_frames": expected_frames,
                "covered_frames": int(seen.sum()),
                "frame_semantic_coverage": coverage,
                "patch_nonvoid_ratio_any_mean": float(np.mean([x["patch_nonvoid_ratio_any"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "patch_nonvoid_ratio_majority_mean": float(np.mean([x["patch_nonvoid_ratio_majority"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "patch_purity_mean": float(np.mean([x["patch_purity_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "semantic_trust_mean": float(np.mean([x["semantic_trust_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "dynamic_mass_mean": float(np.mean([x["dynamic_mass_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "stable_mass_mean": float(np.mean([x["stable_mass_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "weak_context_mass_mean": float(np.mean([x["weak_context_mass_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "sky_mass_mean": float(np.mean([x["sky_mass_mean"] for x in seq_frame_stats])) if seq_frame_stats else math.nan,
                "runtime_arrays": {
                    "role_ids": str(sem_dir / f"seq{seq}_role_ids.npy"),
                    "label_ids": str(sem_dir / f"seq{seq}_label_ids.npy"),
                    "risk": str(sem_dir / f"seq{seq}_risk.npy"),
                    "stable": str(sem_dir / f"seq{seq}_stable.npy"),
                    "confidence": str(sem_dir / f"seq{seq}_confidence.npy"),
                },
                "label_names": seq_label_names,
                "semantic_sources": sorted(seq_sources),
            }
            summary["seqs"][seq] = seq_stats
            provenance["seqs"][seq] = {
                "cache_dir": str(cache_dir),
                "conversion_summary": conversion_summary,
                "semantic_sources": sorted(seq_sources),
                "label_names": seq_label_names,
            }

    mapped_observed = sorted(label for label in observed_labels if label in LABEL_TO_ROLE)
    summary["role_mapping_coverage"] = float(len(mapped_observed) / max(len(observed_labels), 1))
    summary["unmapped_observed_labels"] = sorted(unmapped_labels)
    summary["semantic_projection_coverage"] = float(np.mean([v["frame_semantic_coverage"] for v in summary["seqs"].values()])) if summary["seqs"] else 0.0
    summary["patch_nonvoid_ratio"] = float(np.mean([x["patch_nonvoid_ratio_any"] for x in all_frame_stats])) if all_frame_stats else 0.0
    summary["patch_nonvoid_ratio_majority"] = float(np.mean([x["patch_nonvoid_ratio_majority"] for x in all_frame_stats])) if all_frame_stats else 0.0
    summary["patch_purity_mean"] = float(np.mean([x["patch_purity_mean"] for x in all_frame_stats])) if all_frame_stats else 0.0
    summary["semantic_trust_mean"] = float(np.mean([x["semantic_trust_mean"] for x in all_frame_stats])) if all_frame_stats else 0.0
    summary["stage2_gate"] = {
        "frame_semantic_coverage_ge_0_99": summary["semantic_projection_coverage"] >= 0.99,
        "patch_nonvoid_ratio_ge_0_95": summary["patch_nonvoid_ratio"] >= 0.95,
        "role_mapping_coverage_eq_1": summary["role_mapping_coverage"] == 1.0,
        "pass": summary["semantic_projection_coverage"] >= 0.99
        and summary["patch_nonvoid_ratio"] >= 0.95
        and summary["role_mapping_coverage"] == 1.0,
    }
    summary["outputs"] = {
        "hs_token_semantic_rows": str(token_rows_path),
        "hs_frame_semantic_rows": str(frame_rows_path),
        "hs_semantic_projection_summary": str(diag_dir / "hs_semantic_projection_summary.json"),
        "legacy_hs_semantic_projection_to_37x37_rows": str(legacy_rows_path),
        "semantic_projection_dir": str(sem_dir),
    }
    provenance["notes"].append(
        "Semantic source is the local KITTI preprocess dense semantic cache, not SemanticKITTI LiDAR/GT labels."
    )
    provenance["notes"].append(
        "The legacy hs_semantic_projection_to_37x37_rows.csv artifact is a hardlink/symlink/copy of hs_token_semantic_rows.csv; content records the actual 11x37 runtime grid."
    )

    with (diag_dir / "hs_semantic_projection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (audit_dir / "hs_semantic_source_provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    if legacy_rows_path.exists() or legacy_rows_path.is_symlink():
        legacy_rows_path.unlink()
    try:
        legacy_rows_path.hardlink_to(token_rows_path)
    except OSError:
        try:
            legacy_rows_path.symlink_to(token_rows_path.resolve())
        except OSError:
            import shutil

            shutil.copy2(token_rows_path, legacy_rows_path)

    print(json.dumps(summary["stage2_gate"], indent=2))
    print(f"wrote {diag_dir / 'hs_semantic_projection_summary.json'}")


if __name__ == "__main__":
    main()
