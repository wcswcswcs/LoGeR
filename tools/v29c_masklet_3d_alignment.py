#!/usr/bin/env python3
"""Align cached VideoMasklet masklets with v29C sparse SemanticKITTI projections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.gt_semantic_provider import SEMANTIC_KITTI_ID_TO_NAME  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


CHUNK_STARTS = {6: 174, 10: 290, 16: 464}
N_MIN = 50


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    if not str(text or "").strip():
        return list(default)
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _selected_frames(chunks: Sequence[int], horizons: Sequence[int]) -> set[int]:
    frames: set[int] = set()
    for chunk in chunks:
        start = CHUNK_STARTS[int(chunk)]
        for horizon in horizons:
            frames.update(range(start, start + 32 + int(horizon) * 29))
    return frames


def _video_group(label: str) -> str:
    label = canonicalize_label(str(label))
    if label in {"road", "sidewalk", "parking", "ground", "floor"}:
        return "GROUND_STRUCTURE"
    if label in {"building", "wall", "fence", "bridge", "railing"}:
        return "STRUCTURE_ANCHOR"
    if label in {"vegetation", "grass", "tree", "plant", "terrain", "trunk"}:
        return "VEGETATION_STUFF"
    if label in {"sky", "cloud"}:
        return "SKY_STUFF"
    if label in {"car", "truck", "bus", "bicycle", "motorcycle", "person", "rider"}:
        return "MOVABLE_THING"
    return "UNKNOWN"


def _projected_group(sem_id: int) -> str:
    sem_id = int(sem_id)
    if sem_id in {40, 44, 48, 49, 60}:
        return "GROUND_STRUCTURE"
    if sem_id in {50, 51, 52}:
        return "STRUCTURE_ANCHOR"
    if sem_id in {70, 71, 72}:
        return "VEGETATION_STUFF"
    if sem_id in {10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259}:
        return "MOVABLE_THING"
    return "UNKNOWN"


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=np.float64)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    return float(ent / max(math.log(len(counter)), 1e-12))


def _mean_temporal_iou(masks: torch.Tensor, visible: torch.Tensor) -> float:
    idx = [i for i, flag in enumerate(visible.tolist()) if bool(flag)]
    vals: List[float] = []
    for a, b in zip(idx[:-1], idx[1:]):
        ma = masks[a].bool()
        mb = masks[b].bool()
        inter = torch.logical_and(ma, mb).sum().item()
        union = torch.logical_or(ma, mb).sum().item()
        if union > 0:
            vals.append(float(inter / union))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _cache_chunks(cache_dir: Path, selected: set[int]) -> List[Path]:
    out = []
    for path in sorted(cache_dir.glob("chunk_*/masklet.pt")):
        manifest = path.with_name("manifest.json")
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            start = int(data.get("start_frame", -1))
            end = int(data.get("end_frame", -1))
        else:
            parts = path.parent.name.split("_")
            start = int(parts[-2])
            end = int(parts[-1])
        if any(start <= frame < end for frame in selected):
            out.append(path)
    return out


def _align_chunk(masklet_path: Path, projection_dir: Path, selected: set[int]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    data = torch.load(masklet_path, map_location="cpu")
    manifest = data.get("manifest") or json.loads(masklet_path.with_name("manifest.json").read_text(encoding="utf-8"))
    start = int(manifest["start_frame"])
    end = int(manifest["end_frame"])
    chunk_idx = int(manifest["chunk_idx"])
    labels = list(data.get("L_sem", []))
    masks = data["M_mask"].bool()
    visible = data["V_mask"].bool()
    qmask = data["Q_mask"].float()
    area = data.get("A_ratio", torch.zeros_like(qmask)).float()
    rows: List[Dict[str, object]] = []

    sem_by_local: Dict[int, np.ndarray] = {}
    valid_by_local: Dict[int, np.ndarray] = {}
    for local_t, frame in enumerate(range(start, end)):
        if frame not in selected:
            continue
        sem_path = projection_dir / f"{frame:06d}_sem_sparse.npy"
        valid_path = projection_dir / f"{frame:06d}_valid_mask.npy"
        if sem_path.exists() and valid_path.exists():
            sem_by_local[local_t] = np.load(sem_path, mmap_mode="r")
            valid_by_local[local_t] = np.load(valid_path, mmap_mode="r").astype(bool)

    for j in range(int(data.get("num_masklets", masks.shape[0]))):
        label = str(labels[j]) if j < len(labels) else "unknown"
        vgroup = _video_group(label)
        support = 0
        sem_counts: Counter[int] = Counter()
        depth_values = []
        frames_visible = 0
        frames_with_projection = 0
        for local_t, frame in enumerate(range(start, end)):
            if frame not in selected or not bool(visible[j, local_t]):
                continue
            frames_visible += 1
            if local_t not in sem_by_local:
                continue
            mask_np = masks[j, local_t].numpy().astype(bool)
            hit = np.logical_and(mask_np, valid_by_local[local_t])
            count = int(hit.sum())
            if count <= 0:
                continue
            frames_with_projection += 1
            support += count
            vals = sem_by_local[local_t][hit].astype(np.int64)
            sem_counts.update(int(x) for x in vals.tolist())
            depth_path = projection_dir / f"{frame:06d}_depth_sparse.npy"
            if depth_path.exists():
                depth_values.append(np.load(depth_path, mmap_mode="r")[hit].astype(np.float32))

        majority_id: Optional[int] = None
        majority_count = 0
        if sem_counts:
            majority_id, majority_count = sem_counts.most_common(1)[0]
        majority_name = SEMANTIC_KITTI_ID_TO_NAME.get(int(majority_id), "unknown") if majority_id is not None else "unknown"
        pgroup = _projected_group(int(majority_id)) if majority_id is not None else "UNKNOWN"
        entropy = _entropy(sem_counts)
        majority_ratio = float(majority_count / support) if support else 0.0
        supported = support >= N_MIN
        agreement: object
        if supported and vgroup != "SKY_STUFF" and pgroup != "UNKNOWN":
            agreement = bool(vgroup == pgroup)
        else:
            agreement = "unknown"
        q3d: object = ""
        if supported:
            q3d = float(min(1.0, support / N_MIN) * (1.0 - entropy))
        temporal = _mean_temporal_iou(masks[j], visible[j])
        qmask_mean = float(qmask[j][visible[j]].mean().item()) if bool(visible[j].any()) else 0.0
        if supported and isinstance(agreement, bool):
            tmask = max(0.0, min(1.0, 0.35 * qmask_mean + 0.25 * temporal + 0.25 * float(q3d) + 0.15 * float(agreement)))
        else:
            denom = 0.35 + 0.25
            tmask = max(0.0, min(1.0, (0.35 * qmask_mean + 0.25 * temporal) / denom if denom else 0.0))
        depths = np.concatenate(depth_values) if depth_values else np.array([], dtype=np.float32)
        rows.append(
            {
                "chunk_idx": chunk_idx,
                "chunk_start": start,
                "chunk_end": end,
                "masklet_id": j,
                "fine_label_pred": label,
                "video_group_pred": vgroup,
                "num_frames_visible": frames_visible,
                "frames_with_projection_support": frames_with_projection,
                "area_mean": float(area[j][visible[j]].mean().item()) if bool(visible[j].any()) else 0.0,
                "mask_temporal_iou_mean": temporal,
                "q_mask_mean": qmask_mean,
                "projected_pixel_support_count": support,
                "projected_majority_semantic_id": majority_id if majority_id is not None else "",
                "projected_majority_semantic_name": majority_name,
                "projected_majority_group": pgroup,
                "projected_majority_ratio": majority_ratio,
                "projected_entropy": entropy,
                "agreement_pred_vs_projected": agreement,
                "q_3d": q3d,
                "t_mask": tmask,
                "support_depth_mean": float(depths.mean()) if depths.size else "",
                "support_depth_p90": float(np.quantile(depths, 0.90)) if depths.size else "",
            }
        )

    summary = {
        "chunk_idx": chunk_idx,
        "chunk_start": start,
        "chunk_end": end,
        "num_masklets": len(rows),
        "supported_masklets": sum(int(int(r["projected_pixel_support_count"]) >= N_MIN) for r in rows),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-c-cache-dir", default="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full")
    parser.add_argument("--projection-cache-dir", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/projection_cache/seq01")
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    args = parser.parse_args()

    stage_cache = Path(args.stage_c_cache_dir).resolve()
    projection_dir = Path(args.projection_cache_dir).resolve()
    results = Path(args.results_root)
    if not results.is_absolute():
        results = REPO_ROOT / results
    out_dir = results / "masklet_3d_alignment"
    selected = _selected_frames(_parse_int_list(args.chunks, [6, 10, 16]), _parse_int_list(args.horizons, [10, 15]))
    masklet_paths = _cache_chunks(stage_cache, selected)

    all_rows: List[Dict[str, object]] = []
    chunk_rows: List[Dict[str, object]] = []
    for idx, path in enumerate(masklet_paths, start=1):
        rows, summary = _align_chunk(path, projection_dir, selected)
        all_rows.extend(rows)
        chunk_rows.append(summary)
        if idx % 5 == 0 or idx == len(masklet_paths):
            print(json.dumps({"aligned_chunks": idx, "total_chunks": len(masklet_paths), "masklets": len(all_rows)}), flush=True)

    supportable = [
        r for r in all_rows
        if str(r["video_group_pred"]) not in {"SKY_STUFF", "UNKNOWN"}
    ]
    supported = [r for r in supportable if int(r["projected_pixel_support_count"]) >= N_MIN]
    sg = [
        r for r in supported
        if str(r["video_group_pred"]) in {"GROUND_STRUCTURE", "STRUCTURE_ANCHOR"}
    ]
    sg_known = [r for r in sg if isinstance(r["agreement_pred_vs_projected"], bool)]
    support_ratio = len(supported) / max(1, len(supportable))
    agree_ratio = sum(int(bool(r["agreement_pred_vs_projected"])) for r in sg_known) / max(1, len(sg_known))
    gate = bool(support_ratio >= 0.30 and agree_ratio >= 0.70)

    _write_csv(out_dir / "masklet_alignment.csv", all_rows)
    _write_jsonl(out_dir / "masklet_alignment.jsonl", all_rows)
    _write_csv(out_dir / "per_chunk_alignment_summary.csv", chunk_rows)
    per_label: Dict[str, Dict[str, object]] = {}
    for label in sorted({str(r["fine_label_pred"]) for r in all_rows}):
        rows = [r for r in all_rows if str(r["fine_label_pred"]) == label]
        known = [r for r in rows if isinstance(r["agreement_pred_vs_projected"], bool)]
        per_label[label] = {
            "fine_label_pred": label,
            "masklets": len(rows),
            "supported_masklets": sum(int(int(r["projected_pixel_support_count"]) >= N_MIN) for r in rows),
            "mean_support": float(np.mean([int(r["projected_pixel_support_count"]) for r in rows])) if rows else 0.0,
            "known_agreement_ratio": sum(int(bool(r["agreement_pred_vs_projected"])) for r in known) / max(1, len(known)),
        }
    _write_csv(out_dir / "per_label_agreement_summary.csv", per_label.values())
    summary = {
        "phase": "v29c_phase2_masklet_3d_alignment",
        "stage_c_cache_dir": str(stage_cache),
        "projection_cache_dir": str(projection_dir),
        "selected_frame_count": len(selected),
        "chunks_aligned": len(masklet_paths),
        "masklets_total": len(all_rows),
        "supportable_non_sky_non_unknown_masklets": len(supportable),
        "supported_supportable_masklets": len(supported),
        "supported_supportable_ratio": support_ratio,
        "supported_structure_ground_masklets": len(sg_known),
        "supported_structure_ground_agreement_ratio": agree_ratio,
        "masklet_3d_alignment_gate_pass": gate,
        "n_min": N_MIN,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masklet_trust_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
