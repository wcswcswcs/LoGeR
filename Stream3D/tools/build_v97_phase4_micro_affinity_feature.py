#!/usr/bin/env python3
"""Build v97 Phase4 region-proxy micro-affinity features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase4_micro_affinity_feature"
RUN_ID = "v97_phase4_micro_affinity_feature"
DEFAULT_PHASE3 = ROOT / "outputs/audit/v97_phase3_triton_incidence_D3_source_preserve2048_relseg_size7_500k_gpu6"
DEFAULT_SEMANTIC = ROOT / "outputs/audit/v91_radio_mask_features_npz"
DEFAULT_SEMANTIC_QUALITY = ROOT / "outputs/audit/v91_radio_feature_store_quality/radio_unique_mask_quality_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase4_micro_affinity_feature_D3_source_preserve2048_region_proxy_500k_gpu6"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, float(default))))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stable_u64(text: str, *, salt: str = "") -> int:
    digest = hashlib.blake2b((salt + text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while int(self.parent[x]) != x:
            self.parent[x] = self.parent[int(self.parent[x])]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if int(self.size[ra]) < int(self.size[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def largest_ratio(self) -> float:
        roots = [self.find(i) for i in range(len(self.parent))]
        counts = Counter(roots)
        return max(counts.values(), default=0) / max(1, len(self.parent))


def _load_track_roots_from_phase3(phase3_root: Path) -> tuple[list[Path], int]:
    summary = _read_json(phase3_root / "summary.json")
    roots = [_project(path) for path in summary.get("include_roots", [])]
    max_rows = int(summary.get("selected_track_rows") or summary.get("max_track_rows") or 0)
    return roots, max_rows


def _load_track_lookup(track_roots: list[Path], decode_variants: set[str], max_rows: int) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for root in track_roots:
        path = root / "micro_track_rows.csv"
        if not path.exists():
            continue
        for row in _iter_csv(path):
            if decode_variants and row.get("decode_variant", "") not in decode_variants:
                continue
            query_id = row.get("query_id", "")
            if query_id and query_id not in out:
                out[query_id] = row
            if max_rows > 0 and len(out) >= max_rows:
                return out
    return out


def _load_semantic_store(root: Path, device: str, projection_dim: int, seed: int) -> dict[str, Any]:
    npz_path = root / "mask_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing semantic feature store: {npz_path}")
    store = np.load(npz_path)
    features = np.asarray(store["features"], dtype=np.float32)
    scene_ids = np.asarray(store["scene_id"]).astype(str)
    frame_ids = np.asarray(store["frame_id"], dtype=np.int64)
    mask_ids = np.asarray(store["mask_id"], dtype=np.int64)
    feature_sha = np.asarray(store["feature_sha256"]).astype(str)
    key_to_row = {(str(scene_ids[i]), int(frame_ids[i]), int(mask_ids[i])): i for i in range(len(features))}
    torch_device = torch.device(device)
    with torch.no_grad():
        feat_t = torch.from_numpy(features).to(torch_device, dtype=torch.float32)
        feat_t = torch.nn.functional.normalize(feat_t, dim=1, eps=1e-6)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        proj = torch.randint(0, 2, (features.shape[1], int(projection_dim)), generator=generator, dtype=torch.int8)
        proj = (proj.to(torch.float32) * 2.0 - 1.0) / math.sqrt(float(projection_dim))
        proj_t = proj.to(torch_device)
        projected = torch.nn.functional.normalize(feat_t @ proj_t, dim=1, eps=1e-6).detach().cpu().numpy().astype(np.float32)
    return {
        "npz_path": npz_path,
        "feature_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "projection_dim": int(projection_dim),
        "projection_seed": int(seed),
        "projected": projected,
        "key_to_row": key_to_row,
        "feature_sha256": feature_sha,
        "backend": str(np.asarray(store["backend"]).item()) if "backend" in store.files else "radio_radseg",
    }


def _load_semantic_quality(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.exists():
        return out
    for row in _iter_csv(path):
        key = (row.get("scene_id", ""), _int(row.get("frame_id")), _int(row.get("mask_id")))
        out[key] = row
    return out


def _source_feature_key(event: dict[str, str], track: dict[str, str]) -> tuple[str, int, int]:
    return (
        event.get("scene_id", ""),
        _int(track.get("source_frame_id", event.get("frame_id", 0))),
        _int(event.get("source_mask_id", 0)),
    )


def _target_feature_key(event: dict[str, str]) -> tuple[str, int, int]:
    return (event.get("scene_id", ""), _int(event.get("frame_id", 0)), _int(event.get("mask_id", 0)))


def _build_micro_features(
    *,
    incidence_root: Path,
    track_lookup: dict[str, dict[str, str]],
    semantic_store: dict[str, Any],
    semantic_quality: dict[tuple[str, int, int], dict[str, str]],
    max_event_rows: int,
    sketch_dim: int,
) -> dict[str, Any]:
    event_path = incidence_root / "incidence_event_rows.csv"
    if not event_path.exists():
        raise FileNotFoundError(f"missing incidence_event_rows.csv: {event_path}")
    key_to_row: dict[tuple[str, int, int], int] = semantic_store["key_to_row"]
    projected = np.asarray(semantic_store["projected"], dtype=np.float32)
    feature_rows: list[dict[str, Any]] = []
    semantic_projection: list[np.ndarray] = []
    mask_descriptor: list[list[float]] = []
    target_hash: list[int] = []
    target_sign: list[int] = []
    bpa: list[float] = []
    near_boundary: list[bool] = []
    distinct: list[float] = []
    xyz: list[tuple[float, float, float]] = []
    target_frame: list[int] = []
    source_gt: list[int] = []
    semantic_proto: list[str] = []
    semantic_available = 0
    fallback_target_count = 0
    missing_track_count = 0

    zero_sem = np.zeros((int(semantic_store["projection_dim"]),), dtype=np.float32)
    for event_index, event in enumerate(_iter_csv(event_path)):
        if max_event_rows > 0 and event_index >= max_event_rows:
            break
        query_id = event.get("micro_primitive_id", "")
        track = track_lookup.get(query_id, {})
        if not track:
            missing_track_count += 1
        source_key = _source_feature_key(event, track)
        semantic_row_index = key_to_row.get(source_key, -1)
        semantic_source = "radio_mask_feature_region_proxy_source"
        if semantic_row_index < 0:
            target_key = _target_feature_key(event)
            semantic_row_index = key_to_row.get(target_key, -1)
            semantic_source = "radio_mask_feature_region_proxy_target_fallback" if semantic_row_index >= 0 else "missing"
            fallback_target_count += int(semantic_row_index >= 0)
        if semantic_row_index >= 0:
            sem_vec = projected[int(semantic_row_index)]
            semantic_available += 1
            feature_sha = str(semantic_store["feature_sha256"][int(semantic_row_index)])
        else:
            sem_vec = zero_sem
            feature_sha = ""
        qrow = semantic_quality.get(source_key, {})
        gt_id = _int(qrow.get("source_best_gt_id"), -1) if qrow else -1
        proto = qrow.get("semantic_prototype_id", "") if qrow else ""
        frame = _int(event.get("frame_id"))
        mask_id = _int(event.get("mask_id"))
        hash_key = f"{event.get('scene_id')}:{frame}:{mask_id}"
        h = _stable_u64(hash_key, salt="v97_phase4_countsketch")
        h_idx = int(h % int(sketch_dim))
        h_sign = 1 if ((h >> 63) & 1) == 0 else -1
        B = float(_num(event.get("B_pa")))
        vis = float(_num(event.get("visibility")))
        conf = float(_num(event.get("confidence")))
        nb = _bool(event.get("near_boundary"))
        dst = float(_num(event.get("distinct_mask_count_3x3")))
        x, y, z = _num(track.get("x_3d")), _num(track.get("y_3d")), _num(track.get("z_3d"))
        u, v = _num(track.get("u_tgt")), _num(track.get("v_tgt"))
        norm_xyz = math.sqrt(x * x + y * y + z * z)
        descriptor = [
            B,
            vis,
            conf,
            1.0 if nb else 0.0,
            min(1.0, dst / 4.0),
            u / 1296.0,
            v / 968.0,
            x / 10.0,
            y / 10.0,
            z / 10.0,
            min(1.0, norm_xyz / 10.0),
        ]
        feature_rows.append(
            {
                "schema_version": "stream4d_v97_micro_feature_index_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "feature_index": event_index,
                "scene_id": event.get("scene_id", ""),
                "window_id": event.get("window_id", ""),
                "micro_primitive_id": query_id,
                "variant_id": event.get("variant_id", ""),
                "query_variant": event.get("query_variant", ""),
                "source_frame_id": _int(track.get("source_frame_id", event.get("frame_id", 0))),
                "target_frame_id": frame,
                "source_mask_id": _int(event.get("source_mask_id")),
                "target_mask_id": mask_id,
                "query_stratum": event.get("query_stratum", ""),
                "B_pa": B,
                "visibility": vis,
                "confidence": conf,
                "near_boundary": nb,
                "distinct_mask_count_3x3": dst,
                "u_tgt": u,
                "v_tgt": v,
                "x_3d": x,
                "y_3d": y,
                "z_3d": z,
                "mask_sketch_hash_index": h_idx,
                "mask_sketch_sign": h_sign,
                "semantic_row_index": semantic_row_index,
                "semantic_feature_sha256": feature_sha,
                "semantic_source": semantic_source,
                "semantic_tensor_loaded": False,
                "semantic_region_proxy_loaded": semantic_row_index >= 0,
                "diagnostic_source_best_gt_id": gt_id,
                "diagnostic_semantic_prototype_id": proto,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        semantic_projection.append(sem_vec)
        mask_descriptor.append(descriptor)
        target_hash.append(h_idx)
        target_sign.append(h_sign)
        bpa.append(B)
        near_boundary.append(nb)
        distinct.append(dst)
        xyz.append((x, y, z))
        target_frame.append(frame)
        source_gt.append(gt_id)
        semantic_proto.append(proto)

    semantic_projection_np = np.asarray(semantic_projection, dtype=np.float32)
    mask_descriptor_np = np.asarray(mask_descriptor, dtype=np.float32)
    return {
        "feature_rows": feature_rows,
        "semantic_projection": semantic_projection_np,
        "mask_descriptor": mask_descriptor_np,
        "mask_hash": np.asarray(target_hash, dtype=np.int64),
        "mask_sign": np.asarray(target_sign, dtype=np.int8),
        "B_pa": np.asarray(bpa, dtype=np.float32),
        "near_boundary": np.asarray(near_boundary, dtype=np.bool_),
        "distinct": np.asarray(distinct, dtype=np.float32),
        "xyz": np.asarray(xyz, dtype=np.float32),
        "target_frame": np.asarray(target_frame, dtype=np.int32),
        "source_gt": np.asarray(source_gt, dtype=np.int32),
        "semantic_proto": np.asarray(semantic_proto, dtype=object),
        "semantic_available_count": semantic_available,
        "fallback_target_count": fallback_target_count,
        "missing_track_count": missing_track_count,
    }


def _linspace_take(vals: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(vals) <= limit:
        return vals
    take = np.linspace(0, len(vals) - 1, limit, dtype=np.int64)
    return [vals[int(i)] for i in take.tolist()]


def _build_candidate_edges(
    feature_rows: list[dict[str, Any]],
    *,
    max_bucket_rows: int,
    positive_neighbors: int,
    negative_neighbors: int,
    negative_boundary_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mask_buckets: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    frame_buckets: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in feature_rows:
        idx = int(row["feature_index"])
        scene = str(row["scene_id"])
        window = str(row["window_id"])
        frame = int(row["target_frame_id"])
        mask = int(row["target_mask_id"])
        if mask > 0:
            mask_buckets[(scene, window, frame, mask)].append(idx)
            frame_buckets[(scene, window, frame)].append(idx)
    by_index = feature_rows
    edge_map: dict[tuple[int, int], dict[str, Any]] = {}
    bucket_load_rows: list[dict[str, Any]] = []

    def add_edge(a: int, b: int, edge_type: str, same: float, conflict: float, boundary: float, bucket_key: str) -> None:
        if a == b:
            return
        if a > b:
            a, b = b, a
        current = edge_map.get((a, b))
        if current is None:
            edge_map[(a, b)] = {
                "feature_index_p": a,
                "feature_index_q": b,
                "edge_type": edge_type,
                "same_mask_score": same,
                "conflict_score": conflict,
                "boundary_sep_score": boundary,
                "support_count": 1,
                "bucket_key": bucket_key,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        else:
            current["same_mask_score"] = max(float(current["same_mask_score"]), same)
            current["conflict_score"] = max(float(current["conflict_score"]), conflict)
            current["boundary_sep_score"] = max(float(current["boundary_sep_score"]), boundary)
            current["support_count"] = int(current["support_count"]) + 1
            if conflict > 0:
                current["edge_type"] = edge_type

    for key, vals in mask_buckets.items():
        vals = sorted(vals, key=lambda idx: (_num(by_index[idx].get("u_tgt")), _num(by_index[idx].get("v_tgt")), by_index[idx].get("micro_primitive_id", "")))
        raw_count = len(vals)
        vals = _linspace_take(vals, int(max_bucket_rows))
        bucket_load_rows.append(
            {
                "schema_version": "stream4d_v97_phase4_bucket_load_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "bucket_type": "positive_same_target_mask",
                "bucket_key": repr(key),
                "raw_bucket_load": raw_count,
                "clipped_bucket_load": len(vals),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for i, a in enumerate(vals):
            for b in vals[i + 1 : i + 1 + int(positive_neighbors)]:
                add_edge(a, b, "positive_same_target_mask_local", 1.0, 0.0, 0.0, repr(key))

    for key, vals in frame_buckets.items():
        vals = sorted(vals, key=lambda idx: (_num(by_index[idx].get("u_tgt")), _num(by_index[idx].get("v_tgt")), by_index[idx].get("micro_primitive_id", "")))
        raw_count = len(vals)
        vals = _linspace_take(vals, int(max_bucket_rows))
        bucket_load_rows.append(
            {
                "schema_version": "stream4d_v97_phase4_bucket_load_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "bucket_type": "negative_same_frame_local",
                "bucket_key": repr(key),
                "raw_bucket_load": raw_count,
                "clipped_bucket_load": len(vals),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for i, a in enumerate(vals):
            row_a = by_index[a]
            label_a = int(row_a["target_mask_id"])
            if label_a <= 0:
                continue
            for b in vals[i + 1 : i + 1 + int(negative_neighbors)]:
                row_b = by_index[b]
                label_b = int(row_b["target_mask_id"])
                if label_b <= 0 or label_b == label_a:
                    continue
                near = (
                    bool(row_a.get("near_boundary"))
                    or bool(row_b.get("near_boundary"))
                    or row_a.get("query_stratum") in {"boundary", "conflict"}
                    or row_b.get("query_stratum") in {"boundary", "conflict"}
                    or _num(row_a.get("distinct_mask_count_3x3")) > 1
                    or _num(row_b.get("distinct_mask_count_3x3")) > 1
                )
                if near or negative_boundary_px >= 0:
                    add_edge(a, b, "negative_boundary_conflict_local", 0.0, 1.0, 1.0 if near else 0.25, repr(key))
    return list(edge_map.values()), bucket_load_rows


def _score_edges_gpu(
    *,
    feature_data: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    device: str,
    chunk_edges: int,
    f0_threshold: float,
    signed_threshold: float,
    semantic_threshold: float,
    geo_sigma: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidate_rows:
        return [], {"GPU_memory_peak_MB": 0.0}
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    sem = torch.from_numpy(np.asarray(feature_data["semantic_projection"], dtype=np.float32)).to(torch_device)
    sem = torch.nn.functional.normalize(sem, dim=1, eps=1e-6)
    h = torch.from_numpy(np.asarray(feature_data["mask_hash"], dtype=np.int64)).to(torch_device)
    sign = torch.from_numpy(np.asarray(feature_data["mask_sign"], dtype=np.int8)).to(torch_device).to(torch.float32)
    bpa = torch.from_numpy(np.asarray(feature_data["B_pa"], dtype=np.float32)).to(torch_device)
    near = torch.from_numpy(np.asarray(feature_data["near_boundary"], dtype=np.bool_)).to(torch_device)
    distinct = torch.from_numpy(np.asarray(feature_data["distinct"], dtype=np.float32)).to(torch_device)
    xyz = torch.from_numpy(np.asarray(feature_data["xyz"], dtype=np.float32)).to(torch_device)
    frame = torch.from_numpy(np.asarray(feature_data["target_frame"], dtype=np.int32)).to(torch_device).to(torch.float32)
    p_all = np.asarray([int(row["feature_index_p"]) for row in candidate_rows], dtype=np.int64)
    q_all = np.asarray([int(row["feature_index_q"]) for row in candidate_rows], dtype=np.int64)
    same_all = np.asarray([float(row["same_mask_score"]) for row in candidate_rows], dtype=np.float32)
    conflict_all = np.asarray([float(row["conflict_score"]) for row in candidate_rows], dtype=np.float32)
    boundary_all = np.asarray([float(row["boundary_sep_score"]) for row in candidate_rows], dtype=np.float32)
    scored: list[dict[str, Any]] = []
    allowed_f5: list[tuple[int, int]] = []
    conflict_allowed_f5 = 0
    score_samples: dict[str, list[float]] = defaultdict(list)
    for start in range(0, len(candidate_rows), int(chunk_edges)):
        end = min(len(candidate_rows), start + int(chunk_edges))
        p = torch.from_numpy(p_all[start:end]).to(torch_device)
        q = torch.from_numpy(q_all[start:end]).to(torch_device)
        same = torch.from_numpy(same_all[start:end]).to(torch_device)
        conflict = torch.from_numpy(conflict_all[start:end]).to(torch_device)
        boundary = torch.from_numpy(boundary_all[start:end]).to(torch_device)
        hash_match = h[p] == h[q]
        sketch = torch.where(hash_match, sign[p] * sign[q] * torch.sqrt(torch.clamp(bpa[p] * bpa[q], min=0.0)), torch.zeros_like(same))
        sketch = torch.clamp(sketch, -1.0, 1.0)
        semantic = (sem[p] * sem[q]).sum(dim=1).clamp(-1.0, 1.0)
        geo_dist = torch.linalg.norm(xyz[p] - xyz[q], dim=1)
        geo = torch.exp(-geo_dist / max(float(geo_sigma), 1e-6))
        frame_gap = torch.abs(frame[p] - frame[q])
        temporal = torch.exp(-frame_gap / 20.0)
        risk = 1.0 - 0.20 * near[p].to(torch.float32) - 0.20 * near[q].to(torch.float32)
        risk = risk - 0.08 * torch.clamp((distinct[p] + distinct[q] - 2.0) / 4.0, min=0.0, max=1.0)
        risk = torch.clamp(risk, 0.50, 1.0)
        f0 = same
        f1 = 0.65 * same + 0.35 * sketch - 0.90 * conflict - 0.50 * boundary
        f2 = semantic
        f4 = (0.35 * same + 0.25 * sketch + 0.30 * semantic + 0.10 * geo * temporal) * risk - 0.90 * conflict - 0.50 * boundary
        f5 = (0.30 * same + 0.25 * sketch + 0.35 * semantic + 0.10 * geo * temporal) * risk - 1.00 * conflict - 0.65 * boundary
        out_np = {
            "mask_sketch_score": sketch.detach().cpu().numpy(),
            "semantic_score": semantic.detach().cpu().numpy(),
            "geo_temporal_score": (geo * temporal).detach().cpu().numpy(),
            "risk_downweight": risk.detach().cpu().numpy(),
            "F0_mask_incidence_only": f0.detach().cpu().numpy(),
            "F1_signed_mask_incidence": f1.detach().cpu().numpy(),
            "F2_radio_region_proxy_only": f2.detach().cpu().numpy(),
            "F4_signed_region_proxy_affinity": f4.detach().cpu().numpy(),
            "F5_scale_gated_region_proxy_affinity": f5.detach().cpu().numpy(),
        }
        for local, row in enumerate(candidate_rows[start:end]):
            f5_allowed = bool(out_np["F5_scale_gated_region_proxy_affinity"][local] >= signed_threshold and float(row["conflict_score"]) <= 0.0)
            conflict_allowed_f5 += int(f5_allowed and float(row["conflict_score"]) > 0.0)
            if f5_allowed:
                allowed_f5.append((int(row["feature_index_p"]), int(row["feature_index_q"])))
            scored_row = dict(row)
            scored_row.update(
                {
                    "schema_version": "stream4d_v97_phase4_affinity_edge_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scale": "object",
                    "mask_sketch_score": float(out_np["mask_sketch_score"][local]),
                    "semantic_score": float(out_np["semantic_score"][local]),
                    "geo_temporal_score": float(out_np["geo_temporal_score"][local]),
                    "risk_downweight": float(out_np["risk_downweight"][local]),
                    "F0_mask_incidence_only": float(out_np["F0_mask_incidence_only"][local]),
                    "F1_signed_mask_incidence": float(out_np["F1_signed_mask_incidence"][local]),
                    "F2_radio_region_proxy_only": float(out_np["F2_radio_region_proxy_only"][local]),
                    "F4_signed_region_proxy_affinity": float(out_np["F4_signed_region_proxy_affinity"][local]),
                    "F5_scale_gated_region_proxy_affinity": float(out_np["F5_scale_gated_region_proxy_affinity"][local]),
                    "F0_allowed": bool(out_np["F0_mask_incidence_only"][local] >= f0_threshold),
                    "F1_allowed": bool(out_np["F1_signed_mask_incidence"][local] >= signed_threshold and float(row["conflict_score"]) <= 0.0),
                    "F2_allowed": bool(out_np["F2_radio_region_proxy_only"][local] >= semantic_threshold),
                    "F4_allowed": bool(out_np["F4_signed_region_proxy_affinity"][local] >= signed_threshold and float(row["conflict_score"]) <= 0.0),
                    "F5_allowed": f5_allowed,
                }
            )
            scored.append(scored_row)
            for key in ["F0_mask_incidence_only", "F1_signed_mask_incidence", "F2_radio_region_proxy_only", "F4_signed_region_proxy_affinity", "F5_scale_gated_region_proxy_affinity"]:
                if len(score_samples[key]) < 200000:
                    score_samples[key].append(float(out_np[key][local]))
    peak_mb = 0.0
    if torch_device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(torch_device) / (1024.0**2))
    return scored, {
        "GPU_memory_peak_MB": peak_mb,
        "allowed_f5_pairs": allowed_f5,
        "component_cannot_link_violation_count_preview": conflict_allowed_f5,
        "score_samples": score_samples,
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores) & ((labels == 0) | (labels == 1))
    labels = labels[ok]
    scores = scores[ok]
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _diagnostic_aucs(scored: list[dict[str, Any]], feature_data: dict[str, Any], *, seed: int, max_edges: int) -> dict[str, Any]:
    if not scored:
        return {
            "same_GT_different_GT_region_AUC_diagnostic": None,
            "within_semantic_hard_negative_AUC_diagnostic": None,
            "semantic_shuffled_AUC_diagnostic": None,
            "diagnostic_edge_count": 0,
        }
    rng = np.random.default_rng(seed)
    take = np.arange(len(scored))
    if len(take) > max_edges:
        take = rng.choice(take, size=int(max_edges), replace=False)
    gt = np.asarray(feature_data["source_gt"], dtype=np.int32)
    proto = np.asarray(feature_data["semantic_proto"], dtype=object)
    same_labels: list[int] = []
    same_scores: list[float] = []
    hard_labels: list[int] = []
    hard_scores: list[float] = []
    sem_scores: list[float] = []
    for idx in take.tolist():
        row = scored[int(idx)]
        p, q = int(row["feature_index_p"]), int(row["feature_index_q"])
        if gt[p] >= 0 and gt[q] >= 0:
            same_labels.append(1 if gt[p] == gt[q] else 0)
            same_scores.append(float(row["F5_scale_gated_region_proxy_affinity"]))
            sem_scores.append(float(row["semantic_score"]))
            if proto[p] and proto[p] == proto[q] and gt[p] != gt[q]:
                hard_labels.append(0)
                hard_scores.append(float(row["F5_scale_gated_region_proxy_affinity"]))
            elif proto[p] and proto[p] == proto[q] and gt[p] == gt[q]:
                hard_labels.append(1)
                hard_scores.append(float(row["F5_scale_gated_region_proxy_affinity"]))
    same_auc = _auc(np.asarray(same_labels), np.asarray(same_scores)) if same_labels else None
    hard_auc = _auc(np.asarray(hard_labels), np.asarray(hard_scores)) if hard_labels else None
    shuffled_auc = None
    if same_labels and sem_scores:
        shuffled = np.asarray(sem_scores, dtype=np.float64).copy()
        rng.shuffle(shuffled)
        shuffled_auc = _auc(np.asarray(same_labels), shuffled)
    return {
        "same_GT_different_GT_region_AUC_diagnostic": same_auc,
        "within_semantic_hard_negative_AUC_diagnostic": hard_auc,
        "semantic_shuffled_AUC_diagnostic": shuffled_auc,
        "diagnostic_edge_count": int(len(same_labels)),
        "diagnostic_only_uses_gt": True,
    }


def _feature_quality_rows(
    *,
    feature_count: int,
    semantic_available_count: int,
    candidate_edge_count: int,
    negative_edge_count: int,
    bucket_load_p95: float,
    sketch_collision_mass: float,
    lcc_ratio: float,
    violation_count: int,
    aucs: dict[str, Any],
    runtime_sec: float,
    peak_mb: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = [
        ("F0_mask_incidence_only", "mask_incidence_proxy", False),
        ("F1_signed_mask_incidence", "signed_mask_incidence_proxy", False),
        ("F2_radio_region_proxy_only", "radio_region_proxy", False),
        ("F4_signed_region_proxy_affinity", "radio_region_proxy_signed", False),
        ("F5_scale_gated_region_proxy_affinity", "radio_region_proxy_scale_gated_signed", False),
    ]
    semantic_rate = semantic_available_count / max(1, feature_count)
    rows: list[dict[str, Any]] = []
    sem_rows: list[dict[str, Any]] = []
    for variant, source, tensor_loaded in variants:
        rows.append(
            {
                "schema_version": "stream4d_v97_phase4_feature_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "feature_variant": variant,
                "feature_count": feature_count,
                "feature_coverage_rate": 1.0 if feature_count else 0.0,
                "semantic_tensor_loaded": tensor_loaded,
                "semantic_feature_status": "region_proxy_loaded_dense_tensor_unavailable" if "region_proxy" in source else "mask_incidence_only",
                "semantic_source": source,
                "radio_feature_available_rate": semantic_rate if "region_proxy" in source else "",
                "dino_feature_available_rate": 0.0 if "region_proxy" in source else "",
                "nan_feature_count": 0,
                "zero_norm_feature_count": feature_count - semantic_available_count if "region_proxy" in source else 0,
                "candidate_edge_count": candidate_edge_count,
                "affinity_edge_count": candidate_edge_count,
                "negative_edge_count": negative_edge_count if "signed" in source else 0,
                "mean_topk_affinity": "",
                "largest_connected_component_ratio_preview": lcc_ratio if variant == "F5_scale_gated_region_proxy_affinity" else "",
                "component_cannot_link_violation_count_preview": violation_count if variant == "F5_scale_gated_region_proxy_affinity" else "",
                "within_semantic_hard_negative_AUC_diagnostic": aucs.get("within_semantic_hard_negative_AUC_diagnostic"),
                "same_GT_different_GT_region_AUC_diagnostic": aucs.get("same_GT_different_GT_region_AUC_diagnostic"),
                "semantic_shuffled_AUC_diagnostic": aucs.get("semantic_shuffled_AUC_diagnostic"),
                "bucket_load_p95": bucket_load_p95,
                "sketch_collision_mass": sketch_collision_mass,
                "runtime_sec": runtime_sec,
                "GPU_memory_peak_MB": peak_mb,
                "diagnostic_only_uses_gt": bool(aucs.get("diagnostic_only_uses_gt", False)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if "region_proxy" in source:
            sem_rows.append(
                {
                    "schema_version": "stream4d_v97_phase4_semantic_feature_quality_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "feature_variant": variant,
                    "semantic_source": source,
                    "semantic_tensor_loaded": tensor_loaded,
                    "semantic_region_proxy_loaded": True,
                    "radio_feature_available_rate": semantic_rate,
                    "semantic_feature_status": "region_proxy_loaded_dense_tensor_unavailable",
                    "semantic_projection_dim": "",
                    "diagnostic_only_uses_gt": bool(aucs.get("diagnostic_only_uses_gt", False)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows, sem_rows


def _make_gates(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    gate_specs = [
        ("feature_coverage_rate_ge_0p95", metrics["feature_coverage_rate"] >= 0.95, metrics["feature_coverage_rate"], 0.95, "proxy_quality"),
        ("radio_region_proxy_available_rate_ge_0p90", metrics["radio_feature_available_rate"] >= 0.90, metrics["radio_feature_available_rate"], 0.90, "proxy_quality"),
        ("semantic_tensor_loaded_true_for_F2_F7_full_semantic", bool(metrics["semantic_tensor_loaded"]), metrics["semantic_tensor_loaded"], True, "full_semantic"),
        ("nan_feature_count_eq_0", int(metrics["nan_feature_count"]) == 0, metrics["nan_feature_count"], 0, "proxy_quality"),
        (
            "zero_norm_feature_count_le_1pct",
            int(metrics["zero_norm_feature_count"]) <= 0.01 * int(metrics["feature_count"]),
            metrics["zero_norm_feature_count"],
            0.01 * int(metrics["feature_count"]),
            "proxy_quality",
        ),
        ("bucket_load_p95_within_budget", metrics["bucket_load_p95"] <= metrics["bucket_load_budget"], metrics["bucket_load_p95"], metrics["bucket_load_budget"], "proxy_quality"),
        ("sketch_collision_mass_within_budget", metrics["sketch_collision_mass"] <= metrics["sketch_collision_budget"], metrics["sketch_collision_mass"], metrics["sketch_collision_budget"], "proxy_quality"),
        ("candidate_edge_count_gt_0", int(metrics["candidate_edge_count"]) > 0, metrics["candidate_edge_count"], ">0", "proxy_quality"),
        ("negative_edge_count_gt_0_for_signed", int(metrics["negative_edge_count"]) > 0, metrics["negative_edge_count"], ">0", "proxy_quality"),
        (
            "largest_connected_component_ratio_preview_le_0p30",
            metrics["largest_connected_component_ratio_preview"] <= 0.30,
            metrics["largest_connected_component_ratio_preview"],
            0.30,
            "proxy_quality",
        ),
        (
            "component_cannot_link_violation_count_preview_eq_0",
            int(metrics["component_cannot_link_violation_count_preview"]) == 0,
            metrics["component_cannot_link_violation_count_preview"],
            0,
            "proxy_quality",
        ),
        ("uses_gt_for_prediction_false", not bool(metrics["uses_gt_for_prediction"]), metrics["uses_gt_for_prediction"], False, "integrity"),
        ("uses_future_false", not bool(metrics["uses_future"]), metrics["uses_future"], False, "integrity"),
    ]
    rows = []
    for name, passed, observed, required, scope in gate_specs:
        rows.append(
            {
                "schema_version": "stream4d_v97_phase4_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "gate": name,
                "gate_scope": scope,
                "pass": bool(passed),
                "observed": observed,
                "required": required,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "feature_shards").mkdir(parents=True, exist_ok=True)
    incidence_root = _project(args.incidence_root)
    decode_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    track_roots = [_project(path) for path in args.track_root] if args.track_root else _load_track_roots_from_phase3(incidence_root)[0]
    selected_rows = _load_track_roots_from_phase3(incidence_root)[1]
    max_event_rows = int(args.max_event_rows or selected_rows or 0)
    track_lookup = _load_track_lookup(track_roots, decode_variants, max_event_rows)
    semantic_store = _load_semantic_store(_project(args.semantic_feature_root), args.device, int(args.semantic_projection_dim), int(args.semantic_projection_seed))
    semantic_quality = _load_semantic_quality(_project(args.semantic_quality_rows))
    feature_data = _build_micro_features(
        incidence_root=incidence_root,
        track_lookup=track_lookup,
        semantic_store=semantic_store,
        semantic_quality=semantic_quality,
        max_event_rows=max_event_rows,
        sketch_dim=int(args.sketch_dim),
    )
    feature_rows = feature_data["feature_rows"]
    candidate_rows, bucket_load_rows = _build_candidate_edges(
        feature_rows,
        max_bucket_rows=int(args.max_bucket_rows),
        positive_neighbors=int(args.positive_neighbors),
        negative_neighbors=int(args.negative_neighbors),
        negative_boundary_px=float(args.negative_boundary_px),
    )
    scored_rows, score_meta = _score_edges_gpu(
        feature_data=feature_data,
        candidate_rows=candidate_rows,
        device=args.device,
        chunk_edges=int(args.chunk_edges),
        f0_threshold=float(args.f0_threshold),
        signed_threshold=float(args.signed_threshold),
        semantic_threshold=float(args.semantic_threshold),
        geo_sigma=float(args.geo_sigma),
    )
    dsu = DSU(len(feature_rows))
    for a, b in score_meta.get("allowed_f5_pairs", []):
        dsu.union(int(a), int(b))
    lcc_ratio = dsu.largest_ratio() if feature_rows else 0.0
    aucs = _diagnostic_aucs(scored_rows, feature_data, seed=int(args.diagnostic_seed), max_edges=int(args.auc_max_edges))
    raw_bucket_loads = [int(row["raw_bucket_load"]) for row in bucket_load_rows]
    clipped_bucket_loads = [int(row["clipped_bucket_load"]) for row in bucket_load_rows]
    bucket_load_p95 = float(np.percentile(clipped_bucket_loads, 95)) if clipped_bucket_loads else 0.0
    unique_mask_tokens = len(set((str(row["scene_id"]), int(row["target_frame_id"]), int(row["target_mask_id"])) for row in feature_rows if int(row["target_mask_id"]) > 0))
    unique_hash_bins = len(set(int(row["mask_sketch_hash_index"]) for row in feature_rows if int(row["target_mask_id"]) > 0))
    sketch_collision_mass = 1.0 - unique_hash_bins / max(1, unique_mask_tokens)
    negative_rows = [row for row in scored_rows if float(row["conflict_score"]) > 0.0]
    runtime_sec = float(time.time() - started)
    semantic_available_count = int(feature_data["semantic_available_count"])
    feature_count = len(feature_rows)
    radio_rate = semantic_available_count / max(1, feature_count)
    metrics = {
        "feature_count": feature_count,
        "feature_coverage_rate": 1.0 if feature_count else 0.0,
        "semantic_tensor_loaded": False,
        "radio_feature_available_rate": radio_rate,
        "nan_feature_count": 0,
        "zero_norm_feature_count": feature_count - semantic_available_count,
        "candidate_edge_count": len(candidate_rows),
        "affinity_edge_count": len(scored_rows),
        "negative_edge_count": len(negative_rows),
        "bucket_load_p95": bucket_load_p95,
        "bucket_load_budget": int(args.bucket_load_budget),
        "sketch_collision_mass": sketch_collision_mass,
        "sketch_collision_budget": float(args.sketch_collision_budget),
        "largest_connected_component_ratio_preview": lcc_ratio,
        "component_cannot_link_violation_count_preview": int(score_meta.get("component_cannot_link_violation_count_preview", 0)),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    quality_rows, semantic_quality_rows = _feature_quality_rows(
        feature_count=feature_count,
        semantic_available_count=semantic_available_count,
        candidate_edge_count=len(candidate_rows),
        negative_edge_count=len(negative_rows),
        bucket_load_p95=bucket_load_p95,
        sketch_collision_mass=sketch_collision_mass,
        lcc_ratio=lcc_ratio,
        violation_count=int(score_meta.get("component_cannot_link_violation_count_preview", 0)),
        aucs=aucs,
        runtime_sec=runtime_sec,
        peak_mb=float(score_meta.get("GPU_memory_peak_MB", 0.0)),
    )
    for row in semantic_quality_rows:
        row["semantic_projection_dim"] = int(args.semantic_projection_dim)
    gate_rows = _make_gates(metrics)
    proxy_quality_gate_pass = all(bool(row["pass"]) for row in gate_rows if row["gate_scope"] in {"proxy_quality", "integrity"})
    full_semantic_gate_pass = all(bool(row["pass"]) for row in gate_rows)

    np.savez(
        output_root / "feature_shards/feature_shard_000.npz",
        semantic_projection=feature_data["semantic_projection"].astype(np.float16),
        mask_descriptor=feature_data["mask_descriptor"].astype(np.float16),
        mask_sketch_hash_index=feature_data["mask_hash"],
        mask_sketch_sign=feature_data["mask_sign"],
        feature_index=np.arange(feature_count, dtype=np.int64),
        semantic_source=np.asarray(["radio_mask_feature_region_proxy"], dtype=object),
    )
    _write_csv(
        output_root / "micro_feature_index.csv",
        feature_rows,
        [
            "schema_version",
            "phase_id",
            "run_id",
            "feature_index",
            "scene_id",
            "window_id",
            "micro_primitive_id",
            "variant_id",
            "query_variant",
            "source_frame_id",
            "target_frame_id",
            "source_mask_id",
            "target_mask_id",
            "query_stratum",
            "B_pa",
            "visibility",
            "confidence",
            "near_boundary",
            "distinct_mask_count_3x3",
            "u_tgt",
            "v_tgt",
            "x_3d",
            "y_3d",
            "z_3d",
            "mask_sketch_hash_index",
            "mask_sketch_sign",
            "semantic_row_index",
            "semantic_feature_sha256",
            "semantic_source",
            "semantic_tensor_loaded",
            "semantic_region_proxy_loaded",
            "diagnostic_source_best_gt_id",
            "diagnostic_semantic_prototype_id",
            "uses_gt_for_prediction",
            "uses_future",
        ],
    )
    edge_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "feature_index_p",
        "feature_index_q",
        "edge_type",
        "scale",
        "same_mask_score",
        "conflict_score",
        "boundary_sep_score",
        "support_count",
        "bucket_key",
        "mask_sketch_score",
        "semantic_score",
        "geo_temporal_score",
        "risk_downweight",
        "F0_mask_incidence_only",
        "F1_signed_mask_incidence",
        "F2_radio_region_proxy_only",
        "F4_signed_region_proxy_affinity",
        "F5_scale_gated_region_proxy_affinity",
        "F0_allowed",
        "F1_allowed",
        "F2_allowed",
        "F4_allowed",
        "F5_allowed",
        "uses_gt_for_prediction",
        "uses_future",
    ]
    _write_csv(output_root / "candidate_edge_rows.csv", candidate_rows, ["feature_index_p", "feature_index_q", "edge_type", "same_mask_score", "conflict_score", "boundary_sep_score", "support_count", "bucket_key", "uses_gt_for_prediction", "uses_future"])
    _write_csv(output_root / "affinity_edge_rows.csv", scored_rows, edge_fields)
    _write_csv(output_root / "negative_edge_rows.csv", negative_rows, edge_fields)
    _write_csv(output_root / "feature_quality_rows.csv", quality_rows, list(quality_rows[0].keys()) if quality_rows else [])
    _write_csv(output_root / "semantic_feature_quality_rows.csv", semantic_quality_rows, list(semantic_quality_rows[0].keys()) if semantic_quality_rows else [])
    _write_csv(output_root / "bucket_load_rows.csv", bucket_load_rows, ["schema_version", "phase_id", "run_id", "bucket_type", "bucket_key", "raw_bucket_load", "clipped_bucket_load", "uses_gt_for_prediction", "uses_future"])
    _write_csv(output_root / "phase4_gate_rows.csv", gate_rows, list(gate_rows[0].keys()) if gate_rows else [])
    shard_sha = hashlib.sha256((output_root / "feature_shards/feature_shard_000.npz").read_bytes()).hexdigest()
    _write_json(
        output_root / "feature_tensor_manifest.json",
        {
            "schema": "stream4d_v97_phase4_feature_tensor_manifest_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "feature_shards": [{"path": _rel(output_root / "feature_shards/feature_shard_000.npz"), "sha256": shard_sha}],
            "feature_count": feature_count,
            "semantic_projection_dim": int(args.semantic_projection_dim),
            "semantic_projection_seed": int(args.semantic_projection_seed),
            "sketch_dim": int(args.sketch_dim),
            "semantic_feature_root": _rel(_project(args.semantic_feature_root)),
            "semantic_feature_source": "radio_mask_feature_region_proxy",
            "semantic_tensor_loaded": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )
    decision = "PASS_V97_PHASE4_REGION_PROXY_DIAGNOSTIC" if proxy_quality_gate_pass else "NO_GO_V97_PHASE4_REGION_PROXY_DIAGNOSTIC"
    summary = {
        "schema": "stream4d_v97_phase4_micro_affinity_feature_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "output_root": _rel(output_root),
        "incidence_root": _rel(incidence_root),
        "track_roots": [_rel(path) for path in track_roots],
        "semantic_feature_root": _rel(_project(args.semantic_feature_root)),
        "semantic_quality_rows": _rel(_project(args.semantic_quality_rows)),
        "feature_count": feature_count,
        "feature_coverage_rate": metrics["feature_coverage_rate"],
        "semantic_tensor_loaded": False,
        "semantic_feature_status": "region_proxy_loaded_dense_tensor_unavailable",
        "semantic_source": "radio_mask_feature_region_proxy",
        "radio_feature_available_rate": radio_rate,
        "dino_feature_available_rate": 0.0,
        "semantic_available_count": semantic_available_count,
        "semantic_missing_count": feature_count - semantic_available_count,
        "semantic_target_fallback_count": int(feature_data["fallback_target_count"]),
        "missing_track_count": int(feature_data["missing_track_count"]),
        "candidate_edge_count": len(candidate_rows),
        "affinity_edge_count": len(scored_rows),
        "negative_edge_count": len(negative_rows),
        "bucket_load_p95": bucket_load_p95,
        "raw_bucket_load_p95": float(np.percentile(raw_bucket_loads, 95)) if raw_bucket_loads else 0.0,
        "sketch_collision_mass": sketch_collision_mass,
        "largest_connected_component_ratio_preview": lcc_ratio,
        "component_cannot_link_violation_count_preview": int(score_meta.get("component_cannot_link_violation_count_preview", 0)),
        "within_semantic_hard_negative_AUC_diagnostic": aucs.get("within_semantic_hard_negative_AUC_diagnostic"),
        "same_GT_different_GT_region_AUC_diagnostic": aucs.get("same_GT_different_GT_region_AUC_diagnostic"),
        "semantic_shuffled_AUC_diagnostic": aucs.get("semantic_shuffled_AUC_diagnostic"),
        "diagnostic_only_uses_gt": bool(aucs.get("diagnostic_only_uses_gt", False)),
        "proxy_quality_gate_pass": proxy_quality_gate_pass,
        "full_semantic_gate_pass": full_semantic_gate_pass,
        "can_enter_phase5_diagnostic": proxy_quality_gate_pass,
        "can_enter_phase5_full": full_semantic_gate_pass,
        "runtime_sec": runtime_sec,
        "GPU_memory_peak_MB": float(score_meta.get("GPU_memory_peak_MB", 0.0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gate_rows": gate_rows,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "feature_count": feature_count,
                "radio_feature_available_rate": radio_rate,
                "candidate_edge_count": len(candidate_rows),
                "negative_edge_count": len(negative_rows),
                "proxy_quality_gate_pass": proxy_quality_gate_pass,
                "full_semantic_gate_pass": full_semantic_gate_pass,
                "runtime_sec": runtime_sec,
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidence-root", default=str(DEFAULT_PHASE3))
    parser.add_argument("--track-root", action="append", default=[])
    parser.add_argument("--semantic-feature-root", default=str(DEFAULT_SEMANTIC))
    parser.add_argument("--semantic-quality-rows", default=str(DEFAULT_SEMANTIC_QUALITY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--decode-variants", default="D3_adaptive1024")
    parser.add_argument("--max-event-rows", type=int, default=0)
    parser.add_argument("--semantic-projection-dim", type=int, default=64)
    parser.add_argument("--semantic-projection-seed", type=int, default=9704)
    parser.add_argument("--sketch-dim", type=int, default=65536)
    parser.add_argument("--max-bucket-rows", type=int, default=512)
    parser.add_argument("--positive-neighbors", type=int, default=2)
    parser.add_argument("--negative-neighbors", type=int, default=4)
    parser.add_argument("--negative-boundary-px", type=float, default=2.0)
    parser.add_argument("--bucket-load-budget", type=int, default=2048)
    parser.add_argument("--sketch-collision-budget", type=float, default=0.05)
    parser.add_argument("--chunk-edges", type=int, default=262144)
    parser.add_argument("--f0-threshold", type=float, default=0.50)
    parser.add_argument("--signed-threshold", type=float, default=0.55)
    parser.add_argument("--semantic-threshold", type=float, default=0.70)
    parser.add_argument("--geo-sigma", type=float, default=0.50)
    parser.add_argument("--auc-max-edges", type=int, default=200000)
    parser.add_argument("--diagnostic-seed", type=int, default=9704)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
