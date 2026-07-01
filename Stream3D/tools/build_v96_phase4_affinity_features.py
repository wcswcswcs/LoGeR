#!/usr/bin/env python3
"""Build v96 Phase4 micro affinity features from Triton incidence rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase4_affinity_features"
RUN_ID = "v96_phase4_affinity_features"
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase4_affinity_features"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def largest_ratio(self) -> float:
        counts = Counter(self.find(i) for i in range(len(self.parent)))
        return max(counts.values(), default=0) / max(1, len(self.parent))


def _feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("scene_id", ""), row.get("window_id", ""), row.get("query_id", ""))


def _load_incidence_rows(root: Path, decode_variants: set[str]) -> list[dict[str, str]]:
    path = root / "incidence_event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing incidence_event_rows.csv: {path}")
    rows = []
    for row in _read_csv(path):
        if decode_variants and row.get("decode_variant", "") not in decode_variants:
            continue
        rows.append(row)
    return rows


def _build_features(rows: list[dict[str, str]], image_width: int, image_height: int) -> tuple[list[dict[str, Any]], np.ndarray, dict[tuple[str, str, str], int]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_feature_key(row)].append(row)
    keys = sorted(grouped)
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    feature_rows: list[dict[str, Any]] = []
    vecs: list[list[float]] = []
    for idx, key in enumerate(keys):
        evs = grouped[key]
        positive = [row for row in evs if _bool(row.get("query_has_positive_mask"))]
        masks = [int(_num(row.get("center_mask_id"))) for row in positive]
        mask_counts = Counter(masks)
        dominant_mask, dominant_count = mask_counts.most_common(1)[0] if mask_counts else (0, 0)
        event_count = len(evs)
        positive_rate = len(positive) / max(1, event_count)
        dominant_ratio = dominant_count / max(1, len(positive))
        multi_rate = sum(1 for row in evs if _bool(row.get("query_has_multiple_masks_3x3"))) / max(1, event_count)
        boundary_rate = sum(1 for row in evs if 0.0 <= _num(row.get("boundary_distance_px"), -1.0) <= 2.0) / max(1, event_count)
        x_mean = float(np.mean([_num(row.get("u_tgt")) for row in evs]))
        y_mean = float(np.mean([_num(row.get("v_tgt")) for row in evs]))
        frames = [int(_num(row.get("target_frame_id"))) for row in evs]
        frame_mean = float(np.mean(frames)) if frames else 0.0
        frame_span = float(max(frames) - min(frames)) if frames else 0.0
        distinct_mean = float(np.mean([_num(row.get("distinct_mask_count_3x3")) for row in evs])) if evs else 0.0
        vec = [
            positive_rate,
            dominant_ratio,
            multi_rate,
            boundary_rate,
            min(1.0, distinct_mean / 4.0),
            x_mean / max(1, image_width - 1),
            y_mean / max(1, image_height - 1),
            frame_mean / 100.0,
            frame_span / 100.0,
        ]
        norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        feature_rows.append(
            {
                "schema_version": "stream4d_v96_micro_feature_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "feature_index": idx,
                "scene_id": key[0],
                "window_id": key[1],
                "query_id": key[2],
                "decode_variant": evs[0].get("decode_variant", ""),
                "query_variant": evs[0].get("query_variant", ""),
                "event_count": event_count,
                "positive_rate": positive_rate,
                "dominant_mask_id": dominant_mask,
                "dominant_mask_ratio": dominant_ratio,
                "multi_mask_rate": multi_rate,
                "boundary_near_rate": boundary_rate,
                "distinct_mask_count_mean": distinct_mean,
                "x_mean": x_mean,
                "y_mean": y_mean,
                "frame_mean": frame_mean,
                "frame_span": frame_span,
                "feature_norm": norm,
                "semantic_feature_source": "mask_incidence_proxy_no_radio_dino_loaded",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        vecs.append(vec)
    return feature_rows, np.asarray(vecs, dtype=np.float32), key_to_index


def _candidate_pairs(
    rows: list[dict[str, str]],
    key_to_index: dict[tuple[str, str, str], int],
    *,
    max_bucket_rows: int,
    positive_neighbors: int,
    negative_neighbors: int,
    negative_boundary_px: float,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[float], list[float], list[float]]:
    buckets: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    frame_buckets: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame = int(_num(row.get("target_frame_id")))
        label = int(_num(row.get("center_mask_id")))
        if label > 0:
            buckets[(scene, window, frame, label)].append(row)
            frame_buckets[(scene, window, frame)].append(row)
    raw_bucket_loads = [len(vals) for vals in buckets.values()]
    edge_map: dict[tuple[int, int], dict[str, Any]] = {}

    def add_edge(a_key: tuple[str, str, str], b_key: tuple[str, str, str], edge_type: str, same: float, conflict: float, boundary: float) -> None:
        if a_key == b_key or a_key not in key_to_index or b_key not in key_to_index:
            return
        a, b = key_to_index[a_key], key_to_index[b_key]
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
            }
        else:
            current["same_mask_score"] = max(float(current["same_mask_score"]), same)
            current["conflict_score"] = max(float(current["conflict_score"]), conflict)
            current["boundary_sep_score"] = max(float(current["boundary_sep_score"]), boundary)
            current["support_count"] = int(current["support_count"]) + 1
            if conflict > float(current["conflict_score"]):
                current["edge_type"] = edge_type

    for vals in buckets.values():
        vals = sorted(vals, key=lambda row: (_num(row.get("u_tgt")), _num(row.get("v_tgt")), row.get("query_id", "")))
        if len(vals) > max_bucket_rows:
            take = np.linspace(0, len(vals) - 1, max_bucket_rows, dtype=np.int64)
            vals = [vals[int(i)] for i in take.tolist()]
        for i, row in enumerate(vals):
            a_key = _feature_key(row)
            for j in range(i + 1, min(len(vals), i + 1 + positive_neighbors)):
                add_edge(a_key, _feature_key(vals[j]), "positive_same_mask_local", 1.0, 0.0, 0.0)

    for vals in frame_buckets.values():
        vals = sorted(vals, key=lambda row: (_num(row.get("u_tgt")), _num(row.get("v_tgt")), row.get("query_id", "")))
        if len(vals) > max_bucket_rows:
            take = np.linspace(0, len(vals) - 1, max_bucket_rows, dtype=np.int64)
            vals = [vals[int(i)] for i in take.tolist()]
        for i, row in enumerate(vals):
            label = int(_num(row.get("center_mask_id")))
            if label <= 0:
                continue
            a_key = _feature_key(row)
            for j in range(i + 1, min(len(vals), i + 1 + negative_neighbors)):
                other = vals[j]
                other_label = int(_num(other.get("center_mask_id")))
                if other_label <= 0 or other_label == label:
                    continue
                near_boundary = (
                    0.0 <= _num(row.get("boundary_distance_px"), -1.0) <= negative_boundary_px
                    or 0.0 <= _num(other.get("boundary_distance_px"), -1.0) <= negative_boundary_px
                    or row.get("query_stratum") in {"boundary", "conflict"}
                    or other.get("query_stratum") in {"boundary", "conflict"}
                )
                if near_boundary:
                    add_edge(a_key, _feature_key(other), "negative_boundary_conflict", 0.0, 1.0, 1.0)

    edge_rows = list(edge_map.values())
    p_idx = [int(row["feature_index_p"]) for row in edge_rows]
    q_idx = [int(row["feature_index_q"]) for row in edge_rows]
    same = [float(row["same_mask_score"]) for row in edge_rows]
    conflict = [float(row["conflict_score"]) for row in edge_rows]
    boundary = [float(row["boundary_sep_score"]) for row in edge_rows]
    for row in edge_rows:
        row["raw_bucket_load_p95"] = float(np.percentile(raw_bucket_loads, 95)) if raw_bucket_loads else 0.0
        row["uses_gt_for_prediction"] = False
        row["uses_future"] = False
    return edge_rows, p_idx, q_idx, same, conflict, boundary


def _score_edges(
    features: np.ndarray,
    edge_rows: list[dict[str, Any]],
    p_idx: list[int],
    q_idx: list[int],
    same: list[float],
    conflict: list[float],
    boundary: list[float],
    *,
    device: str,
    union_threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    if not edge_rows:
        return [], 0.0
    torch.cuda.reset_peak_memory_stats()
    feat = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    feat = torch.nn.functional.normalize(feat, dim=1, eps=1e-6)
    p = torch.tensor(p_idx, device=device, dtype=torch.long)
    q = torch.tensor(q_idx, device=device, dtype=torch.long)
    same_t = torch.tensor(same, device=device, dtype=torch.float32)
    conflict_t = torch.tensor(conflict, device=device, dtype=torch.float32)
    boundary_t = torch.tensor(boundary, device=device, dtype=torch.float32)
    cos = (feat[p] * feat[q]).sum(dim=1)
    temporal = torch.clamp(cos, min=-1.0, max=1.0)
    f0 = same_t
    f3 = 0.85 * same_t + 0.35 * temporal - 0.95 * conflict_t - 0.55 * boundary_t
    f5 = 0.75 * same_t + 0.40 * temporal - 1.05 * conflict_t - 0.65 * boundary_t
    f0_np = f0.detach().cpu().numpy()
    temporal_np = temporal.detach().cpu().numpy()
    f3_np = f3.detach().cpu().numpy()
    f5_np = f5.detach().cpu().numpy()
    scored: list[dict[str, Any]] = []
    for idx, row in enumerate(edge_rows):
        for variant, signed in [("F0_same_mask_only", f0_np[idx]), ("F3_signed_conflict_aware", f3_np[idx]), ("F5_full_signed_proxy", f5_np[idx])]:
            allowed = bool(row["same_mask_score"] > 0 and signed >= union_threshold)
            scored.append(
                {
                    "schema_version": "stream4d_v96_micro_affinity_edge_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "feature_variant": variant,
                    "feature_index_p": int(row["feature_index_p"]),
                    "feature_index_q": int(row["feature_index_q"]),
                    "scale": "object",
                    "positive_score": float(row["same_mask_score"]),
                    "semantic_score": float(temporal_np[idx]),
                    "temporal_score": float(temporal_np[idx]),
                    "boundary_sep_score": float(row["boundary_sep_score"]),
                    "conflict_score": float(row["conflict_score"]),
                    "partwhole_score": 0.0,
                    "signed_affinity": float(signed),
                    "edge_type": row["edge_type"],
                    "edge_allowed_for_union": allowed,
                    "support_count": int(row["support_count"]),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0**2))
    return scored, peak_mb


def _variant_diagnostics(edge_rows: list[dict[str, Any]], feature_count: int, raw_bucket_p95: float, feature_coverage_rate: float, feature_norms: np.ndarray, runtime_feature_sec: float, peak_mb: float) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        by_variant[row["feature_variant"]].append(row)
    out: list[dict[str, Any]] = []
    for variant, vals in sorted(by_variant.items()):
        dsu = DSU(feature_count)
        for row in vals:
            if _bool(row.get("edge_allowed_for_union")):
                dsu.union(int(row["feature_index_p"]), int(row["feature_index_q"]))
        out.append(
            {
                "schema_version": "stream4d_v96_phase4_feature_diagnostic_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "feature_variant": variant,
                "feature_coverage_rate": feature_coverage_rate,
                "feature_norm_mean": float(np.mean(feature_norms)) if feature_norms.size else 0.0,
                "feature_norm_p10": float(np.percentile(feature_norms, 10)) if feature_norms.size else 0.0,
                "bucket_load_p95": raw_bucket_p95,
                "bucket_load_budget": 4096,
                "sketch_collision_mass": 0.0,
                "positive_edge_count": sum(1 for row in vals if float(row["positive_score"]) > 0),
                "negative_edge_count": sum(1 for row in vals if float(row["conflict_score"]) > 0),
                "cannot_link_candidate_count": sum(1 for row in vals if float(row["conflict_score"]) > 0),
                "within_semantic_hard_negative_AUC_diagnostic": "",
                "partwhole_consistency_AUC_diagnostic": "",
                "topk_neighbor_recall_proxy": "",
                "largest_component_ratio": dsu.largest_ratio(),
                "runtime_feature_sec": runtime_feature_sec,
                "GPU_memory_peak_MB": peak_mb,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    incidence_root = _project(args.incidence_root)
    decode_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    rows = _load_incidence_rows(incidence_root, decode_variants)
    if not rows:
        raise RuntimeError("No incidence rows selected for Phase4.")
    feature_rows, feature_np, key_to_index = _build_features(rows, int(args.image_width), int(args.image_height))
    candidate_rows, p_idx, q_idx, same, conflict, boundary = _candidate_pairs(
        rows,
        key_to_index,
        max_bucket_rows=int(args.max_bucket_rows),
        positive_neighbors=int(args.positive_neighbors),
        negative_neighbors=int(args.negative_neighbors),
        negative_boundary_px=float(args.negative_boundary_px),
    )
    raw_bucket_p95 = float(np.percentile([row["raw_bucket_load_p95"] for row in candidate_rows], 95)) if candidate_rows else 0.0
    scored_edges, peak_mb = _score_edges(
        feature_np,
        candidate_rows,
        p_idx,
        q_idx,
        same,
        conflict,
        boundary,
        device=args.device,
        union_threshold=float(args.union_threshold),
    )
    runtime_feature_sec = float(time.time() - started)
    feature_norms = np.asarray([_num(row.get("feature_norm")) for row in feature_rows], dtype=np.float32)
    feature_coverage_rate = float(np.mean(feature_norms > 0.0)) if feature_norms.size else 0.0
    diagnostics = _variant_diagnostics(
        scored_edges,
        len(feature_rows),
        raw_bucket_p95,
        feature_coverage_rate,
        feature_norms,
        runtime_feature_sec,
        peak_mb,
    )
    gates: list[dict[str, Any]] = []
    for row in diagnostics:
        variant = row["feature_variant"]
        gates.extend(
            [
                {
                    "schema_version": "stream4d_v96_phase4_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "feature_variant": variant,
                    "gate": "feature_coverage_rate_ge_0p95",
                    "pass": bool(_num(row["feature_coverage_rate"]) >= 0.95),
                    "observed": row["feature_coverage_rate"],
                    "required": 0.95,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v96_phase4_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "feature_variant": variant,
                    "gate": "bucket_load_p95_within_budget",
                    "pass": bool(_num(row["bucket_load_p95"]) <= _num(row["bucket_load_budget"])),
                    "observed": row["bucket_load_p95"],
                    "required": row["bucket_load_budget"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
            ]
        )
        if variant in {"F3_signed_conflict_aware", "F5_full_signed_proxy"}:
            gates.extend(
                [
                    {
                        "schema_version": "stream4d_v96_phase4_gate_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "feature_variant": variant,
                        "gate": "negative_edge_count_gt_0",
                        "pass": bool(int(row["negative_edge_count"]) > 0),
                        "observed": row["negative_edge_count"],
                        "required": ">0",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    },
                    {
                        "schema_version": "stream4d_v96_phase4_gate_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "feature_variant": variant,
                        "gate": "largest_component_ratio_le_0p30",
                        "pass": bool(_num(row["largest_component_ratio"]) <= 0.30),
                        "observed": row["largest_component_ratio"],
                        "required": 0.30,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    },
                ]
            )
    phase4_pass = all(bool(row["pass"]) for row in gates)
    topk_rows = [
        {
            "schema_version": "stream4d_v96_topk_neighbor_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "feature_index_p": row["feature_index_p"],
            "feature_index_q": row["feature_index_q"],
            "feature_variant": row["feature_variant"],
            "signed_affinity": row["signed_affinity"],
            "edge_type": row["edge_type"],
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in scored_edges
        if row["feature_variant"] == "F5_full_signed_proxy"
    ]
    _write_csv(output_root / "micro_feature_rows.csv", feature_rows)
    _write_csv(output_root / "micro_affinity_edge_rows.csv", scored_edges)
    _write_csv(output_root / "topk_neighbor_rows.csv", topk_rows)
    _write_csv(output_root / "feature_diagnostic_rows.csv", diagnostics)
    _write_csv(output_root / "phase4_gate_rows.csv", gates)
    _write_json(
        output_root / "micro_feature_tensor_manifest.json",
        {
            "schema": "stream4d_v96_micro_feature_tensor_manifest_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "feature_shape": list(feature_np.shape),
            "incidence_root": _rel(incidence_root),
            "candidate_edge_count": len(candidate_rows),
            "scored_affinity_edge_count": len(scored_edges),
            "device": args.device,
            "GPU_memory_peak_MB": peak_mb,
        },
    )
    summary = {
        "schema": "stream4d_v96_phase4_affinity_features_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE4_AFFINITY_FEATURES" if phase4_pass else "NO_GO_V96_PHASE4_AFFINITY_FEATURES",
        "output_root": _rel(output_root),
        "incidence_root": _rel(incidence_root),
        "incidence_row_count": len(rows),
        "feature_count": len(feature_rows),
        "candidate_edge_count": len(candidate_rows),
        "scored_affinity_edge_count": len(scored_edges),
        "runtime_feature_sec": runtime_feature_sec,
        "GPU_memory_peak_MB": peak_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "diagnostics": diagnostics,
        "gate_rows": gates,
        "semantic_feature_status": "proxy_only_no_radio_dino_tensor_loaded",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "feature_count": len(feature_rows), "candidate_edge_count": len(candidate_rows), "runtime_feature_sec": runtime_feature_sec}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 Phase4 affinity features.")
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--decode-variants", default="D3_adaptive1024")
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--max-bucket-rows", type=int, default=2048)
    parser.add_argument("--positive-neighbors", type=int, default=2)
    parser.add_argument("--negative-neighbors", type=int, default=4)
    parser.add_argument("--negative-boundary-px", type=float, default=2.0)
    parser.add_argument("--union-threshold", type=float, default=1.20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
