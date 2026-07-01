#!/usr/bin/env python3
"""Build v97 Phase5 local object-birth diagnostics on micro-primitives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase5_object_birth"
RUN_ID = "v97_phase5_object_birth"
DEFAULT_PHASE4 = ROOT / "outputs/audit/v97_phase4_micro_affinity_feature_D3_source_preserve2048_region_proxy_500k_gpu6"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase5_object_birth_region_proxy_500k"


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.cannot: list[set[int]] = [set() for _ in range(n)]

    def find(self, x: int) -> int:
        while int(self.parent[x]) != x:
            self.parent[x] = self.parent[int(self.parent[x])]
            x = int(self.parent[x])
        return x

    def add_cannot_link(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.cannot[ra].add(rb)
        self.cannot[rb].add(ra)

    def can_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        return rb not in self.cannot[ra] and ra not in self.cannot[rb]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if rb in self.cannot[ra] or ra in self.cannot[rb]:
            return False
        if int(self.size[ra]) < int(self.size[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.cannot[ra].update(self.cannot[rb])
        for other in list(self.cannot[rb]):
            self.cannot[other].discard(rb)
            if other != ra:
                self.cannot[other].add(ra)
        self.cannot[rb].clear()
        self.cannot[ra].discard(ra)
        return True


def _load_features(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in _iter_csv(root / "micro_feature_index.csv"):
        rows.append(row)
    n = len(rows)
    arrays = {
        "scene": np.asarray([row["scene_id"] for row in rows], dtype=object),
        "window": np.asarray([row["window_id"] for row in rows], dtype=object),
        "source_frame": np.asarray([_int(row.get("source_frame_id")) for row in rows], dtype=np.int32),
        "target_frame": np.asarray([_int(row.get("target_frame_id")) for row in rows], dtype=np.int32),
        "source_mask": np.asarray([_int(row.get("source_mask_id")) for row in rows], dtype=np.int32),
        "target_mask": np.asarray([_int(row.get("target_mask_id")) for row in rows], dtype=np.int32),
        "visibility": np.asarray([_num(row.get("visibility")) for row in rows], dtype=np.float32),
        "confidence": np.asarray([_num(row.get("confidence")) for row in rows], dtype=np.float32),
        "B_pa": np.asarray([_num(row.get("B_pa")) for row in rows], dtype=np.float32),
        "near_boundary": np.asarray([_bool(row.get("near_boundary")) for row in rows], dtype=np.bool_),
        "distinct": np.asarray([_num(row.get("distinct_mask_count_3x3")) for row in rows], dtype=np.float32),
        "semantic_proto": np.asarray([row.get("diagnostic_semantic_prototype_id", "") for row in rows], dtype=object),
        "gt_diag": np.asarray([_int(row.get("diagnostic_source_best_gt_id"), -1) for row in rows], dtype=np.int32),
    }
    return {"rows": rows, "n": n, **arrays}


def _load_edges(root: Path) -> dict[str, np.ndarray]:
    p: list[int] = []
    q: list[int] = []
    conflict: list[float] = []
    boundary: list[float] = []
    f5: list[float] = []
    f4: list[float] = []
    f2: list[float] = []
    same: list[float] = []
    for row in _iter_csv(root / "affinity_edge_rows.csv"):
        p.append(_int(row.get("feature_index_p")))
        q.append(_int(row.get("feature_index_q")))
        conflict.append(_num(row.get("conflict_score")))
        boundary.append(_num(row.get("boundary_sep_score")))
        f5.append(_num(row.get("F5_scale_gated_region_proxy_affinity")))
        f4.append(_num(row.get("F4_signed_region_proxy_affinity")))
        f2.append(_num(row.get("F2_radio_region_proxy_only")))
        same.append(_num(row.get("same_mask_score")))
    return {
        "p": np.asarray(p, dtype=np.int64),
        "q": np.asarray(q, dtype=np.int64),
        "conflict": np.asarray(conflict, dtype=np.float32),
        "boundary": np.asarray(boundary, dtype=np.float32),
        "F5": np.asarray(f5, dtype=np.float32),
        "F4": np.asarray(f4, dtype=np.float32),
        "F2": np.asarray(f2, dtype=np.float32),
        "same": np.asarray(same, dtype=np.float32),
    }


def _labels_from_keys(features: dict[str, Any], key_fn, min_size: int) -> np.ndarray:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    rows = features["rows"]
    for idx, row in enumerate(rows):
        key = key_fn(idx, row)
        if key:
            groups[key].append(idx)
    labels = np.full(features["n"], -1, dtype=np.int64)
    obj_idx = 0
    for _key, vals in sorted(groups.items(), key=lambda item: (str(item[0]), len(item[1]))):
        if len(vals) < int(min_size):
            continue
        labels[np.asarray(vals, dtype=np.int64)] = obj_idx
        obj_idx += 1
    return labels


def _labels_from_edges(
    features: dict[str, Any],
    edges: dict[str, np.ndarray],
    *,
    score_name: str,
    threshold: float,
    min_size: int,
    constrained: bool,
) -> np.ndarray:
    n = int(features["n"])
    dsu = DSU(n)
    conflict_idx = np.flatnonzero(edges["conflict"] > 0)
    if constrained:
        for idx in conflict_idx.tolist():
            dsu.add_cannot_link(int(edges["p"][idx]), int(edges["q"][idx]))
    pos_idx = np.flatnonzero((edges[score_name] >= float(threshold)) & (edges["conflict"] <= 0))
    pos_idx = pos_idx[np.argsort(edges[score_name][pos_idx])[::-1]]
    for idx in pos_idx.tolist():
        a, b = int(edges["p"][idx]), int(edges["q"][idx])
        if constrained and not dsu.can_union(a, b):
            continue
        dsu.union(a, b)
    root_to_count: Counter[int] = Counter()
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = dsu.find(i)
        roots[i] = r
        root_to_count[int(r)] += 1
    root_to_obj: dict[int, int] = {}
    labels = np.full(n, -1, dtype=np.int64)
    for i, root in enumerate(roots.tolist()):
        if root_to_count[int(root)] < int(min_size):
            continue
        if int(root) not in root_to_obj:
            root_to_obj[int(root)] = len(root_to_obj)
        labels[i] = root_to_obj[int(root)]
    return labels


def _prune_same_frame_dominant(labels: np.ndarray, features: dict[str, Any]) -> np.ndarray:
    out = labels.copy()
    valid_labels = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    for label in valid_labels:
        idxs = np.flatnonzero(out == label)
        by_frame: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
        for idx in idxs.tolist():
            key = (str(features["scene"][idx]), str(features["window"][idx]), int(features["target_frame"][idx]))
            by_frame[key][int(features["target_mask"][idx])] += 1
        keep = np.ones(idxs.shape[0], dtype=np.bool_)
        dominant = {key: counts.most_common(1)[0][0] for key, counts in by_frame.items()}
        for pos, idx in enumerate(idxs.tolist()):
            key = (str(features["scene"][idx]), str(features["window"][idx]), int(features["target_frame"][idx]))
            keep[pos] = int(features["target_mask"][idx]) == int(dominant[key])
        out[idxs[~keep]] = -1
    return _reindex_labels(out)


def _drop_risky(labels: np.ndarray, features: dict[str, Any]) -> np.ndarray:
    out = labels.copy()
    risky = features["near_boundary"] & (features["distinct"] > 1)
    out[risky] = -1
    return _reindex_labels(out)


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape, -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    for idx, label in enumerate(labels.tolist()):
        if int(label) < 0:
            continue
        if int(label) not in mapping:
            mapping[int(label)] = len(mapping)
        out[idx] = mapping[int(label)]
    return out


def _expand_from_seeds(seed_labels: np.ndarray, edges: dict[str, np.ndarray], *, score_name: str, threshold: float, conflict_veto: bool) -> np.ndarray:
    labels = seed_labels.copy()
    for idx in np.argsort(edges[score_name])[::-1].tolist():
        if float(edges[score_name][idx]) < float(threshold):
            break
        if conflict_veto and float(edges["conflict"][idx]) > 0:
            continue
        a, b = int(edges["p"][idx]), int(edges["q"][idx])
        la, lb = int(labels[a]), int(labels[b])
        if la >= 0 and lb < 0:
            labels[b] = la
        elif lb >= 0 and la < 0:
            labels[a] = lb
    return _reindex_labels(labels)


def _entropy(values: Iterable[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    counts = np.asarray(list(Counter(vals).values()), dtype=np.float64)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _edge_stats_for_labels(labels: np.ndarray, edges: dict[str, np.ndarray], score_name: str) -> dict[str, Any]:
    p_label = labels[edges["p"]]
    q_label = labels[edges["q"]]
    same_obj = (p_label >= 0) & (p_label == q_label)
    diff_obj = (p_label >= 0) & (q_label >= 0) & (p_label != q_label)
    cannot = same_obj & (edges["conflict"] > 0)
    internal = edges[score_name][same_obj]
    external = edges[score_name][diff_obj]
    internal_mean = float(np.mean(internal)) if internal.size else 0.0
    external_mean = float(np.mean(external)) if external.size else 0.0
    return {
        "cannot_link_violation_count": int(np.count_nonzero(cannot)),
        "cluster_internal_affinity_mean": internal_mean,
        "cluster_external_margin_mean": float(internal_mean - external_mean),
        "external_affinity_mean": external_mean,
    }


def _same_frame_violation(labels: np.ndarray, features: dict[str, Any]) -> int:
    count = 0
    for label in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
        idxs = np.flatnonzero(labels == label)
        by_frame: dict[tuple[str, str, int], set[int]] = defaultdict(set)
        for idx in idxs.tolist():
            by_frame[(str(features["scene"][idx]), str(features["window"][idx]), int(features["target_frame"][idx]))].add(int(features["target_mask"][idx]))
        count += sum(max(0, len(masks) - 1) for masks in by_frame.values())
    return int(count)


def _candidate_rows_for_variant(
    *,
    variant_id: str,
    birth_family: str,
    labels: np.ndarray,
    features: dict[str, Any],
    edge_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = max(1, int(np.count_nonzero(labels >= 0)))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
        idxs = np.flatnonzero(labels == label)
        if idxs.size == 0:
            continue
        scenes = Counter(str(features["scene"][idx]) for idx in idxs.tolist())
        windows = Counter(str(features["window"][idx]) for idx in idxs.tolist())
        scene = scenes.most_common(1)[0][0]
        window = windows.most_common(1)[0][0]
        frames = set(int(features["target_frame"][idx]) for idx in idxs.tolist())
        target_masks = set((int(features["target_frame"][idx]), int(features["target_mask"][idx])) for idx in idxs.tolist())
        source_masks = set((int(features["source_frame"][idx]), int(features["source_mask"][idx])) for idx in idxs.tolist())
        score = float(np.mean(features["B_pa"][idxs])) * math.log1p(float(idxs.size))
        rows.append(
            {
                "schema_version": "stream4d_v97_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": window,
                "object_id": f"{variant_id}_obj_{label:06d}",
                "birth_family": birth_family,
                "micro_primitive_count": int(idxs.size),
                "frame_support_count": int(len(frames)),
                "masklet_support_count": int(len(target_masks)),
                "mean_affinity_internal": edge_stats.get("cluster_internal_affinity_mean", ""),
                "mean_affinity_external_margin": edge_stats.get("cluster_external_margin_mean", ""),
                "cannot_link_violation_count": "",
                "source_mask_count": int(len(source_masks)),
                "semantic_proto_entropy": _entropy(features["semantic_proto"][idxs].tolist()),
                "d4rt_visibility_mean": float(np.mean(features["visibility"][idxs])),
                "support_area_ratio_mean": float(idxs.size / total),
                "score": score,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _metric_row(
    *,
    variant_id: str,
    birth_family: str,
    labels: np.ndarray,
    features: dict[str, Any],
    edge_stats: dict[str, Any],
    same_frame_violation_count: int,
    baseline_object_count_per_window_mean: float,
    runtime_sec: float,
) -> dict[str, Any]:
    valid = labels >= 0
    assigned = int(np.count_nonzero(valid))
    object_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
    counts = np.asarray([int(np.count_nonzero(labels == label)) for label in object_labels], dtype=np.float64)
    windows = sorted(set(str(v) for v in features["window"].tolist()))
    object_count_by_window = Counter()
    for label in object_labels:
        idxs = np.flatnonzero(labels == label)
        if idxs.size:
            object_count_by_window[str(Counter(str(features["window"][idx]) for idx in idxs.tolist()).most_common(1)[0][0])] += 1
    object_count_per_window_mean = float(np.mean([object_count_by_window.get(w, 0) for w in windows])) if windows else 0.0
    largest_object_ratio = float(np.max(counts) / max(1.0, float(features["n"]))) if counts.size else 0.0
    cluster_spans = []
    source_mask_counts = []
    target_mask_counts = []
    for label in object_labels:
        idxs = np.flatnonzero(labels == label)
        if idxs.size == 0:
            continue
        cluster_spans.append(int(np.max(features["target_frame"][idxs]) - np.min(features["target_frame"][idxs]) + 1))
        source_mask_counts.append(len(set((int(features["source_frame"][idx]), int(features["source_mask"][idx])) for idx in idxs.tolist())))
        target_mask_counts.append(len(set((int(features["target_frame"][idx]), int(features["target_mask"][idx])) for idx in idxs.tolist())))
    keypoint_coverage_rate = float(assigned / max(1, int(features["n"])))
    fragmentation = float(object_count_per_window_mean / max(1.0, baseline_object_count_per_window_mean))
    return {
        "schema_version": "stream4d_v97_object_birth_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "birth_family": birth_family,
        "object_count": int(len(object_labels)),
        "object_count_per_window_mean": object_count_per_window_mean,
        "micro_primitives_per_object_mean": float(np.mean(counts)) if counts.size else 0.0,
        "micro_primitives_per_object_p90": float(np.percentile(counts, 90)) if counts.size else 0.0,
        "largest_object_ratio": largest_object_ratio,
        "keypoint_coverage_rate": keypoint_coverage_rate,
        "masklet_cover_count": int(sum(target_mask_counts)),
        "cannot_link_violation_count": int(edge_stats.get("cannot_link_violation_count", 0)),
        "same_frame_violation_count": int(same_frame_violation_count),
        "object_fragmentation_proxy": fragmentation,
        "object_overmerge_proxy": largest_object_ratio,
        "cluster_temporal_span_mean": float(np.mean(cluster_spans)) if cluster_spans else 0.0,
        "cluster_internal_affinity_mean": edge_stats.get("cluster_internal_affinity_mean", 0.0),
        "cluster_external_margin_mean": edge_stats.get("cluster_external_margin_mean", 0.0),
        "preview_support_to_mask_iou_mean": "",
        "GT_best_IoU_diagnostic": "",
        "within_semantic_hard_negative_AUC_diagnostic": "",
        "runtime_sec": runtime_sec,
        "assigned_micro_primitive_count": assigned,
        "unassigned_micro_primitive_count": int(features["n"] - assigned),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_rows_for_metric(row: dict[str, Any], *, baseline_plus_tolerance: float) -> list[dict[str, Any]]:
    variant = row["variant_id"]
    family = row["birth_family"]
    needs_coverage = family in {"A_keypoint_cover", "C_hybrid_cover_cluster"}
    needs_margin = family in {"B_signed_constrained_clustering", "C_hybrid_cover_cluster"}
    specs = [
        ("object_count_per_window_mean_within_20_400", 20 <= _num(row["object_count_per_window_mean"]) <= 400, row["object_count_per_window_mean"], "[20, 400]"),
        ("largest_object_ratio_le_0p30", _num(row["largest_object_ratio"]) <= 0.30, row["largest_object_ratio"], 0.30),
        ("cannot_link_violation_count_eq_0", int(row["cannot_link_violation_count"]) == 0, row["cannot_link_violation_count"], 0),
        ("same_frame_violation_count_eq_0", int(row["same_frame_violation_count"]) == 0, row["same_frame_violation_count"], 0),
        ("keypoint_coverage_rate_ge_0p50_for_cover", (not needs_coverage) or _num(row["keypoint_coverage_rate"]) >= 0.50, row["keypoint_coverage_rate"], ">=0.50 for A/C"),
        ("cluster_external_margin_mean_gt_0_for_cluster", (not needs_margin) or _num(row["cluster_external_margin_mean"]) > 0, row["cluster_external_margin_mean"], ">0 for B/C"),
        ("object_fragmentation_proxy_le_baseline_plus_tolerance", _num(row["object_fragmentation_proxy"]) <= baseline_plus_tolerance, row["object_fragmentation_proxy"], baseline_plus_tolerance),
        ("uses_gt_for_prediction_false", not _bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
        ("uses_future_false", not _bool(row["uses_future"]), row["uses_future"], False),
    ]
    gates = []
    for gate, passed, observed, required in specs:
        gates.append(
            {
                "schema_version": "stream4d_v97_object_birth_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant,
                "birth_family": family,
                "gate": gate,
                "pass": bool(passed),
                "observed": observed,
                "required": required,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    family_pass = all(bool(row["pass"]) for row in gates)
    gates.append(
        {
            "schema_version": "stream4d_v97_object_birth_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": variant,
            "birth_family": family,
            "gate": "family_gate_to_phase6_diagnostic",
            "pass": family_pass,
            "observed": family_pass,
            "required": "all Phase5 diagnostic sanity gates",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )
    return gates


def _write_micro_rows(path: Path, variant_id: str, birth_family: str, labels: np.ndarray, features: dict[str, Any]) -> None:
    exists = path.exists()
    fieldnames = [
        "schema_version",
        "phase_id",
        "run_id",
        "variant_id",
        "birth_family",
        "object_id",
        "feature_index",
        "micro_primitive_id",
        "scene_id",
        "window_id",
        "target_frame_id",
        "target_mask_id",
        "source_frame_id",
        "source_mask_id",
        "assignment_status",
        "uses_gt_for_prediction",
        "uses_future",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        rows = features["rows"]
        for idx in np.flatnonzero(labels >= 0).tolist():
            label = int(labels[idx])
            row = rows[idx]
            writer.writerow(
                {
                    "schema_version": "stream4d_v97_object_micro_primitive_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "birth_family": birth_family,
                    "object_id": f"{variant_id}_obj_{label:06d}",
                    "feature_index": idx,
                    "micro_primitive_id": row.get("micro_primitive_id", ""),
                    "scene_id": row.get("scene_id", ""),
                    "window_id": row.get("window_id", ""),
                    "target_frame_id": row.get("target_frame_id", ""),
                    "target_mask_id": row.get("target_mask_id", ""),
                    "source_frame_id": row.get("source_frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "assignment_status": "assigned",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase4_root = _project(args.phase4_root)
    features = _load_features(phase4_root)
    edges = _load_edges(phase4_root)
    variants: list[tuple[str, str, np.ndarray, str]] = []
    a0 = _labels_from_keys(features, lambda _idx, row: (row["scene_id"], row["window_id"], row["source_frame_id"], row["source_mask_id"]), int(args.min_object_micro_count))
    a2 = _labels_from_keys(features, lambda _idx, row: (row["scene_id"], row["window_id"], row.get("diagnostic_semantic_prototype_id", ""), row["source_mask_id"]), int(args.min_object_micro_count))
    a4 = _prune_same_frame_dominant(_drop_risky(a0, features), features)
    variants.extend(
        [
            ("A0_mask_cover_uniform_keys", "A_keypoint_cover", a0, "F5"),
            ("A2_geo_semantic_fps_keys", "A_keypoint_cover", a2, "F5"),
            ("A4_setcover_plus_risk_gate", "A_keypoint_cover", a4, "F5"),
        ]
    )
    b0 = _labels_from_edges(features, edges, score_name="F5", threshold=float(args.b0_threshold), min_size=int(args.min_object_micro_count), constrained=False)
    b1 = _labels_from_edges(features, edges, score_name="F5", threshold=float(args.b1_threshold), min_size=int(args.min_object_micro_count), constrained=True)
    b2 = _prune_same_frame_dominant(b1, features)
    b4 = _labels_from_edges(features, edges, score_name="F4", threshold=float(args.b4_threshold), min_size=int(args.min_object_micro_count), constrained=True)
    variants.extend(
        [
            ("B0_connected_components_diagnostic", "B_signed_constrained_clustering", b0, "F5"),
            ("B1_constrained_union_find", "B_signed_constrained_clustering", b1, "F5"),
            ("B2_constrained_union_find_with_postsplit", "B_signed_constrained_clustering", b2, "F5"),
            ("B4_scale_gate_object_only", "B_signed_constrained_clustering", b4, "F4"),
        ]
    )
    c0 = _expand_from_seeds(a4, edges, score_name="F5", threshold=float(args.c0_threshold), conflict_veto=False)
    c1 = _expand_from_seeds(a4, edges, score_name="F5", threshold=float(args.c1_threshold), conflict_veto=True)
    c4 = _prune_same_frame_dominant(c1, features)
    variants.extend(
        [
            ("C0_cover_seed_plus_affinity_expand", "C_hybrid_cover_cluster", c0, "F5"),
            ("C1_cover_seed_plus_signed_expand", "C_hybrid_cover_cluster", c1, "F5"),
            ("C4_cover_cluster_with_conflict_veto", "C_hybrid_cover_cluster", c4, "F5"),
        ]
    )

    baseline_metric = _metric_row(
        variant_id="A0_mask_cover_uniform_keys",
        birth_family="A_keypoint_cover",
        labels=a0,
        features=features,
        edge_stats=_edge_stats_for_labels(a0, edges, "F5"),
        same_frame_violation_count=_same_frame_violation(a0, features),
        baseline_object_count_per_window_mean=1.0,
        runtime_sec=0.0,
    )
    baseline_count = max(1.0, _num(baseline_metric["object_count_per_window_mean"]))
    baseline_plus_tolerance = float(args.fragmentation_baseline_tolerance)

    object_candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    micro_path = output_root / "object_micro_primitive_rows.csv"
    if micro_path.exists():
        micro_path.unlink()
    for variant_id, family, labels, score_name in variants:
        labels = _reindex_labels(labels)
        edge_stats = _edge_stats_for_labels(labels, edges, score_name)
        same_frame_violations = _same_frame_violation(labels, features)
        metric = _metric_row(
            variant_id=variant_id,
            birth_family=family,
            labels=labels,
            features=features,
            edge_stats=edge_stats,
            same_frame_violation_count=same_frame_violations,
            baseline_object_count_per_window_mean=baseline_count,
            runtime_sec=float(time.time() - started),
        )
        metric_rows.append(metric)
        object_candidate_rows.extend(
            _candidate_rows_for_variant(
                variant_id=variant_id,
                birth_family=family,
                labels=labels,
                features=features,
                edge_stats=edge_stats,
            )
        )
        gates = _gate_rows_for_metric(metric, baseline_plus_tolerance=baseline_plus_tolerance)
        gate_rows.extend(gates)
        failed = [row for row in gates if not _bool(row.get("pass"))]
        for fail in failed:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v97_object_birth_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "birth_family": family,
                    "failed_gate": fail["gate"],
                    "observed": fail["observed"],
                    "required": fail["required"],
                    "repair_direction": "follow Phase5 repair ladder: broad/risk gate, negative density, cannot-link audit, object NMS, or return Phase4 if all families fail",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        casebook_rows.append(
            {
                "schema_version": "stream4d_v97_object_birth_casebook_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "birth_family": family,
                "object_count": metric["object_count"],
                "object_count_per_window_mean": metric["object_count_per_window_mean"],
                "largest_object_ratio": metric["largest_object_ratio"],
                "coverage": metric["keypoint_coverage_rate"],
                "cannot_link_violation_count": metric["cannot_link_violation_count"],
                "same_frame_violation_count": metric["same_frame_violation_count"],
                "cluster_external_margin_mean": metric["cluster_external_margin_mean"],
                "diagnosis": "pass_candidate" if any(g["gate"] == "family_gate_to_phase6_diagnostic" and _bool(g["pass"]) for g in gates) else "failed_phase5_sanity",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        _write_micro_rows(micro_path, variant_id, family, labels, features)

    family_pass_rows = [row for row in gate_rows if row["gate"] == "family_gate_to_phase6_diagnostic" and _bool(row["pass"])]
    best_metric = max(metric_rows, key=lambda row: (int(_bool(next((g["pass"] for g in gate_rows if g["variant_id"] == row["variant_id"] and g["gate"] == "family_gate_to_phase6_diagnostic"), False))), _num(row["keypoint_coverage_rate"]), -_num(row["largest_object_ratio"])), default={})
    decision = "PASS_V97_PHASE5_OBJECT_BIRTH_DIAGNOSTIC" if family_pass_rows else "NO_GO_V97_PHASE5_OBJECT_BIRTH_DIAGNOSTIC"
    candidate_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "variant_id",
        "scene_id",
        "window_id",
        "object_id",
        "birth_family",
        "micro_primitive_count",
        "frame_support_count",
        "masklet_support_count",
        "mean_affinity_internal",
        "mean_affinity_external_margin",
        "cannot_link_violation_count",
        "source_mask_count",
        "semantic_proto_entropy",
        "d4rt_visibility_mean",
        "support_area_ratio_mean",
        "score",
        "uses_gt_for_prediction",
        "uses_future",
    ]
    _write_csv(output_root / "object_candidate_rows.csv", object_candidate_rows, candidate_fields)
    _write_csv(output_root / "object_birth_metric_rows.csv", metric_rows, list(metric_rows[0].keys()) if metric_rows else [])
    _write_csv(output_root / "object_birth_gate_rows.csv", gate_rows, list(gate_rows[0].keys()) if gate_rows else [])
    _write_csv(output_root / "object_birth_failure_rows.csv", failure_rows, list(failure_rows[0].keys()) if failure_rows else ["schema_version"])
    _write_csv(output_root / "object_birth_casebook_rows.csv", casebook_rows, list(casebook_rows[0].keys()) if casebook_rows else [])
    summary = {
        "schema": "stream4d_v97_phase5_object_birth_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "output_root": _rel(output_root),
        "phase4_root": _rel(phase4_root),
        "semantic_source": "radio_mask_feature_region_proxy",
        "semantic_tensor_loaded": False,
        "diagnostic_scope": "region_proxy_phase4_500k",
        "feature_count": int(features["n"]),
        "affinity_edge_count": int(len(edges["p"])),
        "variant_count": len(variants),
        "evaluated_families": sorted(set(family for _variant_id, family, _labels, _score in variants)),
        "passing_variant_count": len(family_pass_rows),
        "passing_variants": [row["variant_id"] for row in family_pass_rows],
        "best_variant": best_metric,
        "metric_rows": metric_rows,
        "gate_rows": gate_rows,
        "runtime_sec": float(time.time() - started),
        "can_enter_phase6_diagnostic": bool(family_pass_rows),
        "can_enter_phase6_full": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "passing_variant_count": len(family_pass_rows),
                "best_variant": best_metric.get("variant_id", ""),
                "runtime_sec": summary["runtime_sec"],
                "output_root": _rel(output_root),
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--min-object-micro-count", type=int, default=32)
    parser.add_argument("--b0-threshold", type=float, default=0.50)
    parser.add_argument("--b1-threshold", type=float, default=0.55)
    parser.add_argument("--b4-threshold", type=float, default=0.60)
    parser.add_argument("--c0-threshold", type=float, default=0.55)
    parser.add_argument("--c1-threshold", type=float, default=0.55)
    parser.add_argument("--fragmentation-baseline-tolerance", type=float, default=1.35)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
