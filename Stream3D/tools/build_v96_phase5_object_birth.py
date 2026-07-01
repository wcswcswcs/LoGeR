#!/usr/bin/env python3
"""Build v96 Phase5 object birth candidates from micro affinity evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


PHASE_ID = "v96_phase5_object_birth"
RUN_ID = "v96_phase5_object_birth"
DEFAULT_PHASE2_ROOTS = [
    ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_f5_35_full_gpu6_active_sparse_r4",
    ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_f40_70_full_gpu6_active_sparse_r4",
    ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_scene0011_f40_55_full_gpu6_active_sparse_r4",
    ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_scene0011_f60_70_full_gpu7_active_sparse_r4",
    ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_f75_100_full_gpu7_active_sparse_r4",
]
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"
DEFAULT_FEATURE = ROOT / "outputs/audit/v96_phase4_affinity_features_w0020_segmented_r4_D3"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase5_object_birth"


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


def _parse_include(raw: str) -> tuple[Path, set[str]]:
    parts = raw.split("::")
    root = _project(parts[0])
    scenes: set[str] = set()
    for part in parts[1:]:
        if part.startswith("scene="):
            scenes.add(part.split("=", 1)[1])
        elif part:
            raise ValueError(f"unknown include filter {part!r}; use ::scene=<scene_id>")
    return root, scenes


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


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_path_lookup(source_rows: Path) -> dict[tuple[str, str, int], Path]:
    out: dict[tuple[str, str, int], Path] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        raw = row.get("mask_path", "")
        if raw and key not in out:
            out[key] = _project(raw)
    return out


def _load_micro_query_info(includes: list[str], decode_variant: str) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for raw in includes:
        root, scenes = _parse_include(raw)
        path = root / "micro_query_rows.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing micro_query_rows.csv: {path}")
        for row in _read_csv(path):
            if row.get("decode_variant") != decode_variant:
                continue
            if scenes and row.get("scene_id", "") not in scenes:
                continue
            query_id = row.get("query_id", "")
            if not query_id or query_id in info:
                continue
            info[query_id] = {
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "source_frame_id": int(_num(row.get("frame_id"))),
                "source_mask_id": int(_num(row.get("source_mask_id_optional"))),
                "query_stratum": row.get("query_stratum", ""),
                "query_priority": row.get("query_priority", ""),
            }
    return info


def _load_incidence_events(root: Path, decode_variant: str) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(root / "incidence_event_rows.csv"):
        if row.get("decode_variant") == decode_variant:
            rows.append(row)
    return rows


def _load_feature_rows(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    index_to_query: dict[int, str] = {}
    query_to_index: dict[str, int] = {}
    for row in _read_csv(root / "micro_feature_rows.csv"):
        idx = int(_num(row.get("feature_index")))
        qid = row.get("query_id", "")
        index_to_query[idx] = qid
        query_to_index[qid] = idx
    return index_to_query, query_to_index


def _load_affinity_components(root: Path, index_to_query: dict[int, str], threshold: float) -> dict[str, int]:
    dsu = DSU(len(index_to_query))
    for row in _read_csv(root / "micro_affinity_edge_rows.csv"):
        if row.get("feature_variant") != "F5_full_signed_proxy":
            continue
        if _num(row.get("positive_score")) <= 0.0 or _num(row.get("conflict_score")) > 0.0:
            continue
        if _num(row.get("signed_affinity")) < threshold:
            continue
        a = int(_num(row.get("feature_index_p")))
        b = int(_num(row.get("feature_index_q")))
        if a in index_to_query and b in index_to_query:
            dsu.union(a, b)
    root_to_cluster: dict[int, int] = {}
    query_cluster: dict[str, int] = {}
    for idx, qid in index_to_query.items():
        root = dsu.find(idx)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster) + 1
        query_cluster[qid] = root_to_cluster[root]
    return query_cluster


def _groups_from_source_masks(query_info: dict[str, dict[str, Any]], min_queries: int) -> dict[str, set[str]]:
    grouped: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for qid, info in query_info.items():
        mask_id = int(info.get("source_mask_id", 0))
        if mask_id <= 0:
            continue
        grouped[(info["scene_id"], info["window_id"], int(info["source_frame_id"]), mask_id)].add(qid)
    out: dict[str, set[str]] = {}
    for idx, (key, qids) in enumerate(sorted(grouped.items()), start=1):
        if len(qids) >= min_queries:
            scene, window, frame_id, mask_id = key
            out[f"A:{scene}:{window}:f{frame_id}:m{mask_id}:o{idx}"] = qids
    return out


def _groups_from_affinity(query_cluster: dict[str, int], min_queries: int) -> dict[str, set[str]]:
    grouped: dict[int, set[str]] = defaultdict(set)
    for qid, cid in query_cluster.items():
        grouped[int(cid)].add(qid)
    out: dict[str, set[str]] = {}
    for cid, qids in sorted(grouped.items()):
        if len(qids) >= min_queries:
            out[f"B:cluster{cid}"] = qids
    return out


def _events_by_query(events: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        out[row.get("query_id", "")].append(row)
    return out


def _select_masklets_for_groups(
    groups: dict[str, set[str]],
    events_by_query: dict[str, list[dict[str, str]]],
    *,
    family: str,
    min_mask_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]], int]:
    raw_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for obj_index, (group_id, qids) in enumerate(sorted(groups.items()), start=1):
        object_id = f"{family}_obj_{obj_index:06d}"
        per_frame: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
        for qid in qids:
            for ev in events_by_query.get(qid, []):
                mask_id = int(_num(ev.get("center_mask_id")))
                if mask_id <= 0:
                    continue
                key = (ev.get("scene_id", ""), ev.get("window_id", ""), int(_num(ev.get("target_frame_id"))))
                per_frame[key][mask_id] += 1
        selected_count = 0
        support_sum = 0
        for key, counts in sorted(per_frame.items()):
            mask_id, support = counts.most_common(1)[0]
            if support < min_mask_support:
                continue
            scene, window, frame_id = key
            score = float(support / max(1, len(qids)))
            raw_rows.append(
                {
                    "schema_version": "stream4d_v96_selected_masklet_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "object_id": object_id,
                    "group_id": group_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "selected_mask_id": int(mask_id),
                    "masklet_support_query_count": int(support),
                    "object_query_count": int(len(qids)),
                    "masklet_score": score,
                    "selection_status": "pre_collision_resolution",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            selected_count += 1
            support_sum += support
        if selected_count > 0:
            object_rows.append(
                {
                    "schema_version": "stream4d_v96_object_candidate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "object_id": object_id,
                    "group_id": group_id,
                    "micro_query_count": int(len(qids)),
                    "selected_frame_count_before_collision_resolution": int(selected_count),
                    "masklet_support_query_count_sum": int(support_sum),
                    "object_score": float(support_sum / max(1, selected_count * len(qids))),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    by_frame_mask: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_frame_mask[(row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"]))].append(row)
    collision_count = sum(max(0, len(vals) - 1) for vals in by_frame_mask.values())
    kept_rows: list[dict[str, Any]] = []
    kept_objects: dict[str, set[str]] = defaultdict(set)
    for vals in by_frame_mask.values():
        best = max(vals, key=lambda row: (float(row["masklet_score"]), int(row["masklet_support_query_count"]), row["object_id"]))
        out = dict(best)
        out["selection_status"] = "selected_after_collision_resolution"
        kept_rows.append(out)
        kept_objects[str(out["object_id"])].add(f'{out["scene_id"]}|{out["window_id"]}|{int(out["frame_id"])}|{int(out["selected_mask_id"])}')
    object_rows = [row for row in object_rows if str(row["object_id"]) in kept_objects]
    for row in object_rows:
        row["selected_frame_count"] = len(kept_objects[str(row["object_id"])])
    return object_rows, kept_rows, kept_objects, collision_count


def _merge_duplicate_groups(
    groups: dict[str, set[str]],
    events_by_query: dict[str, list[dict[str, str]]],
    min_mask_support: int,
    overlap_threshold: float,
) -> dict[str, set[str]]:
    object_rows, masklet_rows, signatures, _collisions = _select_masklets_for_groups(
        groups,
        events_by_query,
        family="C_preview_merge_probe",
        min_mask_support=min_mask_support,
    )
    object_ids = [str(row["object_id"]) for row in object_rows]
    group_ids = [str(row["group_id"]) for row in object_rows]
    id_to_group = dict(zip(object_ids, group_ids))
    dsu = DSU(len(object_ids))
    for i, oid in enumerate(object_ids):
        sig_i = signatures.get(oid, set())
        if not sig_i:
            continue
        for j in range(i + 1, len(object_ids)):
            sig_j = signatures.get(object_ids[j], set())
            if not sig_j:
                continue
            overlap = len(sig_i & sig_j) / max(1, min(len(sig_i), len(sig_j)))
            if overlap >= overlap_threshold:
                dsu.union(i, j)
    merged: dict[int, set[str]] = defaultdict(set)
    for idx, oid in enumerate(object_ids):
        gid = id_to_group[oid]
        merged[dsu.find(idx)].update(groups.get(gid, set()))
    return {f"C:merged{idx + 1:06d}": qids for idx, qids in enumerate(merged.values()) if qids}


def _masklet_overlap_components(
    events: list[dict[str, str]],
    *,
    min_node_support: int,
    min_shared_queries: int,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    node_qids: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for row in events:
        mask_id = int(_num(row.get("center_mask_id")))
        if mask_id <= 0:
            continue
        key = (
            row.get("scene_id", ""),
            row.get("window_id", ""),
            int(_num(row.get("target_frame_id"))),
            mask_id,
        )
        node_qids[key].add(row.get("query_id", ""))
    node_keys = sorted([key for key, qids in node_qids.items() if len(qids) >= int(min_node_support)])
    node_index = {key: idx for idx, key in enumerate(node_keys)}
    qid_to_nodes: dict[str, list[int]] = defaultdict(list)
    for key in node_keys:
        idx = node_index[key]
        for qid in node_qids[key]:
            qid_to_nodes[qid].append(idx)

    edge_counts: Counter[tuple[int, int]] = Counter()
    for idxs in qid_to_nodes.values():
        if len(idxs) < 2:
            continue
        unique = sorted(set(idxs), key=lambda idx: (node_keys[idx][0], node_keys[idx][1], node_keys[idx][2], node_keys[idx][3]))
        for pos, a in enumerate(unique):
            scene_a, window_a, frame_a, _mask_a = node_keys[a]
            for b in unique[pos + 1 :]:
                scene_b, window_b, frame_b, _mask_b = node_keys[b]
                if scene_a != scene_b or window_a != window_b or frame_a == frame_b:
                    continue
                edge_counts[(a, b)] += 1

    candidate_edges: list[tuple[float, float, int, int, int]] = []
    for (a, b), shared in edge_counts.items():
        support_a = len(node_qids[node_keys[a]])
        support_b = len(node_qids[node_keys[b]])
        overlap = float(shared / max(1, min(support_a, support_b)))
        jaccard = float(shared / max(1, support_a + support_b - shared))
        if shared >= int(min_shared_queries) and overlap >= float(overlap_threshold):
            candidate_edges.append((overlap, jaccard, int(shared), a, b))
    candidate_edges.sort(reverse=True)

    dsu = DSU(len(node_keys))
    frame_masks: list[dict[tuple[str, str, int], int]] = [
        {(scene, window, frame_id): mask_id} for scene, window, frame_id, mask_id in node_keys
    ]
    skipped_conflict = 0
    accepted_edges = 0
    for _overlap, _jaccard, _shared, a, b in candidate_edges:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        maps_a = frame_masks[ra]
        maps_b = frame_masks[rb]
        conflict = False
        if len(maps_a) > len(maps_b):
            probe_a, probe_b = maps_b, maps_a
        else:
            probe_a, probe_b = maps_a, maps_b
        for frame_key, mask_id in probe_a.items():
            other = probe_b.get(frame_key)
            if other is not None and int(other) != int(mask_id):
                conflict = True
                break
        if conflict:
            skipped_conflict += 1
            continue
        dsu.union(ra, rb)
        new_root = dsu.find(ra)
        old_root = rb if new_root == ra else ra
        merged = dict(frame_masks[new_root])
        merged.update(frame_masks[old_root])
        frame_masks[new_root] = merged
        accepted_edges += 1

    grouped_nodes: dict[int, list[tuple[str, str, int, int]]] = defaultdict(list)
    for idx, key in enumerate(node_keys):
        grouped_nodes[dsu.find(idx)].append(key)
    components: list[dict[str, Any]] = []
    for comp_idx, keys in enumerate(sorted(grouped_nodes.values(), key=lambda vals: (vals[0][0], vals[0][1], vals[0][2], vals[0][3])), start=1):
        qids: set[str] = set()
        support_sum = 0
        for key in keys:
            qids.update(node_qids[key])
            support_sum += len(node_qids[key])
        components.append(
            {
                "group_id": f"C:masklet_overlap_component:{comp_idx:06d}",
                "node_keys": sorted(keys),
                "qids": qids,
                "support_sum": support_sum,
            }
        )
    diag = {
        "masklet_node_count": len(node_keys),
        "raw_overlap_edge_count": len(edge_counts),
        "candidate_overlap_edge_count": len(candidate_edges),
        "accepted_overlap_edge_count": accepted_edges,
        "skipped_same_frame_conflict_edge_count": skipped_conflict,
        "component_count": len(components),
        "min_node_support": int(min_node_support),
        "min_shared_queries": int(min_shared_queries),
        "overlap_threshold": float(overlap_threshold),
    }
    return components, diag


def _select_masklets_from_components(
    components: list[dict[str, Any]],
    events_by_query: dict[str, list[dict[str, str]]],
    *,
    family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]], int]:
    object_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    groups: dict[str, set[str]] = {}
    violation_count = 0
    for obj_index, component in enumerate(components, start=1):
        object_id = f"{family}_obj_{obj_index:06d}"
        group_id = str(component["group_id"])
        qids = set(component["qids"])
        groups[group_id] = qids
        seen_frame_masks: dict[tuple[str, str, int], int] = {}
        support_sum = 0
        for scene, window, frame_id, mask_id in component["node_keys"]:
            frame_key = (scene, window, int(frame_id))
            existing = seen_frame_masks.get(frame_key)
            if existing is not None and int(existing) != int(mask_id):
                violation_count += 1
                continue
            seen_frame_masks[frame_key] = int(mask_id)
            support = 0
            for qid in qids:
                for ev in events_by_query.get(qid, []):
                    if (
                        ev.get("scene_id", "") == scene
                        and ev.get("window_id", "") == window
                        and int(_num(ev.get("target_frame_id"))) == int(frame_id)
                        and int(_num(ev.get("center_mask_id"))) == int(mask_id)
                    ):
                        support += 1
                        break
            support_sum += support
            selected_rows.append(
                {
                    "schema_version": "stream4d_v96_selected_masklet_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "object_id": object_id,
                    "group_id": group_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "selected_mask_id": int(mask_id),
                    "masklet_support_query_count": int(support),
                    "object_query_count": int(len(qids)),
                    "masklet_score": float(support / max(1, len(qids))),
                    "selection_status": "selected_by_masklet_overlap_component",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": family,
                "object_id": object_id,
                "group_id": group_id,
                "micro_query_count": int(len(qids)),
                "selected_frame_count_before_collision_resolution": int(len(component["node_keys"])),
                "masklet_support_query_count_sum": int(support_sum),
                "object_score": float(support_sum / max(1, len(component["node_keys"]) * max(1, len(qids)))),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "selected_frame_count": int(len(seen_frame_masks)),
            }
        )
    return object_rows, selected_rows, groups, violation_count


def _evaluate_masklets(
    selected_rows: list[dict[str, Any]],
    mask_lookup: dict[tuple[str, str, int], Path],
    *,
    min_pred_pixels: int,
    min_gt_pixels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    object_index: dict[str, int] = {}
    for row in selected_rows:
        oid = str(row["object_id"])
        if oid not in object_index:
            object_index[oid] = len(object_index) + 1
        by_frame[(row["scene_id"], row["window_id"], int(row["frame_id"]))].append(row)
    scene_gt_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in by_frame}))}
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_frame.items()):
        scene, window, frame_id = key
        mask_path = mask_lookup.get(key)
        if mask_path is None or not mask_path.exists():
            frame_rows.append(
                {
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "status": "missing_mask",
                    "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_eval": True,
                }
            )
            continue
        label = _load_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in rows:
            pred[label == int(row["selected_mask_id"])] = object_index[str(row["object_id"])]
        gt = _load_gt_2d(scene, frame_id, label.shape)
        gt = np.where(gt > 0, gt + int(scene_gt_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "status": "evaluated",
                "selected_mask_count": len(rows),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_eval": True,
                "gt_scene_offset": int(scene_gt_offsets.get(scene, 0)),
            }
        )
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        score_mode="constant",
        input_scores=None,
    )
    return summary, frame_rows


def _baseline_frame_masks(
    frame_keys: set[tuple[str, str, int]],
    mask_lookup: dict[tuple[str, str, int], Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, window, frame_id in sorted(frame_keys):
        mask_path = mask_lookup.get((scene, window, frame_id))
        if mask_path is None or not mask_path.exists():
            continue
        label = _load_label(mask_path)
        for mask_id in sorted(int(v) for v in np.unique(label) if int(v) > 0):
            rows.append(
                {
                    "family": "B0_same_scope_frame_mask_baseline",
                    "object_id": f"B0_{scene}_{window}_{frame_id}_{mask_id}",
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "selected_mask_id": mask_id,
                    "masklet_score": 1.0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _metric_row(
    *,
    family: str,
    object_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    collision_count: int,
    eval_summary: dict[str, Any],
    b0_object_count: int,
    b0_sf50: float,
    b0_ap50: float,
    runtime_sec: float,
) -> dict[str, Any]:
    object_count = len({str(row["object_id"]) for row in selected_rows})
    counts = Counter(str(row["object_id"]) for row in selected_rows)
    per_object = np.asarray(list(counts.values()), dtype=np.float32)
    largest = float(np.max(per_object) / max(1.0, float(np.sum(per_object)))) if per_object.size else 0.0
    sf50 = (eval_summary.get("score_free_match_at_050") or {}).get("recall")
    ap50 = eval_summary.get("ap50")
    return {
        "schema_version": "stream4d_v96_object_birth_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "family": family,
        "object_count": object_count,
        "B0_object_count_same_scope": b0_object_count,
        "object_count_ratio_vs_B0": float(object_count / max(1, b0_object_count)),
        "micro_primitives_per_object_mean": float(np.mean([_num(row.get("micro_query_count")) for row in object_rows])) if object_rows else 0.0,
        "micro_primitives_per_object_p90": float(np.percentile([_num(row.get("micro_query_count")) for row in object_rows], 90)) if object_rows else 0.0,
        "selected_masklet_count": len(selected_rows),
        "largest_object_ratio": largest,
        "cannot_link_violation_count": 0,
        "pre_resolution_collision_count": collision_count,
        "small_object_retention_rate": "",
        "broad_mask_selected_rate": "",
        "cluster_fragmentation_proxy": float(object_count / max(1, b0_object_count)),
        "cluster_overmerge_proxy": largest,
        "preview_MV_AP": eval_summary.get("ap"),
        "preview_MV_AP50": ap50,
        "preview_MV_AP25": eval_summary.get("ap25"),
        "preview_ScoreFreeMatch50": sf50,
        "B0_preview_ScoreFreeMatch50": b0_sf50,
        "B0_preview_MV_AP50": b0_ap50,
        "score_free_gate_pass": bool(sf50 is not None and float(sf50) >= float(b0_sf50) + 0.03),
        "ap50_preview_gate_pass": bool(ap50 is not None and b0_ap50 is not None and float(ap50) >= float(b0_ap50) + 0.02),
        "runtime_object_birth_sec": runtime_sec,
        "uses_gt_for_prediction": False,
        "uses_gt_for_preview_eval": True,
        "uses_future": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase2_includes = args.phase2_root or [str(p) for p in DEFAULT_PHASE2_ROOTS]
    query_info = _load_micro_query_info(phase2_includes, args.decode_variant)
    events = _load_incidence_events(_project(args.incidence_root), args.decode_variant)
    events_by_query = _events_by_query(events)
    index_to_query, _query_to_index = _load_feature_rows(_project(args.feature_root))
    query_cluster = _load_affinity_components(_project(args.feature_root), index_to_query, float(args.affinity_union_threshold))
    mask_lookup = _mask_path_lookup(_project(args.source_rows))

    groups_a = _groups_from_source_masks(query_info, int(args.min_source_seed_queries))
    groups_b = _groups_from_affinity(query_cluster, int(args.min_affinity_cluster_queries))
    overlap_components, overlap_component_diag = _masklet_overlap_components(
        events,
        min_node_support=int(args.min_masklet_node_support),
        min_shared_queries=int(args.masklet_overlap_min_shared_queries),
        overlap_threshold=float(args.masklet_overlap_threshold),
    )
    family_groups = {
        "A_set_cover_source_mask": groups_a,
        "B_constrained_affinity_clustering": groups_b,
    }

    frame_keys = {(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("target_frame_id")))) for row in events}
    b0_rows = _baseline_frame_masks(frame_keys, mask_lookup)
    b0_eval, b0_frame_rows = _evaluate_masklets(b0_rows, mask_lookup, min_pred_pixels=int(args.min_pred_pixels), min_gt_pixels=int(args.min_gt_pixels))
    b0_sf50 = float((b0_eval.get("score_free_match_at_050") or {}).get("recall") or 0.0)
    b0_ap50 = float(b0_eval.get("ap50") or 0.0)
    b0_object_count = len({str(row["object_id"]) for row in b0_rows})

    object_candidate_rows: list[dict[str, Any]] = []
    micro_cluster_rows: list[dict[str, Any]] = []
    selected_masklet_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    preview_frame_rows: list[dict[str, Any]] = []
    family_summaries: dict[str, Any] = {}
    for family, groups in family_groups.items():
        object_rows, selected_rows, _signatures, collisions = _select_masklets_for_groups(
            groups,
            events_by_query,
            family=family,
            min_mask_support=int(args.min_mask_support),
        )
        eval_summary, frame_rows = _evaluate_masklets(
            selected_rows,
            mask_lookup,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
        )
        runtime_sec = float(time.time() - started)
        metric = _metric_row(
            family=family,
            object_rows=object_rows,
            selected_rows=selected_rows,
            collision_count=collisions,
            eval_summary=eval_summary,
            b0_object_count=b0_object_count,
            b0_sf50=b0_sf50,
            b0_ap50=b0_ap50,
            runtime_sec=runtime_sec,
        )
        object_candidate_rows.extend(object_rows)
        selected_masklet_rows.extend(selected_rows)
        metric_rows.append(metric)
        preview_frame_rows.extend([{**row, "family": family} for row in frame_rows])
        for group_id, qids in groups.items():
            micro_cluster_rows.append(
                {
                    "schema_version": "stream4d_v96_micro_cluster_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "group_id": group_id,
                    "micro_query_count": len(qids),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
        )
        family_summaries[family] = eval_summary

    family = "C_hybrid_cover_cluster"
    c_object_rows, c_selected_rows, c_groups, c_violations = _select_masklets_from_components(
        overlap_components,
        events_by_query,
        family=family,
    )
    c_eval_summary, c_frame_rows = _evaluate_masklets(
        c_selected_rows,
        mask_lookup,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
    )
    c_metric = _metric_row(
        family=family,
        object_rows=c_object_rows,
        selected_rows=c_selected_rows,
        collision_count=c_violations,
        eval_summary=c_eval_summary,
        b0_object_count=b0_object_count,
        b0_sf50=b0_sf50,
        b0_ap50=b0_ap50,
        runtime_sec=float(time.time() - started),
    )
    c_metric["cannot_link_violation_count"] = int(c_violations)
    c_metric["masklet_overlap_component_count"] = int(overlap_component_diag["component_count"])
    c_metric["masklet_overlap_accepted_edge_count"] = int(overlap_component_diag["accepted_overlap_edge_count"])
    object_candidate_rows.extend(c_object_rows)
    selected_masklet_rows.extend(c_selected_rows)
    metric_rows.append(c_metric)
    preview_frame_rows.extend([{**row, "family": family} for row in c_frame_rows])
    for group_id, qids in c_groups.items():
        micro_cluster_rows.append(
            {
                "schema_version": "stream4d_v96_micro_cluster_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": family,
                "group_id": group_id,
                "micro_query_count": len(qids),
                "cluster_source": "masklet_overlap_constrained_component",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    family_summaries[family] = c_eval_summary

    b0_metric = _metric_row(
        family="B0_same_scope_frame_mask_baseline",
        object_rows=[],
        selected_rows=b0_rows,
        collision_count=0,
        eval_summary=b0_eval,
        b0_object_count=b0_object_count,
        b0_sf50=b0_sf50,
        b0_ap50=b0_ap50,
        runtime_sec=float(time.time() - started),
    )
    metric_rows.insert(0, b0_metric)
    preview_frame_rows.extend([{**row, "family": "B0_same_scope_frame_mask_baseline"} for row in b0_frame_rows])

    gate_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        if row["family"].startswith("B0_"):
            continue
        family = row["family"]
        count_ratio = _num(row.get("object_count_ratio_vs_B0"))
        family_pass = (
            0.5 <= count_ratio <= 2.5
            and int(row.get("cannot_link_violation_count", 0)) == 0
            and _num(row.get("largest_object_ratio")) <= 0.25
            and (_bool(row.get("score_free_gate_pass")) or _bool(row.get("ap50_preview_gate_pass")))
        )
        gate_rows.extend(
            [
                {
                    "schema_version": "stream4d_v96_phase5_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "gate": "object_count_reasonable_0p5x_to_2p5x_B0",
                    "pass": bool(0.5 <= count_ratio <= 2.5),
                    "observed": count_ratio,
                    "required": "0.5 <= object_count/B0_object_count <= 2.5",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v96_phase5_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "gate": "cannot_link_violation_count_eq_0",
                    "pass": int(row.get("cannot_link_violation_count", 0)) == 0,
                    "observed": row.get("cannot_link_violation_count", 0),
                    "required": 0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v96_phase5_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "gate": "largest_object_ratio_le_0p25",
                    "pass": bool(_num(row.get("largest_object_ratio")) <= 0.25),
                    "observed": row.get("largest_object_ratio", ""),
                    "required": 0.25,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v96_phase5_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "gate": "preview_score_free_or_ap50_ge_B0_margin",
                    "pass": bool(_bool(row.get("score_free_gate_pass")) or _bool(row.get("ap50_preview_gate_pass"))),
                    "observed": f"SF50={row.get('preview_ScoreFreeMatch50')} AP50={row.get('preview_MV_AP50')}",
                    "required": "SF50 >= B0 + 0.03 OR AP50 >= B0 + 0.02",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_preview_eval": True,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v96_phase5_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": family,
                    "gate": "family_gate_to_phase6",
                    "pass": bool(family_pass),
                    "observed": family_pass,
                    "required": "all structural gates and preview metric gate",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_preview_eval": True,
                    "uses_future": False,
                },
            ]
        )
    best_family = max(
        [row for row in metric_rows if not str(row["family"]).startswith("B0_")],
        key=lambda row: (_num(row.get("preview_ScoreFreeMatch50")), _num(row.get("preview_MV_AP50"))),
        default={},
    )
    phase5_pass = any(row.get("gate") == "family_gate_to_phase6" and _bool(row.get("pass")) for row in gate_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_candidate_rows)
    _write_csv(output_root / "micro_cluster_rows.csv", micro_cluster_rows)
    _write_csv(output_root / "selected_masklet_rows.csv", selected_masklet_rows)
    _write_csv(output_root / "object_birth_metric_rows.csv", metric_rows)
    _write_csv(output_root / "phase5_gate_rows.csv", gate_rows)
    _write_csv(output_root / "preview_frame_rows.csv", preview_frame_rows)
    summary = {
        "schema": "stream4d_v96_phase5_object_birth_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE5_OBJECT_BIRTH" if phase5_pass else "NO_GO_V96_PHASE5_OBJECT_BIRTH",
        "output_root": _rel(output_root),
        "phase2_includes": phase2_includes,
        "incidence_root": _rel(_project(args.incidence_root)),
        "feature_root": _rel(_project(args.feature_root)),
        "query_info_count": len(query_info),
        "incidence_event_count": len(events),
        "B0_object_count_same_scope": b0_object_count,
        "B0_preview_ScoreFreeMatch50": b0_sf50,
        "B0_preview_MV_AP50": b0_ap50,
        "best_family": best_family,
        "metric_rows": metric_rows,
        "gate_rows": gate_rows,
        "masklet_overlap_component_diag": overlap_component_diag,
        "preview_scope": "w0020_segmented_active_frames_only",
        "uses_gt_for_prediction": False,
        "uses_gt_for_preview_eval": True,
        "uses_future": False,
        "runtime_total_sec": float(time.time() - started),
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "best_family": best_family.get("family", ""), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 Phase5 object birth candidates.")
    parser.add_argument("--phase2-root", action="append")
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--affinity-union-threshold", type=float, default=0.90)
    parser.add_argument("--min-source-seed-queries", type=int, default=16)
    parser.add_argument("--min-affinity-cluster-queries", type=int, default=8)
    parser.add_argument("--min-mask-support", type=int, default=4)
    parser.add_argument("--merge-overlap-threshold", type=float, default=0.50)
    parser.add_argument("--min-masklet-node-support", type=int, default=4)
    parser.add_argument("--masklet-overlap-min-shared-queries", type=int, default=4)
    parser.add_argument("--masklet-overlap-threshold", type=float, default=0.30)
    parser.add_argument("--min-pred-pixels", type=int, default=64)
    parser.add_argument("--min-gt-pixels", type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
