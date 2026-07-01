#!/usr/bin/env python3
"""Repair v96 C-family object birth by semantic/spatial/D4RT masklet merging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase5_semantic_spatial_d4rt_merge_repair"
RUN_ID = "v96_phase5_semantic_spatial_d4rt_merge_repair"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SEMANTIC_ROWS = ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
        if key not in out and row.get("mask_path"):
            out[key] = _project(row["mask_path"])
    return out


def _semantic_lookup(path: Path, wanted: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("scene_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("mask_id"))))
            if key in wanted and key not in out:
                out[key] = row
    return out


def _load_incidence_qids(root: Path, decode_variant: str, wanted_nodes: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], set[str]]:
    out: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for row in _read_csv(root / "incidence_event_rows.csv"):
        if row.get("decode_variant") != decode_variant:
            continue
        mask_id = int(_num(row.get("center_mask_id")))
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("target_frame_id"))), mask_id)
        if key in wanted_nodes and mask_id > 0:
            out[key].add(row.get("query_id", ""))
    return out


def _node_geometry(label: np.ndarray, mask_id: int) -> tuple[float, float, float, int]:
    ys, xs = np.nonzero(label == int(mask_id))
    area = int(xs.size)
    if area == 0:
        return 0.0, 0.0, 0.0, 0
    return float(np.mean(xs)), float(np.mean(ys)), float(area / max(1, label.size)), area


def _safe_proto(feature: dict[str, str] | None) -> str:
    if not feature or not _bool(feature.get("feature_available")):
        return "semantic_missing"
    return feature.get("semantic_prototype_id") or "semantic_missing"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    phase5_root = _project(args.phase5_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows_path = _project(args.source_rows)
    selected_source = [
        row for row in _read_csv(phase5_root / "selected_masklet_rows.csv") if row.get("family") == args.source_family
    ]
    if not selected_source:
        raise ValueError(f"no rows for source family {args.source_family}")
    wanted_sem = {
        (row.get("scene_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id"))))
        for row in selected_source
    }
    semantic = _semantic_lookup(_project(args.semantic_feature_rows), wanted_sem)
    mask_paths = _mask_path_lookup(source_rows_path)
    label_cache: dict[tuple[str, str, int], np.ndarray] = {}
    node_rows: list[dict[str, Any]] = []
    wanted_nodes: set[tuple[str, str, int, int]] = set()
    for idx, row in enumerate(selected_source):
        frame_key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        mask_id = int(_num(row.get("selected_mask_id")))
        path = mask_paths.get(frame_key)
        if path is None:
            cx = cy = area_ratio = 0.0
            area_px = 0
            width = height = 1
        else:
            if frame_key not in label_cache:
                label_cache[frame_key] = _load_label(path)
            label = label_cache[frame_key]
            height, width = int(label.shape[0]), int(label.shape[1])
            cx, cy, area_ratio, area_px = _node_geometry(label, mask_id)
        sem_key = (row.get("scene_id", ""), int(_num(row.get("frame_id"))), mask_id)
        proto = _safe_proto(semantic.get(sem_key))
        node_key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), mask_id)
        wanted_nodes.add(node_key)
        node_rows.append(
            {
                "node_index": idx,
                "source_row": row,
                "node_key": node_key,
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "frame_id": int(_num(row.get("frame_id"))),
                "selected_mask_id": mask_id,
                "semantic_prototype_id": proto,
                "semantic_broad_background_risk": _bool((semantic.get(sem_key) or {}).get("broad_background_risk")),
                "cx": cx,
                "cy": cy,
                "image_width": width,
                "image_height": height,
                "area_ratio": area_ratio,
                "area_px": area_px,
            }
        )
    node_qids = _load_incidence_qids(_project(args.incidence_root), args.decode_variant, wanted_nodes)
    for node in node_rows:
        node["qid_count"] = len(node_qids.get(node["node_key"], set()))

    dsu = DSU(len(node_rows))
    frame_masks: list[dict[tuple[str, str, int], int]] = [
        {(node["scene_id"], node["window_id"], int(node["frame_id"])): int(node["selected_mask_id"])}
        for node in node_rows
    ]
    initial_source_union_count = 0
    initial_source_conflict_count = 0
    by_source_object: dict[str, list[int]] = defaultdict(list)
    for node in node_rows:
        by_source_object[str(node["source_row"].get("object_id", ""))].append(int(node["node_index"]))
    for _source_object_id, idxs in by_source_object.items():
        if len(idxs) < 2:
            continue
        root_idx = idxs[0]
        for other_idx in idxs[1:]:
            ra, rb = dsu.find(root_idx), dsu.find(other_idx)
            if ra == rb:
                continue
            conflict = False
            for frame_key, mask_id in (frame_masks[ra].items() if len(frame_masks[ra]) <= len(frame_masks[rb]) else frame_masks[rb].items()):
                other = (frame_masks[rb] if len(frame_masks[ra]) <= len(frame_masks[rb]) else frame_masks[ra]).get(frame_key)
                if other is not None and int(other) != int(mask_id):
                    conflict = True
                    break
            if conflict:
                initial_source_conflict_count += 1
                continue
            new_root = dsu.union(ra, rb)
            old_root = rb if new_root == ra else ra
            merged = dict(frame_masks[new_root])
            merged.update(frame_masks[old_root])
            frame_masks[new_root] = merged
            initial_source_union_count += 1

    candidate_rows: list[dict[str, Any]] = []
    by_proto: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for node in node_rows:
        by_proto[(node["scene_id"], node["window_id"], node["semantic_prototype_id"])].append(int(node["node_index"]))

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for (_scene, _window, proto), idxs in by_proto.items():
        if proto == "semantic_missing":
            continue
        for pos, a in enumerate(idxs):
            node_a = node_rows[a]
            qids_a = node_qids.get(node_a["node_key"], set())
            for b in idxs[pos + 1 :]:
                node_b = node_rows[b]
                frame_gap = abs(int(node_a["frame_id"]) - int(node_b["frame_id"]))
                if frame_gap == 0 or frame_gap > int(args.max_frame_gap):
                    continue
                area_a = max(1e-6, float(node_a["area_ratio"]))
                area_b = max(1e-6, float(node_b["area_ratio"]))
                area_ratio = max(area_a, area_b) / max(1e-6, min(area_a, area_b))
                if area_ratio > float(args.max_area_ratio):
                    continue
                diag = math.sqrt(float(node_a["image_width"]) ** 2 + float(node_a["image_height"]) ** 2)
                centroid_dist = math.sqrt((float(node_a["cx"]) - float(node_b["cx"])) ** 2 + (float(node_a["cy"]) - float(node_b["cy"])) ** 2) / max(1.0, diag)
                if centroid_dist > float(args.max_centroid_dist):
                    continue
                qids_b = node_qids.get(node_b["node_key"], set())
                shared = len(qids_a & qids_b)
                union = len(qids_a | qids_b)
                qid_jaccard = float(shared / max(1, union))
                if shared < int(args.min_shared_qids) and qid_jaccard < float(args.min_qid_jaccard):
                    continue
                priority = float(shared) + 10.0 * qid_jaccard - centroid_dist - 0.01 * frame_gap
                edge = {
                    "schema_version": "stream4d_v96_semantic_spatial_merge_edge_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "node_index_a": a,
                    "node_index_b": b,
                    "scene_id": node_a["scene_id"],
                    "window_id": node_a["window_id"],
                    "semantic_prototype_id": proto,
                    "frame_gap": frame_gap,
                    "centroid_dist_norm": centroid_dist,
                    "area_ratio_pair": area_ratio,
                    "shared_qids": shared,
                    "qid_jaccard": qid_jaccard,
                    "priority": priority,
                    "accepted_for_union": False,
                    "reject_reason": "",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
                candidates.append((priority, a, b, edge))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted_edges = 0
    skipped_same_frame_conflict = 0
    for _priority, a, b, edge in candidates:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            edge["accepted_for_union"] = False
            edge["reject_reason"] = "already_same_component"
            candidate_rows.append(edge)
            continue
        maps_a = frame_masks[ra]
        maps_b = frame_masks[rb]
        conflict = False
        for frame_key, mask_id in (maps_a.items() if len(maps_a) <= len(maps_b) else maps_b.items()):
            other = (maps_b if len(maps_a) <= len(maps_b) else maps_a).get(frame_key)
            if other is not None and int(other) != int(mask_id):
                conflict = True
                break
        if conflict:
            skipped_same_frame_conflict += 1
            edge["accepted_for_union"] = False
            edge["reject_reason"] = "same_frame_cannot_link"
            candidate_rows.append(edge)
            continue
        new_root = dsu.union(ra, rb)
        old_root = rb if new_root == ra else ra
        merged = dict(frame_masks[new_root])
        merged.update(frame_masks[old_root])
        frame_masks[new_root] = merged
        accepted_edges += 1
        edge["accepted_for_union"] = True
        candidate_rows.append(edge)

    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in node_rows:
        components[dsu.find(int(node["node_index"]))].append(node)

    selected_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    micro_cluster_rows: list[dict[str, Any]] = []
    object_idx = 0
    for _root_idx, nodes in sorted(components.items(), key=lambda item: min(int(n["frame_id"]) for n in item[1])):
        object_idx += 1
        object_id = f"{args.output_family}_obj_{object_idx:06d}"
        proto_counts = Counter(str(node["semantic_prototype_id"]) for node in nodes)
        main_proto = proto_counts.most_common(1)[0][0] if proto_counts else "semantic_missing"
        group_id = f"{args.output_family}:{object_idx:06d}:proto:{main_proto}"
        qids: set[str] = set()
        support_sum = 0
        score_vals: list[float] = []
        for node in sorted(nodes, key=lambda n: (int(n["frame_id"]), int(n["selected_mask_id"]))):
            source_row = dict(node["source_row"])
            qids.update(node_qids.get(node["node_key"], set()))
            support_sum += int(_num(source_row.get("masklet_support_query_count")))
            score_vals.append(float(_num(source_row.get("masklet_score"))))
            selected_rows.append(
                {
                    **source_row,
                    "schema_version": "stream4d_v96_selected_masklet_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": args.output_family,
                    "object_id": object_id,
                    "group_id": group_id,
                    "selection_status": "semantic_spatial_d4rt_merged",
                    "semantic_prototype_id": node["semantic_prototype_id"],
                    "mask_centroid_x": node["cx"],
                    "mask_centroid_y": node["cy"],
                    "mask_area_ratio": node["area_ratio"],
                    "node_qid_count": node["qid_count"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "object_id": object_id,
                "group_id": group_id,
                "micro_query_count": len(qids),
                "selected_frame_count_before_collision_resolution": len(nodes),
                "masklet_support_query_count_sum": support_sum,
                "object_score": float(np.mean(score_vals)) if score_vals else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "selected_frame_count": len({(node["scene_id"], node["window_id"], int(node["frame_id"])) for node in nodes}),
            }
        )
        micro_cluster_rows.append(
            {
                "schema_version": "stream4d_v96_micro_cluster_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "group_id": group_id,
                "micro_query_count": len(qids),
                "cluster_source": "semantic_spatial_d4rt_merge_repair",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    frame_mask_counter = Counter((row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"])) for row in selected_rows)
    duplicate_frame_mask_count = sum(max(0, count - 1) for count in frame_mask_counter.values())
    frame_counts = Counter(str(row["object_id"]) for row in selected_rows)
    summary = {
        "schema": "stream4d_v96_semantic_spatial_d4rt_merge_repair_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "MATERIALIZED_V96_PHASE5_SEMANTIC_SPATIAL_D4RT_MERGE_REPAIR",
        "source_phase5_root": _rel(phase5_root),
        "output_root": _rel(output_root),
        "source_family": args.source_family,
        "output_family": args.output_family,
        "selected_masklet_count": len(selected_rows),
        "source_selected_masklet_count": len(selected_source),
        "object_count": len(object_rows),
        "source_object_count": len({row.get("object_id", "") for row in selected_source}),
        "single_frame_object_count": sum(1 for count in frame_counts.values() if count == 1),
        "max_selected_frames_per_object": max(frame_counts.values(), default=0),
        "candidate_edge_count": len(candidate_rows),
        "initial_source_union_count": initial_source_union_count,
        "initial_source_conflict_count": initial_source_conflict_count,
        "accepted_edge_count": accepted_edges,
        "skipped_same_frame_conflict_edge_count": skipped_same_frame_conflict,
        "duplicate_frame_mask_count": duplicate_frame_mask_count,
        "semantic_feature_rows": _rel(_project(args.semantic_feature_rows)),
        "incidence_root": _rel(_project(args.incidence_root)),
        "decode_variant": args.decode_variant,
        "min_shared_qids": int(args.min_shared_qids),
        "min_qid_jaccard": float(args.min_qid_jaccard),
        "max_centroid_dist": float(args.max_centroid_dist),
        "max_area_ratio": float(args.max_area_ratio),
        "max_frame_gap": int(args.max_frame_gap),
        "runtime_total_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_rows)
    _write_csv(output_root / "micro_cluster_rows.csv", micro_cluster_rows)
    _write_csv(output_root / "semantic_spatial_merge_edge_rows.csv", candidate_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "object_count": len(object_rows), "accepted_edge_count": accepted_edges, "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a v96 semantic/spatial/D4RT merge repair from C masklets.")
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--source-family", default="C_hybrid_cover_cluster")
    parser.add_argument("--output-family", default="C_semantic_spatial_d4rt_merge")
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--semantic-feature-rows", default=str(DEFAULT_SEMANTIC_ROWS))
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--min-shared-qids", type=int, default=2)
    parser.add_argument("--min-qid-jaccard", type=float, default=0.02)
    parser.add_argument("--max-centroid-dist", type=float, default=0.18)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--max-frame-gap", type=int, default=35)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
