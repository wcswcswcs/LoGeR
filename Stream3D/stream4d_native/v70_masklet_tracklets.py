from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v68_local_graph_solver import DSU, _row_from_mapping, _same_frame_violation_count, _summarize_variant_all  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _float_or_none, _frame_data, _load_csv_rows, _mean, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


TRACKLET_VARIANTS = [
    "TR0_DINO_mutual_top1",
    "TR1_signature_mutual_top1",
    "TR2_carrier_witness_signature",
    "TR3_TR2_outside_veto",
    "TR4_TR3_underseg_shared",
    "TR5_greedy_adjacent_flow",
    "TR6_no_temporal_control",
]
CONTROL_VARIANTS = {"TR6_no_temporal_control"}


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _stable_unit_interval(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _dino_match(row: dict[str, Any]) -> float:
    left = str(row.get("anchor_DINO_mode_id") or "")
    right = str(row.get("candidate_DINO_mode_id") or "")
    return 1.0 if left and right and left == right else 0.0


def _signature_match(row: dict[str, Any]) -> float:
    left = str(row.get("anchor_repeated_signature_id") or "")
    right = str(row.get("candidate_repeated_signature_id") or "")
    return 1.0 if left and right and left == right else 0.0


def _temporal_score(frame_delta: int) -> float:
    delta_steps = max(1.0, abs(float(frame_delta)) / 5.0)
    return float(1.0 / (1.0 + 0.20 * max(0.0, delta_steps - 1.0)))


def _load_anchor_pairs(path: Path, variant: str, scenes: set[str]) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in _load_csv_rows(path):
        if str(row.get("anchor_variant")) != variant:
            continue
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        out[str(row.get("chunk_id"))].add((_safe_int(row.get("frame_id")), _safe_int(row.get("mask_id"))))
    return out


def _load_witness_edges(path: Path, scenes: set[str], max_frame_delta: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        frame_delta = _safe_int(row.get("frame_delta"))
        if frame_delta <= 0 or frame_delta > int(max_frame_delta):
            continue
        left = (_safe_int(row.get("anchor_frame")), _safe_int(row.get("anchor_mask")))
        right = (_safe_int(row.get("candidate_frame")), _safe_int(row.get("candidate_mask")))
        if left[0] == right[0] or left[1] <= 0 or right[1] <= 0:
            continue
        if right[0] < left[0]:
            left, right = right, left
        inside = _safe_float(row.get("inside_ratio"))
        outside = _safe_float(row.get("outside_ratio"))
        residual = inside - outside
        edge = {
            "scene_id": scene,
            "chunk_id": str(row.get("chunk_id")),
            "left": left,
            "right": right,
            "inside_ratio": inside,
            "outside_ratio": outside,
            "residual": residual,
            "frame_delta": frame_delta,
            "visible_carrier_count": _safe_int(row.get("visible_carrier_count")),
            "inside_candidate_count": _safe_int(row.get("inside_candidate_count")),
            "underseg": _parse_bool(row.get("candidate_underseg_risk")),
            "dino": _dino_match(row),
            "signature": _signature_match(row),
            "temporal": _temporal_score(frame_delta),
            "shuffle": _stable_unit_interval("v70_tracklet_shuffle", row.get("scene_id"), row.get("chunk_id"), left, right),
        }
        out[edge["chunk_id"]].append(edge)
    return out


def _tracklet_score(variant: str, edge: dict[str, Any]) -> tuple[float, str]:
    inside = float(edge["inside_ratio"])
    outside = float(edge["outside_ratio"])
    residual = max(0.0, float(edge["residual"]))
    dino = float(edge["dino"])
    signature = float(edge["signature"])
    temporal = float(edge["temporal"])
    underseg = bool(edge["underseg"])
    if variant == "TR0_DINO_mutual_top1":
        score = dino
        return score, "core" if score >= 1.0 and not underseg else "reject"
    if variant == "TR1_signature_mutual_top1":
        score = 0.80 * signature + 0.20 * inside
        return score, "core" if signature > 0.0 and inside >= 0.45 and not underseg else "reject"
    if variant == "TR2_carrier_witness_signature":
        score = 0.46 * residual + 0.24 * inside + 0.20 * signature + 0.10 * dino
        if underseg:
            return score, "shared" if inside >= 0.86 and residual >= 0.20 else "reject"
        return score, "core" if score >= 0.42 and inside >= 0.50 and (signature > 0.0 or inside >= 0.75) else "reject"
    if variant == "TR3_TR2_outside_veto":
        score = 0.50 * residual + 0.22 * inside + 0.20 * signature + 0.08 * dino
        if outside > 0.46:
            return score, "reject"
        if underseg:
            return score, "shared" if inside >= 0.86 and residual >= 0.25 else "reject"
        return score, "core" if score >= 0.42 and inside >= 0.54 and (signature > 0.0 or inside >= 0.78) else "reject"
    if variant == "TR4_TR3_underseg_shared":
        score = 0.48 * residual + 0.22 * inside + 0.18 * signature + 0.12 * temporal
        if underseg:
            return score, "shared" if score >= 0.60 and inside >= 0.88 else "reject"
        return score, "core" if score >= 0.40 and inside >= 0.52 and (signature > 0.0 or inside >= 0.76) else "reject"
    if variant == "TR5_greedy_adjacent_flow":
        score = 0.38 * residual + 0.24 * inside + 0.14 * signature + 0.24 * temporal
        if underseg:
            return score, "shared" if score >= 0.62 and inside >= 0.88 else "reject"
        return score, "core" if score >= 0.42 and inside >= 0.50 and (signature > 0.0 or inside >= 0.74) else "reject"
    if variant == "TR6_no_temporal_control":
        score = 0.48 * residual + 0.28 * inside + 0.18 * signature + 0.06 * dino
        if underseg:
            return score, "shared" if score >= 0.60 and inside >= 0.88 else "reject"
        return score, "core" if score >= 0.42 and inside >= 0.52 and (signature > 0.0 or inside >= 0.76) else "reject"
    raise ValueError(f"unknown v70 tracklet variant: {variant}")


def _build_tracklet_mapping(
    *,
    variant: str,
    anchors: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    max_edges_per_node: int,
    tracklet_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    nodes = set(anchors)
    scored: list[tuple[float, str, tuple[int, int], tuple[int, int], dict[str, Any]]] = []
    per_left: dict[tuple[int, int], list[tuple[float, str, tuple[int, int], tuple[int, int], dict[str, Any]]]] = defaultdict(list)
    for edge in edges:
        score, role = _tracklet_score(variant, edge)
        left, right = edge["left"], edge["right"]
        if role != "reject":
            nodes.add(left)
            if role == "core":
                nodes.add(right)
        per_left[left].append((score, role, left, right, edge))
    for left, items in per_left.items():
        scored.extend(sorted(items, key=lambda item: item[0], reverse=True)[: int(max_edges_per_node)])
    active = sorted(nodes)
    dsu = DSU(active)
    members: dict[tuple[int, int], set[tuple[int, int]]] = {node: {node} for node in active}
    pred: dict[tuple[int, int], tuple[int, int]] = {}
    succ: dict[tuple[int, int], tuple[int, int]] = {}
    selected_edges = 0
    shared_edges = 0
    rejected_edges = 0
    underseg_core_edges = 0
    for score, role, left, right, edge in sorted(scored, key=lambda item: item[0], reverse=True):
        if role == "reject" or score <= 0.0:
            rejected_edges += 1
            continue
        if role == "shared":
            shared_edges += 1
            continue
        if left not in members or right not in members or left in succ or right in pred:
            rejected_edges += 1
            continue
        root_left = dsu.find(left)
        root_right = dsu.find(right)
        if root_left == root_right:
            rejected_edges += 1
            continue
        frames = {frame for frame, _mask in members[root_left]}
        if any(frame in frames for frame, _mask in members[root_right]):
            rejected_edges += 1
            continue
        dsu.union(root_left, root_right)
        new_root = dsu.find(root_left)
        merged = members.pop(root_left, {root_left}) | members.pop(root_right, {root_right})
        members[new_root] = merged
        pred[right] = left
        succ[left] = right
        selected_edges += 1
        if bool(edge["underseg"]):
            underseg_core_edges += 1
        if len(tracklet_rows) < 1_000_000:
            tracklet_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "tracklet_variant": variant,
                    "left_frame": left[0],
                    "left_mask": left[1],
                    "right_frame": right[0],
                    "right_mask": right[1],
                    "edge_score": score,
                    "inside_ratio": edge["inside_ratio"],
                    "outside_ratio": edge["outside_ratio"],
                    "frame_delta": edge["frame_delta"],
                    "signature_match": edge["signature"],
                    "dino_match": edge["dino"],
                    "candidate_underseg_risk": edge["underseg"],
                    "uses_gt_for_prediction": False,
                    "diagnostic_only": False,
                    "forbidden_for_method_table": False,
                }
            )
    root_to_id: dict[tuple[int, int], int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for node in active:
        root = dsu.find(node)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[node] = root_to_id[root]
    lengths = [len(items) for items in members.values()]
    diag = {
        "tracklet_count": int(len(lengths)),
        "tracklet_length_mean": float(np.mean(lengths)) if lengths else None,
        "tracklet_length_p50": float(np.median(lengths)) if lengths else None,
        "tracklet_single_frame_rate": float(sum(1 for value in lengths if value <= 1) / max(1, len(lengths))),
        "selected_edge_count": int(selected_edges),
        "shared_edge_count": int(shared_edges),
        "reject_edge_count": int(rejected_edges),
        "underseg_bridge_rate": float(underseg_core_edges / max(1, selected_edges)),
        "same_frame_cannot_link_violation_count": _same_frame_violation_count(mapping),
        "shared_mask_count": int(shared_edges),
        "reject_mask_count": int(rejected_edges),
        "unknown_mask_count": 0,
    }
    return mapping, diag


def _mapping_purity(frame_data: list[dict[str, Any]], mapping: dict[tuple[int, int], int]) -> float | None:
    gt_counts_by_object: dict[int, Counter[int]] = defaultdict(Counter)
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        gt = item["gt"]
        if mask is None:
            continue
        for mask_id in np.unique(mask):
            mask_id_i = int(mask_id)
            object_id = int(mapping.get((frame_id, mask_id_i), 0))
            if mask_id_i <= 0 or object_id <= 0:
                continue
            pixels = gt[mask == mask_id_i]
            pixels = pixels[pixels > 0]
            if pixels.size == 0:
                continue
            labels, counts = np.unique(pixels, return_counts=True)
            for label, count in zip(labels.tolist(), counts.tolist()):
                gt_counts_by_object[object_id][int(label)] += int(count)
    purities = []
    for counts in gt_counts_by_object.values():
        total = sum(counts.values())
        if total > 0:
            purities.append(max(counts.values()) / total)
    return float(np.mean(purities)) if purities else None


def _object_temporal_span(mapping: dict[tuple[int, int], int]) -> float | None:
    frames_by_object: dict[int, set[int]] = defaultdict(set)
    for (frame_id, _mask_id), object_id in mapping.items():
        frames_by_object[int(object_id)].add(int(frame_id))
    spans = [len(frames) for frames in frames_by_object.values()]
    return float(np.mean(spans)) if spans else None


def _summarize_tracklet_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "tracklet_variant": variant,
            "tracklet_count": int(sum(int(float(row.get("tracklet_count") or 0)) for row in subset)),
            "tracklet_length_mean": _mean([_float_or_none(row.get("tracklet_length_mean")) for row in subset]),
            "tracklet_length_p50": _mean([_float_or_none(row.get("tracklet_length_p50")) for row in subset]),
            "tracklet_single_frame_rate": _mean([_float_or_none(row.get("tracklet_single_frame_rate")) for row in subset]),
            "tracklet_purity_diagnostic": _mean([_float_or_none(row.get("tracklet_purity_diagnostic")) for row in subset]),
            "tracklet_SF50": base.get("local_score_free_match50_recall_mean"),
            "tracklet_AP50": base.get("local_AP50_mean"),
            "tracklet_GT_best_IoU_mean": base.get("local_GT_best_IoU_mean_mean"),
            "underseg_bridge_rate": _mean([_float_or_none(row.get("underseg_bridge_rate")) for row in subset]),
            "real_minus_shuffled_SF50": None,
            "real_minus_no_temporal_SF50": None,
        }
    )
    return base


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    scene_set = set(scenes)
    variants = _parse_csv_list(args.variants) or TRACKLET_VARIANTS
    anchors_by_chunk = _load_anchor_pairs(_rooted(args.anchor_rows), str(args.anchor_variant), scene_set)
    edges_by_chunk = _load_witness_edges(_rooted(args.witness_rows), scene_set, int(args.max_frame_delta))
    chunk_rows: list[dict[str, Any]] = []
    tracklet_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v70-masklet-tracklets] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            t0 = time.time()
            chunk_id = str(chunk.get("chunk_id"))
            anchors = anchors_by_chunk.get(chunk_id, set())
            edges = edges_by_chunk.get(chunk_id, [])
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids or not anchors:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for variant in variants:
                mapping, diag = _build_tracklet_mapping(
                    variant=variant,
                    anchors=anchors,
                    edges=edges,
                    max_edges_per_node=int(args.max_edges_per_node),
                    tracklet_rows=tracklet_rows,
                    scene=scene,
                    chunk_id=chunk_id,
                )
                diag["tracklet_purity_diagnostic"] = _mapping_purity(frame_data, mapping)
                row = _row_from_mapping(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    diag=diag,
                    pipeline_root=pipeline_root,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                )
                row.update(diag)
                row["tracklet_temporal_span_mean"] = _object_temporal_span(mapping)
                row["runtime_sec"] = float(time.time() - t0)
                row["max_frame_delta"] = int(args.max_frame_delta)
                row["anchor_variant"] = str(args.anchor_variant)
                chunk_rows.append(row)
    metric_rows = [_summarize_tracklet_variant(chunk_rows, variant) for variant in variants]
    by_variant = {str(row.get("tracklet_variant")): row for row in metric_rows}
    for row in metric_rows:
        if row.get("tracklet_variant") in CONTROL_VARIANTS:
            continue
        control = by_variant.get("TR6_no_temporal_control", {})
        if _float_or_none(row.get("tracklet_SF50")) is not None and _float_or_none(control.get("tracklet_SF50")) is not None:
            row["real_minus_no_temporal_SF50"] = float(row["tracklet_SF50"]) - float(control["tracklet_SF50"])
    method_rows = [row for row in metric_rows if row.get("tracklet_variant") not in CONTROL_VARIANTS]
    best = max(method_rows, key=lambda row: float(row.get("tracklet_SF50") or 0.0), default={})
    baseline_summary = json.loads(_rooted(args.true_closure_summary).read_text(encoding="utf-8"))
    true_best = baseline_summary.get("best_closure_variant") or {}
    v69_c10 = baseline_summary.get("v69r2_C10_reproduction") or {}
    best_sf50 = _float_or_none(best.get("tracklet_SF50"))
    true_sf50 = _float_or_none(true_best.get("single_anchor_SF50"))
    c10_sf50 = _float_or_none(v69_c10.get("single_anchor_SF50"))
    best_single = _float_or_none(best.get("tracklet_single_frame_rate"))
    true_single = _float_or_none(true_best.get("single_anchor_single_frame_rate"))
    best_purity = _float_or_none(best.get("tracklet_purity_diagnostic"))
    best_violation = int(float(best.get("same_frame_cannot_link_violation_count_sum") or 0)) if best else 0
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "tracklet_SF50_ge_max_single_anchor_v69r2_plus_0p08": best_sf50 is not None
        and best_sf50 >= max(float(true_sf50 or 0.0), float(c10_sf50 or 0.0)) + 0.08,
        "tracklet_length_mean_ge_2": _float_or_none(best.get("tracklet_length_mean")) is not None and float(best.get("tracklet_length_mean")) >= 2.0,
        "tracklet_single_frame_rate_le_0p50": best_single is not None and best_single <= 0.50,
        "tracklet_purity_diagnostic_ge_0p80": best_purity is not None and best_purity >= 0.80,
        "same_frame_violation_count_eq_0": best_violation == 0,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    summary = {
        "phase": "v70_masklet_tracklets",
        "decision": "PASS_MASKLET_TRACKLETS" if gate["pass"] else "NO_GO_MASKLET_TRACKLETS",
        "gate": gate,
        "best_tracklet_variant": best,
        "baseline_true_closure_best": true_best,
        "baseline_v69r2_C10": v69_c10,
        "anchor_rows": _rel(_rooted(args.anchor_rows)),
        "witness_rows": _rel(_rooted(args.witness_rows)),
        "true_closure_summary": _rel(_rooted(args.true_closure_summary)),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "variants": variants,
        "max_frame_delta": int(args.max_frame_delta),
        "max_edges_per_node": int(args.max_edges_per_node),
        "rows": {
            "tracklet_summary_json": _rel(output_root / "tracklet_summary.json"),
            "tracklet_rows_csv": _rel(output_root / "tracklet_rows.csv"),
            "tracklet_metric_rows_csv": _rel(output_root / "tracklet_metric_rows.csv"),
            "tracklet_chunk_rows_csv": _rel(output_root / "tracklet_chunk_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Tracklet construction uses one-predecessor/one-successor greedy matching over carrier-witness candidate edges.",
            "GT labels are used only for diagnostic evaluation/purity, not for prediction.",
        ],
    }
    _write_csv(output_root / "tracklet_rows.csv", tracklet_rows)
    _write_csv(output_root / "tracklet_metric_rows.csv", metric_rows)
    _write_csv(output_root / "tracklet_chunk_rows.csv", chunk_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "tracklet_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "tracklet_summary.json",
        output_root / "tracklet_rows.csv",
        output_root / "tracklet_metric_rows.csv",
        output_root / "tracklet_chunk_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 masklet tracklet builder/evaluator.")
    parser.add_argument("--output-root", default="outputs/audit/v70_masklet_tracklets")
    parser.add_argument("--anchor-rows", default="outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_rows.csv")
    parser.add_argument("--anchor-variant", default="A9_clean_recall_support_floor_u15")
    parser.add_argument("--witness-rows", default="outputs/audit/v70_carrier_witness/witness_rows.csv")
    parser.add_argument("--true-closure-summary", default="outputs/audit/v70_true_material_closure/closure_summary.json")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--variants", default=",".join(TRACKLET_VARIANTS))
    parser.add_argument("--max-frame-delta", type=int, default=30)
    parser.add_argument("--max-edges-per-node", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
