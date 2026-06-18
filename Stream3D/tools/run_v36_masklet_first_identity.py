from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv, assign_gt_labels
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


LOCAL_GATE = {
    "ARI": 0.40,
    "purity": 0.85,
    "completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_ARI": 0.20,
}


SOURCE_GROUPS = {
    "R0_current_cropformer": {"R0_full_mask_region"},
    "R1_boundary_watershed": {"R1_boundary_eroded_interior", "R2_distance_watershed_region"},
    "R3_d4rt_tube_seeded_split": {"R3_d4rt_tube_seeded_voronoi", "R5_d4rt_canonical_adjacency_split"},
    "R4_hybrid_split": {
        "R1_boundary_eroded_interior",
        "R2_distance_watershed_region",
        "R3_d4rt_tube_seeded_voronoi",
        "R4_image_gradient_split",
        "R5_d4rt_canonical_adjacency_split",
        "R6_mask_overlap_consensus_region",
        "R7_high_purity_core_region",
    },
    "R6_hybrid_union": {
        "R0_full_mask_region",
        "R1_boundary_eroded_interior",
        "R2_distance_watershed_region",
        "R3_d4rt_tube_seeded_voronoi",
        "R4_image_gradient_split",
        "R5_d4rt_canonical_adjacency_split",
        "R6_mask_overlap_consensus_region",
        "R6_mask_overlap_consensus_union",
        "R7_high_purity_core_region",
    },
}

MASK_ONLY_TYPES = {"R0_full_mask_region", "R1_boundary_eroded_interior"}
MAX_REGIONS_PER_SCENE = 4500


@dataclass
class MaskletResult:
    variant: str
    control_kind: str
    rows: list[dict[str, Any]]
    masklets: list[dict[str, Any]]
    scene_metrics: list[dict[str, Any]]
    all_metrics: dict[str, Any]


class UnionFind:
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[Any]) -> float | None:
    vals = [_float(v) for v in values]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _parse_core_tube_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if isinstance(row.get("_core_tube_ids"), list):
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or []))
    text = str(row.get("core_tube_ids") or "")
    return tuple(sorted(int(part) for part in text.split(";") if part.strip()))


def _overlap_counts(row: dict[str, Any]) -> dict[int, int]:
    raw = row.get("_gt_overlap_counts") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return {int(k): int(v) for k, v in dict(raw).items() if int(v) > 0}


def _proposal_type(row: dict[str, Any]) -> str:
    return str(row.get("proposal_type") or "")


def _source_group(row: dict[str, Any]) -> str:
    ptype = _proposal_type(row)
    for source, types in SOURCE_GROUPS.items():
        if ptype in types:
            return source
    if ptype.startswith(("R8_", "R9_", "R10_", "R11_", "R12_")):
        return "R6_hybrid_union"
    return f"unknown::{ptype}"


def _rows_for_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    if source == "R6_hybrid_union":
        return [row for row in rows if not _source_group(row).startswith("unknown::")]
    types = SOURCE_GROUPS.get(source, set())
    return [row for row in rows if _proposal_type(row) in types]


def _load_gt_labels(args: argparse.Namespace, scenes: list[str]) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    cache_root = Path(args.cache_root)
    for scene in scenes:
        chunks, _ = load_scene_chunks_from_cache(
            cache_root / scene,
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
        )
        builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
        records = chunks_to_records(builder.stitch_to_canonical(chunks))
        out[scene] = assign_gt_labels(
            records,
            stream=ScanNetStream(seq_name=scene),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
    return out


def _gt_counts(gt_labels: dict[int, int]) -> Counter[int]:
    return Counter(int(v) for v in gt_labels.values() if int(v) > 0)


def _source_metrics_for_rows(
    *,
    source: str,
    rows: list[dict[str, Any]],
    gt_counts_by_scene: dict[str, Counter[int]],
) -> dict[str, Any]:
    total_regions = len(rows)
    mixed_count = 0
    pair_total = 0
    same_pairs = 0
    diff_pairs = 0
    scene0081_regions = 0
    scene0081_mixed = 0
    best_iou: dict[tuple[str, int], float] = {}
    best_iou_scene0081: dict[int, float] = {}
    for row in rows:
        scene = str(row.get("scene") or "")
        counts = _overlap_counts(row)
        positive = {gt: count for gt, count in counts.items() if int(gt) > 0 and int(count) > 0}
        labeled = sum(positive.values())
        if len(positive) > 1:
            mixed_count += 1
            if scene == "scene0081_01":
                scene0081_mixed += 1
        if scene == "scene0081_01":
            scene0081_regions += 1
        if labeled >= 2:
            total = labeled * (labeled - 1) // 2
            same = sum(count * (count - 1) // 2 for count in positive.values())
            pair_total += total
            same_pairs += same
            diff_pairs += total - same
        for gt, overlap in positive.items():
            gt_total = int(gt_counts_by_scene.get(scene, Counter()).get(int(gt), 0))
            denom = labeled + gt_total - int(overlap)
            iou = float(overlap / max(denom, 1))
            key = (scene, int(gt))
            best_iou[key] = max(best_iou.get(key, 0.0), iou)
            if scene == "scene0081_01":
                best_iou_scene0081[int(gt)] = max(best_iou_scene0081.get(int(gt), 0.0), iou)
    gt_keys = [(scene, gt) for scene, counts in gt_counts_by_scene.items() for gt in counts]
    scene0081_gt = list(gt_counts_by_scene.get("scene0081_01", Counter()).keys())
    return {
        "source": source,
        "region_count": int(total_regions),
        "mixed_region_rate": float(mixed_count / max(total_regions, 1)),
        "same_region_same_GT_ratio": float(same_pairs / max(pair_total, 1)),
        "same_region_diff_GT_ratio": float(diff_pairs / max(pair_total, 1)),
        "GT_object_coverage@0.05": float(sum(1 for key in gt_keys if best_iou.get(key, 0.0) >= 0.05) / max(len(gt_keys), 1)),
        "GT_object_coverage@0.10": float(sum(1 for key in gt_keys if best_iou.get(key, 0.0) >= 0.10) / max(len(gt_keys), 1)),
        "GT_object_coverage@0.25": float(sum(1 for key in gt_keys if best_iou.get(key, 0.0) >= 0.25) / max(len(gt_keys), 1)),
        "scene0081_mixed_region_rate": float(scene0081_mixed / max(scene0081_regions, 1)),
        "scene0081_GT_object_coverage@0.10": float(
            sum(1 for gt in scene0081_gt if best_iou_scene0081.get(int(gt), 0.0) >= 0.10)
            / max(len(scene0081_gt), 1)
        ),
    }


def _region_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        counts = _overlap_counts(row)
        source = _source_group(row)
        out.append(
            {
                "region_id": row.get("proposal_id"),
                "scene": row.get("scene"),
                "frame_id": int(row.get("frame_id", -1)),
                "mask_id": int(row.get("mask_id", -1)),
                "source": source,
                "proposal_type": row.get("proposal_type"),
                "area": _float(row.get("region_area")),
                "inside_tube_count": int(row.get("num_core_tubes") or len(_parse_core_tube_ids(row))),
                "boundary_tube_count": int(row.get("num_boundary_tubes") or 0),
                "visual_compactness": _float(row.get("appearance_variance")),
                "D4RT_support_count": int(row.get("num_core_tubes") or len(_parse_core_tube_ids(row))),
                "dominant_GT_ratio_diagnostic": _float(row.get("proposal_purity")),
                "num_GT_instances_inside_diagnostic": int(len([gt for gt, count in counts.items() if gt > 0 and count > 0])),
                "mixed_flag_diagnostic": bool(len([gt for gt, count in counts.items() if gt > 0 and count > 0]) > 1),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return out


def _phase_b_region_sources(rows: list[dict[str, Any]], gt_labels: dict[str, dict[int, int]], out_dir: Path) -> dict[str, Any]:
    gt_counts_by_scene = {scene: _gt_counts(labels) for scene, labels in gt_labels.items()}
    metrics = []
    for source in ["R0_current_cropformer", "R1_boundary_watershed", "R3_d4rt_tube_seeded_split", "R4_hybrid_split", "R6_hybrid_union"]:
        metrics.append(_source_metrics_for_rows(source=source, rows=_rows_for_source(rows, source), gt_counts_by_scene=gt_counts_by_scene))
    r0 = next((row for row in metrics if row["source"] == "R0_current_cropformer"), None)
    for row in metrics:
        row["phaseB_success"] = False
        row["phaseB_success_reasons"] = ""
        if r0 is not None and row["source"] != "R0_current_cropformer":
            checks = {
                "mixed_region_rate": row["mixed_region_rate"] <= 0.70 * float(r0["mixed_region_rate"]),
                "same_region_diff_GT_ratio": row["same_region_diff_GT_ratio"] <= 0.70 * float(r0["same_region_diff_GT_ratio"]),
                "GT_object_coverage@0.10": row["GT_object_coverage@0.10"] >= 0.70,
                "scene0081_mixed_region_rate": row["scene0081_mixed_region_rate"] <= 0.80 * float(r0["scene0081_mixed_region_rate"]),
            }
            row["phaseB_success"] = bool(all(checks.values()))
            row["phaseB_success_reasons"] = json.dumps(checks, sort_keys=True)
    audit_rows = _region_audit_rows(rows)
    _write_csv(out_dir / "region_source_metrics.csv", metrics)
    _write_json(out_dir / "region_source_metrics.json", metrics)
    _write_csv(out_dir / "region_rows.csv", audit_rows)
    _write_json(out_dir / "region_rows.json", audit_rows)
    manifest = {
        "phase": "v36_phaseB",
        "region_source_generated_from_rgb_masks": True,
        "uses_old_proposal_rows_only": False,
        "generator": "tools/run_v28_proposal_oracle.py regenerated under outputs/audit/v36_region_sources",
        "source_count": len(metrics),
        "region_count": len(rows),
        "phaseB_any_success": any(bool(row["phaseB_success"]) for row in metrics),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {"metrics": metrics, "manifest": manifest}


def _filter_rows_for_variant(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "D0_greedy_temporal_R1":
        return _cap_rows_per_scene(_rows_for_source(rows, "R1_boundary_watershed"))
    if variant == "D1_correlation_R4":
        return _cap_rows_per_scene(_rows_for_source(rows, "R4_hybrid_split"))
    if variant == "D2_adjacent_matching_R6":
        return _cap_rows_per_scene(_rows_for_source(rows, "R6_hybrid_union"))
    if variant == "D3_hybrid_unknown_R6":
        return _cap_rows_per_scene(_rows_for_source(rows, "R6_hybrid_union"))
    if variant == "D4_shuffled_d4rt_control":
        return _cap_rows_per_scene(_rows_for_source(rows, "R6_hybrid_union"))
    if variant == "D5_mask_only_control":
        return _cap_rows_per_scene([row for row in rows if _proposal_type(row) in MASK_ONLY_TYPES])
    if variant == "D6_no_temporal_control":
        return _cap_rows_per_scene(_rows_for_source(rows, "R6_hybrid_union"))
    return _cap_rows_per_scene(rows)


def _method_score(row: dict[str, Any]) -> float:
    core = max(float(len(_parse_core_tube_ids(row))), 1.0)
    score = 0.0
    score += 0.25 * (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0)
    score += 0.15 * (_float(row.get("visibility_mean"), 0.5) or 0.5)
    score += 0.12 * (_float(row.get("confidence_mean"), 0.5) or 0.5)
    score += 0.10 * min(math.log1p(core) / math.log(128.0), 1.0)
    score += 0.08 * (_float(row.get("mask_temporal_repeat_score"), 1.0) or 1.0)
    score -= 0.18 * (_float(row.get("boundary_contact_ratio"), 0.0) or 0.0)
    score -= 0.15 * math.log1p(max(_float(row.get("visible_outside_negative_rate"), 0.0) or 0.0, 0.0))
    score -= 0.15 * math.log1p(max(_float(row.get("same_frame_cannot_link_rate"), 0.0) or 0.0, 0.0))
    return float(score)


def _cap_rows_per_scene(rows: list[dict[str, Any]], max_rows: int = MAX_REGIONS_PER_SCENE) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene"))].append(row)
    out: list[dict[str, Any]] = []
    for scene_rows in by_scene.values():
        if len(scene_rows) <= int(max_rows):
            out.extend(scene_rows)
        else:
            out.extend(sorted(scene_rows, key=_method_score, reverse=True)[: int(max_rows)])
    return out


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _make_masklets(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows = list(rows)
    if variant in {"D0_greedy_temporal_R1", "D1_correlation_R4", "D3_hybrid_unknown_R6", "D4_shuffled_d4rt_control", "D5_mask_only_control"}:
        groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            source = _source_group(row) if variant in {"D1_correlation_R4", "D3_hybrid_unknown_R6", "D4_shuffled_d4rt_control"} else "mask"
            groups[(str(row.get("scene")), int(row.get("mask_id", -1)), source)].append(idx)
        return _groups_to_masklets(rows, list(groups.values()), variant)
    if variant == "D2_adjacent_matching_R6":
        groups = defaultdict(list)
        for idx, row in enumerate(rows):
            frame_bucket = int(row.get("frame_id", 0)) // 20
            groups[(str(row.get("scene")), int(row.get("mask_id", -1)), frame_bucket)].append(idx)
        return _groups_to_masklets(rows, list(groups.values()), variant)
    if variant == "D6_no_temporal_control":
        return _groups_to_masklets(rows, [[idx] for idx in range(len(rows))], variant)

    # Fallback sparse shared-tube graph for future variants; current v36 baseline
    # variants use deterministic masklet components above to keep runtime auditable.
    uf = UnionFind(len(rows))
    by_scene = defaultdict(list)
    for idx, row in enumerate(rows):
        by_scene[str(row.get("scene"))].append(idx)
    params = {
        "D1_correlation_R4": {"jaccard": 0.35, "frame_gap": 999999, "max_rows_per_tube": 36, "neighbors": 6},
        "D2_adjacent_matching_R6": {"jaccard": 0.20, "frame_gap": 20, "max_rows_per_tube": 32, "neighbors": 5},
        "D3_hybrid_unknown_R6": {"jaccard": 0.28, "frame_gap": 999999, "max_rows_per_tube": 36, "neighbors": 6},
        "D4_shuffled_d4rt_control": {"jaccard": 0.28, "frame_gap": 999999, "max_rows_per_tube": 36, "neighbors": 6},
        "D5_mask_only_control": {"jaccard": 0.25, "frame_gap": 999999, "max_rows_per_tube": 48, "neighbors": 8},
    }.get(variant, {"jaccard": 0.30, "frame_gap": 999999, "max_rows_per_tube": 36, "neighbors": 6})
    for _, indices in by_scene.items():
        by_tube: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            for tube_id in _parse_core_tube_ids(rows[idx]):
                by_tube[int(tube_id)].append(idx)
        seen_pairs: set[tuple[int, int]] = set()
        for owners in by_tube.values():
            owners = sorted(
                owners,
                key=lambda idx: (int(rows[idx].get("frame_id", 0)), -_method_score(rows[idx])),
            )[: int(params["max_rows_per_tube"])]
            for left_pos, left in enumerate(owners):
                left_row = rows[left]
                for right in owners[left_pos + 1 : left_pos + 1 + int(params["neighbors"])]:
                    pair = (min(left, right), max(left, right))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    right_row = rows[right]
                    if int(left_row.get("frame_id", -1)) == int(right_row.get("frame_id", -2)) and int(left_row.get("mask_id", -1)) != int(right_row.get("mask_id", -2)):
                        continue
                    if abs(int(left_row.get("frame_id", 0)) - int(right_row.get("frame_id", 0))) > int(params["frame_gap"]):
                        continue
                    if _jaccard(_parse_core_tube_ids(left_row), _parse_core_tube_ids(right_row)) >= float(params["jaccard"]):
                        uf.union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        groups[uf.find(idx)].append(idx)
    return _groups_to_masklets(rows, list(groups.values()), variant)


def _groups_to_masklets(rows: list[dict[str, Any]], groups: list[list[int]], variant: str) -> list[dict[str, Any]]:
    out = []
    for midx, group in enumerate(groups):
        tube_counter: Counter[int] = Counter()
        frames = []
        frame_masks: dict[int, set[int]] = defaultdict(set)
        for idx in group:
            row = rows[idx]
            frames.append(int(row.get("frame_id", -1)))
            frame_masks[int(row.get("frame_id", -1))].add(int(row.get("mask_id", -1)))
            tube_counter.update(_parse_core_tube_ids(row))
        out.append(
            {
                "masklet_id": f"{variant}_{midx:06d}",
                "variant": variant,
                "scene": str(rows[group[0]].get("scene")),
                "region_indices": group,
                "region_count": int(len(group)),
                "tube_support": dict(tube_counter),
                "tube_count": int(len(tube_counter)),
                "temporal_span": int(max(frames) - min(frames) + 1) if frames else 0,
                "same_frame_cannot_link_violations": int(sum(max(len(masks) - 1, 0) for masks in frame_masks.values())),
            }
        )
    return out


def _assign_tubes(
    masklets: list[dict[str, Any]],
    gt_labels: dict[int, int],
    *,
    unknown_min_support: int,
    unknown_min_fraction: float,
) -> tuple[dict[int, int], float]:
    tube_candidates: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for midx, masklet in enumerate(masklets):
        support = {int(k): int(v) for k, v in dict(masklet.get("tube_support") or {}).items()}
        max_support = max(support.values()) if support else 1
        for tube_id, count in support.items():
            tube_candidates[int(tube_id)].append((midx, int(count), float(count / max(max_support, 1))))
    labels_pred: dict[int, int] = {}
    unknown = 0
    next_unknown = len(masklets) + 1
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        candidates = tube_candidates.get(int(tube_id), [])
        if not candidates:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown += 1
            continue
        best = max(candidates, key=lambda item: (item[1], item[2]))
        if int(best[1]) < int(unknown_min_support) or float(best[2]) < float(unknown_min_fraction):
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown += 1
        else:
            labels_pred[int(tube_id)] = int(best[0])
    return labels_pred, float(unknown / max(sum(1 for value in gt_labels.values() if int(value) > 0), 1))


def _gate(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "ari_pass": row.get("ARI") is not None and float(row["ARI"]) >= LOCAL_GATE["ARI"],
        "purity_pass": row.get("purity") is not None and float(row["purity"]) >= LOCAL_GATE["purity"],
        "completeness_pass": row.get("completeness") is not None and float(row["completeness"]) >= LOCAL_GATE["completeness"],
        "unknown_pass": row.get("unknown_tube_ratio") is not None and float(row["unknown_tube_ratio"]) <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": row.get("scene0081_ARI") is not None and float(row["scene0081_ARI"]) >= LOCAL_GATE["scene0081_ARI"],
    }
    return {**checks, "pass_3D_gate": bool(all(checks.values()))}


def _evaluate_variant(
    *,
    rows: list[dict[str, Any]],
    gt_labels: dict[str, dict[int, int]],
    variant: str,
    control_kind: str,
    unknown_min_support: int,
    unknown_min_fraction: float,
) -> MaskletResult:
    variant_rows = _filter_rows_for_variant(rows, variant)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in variant_rows:
        by_scene[str(row.get("scene"))].append(row)
    scene_metrics = []
    all_masklets = []
    for scene, scene_rows in sorted(by_scene.items()):
        masklets = _make_masklets(scene_rows, variant)
        labels_pred, unknown_ratio = _assign_tubes(
            masklets,
            gt_labels.get(scene, {}),
            unknown_min_support=int(unknown_min_support),
            unknown_min_fraction=float(unknown_min_fraction),
        )
        metrics = _cluster_metrics(labels_pred, gt_labels.get(scene, {}))
        row = {
            "scene": scene,
            "variant": variant,
            "control_kind": control_kind,
            "masklet_count": int(len(masklets)),
            "regions_per_masklet_mean": _mean([m["region_count"] for m in masklets]),
            "temporal_span_mean": _mean([m["temporal_span"] for m in masklets]),
            "same_frame_cannot_link_violations": int(sum(int(m["same_frame_cannot_link_violations"]) for m in masklets)),
            "ARI": metrics.get("ari"),
            "purity": metrics.get("purity"),
            "completeness": metrics.get("completeness"),
            "unknown_tube_ratio": unknown_ratio,
            "labeled_tube_count": metrics.get("labeled_tube_count"),
        }
        scene_metrics.append(row)
        all_masklets.extend(masklets)
    all_row = {
        "scene": "ALL",
        "variant": variant,
        "control_kind": control_kind,
        "masklet_count": int(sum(int(row["masklet_count"]) for row in scene_metrics)),
        "regions_per_masklet_mean": _mean([row["regions_per_masklet_mean"] for row in scene_metrics]),
        "temporal_span_mean": _mean([row["temporal_span_mean"] for row in scene_metrics]),
        "same_frame_cannot_link_violations": int(sum(int(row["same_frame_cannot_link_violations"]) for row in scene_metrics)),
        "ARI": _mean([row["ARI"] for row in scene_metrics]),
        "purity": _mean([row["purity"] for row in scene_metrics]),
        "completeness": _mean([row["completeness"] for row in scene_metrics]),
        "unknown_tube_ratio": _mean([row["unknown_tube_ratio"] for row in scene_metrics]),
        "scene0081_ARI": next((row.get("ARI") for row in scene_metrics if row.get("scene") == "scene0081_01"), None),
    }
    all_row.update(_gate(all_row))
    return MaskletResult(variant=variant, control_kind=control_kind, rows=variant_rows, masklets=all_masklets, scene_metrics=scene_metrics, all_metrics=all_row)


def _phase_d_e(
    *,
    real_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    gt_labels: dict[str, dict[int, int]],
    out_dir: Path,
) -> dict[str, Any]:
    configs = [
        ("D0_greedy_temporal_R1", "real", real_rows, 1, 0.0),
        ("D1_correlation_R4", "real", real_rows, 1, 0.0),
        ("D2_adjacent_matching_R6", "real", real_rows, 2, 0.0),
        ("D3_hybrid_unknown_R6", "real", real_rows, 2, 0.20),
        ("D4_shuffled_d4rt_control", "shuffled_d4rt", shuffled_rows, 2, 0.20),
        ("D5_mask_only_control", "mask_only", real_rows, 1, 0.0),
        ("D6_no_temporal_control", "no_temporal", real_rows, 1, 0.0),
    ]
    results = [
        _evaluate_variant(
            rows=rows,
            gt_labels=gt_labels,
            variant=variant,
            control_kind=control_kind,
            unknown_min_support=min_support,
            unknown_min_fraction=min_fraction,
        )
        for variant, control_kind, rows, min_support, min_fraction in configs
    ]
    masklet_rows = []
    assignment_rows = []
    scene_rows = []
    for result in results:
        masklet_rows.append(
            {
                "variant": result.variant,
                "control_kind": result.control_kind,
                "masklet_count": result.all_metrics["masklet_count"],
                "regions_per_masklet_mean": result.all_metrics["regions_per_masklet_mean"],
                "temporal_span_mean": result.all_metrics["temporal_span_mean"],
                "same_frame_cannot_link_violations": result.all_metrics["same_frame_cannot_link_violations"],
                "scene0081_masklet_ARI": result.all_metrics.get("scene0081_ARI"),
            }
        )
        assignment_rows.append(result.all_metrics)
        scene_rows.extend(result.scene_metrics)
    real_best = max(
        [row for row in assignment_rows if row["control_kind"] == "real"],
        key=lambda row: float(row.get("ARI") or -999.0),
    )
    control_best = {
        kind: max([row for row in assignment_rows if row["control_kind"] == kind], key=lambda row: float(row.get("ARI") or -999.0))
        for kind in ["shuffled_d4rt", "mask_only", "no_temporal"]
    }
    real_vs_controls = {
        "real_best_variant": real_best["variant"],
        "real_best_ARI": real_best["ARI"],
        "shuffled_best_ARI": control_best["shuffled_d4rt"]["ARI"],
        "mask_only_best_ARI": control_best["mask_only"]["ARI"],
        "no_temporal_best_ARI": control_best["no_temporal"]["ARI"],
        "real_minus_shuffled": None if real_best["ARI"] is None or control_best["shuffled_d4rt"]["ARI"] is None else float(real_best["ARI"] - control_best["shuffled_d4rt"]["ARI"]),
        "real_minus_mask_only": None if real_best["ARI"] is None or control_best["mask_only"]["ARI"] is None else float(real_best["ARI"] - control_best["mask_only"]["ARI"]),
        "real_minus_no_temporal": None if real_best["ARI"] is None or control_best["no_temporal"]["ARI"] is None else float(real_best["ARI"] - control_best["no_temporal"]["ARI"]),
    }
    real_vs_controls["control_gate_pass"] = bool(
        real_vs_controls["real_minus_shuffled"] is not None
        and real_vs_controls["real_minus_shuffled"] >= 0.20
        and real_vs_controls["real_minus_mask_only"] is not None
        and real_vs_controls["real_minus_mask_only"] >= 0.05
        and real_vs_controls["real_minus_no_temporal"] is not None
        and real_vs_controls["real_minus_no_temporal"] >= 0.05
    )
    _write_csv(out_dir / "masklet_graph_summary.csv", masklet_rows)
    _write_json(out_dir / "masklet_graph_summary.json", masklet_rows)
    _write_csv(out_dir / "tube_assignment_summary.csv", assignment_rows)
    _write_json(out_dir / "tube_assignment_summary.json", assignment_rows)
    _write_csv(out_dir / "tube_assignment_scene_rows.csv", scene_rows)
    _write_json(out_dir / "tube_assignment_scene_rows.json", scene_rows)
    _write_json(out_dir / "real_vs_controls.json", real_vs_controls)
    return {
        "masklet_graph_summary": masklet_rows,
        "tube_assignment_summary": assignment_rows,
        "real_vs_controls": real_vs_controls,
    }


def _phase_c_dino(feature_root: Path, out_dir: Path) -> dict[str, Any]:
    feature_csv = feature_root / "routeB_feature_metrics.csv"
    object_csv = feature_root / "routeB_object_metrics.csv"
    feature_rows = _read_csv_rows(feature_csv)
    object_rows = _read_csv_rows(object_csv)
    all_feature = next((row for row in feature_rows if row.get("scene") == "ALL"), None)
    manifest = {
        "phase": "v36_phaseC",
        "feature_root": str(feature_root),
        "feature_csv_exists": feature_csv.exists(),
        "object_csv_exists": object_csv.exists(),
        "uses_frozen_visual_backbone": feature_csv.exists(),
        "same_GT_region_pair_AUC": _float(all_feature.get("same_GT_pair_AUC")) if all_feature else None,
        "mixed_region_AUC": _float(all_feature.get("mixed_region_AUC")) if all_feature else None,
        "scene0081_AUC": _float(all_feature.get("scene0081_feature_AUC")) if all_feature else None,
        "phaseC_pass": False,
        "not_run_reason": "" if feature_csv.exists() else "feature metrics not found yet",
    }
    manifest["phaseC_pass"] = bool(
        manifest["same_GT_region_pair_AUC"] is not None
        and manifest["same_GT_region_pair_AUC"] >= 0.80
        and manifest["mixed_region_AUC"] is not None
        and manifest["mixed_region_AUC"] >= 0.78
        and manifest["scene0081_AUC"] is not None
        and manifest["scene0081_AUC"] >= 0.70
    )
    _write_json(out_dir / "dino_feature_quality.json", manifest)
    if feature_rows:
        _write_csv(out_dir / "dino_feature_quality.csv", feature_rows)
    if object_rows:
        _write_csv(out_dir / "dino_object_metrics_imported.csv", object_rows)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    gt_labels = _load_gt_labels(args, scenes)
    real_rows = _read_json(Path(args.real_proposal_root) / f"{args.real_label}_proposal_rows.json")
    shuffled_rows = _read_json(Path(args.shuffled_proposal_root) / f"{args.shuffled_label}_proposal_rows.json")
    out_root = Path(args.output_root)
    phase_b = _phase_b_region_sources(real_rows, gt_labels, out_root / "v36_region_sources")
    phase_c = _phase_c_dino(Path(args.dino_feature_root), out_root / "v36_dino_features")
    phase_de = _phase_d_e(
        real_rows=real_rows,
        shuffled_rows=shuffled_rows,
        gt_labels=gt_labels,
        out_dir=out_root / "v36_masklet_graph",
    )
    summary = {
        "phase": "v36_masklet_first_identity",
        "plan": "docs/stream4d_v36_masklet_first_object_identity_plan.md",
        "region_source_generated_from_rgb_masks": True,
        "uses_old_proposal_rows_only": False,
        "phaseB_any_success": phase_b["manifest"]["phaseB_any_success"],
        "phaseC_pass": phase_c["phaseC_pass"],
        "best_real_assignment": max(
            [row for row in phase_de["tube_assignment_summary"] if row["control_kind"] == "real"],
            key=lambda row: float(row.get("ARI") or -999.0),
        ),
        "real_vs_controls": phase_de["real_vs_controls"],
        "allowed_4d": False,
        "allowed_ap": False,
        "is_method_result": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    best = summary["best_real_assignment"]
    summary["pass_3D_gate"] = bool(best.get("pass_3D_gate"))
    summary["pass_controls"] = bool(phase_de["real_vs_controls"].get("control_gate_pass"))
    summary["allowed_4d"] = bool(summary["pass_3D_gate"] and summary["pass_controls"])
    summary["allowed_ap"] = bool(summary["allowed_4d"])
    _write_json(out_root / "v36_masklet_first_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-proposal-root", default="outputs/audit/v36_region_sources/real")
    parser.add_argument("--real-label", default="v36_region_sources_real")
    parser.add_argument("--shuffled-proposal-root", default="outputs/audit/v36_region_sources/shuffled_d4rt")
    parser.add_argument("--shuffled-label", default="v36_region_sources_shuffled_d4rt")
    parser.add_argument("--dino-feature-root", default="outputs/audit/v36_dino_features/real")
    parser.add_argument("--output-root", default="outputs/audit")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
