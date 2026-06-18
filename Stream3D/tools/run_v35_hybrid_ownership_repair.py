from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _load_gt_labels


LOCAL_GATE = {
    "local_ARI": 0.40,
    "local_purity": 0.85,
    "local_completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_local_ARI": 0.20,
}

CONTROL_GATE = {
    "real_vs_shuffled_margin": 0.20,
    "real_vs_no_temporal_margin": 0.05,
    "real_vs_mask_only_margin": 0.05,
    "window0_baseline_ari": 0.35140920829133926,
}

O5_BASE_TYPES = {
    "R0_full_mask_region",
    "R1_boundary_eroded_interior",
    "R2_distance_watershed_region",
    "R3_d4rt_tube_seeded_voronoi",
    "R4_image_gradient_split",
    "R5_d4rt_canonical_adjacency_split",
    "R6_mask_overlap_consensus_region",
    "R6_mask_overlap_consensus_union",
    "R7_high_purity_core_region",
}

MASK_ONLY_TYPES = {
    "R0_full_mask_region",
    "R1_boundary_eroded_interior",
    "R2_distance_watershed_region",
    "R4_image_gradient_split",
    "R6_mask_overlap_consensus_region",
    "R6_mask_overlap_consensus_union",
    "R7_high_purity_core_region",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
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
    vals = [_float(value) for value in values]
    vals = [float(value) for value in vals if value is not None]
    return float(np.mean(vals)) if vals else None


def _parse_core_tube_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_core_tube_ids" in row and isinstance(row.get("_core_tube_ids"), list):
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or []))
    text = str(row.get("core_tube_ids") or "")
    if not text:
        return ()
    return tuple(sorted(int(part) for part in text.split(";") if part.strip()))


def _is_temporal(row: dict[str, Any]) -> bool:
    return str(row.get("proposal_type") or "").startswith(("R8_", "R9_", "R10_", "R11_", "R12_"))


def _is_o5(row: dict[str, Any]) -> bool:
    ptype = str(row.get("proposal_type") or "")
    return ptype in O5_BASE_TYPES or _is_temporal(row)


def _is_mask_only(row: dict[str, Any]) -> bool:
    return str(row.get("proposal_type") or "") in MASK_ONLY_TYPES


def _is_anchor_source(row: dict[str, Any]) -> bool:
    return str(row.get("proposal_type") or "") in {
        "R1_boundary_eroded_interior",
        "R2_distance_watershed_region",
        "R3_d4rt_tube_seeded_voronoi",
        "R4_image_gradient_split",
        "R5_d4rt_canonical_adjacency_split",
        "R7_high_purity_core_region",
    }


def _is_pack_variant(variant: str) -> bool:
    return variant.startswith(
        ("H7_", "H8_", "H9_", "H10_", "H11_", "H12_", "H13_", "H14_", "H15_", "H16_", "H17_")
    )


def _source_weight(row: dict[str, Any]) -> float:
    ptype = str(row.get("proposal_type") or "")
    if ptype.startswith("R7_"):
        return 0.36
    if ptype.startswith("R1_"):
        return 0.26
    if ptype.startswith("R5_"):
        return 0.20
    if ptype.startswith("R3_"):
        return 0.16
    if ptype.startswith("R4_"):
        return 0.12
    if ptype.startswith(("R10_", "R12_")):
        return 0.12
    if ptype.startswith("R2_"):
        return 0.06
    if ptype.startswith("R0_"):
        return -0.24
    if ptype.startswith(("R8_", "R9_")):
        return -0.08
    return 0.0


def _row_score(row: dict[str, Any], *, strict: bool) -> float:
    core = max(float(len(_parse_core_tube_ids(row))), 1.0)
    score = 0.0
    score += _source_weight(row)
    score += 0.26 * (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0)
    score += 0.12 * (_float(row.get("visibility_mean"), 0.5) or 0.5)
    score += 0.10 * (_float(row.get("confidence_mean"), 0.5) or 0.5)
    score += 0.08 * (_float(row.get("mask_temporal_repeat_score"), 0.0) or 0.0)
    score += 0.07 * min(math.log1p(core) / math.log(96.0), 1.0)
    score -= 0.11 * math.log1p(max(_float(row.get("same_frame_cannot_link_rate"), 0.0) or 0.0, 0.0))
    score -= 0.10 * math.log1p(max(_float(row.get("visible_outside_negative_rate"), 0.0) or 0.0, 0.0))
    score -= 0.08 * (_float(row.get("boundary_contact_ratio"), 0.0) or 0.0)
    score -= 0.04 * math.log1p(max(_float(row.get("image_gradient_boundary_score"), 0.0) or 0.0, 0.0))
    if strict:
        score -= 0.06 * (_float(row.get("appearance_variance"), 0.0) or 0.0)
        score -= 0.08 * math.log1p(max(_float(row.get("overlap_with_other_proposals"), 0.0) or 0.0, 0.0))
    return float(score)


def _row_allowed(row: dict[str, Any], *, variant: str) -> bool:
    ptype = str(row.get("proposal_type") or "")
    if not _is_o5(row):
        return False
    if variant.endswith("_no_temporal") and _is_temporal(row):
        return False
    if variant.endswith("_mask_only") and not _is_mask_only(row):
        return False
    if variant in {"H1_o5_strict_graph", "H2_o5_core_expand", "H3_o5_proxy_purity_graph"}:
        if (_float(row.get("same_frame_cannot_link_rate"), 0.0) or 0.0) > 12.0:
            return False
        if (_float(row.get("visible_outside_negative_rate"), 0.0) or 0.0) > 12.0:
            return False
    if variant == "H3_o5_proxy_purity_graph":
        if (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0) < 0.50:
            return False
        if (_float(row.get("visibility_mean"), 0.0) or 0.0) < 0.55:
            return False
    if variant == "H2_o5_core_expand":
        return ptype.startswith(("R1_", "R3_", "R4_", "R5_", "R7_", "R10_", "R12_"))
    return True


def _variant_params(variant: str) -> dict[str, Any]:
    params = {
        "score_floor": -0.12,
        "edge_floor": 0.55,
        "support_floor": 1.0,
        "min_component_tubes": 2,
        "max_row_tubes": 120,
        "strict": False,
        "expand": False,
    }
    if variant in {"H0_o5_all_graph", "H4_o5_shuffled", "H5_o5_no_temporal", "H6_o5_mask_only"}:
        return params
    if variant == "H1_o5_strict_graph":
        params.update(score_floor=0.02, edge_floor=0.82, support_floor=1.4, min_component_tubes=2, strict=True)
    elif variant == "H2_o5_core_expand":
        params.update(score_floor=0.04, edge_floor=0.72, support_floor=1.2, min_component_tubes=2, strict=True, expand=True)
    elif variant == "H3_o5_proxy_purity_graph":
        params.update(score_floor=0.08, edge_floor=0.92, support_floor=1.6, min_component_tubes=3, strict=True)
    return params


def _pack_base_variant(variant: str) -> str:
    return {
        "H10_o5_anchor_shuffled": "H7_o5_anchor_pack",
        "H11_o5_anchor_no_temporal": "H9_o5_anchor_temporal_rescue_no_temporal",
        "H12_o5_anchor_mask_only": "H9_o5_anchor_temporal_rescue_mask_only",
    }.get(variant, variant)


def _pack_params(variant: str) -> dict[str, Any]:
    base = _pack_base_variant(variant).replace("_no_temporal", "").replace("_mask_only", "")
    params = {
        "anchor_types": {
            "R1_boundary_eroded_interior",
            "R2_distance_watershed_region",
            "R3_d4rt_tube_seeded_voronoi",
            "R4_image_gradient_split",
            "R5_d4rt_canonical_adjacency_split",
            "R7_high_purity_core_region",
        },
        "min_row_tubes": 3,
        "max_row_tubes": 120,
        "min_visibility": 0.62,
        "min_eroded": 0.78,
        "min_score": 0.96,
        "jaccard_floor": 0.42,
        "containment_floor": 0.72,
        "min_object_tubes": 3,
        "residual_ratio_floor": 0.32,
        "duplicate_overlap_ratio": 0.78,
        "temporal_rescue": False,
        "max_temporal_row_tubes": 420,
        "temporal_overlap_floor": 0.25,
        "temporal_new_ratio_cap": 0.18,
    }
    if base == "H8_o5_anchor_strict_pack":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R5_d4rt_canonical_adjacency_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.72,
            min_eroded=0.92,
            min_score=1.08,
            jaccard_floor=0.52,
            containment_floor=0.82,
            min_object_tubes=5,
            residual_ratio_floor=0.45,
        )
    elif base == "H9_o5_anchor_temporal_rescue":
        params.update(temporal_rescue=True, min_score=0.98, jaccard_floor=0.45, containment_floor=0.76)
    elif base == "H13_o5_mask_balanced_pack":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R2_distance_watershed_region",
                "R4_image_gradient_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.66,
            min_eroded=0.84,
            min_score=1.00,
            jaccard_floor=0.40,
            containment_floor=0.70,
            min_object_tubes=3,
            residual_ratio_floor=0.34,
        )
    elif base == "H14_o5_mask_strict_fill":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R4_image_gradient_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.72,
            min_eroded=0.92,
            min_score=1.08,
            jaccard_floor=0.48,
            containment_floor=0.78,
            min_object_tubes=4,
            residual_ratio_floor=0.42,
            mask_rescue=True,
            rescue_types={
                "R0_full_mask_region",
                "R1_boundary_eroded_interior",
                "R4_image_gradient_split",
                "R6_mask_overlap_consensus_region",
                "R6_mask_overlap_consensus_union",
                "R7_high_purity_core_region",
            },
            max_rescue_row_tubes=160,
            rescue_min_eroded=0.72,
            rescue_min_visibility=0.58,
            rescue_score_floor=0.78,
            rescue_overlap_floor=0.30,
            rescue_new_ratio_cap=0.35,
        )
    elif base == "H15_o5_strict_core_mask_fill":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R5_d4rt_canonical_adjacency_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.72,
            min_eroded=0.92,
            min_score=1.08,
            jaccard_floor=0.52,
            containment_floor=0.82,
            min_object_tubes=5,
            residual_ratio_floor=0.45,
            mask_rescue=True,
            rescue_types={
                "R0_full_mask_region",
                "R1_boundary_eroded_interior",
                "R4_image_gradient_split",
                "R6_mask_overlap_consensus_region",
                "R6_mask_overlap_consensus_union",
                "R7_high_purity_core_region",
            },
            max_rescue_row_tubes=160,
            rescue_min_eroded=0.70,
            rescue_min_visibility=0.56,
            rescue_score_floor=0.76,
            rescue_overlap_floor=0.24,
            rescue_new_ratio_cap=0.45,
        )
    elif base == "H16_o5_visual_mask_balanced":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R2_distance_watershed_region",
                "R4_image_gradient_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.64,
            min_eroded=0.82,
            min_score=1.08,
            jaccard_floor=0.40,
            containment_floor=0.70,
            min_object_tubes=3,
            residual_ratio_floor=0.34,
            visual_min_compactness=0.42,
        )
    elif base == "H17_o5_visual_strict_fill":
        params.update(
            anchor_types={
                "R1_boundary_eroded_interior",
                "R4_image_gradient_split",
                "R7_high_purity_core_region",
            },
            min_visibility=0.70,
            min_eroded=0.88,
            min_score=1.14,
            jaccard_floor=0.46,
            containment_floor=0.76,
            min_object_tubes=4,
            residual_ratio_floor=0.40,
            visual_min_compactness=0.45,
            mask_rescue=True,
            rescue_types={
                "R0_full_mask_region",
                "R1_boundary_eroded_interior",
                "R4_image_gradient_split",
                "R6_mask_overlap_consensus_region",
                "R6_mask_overlap_consensus_union",
                "R7_high_purity_core_region",
            },
            max_rescue_row_tubes=160,
            rescue_min_eroded=0.70,
            rescue_min_visibility=0.56,
            rescue_score_floor=0.84,
            rescue_overlap_floor=0.28,
            rescue_new_ratio_cap=0.35,
        )
    return params


def _pack_source_weight(row: dict[str, Any]) -> float:
    ptype = str(row.get("proposal_type") or "")
    if ptype.startswith("R7_"):
        return 0.92
    if ptype.startswith("R1_"):
        return 0.88
    if ptype.startswith(("R3_", "R5_")):
        return 0.78
    if ptype.startswith("R2_"):
        return 0.70
    if ptype.startswith("R4_"):
        return 0.68
    if ptype.startswith(("R0_", "R6_")):
        return 0.46
    if _is_temporal(row):
        return 0.18
    return 0.0


def _pack_score(row: dict[str, Any]) -> float:
    tube_count = max(len(_parse_core_tube_ids(row)), 1)
    size_score = min(math.log1p(tube_count) / math.log(80.0), 1.0)
    score = _pack_source_weight(row)
    score += 0.12 * (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0)
    score += 0.10 * (_float(row.get("visibility_mean"), 0.0) or 0.0)
    score += 0.04 * (_float(row.get("confidence_mean"), 0.0) or 0.0)
    score += 0.12 * size_score
    if bool(row.get("_use_visual_compactness")) and _float(row.get("visual_compactness")) is not None:
        score += 0.18 * (_float(row.get("visual_compactness"), 0.0) or 0.0)
    return float(score)


def _pack_row_allowed(row: dict[str, Any], *, variant: str) -> bool:
    params = _pack_params(variant)
    ptype = str(row.get("proposal_type") or "")
    if variant.endswith("_mask_only") and not _is_mask_only(row):
        return False
    if variant.endswith("_no_temporal") and _is_temporal(row):
        return False
    if ptype not in params["anchor_types"]:
        return False
    tube_count = len(_parse_core_tube_ids(row))
    if tube_count < int(params["min_row_tubes"]) or tube_count > int(params["max_row_tubes"]):
        return False
    if (_float(row.get("visibility_mean"), 0.0) or 0.0) < float(params["min_visibility"]):
        return False
    if (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0) < float(params["min_eroded"]):
        return False
    if "visual_min_compactness" in params:
        vc = _float(row.get("visual_compactness"))
        if vc is None or vc < float(params["visual_min_compactness"]):
            return False
    return _pack_score(row) >= float(params["min_score"])


def _stable_scene_seed(seed: int, scene: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _shuffled_rows(rows: list[dict[str, Any]], *, seed: int, scene: str) -> list[dict[str, Any]]:
    tube_ids = sorted({int(tid) for row in rows for tid in _parse_core_tube_ids(row)})
    if not tube_ids:
        return []
    shuffled = list(tube_ids)
    rng = np.random.default_rng(_stable_scene_seed(seed, scene))
    rng.shuffle(shuffled)
    if shuffled == tube_ids:
        shuffled = shuffled[1:] + shuffled[:1]
    remap = dict(zip(tube_ids, shuffled))
    out = []
    for row in rows:
        item = dict(row)
        core = tuple(sorted(remap[int(tid)] for tid in _parse_core_tube_ids(row)))
        item["_core_tube_ids"] = list(core)
        item["core_tube_ids"] = ";".join(str(tid) for tid in core)
        item["proposal_id"] = f"{row.get('proposal_id')}_hybrid_shuf"
        out.append(item)
    return out


def _selected_rows(rows: list[dict[str, Any]], *, variant: str, scene: str, seed: int) -> list[dict[str, Any]]:
    if _is_pack_variant(variant):
        base_variant = _pack_base_variant(variant)
        selected = [dict(row) for row in rows if _is_o5(row)]
        if base_variant.startswith(("H16_", "H17_")):
            for row in selected:
                row["_use_visual_compactness"] = True
        if base_variant.endswith("_no_temporal"):
            selected = [row for row in selected if not _is_temporal(row)]
        if base_variant.endswith("_mask_only"):
            selected = [row for row in selected if _is_mask_only(row)]
        if variant == "H10_o5_anchor_shuffled":
            selected = _shuffled_rows(selected, seed=seed, scene=scene)
        return selected
    base_variant = {
        "H4_o5_shuffled": "H0_o5_all_graph",
        "H5_o5_no_temporal": "H0_o5_all_graph_no_temporal",
        "H6_o5_mask_only": "H0_o5_all_graph_mask_only",
    }.get(variant, variant)
    selected = [row for row in rows if _row_allowed(row, variant=base_variant)]
    params = _variant_params(variant if variant not in {"H4_o5_shuffled", "H5_o5_no_temporal", "H6_o5_mask_only"} else "H0_o5_all_graph")
    selected = [row for row in selected if _row_score(row, strict=bool(params["strict"])) >= float(params["score_floor"])]
    if variant == "H4_o5_shuffled":
        selected = _shuffled_rows(selected, seed=seed, scene=scene)
    return selected


def _build_graph_components(rows: list[dict[str, Any]], *, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = _variant_params(variant)
    parent: dict[int, int] = {}
    edge_weight: Counter[tuple[int, int]] = Counter()
    edge_support: Counter[tuple[int, int]] = Counter()
    row_support = 0
    for row in rows:
        ids = list(_parse_core_tube_ids(row))
        if len(ids) < 2:
            continue
        ids = ids[: int(params["max_row_tubes"])]
        score = _row_score(row, strict=bool(params["strict"]))
        if score < float(params["score_floor"]):
            continue
        row_support += 1
        row_weight = max(0.05, score + 0.35)
        for tid in ids:
            parent.setdefault(int(tid), int(tid))
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                key = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
                edge_weight[key] += row_weight
                edge_support[key] += 1

    def find(tid: int) -> int:
        cur = int(tid)
        while parent[cur] != cur:
            parent[cur] = parent[parent[cur]]
            cur = parent[cur]
        return cur

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    kept_edges = 0
    for (a, b), weight in edge_weight.items():
        support = edge_support[(a, b)]
        if float(weight) >= float(params["edge_floor"]) and float(support) >= float(params["support_floor"]):
            union(a, b)
            kept_edges += 1
    comps: dict[int, set[int]] = defaultdict(set)
    for tid in list(parent):
        comps[find(tid)].add(int(tid))
    objects = []
    for comp in sorted(comps.values(), key=lambda item: (len(item), min(item)), reverse=True):
        if len(comp) >= int(params["min_component_tubes"]):
            objects.append({"tube_set": set(comp), "supporting_regions": [], "confidence": float(len(comp))})
    if bool(params.get("expand")):
        assigned = {tid for obj in objects for tid in obj["tube_set"]}
        for row in rows:
            ids = set(_parse_core_tube_ids(row))
            if len(ids) < 2:
                continue
            best_idx = None
            best_overlap = 0
            for idx, obj in enumerate(objects):
                overlap = len(ids & obj["tube_set"])
                if overlap > best_overlap:
                    best_idx = idx
                    best_overlap = overlap
            if best_idx is not None and best_overlap >= 2:
                safe = {tid for tid in ids if tid not in assigned}
                if len(safe) <= max(6, 2 * best_overlap):
                    objects[int(best_idx)]["tube_set"].update(safe)
                    assigned.update(safe)
    diag = {
        "selected_row_count": int(len(rows)),
        "row_support_count": int(row_support),
        "edge_count": int(len(edge_weight)),
        "kept_edge_count": int(kept_edges),
        "object_count": int(len(objects)),
    }
    return objects, diag


def _build_anchor_pack_objects(rows: list[dict[str, Any]], *, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = _pack_params(variant)
    anchors = [row for row in rows if _pack_row_allowed(row, variant=_pack_base_variant(variant))]
    tube_sets = [set(_parse_core_tube_ids(row)) for row in anchors]
    parent = {idx: idx for idx in range(len(anchors))}

    def find(idx: int) -> int:
        cur = int(idx)
        while parent[cur] != cur:
            parent[cur] = parent[parent[cur]]
            cur = parent[cur]
        return cur

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_tube: dict[int, list[int]] = defaultdict(list)
    for idx, tube_set in enumerate(tube_sets):
        for tid in tube_set:
            by_tube[int(tid)].append(idx)

    pair_counts: Counter[tuple[int, int]] = Counter()
    for indices in by_tube.values():
        if len(indices) < 2:
            continue
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                key = (a, b) if a < b else (b, a)
                pair_counts[key] += 1

    kept_edges = 0
    for (a, b), inter in pair_counts.items():
        size_a = len(tube_sets[a])
        size_b = len(tube_sets[b])
        union_size = max(size_a + size_b - int(inter), 1)
        jaccard = float(inter / union_size)
        containment = float(inter / max(min(size_a, size_b), 1))
        if jaccard >= float(params["jaccard_floor"]) or containment >= float(params["containment_floor"]):
            union(a, b)
            kept_edges += 1

    comps: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(anchors)):
        comps[find(idx)].append(idx)

    candidates = []
    for indices in comps.values():
        tube_set: set[int] = set()
        score_sum = 0.0
        for idx in indices:
            tube_set.update(tube_sets[idx])
            score_sum += _pack_score(anchors[idx])
        if len(tube_set) < int(params["min_object_tubes"]):
            continue
        candidates.append(
            {
                "tube_set": tube_set,
                "supporting_regions": [anchors[idx].get("proposal_id") for idx in indices],
                "confidence": float(score_sum / max(len(indices), 1)),
                "support_count": int(len(indices)),
            }
        )

    candidates.sort(key=lambda item: (float(item["confidence"]), int(item["support_count"]), len(item["tube_set"])), reverse=True)
    objects = []
    assigned: set[int] = set()
    duplicate_skips = 0
    residual_skips = 0
    for cand in candidates:
        tube_set = set(int(tid) for tid in cand["tube_set"])
        overlap = len(tube_set & assigned)
        if overlap / max(len(tube_set), 1) >= float(params["duplicate_overlap_ratio"]):
            duplicate_skips += 1
            continue
        residual = tube_set - assigned
        if len(residual) < int(params["min_object_tubes"]):
            residual_skips += 1
            continue
        if len(residual) / max(len(tube_set), 1) < float(params["residual_ratio_floor"]):
            residual_skips += 1
            continue
        objects.append({"tube_set": set(residual), "supporting_regions": cand["supporting_regions"], "confidence": cand["confidence"]})
        assigned.update(residual)

    temporal_rescue_count = 0
    temporal_added_tubes = 0
    if bool(params["temporal_rescue"]) and objects:
        temporal_rows = []
        for row in rows:
            ids = set(_parse_core_tube_ids(row))
            if not _is_temporal(row) or len(ids) < 2 or len(ids) > int(params["max_temporal_row_tubes"]):
                continue
            temporal_rows.append((row, ids))
        for row, ids in temporal_rows:
            best_idx = None
            best_overlap = 0.0
            for idx, obj in enumerate(objects):
                overlap = len(ids & obj["tube_set"]) / max(len(obj["tube_set"]), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            if best_idx is None or best_overlap < float(params["temporal_overlap_floor"]):
                continue
            new_tubes = ids - assigned
            cap = max(2, int(float(params["temporal_new_ratio_cap"]) * len(objects[int(best_idx)]["tube_set"])))
            if not new_tubes or len(new_tubes) > cap:
                continue
            objects[int(best_idx)]["tube_set"].update(new_tubes)
            assigned.update(new_tubes)
            temporal_rescue_count += 1
            temporal_added_tubes += len(new_tubes)

    mask_rescue_count = 0
    mask_added_tubes = 0
    if bool(params.get("mask_rescue")) and objects:
        rescue_types = set(params.get("rescue_types") or [])
        rescue_rows = []
        for row in rows:
            ids = set(_parse_core_tube_ids(row))
            if _is_temporal(row) or str(row.get("proposal_type") or "") not in rescue_types:
                continue
            if len(ids) < 2 or len(ids) > int(params.get("max_rescue_row_tubes", 160)):
                continue
            if (_float(row.get("eroded_interior_ratio"), 0.0) or 0.0) < float(params.get("rescue_min_eroded", 0.0)):
                continue
            if (_float(row.get("visibility_mean"), 0.0) or 0.0) < float(params.get("rescue_min_visibility", 0.0)):
                continue
            if _pack_score(row) < float(params.get("rescue_score_floor", 0.0)):
                continue
            rescue_rows.append((row, ids))
        rescue_rows.sort(key=lambda item: (_pack_score(item[0]), len(item[1])), reverse=True)
        for row, ids in rescue_rows:
            best_idx = None
            best_overlap = 0.0
            for idx, obj in enumerate(objects):
                inter = len(ids & obj["tube_set"])
                overlap = inter / max(min(len(ids), len(obj["tube_set"])), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            if best_idx is None or best_overlap < float(params.get("rescue_overlap_floor", 0.0)):
                continue
            new_tubes = ids - assigned
            cap = max(2, int(float(params.get("rescue_new_ratio_cap", 0.0)) * len(objects[int(best_idx)]["tube_set"])))
            if not new_tubes or len(new_tubes) > cap:
                continue
            objects[int(best_idx)]["tube_set"].update(new_tubes)
            assigned.update(new_tubes)
            mask_rescue_count += 1
            mask_added_tubes += len(new_tubes)

    diag = {
        "selected_row_count": int(len(rows)),
        "row_support_count": int(len(anchors)),
        "edge_count": int(len(pair_counts)),
        "kept_edge_count": int(kept_edges),
        "object_count": int(len(objects)),
        "anchor_candidate_count": int(len(anchors)),
        "anchor_component_count": int(len(candidates)),
        "pack_duplicate_skips": int(duplicate_skips),
        "pack_residual_skips": int(residual_skips),
        "temporal_rescue_count": int(temporal_rescue_count),
        "temporal_added_tubes": int(temporal_added_tubes),
        "mask_rescue_count": int(mask_rescue_count),
        "mask_added_tubes": int(mask_added_tubes),
    }
    return objects, diag


def _evaluate_objects(objects: list[dict[str, Any]], gt_labels: dict[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
    labels_pred: dict[int, int] = {}
    object_rows: list[dict[str, Any]] = []
    for object_id, obj in enumerate(objects):
        tube_ids = sorted(int(tid) for tid in obj["tube_set"])
        assigned = []
        for tid in tube_ids:
            if int(tid) in labels_pred:
                continue
            labels_pred[int(tid)] = int(object_id)
            assigned.append(int(tid))
        object_rows.append(
            {
                "object_id": int(object_id),
                "tube_ids": assigned,
                "confidence": float(obj.get("confidence", 0.0)),
                "unknown/reject policy": "low-score rows and weak graph edges rejected; unassigned labeled tubes become unknown",
            }
        )
    next_label = len(object_rows)
    unknown_tubes = []
    for tid in labeled_tubes:
        if int(tid) not in labels_pred:
            labels_pred[int(tid)] = int(next_label)
            next_label += 1
            unknown_tubes.append(int(tid))
    metrics = _cluster_metrics(labels_pred, gt_labels) if gt_labels else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
    }
    metric_row = {
        "labeled_tube_count": int(len(labeled_tubes)),
        "object_count": int(len(object_rows)),
        "unknown_tube_count": int(len(unknown_tubes)),
        "unknown_tube_ratio": float(len(unknown_tubes) / max(len(labeled_tubes), 1)),
        "local_ARI": metrics["ari"],
        "local_purity": metrics["purity"],
        "local_completeness": metrics["completeness"],
        "local_overmerge": metrics["overmerge"],
        "local_oversplit": metrics["oversplit"],
    }
    return metric_row, object_rows, unknown_tubes


def _gate_status(row: dict[str, Any]) -> dict[str, Any]:
    ari = _float(row.get("local_ARI"))
    purity = _float(row.get("local_purity"))
    completeness = _float(row.get("local_completeness"))
    unknown = _float(row.get("unknown_tube_ratio"))
    scene0081 = _float(row.get("scene0081_local_ARI"))
    checks = {
        "ari_pass": ari is not None and ari >= LOCAL_GATE["local_ARI"],
        "purity_pass": purity is not None and purity >= LOCAL_GATE["local_purity"],
        "completeness_pass": completeness is not None and completeness >= LOCAL_GATE["local_completeness"],
        "unknown_pass": unknown is not None and unknown <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": scene0081 is not None and scene0081 >= LOCAL_GATE["scene0081_local_ARI"],
    }
    return {**checks, "local_gate_pass": bool(all(checks.values())), "local_gate_thresholds": dict(LOCAL_GATE)}


def _control_status(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_variant = {row["variant"]: row for row in rows}
    out: dict[str, dict[str, Any]] = {}
    for variant, row in by_variant.items():
        if variant.startswith(("H7_", "H8_", "H9_", "H13_", "H14_", "H15_", "H16_", "H17_")):
            control_variants = {
                "shuffled": "H10_o5_anchor_shuffled",
                "no_temporal": "H11_o5_anchor_no_temporal",
                "mask_only": "H12_o5_anchor_mask_only",
            }
        else:
            control_variants = {
                "shuffled": "H4_o5_shuffled",
                "no_temporal": "H5_o5_no_temporal",
                "mask_only": "H6_o5_mask_only",
            }
        shuffled = _float(by_variant.get(control_variants["shuffled"], {}).get("local_ARI"))
        no_temporal = _float(by_variant.get(control_variants["no_temporal"], {}).get("local_ARI"))
        mask_only = _float(by_variant.get(control_variants["mask_only"], {}).get("local_ARI"))
        ari = _float(row.get("local_ARI"))
        if ari is None or variant in {
            "H4_o5_shuffled",
            "H5_o5_no_temporal",
            "H6_o5_mask_only",
            "H10_o5_anchor_shuffled",
            "H11_o5_anchor_no_temporal",
            "H12_o5_anchor_mask_only",
        }:
            out[variant] = {"control_gate_pass": False}
            continue
        shuffled_pass = shuffled is not None and ari >= shuffled + CONTROL_GATE["real_vs_shuffled_margin"]
        no_temporal_pass = no_temporal is not None and ari >= no_temporal + CONTROL_GATE["real_vs_no_temporal_margin"]
        mask_pass = mask_only is not None and ari >= mask_only + CONTROL_GATE["real_vs_mask_only_margin"]
        window0_pass = ari >= CONTROL_GATE["window0_baseline_ari"]
        out[variant] = {
            "real_vs_shuffled": None if shuffled is None else float(ari - shuffled),
            "real_vs_no_temporal": None if no_temporal is None else float(ari - no_temporal),
            "real_vs_mask_only": None if mask_only is None else float(ari - mask_only),
            "control_shuffled_pass": shuffled_pass,
            "control_no_temporal_pass": no_temporal_pass,
            "control_mask_only_pass": mask_pass,
            "control_window0_baseline_pass": window0_pass,
            "control_gate_pass": bool(shuffled_pass and no_temporal_pass and mask_pass and window0_pass),
        }
    return out


def _attach_visual_compactness(
    args: argparse.Namespace, rows_by_scene: dict[str, list[dict[str, Any]]], scenes: list[str]
) -> list[dict[str, Any]]:
    from tools.run_v35_frozen_embedding_route import _extract_scene_embeddings, _make_model

    model = _make_model(args)
    diag_rows = []
    for scene in scenes:
        embeddings, _unused_gt_labels, diag = _extract_scene_embeddings(args, model, scene)
        attached = 0
        skipped_small = 0
        for row in rows_by_scene.get(scene, []):
            tube_ids = [int(tid) for tid in _parse_core_tube_ids(row) if int(tid) in embeddings]
            if len(tube_ids) < 3:
                skipped_small += 1
                continue
            vecs = np.stack([embeddings[int(tid)] for tid in tube_ids], axis=0)
            centroid = vecs.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
            row["visual_compactness"] = float(np.mean(vecs @ centroid))
            attached += 1
        diag_rows.append(
            {
                "scene": scene,
                "attached_visual_compactness_rows": int(attached),
                "skipped_small_or_unembedded_rows": int(skipped_small),
                "embedded_tube_count": int(diag.get("embedded_tube_count") or 0),
                "frame_count": int(diag.get("frame_count") or 0),
                "uses_gt_for_prediction": False,
                "note": "DINOv2 compactness uses RGB patch embeddings and D4RT tube UVs; GT returned by helper is ignored for prediction.",
            }
        )
    return diag_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_json(Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json")
    scenes = _read_split(Path(args.split))
    rows_by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in rows:
        scene = str(row.get("scene") or "")
        if scene in rows_by_scene:
            rows_by_scene[scene].append(row)
    variants = [
        "H0_o5_all_graph",
        "H1_o5_strict_graph",
        "H2_o5_core_expand",
        "H3_o5_proxy_purity_graph",
        "H4_o5_shuffled",
        "H5_o5_no_temporal",
        "H6_o5_mask_only",
        "H7_o5_anchor_pack",
        "H8_o5_anchor_strict_pack",
        "H9_o5_anchor_temporal_rescue",
        "H10_o5_anchor_shuffled",
        "H11_o5_anchor_no_temporal",
        "H12_o5_anchor_mask_only",
        "H13_o5_mask_balanced_pack",
        "H14_o5_mask_strict_fill",
        "H15_o5_strict_core_mask_fill",
        "H16_o5_visual_mask_balanced",
        "H17_o5_visual_strict_fill",
    ]
    if str(getattr(args, "variants", "") or "").strip():
        requested = [part.strip() for part in str(args.variants).split(",") if part.strip()]
        unknown = sorted(set(requested) - set(variants))
        if unknown:
            raise ValueError(f"Unknown variants requested: {unknown}")
        variants = requested
    visual_compactness_diag = []
    if any(variant.startswith(("H16_", "H17_")) for variant in variants):
        visual_compactness_diag = _attach_visual_compactness(args, rows_by_scene, scenes)
    gt_by_scene = {
        scene: _load_gt_labels(Path(args.cache_root), scene, int(args.max_tubes_per_window), int(args.image_width), int(args.image_height))
        for scene in scenes
    }
    summary_rows = []
    scene_rows = []
    for variant in variants:
        variant_scene_rows = []
        for scene in scenes:
            selected = _selected_rows(rows_by_scene.get(scene, []), variant=variant, scene=scene, seed=int(args.seed))
            if _is_pack_variant(variant):
                objects, graph_diag = _build_anchor_pack_objects(selected, variant=variant)
            else:
                objects, graph_diag = _build_graph_components(selected, variant=variant)
            metrics, object_rows, unknown_tubes = _evaluate_objects(objects, gt_by_scene.get(scene, {}))
            row = {
                "scene": scene,
                "variant": variant,
                "route": variant,
                "mask_source": "O5_hybrid_first_class",
                **graph_diag,
                **metrics,
                "is_method_result": True,
                "is_diagnostic_only": False,
                "forbidden_for_method_table": False,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "uses_rgbd_for_prediction": False,
                "uses_pose_for_prediction": False,
                "uses_scannet_mesh_for_prediction": False,
                "uses_eval_sim3_for_prediction": False,
                "uses_d4rt_self_sim3": True,
                "uses_frozen_visual_backbone": bool(variant.startswith(("H16_", "H17_"))),
                "visual_backbone_name": "DINOv2" if variant.startswith(("H16_", "H17_")) else "none",
                "geometry_field": "D4RT uv/visibility/confidence and non-GT proposal features from current v35 O5 hybrid pool",
                "coordinate_frame": "D4RT canonical tube ids; image-space mask observations",
                "alignment_source": "D4RT self-Sim3 inherited from proposal oracle manifest",
            }
            row = {**row, **_gate_status(row)}
            variant_scene_rows.append(row)
            scene_rows.append(row)
            scene_dir = output_root / variant / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                scene_dir / "objects.json",
                {
                    "scene": scene,
                    "variant": variant,
                    "objects": object_rows,
                    "unknown_tubes": unknown_tubes,
                    "method_manifest": {
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                        "mask_source": "O5_hybrid_first_class",
                        "uses_frozen_visual_backbone": bool(variant.startswith(("H16_", "H17_"))),
                    },
                },
            )
        aggregate = {
            "scene": "ALL",
            "variant": variant,
            "route": variant,
            "mask_source": "O5_hybrid_first_class",
            "selected_row_count": int(sum(int(row["selected_row_count"]) for row in variant_scene_rows)),
            "row_support_count": int(sum(int(row["row_support_count"]) for row in variant_scene_rows)),
            "edge_count": int(sum(int(row["edge_count"]) for row in variant_scene_rows)),
            "kept_edge_count": int(sum(int(row["kept_edge_count"]) for row in variant_scene_rows)),
            "labeled_tube_count": int(sum(int(row["labeled_tube_count"]) for row in variant_scene_rows)),
            "object_count": int(sum(int(row["object_count"]) for row in variant_scene_rows)),
            "unknown_tube_count": int(sum(int(row["unknown_tube_count"]) for row in variant_scene_rows)),
            "unknown_tube_ratio": _mean([row["unknown_tube_ratio"] for row in variant_scene_rows]),
            "local_ARI": _mean([row["local_ARI"] for row in variant_scene_rows]),
            "local_purity": _mean([row["local_purity"] for row in variant_scene_rows]),
            "local_completeness": _mean([row["local_completeness"] for row in variant_scene_rows]),
            "local_overmerge": _mean([row["local_overmerge"] for row in variant_scene_rows]),
            "local_oversplit": _mean([row["local_oversplit"] for row in variant_scene_rows]),
            "scene0081_local_ARI": next((row["local_ARI"] for row in variant_scene_rows if row["scene"] == "scene0081_01"), None),
            "is_method_result": True,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "uses_frozen_visual_backbone": bool(variant.startswith(("H16_", "H17_"))),
        }
        aggregate = {**aggregate, **_gate_status(aggregate)}
        summary_rows.append(aggregate)
    controls = _control_status(summary_rows)
    for row in summary_rows:
        row.update(controls.get(row["variant"], {}))
    payload = {
        "proposal_root": str(args.proposal_root),
        "proposal_label": str(args.proposal_label),
        "summary_rows": summary_rows,
        "scene_rows": scene_rows,
        "route": "RouteA_Hybrid_O5_first_class_repair",
        "visual_compactness_diag": visual_compactness_diag,
    }
    _write_csv(output_root / "hybrid_ownership_summary.csv", summary_rows)
    _write_csv(output_root / "hybrid_ownership_scene_rows.csv", scene_rows)
    _write_json(output_root / "hybrid_ownership_summary.json", payload)
    best = max(summary_rows, key=lambda row: _float(row.get("local_ARI"), -9.0) or -9.0)
    print(json.dumps(_json_safe({"output_root": str(output_root), "best_variant": best["variant"], "best_ARI": best["local_ARI"], "best_gate": best["local_gate_pass"]}), indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v35 O5 hybrid first-class ownership repair.")
    parser.add_argument("--proposal-root", default="outputs/audit/v35_mask_source_audit/proposal_rebuild_conda")
    parser.add_argument("--proposal-label", default="v35_mask_source_rebuild_conda")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-root", default="outputs/audit/v35_hybrid_ownership_repair")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--seed", type=int, default=3511)
    parser.add_argument("--variants", default="", help="Comma-separated variant subset; empty runs all variants.")
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2")
    parser.add_argument("--checkpoint", default="/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames-per-scene", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
