from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv, assign_gt_labels
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


def _parse_core_tube_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_core_tube_ids" in row:
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or []))
    text = str(row.get("core_tube_ids") or "")
    if not text:
        return ()
    return tuple(sorted(int(v) for v in text.split(";") if str(v).strip()))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _selection_manifest_policy_fields() -> dict[str, Any]:
    return {
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
        "geometry_field": "D4RT canonical proposal tube memberships plus image-space mask features",
        "coordinate_frame": "d4rt_canonical for tube geometry; image space for mask/proposal features",
        "alignment_source": "D4RT self-Sim3 inherited from proposal oracle manifest",
    }


def _is_o5_method_candidate(row: dict[str, Any]) -> bool:
    proposal_type = str(row.get("proposal_type") or "")
    return proposal_type.startswith(
        (
            "R0_",
            "R1_",
            "R2_",
            "R3_",
            "R4_",
            "R5_",
            "R6_",
            "R7_",
            "R8_temporal_tube_overlap_track_union",
            "R9_temporal_tube_overlap_track_consensus",
            "R10_temporal_tube_overlap_visible_negative_pruned",
            "R12_temporal_visible_negative_eroded_pruned",
        )
    )


def _proposal_score(
    row: dict[str, Any],
    *,
    cannot_link_weight: float = 0.20,
    visible_negative_weight: float = 0.14,
    boundary_weight: float = 0.20,
    small_proposal_bonus: float = 0.0,
) -> float:
    proposal_type = str(row.get("proposal_type") or "")
    type_bonus = 0.0
    if proposal_type.startswith("R10_"):
        type_bonus += 0.30
    elif proposal_type.startswith("R7_"):
        type_bonus += 0.22
    elif proposal_type.startswith("R1_"):
        type_bonus += 0.12
    elif proposal_type.startswith("R2_"):
        type_bonus += 0.08
    elif proposal_type.startswith("R3_"):
        type_bonus += 0.08
    elif proposal_type.startswith("R5_"):
        type_bonus += 0.05
    elif proposal_type.startswith("R4_"):
        type_bonus += 0.03
    elif proposal_type.startswith("R6_"):
        type_bonus += 0.03
    elif proposal_type.startswith("R8_"):
        type_bonus += 0.06
    elif proposal_type.startswith("R0_"):
        type_bonus -= 0.12
    core = max(_float(row, "num_core_tubes"), 1.0)
    size_score = min(np.log1p(core) / np.log(128.0), 1.0)
    return float(
        type_bonus
        + 0.35 * _float(row, "eroded_interior_ratio")
        + 0.18 * _float(row, "visibility_mean", 0.5)
        + 0.10 * _float(row, "confidence_mean", 0.5)
        + 0.10 * size_score
        + float(small_proposal_bonus) * (1.0 - size_score)
        - float(boundary_weight) * _float(row, "boundary_contact_ratio")
        - float(visible_negative_weight) * np.log1p(max(_float(row, "visible_outside_negative_rate"), 0.0))
        - float(cannot_link_weight) * np.log1p(max(_float(row, "same_frame_cannot_link_rate"), 0.0))
        - 0.05 * _float(row, "appearance_variance")
    )


def _rows_for_variant(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "P0_full_mask":
        return [row for row in rows if str(row.get("proposal_type")) == "R0_full_mask_region"]
    if variant == "P1_eroded_core":
        return [row for row in rows if str(row.get("proposal_type")) in {"R1_boundary_eroded_interior", "R7_high_purity_core_region"}]
    if variant == "P2_watershed":
        return [row for row in rows if str(row.get("proposal_type")) == "R2_distance_watershed_region"]
    if variant == "P3_d4rt_seeded":
        return [row for row in rows if str(row.get("proposal_type")) in {"R3_d4rt_tube_seeded_voronoi", "R5_d4rt_canonical_adjacency_split"}]
    if variant in {
        "P4_greedy_set_packing",
        "P5_local_search",
        "P6_competition_unknown",
        "P8_shuffled_membership_control",
        "P11_calibrated_ownership_expansion",
    }:
        return [row for row in rows if _is_o5_method_candidate(row)]
    if variant == "P9_no_temporal_control":
        return [
            row
            for row in rows
            if _is_o5_method_candidate(row)
            and not str(row.get("proposal_type")).startswith(
                (
                    "R8_temporal_",
                    "R9_temporal_",
                    "R10_temporal_",
                    "R12_temporal_",
                )
            )
        ]
    if variant == "P10_mask_only_control":
        return [
            row
            for row in rows
            if str(row.get("proposal_type"))
            in {
                "R0_full_mask_region",
                "R1_boundary_eroded_interior",
                "R2_distance_watershed_region",
                "R4_image_gradient_split",
                "R6_mask_overlap_consensus_union",
            }
        ]
    raise ValueError(f"unknown selection variant: {variant}")


def _stable_scene_seed(seed: int, scene: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _shuffle_candidate_memberships(rows: list[dict[str, Any]], *, seed: int, scene: str) -> list[dict[str, Any]]:
    tube_ids = sorted({tid for row in rows for tid in _parse_core_tube_ids(row)})
    if not tube_ids:
        return []
    shuffled = list(tube_ids)
    rng = np.random.default_rng(_stable_scene_seed(seed, scene))
    rng.shuffle(shuffled)
    remap = dict(zip(tube_ids, shuffled))
    out: list[dict[str, Any]] = []
    for row in rows:
        core = tuple(sorted(remap[int(tid)] for tid in _parse_core_tube_ids(row)))
        item = dict(row)
        item["_core_tube_ids"] = core
        item["core_tube_ids"] = ";".join(str(tid) for tid in core)
        item["proposal_id"] = f"{row.get('proposal_id')}_p8shuf{int(seed)}"
        item["control_kind"] = "deterministic_tube_membership_shuffle_proxy"
        out.append(item)
    return out


def _select_greedy(
    rows: list[dict[str, Any]],
    *,
    min_new_tubes: int,
    max_overlap_ratio: float,
    min_score: float,
    score_kwargs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    score_kwargs = score_kwargs or {}
    selected: list[dict[str, Any]] = []
    owned: set[int] = set()
    ranked = sorted(rows, key=lambda row: (_proposal_score(row, **score_kwargs), len(_parse_core_tube_ids(row))), reverse=True)
    for row in ranked:
        core = set(_parse_core_tube_ids(row))
        if len(core) < int(min_new_tubes):
            continue
        score = _proposal_score(row, **score_kwargs)
        if score < float(min_score):
            continue
        new = core - owned
        overlap_ratio = float((len(core) - len(new)) / max(len(core), 1))
        if len(new) < int(min_new_tubes) or overlap_ratio > float(max_overlap_ratio):
            continue
        selected.append(row)
        owned.update(core)
    return selected


def _select_local_search(
    rows: list[dict[str, Any]],
    *,
    min_new_tubes: int,
    max_overlap_ratio: float,
    min_score: float,
    score_kwargs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    score_kwargs = score_kwargs or {}
    selected = _select_greedy(
        rows,
        min_new_tubes=min_new_tubes,
        max_overlap_ratio=max_overlap_ratio,
        min_score=min_score,
        score_kwargs=score_kwargs,
    )
    selected_ids = {id(row) for row in selected}
    ranked = sorted(rows, key=lambda row: (_proposal_score(row, **score_kwargs), len(_parse_core_tube_ids(row))), reverse=True)
    for row in ranked:
        if id(row) in selected_ids:
            continue
        core = set(_parse_core_tube_ids(row))
        if len(core) < int(min_new_tubes):
            continue
        row_score = _proposal_score(row, **score_kwargs)
        if row_score < float(min_score):
            continue
        conflicts: list[dict[str, Any]] = []
        for item in selected:
            other = set(_parse_core_tube_ids(item))
            overlap = len(core & other) / max(min(len(core), len(other)), 1)
            if overlap > float(max_overlap_ratio):
                conflicts.append(item)
        if not conflicts:
            continue
        conflict_score = sum(_proposal_score(item, **score_kwargs) for item in conflicts)
        conflict_core = {tid for item in conflicts for tid in _parse_core_tube_ids(item)}
        retained_core = {tid for item in selected if item not in conflicts for tid in _parse_core_tube_ids(item)}
        new_gain = len(core - retained_core)
        lost_gain = len(conflict_core - core)
        if new_gain < int(min_new_tubes):
            continue
        if row_score + 0.02 * new_gain <= conflict_score + 0.02 * lost_gain:
            continue
        selected = [item for item in selected if item not in conflicts]
        selected.append(row)
        selected_ids = {id(item) for item in selected}
    return sorted(selected, key=lambda row: (_proposal_score(row, **score_kwargs), len(_parse_core_tube_ids(row))), reverse=True)


def _select_calibrated_ownership_expansion(
    rows: list[dict[str, Any]],
    *,
    min_new_tubes: int,
    seed_max_overlap_ratio: float,
    seed_min_score: float,
    expand_min_score: float,
    expand_min_overlap_ratio: float,
    expand_min_votes: int,
    expand_margin: float,
    max_expanded_core_ratio: float,
    score_kwargs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    score_kwargs = score_kwargs or {}
    seeds = _select_greedy(
        rows,
        min_new_tubes=min_new_tubes,
        max_overlap_ratio=seed_max_overlap_ratio,
        min_score=seed_min_score,
        score_kwargs=score_kwargs,
    )
    if not seeds:
        return []
    seed_cores = [set(_parse_core_tube_ids(row)) for row in seeds]
    owned = {tid for core in seed_cores for tid in core}
    votes: dict[int, Counter[int]] = {}
    ranked = sorted(rows, key=lambda row: (_proposal_score(row, **score_kwargs), len(_parse_core_tube_ids(row))), reverse=True)
    for row in ranked:
        score = _proposal_score(row, **score_kwargs)
        if score < float(expand_min_score):
            continue
        core = set(_parse_core_tube_ids(row))
        if len(core) < int(min_new_tubes):
            continue
        overlaps: list[tuple[float, int]] = []
        for idx, seed_core in enumerate(seed_cores):
            shared = len(core & seed_core)
            if shared <= 0:
                continue
            ratio = shared / max(min(len(core), len(seed_core)), 1)
            if ratio >= float(expand_min_overlap_ratio):
                overlaps.append((float(ratio), idx))
        if not overlaps:
            continue
        overlaps.sort(reverse=True)
        best_ratio, best_idx = overlaps[0]
        second_ratio = overlaps[1][0] if len(overlaps) > 1 else 0.0
        if second_ratio > 0.0 and best_ratio < second_ratio * float(expand_margin):
            continue
        weight = max(score - float(expand_min_score), 0.01) * best_ratio
        for tid in core - owned:
            votes.setdefault(int(tid), Counter())[int(best_idx)] += int(round(weight * 1000.0))

    additions: list[set[int]] = [set() for _ in seeds]
    for tid, counter in votes.items():
        if not counter:
            continue
        (best_idx, best_vote), *rest = counter.most_common(2)
        second_vote = rest[0][1] if rest else 0
        if best_vote < int(expand_min_votes):
            continue
        if second_vote > 0 and best_vote < second_vote * float(expand_margin):
            continue
        max_size = max(int(np.ceil(len(seed_cores[best_idx]) * float(max_expanded_core_ratio))), len(seed_cores[best_idx]))
        if len(seed_cores[best_idx]) + len(additions[best_idx]) >= max_size:
            continue
        additions[best_idx].add(int(tid))

    expanded: list[dict[str, Any]] = []
    for row, seed_core, extra in zip(seeds, seed_cores, additions):
        core = tuple(sorted(seed_core | extra))
        item = dict(row)
        item["_core_tube_ids"] = core
        item["core_tube_ids"] = ";".join(str(tid) for tid in core)
        item["num_core_tubes"] = int(len(core))
        item["proposal_id"] = f"{row.get('proposal_id')}_p11own"
        item["selection_transform"] = "calibrated_ownership_expansion"
        item["ownership_added_tube_count"] = int(len(extra))
        expanded.append(item)
    return sorted(expanded, key=lambda row: (_proposal_score(row, **score_kwargs), len(_parse_core_tube_ids(row))), reverse=True)


def _labels_from_selected(selected: list[dict[str, Any]], labeled_tubes: list[int]) -> tuple[dict[int, int], int]:
    labels_pred: dict[int, int] = {}
    for idx, row in enumerate(selected):
        for tid in _parse_core_tube_ids(row):
            tid = int(tid)
            if tid not in labels_pred:
                labels_pred[tid] = idx
    next_label = len(selected)
    unknown_count = 0
    for tid in labeled_tubes:
        if int(tid) not in labels_pred:
            labels_pred[int(tid)] = next_label
            next_label += 1
            unknown_count += 1
    return labels_pred, unknown_count


def _load_gt_labels(cache_root: Path, scene: str, max_tubes_per_window: int, image_width: int, image_height: int) -> dict[int, int]:
    chunks, _ = load_scene_chunks_from_cache(
        cache_root / scene,
        max_tubes_per_window=max_tubes_per_window,
        image_width=image_width,
        image_height=image_height,
    )
    builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
    records = chunks_to_records(builder.stitch_to_canonical(chunks))
    return assign_gt_labels(records, stream=ScanNetStream(seq_name=scene), min_visibility=0.5, min_confidence=0.5)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    proposal_root = Path(args.proposal_root)
    proposal_label = str(args.proposal_label)
    proposal_rows = json.loads((proposal_root / f"{proposal_label}_proposal_rows.json").read_text(encoding="utf-8"))
    proposal_manifest_path = proposal_root / f"{proposal_label}_manifest.json"
    proposal_manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8")) if proposal_manifest_path.exists() else {}
    proposal_root_is_shuffle = bool(proposal_manifest.get("shuffle_d4rt_control", False))
    scenes = _read_split(Path(args.split))
    rows_by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in proposal_rows:
        scene = str(row.get("scene"))
        if scene in rows_by_scene:
            rows_by_scene[scene].append(row)

    variants = [
        "P0_full_mask",
        "P1_eroded_core",
        "P2_watershed",
        "P3_d4rt_seeded",
        "P4_greedy_set_packing",
        "P5_local_search",
        "P6_competition_unknown",
        "P8_shuffled_membership_control",
        "P9_no_temporal_control",
        "P10_mask_only_control",
        "P11_calibrated_ownership_expansion",
    ]
    score_kwargs = {
        "cannot_link_weight": float(args.cannot_link_weight),
        "visible_negative_weight": float(args.visible_negative_weight),
        "boundary_weight": float(args.boundary_weight),
        "small_proposal_bonus": float(args.small_proposal_bonus),
    }
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            continue
        gt_labels = _load_gt_labels(Path(args.cache_root), scene, int(args.max_tubes_per_window), int(args.image_width), int(args.image_height))
        labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
        scene_rows = rows_by_scene.get(scene, [])
        for variant in variants:
            candidates = _rows_for_variant(scene_rows, variant)
            if variant == "P8_shuffled_membership_control":
                candidates = _shuffle_candidate_memberships(candidates, seed=int(args.shuffle_control_seed), scene=scene)
            if variant in {"P0_full_mask", "P1_eroded_core", "P2_watershed", "P3_d4rt_seeded", "P9_no_temporal_control", "P10_mask_only_control"}:
                selected = _select_greedy(
                    candidates,
                    min_new_tubes=3,
                    max_overlap_ratio=0.05 if variant in {"P0_full_mask", "P1_eroded_core"} else float(args.control_max_overlap_ratio),
                    min_score=-999.0 if variant in {"P0_full_mask", "P1_eroded_core", "P2_watershed", "P3_d4rt_seeded"} else float(args.min_score),
                    score_kwargs=score_kwargs,
                )
            elif variant == "P5_local_search":
                selected = _select_local_search(
                    candidates,
                    min_new_tubes=int(args.min_new_tubes),
                    max_overlap_ratio=float(args.max_overlap_ratio),
                    min_score=float(args.min_score),
                    score_kwargs=score_kwargs,
                )
            elif variant == "P11_calibrated_ownership_expansion":
                selected = _select_calibrated_ownership_expansion(
                    candidates,
                    min_new_tubes=int(args.min_new_tubes),
                    seed_max_overlap_ratio=float(args.ownership_seed_max_overlap_ratio),
                    seed_min_score=float(args.ownership_seed_min_score),
                    expand_min_score=float(args.ownership_expand_min_score),
                    expand_min_overlap_ratio=float(args.ownership_expand_min_overlap_ratio),
                    expand_min_votes=int(args.ownership_expand_min_votes),
                    expand_margin=float(args.ownership_expand_margin),
                    max_expanded_core_ratio=float(args.ownership_max_expanded_core_ratio),
                    score_kwargs=score_kwargs,
                )
            else:
                selected = _select_greedy(
                    candidates,
                    min_new_tubes=int(args.min_new_tubes),
                    max_overlap_ratio=float(args.max_overlap_ratio),
                    min_score=float(args.min_score),
                    score_kwargs=score_kwargs,
                )
            labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes)
            metrics = _cluster_metrics(labels_pred, gt_labels)
            selected_tube_count = len({tid for row in selected for tid in _parse_core_tube_ids(row)})
            row = {
                "scene": scene,
                "variant": variant,
                "candidate_proposal_count": int(len(candidates)),
                "selected_proposal_count": int(len(selected)),
                "selected_tube_count": int(selected_tube_count),
                "labeled_tube_count": int(len(labeled_tubes)),
                "unknown_tube_count": int(unknown_count),
                "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
                "local_ARI": metrics["ari"],
                "local_purity": metrics["purity"],
                "local_completeness": metrics["completeness"],
                "local_overmerge": metrics["overmerge"],
                "local_oversplit": metrics["oversplit"],
                "is_method_result": variant != "P7_oracle",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
            summary_rows.append(row)
            for rank, item in enumerate(selected):
                selected_rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "rank": int(rank),
                        "proposal_id": item.get("proposal_id"),
                        "proposal_type": item.get("proposal_type"),
                        "control_kind": item.get("control_kind", ""),
                        "selection_transform": item.get("selection_transform", ""),
                        "ownership_added_tube_count": int(item.get("ownership_added_tube_count") or 0),
                        "score": _proposal_score(item, **score_kwargs),
                        "num_core_tubes": int(item.get("num_core_tubes") or 0),
                        "same_frame_cannot_link_rate": _float(item, "same_frame_cannot_link_rate"),
                        "visible_outside_negative_rate": _float(item, "visible_outside_negative_rate"),
                        "is_method_result": True,
                        "uses_gt_for_prediction": False,
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for variant in variants:
        items = [row for row in summary_rows if row["variant"] == variant]
        if not items:
            continue
        aggregate_rows.append(
            {
                "scene": "ALL",
                "variant": variant,
                "candidate_proposal_count": int(sum(int(row["candidate_proposal_count"]) for row in items)),
                "selected_proposal_count": int(sum(int(row["selected_proposal_count"]) for row in items)),
                "selected_tube_count": int(sum(int(row["selected_tube_count"]) for row in items)),
                "labeled_tube_count": int(sum(int(row["labeled_tube_count"]) for row in items)),
                "unknown_tube_count": int(sum(int(row["unknown_tube_count"]) for row in items)),
                "unknown_tube_ratio": float(np.mean([float(row["unknown_tube_ratio"]) for row in items])),
                "local_ARI": float(np.mean([float(row["local_ARI"]) for row in items if row["local_ARI"] is not None])),
                "local_purity": float(np.mean([float(row["local_purity"]) for row in items])),
                "local_completeness": float(np.mean([float(row["local_completeness"]) for row in items])),
                "local_overmerge": float(np.mean([float(row["local_overmerge"]) for row in items])),
                "local_oversplit": float(np.mean([float(row["local_oversplit"]) for row in items])),
                "scene0081_local_ARI": next((row["local_ARI"] for row in items if row["scene"] == "scene0081_01"), None),
                "is_method_result": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    label = str(args.label)
    _write_csv(output_root / f"{label}_selection_summary.csv", summary_rows + aggregate_rows)
    (output_root / f"{label}_selection_summary.json").write_text(
        json.dumps(_json_safe(summary_rows + aggregate_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_root / f"{label}_selected_proposals.csv", selected_rows)
    manifest = {
        **_selection_manifest_policy_fields(),
        "proposal_root": str(proposal_root),
        "proposal_label": proposal_label,
        "proposal_manifest_shuffle_d4rt_control": bool(proposal_manifest.get("shuffle_d4rt_control", False)),
        "proposal_manifest_shuffle_d4rt_seed": proposal_manifest.get("shuffle_d4rt_seed"),
        "proposal_manifest_path": str(proposal_manifest_path) if proposal_manifest_path.exists() else "",
        "split": str(args.split),
        "cache_root": str(args.cache_root),
        "variant_count": int(len(variants)),
        "score_kwargs": dict(score_kwargs),
        "control_max_overlap_ratio": float(args.control_max_overlap_ratio),
        "shuffle_control_seed": int(args.shuffle_control_seed),
        "ownership_solver": {
            "variant": "P11_calibrated_ownership_expansion",
            "seed_max_overlap_ratio": float(args.ownership_seed_max_overlap_ratio),
            "seed_min_score": float(args.ownership_seed_min_score),
            "expand_min_score": float(args.ownership_expand_min_score),
            "expand_min_overlap_ratio": float(args.ownership_expand_min_overlap_ratio),
            "expand_min_votes": int(args.ownership_expand_min_votes),
            "expand_margin": float(args.ownership_expand_margin),
            "max_expanded_core_ratio": float(args.ownership_max_expanded_core_ratio),
        },
        "p8_control_kind": (
            "full_shuffled_d4rt_proposal_pool"
            if proposal_root_is_shuffle
            else "deterministic_tube_membership_shuffle_proxy"
        ),
        "p8_full_shuffled_d4rt_proposal_generation_control": proposal_root_is_shuffle,
        "plan_variant_limitations": [
            "P7 remains oracle-only diagnostic in proposal oracle artifacts.",
        ]
        + (
            []
            if proposal_root_is_shuffle
            else [
                "P8 here is a deterministic tube-membership shuffle proxy over the final proposal rows, not a fully regenerated shuffled-D4RT proposal pool.",
            ]
        ),
        "missing_plan_variants": ["P7_oracle_proposal_selection_diagnostic"],
        "missing_plan_variant_reason": "P7 remains oracle-only diagnostic in proposal oracle artifacts.",
        "phase_d_selection_attempt_complete": True,
        "phase_d_plan_variants_complete": False,
        "phase_d_complete": False,
    }
    (output_root / f"{label}_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "manifest": manifest,
        "selection_all": aggregate_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v28 non-GT proposal selection and ownership diagnostics.")
    parser.add_argument("--proposal-root", required=True)
    parser.add_argument("--proposal-label", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-root", default="outputs/audit/v28_proposal_selection")
    parser.add_argument("--label", default="v28_proposal_selection")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-new-tubes", type=int, default=3)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.35)
    parser.add_argument("--control-max-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--min-score", type=float, default=-0.45)
    parser.add_argument("--cannot-link-weight", type=float, default=0.20)
    parser.add_argument("--visible-negative-weight", type=float, default=0.14)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    parser.add_argument("--small-proposal-bonus", type=float, default=0.0)
    parser.add_argument("--shuffle-control-seed", type=int, default=2808)
    parser.add_argument("--ownership-seed-max-overlap-ratio", type=float, default=0.10)
    parser.add_argument("--ownership-seed-min-score", type=float, default=0.20)
    parser.add_argument("--ownership-expand-min-score", type=float, default=-0.05)
    parser.add_argument("--ownership-expand-min-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--ownership-expand-min-votes", type=int, default=20)
    parser.add_argument("--ownership-expand-margin", type=float, default=1.25)
    parser.add_argument("--ownership-max-expanded-core-ratio", type=float, default=1.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
