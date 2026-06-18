from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_selection import _load_gt_labels, _rows_for_variant, _select_calibrated_ownership_expansion
from tools.run_v29_constrained_ownership_solver import (
    MASK_ONLY_TYPES,
    TEMPORAL_PREFIXES,
    _annotate_features,
    _core_ids,
    _dedupe_rows,
    _eval_selected,
    _float,
    _generate_medium,
    _is_o5,
    _quality,
    _set_core_ids,
)


SMALL_REPLACEMENT_PREFIXES = ("R1_", "R2_", "R3_", "R4_", "R5_", "R7_", "R12_", "R13_")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_rows(root: Path, label: str, scenes: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows = json.loads((root / f"{label}_proposal_rows.json").read_text(encoding="utf-8"))
    by_scene = {scene: [] for scene in scenes}
    for row in rows:
        scene = str(row.get("scene"))
        if scene in by_scene and _is_o5(row):
            _set_core_ids(row, _core_ids(row))
            by_scene[scene].append(row)
    return by_scene


def _filter_candidates(rows: list[dict[str, Any]], control_kind: str) -> list[dict[str, Any]]:
    if control_kind == "real" or control_kind == "shuffled_d4rt":
        return [row for row in rows if _is_o5(row)]
    if control_kind == "no_temporal":
        return [row for row in rows if _is_o5(row) and not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
    if control_kind == "mask_only":
        return [row for row in rows if str(row.get("proposal_type")) in MASK_ONLY_TYPES]
    raise ValueError(control_kind)


def _p11_select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_kwargs = {
        "cannot_link_weight": 0.20,
        "visible_negative_weight": 0.14,
        "boundary_weight": 0.20,
        "small_proposal_bonus": 0.0,
    }
    return _select_calibrated_ownership_expansion(
        candidates,
        min_new_tubes=3,
        seed_max_overlap_ratio=0.10,
        seed_min_score=0.20,
        expand_min_score=-0.05,
        expand_min_overlap_ratio=0.25,
        expand_min_votes=20,
        expand_margin=1.25,
        max_expanded_core_ratio=1.25,
        score_kwargs=score_kwargs,
    )


def _risk_score(row: dict[str, Any]) -> float:
    feats = row.get("_v29_features") or {}
    proposal_type = str(row.get("proposal_type", ""))
    n_core = len(_core_ids(row))
    added = float(row.get("ownership_added_tube_count") or 0.0)
    medium_prior = float(feats.get("medium_prior_score", 0.0))
    area_norm_cannot = float(feats.get("area_normalized_cannot_link_rate", 0.0))
    return float(
        0.70 * (1.0 - medium_prior)
        + 0.15 * min(max(area_norm_cannot, 0.0) / 5.0, 1.0)
        + 0.25 * float(proposal_type.startswith("R10_") and n_core > 120)
        + 0.10 * added / max(n_core, 1)
    )


def _is_risky(row: dict[str, Any], *, risk_threshold: float) -> bool:
    feats = row.get("_v29_features") or {}
    proposal_type = str(row.get("proposal_type", ""))
    n_core = len(_core_ids(row))
    medium_prior = float(feats.get("medium_prior_score", 0.0))
    return bool(
        _risk_score(row) >= float(risk_threshold)
        or (proposal_type.startswith("R10_") and (medium_prior < 0.20 or n_core > 220))
        or (proposal_type.startswith("R7_") and float(feats.get("area_normalized_cannot_link_rate", 0.0)) > 8.0)
    )


def _base_id(row: dict[str, Any]) -> str:
    pid = str(row.get("proposal_id", ""))
    return pid[:-7] if pid.endswith("_p11own") else pid


def _replacement_pool(rows: list[dict[str, Any]], medium_rows: list[dict[str, Any]], *, control_kind: str) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for row in rows + ([] if control_kind in {"shuffled_d4rt", "mask_only"} else medium_rows):
        proposal_type = str(row.get("proposal_type", ""))
        if not proposal_type.startswith(SMALL_REPLACEMENT_PREFIXES):
            continue
        if control_kind == "no_temporal" and proposal_type.startswith(TEMPORAL_PREFIXES):
            continue
        if control_kind == "mask_only" and proposal_type not in MASK_ONLY_TYPES:
            continue
        if len(_core_ids(row)) >= 3:
            pool.append(row)
    return sorted(pool, key=lambda item: (_quality(item, medium_bonus=0.35), len(_core_ids(item))), reverse=True)


def _resolve_overlaps(rows: list[dict[str, Any]], *, min_new_ratio: float) -> list[dict[str, Any]]:
    final: list[dict[str, Any]] = []
    owned: set[int] = set()
    for row in sorted(rows, key=lambda item: (_quality(item, medium_bonus=0.35), len(_core_ids(item))), reverse=True):
        core = set(_core_ids(row))
        if not core:
            continue
        if len(core - owned) / max(len(core), 1) < float(min_new_ratio):
            continue
        final.append(row)
        owned.update(core)
    return final


def _replace_with_children(
    selected: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    mode: str,
    risk_threshold: float,
    quality_threshold: float,
    overlap_need: float,
    cover_need: float,
    max_children: int,
    duplicate_threshold: float,
    fallback: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {"risky_seen": 0, "replaced": 0, "reverted": 0, "dropped": 0, "kept": 0}
    for row in selected:
        if not _is_risky(row, risk_threshold=risk_threshold):
            out.append(row)
            continue
        stats["risky_seen"] += 1
        core = set(_core_ids(row))
        chosen: list[dict[str, Any]] = []
        owned: set[int] = set()
        for candidate in candidates:
            cand_core = set(_core_ids(candidate))
            if not cand_core:
                continue
            if len(cand_core & core) / max(len(cand_core), 1) < float(overlap_need):
                continue
            if _quality(candidate, medium_bonus=0.35) < float(quality_threshold):
                continue
            if any(
                len(cand_core & set(_core_ids(other))) / max(min(len(cand_core), len(_core_ids(other))), 1) > float(duplicate_threshold)
                for other in chosen
            ):
                continue
            if len(cand_core - owned) < 3:
                continue
            chosen.append(candidate)
            owned.update(cand_core)
            if len(chosen) >= int(max_children):
                break
        if chosen and len(owned & core) / max(len(core), 1) >= float(cover_need):
            out.extend(chosen)
            stats["replaced"] += 1
            continue
        if fallback == "revert":
            base = id_map.get(_base_id(row))
            out.append(base if base is not None else row)
            stats["reverted"] += 1
        elif fallback == "drop":
            stats["dropped"] += 1
        else:
            out.append(row)
            stats["kept"] += 1
    return _resolve_overlaps(out, min_new_ratio=0.35 if mode != "keep_order" else 0.0), stats


def _make_partition_row(
    parent: dict[str, Any],
    *,
    ids: set[int],
    index: int,
    kind: str,
    seed: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "proposal_id": f"{parent.get('proposal_id')}_v29tube_{index:02d}_{kind}",
        "scene": parent.get("scene"),
        "frame_id": parent.get("frame_id"),
        "mask_id": parent.get("mask_id"),
        "proposal_type": f"R14_v29_tube_partition_{kind}",
        "v29_partition_parent_id": parent.get("proposal_id"),
        "v29_partition_seed_id": seed.get("proposal_id") if seed is not None else "",
        "v29_partition_kind": kind,
        "is_diagnostic_only": False,
        "uses_gt_for_prediction": False,
    }
    for key in (
        "appearance_variance",
        "boundary_contact_ratio",
        "boundary_risk",
        "confidence_mean",
        "eroded_interior_ratio",
        "image_gradient_boundary_score",
        "mask_area",
        "mask_distance_mean",
        "mask_temporal_repeat_score",
        "overlap_with_other_proposals",
        "proposal_area",
        "proposal_area_ratio",
        "region_area",
        "same_frame_cannot_link_rate",
        "tube_canonical_compactness",
        "tube_density",
        "tube_temporal_length_mean",
        "tube_xy_compactness",
        "visibility_mean",
        "visible_outside_negative_rate",
    ):
        row[key] = (seed or parent).get(key, parent.get(key, 0.0))
    row["_v29_features"] = dict((seed or parent).get("_v29_features") or parent.get("_v29_features") or {})
    _set_core_ids(row, tuple(sorted(ids)))
    return row


def _partition_parent(
    parent: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    quality_threshold: float,
    child_containment: float,
    max_parts: int,
    min_part_tubes: int,
    min_covered_ratio: float,
    keep_residual: bool,
    residual_max_ratio: float,
) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    parent_core = set(_core_ids(parent))
    if len(parent_core) < max(int(min_part_tubes) * 2, 8):
        return None, {"partition_parts": 0, "residual_parts": 0}
    scored: list[tuple[float, dict[str, Any], set[int]]] = []
    parent_id = str(parent.get("proposal_id"))
    parent_base = _base_id(parent)
    for candidate in candidates:
        cand_id = str(candidate.get("proposal_id"))
        if cand_id in {parent_id, parent_base}:
            continue
        cand_core = set(_core_ids(candidate))
        if not cand_core:
            continue
        inter = cand_core & parent_core
        if len(inter) < int(min_part_tubes):
            continue
        if len(inter) / max(len(cand_core), 1) < float(child_containment):
            continue
        if len(inter) / max(len(parent_core), 1) > 0.82:
            continue
        quality = _quality(candidate, medium_bonus=0.35)
        if quality < float(quality_threshold):
            continue
        score = float(quality + 0.05 * np.log1p(len(inter)) - 0.08 * len(inter) / max(len(parent_core), 1))
        scored.append((score, candidate, inter))
    chosen: list[tuple[dict[str, Any], set[int]]] = []
    owned: set[int] = set()
    for _, candidate, inter in sorted(scored, key=lambda item: item[0], reverse=True):
        part = inter - owned
        if len(part) < int(min_part_tubes):
            continue
        if any(len(part & other) / max(min(len(part), len(other)), 1) > 0.50 for _, other in chosen):
            continue
        chosen.append((candidate, set(part)))
        owned.update(part)
        if len(chosen) >= int(max_parts):
            break
    covered_ratio = len(owned) / max(len(parent_core), 1)
    if len(chosen) < 2 or covered_ratio < float(min_covered_ratio):
        return None, {"partition_parts": len(chosen), "residual_parts": 0}
    out: list[dict[str, Any]] = []
    for index, (seed, ids) in enumerate(chosen):
        out.append(_make_partition_row(parent, ids=ids, index=index, kind="seed", seed=seed))
    residual = parent_core - owned
    residual_parts = 0
    if residual and keep_residual and len(residual) / max(len(parent_core), 1) <= float(residual_max_ratio):
        out.append(_make_partition_row(parent, ids=residual, index=len(out), kind="residual", seed=None))
        residual_parts = 1
    elif residual and not keep_residual:
        pass
    elif residual:
        return None, {"partition_parts": len(chosen), "residual_parts": 0}
    return out, {"partition_parts": len(out), "residual_parts": residual_parts}


def _split_with_tube_partitions(
    selected: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    risk_threshold: float,
    quality_threshold: float,
    child_containment: float,
    max_parts: int,
    min_part_tubes: int,
    min_covered_ratio: float,
    keep_residual: bool,
    residual_max_ratio: float,
    fallback: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {
        "risky_seen": 0,
        "replaced": 0,
        "reverted": 0,
        "dropped": 0,
        "kept": 0,
        "split_parents": 0,
        "partition_parts": 0,
        "residual_parts": 0,
    }
    for row in selected:
        if not _is_risky(row, risk_threshold=risk_threshold):
            out.append(row)
            stats["kept"] += 1
            continue
        stats["risky_seen"] += 1
        parts, part_stats = _partition_parent(
            row,
            candidates,
            quality_threshold=quality_threshold,
            child_containment=child_containment,
            max_parts=max_parts,
            min_part_tubes=min_part_tubes,
            min_covered_ratio=min_covered_ratio,
            keep_residual=keep_residual,
            residual_max_ratio=residual_max_ratio,
        )
        if parts:
            out.extend(parts)
            stats["replaced"] += 1
            stats["split_parents"] += 1
            stats["partition_parts"] += int(part_stats["partition_parts"])
            stats["residual_parts"] += int(part_stats["residual_parts"])
            continue
        if fallback == "revert":
            base = id_map.get(_base_id(row))
            out.append(base if base is not None else row)
            stats["reverted"] += 1
        elif fallback == "drop":
            stats["dropped"] += 1
        else:
            out.append(row)
            stats["kept"] += 1
    return _resolve_overlaps(out, min_new_ratio=0.20), stats


def _variant_selected(
    variant: str,
    *,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if variant == "C0_p11_reconstructed":
        return list(selected), {"risky_seen": 0, "replaced": 0, "reverted": 0, "dropped": 0, "kept": 0, "split_parents": 0, "partition_parts": 0, "residual_parts": 0}
    if variant == "C1_drop_risky_strict":
        out = [row for row in selected if not _is_risky(row, risk_threshold=0.92)]
        return out, {"risky_seen": len(selected) - len(out), "replaced": 0, "reverted": 0, "dropped": len(selected) - len(out), "kept": len(out)}
    if variant == "C2_seed_revert_risky":
        out = []
        stats = {"risky_seen": 0, "replaced": 0, "reverted": 0, "dropped": 0, "kept": 0}
        for row in selected:
            if _is_risky(row, risk_threshold=0.92):
                stats["risky_seen"] += 1
                base = id_map.get(_base_id(row))
                out.append(base if base is not None else row)
                stats["reverted"] += 1
            else:
                out.append(row)
                stats["kept"] += 1
        return out, stats
    if variant == "C3_child_replace_conservative":
        return _replace_with_children(
            selected,
            candidates=candidates,
            id_map=id_map,
            mode="resolve",
            risk_threshold=0.92,
            quality_threshold=0.15,
            overlap_need=0.80,
            cover_need=0.30,
            max_children=6,
            duplicate_threshold=0.45,
            fallback="keep",
        )
    if variant == "C4_child_replace_aggressive":
        return _replace_with_children(
            selected,
            candidates=candidates,
            id_map=id_map,
            mode="resolve",
            risk_threshold=0.82,
            quality_threshold=0.00,
            overlap_need=0.60,
            cover_need=0.15,
            max_children=10,
            duplicate_threshold=0.65,
            fallback="drop",
        )
    if variant == "C5_child_replace_or_seed_revert":
        return _replace_with_children(
            selected,
            candidates=candidates,
            id_map=id_map,
            mode="resolve",
            risk_threshold=0.82,
            quality_threshold=0.08,
            overlap_need=0.70,
            cover_need=0.25,
            max_children=8,
            duplicate_threshold=0.55,
            fallback="revert",
        )
    if variant == "C6_tube_partition_cover":
        return _split_with_tube_partitions(
            selected,
            candidates=candidates,
            id_map=id_map,
            risk_threshold=0.82,
            quality_threshold=0.08,
            child_containment=0.55,
            max_parts=10,
            min_part_tubes=18,
            min_covered_ratio=0.20,
            keep_residual=True,
            residual_max_ratio=0.82,
            fallback="keep",
        )
    if variant == "C7_tube_partition_seed_revert":
        return _split_with_tube_partitions(
            selected,
            candidates=candidates,
            id_map=id_map,
            risk_threshold=0.82,
            quality_threshold=0.12,
            child_containment=0.65,
            max_parts=8,
            min_part_tubes=20,
            min_covered_ratio=0.28,
            keep_residual=True,
            residual_max_ratio=0.65,
            fallback="revert",
        )
    raise ValueError(variant)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["solver"]), str(row["control_kind"]))].append(row)
    out: list[dict[str, Any]] = []
    for (solver, control_kind), items in sorted(by_key.items()):
        out.append(
            {
                "scene": "ALL",
                "solver": solver,
                "control_kind": control_kind,
                "selected_proposal_count": int(sum(_as_int(row.get("selected_proposal_count")) for row in items)),
                "risky_seen": int(sum(_as_int(row.get("risky_seen")) for row in items)),
                "replaced": int(sum(_as_int(row.get("replaced")) for row in items)),
                "reverted": int(sum(_as_int(row.get("reverted")) for row in items)),
                "dropped": int(sum(_as_int(row.get("dropped")) for row in items)),
                "split_parents": int(sum(_as_int(row.get("split_parents")) for row in items)),
                "partition_parts": int(sum(_as_int(row.get("partition_parts")) for row in items)),
                "residual_parts": int(sum(_as_int(row.get("residual_parts")) for row in items)),
                "ARI": _mean([_float(row, "ARI") for row in items]),
                "purity": _mean([_float(row, "purity") for row in items]),
                "completeness": _mean([_float(row, "completeness") for row in items]),
                "unknown_tube_ratio": _mean([_float(row, "unknown_tube_ratio") for row in items]),
                "scene0081_ARI": next((row.get("ARI") for row in items if row.get("scene") == "scene0081_01"), None),
            }
        )
    return out


def _control_rows(
    *,
    rows_by_scene: dict[str, list[dict[str, Any]]],
    medium_by_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
    scenes: list[str],
    control_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = [
        "C0_p11_reconstructed",
        "C1_drop_risky_strict",
        "C2_seed_revert_risky",
        "C3_child_replace_conservative",
        "C4_child_replace_aggressive",
        "C5_child_replace_or_seed_revert",
        "C6_tube_partition_cover",
        "C7_tube_partition_seed_revert",
    ]
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for scene in scenes:
        rows = _filter_candidates(rows_by_scene[scene], control_kind)
        _annotate_features(rows + medium_by_scene.get(scene, []), set(), set())
        id_map = {str(row.get("proposal_id")): row for row in rows}
        selected = _p11_select(_rows_for_variant(rows, "P11_calibrated_ownership_expansion"))
        replacement_candidates = _replacement_pool(rows, medium_by_scene.get(scene, []), control_kind=control_kind)
        for variant in variants:
            chosen, stats = _variant_selected(variant, selected=selected, candidates=replacement_candidates, id_map=id_map)
            metrics = _eval_selected(chosen, gt_by_scene[scene])
            row = {
                "scene": scene,
                "solver": variant,
                "control_kind": control_kind,
                "selected_proposal_count": int(len(chosen)),
                "ARI": metrics["ARI"],
                "purity": metrics["purity"],
                "completeness": metrics["completeness"],
                "unknown_tube_ratio": metrics["unknown_tube_ratio"],
                **stats,
            }
            summary_rows.append(row)
            for rank, item in enumerate(chosen):
                selected_rows.append(
                    {
                        "scene": scene,
                        "solver": variant,
                        "control_kind": control_kind,
                        "rank": int(rank),
                        "proposal_id": item.get("proposal_id"),
                        "proposal_type": item.get("proposal_type"),
                        "num_core_tubes": int(len(_core_ids(item))),
                        "risk_score": _risk_score(item),
                        "quality": _quality(item, medium_bonus=0.35),
                        "uses_gt_for_prediction": False,
                    }
                )
    return summary_rows + _aggregate(summary_rows), selected_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_split(Path(args.split))
    gt_by_scene = {
        scene: _load_gt_labels(Path(args.cache_root), scene, int(args.max_tubes_per_window), int(args.image_width), int(args.image_height))
        for scene in scenes
    }
    real_rows = _load_rows(Path(args.proposal_root), args.proposal_label, scenes)
    shuffled_rows = _load_rows(Path(args.shuffle_proposal_root), args.shuffle_proposal_label, scenes)
    _annotate_features([row for rows in real_rows.values() for row in rows], set(), set())
    medium_rows, _ = _generate_medium(real_rows, gt_by_scene)
    medium_rows = _dedupe_rows([row for row in medium_rows if row.get("v29_medium_generator") in {"D1_sibling_merge", "D2_boundary_grow"}])
    for row in medium_rows:
        row["_v29_features"] = {}
    _annotate_features([row for rows in real_rows.values() for row in rows] + medium_rows, set(), set())
    medium_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in medium_rows:
        medium_by_scene[str(row.get("scene"))].append(row)

    all_summary: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    for control_kind, rows_by_scene, medium in [
        ("real", real_rows, medium_by_scene),
        ("shuffled_d4rt", shuffled_rows, {}),
        ("no_temporal", real_rows, medium_by_scene),
        ("mask_only", real_rows, {}),
    ]:
        summary, selected = _control_rows(
            rows_by_scene=rows_by_scene,
            medium_by_scene=medium,
            gt_by_scene=gt_by_scene,
            scenes=scenes,
            control_kind=control_kind,
        )
        all_summary.extend(summary)
        all_selected.extend(selected)
    _write_csv(output_root / "continuation_solver_summary.csv", all_summary)
    _write_csv(output_root / "continuation_selected_proposals.csv", all_selected)
    all_rows = [row for row in all_summary if row.get("scene") == "ALL"]
    real_rows_all = [row for row in all_rows if row.get("control_kind") == "real"]
    best_real = max(real_rows_all, key=lambda row: float(row.get("ARI") or 0.0), default={})
    best_shuffle = max([row for row in all_rows if row.get("control_kind") == "shuffled_d4rt"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    best_no_temporal = max([row for row in all_rows if row.get("control_kind") == "no_temporal"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    best_mask = max([row for row in all_rows if row.get("control_kind") == "mask_only"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    gates = {
        "best_real_solver": best_real.get("solver"),
        "best_real_ARI": best_real.get("ARI"),
        "best_real_purity": best_real.get("purity"),
        "best_real_completeness": best_real.get("completeness"),
        "best_real_scene0081_ARI": best_real.get("scene0081_ARI"),
        "best_shuffled_ARI": best_shuffle.get("ARI"),
        "best_no_temporal_ARI": best_no_temporal.get("ARI"),
        "best_mask_only_ARI": best_mask.get("ARI"),
        "local_gate_pass": bool(
            float(best_real.get("ARI") or 0.0) >= 0.35
            and float(best_real.get("purity") or 0.0) >= 0.85
            and float(best_real.get("completeness") or 0.0) >= 0.50
            and float(best_real.get("scene0081_ARI") or 0.0) >= 0.20
        ),
        "real_beats_shuffled_by_0_20": float(best_real.get("ARI") or 0.0) >= float(best_shuffle.get("ARI") or 0.0) + 0.20,
        "real_beats_no_temporal_by_0_05": float(best_real.get("ARI") or 0.0) >= float(best_no_temporal.get("ARI") or 0.0) + 0.05,
        "real_beats_mask_only_by_0_05": float(best_real.get("ARI") or 0.0) >= float(best_mask.get("ARI") or 0.0) + 0.05,
    }
    gates["continuation_gate_pass"] = bool(
        gates["local_gate_pass"]
        and gates["real_beats_shuffled_by_0_20"]
        and gates["real_beats_no_temporal_by_0_05"]
        and gates["real_beats_mask_only_by_0_05"]
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v29_continuation_solver_repair",
        "is_method_result": True,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": not gates["continuation_gate_pass"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "geometry_field": "D4RT canonical proposal tube memberships plus image-space mask/proposal features",
        "coordinate_frame": "d4rt_canonical for tube geometry; image space for masks/features",
        "alignment_source": "D4RT self-Sim3 inherited from v28 final artifacts",
        "solver_variants": [
            "C0_p11_reconstructed",
            "C1_drop_risky_strict",
            "C2_seed_revert_risky",
            "C3_child_replace_conservative",
            "C4_child_replace_aggressive",
            "C5_child_replace_or_seed_revert",
            "C6_tube_partition_cover",
            "C7_tube_partition_seed_revert",
        ],
        "gates": gates,
    }
    (output_root / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return {"output_root": str(output_root), "gates": gates}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v29 continuation solver repair variants.")
    parser.add_argument("--proposal-root", required=True)
    parser.add_argument("--proposal-label", required=True)
    parser.add_argument("--shuffle-proposal-root", required=True)
    parser.add_argument("--shuffle-proposal-label", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-root", default="outputs/audit/v29_solver_continuation")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
