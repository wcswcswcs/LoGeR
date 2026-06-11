from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.v11_candidate_pool_oracle import _conflict_rate, _json_safe, _load_prediction, _read_seq_list


@dataclass
class FeaturePack:
    area: np.ndarray
    score: np.ndarray
    score_norm: np.ndarray
    area_norm: np.ndarray
    conflict_rate: np.ndarray
    max_candidate_overlap: np.ndarray
    anchor_agreement: np.ndarray
    seed_overlap: np.ndarray
    visible_outside_negative_proxy: np.ndarray
    boundary_risk_proxy: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _overlap_features(candidates: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = candidates.shape[1]
    if anchors.shape[1] == 0 or n == 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    cand_area = candidates.sum(axis=0).astype(np.float64)
    anchor_area = anchors.sum(axis=0).astype(np.float64)
    inter = candidates.astype(np.int64).T @ anchors.astype(np.int64)
    union = cand_area[:, None] + anchor_area[None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    min_ioc = inter / np.maximum(np.minimum(cand_area[:, None], anchor_area[None, :]), 1.0)
    cand_ioc = inter / np.maximum(cand_area[:, None], 1.0)
    return np.max(iou, axis=1), np.max(min_ioc, axis=1), np.max(cand_ioc, axis=1)


def _candidate_conflict(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = masks.shape[1]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    areas = masks.sum(axis=0).astype(np.float64)
    point_counts = masks.astype(np.int16).sum(axis=1)
    overlap_mass = (masks.T @ np.maximum(point_counts - 1, 0)).astype(np.float64)
    conflict_rate = overlap_mass / np.maximum(areas, 1.0)
    inter = masks.astype(np.int64).T @ masks.astype(np.int64)
    np.fill_diagonal(inter, 0)
    max_ioc = inter / np.maximum(np.minimum(areas[:, None], areas[None, :]), 1.0)
    return conflict_rate, np.max(max_ioc, axis=1)


def _build_features(
    *,
    masks: np.ndarray,
    scores: np.ndarray,
    seed_masks: np.ndarray,
    anchor_masks: np.ndarray,
    variant: str,
    rng: np.random.Generator,
) -> FeaturePack:
    area = masks.sum(axis=0).astype(np.float64)
    log_score = np.log1p(np.maximum(scores.astype(np.float64), 0.0))
    score_norm = _normalize(log_score)
    area_norm = _normalize(np.log1p(area))
    conflict_rate, max_candidate_overlap = _candidate_conflict(masks)
    _, anchor_min_ioc, anchor_candidate_coverage = _overlap_features(masks, anchor_masks)
    _, seed_min_ioc, _ = _overlap_features(masks, seed_masks)
    anchor_agreement = np.maximum(anchor_min_ioc, 0.5 * anchor_candidate_coverage)
    if variant == "shuffle":
        if anchor_agreement.size:
            anchor_agreement = anchor_agreement[rng.permutation(anchor_agreement.size)]
        if seed_min_ioc.size:
            seed_min_ioc = seed_min_ioc[rng.permutation(seed_min_ioc.size)]
    if variant == "no_temporal":
        score_norm = np.zeros_like(score_norm)
    negative = conflict_rate * (1.0 - np.clip(anchor_candidate_coverage, 0.0, 1.0))
    if variant == "no_negative":
        negative = np.zeros_like(negative)
    boundary = max_candidate_overlap * conflict_rate
    return FeaturePack(
        area=area,
        score=scores.astype(np.float64),
        score_norm=score_norm,
        area_norm=area_norm,
        conflict_rate=conflict_rate,
        max_candidate_overlap=max_candidate_overlap,
        anchor_agreement=anchor_agreement,
        seed_overlap=seed_min_ioc,
        visible_outside_negative_proxy=negative,
        boundary_risk_proxy=boundary,
    )


def _candidate_base_score(features: FeaturePack, args: argparse.Namespace, variant: str) -> np.ndarray:
    if variant == "area_only":
        return features.area_norm.copy()
    return (
        float(args.w_anchor) * features.anchor_agreement
        + float(args.w_score) * features.score_norm
        + float(args.w_area) * features.area_norm
        - float(args.w_conflict) * features.conflict_rate
        - float(args.w_negative) * features.visible_outside_negative_proxy
        - float(args.w_boundary) * features.boundary_risk_proxy
    )


def _min_ioc(mask: np.ndarray, other: np.ndarray) -> float:
    inter = int(np.count_nonzero(mask & other))
    denom = max(min(int(np.count_nonzero(mask)), int(np.count_nonzero(other))), 1)
    return float(inter / denom)


def _iou(mask: np.ndarray, other: np.ndarray) -> float:
    inter = int(np.count_nonzero(mask & other))
    union = int(np.count_nonzero(mask | other))
    return float(inter / max(union, 1))


def _grow_slot(
    *,
    seed_idx: int,
    masks: np.ndarray,
    features: FeaturePack,
    base_score: np.ndarray,
    args: argparse.Namespace,
    variant: str,
) -> dict[str, Any]:
    selected = [int(seed_idx)]
    selected_set = {int(seed_idx)}
    slot_mask = masks[:, seed_idx].copy()
    slot_score = float(base_score[seed_idx])
    add_records: list[dict[str, Any]] = [
        {
            "candidate_index": int(seed_idx),
            "delta": float(slot_score),
            "new_support_ratio": 1.0,
            "slot_overlap": 1.0,
        }
    ]
    for _ in range(max(int(args.max_measurements_per_slot) - 1, 0)):
        slot_area = float(np.count_nonzero(slot_mask))
        best_idx = -1
        best_delta = float(args.grow_min_delta)
        best_record: dict[str, Any] | None = None
        for idx in range(masks.shape[1]):
            if idx in selected_set:
                continue
            cand = masks[:, idx]
            cand_area = float(features.area[idx])
            if cand_area < float(args.min_candidate_points):
                continue
            inter = float(np.count_nonzero(cand & slot_mask))
            if inter <= 0.0:
                continue
            slot_overlap = inter / max(min(cand_area, slot_area), 1.0)
            if slot_overlap < float(args.grow_neighbor_min_ioc) and features.seed_overlap[idx] < float(args.seed_min_ioc):
                continue
            new_support = float(np.count_nonzero(cand & ~slot_mask))
            new_ratio = new_support / max(cand_area, 1.0)
            if new_ratio < float(args.min_new_support_ratio):
                continue
            area_jump = cand_area / max(slot_area, 1.0)
            if area_jump > float(args.max_area_jump) and new_ratio < float(args.area_jump_min_new_ratio):
                continue
            delta = (
                float(base_score[idx])
                + float(args.w_new_support) * new_ratio
                + float(args.w_slot_overlap) * min(slot_overlap, 1.0)
                - float(args.w_complexity) * len(selected)
                - float(args.w_area_jump) * max(area_jump - 1.0, 0.0)
            )
            if variant == "area_only":
                delta = float(features.area_norm[idx] + 0.25 * new_ratio - 0.15 * slot_overlap)
            if delta > best_delta:
                best_idx = int(idx)
                best_delta = float(delta)
                best_record = {
                    "candidate_index": int(idx),
                    "delta": float(delta),
                    "new_support_ratio": float(new_ratio),
                    "slot_overlap": float(slot_overlap),
                    "area_jump": float(area_jump),
                    "positive_evidence": float(base_score[idx]),
                    "negative_evidence": float(features.visible_outside_negative_proxy[idx]),
                    "boundary_risk": float(features.boundary_risk_proxy[idx]),
                    "d4rt_consistency": float(features.score_norm[idx]),
                    "appearance_consistency": 0.0,
                }
        if best_idx < 0 or best_record is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        slot_mask |= masks[:, best_idx]
        slot_score += best_delta
        add_records.append(best_record)
    return {
        "selected_indices": selected,
        "mask": slot_mask,
        "score": float(slot_score),
        "add_records": add_records,
        "area": int(np.count_nonzero(slot_mask)),
    }


def _pack_slots(slots: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    for slot in sorted(slots, key=lambda item: (float(item["score"]), int(item["area"])), reverse=True):
        if int(slot["area"]) < int(args.min_output_points):
            continue
        overlaps = [_min_ioc(slot["mask"], kept["mask"]) for kept in packed]
        if overlaps and max(overlaps) >= float(args.packing_max_min_ioc):
            continue
        packed.append(slot)
        if int(args.max_slots) > 0 and len(packed) >= int(args.max_slots):
            break
    return packed


def _select_fallback_candidates(
    *,
    selected_slots: list[dict[str, Any]],
    masks: np.ndarray,
    features: FeaturePack,
    base_score: np.ndarray,
    target_points: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if target_points <= 0:
        return selected_slots
    current = np.zeros((masks.shape[0],), dtype=bool)
    for slot in selected_slots:
        current |= slot["mask"]
    order = sorted(
        range(masks.shape[1]),
        key=lambda idx: (float(base_score[idx]), float(features.area[idx]), -int(idx)),
        reverse=True,
    )
    existing_indices = {idx for slot in selected_slots for idx in slot["selected_indices"]}
    added = 0
    for idx in order:
        if int(idx) in existing_indices:
            continue
        cand = masks[:, idx]
        cand_area = int(features.area[idx])
        if cand_area < int(args.min_output_points):
            continue
        if selected_slots:
            max_overlap = max(_min_ioc(cand, slot["mask"]) for slot in selected_slots)
            if max_overlap >= float(args.fallback_max_min_ioc):
                continue
        new_points = int(np.count_nonzero(cand & ~current))
        if new_points < int(args.fallback_min_new_points):
            continue
        slot = {
            "selected_indices": [int(idx)],
            "mask": cand.copy(),
            "score": float(base_score[idx]),
            "add_records": [
                {
                    "candidate_index": int(idx),
                    "delta": float(base_score[idx]),
                    "new_support_ratio": float(new_points / max(cand_area, 1)),
                    "slot_overlap": 0.0,
                    "fallback_support_floor": True,
                }
            ],
            "area": cand_area,
            "fallback_support_floor": True,
        }
        selected_slots.append(slot)
        current |= cand
        added += 1
        if int(np.count_nonzero(current)) >= target_points:
            break
        if int(args.max_fallback_slots) > 0 and added >= int(args.max_fallback_slots):
            break
    return selected_slots


def _random_same_count(
    *,
    root: Path,
    args: argparse.Namespace,
    scene: str,
    masks: np.ndarray,
    features: FeaturePack,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    same_count = 0
    if args.same_count_from_config:
        source_path = root / "data" / "prediction" / f"{args.same_count_from_config}_class_agnostic" / f"{scene}.npz"
        if source_path.exists():
            with np.load(source_path) as data:
                same_count = int(np.asarray(data["pred_masks"]).shape[1])
    if same_count <= 0:
        same_count = int(args.random_default_count)
    valid = np.flatnonzero(features.area >= int(args.min_output_points))
    if valid.size == 0:
        return []
    count = min(int(same_count), int(valid.size))
    chosen = rng.choice(valid, size=count, replace=False)
    slots = []
    for idx in chosen.tolist():
        slots.append(
            {
                "selected_indices": [int(idx)],
                "mask": masks[:, int(idx)].copy(),
                "score": float(rng.random()),
                "add_records": [{"candidate_index": int(idx), "delta": 0.0, "new_support_ratio": 1.0, "slot_overlap": 0.0}],
                "area": int(features.area[int(idx)]),
            }
        )
    return slots


def _scene_solve(root: Path, args: argparse.Namespace, scene: str, rng: np.random.Generator) -> dict[str, Any]:
    start = time.perf_counter()
    hybrid = _load_prediction(root, args.pred_config, args.pred_suffix, scene)
    regionlet = _load_prediction(root, args.regionlet_config, args.pred_suffix, scene)
    surfel = _load_prediction(root, args.surfel_config, args.pred_suffix, scene)
    mask = _load_prediction(root, args.mask_config, args.pred_suffix, scene)
    h_masks = np.asarray(hybrid["pred_masks"], dtype=bool)
    h_scores = np.asarray(hybrid["pred_score"], dtype=np.float64)
    r_masks = np.asarray(regionlet["pred_masks"], dtype=bool)
    s_masks = np.asarray(surfel["pred_masks"], dtype=bool)
    m_masks = np.asarray(mask["pred_masks"], dtype=bool)
    seed_parts = [r_masks]
    if bool(args.use_surfel_seeds):
        seed_parts.append(s_masks)
    seed_masks = np.concatenate(seed_parts, axis=1) if seed_parts else np.zeros((h_masks.shape[0], 0), dtype=bool)
    anchor_masks = np.concatenate([r_masks, s_masks, m_masks], axis=1)
    features = _build_features(
        masks=h_masks,
        scores=h_scores,
        seed_masks=seed_masks,
        anchor_masks=anchor_masks,
        variant=args.variant,
        rng=rng,
    )
    base_score = _candidate_base_score(features, args, args.variant)
    valid = np.flatnonzero(features.area >= int(args.min_candidate_points))
    if args.variant == "random_same_count":
        slots = _random_same_count(root=root, args=args, scene=scene, masks=h_masks, features=features, rng=rng)
    else:
        if args.variant == "area_only":
            seed_order = sorted(valid.tolist(), key=lambda idx: (float(features.area[idx]), float(h_scores[idx])), reverse=True)
        else:
            seedable = [
                int(idx)
                for idx in valid.tolist()
                if features.seed_overlap[idx] >= float(args.seed_min_ioc)
                or features.anchor_agreement[idx] >= float(args.anchor_seed_min_score)
            ]
            if not seedable:
                seedable = valid.tolist()
            seed_order = sorted(seedable, key=lambda idx: (float(base_score[idx]), float(features.area[idx])), reverse=True)
        if int(args.max_seed_candidates) > 0:
            seed_order = seed_order[: int(args.max_seed_candidates)]
        slots = [
            _grow_slot(seed_idx=int(idx), masks=h_masks, features=features, base_score=base_score, args=args, variant=args.variant)
            for idx in seed_order
        ]
        slots = _pack_slots(slots, args)
        target_points = int(float(args.target_union_ratio) * float(h_masks.shape[0]))
        if float(args.target_union_ratio) > 0:
            slots = _select_fallback_candidates(
                selected_slots=slots,
                masks=h_masks,
                features=features,
                base_score=base_score,
                target_points=target_points,
                args=args,
            )
    if slots:
        out_masks = np.stack([slot["mask"] for slot in slots], axis=1).astype(bool)
        out_scores = np.asarray([float(slot["score"]) for slot in slots], dtype=np.float32)
        order = np.argsort(-out_scores, kind="mergesort")
        out_masks = out_masks[:, order]
        out_scores = out_scores[order]
        slots = [slots[int(idx)] for idx in order.tolist()]
    else:
        out_masks = np.zeros((h_masks.shape[0], 0), dtype=bool)
        out_scores = np.zeros((0,), dtype=np.float32)
    pred_classes = np.zeros((out_masks.shape[1],), dtype=np.int32)
    support = np.flatnonzero(np.any(out_masks, axis=1)).astype(np.int64)
    pred_dir = root / "data" / "prediction" / f"{args.output_config}_class_agnostic"
    tmp_dir = root / "data" / "TMP" / args.output_config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{scene}.npz", pred_masks=out_masks, pred_score=out_scores, pred_classes=pred_classes)
    np.save(tmp_dir / f"{scene}_pre_points.npy", support)

    selected_indices = sorted({int(idx) for slot in slots for idx in slot["selected_indices"]})
    add_records = [record for slot in slots for record in slot.get("add_records", [])]
    runtime = float(time.perf_counter() - start)
    union_ratio = float(support.shape[0] / max(h_masks.shape[0], 1))
    return {
        "scene": scene,
        "variant": args.variant,
        "num_candidates": int(h_masks.shape[1]),
        "num_valid_candidates": int(valid.shape[0]),
        "num_seeds": int(len(seed_order) if args.variant != "random_same_count" else 0),
        "num_candidate_neighbors_per_seed": None,
        "num_slots_before_packing": int(len(seed_order) if args.variant != "random_same_count" else len(slots)),
        "num_slots_after_packing": int(len(slots)),
        "num_measurements_selected": int(sum(len(slot["selected_indices"]) for slot in slots)),
        "num_unique_measurements_selected": int(len(selected_indices)),
        "num_measurements_rejected": int(max(h_masks.shape[1] - len(selected_indices), 0)),
        "selected_measurement_indices": selected_indices,
        "mean_selected_measurements_per_object": float(np.mean([len(slot["selected_indices"]) for slot in slots])) if slots else 0.0,
        "mean_beam_size": float(np.mean([len(slot["selected_indices"]) for slot in slots])) if slots else 0.0,
        "slot_area_expansion_ratio": float(
            np.mean(
                [
                    int(slot["area"]) / max(int(features.area[slot["selected_indices"][0]]), 1)
                    for slot in slots
                    if slot["selected_indices"]
                ]
            )
        )
        if slots
        else 0.0,
        "new_support_ratio_per_added_measurement": float(
            np.mean([float(item.get("new_support_ratio", 0.0)) for item in add_records])
        )
        if add_records
        else 0.0,
        "positive_evidence_mean": float(np.mean([float(base_score[idx]) for idx in selected_indices])) if selected_indices else 0.0,
        "negative_evidence_mean": float(np.mean([float(features.visible_outside_negative_proxy[idx]) for idx in selected_indices]))
        if selected_indices
        else 0.0,
        "boundary_risk_mean": float(np.mean([float(features.boundary_risk_proxy[idx]) for idx in selected_indices]))
        if selected_indices
        else 0.0,
        "appearance_consistency_mean": 0.0,
        "d4rt_consistency_mean": float(np.mean([float(features.score_norm[idx]) for idx in selected_indices])) if selected_indices else 0.0,
        "packing_overlap_penalty": float(_conflict_rate(out_masks[support, :])) if support.size and out_masks.shape[1] else 0.0,
        "candidate_support_ratio": float(
            np.load(root / "data" / "TMP" / args.pred_config / f"{scene}_pre_points.npy").shape[0] / max(h_masks.shape[0], 1)
        ),
        "prediction_union_ratio": union_ratio,
        "support_pre_ratio": union_ratio,
        "num_output_objects": int(out_masks.shape[1]),
        "mean_points_per_object": float(np.mean(out_masks.sum(axis=0))) if out_masks.shape[1] else 0.0,
        "tiny_mask_ratio_lt100": float(np.mean(out_masks.sum(axis=0) < 100)) if out_masks.shape[1] else 0.0,
        "large_mask_ratio_gt1000": float(np.mean(out_masks.sum(axis=0) > 1000)) if out_masks.shape[1] else 0.0,
        "fallback_slots": int(sum(1 for slot in slots if slot.get("fallback_support_floor"))),
        "selection_runtime_seconds": runtime,
    }


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm_name": "v17_non_gt_object_explanation_solver",
        "variant": args.variant,
        "output_config": args.output_config,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }


def _write_summary(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _aggregate(rows, args)
    payload = {"args": vars(args), "summary": summary, "scenes": rows}
    summary_root = root / args.summary_root
    summary_root.mkdir(parents=True, exist_ok=True)
    prefix = summary_root / f"{args.output_config}_summary"
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if key != "selected_measurement_indices"})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    mean = summary["numeric_mean"]
    lines = [
        f"# {args.output_config}",
        "",
        "Non-GT v17 candidate-union object explanation solver. It reads existing candidate masks/scores and never reads GT.",
        "",
        "## Aggregate",
        "",
    ]
    for key in (
        "support_pre_ratio",
        "prediction_union_ratio",
        "num_output_objects",
        "mean_selected_measurements_per_object",
        "slot_area_expansion_ratio",
        "new_support_ratio_per_added_measurement",
        "positive_evidence_mean",
        "negative_evidence_mean",
        "boundary_risk_mean",
        "d4rt_consistency_mean",
        "packing_overlap_penalty",
        "fallback_slots",
        "selection_runtime_seconds",
    ):
        lines.append(f"- {key}: `{mean.get(key)}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | candidates | seeds | slots | selected meas | support% | fallback | pos | neg | boundary | runtime |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["num_candidates"]),
                    str(row["num_seeds"]),
                    str(row["num_slots_after_packing"]),
                    str(row["num_unique_measurements_selected"]),
                    f"{float(row['support_pre_ratio']) * 100.0:.4f}",
                    str(row["fallback_slots"]),
                    f"{float(row['positive_evidence_mean']):.4f}",
                    f"{float(row['negative_evidence_mean']):.4f}",
                    f"{float(row['boundary_risk_mean']):.4f}",
                    f"{float(row['selection_runtime_seconds']):.3f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", default="stream4d_v13_c_hybrid_unsup_probe5")
    parser.add_argument("--regionlet-config", default="stream4d_v13_c_regionlet_unsup_probe5")
    parser.add_argument("--mask-config", default="stream4d_v13_c_mask_unsup_probe5")
    parser.add_argument("--surfel-config", default="stream4d_v13_c_surfel_unsup_probe5")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--summary-root", default="outputs/v17_object_explanation_solver")
    parser.add_argument(
        "--variant",
        choices=["real", "shuffle", "no_temporal", "no_negative", "area_only", "random_same_count"],
        default="real",
    )
    parser.add_argument("--same-count-from-config", default="")
    parser.add_argument("--random-default-count", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--min-candidate-points", type=int, default=100)
    parser.add_argument("--min-output-points", type=int, default=100)
    parser.add_argument("--max-seed-candidates", type=int, default=96)
    parser.add_argument("--max-slots", type=int, default=64)
    parser.add_argument("--max-measurements-per-slot", type=int, default=8)
    parser.add_argument("--seed-min-ioc", type=float, default=0.18)
    parser.add_argument("--anchor-seed-min-score", type=float, default=0.12)
    parser.add_argument("--grow-neighbor-min-ioc", type=float, default=0.10)
    parser.add_argument("--grow-min-delta", type=float, default=0.18)
    parser.add_argument("--min-new-support-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-jump", type=float, default=5.0)
    parser.add_argument("--area-jump-min-new-ratio", type=float, default=0.35)
    parser.add_argument("--packing-max-min-ioc", type=float, default=0.72)
    parser.add_argument("--target-union-ratio", type=float, default=0.30)
    parser.add_argument("--fallback-max-min-ioc", type=float, default=0.78)
    parser.add_argument("--fallback-min-new-points", type=int, default=400)
    parser.add_argument("--max-fallback-slots", type=int, default=64)
    parser.add_argument("--w-anchor", type=float, default=0.95)
    parser.add_argument("--w-score", type=float, default=0.55)
    parser.add_argument("--w-area", type=float, default=0.10)
    parser.add_argument("--w-conflict", type=float, default=0.0)
    parser.add_argument("--w-negative", type=float, default=0.75)
    parser.add_argument("--w-boundary", type=float, default=0.30)
    parser.add_argument("--w-new-support", type=float, default=0.35)
    parser.add_argument("--w-slot-overlap", type=float, default=0.20)
    parser.add_argument("--w-complexity", type=float, default=0.05)
    parser.add_argument("--w-area-jump", type=float, default=0.05)
    parser.add_argument("--use-surfel-seeds", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rng = np.random.default_rng(int(args.random_seed))
    rows = [_scene_solve(root, args, scene, rng) for scene in _read_seq_list((root / args.seq_list).resolve())]
    payload = _write_summary(root, args, rows)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.pred_config, args.regionlet_config, args.mask_config, args.surfel_config],
        pre_points_policy="own_recompute_paper_style",
        support_policy="selected_candidate_union",
        notes=(
            "v17 non-GT candidate-union object explanation solver. The available candidate NPZ files only expose "
            "mask/score/class arrays, so D4RT/temporal terms are score/anchor-overlap proxies rather than raw frame-level features."
        ),
        extra={
            "algorithm_name": "v17_non_gt_object_explanation_solver",
            "algorithm": "v17_non_gt_object_explanation_solver",
            "variant": args.variant,
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "alignment_source": "none",
            "sim3_alignment_used_for_prediction": False,
            "sim3_alignment_used_for_evaluation": False,
            "alignment_used_for_prediction": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "gt_selected_output": False,
            "forbidden_for_method_table": False,
            "is_method_result": True,
            "is_diagnostic_only": False,
            "summary_path": str(Path(args.summary_root) / f"{args.output_config}_summary.json"),
            "runtime_seconds": payload["summary"]["numeric_mean"].get("selection_runtime_seconds"),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
