from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{seq_name}.npz"


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _load_prediction(root: Path, config: str, suffix: str, seq_name: str) -> dict[str, np.ndarray]:
    path = _prediction_path(root, config, suffix, seq_name)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {
            "pred_masks": data["pred_masks"].astype(bool, copy=False),
            "pred_score": data["pred_score"].astype(np.float32, copy=False),
            "pred_classes": data["pred_classes"].astype(np.int32, copy=False),
        }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _candidate_conflict_ratio(candidate_support_masks: np.ndarray) -> np.ndarray:
    if candidate_support_masks.size == 0:
        return np.zeros((candidate_support_masks.shape[1],), dtype=np.float32)
    owner_counts = candidate_support_masks.sum(axis=1)
    conflict_area = candidate_support_masks[owner_counts > 1, :].sum(axis=0).astype(np.float64)
    area = np.maximum(candidate_support_masks.sum(axis=0).astype(np.float64), 1.0)
    return (conflict_area / area).astype(np.float32)


def _edge_quality(
    *,
    score_norm: np.ndarray,
    conflict_ratio: np.ndarray,
    slot_ioc: np.ndarray,
    candidate_ioc: np.ndarray,
    iou: np.ndarray,
    area_ratio: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    ratio = np.maximum(area_ratio, 1.0e-6)
    log_ratio_error = np.abs(np.log(ratio))
    max_log_error = max(abs(np.log(max(float(args.max_area_ratio), 1.0e-6))), abs(np.log(max(float(args.min_area_ratio), 1.0e-6))), 1.0)
    area_match = np.clip(1.0 - log_ratio_error / max_log_error, 0.0, 1.0)
    quality = (
        float(args.iou_weight) * iou
        + float(args.slot_ioc_weight) * slot_ioc
        + float(args.candidate_ioc_weight) * candidate_ioc
        + float(args.score_weight) * score_norm
        + float(args.area_match_weight) * area_match
        - float(args.conflict_weight) * conflict_ratio
    )
    return quality.astype(np.float32)


def _greedy_assign(
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_quality: np.ndarray,
    num_slots: int,
    num_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((edge_cols, edge_rows, -edge_quality))
    slot_to_candidate = np.full((num_slots,), -1, dtype=np.int64)
    slot_quality = np.zeros((num_slots,), dtype=np.float32)
    used_candidates = np.zeros((num_candidates,), dtype=bool)
    for edge_idx in order.tolist():
        slot = int(edge_rows[edge_idx])
        cand = int(edge_cols[edge_idx])
        if slot_to_candidate[slot] >= 0 or used_candidates[cand]:
            continue
        slot_to_candidate[slot] = cand
        slot_quality[slot] = float(edge_quality[edge_idx])
        used_candidates[cand] = True
    return slot_to_candidate, slot_quality, used_candidates


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    slots = _load_prediction(root, args.slot_config, args.pred_suffix, seq_name)
    candidates = _load_prediction(root, args.candidate_config, args.pred_suffix, seq_name)
    slot_masks = slots["pred_masks"]
    candidate_masks = candidates["pred_masks"]
    if slot_masks.shape[0] != candidate_masks.shape[0]:
        raise ValueError(
            f"{seq_name}: vertex mismatch slot={slot_masks.shape[0]} candidate={candidate_masks.shape[0]}"
        )

    support_path = _tmp_path(root, args.score_pre_points_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    support_ids = np.load(support_path).astype(np.int64)
    slot_support = slot_masks[support_ids, :] if support_ids.size else np.zeros((0, slot_masks.shape[1]), dtype=bool)
    candidate_support = (
        candidate_masks[support_ids, :] if support_ids.size else np.zeros((0, candidate_masks.shape[1]), dtype=bool)
    )
    slot_area = slot_support.sum(axis=0).astype(np.float64)
    candidate_area = candidate_support.sum(axis=0).astype(np.float64)
    valid_slots = slot_area >= float(args.min_slot_area)
    valid_candidates = candidate_area >= float(args.min_candidate_area)

    if slot_support.shape[1] == 0 or candidate_support.shape[1] == 0:
        intersections = np.zeros((slot_support.shape[1], candidate_support.shape[1]), dtype=np.float32)
    else:
        intersections = slot_support.astype(np.int32).T @ candidate_support.astype(np.int32)
        intersections = intersections.astype(np.float64)

    slot_ioc = intersections / np.maximum(slot_area.reshape(-1, 1), 1.0)
    candidate_ioc = intersections / np.maximum(candidate_area.reshape(1, -1), 1.0)
    union = slot_area.reshape(-1, 1) + candidate_area.reshape(1, -1) - intersections
    iou = intersections / np.maximum(union, 1.0)
    area_ratio = candidate_area.reshape(1, -1) / np.maximum(slot_area.reshape(-1, 1), 1.0)

    score_norm = _normalize(candidates["pred_score"]).reshape(1, -1)
    conflict_ratio = _candidate_conflict_ratio(candidate_support).reshape(1, -1)
    edge_quality = _edge_quality(
        score_norm=score_norm,
        conflict_ratio=conflict_ratio,
        slot_ioc=slot_ioc,
        candidate_ioc=candidate_ioc,
        iou=iou,
        area_ratio=area_ratio,
        args=args,
    )
    valid_edges = (
        valid_slots.reshape(-1, 1)
        & valid_candidates.reshape(1, -1)
        & (slot_ioc >= float(args.min_slot_ioc))
        & (candidate_ioc >= float(args.min_candidate_ioc))
        & (iou >= float(args.min_iou))
        & (area_ratio >= float(args.min_area_ratio))
        & (area_ratio <= float(args.max_area_ratio))
        & (edge_quality >= float(args.min_edge_quality))
    )
    edge_rows, edge_cols = np.nonzero(valid_edges)
    qualities = edge_quality[edge_rows, edge_cols] if edge_rows.size else np.zeros((0,), dtype=np.float32)
    slot_to_candidate, slot_quality, used_candidates = _greedy_assign(
        edge_rows.astype(np.int64),
        edge_cols.astype(np.int64),
        qualities.astype(np.float32),
        num_slots=slot_masks.shape[1],
        num_candidates=candidate_masks.shape[1],
    )

    out_masks: list[np.ndarray] = []
    out_classes: list[int] = []
    out_scores: list[float] = []
    selected_slots = 0
    fallback_slots = 0
    selected_candidate_ids: list[int] = []
    for slot_idx in range(slot_masks.shape[1]):
        cand_idx = int(slot_to_candidate[slot_idx])
        if cand_idx >= 0:
            selected_slots += 1
            selected_candidate_ids.append(cand_idx)
            out_masks.append(candidate_masks[:, cand_idx].copy())
            out_classes.append(int(candidates["pred_classes"][cand_idx]))
            out_scores.append(float(args.selected_score_base) + float(args.selected_score_scale) * float(slot_quality[slot_idx]))
        elif bool(args.keep_unassigned_slots) and valid_slots[slot_idx]:
            fallback_slots += 1
            out_masks.append(slot_masks[:, slot_idx].copy())
            out_classes.append(int(slots["pred_classes"][slot_idx]))
            out_scores.append(float(args.fallback_slot_score))

    unmatched_added = 0
    if int(args.add_unmatched_top_k) != 0:
        unmatched = np.flatnonzero(valid_candidates & ~used_candidates)
        if unmatched.size:
            unmatched_quality = (
                float(args.score_weight) * _normalize(candidates["pred_score"])[unmatched]
                + float(args.area_match_weight) * _normalize(np.log1p(candidate_area))[unmatched]
                - float(args.conflict_weight) * _candidate_conflict_ratio(candidate_support)[unmatched]
            )
            order = np.argsort(-unmatched_quality)
            if int(args.add_unmatched_top_k) > 0:
                order = order[: int(args.add_unmatched_top_k)]
            for pos in order.tolist():
                cand_idx = int(unmatched[pos])
                out_masks.append(candidate_masks[:, cand_idx].copy())
                out_classes.append(int(candidates["pred_classes"][cand_idx]))
                out_scores.append(float(args.unmatched_score))
                unmatched_added += 1

    if out_masks:
        masks_out = np.stack(out_masks, axis=1).astype(bool, copy=False)
        scores_out = np.asarray(out_scores, dtype=np.float32)
        classes_out = np.asarray(out_classes, dtype=np.int32)
        order = np.argsort(-scores_out, kind="stable")
        masks_out = masks_out[:, order]
        scores_out = scores_out[order]
        classes_out = classes_out[order]
    else:
        masks_out = np.zeros((slot_masks.shape[0], 0), dtype=bool)
        scores_out = np.zeros((0,), dtype=np.float32)
        classes_out = np.zeros((0,), dtype=np.int32)

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{seq_name}.npz",
        pred_masks=masks_out,
        pred_score=scores_out,
        pred_classes=classes_out,
    )

    tmp_in = _tmp_path(root, args.slot_config if args.copy_tmp_from == "slot" else args.candidate_config, seq_name)
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
    else:
        np.save(tmp_out, np.flatnonzero(masks_out.any(axis=1)).astype(np.int64))

    selected_quality = slot_quality[slot_to_candidate >= 0]
    selected_arr = np.asarray(selected_candidate_ids, dtype=np.int64)
    selected_iou = []
    selected_slot_ioc = []
    selected_candidate_ioc = []
    if selected_arr.size:
        selected_slot_ids = np.flatnonzero(slot_to_candidate >= 0)
        selected_iou = iou[selected_slot_ids, selected_arr].tolist()
        selected_slot_ioc = slot_ioc[selected_slot_ids, selected_arr].tolist()
        selected_candidate_ioc = candidate_ioc[selected_slot_ids, selected_arr].tolist()
    return {
        "seq_name": seq_name,
        "slot_config": args.slot_config,
        "candidate_config": args.candidate_config,
        "output_config": args.output_config,
        "num_slots": int(slot_masks.shape[1]),
        "num_candidates": int(candidate_masks.shape[1]),
        "num_valid_slots": int(np.count_nonzero(valid_slots)),
        "num_valid_candidates": int(np.count_nonzero(valid_candidates)),
        "num_candidate_edges": int(edge_rows.shape[0]),
        "num_selected_slots": int(selected_slots),
        "num_fallback_slots": int(fallback_slots),
        "num_unmatched_added": int(unmatched_added),
        "num_output_instances": int(masks_out.shape[1]),
        "output_union": int(masks_out.any(axis=1).sum()) if masks_out.shape[1] else 0,
        "selected_quality_mean": float(np.mean(selected_quality)) if selected_quality.size else 0.0,
        "selected_iou_mean": float(np.mean(selected_iou)) if selected_iou else 0.0,
        "selected_slot_ioc_mean": float(np.mean(selected_slot_ioc)) if selected_slot_ioc else 0.0,
        "selected_candidate_ioc_mean": float(np.mean(selected_candidate_ioc)) if selected_candidate_ioc else 0.0,
    }


def _write_manifest(args: argparse.Namespace) -> None:
    diagnostic_only = bool(args.diagnostic_only)
    manifest = build_prediction_manifest(
        root=args.root,
        output_config=args.output_config,
        is_method_result=not diagnostic_only,
        is_diagnostic_only=diagnostic_only,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.slot_config, args.candidate_config, args.score_pre_points_config],
        pre_points_policy=f"copy_tmp_from:{args.copy_tmp_from}",
        support_policy="slotwise_candidate_select",
        notes=(
            "Assign candidate full-scene masks to object slots using only prediction/support overlap. "
            "Mark diagnostic-only when candidates include Stream3D/scannet baseline predictions."
        ),
        extra={
            "algorithm": "slotwise_candidate_select",
            "eval_policy": args.eval_policy,
            "slot_config": args.slot_config,
            "candidate_config": args.candidate_config,
            "score_pre_points_config": args.score_pre_points_config,
            "copy_tmp_from": args.copy_tmp_from,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Stream4D candidates to fixed-support object slots without GT.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--slot-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--copy-tmp-from", default="slot", choices=["slot", "candidate"])
    parser.add_argument("--min-slot-area", type=float, default=1.0)
    parser.add_argument("--min-candidate-area", type=float, default=1.0)
    parser.add_argument("--min-slot-ioc", type=float, default=0.20)
    parser.add_argument("--min-candidate-ioc", type=float, default=0.30)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--min-area-ratio", type=float, default=0.20)
    parser.add_argument("--max-area-ratio", type=float, default=3.00)
    parser.add_argument("--min-edge-quality", type=float, default=-999.0)
    parser.add_argument("--iou-weight", type=float, default=0.25)
    parser.add_argument("--slot-ioc-weight", type=float, default=0.20)
    parser.add_argument("--candidate-ioc-weight", type=float, default=0.25)
    parser.add_argument("--score-weight", type=float, default=0.20)
    parser.add_argument("--area-match-weight", type=float, default=0.10)
    parser.add_argument("--conflict-weight", type=float, default=0.20)
    parser.add_argument("--selected-score-base", type=float, default=1.0)
    parser.add_argument("--selected-score-scale", type=float, default=0.25)
    parser.add_argument("--keep-unassigned-slots", action="store_true")
    parser.add_argument("--fallback-slot-score", type=float, default=0.05)
    parser.add_argument("--add-unmatched-top-k", type=int, default=0)
    parser.add_argument("--unmatched-score", type=float, default=0.01)
    parser.add_argument("--summary-root", default="outputs/slotwise_candidate_select")
    parser.add_argument("--eval-policy", default="slotwise_candidate_select")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    aggregate: dict[str, float] = {}
    if rows:
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        for key in numeric_keys:
            aggregate[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    summary = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.output_config}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    _write_manifest(args)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
