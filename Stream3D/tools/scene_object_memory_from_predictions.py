from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{seq_name}.npz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _order_candidates(scores: np.ndarray, areas: np.ndarray, mode: str) -> np.ndarray:
    idx = np.arange(scores.shape[0], dtype=np.int64)
    if mode == "score_area":
        return np.asarray(sorted(idx.tolist(), key=lambda i: (-float(scores[i]), -int(areas[i]), int(i))), dtype=np.int64)
    if mode == "area":
        return np.asarray(sorted(idx.tolist(), key=lambda i: (-int(areas[i]), -float(scores[i]), int(i))), dtype=np.int64)
    if mode == "small_area":
        return np.asarray(sorted(idx.tolist(), key=lambda i: (int(areas[i]), -float(scores[i]), int(i))), dtype=np.int64)
    if mode == "score":
        return np.asarray(sorted(idx.tolist(), key=lambda i: (-float(scores[i]), int(i))), dtype=np.int64)
    raise ValueError(f"Unsupported order mode: {mode}")


def _exclusive_masks(masks: np.ndarray, scores: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    if mode == "none" or masks.shape[1] <= 1:
        return masks, int(np.count_nonzero(masks.sum(axis=1) > 1)) if masks.shape[1] else 0
    owner_counts = masks.sum(axis=1)
    conflict_ids = np.flatnonzero(owner_counts > 1)
    if conflict_ids.size == 0:
        return masks, 0
    out = masks.copy()
    areas = out.sum(axis=0).astype(np.float64)
    for point_id in conflict_ids.tolist():
        owners = np.flatnonzero(out[int(point_id), :])
        if owners.size <= 1:
            continue
        if mode == "score":
            keep = int(owners[np.argmax(scores[owners])])
        elif mode == "small_area":
            keep = int(owners[np.argmin(areas[owners])])
        elif mode == "large_area":
            keep = int(owners[np.argmax(areas[owners])])
        else:
            raise ValueError(f"Unsupported exclusive mode: {mode}")
        out[int(point_id), owners] = False
        out[int(point_id), keep] = True
    return out, int(conflict_ids.shape[0])


def _class_vote(classes: np.ndarray, members: list[int]) -> int:
    if classes.shape[0] == 0:
        return 0
    vals = [int(classes[idx]) for idx in members]
    return int(Counter(vals).most_common(1)[0][0]) if vals else 0


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    if scores.shape[0] != masks.shape[1]:
        scores = np.ones((masks.shape[1],), dtype=np.float32)
    if classes.shape[0] != masks.shape[1]:
        classes = np.zeros((masks.shape[1],), dtype=np.int32)

    areas = masks.sum(axis=0).astype(np.int64)
    order = _order_candidates(scores, areas, args.order_mode)
    slots: list[np.ndarray] = []
    slot_scores: list[float] = []
    slot_members: list[list[int]] = []
    slot_classes: list[int] = []

    stats = Counter()
    for idx in order.tolist():
        area = int(areas[idx])
        if area < int(args.min_area):
            stats["rejected_small"] += 1
            continue
        cand = masks[:, int(idx)]
        cand_ids = np.flatnonzero(cand)
        if not slots:
            slots.append(cand.copy())
            slot_scores.append(float(scores[idx]))
            slot_members.append([int(idx)])
            slot_classes.append(int(classes[idx]))
            stats["birth"] += 1
            continue

        slot_matrix = np.stack(slots, axis=1)
        slot_areas = slot_matrix.sum(axis=0).astype(np.float64)
        intersections = slot_matrix[cand_ids, :].sum(axis=0).astype(np.float64) if cand_ids.size else np.zeros(len(slots))
        cand_area = float(max(area, 1))
        cand_ioc = intersections / cand_area
        slot_ioc = intersections / np.maximum(slot_areas, 1.0)
        iou = intersections / np.maximum(slot_areas + cand_area - intersections, 1.0)
        update_score = np.maximum.reduce([cand_ioc, slot_ioc, iou])
        best = int(np.argmax(update_score))

        ambiguous = np.flatnonzero(
            (cand_ioc >= float(args.ambiguous_candidate_ioc))
            | (slot_ioc >= float(args.ambiguous_slot_ioc))
            | (iou >= float(args.ambiguous_iou))
        )
        if bool(args.reject_ambiguous) and ambiguous.size > 1:
            stats["rejected_ambiguous"] += 1
            continue

        should_update = (
            cand_ioc[best] >= float(args.attach_candidate_ioc)
            or slot_ioc[best] >= float(args.attach_slot_ioc)
            or iou[best] >= float(args.attach_iou)
        )
        if should_update:
            if args.update_mode == "union":
                slots[best] = np.logical_or(slots[best], cand)
            elif args.update_mode == "keep_slot":
                pass
            elif args.update_mode == "new_points_only":
                new_points = cand & ~slot_matrix.any(axis=1)
                slots[best] = np.logical_or(slots[best], new_points)
            else:
                raise ValueError(f"Unsupported update mode: {args.update_mode}")
            slot_scores[best] = max(float(slot_scores[best]), float(scores[idx]))
            slot_members[best].append(int(idx))
            slot_classes[best] = _class_vote(classes, slot_members[best])
            stats["update"] += 1
            continue

        if update_score[best] <= float(args.birth_max_overlap):
            slots.append(cand.copy())
            slot_scores.append(float(scores[idx]))
            slot_members.append([int(idx)])
            slot_classes.append(int(classes[idx]))
            stats["birth"] += 1
        else:
            stats["rejected_overlap"] += 1

    out_masks = np.stack(slots, axis=1).astype(bool, copy=False) if slots else np.zeros((masks.shape[0], 0), dtype=bool)
    out_scores = np.asarray(slot_scores, dtype=np.float32)
    out_classes = np.asarray(slot_classes, dtype=np.int32)
    if out_masks.shape[1]:
        keep = out_masks.sum(axis=0) >= int(args.min_output_area)
        stats["dropped_small_output"] = int(np.count_nonzero(~keep))
        out_masks = out_masks[:, keep]
        out_scores = out_scores[keep]
        out_classes = out_classes[keep]

    conflict_before_exclusive = int(np.count_nonzero(out_masks.sum(axis=1) > 1)) if out_masks.shape[1] else 0
    out_masks, exclusive_conflict_points = _exclusive_masks(out_masks, out_scores, args.exclusive_mode)
    keep_nonempty = np.flatnonzero(out_masks.any(axis=0))
    if keep_nonempty.shape[0] != out_masks.shape[1]:
        out_masks = out_masks[:, keep_nonempty] if keep_nonempty.size else np.zeros((masks.shape[0], 0), dtype=bool)
        out_scores = out_scores[keep_nonempty] if keep_nonempty.size else np.zeros((0,), dtype=np.float32)
        out_classes = out_classes[keep_nonempty] if keep_nonempty.size else np.zeros((0,), dtype=np.int32)
    if out_masks.shape[1]:
        order_out = np.asarray(sorted(range(out_masks.shape[1]), key=lambda i: (-float(out_scores[i]), int(i))), dtype=np.int64)
        out_masks = out_masks[:, order_out]
        out_scores = out_scores[order_out]
        out_classes = out_classes[order_out]

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores,
        pred_classes=out_classes,
    )
    pre_points = np.flatnonzero(out_masks.any(axis=1)).astype(np.int64)
    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / f"{seq_name}_pre_points.npy", pre_points)
    conflict_after = int(np.count_nonzero(out_masks.sum(axis=1) > 1)) if out_masks.shape[1] else 0

    return {
        "seq_name": seq_name,
        "num_input_objects": int(masks.shape[1]),
        "num_output_objects": int(out_masks.shape[1]),
        "num_output_points": int(pre_points.shape[0]),
        "output_point_ratio": float(pre_points.shape[0] / max(masks.shape[0], 1)),
        "conflict_before_exclusive": int(conflict_before_exclusive),
        "exclusive_conflict_points": int(exclusive_conflict_points),
        "conflict_after": int(conflict_after),
        "conflict_rate_after": float(conflict_after / max(pre_points.shape[0], 1)),
        "mean_members_per_slot": float(np.mean([len(m) for m in slot_members])) if slot_members else 0.0,
        "max_members_per_slot": int(max((len(m) for m in slot_members), default=0)),
        **{key: int(value) for key, value in stats.items()},
    }


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted(key for row in rows for key, value in row.items() if isinstance(value, (int, float)))
    return {
        "algorithm": "scene_object_memory_from_predictions",
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_scenes": len(rows),
        **{
            f"{key}_mean": float(np.mean([float(row[key]) for row in rows if key in row]))
            for key in numeric_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight scene-level object memory from predicted masks.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--order-mode", default="score_area", choices=["score_area", "area", "small_area", "score"])
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--min-output-area", type=int, default=20)
    parser.add_argument("--attach-candidate-ioc", type=float, default=0.50)
    parser.add_argument("--attach-slot-ioc", type=float, default=0.30)
    parser.add_argument("--attach-iou", type=float, default=0.12)
    parser.add_argument("--birth-max-overlap", type=float, default=0.12)
    parser.add_argument("--ambiguous-candidate-ioc", type=float, default=0.35)
    parser.add_argument("--ambiguous-slot-ioc", type=float, default=0.35)
    parser.add_argument("--ambiguous-iou", type=float, default=0.20)
    parser.add_argument("--reject-ambiguous", action="store_true")
    parser.add_argument("--update-mode", default="union", choices=["union", "keep_slot", "new_points_only"])
    parser.add_argument("--exclusive-mode", default="none", choices=["none", "score", "small_area", "large_area"])
    parser.add_argument("--summary-root", default="outputs/v9_scene_object_memory")
    parser.add_argument("--eval-policy", default="own_recompute_scene_object_memory")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_prediction_manifest(
        root=args.root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config],
        pre_points_policy="recompute",
        support_policy=(
            "scene_object_memory:"
            f"order={args.order_mode}:attach_cioc={args.attach_candidate_ioc}:"
            f"attach_sioc={args.attach_slot_ioc}:attach_iou={args.attach_iou}:"
            f"reject_ambiguous={bool(args.reject_ambiguous)}:exclusive={args.exclusive_mode}"
        ),
        notes=(
            "Builds scene-level object slots from predicted masks with birth/update/reject memory "
            "transitions and optional ambiguous-bridge rejection. Uses prediction masks only; no GT is read."
        ),
        extra={
            "algorithm": "scene_object_memory_from_predictions",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "summary_path": str(out_path),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[scene-object-memory] wrote {out_path}")


if __name__ == "__main__":
    main()
