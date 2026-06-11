from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


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


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _support_features(masks: np.ndarray, support_ids: np.ndarray) -> dict[str, np.ndarray]:
    if support_ids.size:
        if int(support_ids.min()) < 0 or int(support_ids.max()) >= masks.shape[0]:
            raise ValueError(
                f"support ids outside prediction vertices: min={int(support_ids.min())}, "
                f"max={int(support_ids.max())}, vertices={masks.shape[0]}"
            )
        support_masks = masks[support_ids, :]
    else:
        support_masks = np.zeros((0, masks.shape[1]), dtype=bool)
    support_area = support_masks.sum(axis=0).astype(np.float64)
    owner_counts = support_masks.sum(axis=1) if support_masks.shape[0] else np.zeros((0,), dtype=np.int32)
    conflict_area = np.zeros((masks.shape[1],), dtype=np.float64)
    unique_area = np.zeros((masks.shape[1],), dtype=np.float64)
    if support_masks.shape[0]:
        conflict_area = support_masks[owner_counts > 1, :].sum(axis=0).astype(np.float64)
        unique_area = support_masks[owner_counts == 1, :].sum(axis=0).astype(np.float64)
    safe_area = np.maximum(support_area, 1.0)
    return {
        "support_masks": support_masks,
        "support_area": support_area,
        "area_norm": _normalize(np.log1p(support_area)),
        "conflict_ratio": conflict_area / safe_area,
        "unique_ratio": unique_area / safe_area,
    }


def _base_quality(scores: np.ndarray, features: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    score_norm = _normalize(scores)
    quality = (
        float(args.score_weight) * score_norm
        + float(args.area_weight) * features["area_norm"]
        + float(args.unique_weight) * features["unique_ratio"]
        - float(args.conflict_weight) * features["conflict_ratio"]
    )
    return np.asarray(quality, dtype=np.float32)


def _greedy_select(
    support_masks: np.ndarray,
    support_area: np.ndarray,
    quality: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = support_area >= float(args.min_support_area)
    remaining = set(np.flatnonzero(valid).tolist())
    selected: list[int] = []
    selected_union = np.zeros((support_masks.shape[0],), dtype=bool)
    step_records: list[dict[str, float | int]] = []

    max_instances = int(args.max_instances)
    while remaining and (max_instances <= 0 or len(selected) < max_instances):
        rem = np.asarray(sorted(remaining), dtype=np.int64)
        if selected_union.any():
            overlap = support_masks[selected_union, :][:, rem].sum(axis=0).astype(np.float64)
        else:
            overlap = np.zeros((rem.shape[0],), dtype=np.float64)
        area = support_area[rem]
        new_area = np.maximum(area - overlap, 0.0)
        safe_area = np.maximum(area, 1.0)
        novelty = new_area / safe_area
        overlap_ratio = overlap / safe_area
        new_area_norm = _normalize(np.log1p(new_area))
        utility = (
            quality[rem].astype(np.float64)
            + float(args.new_area_weight) * new_area_norm
            + float(args.novelty_weight) * novelty
            - float(args.overlap_penalty) * overlap_ratio
        )
        best_pos = int(np.argmax(utility))
        best_idx = int(rem[best_pos])
        best_new_area = float(new_area[best_pos])
        best_utility = float(utility[best_pos])
        if best_new_area < float(args.min_new_area):
            break
        if best_utility < float(args.min_utility):
            break
        selected.append(best_idx)
        selected_union |= support_masks[:, best_idx]
        remaining.remove(best_idx)
        if bool(args.suppress_overlapped):
            area_remaining = support_area[rem]
            overlapped_ratio = overlap / np.maximum(area_remaining, 1.0)
            for cand in rem[overlapped_ratio >= float(args.suppress_overlap_ratio)].tolist():
                remaining.discard(int(cand))
        step_records.append(
            {
                "step": len(selected),
                "selected_index": best_idx,
                "utility": best_utility,
                "new_area": best_new_area,
                "area": float(support_area[best_idx]),
                "novelty": float(best_new_area / max(float(support_area[best_idx]), 1.0)),
            }
        )

    selected_arr = np.asarray(selected, dtype=np.int64)
    diag = {
        "num_valid_candidates": int(np.count_nonzero(valid)),
        "num_selected": int(selected_arr.shape[0]),
        "selected_union_count": int(selected_union.sum()),
        "first_steps": step_records[:20],
    }
    return selected_arr, diag


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
    if scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: inconsistent prediction arrays")

    support_path = _tmp_path(root, args.score_pre_points_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    support_ids = np.load(support_path).astype(np.int64)
    features = _support_features(masks, support_ids)
    quality = _base_quality(scores, features, args)
    selected, select_diag = _greedy_select(
        support_masks=features["support_masks"],
        support_area=features["support_area"],
        quality=quality,
        args=args,
    )

    out_masks = masks[:, selected] if selected.size else np.zeros((masks.shape[0], 0), dtype=bool)
    out_classes = classes[selected] if selected.size else np.zeros((0,), dtype=np.int32)
    if selected.size:
        if bool(args.preserve_quality_score):
            selected_quality = quality[selected].astype(np.float32, copy=False)
            rank_bias = np.linspace(1.0, 0.0, selected.size, endpoint=False, dtype=np.float32)
            out_scores = (selected_quality + 0.001 * rank_bias).astype(np.float32)
        else:
            out_scores = np.linspace(1.0, 0.5, selected.size, endpoint=False, dtype=np.float32)
    else:
        out_scores = np.zeros((0,), dtype=np.float32)

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores,
        pred_classes=out_classes,
    )

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
    else:
        np.save(tmp_out, np.flatnonzero(out_masks.any(axis=1)).astype(np.int64))

    selected_support_area = features["support_area"][selected] if selected.size else np.zeros((0,), dtype=np.float64)
    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "score_pre_points_config": args.score_pre_points_config,
        "num_instances_before": int(masks.shape[1]),
        "num_score_support_points": int(support_ids.shape[0]),
        "support_area_mean": float(np.mean(features["support_area"])) if features["support_area"].size else 0.0,
        "quality_mean": float(np.mean(quality)) if quality.size else 0.0,
        "selected_support_area_mean": float(np.mean(selected_support_area)) if selected_support_area.size else 0.0,
        **select_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy support-novelty object selection for diagnostics.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--score-weight", type=float, default=0.55)
    parser.add_argument("--area-weight", type=float, default=0.15)
    parser.add_argument("--unique-weight", type=float, default=0.20)
    parser.add_argument("--conflict-weight", type=float, default=0.20)
    parser.add_argument("--new-area-weight", type=float, default=0.30)
    parser.add_argument("--novelty-weight", type=float, default=0.20)
    parser.add_argument("--overlap-penalty", type=float, default=0.30)
    parser.add_argument("--min-support-area", type=float, default=1.0)
    parser.add_argument("--min-new-area", type=float, default=10.0)
    parser.add_argument("--min-utility", type=float, default=-999.0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--suppress-overlapped", action="store_true")
    parser.add_argument("--suppress-overlap-ratio", type=float, default=0.90)
    parser.add_argument("--preserve-quality-score", action="store_true")
    parser.add_argument("--summary-root", default="outputs/greedy_support_select")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    aggregate: dict[str, float] = {}
    if rows:
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        for key in numeric_keys:
            aggregate[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))

    summary = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    summary_root = root / args.summary_root
    summary_root.mkdir(parents=True, exist_ok=True)
    with (summary_root / f"{args.output_config}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(_json_safe({"aggregate": aggregate}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
