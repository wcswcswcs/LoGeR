from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    return root / "data" / "prediction" / f"{config}{suffix}" / f"{seq_name}.npz"


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _load_prediction(root: Path, config: str, suffix: str, seq_name: str) -> dict[str, np.ndarray]:
    path = _prediction_path(root, config, suffix, seq_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction: {path}")
    with np.load(path) as data:
        return {
            "pred_masks": data["pred_masks"].astype(bool, copy=False),
            "pred_score": data["pred_score"].astype(np.float32, copy=False),
            "pred_classes": data["pred_classes"].astype(np.int32, copy=False),
        }


def _score_array_or_preserve(count: int, value: float, source_scores: np.ndarray) -> np.ndarray:
    if float(value) < 0.0 and source_scores.shape[0] == count:
        return source_scores.astype(np.float32, copy=False)
    return np.full((int(count),), float(value), dtype=np.float32)


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    primary = _load_prediction(root, args.primary_config, args.pred_suffix, seq_name)
    secondary = _load_prediction(root, args.secondary_config, args.pred_suffix, seq_name)
    primary_masks = primary["pred_masks"]
    secondary_masks = secondary["pred_masks"]
    if primary_masks.shape[0] != secondary_masks.shape[0]:
        raise ValueError(
            f"{seq_name}: vertex count mismatch primary={primary_masks.shape[0]}, "
            f"secondary={secondary_masks.shape[0]}"
        )

    support_path = _tmp_path(root, args.support_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(f"Missing support pre_points: {support_path}")
    support_ids = np.load(support_path).astype(np.int64)
    if support_ids.size:
        if int(support_ids.min()) < 0 or int(support_ids.max()) >= primary_masks.shape[0]:
            raise ValueError(
                f"{seq_name}: support ids outside prediction range: "
                f"min={int(support_ids.min())}, max={int(support_ids.max())}, vertices={primary_masks.shape[0]}"
            )
        primary_support = primary_masks[support_ids, :]
        secondary_support = secondary_masks[support_ids, :]
    else:
        primary_support = np.zeros((0, primary_masks.shape[1]), dtype=bool)
        secondary_support = np.zeros((0, secondary_masks.shape[1]), dtype=bool)

    primary_union_support = np.any(primary_support, axis=1) if primary_support.shape[1] else np.zeros((support_ids.shape[0],), dtype=bool)
    uncovered_support = ~primary_union_support
    residual_masks: list[np.ndarray] = []
    residual_scores: list[float] = []
    residual_classes: list[int] = []
    residual_areas: list[float] = []
    support_areas: list[float] = []
    residual_ratios: list[float] = []
    dropped_empty = 0
    dropped_small = 0
    dropped_ratio = 0

    for idx in range(secondary_masks.shape[1]):
        sec_support = secondary_support[:, idx] if secondary_support.shape[0] else np.zeros((0,), dtype=bool)
        support_area = int(np.count_nonzero(sec_support))
        if support_area <= 0:
            dropped_empty += 1
            continue
        residual_support = sec_support & uncovered_support
        residual_area = int(np.count_nonzero(residual_support))
        residual_ratio = residual_area / max(float(support_area), 1.0)
        if residual_area < int(args.min_residual_area):
            dropped_small += 1
            continue
        if residual_ratio < float(args.min_residual_ratio):
            dropped_ratio += 1
            continue

        if args.secondary_mode == "residual_support":
            out_mask = np.zeros((secondary_masks.shape[0],), dtype=bool)
            out_mask[support_ids[residual_support]] = True
        elif args.secondary_mode == "support_full":
            out_mask = np.zeros((secondary_masks.shape[0],), dtype=bool)
            out_mask[support_ids[sec_support]] = True
        elif args.secondary_mode == "full":
            out_mask = secondary_masks[:, idx].copy()
        else:
            raise ValueError(f"Unsupported secondary mode: {args.secondary_mode}")

        if int(np.count_nonzero(out_mask)) <= 0:
            dropped_empty += 1
            continue
        residual_masks.append(out_mask)
        residual_scores.append(float(args.secondary_score))
        residual_classes.append(int(secondary["pred_classes"][idx]) if idx < secondary["pred_classes"].shape[0] else 0)
        residual_areas.append(float(residual_area))
        support_areas.append(float(support_area))
        residual_ratios.append(float(residual_ratio))

    if residual_masks:
        secondary_out = np.stack(residual_masks, axis=1).astype(bool, copy=False)
        secondary_scores = np.asarray(residual_scores, dtype=np.float32)
        secondary_classes = np.asarray(residual_classes, dtype=np.int32)
        out_masks = np.concatenate([primary_masks, secondary_out], axis=1)
        out_scores = np.concatenate(
            [
                _score_array_or_preserve(primary_masks.shape[1], args.primary_score, primary["pred_score"]),
                secondary_scores,
            ],
            axis=0,
        )
        out_classes = np.concatenate(
            [
                primary["pred_classes"].astype(np.int32, copy=False),
                secondary_classes,
            ],
            axis=0,
        )
    else:
        out_masks = primary_masks.copy()
        out_scores = _score_array_or_preserve(primary_masks.shape[1], args.primary_score, primary["pred_score"])
        out_classes = primary["pred_classes"].astype(np.int32, copy=False)

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores.astype(np.float32, copy=False),
        pred_classes=out_classes.astype(np.int32, copy=False),
    )

    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_out_dir / f"{seq_name}_pre_points.npy", support_ids)

    out_support = out_masks[support_ids, :] if support_ids.size else np.zeros((0, out_masks.shape[1]), dtype=bool)
    output_support_union = int(np.count_nonzero(np.any(out_support, axis=1))) if out_support.shape[1] else 0
    output_support_conflict = int(np.count_nonzero(out_support.sum(axis=1) > 1)) if out_support.shape[1] else 0
    return {
        "seq_name": seq_name,
        "primary_config": args.primary_config,
        "secondary_config": args.secondary_config,
        "output_config": args.output_config,
        "support_config": args.support_config,
        "secondary_mode": args.secondary_mode,
        "num_support_points": float(support_ids.shape[0]),
        "num_primary_instances": float(primary_masks.shape[1]),
        "num_secondary_instances": float(secondary_masks.shape[1]),
        "num_residual_instances": float(len(residual_masks)),
        "num_output_instances": float(out_masks.shape[1]),
        "primary_support_union": float(np.count_nonzero(primary_union_support)),
        "uncovered_support_count": float(np.count_nonzero(uncovered_support)),
        "output_support_union": float(output_support_union),
        "output_support_conflict_points": float(output_support_conflict),
        "output_support_conflict_ratio": float(output_support_conflict / max(output_support_union, 1)),
        "dropped_empty": float(dropped_empty),
        "dropped_small": float(dropped_small),
        "dropped_ratio": float(dropped_ratio),
        "mean_secondary_support_area": float(np.mean(support_areas)) if support_areas else 0.0,
        "mean_residual_area": float(np.mean(residual_areas)) if residual_areas else 0.0,
        "mean_residual_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
    }


def aggregate(rows: list[dict[str, float | str]], args: argparse.Namespace) -> dict[str, float | str]:
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    )
    means: dict[str, float] = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            means[f"mean_{key}"] = float(np.mean(vals))
    return {
        "primary_config": args.primary_config,
        "secondary_config": args.secondary_config,
        "output_config": args.output_config,
        "support_config": args.support_config,
        "secondary_mode": args.secondary_mode,
        "primary_score": float(args.primary_score),
        "secondary_score": float(args.secondary_score),
        "min_residual_area": int(args.min_residual_area),
        "min_residual_ratio": float(args.min_residual_ratio),
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse primary predictions with secondary residual recall only.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--secondary-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--support-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--primary-score", type=float, default=-1.0)
    parser.add_argument("--secondary-score", type=float, default=0.005)
    parser.add_argument("--min-residual-area", type=int, default=10)
    parser.add_argument("--min-residual-ratio", type=float, default=0.01)
    parser.add_argument(
        "--secondary-mode",
        default="residual_support",
        choices=["residual_support", "support_full", "full"],
        help="what mask to output for a secondary candidate after it passes residual tests",
    )
    parser.add_argument("--summary-root", default="outputs/residual_recall_fuse_v4_1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[residual-recall-fuse] wrote {out_path}")


if __name__ == "__main__":
    main()
