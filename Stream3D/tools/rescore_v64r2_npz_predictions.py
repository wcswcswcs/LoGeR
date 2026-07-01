from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _load_scene_names(seq_list: str) -> list[str]:
    with Path(seq_list).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _scores_from_masks(masks: np.ndarray, mode: str) -> np.ndarray:
    areas = np.asarray(masks, dtype=bool).sum(axis=0).astype(np.float32)
    if mode == "one":
        return np.ones_like(areas, dtype=np.float32)
    if mode == "area":
        return areas
    if mode == "inverse_area":
        return 1.0 / np.maximum(areas, 1.0)
    if mode == "sqrt_area":
        return np.sqrt(np.maximum(areas, 0.0)).astype(np.float32)
    raise ValueError(f"Unsupported score mode: {mode}")


def _copy_support(input_config: str, output_config: str) -> None:
    src = Path("data/TMP") / input_config
    dst = Path("data/TMP") / output_config
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*_pre_points.npy"):
        shutil.copy2(path, dst / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--score-mode", required=True, choices=["one", "area", "inverse_area", "sqrt_area"])
    parser.add_argument("--audit-root", default="outputs/audit/v64r2_rescore_predictions")
    args = parser.parse_args()

    input_pred_dir = Path("data/prediction") / f"{args.input_config}_class_agnostic"
    output_pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    _copy_support(args.input_config, args.output_config)

    summaries = []
    for seq_name in _load_scene_names(args.seq_list):
        src_path = input_pred_dir / f"{seq_name}.npz"
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        payload = np.load(src_path, allow_pickle=True)
        masks = np.asarray(payload["pred_masks"], dtype=bool)
        scores = _scores_from_masks(masks, args.score_mode)
        pred_classes = np.asarray(payload.get("pred_classes", np.zeros((masks.shape[1],), dtype=np.int32)))
        np.savez_compressed(
            output_pred_dir / f"{seq_name}.npz",
            pred_masks=masks,
            pred_score=scores.astype(np.float32),
            pred_classes=pred_classes.astype(np.int32),
        )
        areas = masks.sum(axis=0).astype(np.float64)
        summaries.append(
            {
                "seq_name": seq_name,
                "pred_count": int(masks.shape[1]),
                "area_mean": float(areas.mean()) if areas.size else 0.0,
                "area_median": float(np.median(areas)) if areas.size else 0.0,
                "score_mean": float(scores.mean()) if scores.size else 0.0,
                "score_median": float(np.median(scores)) if scores.size else 0.0,
            }
        )

    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        source_configs=[args.input_config],
        pre_points_policy="copied_from_input_config",
        support_policy="same_as_input_config",
        notes="Diagnostic-only rescore of existing predicted masks; pred_masks and pre_points are unchanged.",
        extra={
            "forbidden_for_method_table": True,
            "rescore_only": True,
            "rescore_mode": args.score_mode,
        },
    )
    write_prediction_manifest(args.output_config, manifest)

    audit_dir = Path(args.audit_root)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{args.output_config}_summary.json").write_text(
        json.dumps({"score_mode": args.score_mode, "summaries": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
