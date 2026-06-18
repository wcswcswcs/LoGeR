from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.frozen_feature_adapter import (
    FrozenFeatureAdapter,
    locate_default_dinov2_checkpoint,
    locate_default_radio_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["rgb_stats", "dinov2_timm", "radio_radseg"], default="rgb_stats")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--output-root", default="outputs/audit/v42_feature_adapter")
    args = parser.parse_args()

    if args.backend == "dinov2_timm":
        checkpoint = args.checkpoint or locate_default_dinov2_checkpoint()
    elif args.backend == "radio_radseg":
        checkpoint = args.checkpoint or locate_default_radio_checkpoint()
    else:
        checkpoint = args.checkpoint
    adapter = FrozenFeatureAdapter(
        backend=args.backend,
        device=args.device,
        checkpoint=checkpoint,
        radio_lang_model=args.radio_lang_model,
        radio_lang_align=bool(args.radio_lang_align),
    )
    frame = np.zeros((224, 320, 3), dtype=np.uint8) if args.backend == "radio_radseg" else np.zeros((64, 96, 3), dtype=np.uint8)
    frame[:, : frame.shape[1] // 2, 0] = 255
    frame[:, frame.shape[1] // 2 :, 1] = 255
    mask_a = np.zeros((64, 96), dtype=bool)
    mask_a = np.zeros(frame.shape[:2], dtype=bool)
    mask_a[:, : frame.shape[1] // 2] = True
    mask_b = np.zeros((64, 96), dtype=bool)
    mask_b = np.zeros(frame.shape[:2], dtype=bool)
    mask_b[:, frame.shape[1] // 2 :] = True
    fmap = adapter.extract_dense_features(frame)
    feat_a = adapter.pool_mask_feature(fmap, mask_a)
    feat_b = adapter.pool_mask_feature(fmap, mask_b)
    affinity = adapter.compute_token_affinity(feat_a, feat_b)
    rows = [
        {
            "backend": args.backend,
            "checkpoint": checkpoint or "",
            "radio_lang_model": args.radio_lang_model if args.backend == "radio_radseg" else "",
            "radio_lang_align": bool(args.radio_lang_align) if args.backend == "radio_radseg" else "",
            "feature_h": int(fmap.features.shape[0]),
            "feature_w": int(fmap.features.shape[1]),
            "feature_c": int(fmap.features.shape[2]),
            "image_h": int(fmap.image_height),
            "image_w": int(fmap.image_width),
            "patch_size": fmap.patch_size or "",
            "mask_a_norm": float(np.linalg.norm(feat_a)),
            "mask_b_norm": float(np.linalg.norm(feat_b)),
            "cross_mask_affinity": float(affinity),
            "boundary_contrast_a": float(adapter.compute_boundary_contrast(fmap, mask_a)),
        }
    ]
    summary = {
        "backend": args.backend,
        "checkpoint": checkpoint,
        "radio_lang_model": args.radio_lang_model if args.backend == "radio_radseg" else "",
        "radio_lang_align": bool(args.radio_lang_align) if args.backend == "radio_radseg" else "",
        "feature_shape_rows_csv": str(ROOT / args.output_root / "feature_shape_rows.csv"),
        "gate_pass": bool(fmap.features.ndim == 3 and feat_a.ndim == 1 and np.isfinite(affinity)),
        "rows": rows,
    }
    out = ROOT / args.output_root
    _write_csv(out / "feature_shape_rows.csv", rows)
    _write_json(out / "feature_smoke.json", summary)
    print(json.dumps({"feature_smoke": str(out / "feature_smoke.json"), "gate_pass": summary["gate_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
