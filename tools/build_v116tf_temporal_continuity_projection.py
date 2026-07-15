#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = ["label_ids", "role_ids", "risk", "stable", "confidence"]


def compute_continuity(label_ids: np.ndarray, confidence: np.ndarray, radius: int) -> np.ndarray:
    labels = np.asarray(label_ids)
    conf = np.asarray(confidence, dtype=np.float32)
    if labels.shape != conf.shape:
        raise ValueError(f"label/confidence shape mismatch: {labels.shape} vs {conf.shape}")
    total = np.zeros(labels.shape, dtype=np.float32)
    count = np.zeros(labels.shape, dtype=np.float32)
    for offset in range(1, radius + 1):
        forward = np.zeros(labels.shape, dtype=np.float32)
        backward = np.zeros(labels.shape, dtype=np.float32)
        valid_forward = np.zeros(labels.shape, dtype=np.float32)
        valid_backward = np.zeros(labels.shape, dtype=np.float32)
        forward[:-offset] = (labels[:-offset] == labels[offset:]).astype(np.float32)
        backward[offset:] = (labels[offset:] == labels[:-offset]).astype(np.float32)
        valid_forward[:-offset] = 1.0
        valid_backward[offset:] = 1.0
        total += forward + backward
        count += valid_forward + valid_backward
    continuity = total / np.maximum(count, 1.0)
    continuity *= np.clip(conf, 0.0, 1.0)
    return continuity.astype(np.float32)


def summarize(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": float(np.min(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "max": float(np.max(flat)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v116 temporal-continuity semantic projection sidecar.")
    parser.add_argument(
        "--source-root",
        default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence/semantic_projection",
    )
    parser.add_argument(
        "--output-root",
        default="results/acl2_v116tf_fast_semantic_causal_memory_influence/semantic_projection_temporal_continuity",
    )
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seqs = [seq.strip() for seq in args.seqs.split(",") if seq.strip()]

    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seqs": seqs,
        "radius": int(args.radius),
        "continuity_definition": "per-patch same-label fraction over +/-radius frames multiplied by semantic confidence",
        "seq_summaries": {},
    }
    for seq in seqs:
        for field in REQUIRED_FIELDS:
            src = source_root / f"seq{seq}_{field}.npy"
            dst = output_root / f"seq{seq}_{field}.npy"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
        label_ids = np.load(source_root / f"seq{seq}_label_ids.npy")
        confidence = np.load(source_root / f"seq{seq}_confidence.npy")
        continuity = compute_continuity(label_ids, confidence, radius=int(args.radius))
        np.save(output_root / f"seq{seq}_continuity.npy", continuity)
        summary["seq_summaries"][seq] = summarize(continuity)

    summary_path = output_root / "temporal_continuity_projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
