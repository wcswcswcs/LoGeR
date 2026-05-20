#!/usr/bin/env python3
"""Audit ACL2 v20 temporal support variants without running the model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.hybrid_memory_controller import _acl2_support_indices


VARIANTS = [
    "past_only",
    "full",
    "full_chunk",
    "full_chunk_no_overlap",
    "past_plus_future_light",
    "near12",
    "near24",
]


def _weights(num_frames: int, t: int, support: str, indices: Iterable[int]) -> Dict[int, float]:
    support = support.lower()
    idx = list(indices)
    if not idx:
        return {}
    if support in {"past_plus_future_light", "past_future_light", "past075_future025"}:
        past = [s for s in idx if s < t]
        future = [s for s in idx if s > t]
        out: Dict[int, float] = {}
        if past:
            for s in past:
                out[s] = 0.75 / len(past)
        if future:
            for s in future:
                out[s] = 0.25 / len(future)
        total = sum(out.values())
        if total > 0:
            out = {s: w / total for s, w in out.items()}
        return out
    w = 1.0 / len(idx)
    return {s: w for s in idx}


def _variant_note(support: str) -> str:
    if support == "full_chunk":
        return "explicit alias of full support inside current chunk"
    if support == "full_chunk_no_overlap":
        return "falls back to full_chunk because HMC cue builder has no external overlap metadata"
    if support == "past_plus_future_light":
        return "weighted centroid: 0.75 past + 0.25 future within current chunk"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--chunk-id", action="append", type=int, default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--support", action="append", default=[])
    args = parser.parse_args()

    variants = args.support or VARIANTS
    chunks = args.chunk_id or [6, 10]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for chunk_id in chunks:
        for support in variants:
            for t in range(args.num_frames):
                indices = _acl2_support_indices(args.num_frames, t, support)
                weights = _weights(args.num_frames, t, support, indices)
                rows.append(
                    {
                        "chunk_id": int(chunk_id),
                        "local_frame": int(t),
                        "support": support,
                        "support_indices": indices,
                        "support_weights": weights,
                        "support_count": len(indices),
                        "uses_future": any(s > t for s in indices),
                        "uses_past": any(s < t for s in indices),
                        "note": _variant_note(support),
                    }
                )

    with (out_dir / "support_indices.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_rows: List[Dict[str, object]] = []
    for support in variants:
        sub = [r for r in rows if r["support"] == support]
        counts = [int(r["support_count"]) for r in sub]
        future_count = sum(1 for r in sub if r["uses_future"])
        past_count = sum(1 for r in sub if r["uses_past"])
        summary_rows.append(
            {
                "support": support,
                "num_rows": len(sub),
                "support_count_min": min(counts) if counts else 0,
                "support_count_max": max(counts) if counts else 0,
                "support_count_mean": sum(counts) / len(counts) if counts else 0.0,
                "frames_with_future": future_count,
                "frames_with_past": past_count,
                "note": _variant_note(support),
            }
        )

    with (out_dir / "support_index_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "support_index_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ACL2 v20 Support Index Audit",
        "",
        f"num_frames = `{args.num_frames}`",
        "",
        "| Support | Count min | Count max | Count mean | Future frames | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['support']}` | {row['support_count_min']} | {row['support_count_max']} | "
            f"{float(row['support_count_mean']):.3f} | {row['frames_with_future']} | {row['note']} |"
        )
    (out_dir / "support_index_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'support_indices.jsonl'}")


if __name__ == "__main__":
    main()
