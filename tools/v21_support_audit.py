#!/usr/bin/env python3
"""Audit ACL2 v21 temporal support semantics without running the model."""

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

from loger.pipeline.hybrid_memory_controller import _acl2_support_indices  # noqa: E402


VARIANTS = [
    "past_only",
    "full_chunk_true",
    "full_chunk_no_overlap",
    "past_plus_near_future12",
    "past_plus_future_light_real",
]


def _support_weights(t: int, support: str, indices: Iterable[int]) -> Dict[int, float]:
    idx = list(indices)
    if not idx:
        return {}
    if support in {"past_plus_future_light", "past_plus_future_light_real", "past_future_light", "past075_future025"}:
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
            return {s: w / total for s, w in out.items()}
        return {}
    w = 1.0 / len(idx)
    return {s: w for s in idx}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--chunk-id", action="append", type=int, default=[])
    parser.add_argument("--support", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    num_frames = int(args.num_frames)
    overlap_frames = max(int(args.overlap_frames), 0)
    chunks = args.chunk_id or [6, 10, 16]
    supports = args.support or VARIANTS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seam = set(range(min(overlap_frames, num_frames)))
    if overlap_frames > 0:
        seam.update(range(max(num_frames - overlap_frames, 0), num_frames))

    frame_rows: List[Dict[str, object]] = []
    for chunk_id in chunks:
        for support in supports:
            for t in range(num_frames):
                indices = _acl2_support_indices(
                    num_frames,
                    t,
                    support,
                    overlap_frames=overlap_frames,
                )
                weights = _support_weights(t, support, indices)
                future_indices = [s for s in indices if s > t]
                past_indices = [s for s in indices if s < t]
                seam_indices = [s for s in indices if s in seam]
                frame_rows.append({
                    "chunk_id": int(chunk_id),
                    "local_frame": int(t),
                    "support": str(support),
                    "support_indices": indices,
                    "support_weights": weights,
                    "support_count": int(len(indices)),
                    "past_support_count": int(len(past_indices)),
                    "future_support_count": int(len(future_indices)),
                    "future_support_ratio": float(len(future_indices) / max(len(indices), 1)),
                    "support_weight_past_mass": float(sum(w for s, w in weights.items() if s < t)),
                    "support_weight_future_mass": float(sum(w for s, w in weights.items() if s > t)),
                    "overlap_seam_indices_in_support": seam_indices,
                    "overlap_seam_support_count": int(len(seam_indices)),
                    "is_query_overlap_seam": bool(t in seam),
                })

    with (out_dir / "support_index_by_frame.jsonl").open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_rows: List[Dict[str, object]] = []
    future_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []
    for support in supports:
        sub = [r for r in frame_rows if r["support"] == support]
        counts = [int(r["support_count"]) for r in sub]
        future_ratios = [float(r["future_support_ratio"]) for r in sub]
        past_mass = [float(r["support_weight_past_mass"]) for r in sub]
        future_mass = [float(r["support_weight_future_mass"]) for r in sub]
        seam_support_total = sum(int(r["overlap_seam_support_count"]) for r in sub)
        summary_rows.append({
            "support": support,
            "num_rows": len(sub),
            "support_count_min": min(counts) if counts else 0,
            "support_count_max": max(counts) if counts else 0,
            "support_count_mean": sum(counts) / len(counts) if counts else 0.0,
            "future_support_ratio_mean": sum(future_ratios) / len(future_ratios) if future_ratios else 0.0,
            "weighted_past_mass_mean": sum(past_mass) / len(past_mass) if past_mass else 0.0,
            "weighted_future_mass_mean": sum(future_mass) / len(future_mass) if future_mass else 0.0,
            "overlap_seam_support_total": seam_support_total,
            "overlap_exclusion_pass": bool(
                support not in {"full_chunk_no_overlap", "full_no_overlap", "no_overlap", "no_overlap_true"}
                or seam_support_total == 0
            ),
            "fallback_forbidden": support in {"full_chunk_no_overlap", "full_no_overlap", "no_overlap", "no_overlap_true"},
        })
        future_rows.append({
            "support": support,
            "weighted_past_mass_mean": summary_rows[-1]["weighted_past_mass_mean"],
            "weighted_future_mass_mean": summary_rows[-1]["weighted_future_mass_mean"],
            "future_support_ratio_mean": summary_rows[-1]["future_support_ratio_mean"],
        })
        overlap_rows.append({
            "support": support,
            "overlap_frames": overlap_frames,
            "seam_frames": sorted(seam),
            "overlap_seam_support_total": seam_support_total,
            "overlap_exclusion_pass": summary_rows[-1]["overlap_exclusion_pass"],
        })

    _write_csv(out_dir / "support_index_summary.csv", summary_rows)
    _write_csv(out_dir / "support_future_mass.csv", future_rows)
    _write_csv(out_dir / "support_overlap_exclusion_check.csv", overlap_rows)
    (out_dir / "support_index_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ACL2 v21 Support Audit",
        "",
        f"num_frames = `{num_frames}`",
        f"overlap_frames = `{overlap_frames}`",
        "",
        "| Support | Count mean | Future ratio | Future weight mass | Seam support | No-overlap pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['support']}` | {float(row['support_count_mean']):.3f} | "
            f"{float(row['future_support_ratio_mean']):.3f} | "
            f"{float(row['weighted_future_mass_mean']):.3f} | "
            f"{int(row['overlap_seam_support_total'])} | `{str(row['overlap_exclusion_pass']).lower()}` |"
        )
    (out_dir / "support_index_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'support_index_summary.csv'}")


if __name__ == "__main__":
    main()
