#!/usr/bin/env python3
"""Audit reset-window parent snapshot coverage for ACL2 v33.

The v33 plan needs reset-relative oracle windows, not hard-coded KITTI01
chunk ids. This script records which H9/C9 parent state and merge snapshots
are actually present so downstream trigger-learning decisions are auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _chunks_in_dir(path: Path) -> set[int]:
    chunks: set[int] = set()
    if not path.exists():
        return chunks
    for item in path.glob("chunk_*_input.pt"):
        match = re.match(r"chunk_(\d+)_input\.pt$", item.name)
        if match:
            chunks.add(int(match.group(1)))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--snapshot-root",
        default=(
            "results/kitti01_hmc_v2/"
            "acl2_v16_ttt_causalfork_candidatebank_target25/"
            "phase1_causalfork"
        ),
    )
    parser.add_argument(
        "--warm-snapshot-dir",
        default="/tmp/loger_v23_warm/snapshots",
    )
    parser.add_argument(
        "--results-root",
        required=True,
    )
    parser.add_argument(
        "--expected-reset-starts",
        default="0,5,10,15,20,25,30",
    )
    parser.add_argument(
        "--parents",
        default="H9_P0_V16_R2,C9_P0_V16_R2",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    snapshot_root = (repo_root / args.snapshot_root).resolve()
    state_root = snapshot_root / "state_snapshots"
    merge_root = snapshot_root / "merge_state_snapshots"
    warm_dir = Path(args.warm_snapshot_dir)
    results_root = (repo_root / args.results_root).resolve()
    out_dir = results_root / "snapshot_coverage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = [int(x) for x in args.expected_reset_starts.split(",") if x]
    parents = [x for x in args.parents.split(",") if x]

    rows: list[dict[str, object]] = []
    parent_available: dict[str, list[int]] = {}
    parent_expected_hits: dict[str, list[int]] = {}
    for parent in parents:
        state_chunks = _chunks_in_dir(state_root / parent)
        merge_chunks = _chunks_in_dir(merge_root / parent)
        both = sorted(state_chunks & merge_chunks)
        parent_available[parent] = both
        parent_expected_hits[parent] = [chunk for chunk in expected if chunk in both]
        for chunk in sorted(set(expected) | set(both)):
            rows.append(
                {
                    "parent": parent,
                    "chunk": chunk,
                    "expected_reset_start": chunk in expected,
                    "state_snapshot": chunk in state_chunks,
                    "merge_snapshot": chunk in merge_chunks,
                    "usable_parent_snapshot": chunk in state_chunks and chunk in merge_chunks,
                }
            )

    common_available = sorted(
        set.intersection(*(set(chunks) for chunks in parent_available.values()))
        if parent_available
        else set()
    )
    common_expected_hits = [chunk for chunk in expected if chunk in common_available]
    warm_chunks = sorted(_chunks_in_dir(warm_dir))

    summary = {
        "snapshot_root": str(snapshot_root),
        "warm_snapshot_dir": str(warm_dir),
        "expected_reset_starts": expected,
        "parents": parents,
        "parent_available_chunks": parent_available,
        "parent_expected_reset_hits": parent_expected_hits,
        "common_available_chunks": common_available,
        "common_expected_reset_hits": common_expected_hits,
        "warm_snapshot_chunks": warm_chunks,
        "complete_expected_reset_coverage": len(common_expected_hits) == len(expected),
        "available_window_count": len(common_available),
        "expected_reset_hit_count": len(common_expected_hits),
        "trigger_training_allowed": len(common_expected_hits) >= len(expected),
        "reason_if_blocked": (
            "incomplete_common_H9_C9_parent_snapshots_for_expected_reset_groups"
            if len(common_expected_hits) < len(expected)
            else ""
        ),
    }

    with (out_dir / "snapshot_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parent",
                "chunk",
                "expected_reset_start",
                "state_snapshot",
                "merge_snapshot",
                "usable_parent_snapshot",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "snapshot_coverage_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# v33 Snapshot Coverage Audit",
        "",
        f"Expected reset starts: {expected}",
        f"Common available chunks: {common_available}",
        f"Common expected reset hits: {common_expected_hits}",
        f"Complete expected reset coverage: {summary['complete_expected_reset_coverage']}",
        f"Trigger training allowed: {summary['trigger_training_allowed']}",
        "",
        "Per parent available chunks:",
    ]
    for parent, chunks in parent_available.items():
        report.append(f"- {parent}: {chunks}")
    report.append("")
    report.append(f"Warm snapshot chunks: {warm_chunks}")
    if summary["reason_if_blocked"]:
        report.append("")
        report.append(f"Blocked reason: {summary['reason_if_blocked']}")
    (out_dir / "snapshot_coverage_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete_expected_reset_coverage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
