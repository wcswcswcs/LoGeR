#!/usr/bin/env python3
"""Register existing true-route visual panels for ACL2 v83 Phase3.

This tool does not synthesize or copy images. It creates the v83 Phase3
manifest by pointing to already audited v82 runtime SWA route/QKV panels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SOURCE_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase3_carrier_alignment"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    source_manifest = args.source_root / "visual_manifest.csv"
    source_review = args.source_root / "visual_review.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    for name in ["read_qk_panels", "swa_qkv_panels", "merge_boundary_panels", "ttt_write_panels"]:
        (args.out_dir / name).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    source_rows = read_csv(source_manifest)
    for row in source_rows:
        out = dict(row)
        out.update(
            {
                "v83_panel_registration": "reuse_audited_v82_runtime_visual_panel",
                "v83_source_root": str(args.source_root),
                "read_qk_panel": "",
                "swa_qkv_panel": row.get("qkv_head_layer_panel", ""),
                "merge_boundary_panel": "",
                "ttt_write_panel": "",
                "v83_note": (
                    "No new bitmap was generated. This v83 manifest references the "
                    "audited v82 runtime SWA true-route/QKV/actual-vs-random panels."
                ),
            }
        )
        manifest_rows.append(out)

    review_rows: list[dict[str, Any]]
    if source_review.is_file():
        review_rows = [dict(row) for row in read_csv(source_review)]
        for row in review_rows:
            row["v83_review_basis"] = "imported_from_v82_true_route_visual_review"
            row["v83_review_note"] = "Visual evidence reused as an audited source artifact; no new panel generated."
    else:
        review_rows = [
            {
                "seq": row.get("seq", ""),
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": row.get("curr_chunk", ""),
                "case_type": row.get("case_type", ""),
                "base_case_type": row.get("base_case_type", ""),
                "review_status": (
                    "reviewed_true_route_qkv_random"
                    if truthy(row.get("has_actual_route_mask"))
                    and truthy(row.get("has_qkv_maps"))
                    and truthy(row.get("actual_vs_random_difference_reviewed"))
                    else "blocker_visual_artifact_incomplete"
                ),
                "v83_review_basis": "generated_from_v82_manifest_fields",
            }
            for row in source_rows
        ]

    write_csv(args.out_dir / "visual_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "visual_review.csv", review_rows)

    seqs = sorted({row.get("seq", "") for row in manifest_rows if row.get("seq", "")})
    status_counts = Counter(row.get("review_status", "") for row in review_rows)
    summary = {
        "schema": "acl2_v83_phase3_visual_panel_registration_v1",
        "source_root": str(args.source_root),
        "out_dir": str(args.out_dir),
        "rows": len(manifest_rows),
        "review_rows": len(review_rows),
        "seq_coverage": seqs,
        "review_status_counts": dict(status_counts),
        "new_bitmap_panels_generated": 0,
        "note": "v83 Phase3 visual manifest references existing v82 audited runtime panels.",
    }
    write_json(args.out_dir / "visual_panel_registration_summary.json", summary)
    (args.out_dir / "visual_insight.md").write_text(
        "# v83 Phase3 Visual Insight\n\n"
        f"rows: {summary['rows']}\n\n"
        f"review_rows: {summary['review_rows']}\n\n"
        f"seq_coverage: {summary['seq_coverage']}\n\n"
        f"new_bitmap_panels_generated: {summary['new_bitmap_panels_generated']}\n\n"
        "conclusion: Existing v82 runtime SWA true-route/QKV/actual-vs-random panels were registered "
        "as v83 Phase3 visual evidence. This only proves visual artifact availability; carrier "
        "specificity is decided separately by carrier_alignment_summary.json.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
