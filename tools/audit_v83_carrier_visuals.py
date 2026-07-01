#!/usr/bin/env python3
"""Audit ACL2 v83 Phase3 carrier visual registrations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase3_carrier_alignment"
)
DEFAULT_SOURCE_AUDIT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation/visual_integrity_audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "visual_manifest.csv"
    review_path = args.root / "visual_review.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_csv(manifest_path)
    review = read_csv(review_path) if review_path.is_file() else []
    source_audit = read_json(args.source_audit) if args.source_audit.is_file() else {}
    source_gate = bool(source_audit.get("gate", {}).get("phase3_gate_pass"))

    required_panel_keys = [
        "true_route_panel",
        "qkv_head_layer_panel",
        "actual_vs_random_panel",
        "confidence_bin_panel",
    ]
    missing_files: list[dict[str, str]] = []
    for row in manifest:
        for key in required_panel_keys:
            value = row.get(key, "")
            if not value or not Path(value).is_file():
                missing_files.append(
                    {
                        "seq": row.get("seq", ""),
                        "prev_chunk": row.get("prev_chunk", ""),
                        "curr_chunk": row.get("curr_chunk", ""),
                        "key": key,
                        "path": value,
                    }
                )

    seqs = sorted({row.get("seq", "") for row in manifest if row.get("seq", "")})
    base_counts = Counter(row.get("base_case_type", "") for row in manifest)
    case_counts = Counter(row.get("case_type", "") for row in manifest)
    true_route_rows = sum(1 for row in manifest if truthy(row.get("has_actual_route_mask")))
    qkv_rows = sum(1 for row in manifest if truthy(row.get("has_qkv_maps")))
    actual_random_rows = sum(1 for row in manifest if truthy(row.get("actual_vs_random_difference_reviewed")))
    missing_overlay_count = sum(1 for row in manifest if truthy(row.get("missing_overlay")))
    reviewed_rows = sum(1 for row in review if str(row.get("review_status", "")).startswith("reviewed"))
    review_coverage = reviewed_rows / len(manifest) if manifest else 0.0
    good_or_false_positive = sum(
        1
        for row in manifest
        if row.get("base_case_type") == "good" or row.get("case_type") == "false_positive_semantic"
    )
    gate = {
        "visual_rows_ge_24": len(manifest) >= 24,
        "bad_rows_ge_12": base_counts.get("bad", 0) >= 12,
        "good_or_false_positive_rows_ge_12": good_or_false_positive >= 12,
        "seq_coverage_ge_3": len(seqs) >= 3,
        "referenced_visual_files_exist": not missing_files,
        "qkv_maps_present_all_rows": qkv_rows == len(manifest) and bool(manifest),
        "true_runtime_route_present_ge_90pct": (true_route_rows / len(manifest) if manifest else 0.0) >= 0.90,
        "actual_vs_random_difference_all_rows": actual_random_rows == len(manifest) and bool(manifest),
        "missing_overlay_count_eq_0": missing_overlay_count == 0,
        "review_coverage_ge_80pct": review_coverage >= 0.80,
        "source_v82_visual_audit_gate_pass": source_gate,
    }
    gate["visual_audit_gate_pass"] = all(gate.values())
    audit = {
        "schema": "acl2_v83_phase3_carrier_visual_audit_v1",
        "root": str(args.root),
        "source_audit": str(args.source_audit),
        "rows": len(manifest),
        "review_rows": len(review),
        "reviewed_rows": reviewed_rows,
        "review_coverage": review_coverage,
        "seq_coverage": seqs,
        "base_case_counts": dict(base_counts),
        "case_counts": dict(case_counts),
        "true_route_rows": true_route_rows,
        "qkv_rows": qkv_rows,
        "actual_vs_random_difference_rows": actual_random_rows,
        "missing_overlay_count": missing_overlay_count,
        "missing_files": missing_files,
        "source_audit_gate": source_audit.get("gate", {}),
        "gate": gate,
        "note": "This audit validates referenced visual artifacts; it does not decide semantic carrier specificity.",
    }
    write_json(args.root / "visual_integrity_audit.json", audit)
    print(json.dumps({"gate": gate, "out_path": str(args.root / "visual_integrity_audit.json")}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
