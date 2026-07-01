#!/usr/bin/env python3
"""Audit v82 Phase3 true-route visual artifact integrity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation"
)
DEFAULT_ROUTE_ROOT = DEFAULT_ROOT / "route_dump"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _route_dump_counts(route_root: Path) -> dict[str, int]:
    return {
        "trajectory_txt": len(list(route_root.glob("seq*/chunk*/P9_*/01.txt"))),
        "runtime_route_pt": len(list(route_root.glob("seq*/chunk*/P9_*/swa_overlap_feature_maps/*.pt"))),
        "metrics_json": len(list(route_root.glob("seq*/phase9_swa_cache_value_metrics.json"))),
        "decision_json": len(list(route_root.glob("seq*/phase9_swa_cache_value_decision.json"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    args = parser.parse_args()

    manifest = _read_csv(args.root / "visual_manifest.csv")
    review_path = args.root / "visual_review.csv"
    review = _read_csv(review_path) if review_path.is_file() else []
    base_counts = Counter(row.get("base_case_type", "") for row in manifest)
    case_counts = Counter(row.get("case_type", "") for row in manifest)
    seqs = sorted({row.get("seq", "") for row in manifest if row.get("seq", "")})
    true_route_rows = sum(1 for row in manifest if _truthy(row.get("has_actual_route_mask")))
    qkv_rows = sum(1 for row in manifest if _truthy(row.get("has_qkv_maps")))
    actual_random_rows = sum(1 for row in manifest if _truthy(row.get("actual_vs_random_difference_reviewed")))
    missing_overlay_count = sum(1 for row in manifest if _truthy(row.get("missing_overlay")))
    reviewed_rows = sum(1 for row in review if str(row.get("review_status", "")).startswith("reviewed"))
    review_coverage = reviewed_rows / len(manifest) if manifest else 0.0
    good_or_false_positive = sum(
        1
        for row in manifest
        if row.get("base_case_type") == "good" or row.get("case_type") == "false_positive_semantic"
    )
    files_exist = all(
        Path(row.get(key, "")).is_file()
        for row in manifest
        for key in ["true_route_panel", "qkv_head_layer_panel", "actual_vs_random_panel", "confidence_bin_panel"]
    )
    gate = {
        "visual_rows_ge_24": len(manifest) >= 24,
        "bad_review_rows_ge_12": base_counts.get("bad", 0) >= 12,
        "good_or_false_positive_review_rows_ge_12": good_or_false_positive >= 12,
        "seq_coverage_ge_3": len(seqs) >= 3,
        "visual_files_exist": files_exist,
        "qkv_maps_present_all_rows": qkv_rows == len(manifest) and bool(manifest),
        "true_runtime_route_present_ge_90pct": (true_route_rows / len(manifest) if manifest else 0.0) >= 0.90,
        "actual_vs_random_difference_all_rows": actual_random_rows == len(manifest) and bool(manifest),
        "missing_overlay_count_eq_0": missing_overlay_count == 0,
        "review_coverage_ge_80pct": review_coverage >= 0.80,
    }
    gate["phase3_gate_pass"] = all(gate.values())
    audit = {
        "schema": "acl2_v82_swa_true_route_visual_artifact_audit_v1",
        "root": str(args.root),
        "route_root": str(args.route_root),
        "rows": len(manifest),
        "review_rows": len(review),
        "reviewed_rows": reviewed_rows,
        "review_coverage": review_coverage,
        "case_counts": dict(case_counts),
        "base_case_counts": dict(base_counts),
        "seq_coverage": seqs,
        "true_route_rows": true_route_rows,
        "qkv_rows": qkv_rows,
        "actual_vs_random_difference_rows": actual_random_rows,
        "missing_overlay_count": missing_overlay_count,
        "route_dump_counts": _route_dump_counts(args.route_root),
        "gate": gate,
        "note": "True route requires runtime_swa_overlap_feature_not_qk_proxy=True in both P9_40 and P9_6 route tensors.",
    }
    out = args.root / "visual_integrity_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "out_path": str(out)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
