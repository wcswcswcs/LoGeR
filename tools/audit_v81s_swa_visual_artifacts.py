#!/usr/bin/env python3
"""Audit v81S S3 visual artifact completeness."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_visual_confirmation"
)
DEFAULT_ROUTE_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_action_route_smoke"
)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _route_smoke_pairs(route_root: Path) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    if not route_root.is_dir():
        return pairs
    for path in route_root.glob("seq*/chunk*/P9_*/swa_overlap_feature_maps/*.pt"):
        parts = path.parts
        seq = ""
        chunk = None
        for part in parts:
            if part.startswith("seq") and len(part) >= 5:
                seq = part[3:5]
            if part.startswith("chunk") and len(part) >= 7 and part[5:7].isdigit():
                try:
                    chunk = int(part[5:7])
                except ValueError:
                    pass
        if seq and chunk is not None:
            pairs.add((seq, chunk))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    args = parser.parse_args()

    manifest_path = args.root / "visual_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_counts = Counter(row.get("case_type", "") for row in rows)
    seqs = sorted({row.get("seq", "") for row in rows if row.get("seq", "")})
    file_exists = all(Path(row.get("visual_file", "")).is_file() for row in rows)
    qkv_ok = all(_truthy(row.get("has_qkv_maps")) for row in rows)
    residual_ok = all(_truthy(row.get("has_overlap_residual_map")) for row in rows)
    route_proxy_ok = all(_truthy(row.get("has_route_proxy_mask")) for row in rows)
    manifest_true_route_ok = all(_truthy(row.get("has_actual_route_mask")) for row in rows)
    manifest_pairs: set[tuple[str, int]] = set()
    for row in rows:
        try:
            manifest_pairs.add((str(row.get("seq", "")), int(row.get("curr_chunk", ""))))
        except ValueError:
            continue
    route_pairs = _route_smoke_pairs(args.route_root)
    route_smoke_present = bool(route_pairs)
    route_smoke_full_visual_coverage = bool(manifest_pairs and manifest_pairs.issubset(route_pairs))
    true_route_ok = bool(manifest_true_route_ok or route_smoke_full_visual_coverage)
    gate = {
        "visual_rows_ge_24": len(rows) >= 24,
        "bad_ge_12": case_counts.get("bad", 0) >= 12,
        "good_ge_12": case_counts.get("good", 0) >= 12,
        "coverage_ge_3": len(seqs) >= 3,
        "visual_files_exist": file_exists,
        "qkv_maps_present": qkv_ok,
        "overlap_residual_maps_present": residual_ok,
        "route_proxy_masks_present": route_proxy_ok,
        "true_swa_action_route_masks_present": true_route_ok,
        "true_swa_action_route_smoke_present": route_smoke_present,
        "true_swa_action_route_smoke_full_visual_coverage": route_smoke_full_visual_coverage,
    }
    gate["phaseS3_visual_artifact_gate_pass"] = all(
        v
        for k, v in gate.items()
        if k
        not in {
            "true_swa_action_route_masks_present",
            "true_swa_action_route_smoke_present",
            "true_swa_action_route_smoke_full_visual_coverage",
        }
    )
    gate["phaseS3_gate_pass"] = bool(gate["phaseS3_visual_artifact_gate_pass"] and gate["true_swa_action_route_masks_present"])
    audit = {
        "schema": "acl2_v81s_swa_visual_artifact_audit_v1",
        "root": str(args.root),
        "route_root": str(args.route_root),
        "rows": len(rows),
        "case_counts": dict(case_counts),
        "seq_coverage": seqs,
        "visual_manifest_pair_count": len(manifest_pairs),
        "route_smoke_pair_count": len(route_pairs),
        "route_smoke_pairs": [f"seq{seq}_chunk{chunk:03d}" for seq, chunk in sorted(route_pairs)],
        "gate": gate,
        "note": (
            "phaseS3_gate_pass remains false unless true route masks cover the visual-manifest rows; "
            "partial route smoke is recorded separately."
        ),
    }
    out = args.root / "visual_integrity_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "out_path": str(out)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
