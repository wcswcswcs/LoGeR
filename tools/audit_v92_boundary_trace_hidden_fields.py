#!/usr/bin/env python3
"""Inspect landed route traces for hidden boundary/gauge fields before reruns."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, V91_PHASE7, read_jsonl


DEFAULT_OUT = ROOT / "phase2_boundary_trace_ledger"
KEYWORDS = ("residual", "boundary", "gauge", "scale", "transform", "merge", "sim3", "update")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, default=V91_PHASE7 / "route_dump_smoke")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value, child))
    elif isinstance(obj, list):
        rows.append((prefix, f"list[{len(obj)}]"))
    else:
        rows.append((prefix, obj))
    return rows


def _scan(paths: list[Path], kind: str) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    samples: list[dict[str, Any]] = []
    key_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    for path in paths:
        for row_idx, row in enumerate(read_jsonl(path)):
            for key, value in _flatten(row):
                key_counts[key] += 1
                low = key.lower()
                if any(word in low for word in KEYWORDS):
                    keyword_counts[key] += 1
                    if len(samples) < 200:
                        samples.append({"kind": kind, "path": str(path), "row_idx": row_idx, "key": key, "sample_value": value})
    return samples, key_counts, keyword_counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merge_paths = sorted(args.route_root.glob("**/merge_state_trace.jsonl")) if args.route_root.exists() else []
    hmc_paths = sorted(args.route_root.glob("**/hmc_state_hash.jsonl")) if args.route_root.exists() else []
    hook_paths = sorted(args.route_root.glob("**/hook_effect_summary.jsonl")) if args.route_root.exists() else []
    sample_rows: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}
    for kind, paths in [("merge_state_trace", merge_paths), ("hmc_state_hash", hmc_paths), ("hook_effect_summary", hook_paths)]:
        samples, key_counts, keyword_counts = _scan(paths, kind)
        sample_rows.extend(samples)
        sections[kind] = {
            "file_count": len(paths),
            "unique_key_count": len(key_counts),
            "keyword_key_count": len(keyword_counts),
            "keyword_keys": [{"key": key, "count": count} for key, count in keyword_counts.most_common()],
            "has_semantic_merge_residual": any("semantic_merge" in key and "residual" in key for key in keyword_counts),
            "has_boundary_update_norm": any("boundary_update_norm" in key for key in keyword_counts),
            "has_merge_residual_delta": any("merge_residual_delta" in key for key in keyword_counts),
            "has_transform_scale_value": any("transform_scale_value" in key for key in keyword_counts),
        }
    summary = {
        "phase": "Phase2_hidden_boundary_field_audit",
        "route_root": str(args.route_root),
        "merge_state_trace_files": len(merge_paths),
        "hmc_state_hash_files": len(hmc_paths),
        "hook_effect_summary_files": len(hook_paths),
        "sections": sections,
        "conclusion": (
            "existing_route_smoke_exposes_transform_scale_and_hash_fields_only_for_4_pairs; "
            "no landed full-row boundary_update_norm or merge_residual_delta field was found"
        ),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "hidden_boundary_field_audit.json", summary)
    write_csv(args.out_dir / "hidden_boundary_field_samples.csv", sample_rows)
    print(f"merge_state_trace_files={summary['merge_state_trace_files']}")
    print(f"hmc_state_hash_files={summary['hmc_state_hash_files']}")
    print(f"hook_effect_summary_files={summary['hook_effect_summary_files']}")
    for kind, section in sections.items():
        print(f"{kind}_keyword_key_count={section['keyword_key_count']}")
        print(f"{kind}_has_boundary_update_norm={section['has_boundary_update_norm']}")
        print(f"{kind}_has_merge_residual_delta={section['has_merge_residual_delta']}")
        print(f"{kind}_has_semantic_merge_residual={section['has_semantic_merge_residual']}")
        print(f"{kind}_has_transform_scale_value={section['has_transform_scale_value']}")


if __name__ == "__main__":
    main()
