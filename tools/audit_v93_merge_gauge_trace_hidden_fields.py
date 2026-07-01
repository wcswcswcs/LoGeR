#!/usr/bin/env python3
"""Scan available merge/gauge traces for hidden v93-required fields."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V91_ROOT, V92_ROOT  # noqa: E402


KEYWORDS = ("residual", "boundary", "gauge", "scale", "transform", "merge", "sim3", "update", "identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase3_merge_gauge_trace_audit")
    parser.add_argument("--v91-route-root", type=Path, default=V91_ROOT / "phase7_carrier_attribution_or_blocked/route_dump_smoke")
    parser.add_argument("--v92-noop-root", type=Path, default=V92_ROOT / "phase2_boundary_trace_ledger/noop_trace_smoke")
    parser.add_argument("--v93-smoke-root", type=Path, default=ROOT / "phase3_merge_gauge_trace_smoke")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten(value, child))
    elif isinstance(obj, list):
        rows.append((prefix, f"list[{len(obj)}]"))
    else:
        rows.append((prefix, obj))
    return rows


def scan(paths: list[Path], kind: str) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    samples = []
    keys: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    for path in paths:
        for row_idx, row in enumerate(read_jsonl(path)):
            for key, value in flatten(row):
                keys[key] += 1
                if any(word in key.lower() for word in KEYWORDS):
                    keywords[key] += 1
                    if len(samples) < 300:
                        samples.append({"kind": kind, "path": str(path), "row_idx": row_idx, "key": key, "sample_value": value})
    return samples, keys, keywords


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = [args.v93_smoke_root, args.v92_noop_root, args.v91_route_root]
    merge_paths: list[Path] = []
    hmc_paths: list[Path] = []
    hook_paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        merge_paths.extend(sorted(root.glob("**/merge_state_trace.jsonl")))
        hmc_paths.extend(sorted(root.glob("**/hmc_state_hash.jsonl")))
        hook_paths.extend(sorted(root.glob("**/hook_effect_summary.jsonl")))
    sample_rows = []
    sections: dict[str, Any] = {}
    for kind, paths in [("merge_state_trace", merge_paths), ("hmc_state_hash", hmc_paths), ("hook_effect_summary", hook_paths)]:
        samples, keys, keywords = scan(paths, kind)
        sample_rows.extend(samples)
        sections[kind] = {
            "file_count": len(paths),
            "unique_key_count": len(keys),
            "keyword_key_count": len(keywords),
            "keyword_keys": [{"key": key, "count": count} for key, count in keywords.most_common()],
            "has_boundary_update_norm": "boundary_update_norm" in keys,
            "has_boundary_update_l2": "boundary_update_l2" in keys,
            "has_boundary_update_scale_component": "boundary_update_scale_component" in keys,
            "has_merge_residual_delta": "merge_residual_delta" in keys,
            "has_merge_residual_before_after": "merge_residual_before" in keys and "merge_residual_after" in keys,
            "has_non_identity_transform_flag": "non_identity_transform_flag" in keys,
            "has_transform_scale_value": "transform_scale_value" in keys,
        }
    summary = {
        "phase": "Phase3_merge_gauge_trace_hidden_field_audit",
        "roots": [str(root) for root in roots],
        "merge_state_trace_files": len(merge_paths),
        "hmc_state_hash_files": len(hmc_paths),
        "hook_effect_summary_files": len(hook_paths),
        "sections": sections,
        "conclusion": "required v93 transform-derived fields may exist only in newly instrumented traces; merge_residual_delta remains unavailable unless future runtime computes residual before/after",
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "hidden_merge_gauge_field_audit.json", summary)
    write_csv(args.out_dir / "hidden_merge_gauge_field_samples.csv", sample_rows)
    print(f"merge_state_trace_files={len(merge_paths)}")
    print(f"hmc_state_hash_files={len(hmc_paths)}")
    print(f"hook_effect_summary_files={len(hook_paths)}")
    for kind, section in sections.items():
        print(f"{kind}_has_boundary_update_norm={section['has_boundary_update_norm']}")
        print(f"{kind}_has_merge_residual_delta={section['has_merge_residual_delta']}")
        print(f"{kind}_has_non_identity_transform_flag={section['has_non_identity_transform_flag']}")


if __name__ == "__main__":
    main()
