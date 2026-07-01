#!/usr/bin/env python3
"""Audit ACL2 v84 Phase1 ruler candidate universe outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PHASE1_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase1_ruler_candidate_universe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.phase1_dir
    tokens = read_csv(root / "ruler_candidate_tokens.csv")
    pairs = read_csv(root / "ruler_candidate_pairs.csv")
    summaries = read_csv(root / "ruler_candidate_pair_summary.csv")
    features = read_csv(root / "feature_availability.csv")
    gate = read_json(root / "phase1_gate_summary.json")

    role_counts: dict[str, int] = {}
    for row in tokens:
        role = row.get("ruler_role", "")
        role_counts[role] = role_counts.get(role, 0) + 1
    missing_read = [
        f"{row.get('seq')}_{row.get('prev_chunk')}_{row.get('curr_chunk')}"
        for row in summaries
        if str(row.get("read_usage_available", "")).lower() != "true"
    ]
    missing_swa = [
        f"{row.get('seq')}_{row.get('prev_chunk')}_{row.get('curr_chunk')}"
        for row in summaries
        if str(row.get("swa_usage_available", "")).lower() != "true"
    ]
    audit = {
        "schema": "acl2_v84_phase1_candidate_universe_audit_v1",
        "phase1_dir": str(root),
        "expected_files": {
            "ruler_candidate_tokens.csv": (root / "ruler_candidate_tokens.csv").is_file(),
            "ruler_candidate_pairs.csv": (root / "ruler_candidate_pairs.csv").is_file(),
            "ruler_candidate_pair_summary.csv": (root / "ruler_candidate_pair_summary.csv").is_file(),
            "feature_availability.csv": (root / "feature_availability.csv").is_file(),
            "missing_artifact_report.md": (root / "missing_artifact_report.md").is_file(),
            "phase1_gate_summary.json": (root / "phase1_gate_summary.json").is_file(),
        },
        "token_rows": len(tokens),
        "candidate_pair_rows": len(pairs),
        "summary_rows": len(summaries),
        "role_counts": role_counts,
        "feature_availability": features,
        "missing_read_pairs": missing_read,
        "missing_swa_pairs": missing_swa,
        "phase1_gate_pass": gate.get("phase1_gate_pass"),
        "read_usage_available_ratio": gate.get("read_usage_available_ratio"),
        "swa_usage_available_ratio": gate.get("swa_usage_available_ratio"),
        "geometry_leverage_high_quality_ratio": gate.get("geometry_leverage_high_quality_ratio"),
    }
    write_json(root / "phase1_candidate_universe_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

