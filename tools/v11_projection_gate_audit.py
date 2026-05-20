#!/usr/bin/env python3
"""Audit whether v11 true projection instrumentation is actually connected."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REQUIRED_ARTIFACTS = [
    "per_token_update_group.pt",
    "per_layer_branch_update_matrix.pt",
    "tri_replay_role_mass.jsonl",
    "window_pose_residual_gt.json",
    "window_drift_direction_gt.pt",
    "update_to_drift_projection.csv",
    "update_cos_to_window_drift.jsonl",
    "projection_helpful_energy.jsonl",
    "ttt_update_conflict_energy.jsonl",
]

DEBUG_KEYS = [
    "update_cos_to_window_drift",
    "projection_helpful_energy",
    "projection_harmful_energy",
    "projection_role_mass",
    "ttt_update_conflict_energy",
]

SOURCE_HOOK_PATTERNS = [
    "window_drift_direction",
    "update_to_drift_projection",
    "projection_helpful_energy",
    "tri_replay_role_mass",
    "per_token_update_group",
    "per_layer_branch_update_matrix",
]


def _find_artifact(run_dir: Path, name: str) -> List[Path]:
    return sorted(path for path in run_dir.rglob(name) if path.is_file())


def _debug_key_hits(path: Path) -> Dict[str, int]:
    hits = {key: 0 for key in DEBUG_KEYS}
    if not path.exists():
        return hits
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = json.dumps(rec, sort_keys=True)
            for key in DEBUG_KEYS:
                if key in blob:
                    hits[key] += 1
    return hits


def _source_hits(repo_root: Path) -> Dict[str, bool]:
    paths = [
        repo_root / "run_pipeline_abc_v2.py",
        repo_root / "loger" / "pipeline" / "ttt_write_controller.py",
        repo_root / "loger" / "pipeline" / "hybrid_memory_controller.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths if path.exists())
    return {pattern: pattern in text for pattern in SOURCE_HOOK_PATTERNS}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    repo_root = Path(args.repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows: List[Dict[str, object]] = []
    present_count = 0
    for name in REQUIRED_ARTIFACTS:
        matches = _find_artifact(run_dir, name)
        present = bool(matches)
        present_count += int(present)
        artifact_rows.append({
            "artifact": name,
            "present": present,
            "count": len(matches),
            "paths": ";".join(str(path) for path in matches[:5]),
        })

    debug_hits = _debug_key_hits(run_dir / "hmc_state_hash.jsonl")
    debug_rows = [{"debug_key": key, "hit_rows": value, "present": value > 0} for key, value in debug_hits.items()]
    src_hits = _source_hits(repo_root)
    source_rows = [{"source_pattern": key, "present": value} for key, value in src_hits.items()]

    coverage = present_count / len(REQUIRED_ARTIFACTS)
    gate_pass = (
        coverage == 1.0
        and debug_hits.get("update_cos_to_window_drift", 0) > 0
        and debug_hits.get("projection_helpful_energy", 0) > 0
        and src_hits.get("window_drift_direction", False)
        and src_hits.get("update_to_drift_projection", False)
    )
    summary = {
        "run_dir": str(run_dir),
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "present_artifact_count": present_count,
        "coverage": coverage,
        "gate_pass": gate_pass,
        "phase2_allowed": gate_pass,
        "reason": "pass" if gate_pass else "required online projection instrumentation is incomplete",
    }

    _write_csv(out_dir / "required_artifact_coverage.csv", artifact_rows)
    _write_csv(out_dir / "debug_key_coverage.csv", debug_rows)
    _write_csv(out_dir / "source_hook_coverage.csv", source_rows)
    (out_dir / "projection_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# V11 Projection Gate Audit",
        "",
        f"run_dir = `{run_dir}`",
        "",
        "| Required artifact | Present | Count |",
        "|---|---:|---:|",
    ]
    for row in artifact_rows:
        lines.append(f"| `{row['artifact']}` | `{row['present']}` | `{row['count']}` |")
    lines.extend([
        "",
        "| Debug key | Present rows |",
        "|---|---:|",
    ])
    for row in debug_rows:
        lines.append(f"| `{row['debug_key']}` | `{row['hit_rows']}` |")
    lines.extend([
        "",
        "| Source hook pattern | Present in checked source |",
        "|---|---:|",
    ])
    for row in source_rows:
        lines.append(f"| `{row['source_pattern']}` | `{row['present']}` |")
    lines.extend([
        "",
        "## Gate",
        "",
        f"- required artifact coverage: `{coverage:.3f}`",
        f"- gate_pass: `{gate_pass}`",
    ])
    if not gate_pass:
        lines.append("- Phase 2 is not allowed by the v11 plan; this is an instrumentation-not-connected result, not a failed projection oracle.")
    (out_dir / "v11_projection_gate_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'v11_projection_gate_audit.md'}")


if __name__ == "__main__":
    main()
