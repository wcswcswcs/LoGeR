#!/usr/bin/env python3
"""Audit whether landed runs contain the true v10 projection-oracle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


REQUIRED_ARTIFACTS = (
    "per_token_update_group.pt",
    "per_layer_branch_update_matrix.pt",
    "tri_replay_role_mass.jsonl",
    "window_pose_residual_gt.json",
    "window_drift_direction_gt.pt",
    "update_to_drift_projection.csv",
)

REQUIRED_DEBUG_KEYS = (
    "update_cos_to_window_drift",
    "projection_helpful_energy",
    "projection_harmful_energy",
    "projection_orthogonal_energy",
    "oracle_projection_energy_total",
    "oracle_harmful_energy_removed",
)

KNOWN_AGGREGATE_KEYS = (
    "ttt_update_conflict_energy_mean",
    "ttt_update_conflict_energy_p90",
    "ttt_update_conflict_risk_mean",
    "ttt_tri_replay_pos_mass",
    "ttt_tri_replay_neu_mass",
    "ttt_tri_replay_neg_mass",
    "ttt_self_cue_update_conflict_present",
)


def _collect_keys(obj: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from _collect_keys(value, name)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj[:4]):
            yield from _collect_keys(value, f"{prefix}[{idx}]")


def _load_jsonl_keys(path: Path, max_lines: int = 80) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.update(_collect_keys(payload))
    return keys


def _source_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {name: (run_dir / name).exists() for name in REQUIRED_ARTIFACTS}
    keys = set()
    for rel in ("hmc_state_hash.jsonl", "hmc_probe_summary.jsonl", "hook_effect_summary.jsonl"):
        keys.update(_load_jsonl_keys(run_dir / rel))
    key_hits = {key: any(key in landed for landed in keys) for key in REQUIRED_DEBUG_KEYS}
    aggregate_hits = {key: any(key in landed for landed in keys) for key in KNOWN_AGGREGATE_KEYS}

    source_files = [
        repo / "loger/pipeline/ttt_write_controller.py",
        repo / "loger/pipeline/hybrid_memory_controller.py",
        repo / "run_pipeline_abc_v2.py",
    ]
    source_hits: Dict[str, bool] = {}
    for needle in (
        "update_cos_to_window_drift",
        "projection_helpful_energy",
        "per_token_update_group",
        "window_drift_direction",
        "tri_replay_role_mass",
        "ttt_update_conflict_energy",
    ):
        source_hits[needle] = any(_source_contains(path, needle) for path in source_files)

    gate_pass = all(artifacts.values()) and all(key_hits.values()) and source_hits.get("window_drift_direction", False)
    summary = {
        "run_dir": str(run_dir),
        "required_artifacts": artifacts,
        "required_debug_keys": key_hits,
        "known_aggregate_debug_keys": aggregate_hits,
        "source_hits": source_hits,
        "gate_pass": gate_pass,
    }
    (out_dir / "projection_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    def yes(value: bool) -> str:
        return "yes" if value else "no"

    lines = [
        "# V10 Projection Oracle Feasibility Audit",
        "",
        f"run_dir = `{run_dir}`",
        f"gate_pass = `{yes(gate_pass)}`",
        "",
        "## Required Artifacts",
        "",
        "| Artifact | Present |",
        "|---|---:|",
    ]
    for name, present in artifacts.items():
        lines.append(f"| `{name}` | {yes(present)} |")
    lines += ["", "## Required Projection Debug Keys", "", "| Key | Present in landed JSONL |", "|---|---:|"]
    for key, present in key_hits.items():
        lines.append(f"| `{key}` | {yes(present)} |")
    lines += ["", "## Existing Aggregate TTT Debug", "", "| Key | Present |", "|---|---:|"]
    for key, present in aggregate_hits.items():
        lines.append(f"| `{key}` | {yes(present)} |")
    lines += ["", "## Source-Level Hooks", "", "| Token | Found in source |", "|---|---:|"]
    for key, present in source_hits.items():
        lines.append(f"| `{key}` | {yes(present)} |")
    lines += [
        "",
        "## Decision",
        "",
    ]
    if gate_pass:
        lines.append("Projection instrumentation evidence is present. Full oracle candidates may be considered after smoke.")
    else:
        lines.append(
            "True v10 projection oracle is not evidenced in the landed run: existing debug contains aggregate "
            "`update_conflict_energy` / tri-replay mass, but not token-update to window-drift projection artifacts. "
            "Per the v10 gate, do not run `ORACLEPROJ_*` full runs from this state."
        )
    (out_dir / "projection_feasibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'projection_feasibility_audit.md'}")


if __name__ == "__main__":
    main()
