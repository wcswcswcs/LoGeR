#!/usr/bin/env python3
"""Audit v66B active-chunk semantic action fidelity.

The check is intentionally narrow: inactive chunks must not consume semantic
prior, expose TTT write prior, or apply role controls; active chunks must do so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


ACTIVE_BALANCED = {2, 3, 6, 8, 10, 17, 18, 21, 23}
ACTIVE_FUTURE = {2, 3, 6, 8, 10, 15, 17, 18, 21, 23, 24}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _control_mode(run_name: str) -> str:
    if "RANDOMCTRL" in run_name:
        return "random_same_mass"
    if "SHUFFLECTRL" in run_name:
        return "shuffled_semantic"
    return "none"


def _active_chunks(run_name: str) -> Set[int]:
    if "ACTIVEFUTURE" in run_name:
        return set(ACTIVE_FUTURE)
    return set(ACTIVE_BALANCED)


def _audit_run(run_dir: Path) -> Dict[str, Any]:
    run_name = run_dir.name
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    active = _active_chunks(run_name)
    mode = _control_mode(run_name)
    failures: List[Dict[str, Any]] = []
    active_rows = 0
    inactive_rows = 0
    changed_fractions: List[float] = []

    for row in rows:
        chunk_idx = int(row.get("chunk_idx", -1))
        gate = bool(row.get("prior_semantic_action_chunk_gate_active"))
        consumed = bool(row.get("prior_semantic_action_prior_consumed"))
        control_applied = row.get("prior_semantic_role_control_applied")
        ttt_present = bool(row.get("prior_ttt_write_present"))
        if chunk_idx in active:
            active_rows += 1
            if not gate or not consumed or not ttt_present:
                failures.append({
                    "chunk_idx": chunk_idx,
                    "reason": "active_missing_action",
                    "gate": gate,
                    "prior_consumed": consumed,
                    "ttt_write_present": ttt_present,
                    "control_applied": control_applied,
                })
            if mode != "none" and control_applied is not True:
                failures.append({
                    "chunk_idx": chunk_idx,
                    "reason": "active_control_not_applied",
                    "expected_control_mode": mode,
                    "control_applied": control_applied,
                })
            if mode == "none" and control_applied not in (False, None):
                failures.append({
                    "chunk_idx": chunk_idx,
                    "reason": "none_control_applied",
                    "control_applied": control_applied,
                })
            changed = row.get("prior_semantic_role_control_changed_fraction")
            if isinstance(changed, (int, float)):
                changed_fractions.append(float(changed))
        else:
            inactive_rows += 1
            if gate or consumed or ttt_present:
                failures.append({
                    "chunk_idx": chunk_idx,
                    "reason": "inactive_leak",
                    "gate": gate,
                    "prior_consumed": consumed,
                    "ttt_write_present": ttt_present,
                    "control_applied": control_applied,
                })
            if control_applied not in (False, None):
                failures.append({
                    "chunk_idx": chunk_idx,
                    "reason": "inactive_control_applied",
                    "control_applied": control_applied,
                })

    return {
        "run": run_name,
        "rows": len(rows),
        "active_rows": active_rows,
        "inactive_rows": inactive_rows,
        "control_mode": mode,
        "active_chunks": sorted(active),
        "control_changed_fraction_mean": (
            sum(changed_fractions) / len(changed_fractions)
            if changed_fractions
            else None
        ),
        "failures": failures,
        "pass": not failures and active_rows > 0 and inactive_rows > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Directory containing active-chunk rollout dirs.")
    parser.add_argument("--json-out", default=None, help="Optional path to write the audit JSON.")
    args = parser.parse_args()

    base = Path(args.base)
    runs = sorted(base.glob("V66B_P9_704_TTT_ROLE_EXTREME_ACTIVE*"))
    results = [_audit_run(run_dir) for run_dir in runs]
    payload = {
        "base": str(base),
        "run_count": len(results),
        "pass": bool(results) and all(row["pass"] for row in results),
        "runs": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
