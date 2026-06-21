#!/usr/bin/env python3
"""Audit ACL2 v76 Phase5 commit-EMA bridge evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (  # noqa: E402
    V45_CLEAN_REGISTRY,
    V45_INTERACTION,
    V45_LEDGER,
    V45_ROOT,
    V76_ROOT,
    first_row,
    read_csv,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


SEM4_COMPONENT_REGISTRY = V45_ROOT / "phase5_semantic_read_extra/report_R1/sem4_component_combos/full_online_registry.csv"


def _ate(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    return safe_float(row.get("ATE_full")) if row else None


def _gain(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return reference - candidate


def run(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(V45_LEDGER)
    clean = read_csv(V45_CLEAN_REGISTRY)
    combos = read_csv(SEM4_COMPONENT_REGISTRY)
    interactions = read_csv(V45_INTERACTION)

    ledger_commit = first_row(ledger, "component", "commit_ema")
    d7 = first_row(clean, "name", "D7")
    d5 = first_row(clean, "name", "D5")
    d6 = first_row(clean, "name", "D6")
    sem4 = first_row(combos, "name", "SEM4")
    no_ema = first_row(combos, "name", "X_NO_EMA")
    i8 = first_row(interactions, "candidate", "I8")
    i7 = first_row(interactions, "candidate", "I7")

    d7_ate = _ate(d7)
    d5_ate = _ate(d5)
    d6_ate = _ate(d6)
    sem4_ate = _ate(sem4)
    no_ema_ate = _ate(no_ema)
    i8_ate = _ate(i8)
    i7_ate = _ate(i7)

    rows = [
        {
            "scope": "c9_component_necessity",
            "artifact": rel(V45_LEDGER),
            "reference": "C9 repeat",
            "candidate": "C9_MINUS_COMMIT_EMA",
            "effect_delta_vs_C9_m": safe_float(ledger_commit.get("effect_delta_vs_C9")) if ledger_commit else None,
            "interpretation": "Commit EMA is necessary in C9/chunk-map context when removal worsens ATE.",
        },
        {
            "scope": "c9clean_fixed_ema_substitution",
            "artifact": rel(V45_CLEAN_REGISTRY),
            "reference": "D7_C9_CLEAN_BEST_FIXED",
            "candidate": "D5_NO_EMA / D6_GLOBAL_A08",
            "D7_ATE_full": d7_ate,
            "D5_no_ema_ATE_full": d5_ate,
            "D6_global_a08_ATE_full": d6_ate,
            "D7_gain_vs_D5_no_ema_m": _gain(d5_ate, d7_ate),
            "D7_gain_vs_D6_global_a08_m": _gain(d6_ate, d7_ate),
            "interpretation": "C9-clean fixed recipe is better than these EMA variants, but this is not semantic-runtime proof.",
        },
        {
            "scope": "semantic_read_adaptive_tri_component",
            "artifact": rel(SEM4_COMPONENT_REGISTRY),
            "reference": "SEM4",
            "candidate": "X_NO_EMA",
            "SEM4_ATE_full": sem4_ate,
            "X_NO_EMA_ATE_full": no_ema_ate,
            "SEM4_gain_vs_X_NO_EMA_m": _gain(no_ema_ate, sem4_ate),
            "interpretation": "Historical SEM4 loses most of its gain when commit EMA is removed, but this row is chunk-map contaminated.",
        },
        {
            "scope": "fixed_read_tri_fixed_ema_interaction",
            "artifact": rel(V45_INTERACTION),
            "reference": "I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST",
            "candidate": "I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST",
            "I7_ATE_full": i7_ate,
            "I8_ATE_full": i8_ate,
            "I8_gain_vs_I7_m": _gain(i7_ate, i8_ate),
            "interpretation": "Adding fixed EMA to fixed READ+tri-gamma worsens ATE in this interaction table.",
        },
    ]
    sem4_gain = _gain(no_ema_ate, sem4_ate)
    i8_gain = _gain(i7_ate, i8_ate)
    summary = {
        "phase5_c9_commit_ema_necessity_available": ledger_commit is not None,
        "phase5_sem4_commit_ema_historical_gain_m": sem4_gain,
        "phase5_fixed_ema_interaction_gain_m": i8_gain,
        "phase5_commit_ema_training_free_bridge_gate_pass": False,
        "phase5_commit_ema_training_free_bridge_reason": (
            "Commit EMA is supported as C9/chunk-map historical necessity and SEM4 historical component, "
            "but available evidence does not prove a chunk-policy-free semantic runtime commit bridge; "
            "fixed EMA interaction can even worsen I8 vs I7."
        ),
    }
    write_csv(out_dir / "phase5_commit_ema_bridge_rows.csv", rows)
    write_json(out_dir / "phase5_commit_ema_bridge_summary.json", summary)
    _write_report(out_dir, rows, summary)
    return {"out_dir": rel(out_dir), **summary}


def _write_report(out_dir: Path, rows: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = ["# v76 Phase5 Commit-EMA Bridge Audit", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rows", "", "| scope | artifact | interpretation |", "|---|---|---|"])
    for row in rows:
        lines.append(f"| `{row.get('scope')}` | `{row.get('artifact')}` | {row.get('interpretation')} |")
    lines.append("")
    write_text(out_dir / "phase5_commit_ema_bridge_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase5_commit_ema_bridge"))
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)


if __name__ == "__main__":
    main()
