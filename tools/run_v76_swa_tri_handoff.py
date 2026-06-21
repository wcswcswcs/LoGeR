#!/usr/bin/env python3
"""Audit ACL2 v76 Phase6 SWA/MERGE handoff evidence around tri-replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (  # noqa: E402
    V46B_REGISTRY,
    V45_ROOT,
    V74_ROOT,
    V76_ROOT,
    first_row,
    read_csv,
    read_json,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


SEM4_COMPONENT_REGISTRY = V45_ROOT / "phase5_semantic_read_extra/report_R1/sem4_component_combos/full_online_registry.csv"
SWA_SMOKE_SUMMARY = V74_ROOT / "phase5_component_leave_one_out_swa_turnoff_top4/radio_swa_online_smoke_summary.json"


def _ate(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    return safe_float(row.get("ATE_full")) if row else None


def _gain(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return reference - candidate


def run(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    v46 = read_csv(V46B_REGISTRY)
    sem = read_csv(SEM4_COMPONENT_REGISTRY)
    swa_summary = read_json(SWA_SMOKE_SUMMARY)
    swa_summary = swa_summary if isinstance(swa_summary, dict) else {}

    f010 = first_row(v46, "row", "F010_ONLY_TTT")
    f011 = first_row(v46, "row", "F011_TTT_SWA")
    f110 = first_row(v46, "row", "F110_FRAME_ATTN_TTT")
    f111 = first_row(v46, "row", "F111_ALL_THREE")
    sem4 = first_row(sem, "name", "SEM4")
    no_swa = first_row(sem, "name", "X_NO_SWA")

    f010_ate = _ate(f010)
    f011_ate = _ate(f011)
    f110_ate = _ate(f110)
    f111_ate = _ate(f111)
    sem4_ate = _ate(sem4)
    no_swa_ate = _ate(no_swa)

    details = swa_summary.get("chunk_details")
    improvements = []
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                value = safe_float(item.get("candidate_improvement_m"))
                if value is not None:
                    improvements.append(value)

    rows = [
        {
            "scope": "start_a_ttt_plus_swa_factorial",
            "artifact": rel(V46B_REGISTRY),
            "reference": "F010_ONLY_TTT",
            "candidate": "F011_TTT_SWA",
            "reference_ATE_full": f010_ate,
            "candidate_ATE_full": f011_ate,
            "candidate_gain_m": _gain(f010_ate, f011_ate),
            "interpretation": "SWA adds only a tiny ATE gain to TTT-only in Start A.",
        },
        {
            "scope": "start_a_read_ttt_plus_swa_factorial",
            "artifact": rel(V46B_REGISTRY),
            "reference": "F110_FRAME_ATTN_TTT",
            "candidate": "F111_ALL_THREE",
            "reference_ATE_full": f110_ate,
            "candidate_ATE_full": f111_ate,
            "candidate_gain_m": _gain(f110_ate, f111_ate),
            "interpretation": "SWA adds only a tiny ATE gain to READ+TTT in Start A.",
        },
        {
            "scope": "historical_sem4_no_swa",
            "artifact": rel(SEM4_COMPONENT_REGISTRY),
            "reference": "X_NO_SWA",
            "candidate": "SEM4",
            "reference_ATE_full": no_swa_ate,
            "candidate_ATE_full": sem4_ate,
            "candidate_gain_m": _gain(no_swa_ate, sem4_ate),
            "interpretation": "SEM4 has a small historical SWA contribution, but this context is chunk-map contaminated.",
        },
        {
            "scope": "v74_online_swa_smoke",
            "artifact": rel(SWA_SMOKE_SUMMARY),
            "reference": "native/control chunks",
            "candidate": swa_summary.get("phase", "v74_swa_smoke"),
            "candidate_pass_chunks": ",".join(str(x) for x in swa_summary.get("candidate_pass_chunks", [])) if isinstance(swa_summary.get("candidate_pass_chunks"), list) else "",
            "best_candidate_improvement_m": max(improvements) if improvements else None,
            "worst_candidate_improvement_m": min(improvements) if improvements else None,
            "gate_pass": bool(swa_summary.get("swa_online_gate_pass")),
            "interpretation": "Existing online SWA smoke does not pass its gate.",
        },
    ]
    f011_gain = _gain(f010_ate, f011_ate)
    f111_gain = _gain(f110_ate, f111_ate)
    sem4_gain = _gain(no_swa_ate, sem4_ate)
    summary = {
        "phase6_start_a_ttt_swa_gain_m": f011_gain,
        "phase6_start_a_read_ttt_swa_gain_m": f111_gain,
        "phase6_sem4_historical_swa_gain_m": sem4_gain,
        "phase6_online_swa_smoke_gate_pass": bool(swa_summary.get("swa_online_gate_pass")),
        "phase6_swa_tri_handoff_gate_pass": False,
        "phase6_swa_tri_handoff_reason": (
            "SWA has at most tiny Start A gains, historical SEM4 SWA support is chunk-map contaminated, "
            "and v74 online SWA smoke fails; no deployable tri-replay handoff gate is proven."
        ),
    }
    write_csv(out_dir / "phase6_swa_tri_handoff_rows.csv", rows)
    write_json(out_dir / "phase6_swa_tri_handoff_summary.json", summary)
    _write_report(out_dir, rows, summary)
    return {"out_dir": rel(out_dir), **summary}


def _write_report(out_dir: Path, rows: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = ["# v76 Phase6 SWA / Tri-Replay Handoff Audit", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rows", "", "| scope | artifact | candidate_gain_m | interpretation |", "|---|---|---:|---|"])
    for row in rows:
        lines.append(
            f"| `{row.get('scope')}` | `{row.get('artifact')}` | `{row.get('candidate_gain_m')}` | {row.get('interpretation')} |"
        )
    lines.append("")
    write_text(out_dir / "phase6_swa_tri_handoff_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase6_swa_tri_handoff"))
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)


if __name__ == "__main__":
    main()
