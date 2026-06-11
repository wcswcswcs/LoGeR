#!/usr/bin/env python3
"""Write v41 final reports from landed artifacts only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"missing_report": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"invalid_report": str(path)}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10f}" if math.isfinite(value) else "NA"
    return str(value)


def _table(rows: Iterable[Tuple[str, Any]]) -> str:
    out = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        out.append(f"| `{key}` | `{_fmt(value)}` |")
    return "\n".join(out) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    reports = root / "final_reports"
    reports.mkdir(parents=True, exist_ok=True)

    phase1 = _read_json(root / "phase1_health_detector/v41_health_detector_summary.json")
    mechanism = _read_json(root / "phase2_read_mechanism/v41_read_mechanism_summary.json")
    h10 = _read_json(root / "phase3_read_h10/report_h10_R1/read_h10_v41_gate_summary.json")
    h15_h9 = _read_json(root / "phase3_read_h15/report_h15_R1_H9/read_h15_H9_v41_gate_summary.json")
    h15_c9 = _read_json(root / "phase3_read_h15/report_h15_R1_C9/read_h15_C9_v41_gate_summary.json")
    wash_h9 = _read_json(root / "phase5_washout/R1_H9/read_washout_summary.json")
    wash_c9 = _read_json(root / "phase5_washout/R1_C9/read_washout_summary.json")
    wash_r3 = _read_json(root / "phase5_washout/R3_H9/read_washout_summary.json")

    h15_pass = bool(h15_h9.get("gate_pass")) or bool(h15_c9.get("gate_pass"))
    full_allowed = h15_pass
    final = {
        "phase1_health_detector_gate_pass": phase1.get("phase1_gate_pass"),
        "selected_bad_chunks": phase1.get("selected_bad_chunks"),
        "selected_bad_chunk_ratio": phase1.get("selected_bad_chunk_ratio"),
        "mechanism_decision": mechanism.get("mechanism_decision"),
        "sky_causality_proven": mechanism.get("sky_causality_proven"),
        "scalar_attention_mass_rows": mechanism.get("scalar_attention_mass_rows"),
        "h10_gate_pass": h10.get("gate_pass"),
        "h10_gate_pass_rows": h10.get("gate_pass_rows"),
        "h10_best_ATE_delta": h10.get("best_ATE_delta_vs_base"),
        "h10_best_rolling100_delta": h10.get("best_rolling_100f_best_delta"),
        "h10_best_stress_delta": h10.get("best_stress_delta"),
        "h15_H9_gate_pass": h15_h9.get("gate_pass"),
        "h15_H9_best_ATE_delta": h15_h9.get("best_ATE_delta_vs_base"),
        "h15_H9_best_rolling100_delta": h15_h9.get("best_rolling_100f_best_delta"),
        "h15_H9_best_stress_delta": h15_h9.get("best_stress_delta"),
        "h15_C9_gate_pass": h15_c9.get("gate_pass"),
        "h15_C9_best_ATE_delta": h15_c9.get("best_ATE_delta_vs_base"),
        "h15_C9_best_rolling100_delta": h15_c9.get("best_rolling_100f_best_delta"),
        "h15_C9_best_stress_delta": h15_c9.get("best_stress_delta"),
        "full_online_allowed": full_allowed,
        "full_online_launched": False,
        "target30_success": False,
        "best_deployable_online": "C9_P0_R2",
        "best_deployable_online_ate": 33.7629421029,
    }

    _write(
        reports / "health_detector_report.md",
        "# v41 Health Detector\n\n"
        + _table([
            ("phase1_gate_pass", phase1.get("phase1_gate_pass")),
            ("selected_bad_chunks", phase1.get("selected_bad_chunks")),
            ("top3_health_risk_chunks", phase1.get("top3_health_risk_chunks")),
            ("selected_bad_chunk_ratio", phase1.get("selected_bad_chunk_ratio")),
            ("top_rolling100_bad_chunk_diagnostic", phase1.get("top_rolling100_bad_chunk_diagnostic")),
            ("selection_uses_ATE", phase1.get("selection_uses_ATE")),
            ("selection_uses_fixed_chunk_or_segment", phase1.get("selection_uses_fixed_chunk_or_segment")),
        ]),
    )
    _write(
        reports / "read_mechanism_report.md",
        "# v41 READ Mechanism\n\n"
        + _table([
            ("mechanism_decision", mechanism.get("mechanism_decision")),
            ("sky_causality_proven", mechanism.get("sky_causality_proven")),
            ("scalar_attention_mass_rows", mechanism.get("scalar_attention_mass_rows")),
            ("proxy_overlays_copied", mechanism.get("proxy_overlays_copied")),
            ("evidence_level", mechanism.get("evidence_level")),
            ("reason", mechanism.get("reason")),
        ]),
    )
    _write(
        reports / "read_h10_candidate_report.md",
        "# v41 READ h10 Candidates\n\n"
        + _table([
            ("gate_pass", h10.get("gate_pass")),
            ("gate_pass_rows", h10.get("gate_pass_rows")),
            ("best_ATE_candidate", h10.get("best_ATE_candidate")),
            ("best_ATE_delta_vs_base", h10.get("best_ATE_delta_vs_base")),
            ("best_rolling_100f_candidate", h10.get("best_rolling_100f_candidate")),
            ("best_rolling_100f_best_delta", h10.get("best_rolling_100f_best_delta")),
            ("best_stress_candidate", h10.get("best_stress_candidate")),
            ("best_stress_delta", h10.get("best_stress_delta")),
        ]),
    )
    _write(
        reports / "read_h15_candidate_report.md",
        "# v41 READ h15 Candidates\n\nH9:\n\n"
        + _table([
            ("gate_pass", h15_h9.get("gate_pass")),
            ("best_ATE_delta_vs_base", h15_h9.get("best_ATE_delta_vs_base")),
            ("best_rolling_100f_best_delta", h15_h9.get("best_rolling_100f_best_delta")),
            ("best_stress_delta", h15_h9.get("best_stress_delta")),
        ])
        + "\nC9:\n\n"
        + _table([
            ("gate_pass", h15_c9.get("gate_pass")),
            ("best_ATE_delta_vs_base", h15_c9.get("best_ATE_delta_vs_base")),
            ("best_rolling_100f_best_delta", h15_c9.get("best_rolling_100f_best_delta")),
            ("best_stress_delta", h15_c9.get("best_stress_delta")),
        ]),
    )
    _write(
        reports / "washout_report.md",
        "# v41 READ Washout\n\nR1 H9:\n\n"
        + _table([
            ("stress_durability", wash_h9.get("metric_durability", {}).get("stress_200_300_delta")),
            ("rolling100_durability", wash_h9.get("metric_durability", {}).get("rolling100_best_delta")),
            ("ttt_tail_over_h10", wash_h9.get("path_tail_over_h10", {}).get("ttt_state")),
            ("frame_bias_tail_over_h10", wash_h9.get("path_tail_over_h10", {}).get("frame_attention_bias")),
        ])
        + "\nR1 C9:\n\n"
        + _table([
            ("stress_durability", wash_c9.get("metric_durability", {}).get("stress_200_300_delta")),
            ("rolling100_durability", wash_c9.get("metric_durability", {}).get("rolling100_best_delta")),
            ("ttt_tail_over_h10", wash_c9.get("path_tail_over_h10", {}).get("ttt_state")),
            ("frame_bias_tail_over_h10", wash_c9.get("path_tail_over_h10", {}).get("frame_attention_bias")),
        ])
        + "\nR3 H9:\n\n"
        + _table([
            ("stress_durability", wash_r3.get("metric_durability", {}).get("stress_200_300_delta")),
            ("rolling100_durability", wash_r3.get("metric_durability", {}).get("rolling100_best_delta")),
            ("ttt_tail_over_h10", wash_r3.get("path_tail_over_h10", {}).get("ttt_state")),
            ("frame_bias_tail_over_h10", wash_r3.get("path_tail_over_h10", {}).get("frame_attention_bias")),
        ])
        + "\nBoundary: proxy-only JSONL attribution; no tensor-state overwrite proof is claimed.\n",
    )
    _write(
        reports / "full_online_report.md",
        "# v41 Full Online\n\n"
        + _table([
            ("full_online_allowed", full_allowed),
            ("full_online_launched", False),
            ("target30_success", False),
            ("reason", "No h15-qualified READ candidate exists." if not full_allowed else "h15-qualified candidate exists"),
        ]),
    )
    _write(reports / "failure_routing_summary.md", "# v41 Failure Routing Summary\n\n" + _table(final.items()))
    _write(reports / "v41_final_summary.json", json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
