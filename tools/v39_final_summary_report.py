#!/usr/bin/env python3
"""Write v39 final per-track reports from landed artifacts only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
        if not math.isfinite(value):
            return "NA"
        return f"{value:.10f}"
    return str(value)


def _table(rows: Iterable[Tuple[str, Any]]) -> str:
    out = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        out.append(f"| `{key}` | `{_fmt(value)}` |")
    return "\n".join(out) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary_rows(summary: Dict[str, Any]) -> List[Tuple[str, Any]]:
    keys = [
        "rows",
        "missing_rows",
        "all_rows_done",
        "gate_pass",
        "best_ATE_candidate",
        "best_ATE_parent",
        "best_ATE_chunk",
        "best_ATE_delta_vs_base",
        "best_rolling_100f_candidate",
        "best_rolling_100f_parent",
        "best_rolling_100f_chunk",
        "best_rolling_100f_best_delta",
        "best_downstream_400_600_delta_for_best_ATE",
    ]
    return [(key, summary.get(key)) for key in keys if key in summary]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _plot_bars(path: Path, items: List[Tuple[str, float, float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    labels = [x[0] for x in items]
    ate = [x[1] for x in items]
    roll = [x[2] for x in items]
    fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.8), 4.8))
    xs = list(range(len(items)))
    ax.bar([x - 0.18 for x in xs], ate, width=0.36, label="best ATE delta")
    ax.bar([x + 0.18 for x in xs], roll, width=0.36, label="best rolling100 delta")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-1.5, color="red", linewidth=0.8, linestyle="--", label="h10 ATE gate")
    ax.axhline(-3.0, color="orange", linewidth=0.8, linestyle="--", label="h10 rolling gate")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("delta vs parent base (m)")
    ax.legend(fontsize=8)
    ax.set_title("v39 short-rollout best deltas")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    reports = root / "final_reports"
    reports.mkdir(parents=True, exist_ok=True)

    phase0 = _read_json(root / "phase0_semantic_appearance/report_R1/phase0_action_influence_summary.json")
    app = _read_json(root / "phase0_semantic_appearance/report_R1/v39_semantic_appearance_summary.json")
    t1_h10 = _read_json(root / "phase1_frameglobal/report_h10_R1/track1_h10_summary.json")
    t1_h15 = _read_json(root / "phase1_frameglobal/report_h15_R1/track1_h15_summary.json")
    t2_h10 = _read_json(root / "phase2_swa/report_h10_R1/track2_h10_summary.json")
    t2_h15 = _read_json(root / "phase2_swa/report_h15_R1/track2_h15_summary.json")
    t3_h10 = _read_json(root / "phase3_ttt/report_h10_R1/track3_h10_summary.json")
    t3_h15 = _read_json(root / "phase3_ttt/report_h15_R1/track3_h15_summary.json")
    t4_h10 = _read_json(root / "phase4_semc23/report_h10_R1/track4_h10_summary.json")
    t4_h15 = _read_json(root / "phase4_semc23/report_h15_R1/track4_h15_summary.json")
    full = _read_json(root / "phase5_full_online/report_R1/full_online_summary.json")

    _write(
        reports / "track0_semantic_appearance_report.md",
        "# Track 0 Semantic-Appearance Influence Atlas\n\n"
        + _table([
            ("track0_gate_pass", phase0.get("track0_gate_pass")),
            ("rows_done", phase0.get("rows_done")),
            ("missing_rows", phase0.get("missing_rows")),
            ("max_skipped_source_influence_mass", phase0.get("max_skipped_source_influence_mass")),
            ("attention_mass_rows", phase0.get("attention_mass_rows")),
            ("appearance_frame_rows", app.get("frame_rows")),
            ("appearance_masklet_rows", app.get("masklet_rows")),
            ("sky_lab_delta_p90", app.get("sky_lab_delta_p90")),
            ("sky_candidate_level_influence_mass_max", app.get("sky_candidate_level_influence_mass_max")),
            ("sky_causality_decision", app.get("sky_causality_decision")),
        ])
        + "\nBoundary: Track 0 is action/influence and appearance audit only; it is not deployable online evidence.\n",
    )
    _write(reports / "track1_frameglobal_report.md", "# Track 1 Frame/Global Source Surgery\n\nh10:\n\n" + _table(_summary_rows(t1_h10)) + "\nh15:\n\n" + _table(_summary_rows(t1_h15)))
    _write(reports / "track2_swa_report.md", "# Track 2 SWA Local Continuity\n\nh10:\n\n" + _table(_summary_rows(t2_h10)) + "\nh15:\n\n" + _table(_summary_rows(t2_h15)))
    _write(reports / "track3_ttt_report.md", "# Track 3 TTT Lifecycle\n\nh10:\n\n" + _table(_summary_rows(t3_h10)) + "\nh15:\n\n" + _table(_summary_rows(t3_h15)))
    _write(reports / "track4_semantic_c23_report.md", "# Track 4 Semantic-Conditioned C23 Residual\n\nh10:\n\n" + _table(_summary_rows(t4_h10)) + "\nh15:\n\n" + _table(_summary_rows(t4_h15)))

    h15_qualified = any(bool(x.get("gate_pass")) for x in (t1_h15, t2_h15, t3_h15, t4_h15))
    track5_launched = bool(full.get("full_online_launched", False)) if "missing_report" not in full else False
    target30_success = bool(full.get("target30_success", False)) if track5_launched else False
    _write(
        reports / "track5_full_online_report.md",
        "# Track 5 Full Online\n\n"
        + _table([
            ("full_online_allowed", h15_qualified),
            ("full_online_launched", track5_launched),
            ("target30_success", target30_success),
            ("reason", "requires h15-qualified Track 1-4 candidate" if not h15_qualified else full.get("reason", "see full_online_summary.json")),
        ]),
    )

    _plot_bars(
        reports / "short_rollout_delta_bar_chart.png",
        [
            ("T1 h10", _float(t1_h10.get("best_ATE_delta_vs_base")), _float(t1_h10.get("best_rolling_100f_best_delta"))),
            ("T2 h10", _float(t2_h10.get("best_ATE_delta_vs_base")), _float(t2_h10.get("best_rolling_100f_best_delta"))),
            ("T3 h10", _float(t3_h10.get("best_ATE_delta_vs_base")), _float(t3_h10.get("best_rolling_100f_best_delta"))),
            ("T4 h10", _float(t4_h10.get("best_ATE_delta_vs_base")), _float(t4_h10.get("best_rolling_100f_best_delta"))),
        ],
    )

    final = {
        "track0_gate_pass": phase0.get("track0_gate_pass"),
        "track1_h10_gate_pass": t1_h10.get("gate_pass"),
        "track1_h15_gate_pass": t1_h15.get("gate_pass"),
        "track2_h10_gate_pass": t2_h10.get("gate_pass"),
        "track2_h15_gate_pass": t2_h15.get("gate_pass"),
        "track3_h10_gate_pass": t3_h10.get("gate_pass"),
        "track3_h15_gate_pass": t3_h15.get("gate_pass"),
        "track4_h10_gate_pass": t4_h10.get("gate_pass"),
        "track4_h15_gate_pass": t4_h15.get("gate_pass"),
        "track5_full_online_allowed": h15_qualified,
        "track5_full_online_launched": track5_launched,
        "target30_success": target30_success,
        "sky_causality_decision": app.get("sky_causality_decision"),
        "best_deployable_online": "C9_P0_R2",
        "best_deployable_online_ate": 33.7629421029,
    }
    _write(
        reports / "failure_routing_summary.md",
        "# v39 Failure Routing Summary\n\n"
        + _table(final.items())
        + "\nVisualization boundary: final charts are generated only from landed report artifacts. Missing spatial tensors are recorded, not reconstructed.\n",
    )
    _write(reports / "v39_final_summary.json", json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
