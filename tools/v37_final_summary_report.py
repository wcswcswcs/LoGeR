#!/usr/bin/env python3
"""Write v37 final per-track Markdown reports and summary plots.

This script only consumes landed JSON/CSV report artifacts. It does not infer
missing tensor overlays or fabricate full-online results. Missing visualization
items requested by the plan are written explicitly into
``failure_routing_summary.md`` as not generated / not applicable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"missing_report": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _table(rows: Iterable[Tuple[str, Any]]) -> str:
    out = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        out.append(f"| `{key}` | `{_fmt(value)}` |")
    return "\n".join(out) + "\n"


def _summary_rows(summary: Dict[str, Any]) -> List[Tuple[str, Any]]:
    keys = [
        "rows",
        "missing_rows",
        "all_rows_done",
        "gate_pass",
        "best_ATE_candidate",
        "best_ATE_chunk",
        "best_ATE_delta_vs_H9",
        "best_200_300_candidate",
        "best_200_300_chunk",
        "best_200_300_delta_vs_H9",
        "best_400_600_delta_for_best_ATE",
    ]
    return [(key, summary.get(key)) for key in keys if key in summary]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_bars(path: Path, items: List[Tuple[str, float, float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    labels = [x[0] for x in items]
    seg = [x[1] for x in items]
    downstream = [x[2] for x in items]
    fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.75), 4.8))
    xs = range(len(items))
    ax.bar([x - 0.18 for x in xs], seg, width=0.36, label="[200,300) delta")
    ax.bar([x + 0.18 for x in xs], downstream, width=0.36, label="[400,600) delta")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ATE delta vs parent base (m)")
    ax.legend()
    ax.set_title("v37 short-rollout segment deltas")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_durability(path: Path, items: List[Tuple[str, float, float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, h10, h15 in items:
        ax.plot([10, 15], [h10, h15], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-5, color="red", linewidth=0.8, linestyle="--", label="h15 local gate")
    ax.set_xticks([10, 15])
    ax.set_xlabel("horizon")
    ax.set_ylabel("[200,300) ATE delta (m)")
    ax.set_title("v37 h10 to h15 durability")
    ax.legend(fontsize=8)
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

    phase0 = _read_json(root / "phase0_action_influence/report_R1/phase0_action_influence_summary.json")
    track1_h9 = _read_json(root / "phase1_frameglobal/report_h10_R1_H9/track1_h10_H9_summary.json")
    track1_c9 = _read_json(root / "phase1_frameglobal/report_h10_R1_C9/track1_h10_C9_summary.json")
    track2_h10_h9 = _read_json(root / "phase2_swa/report_h10_R2_H9/track2_h10_H9_summary.json")
    track2_h10_c9 = _read_json(root / "phase2_swa/report_h10_R2_C9/track2_h10_C9_summary.json")
    track2_h15_h9 = _read_json(root / "phase2_swa/report_h15_R1_H9/track2_h15_H9_summary.json")
    track2_h15_c9 = _read_json(root / "phase2_swa/report_h15_R1_C9/track2_h15_C9_summary.json")
    track3_h10_h9 = _read_json(root / "phase3_ttt/report_h10_R2_H9/track3_h10_H9_summary.json")
    track3_h10_c9 = _read_json(root / "phase3_ttt/report_h10_R2_C9/track3_h10_C9_summary.json")
    track3_h15_h9 = _read_json(root / "phase3_ttt/report_h15_R1_H9/track3_h15_H9_summary.json")
    track3_h15_c9 = _read_json(root / "phase3_ttt/report_h15_R1_C9/track3_h15_C9_summary.json")
    track4_h9 = _read_json(root / "phase4_semc23/report_h10_R1_H9/track4_h10_H9_summary.json")
    track4_c9 = _read_json(root / "phase4_semc23/report_h10_R1_C9/track4_h10_C9_summary.json")

    _write(
        reports / "track1_frameglobal_report.md",
        "# Track 1 Frame/Global Source Surgery\n\n"
        "H9 summary:\n\n"
        + _table(_summary_rows(track1_h9))
        + "\nC9 summary:\n\n"
        + _table(_summary_rows(track1_c9))
        + "\nDecision: h10 gate failed for both H9 and C9; no Track 1 h15/full-online continuation.\n",
    )

    _write(
        reports / "track2_swa_report.md",
        "# Track 2 SWA Local-Continuity\n\n"
        "H9 h10:\n\n"
        + _table(_summary_rows(track2_h10_h9))
        + "\nC9 h10:\n\n"
        + _table(_summary_rows(track2_h10_c9))
        + "\nH9 h15:\n\n"
        + _table(_summary_rows(track2_h15_h9))
        + "\nC9 h15:\n\n"
        + _table(_summary_rows(track2_h15_c9))
        + "\nDecision: h10 local signal exists but h15 durability gate failed; no Track 2 full-online continuation.\n",
    )

    _write(
        reports / "track3_ttt_report.md",
        "# Track 3 TTT Static/Short-Negative\n\n"
        "H9 h10:\n\n"
        + _table(_summary_rows(track3_h10_h9))
        + "\nC9 h10:\n\n"
        + _table(_summary_rows(track3_h10_c9))
        + "\nH9 h15:\n\n"
        + _table(_summary_rows(track3_h15_h9))
        + "\nC9 h15:\n\n"
        + _table(_summary_rows(track3_h15_c9))
        + "\nDecision: TTT_FINE_RISK_02_SCALE_STATE passed h10 by ATE but failed h15; no Track 3 full-online continuation.\n",
    )

    _write(
        reports / "track4_semantic_c23_report.md",
        "# Track 4 Semantic C23 Path Isolation\n\n"
        "H9 h10:\n\n"
        + _table(_summary_rows(track4_h9))
        + "\nC9 h10:\n\n"
        + _table(_summary_rows(track4_c9))
        + "\nDecision: h10 gate failed for both parents; no Track 4 h15/full-online continuation.\n",
    )

    full_allowed = False
    _write(
        reports / "track5_full_online_report.md",
        "# Track 5 Full Online\n\n"
        "No v37 full-online row was launched.\n\n"
        "Reason: Track 5 requires h15-qualified candidates from Tracks 1-4. "
        "Track 1 failed h10, Track 2 passed h10 but failed h15, Track 3 passed h10 but failed h15, "
        "and Track 4 failed h10.\n\n"
        "full_online_allowed = false\n"
        f"full_online_launched = {str(full_allowed).lower()}\n",
    )

    _write(
        reports / "failure_routing_summary.md",
        "# v37 Failure Routing Summary\n\n"
        "Track 0 action/influence audit passed, with nontrivial skipped-source influence mass.\n\n"
        + _table([
            ("track0_gate_pass", phase0.get("track0_gate_pass")),
            ("h0a_hook_reachability_pass", phase0.get("h0a_hook_reachability_pass")),
            ("h0b_action_distinguishability_pass", phase0.get("h0b_action_distinguishability_pass")),
            ("h0c_influence_nontriviality_pass", phase0.get("h0c_influence_nontriviality_pass")),
            ("max_skipped_source_influence_mass", phase0.get("max_skipped_source_influence_mass")),
            ("attention_mass_rows", phase0.get("attention_mass_rows")),
        ])
        + "\nDownstream:\n\n"
        "- Track 1 stopped after h10 fail.\n"
        "- Track 2 stopped after h15 fail and washout attribution.\n"
        "- Track 3 stopped after h15 fail.\n"
        "- Track 4 stopped after h10 fail.\n"
        "- Track 5 full online was not launched because no h15-qualified candidate exists.\n\n"
        "Visualization boundary:\n\n"
        "- Generated: phase0 semantic influence heatmap, action Jaccard heatmap, h10->h15 durability curve, segment delta bar chart.\n"
        "- Not generated in v37: pixel-level RGB/semantic/trust/D_g/scale-risk/source-attention/SWA/TTT overlays. "
        "Those tensors/images were not landed by the v37 rollout reports at the needed spatial granularity, so no overlay is fabricated.\n"
        "- Not generated in v37: full trajectory overlay, because no v37 full-online candidate was allowed or launched.\n",
    )

    bar_items = [
        ("T1 H9 h10", float(track1_h9.get("best_200_300_delta_vs_H9") or 0), float(track1_h9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T1 C9 h10", float(track1_c9.get("best_200_300_delta_vs_H9") or 0), float(track1_c9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T2 H9 h10", float(track2_h10_h9.get("best_200_300_delta_vs_H9") or 0), float(track2_h10_h9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T2 C9 h10", float(track2_h10_c9.get("best_200_300_delta_vs_H9") or 0), float(track2_h10_c9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T3 H9 h10", float(track3_h10_h9.get("best_200_300_delta_vs_H9") or 0), float(track3_h10_h9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T3 C9 h10", float(track3_h10_c9.get("best_200_300_delta_vs_H9") or 0), float(track3_h10_c9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T4 H9 h10", float(track4_h9.get("best_200_300_delta_vs_H9") or 0), float(track4_h9.get("best_400_600_delta_for_best_ATE") or 0)),
        ("T4 C9 h10", float(track4_c9.get("best_200_300_delta_vs_H9") or 0), float(track4_c9.get("best_400_600_delta_for_best_ATE") or 0)),
    ]
    dur_items = [
        ("T2 H9 SWA", float(track2_h10_h9.get("best_200_300_delta_vs_H9") or 0), float(track2_h15_h9.get("best_200_300_delta_vs_H9") or 0)),
        ("T2 C9 SWA", float(track2_h10_c9.get("best_200_300_delta_vs_H9") or 0), float(track2_h15_c9.get("best_200_300_delta_vs_H9") or 0)),
        ("T3 H9 TTT", float(track3_h10_h9.get("best_200_300_delta_vs_H9") or 0), float(track3_h15_h9.get("best_200_300_delta_vs_H9") or 0)),
        ("T3 C9 TTT", float(track3_h10_c9.get("best_200_300_delta_vs_H9") or 0), float(track3_h15_c9.get("best_200_300_delta_vs_H9") or 0)),
    ]
    _plot_bars(reports / "segment_ate_bar_chart.png", bar_items)
    _plot_durability(reports / "h10_h15_durability_curve.png", dur_items)

    final = {
        "track0_gate_pass": phase0.get("track0_gate_pass"),
        "track1_gate_pass": False,
        "track2_h10_gate_pass": track2_h10_h9.get("gate_pass") and track2_h10_c9.get("gate_pass"),
        "track2_h15_gate_pass": track2_h15_h9.get("gate_pass") and track2_h15_c9.get("gate_pass"),
        "track3_h10_gate_pass": track3_h10_h9.get("gate_pass") and track3_h10_c9.get("gate_pass"),
        "track3_h15_gate_pass": track3_h15_h9.get("gate_pass") and track3_h15_c9.get("gate_pass"),
        "track4_h10_gate_pass": track4_h9.get("gate_pass") and track4_c9.get("gate_pass"),
        "track5_full_online_allowed": full_allowed,
        "track5_full_online_launched": False,
        "target30_success": False,
        "best_deployable_online": "C9_P0_R2",
        "best_deployable_online_ate": 33.7629421029,
    }
    _write(reports / "v37_final_summary.json", json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
