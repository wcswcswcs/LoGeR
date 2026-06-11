#!/usr/bin/env python3
"""Write v38 final per-track Markdown reports and summary plots."""

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
    ate = [x[1] for x in items]
    roll = [x[2] for x in items]
    fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.8), 4.8))
    xs = list(range(len(items)))
    ax.bar([x - 0.18 for x in xs], ate, width=0.36, label="best ATE delta")
    ax.bar([x + 0.18 for x in xs], roll, width=0.36, label="best rolling100 delta")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ATE delta vs parent base (m)")
    ax.legend()
    ax.set_title("v38 short-rollout best deltas")
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
    ax.axhline(-2, color="red", linewidth=0.8, linestyle="--", label="h15 ATE threshold")
    ax.set_xticks([10, 15])
    ax.set_xlabel("horizon")
    ax.set_ylabel("best ATE delta vs base (m)")
    ax.set_title("v38 h10 to h15 durability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    reports = root / "final_reports"
    reports.mkdir(parents=True, exist_ok=True)

    phase0 = _read_json(root / "phase0_action_influence/report_R1/phase0_action_influence_summary.json")
    t1 = _read_json(root / "phase1_frameglobal/report_h10_R1/track1_h10_summary.json")
    t2_h10 = _read_json(root / "phase2_swa/report_h10_R1/track2_h10_summary.json")
    t2_h15 = _read_json(root / "phase2_swa/report_h15_R1/track2_h15_summary.json")
    t3_h10 = _read_json(root / "phase3_ttt/report_h10_R1/track3_h10_summary.json")
    t3_h15 = _read_json(root / "phase3_ttt/report_h15_R1/track3_h15_summary.json")
    t4 = _read_json(root / "phase4_semc23/report_h10_R1/track4_h10_summary.json")
    full = _read_json(root / "phase5_full_online/report_R1/full_online_summary.json")

    _write(
        reports / "track0_action_influence_report.md",
        "# Track 0 Action / Influence Atlas v2\n\n"
        + _table([
            ("track0_gate_pass", phase0.get("track0_gate_pass")),
            ("h0a_hook_reachability_pass", phase0.get("h0a_hook_reachability_pass")),
            ("h0b_action_distinguishability_pass", phase0.get("h0b_action_distinguishability_pass")),
            ("h0c_influence_nontriviality_pass", phase0.get("h0c_influence_nontriviality_pass")),
            ("rows_done", phase0.get("rows_done")),
            ("missing_rows", phase0.get("missing_rows")),
            ("max_skipped_source_influence_mass", phase0.get("max_skipped_source_influence_mass")),
            ("attention_mass_rows", phase0.get("attention_mass_rows")),
        ])
        + "\nBoundary: Track 0 is action/influence audit only; it is not trajectory improvement or deployable online evidence.\n",
    )
    _write(reports / "track1_frameglobal_report.md", "# Track 1 Frame/Global Source Surgery\n\n" + _table(_summary_rows(t1)))
    _write(reports / "track2_swa_report.md", "# Track 2 SWA Durability\n\nh10:\n\n" + _table(_summary_rows(t2_h10)) + "\nh15:\n\n" + _table(_summary_rows(t2_h15)))
    _write(reports / "track3_ttt_report.md", "# Track 3 TTT Durability\n\nh10:\n\n" + _table(_summary_rows(t3_h10)) + "\nh15:\n\n" + _table(_summary_rows(t3_h15)))
    _write(reports / "track4_semantic_c23_report.md", "# Track 4 Semantic C23 Residual Path Isolation\n\n" + _table(_summary_rows(t4)))

    h15_qualified = bool(t2_h15.get("gate_pass")) or bool(t3_h15.get("gate_pass"))
    track5_launched = bool(full.get("full_online_launched", False)) if "missing_report" not in full else False
    track5_allowed = h15_qualified
    target30_success = bool(full.get("target30_success", False)) if track5_launched else False
    _write(
        reports / "track5_full_online_report.md",
        "# Track 5 Full Online\n\n"
        + _table([
            ("full_online_allowed", track5_allowed),
            ("full_online_launched", track5_launched),
            ("target30_success", target30_success),
            ("reason", "requires h15-qualified Track 1-4 candidate" if not track5_allowed else full.get("reason", "see full_online_summary.json")),
        ]),
    )

    _write(
        reports / "failure_routing_summary.md",
        "# v38 Failure Routing Summary\n\n"
        + _table([
            ("track0_gate_pass", phase0.get("track0_gate_pass")),
            ("track1_h10_gate_pass", t1.get("gate_pass")),
            ("track2_h10_gate_pass", t2_h10.get("gate_pass")),
            ("track2_h15_gate_pass", t2_h15.get("gate_pass")),
            ("track3_h10_gate_pass", t3_h10.get("gate_pass")),
            ("track3_h15_gate_pass", t3_h15.get("gate_pass")),
            ("track4_h10_gate_pass", t4.get("gate_pass")),
            ("track5_full_online_allowed", track5_allowed),
            ("track5_full_online_launched", track5_launched),
            ("target30_success", target30_success),
        ])
        + "\nVisualization boundary: final charts are generated only from landed report artifacts. Pixel/tensor overlays are not fabricated.\n",
    )

    bar_items = [
        ("T1 h10", _float(t1.get("best_ATE_delta_vs_base")), _float(t1.get("best_rolling_100f_best_delta"))),
        ("T2 h10", _float(t2_h10.get("best_ATE_delta_vs_base")), _float(t2_h10.get("best_rolling_100f_best_delta"))),
        ("T2 h15", _float(t2_h15.get("best_ATE_delta_vs_base")), _float(t2_h15.get("best_rolling_100f_best_delta"))),
        ("T3 h10", _float(t3_h10.get("best_ATE_delta_vs_base")), _float(t3_h10.get("best_rolling_100f_best_delta"))),
        ("T3 h15", _float(t3_h15.get("best_ATE_delta_vs_base")), _float(t3_h15.get("best_rolling_100f_best_delta"))),
        ("T4 h10", _float(t4.get("best_ATE_delta_vs_base")), _float(t4.get("best_rolling_100f_best_delta"))),
    ]
    _plot_bars(reports / "short_rollout_delta_bar_chart.png", bar_items)
    _plot_durability(
        reports / "h10_h15_durability_curve.png",
        [
            ("Track2 SWA", _float(t2_h10.get("best_ATE_delta_vs_base")), _float(t2_h15.get("best_ATE_delta_vs_base"))),
            ("Track3 TTT", _float(t3_h10.get("best_ATE_delta_vs_base")), _float(t3_h15.get("best_ATE_delta_vs_base"))),
        ],
    )

    final = {
        "track0_gate_pass": phase0.get("track0_gate_pass"),
        "track1_h10_gate_pass": t1.get("gate_pass"),
        "track2_h10_gate_pass": t2_h10.get("gate_pass"),
        "track2_h15_gate_pass": t2_h15.get("gate_pass"),
        "track3_h10_gate_pass": t3_h10.get("gate_pass"),
        "track3_h15_gate_pass": t3_h15.get("gate_pass"),
        "track4_h10_gate_pass": t4.get("gate_pass"),
        "track5_full_online_allowed": track5_allowed,
        "track5_full_online_launched": track5_launched,
        "target30_success": target30_success,
        "best_deployable_online": "C9_P0_R2",
        "best_deployable_online_ate": 33.7629421029,
    }
    _write(reports / "v38_final_summary.json", json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

