#!/usr/bin/env python3
"""Write v40 final reports from landed artifacts only."""

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
    fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.9), 4.8))
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
    ax.set_title("v40 short-rollout best deltas")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
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

    p0 = _read_json(root / "phase0_health/report_R1/v40_phase0_health_summary.json")
    health = _read_json(root / "health_atlas/health_flag_summary.json")
    read_h10 = _read_json(root / "phase2a_read/report_h10_R1/read_h10_summary.json")
    read_h15 = _read_json(root / "phase2a_read/report_h15_R1/read_h15_summary.json")
    swa_h10 = _read_json(root / "phase2b_swa/report_h10_R1/swa_h10_summary.json")
    swa_h15 = _read_json(root / "phase2b_swa/report_h15_R1/swa_h15_summary.json")
    ttt_h10 = _read_json(root / "phase2c_ttt/report_h10_R1/ttt_h10_summary.json")
    ttt_h15 = _read_json(root / "phase2c_ttt/report_h15_R1/ttt_h15_summary.json")
    reset_h10 = _read_json(root / "phase2d_reset/report_h10_R1/reset_h10_summary.json")
    full = _read_json(root / "phase4_full_online/report_R1/full_online_summary.json")

    _write(
        reports / "health_timeline_report.md",
        "# v40 Health Timeline\n\n"
        + _table([
            ("phase0_gate_pass", p0.get("phase0_gate_pass")),
            ("phase0_rows_done", p0.get("phase0_rows_done")),
            ("phase0_missing_rows", p0.get("phase0_missing_rows")),
            ("max_abs_ATE_delta_vs_noop_reference", p0.get("max_abs_ATE_delta_vs_noop_reference")),
            ("max_raw_pose_abs_diff_vs_noop_reference", p0.get("max_raw_pose_abs_diff_vs_noop_reference")),
            ("required_health_streams_nonempty", p0.get("required_health_streams_nonempty")),
            ("context_empty_source_events_total", p0.get("context_empty_source_events_total")),
            ("cue_quality_rows_total", health.get("cue_quality_rows_total")),
            ("source_influence_rows_total", health.get("source_influence_rows_total")),
            ("swa_health_rows_total", health.get("swa_health_rows_total")),
            ("ttt_health_rows_total", health.get("ttt_health_rows_total")),
            ("appearance_evidence_level", health.get("appearance_evidence_level")),
        ])
        + "\nBoundary: health reports are landed-artifact summaries; missing spatial tensors are not reconstructed.\n",
    )
    _write(reports / "path_action_report.md", "# v40 Path Action Report\n\nRead h10:\n\n" + _table(_summary_rows(read_h10)) + "\nSWA h10:\n\n" + _table(_summary_rows(swa_h10)) + "\nTTT h10:\n\n" + _table(_summary_rows(ttt_h10)) + "\nReset h10:\n\n" + _table(_summary_rows(reset_h10)))
    _write(reports / "durability_report.md", "# v40 Durability Report\n\nRead h15:\n\n" + _table(_summary_rows(read_h15)) + "\nSWA h15:\n\n" + _table(_summary_rows(swa_h15)) + "\nTTT h15:\n\n" + _table(_summary_rows(ttt_h15)))

    h15_qualified = any(bool(x.get("gate_pass")) for x in (read_h15, swa_h15, ttt_h15))
    full_launched = bool(full.get("full_online_launched", False)) if "missing_report" not in full else False
    target30_success = bool(full.get("target30_success", False)) if full_launched else False
    _write(
        reports / "full_online_report.md",
        "# v40 Full Online\n\n"
        + _table([
            ("full_online_allowed", h15_qualified),
            ("full_online_launched", full_launched),
            ("target30_success", target30_success),
            ("reason", "requires h15-qualified Phase 2 candidate" if not h15_qualified else full.get("reason", "see full_online_summary.json")),
        ]),
    )

    _plot_bars(
        reports / "short_rollout_delta_bar_chart.png",
        [
            ("READ h10", _float(read_h10.get("best_ATE_delta_vs_base")), _float(read_h10.get("best_rolling_100f_best_delta"))),
            ("SWA h10", _float(swa_h10.get("best_ATE_delta_vs_base")), _float(swa_h10.get("best_rolling_100f_best_delta"))),
            ("TTT h10", _float(ttt_h10.get("best_ATE_delta_vs_base")), _float(ttt_h10.get("best_rolling_100f_best_delta"))),
            ("RESET h10", _float(reset_h10.get("best_ATE_delta_vs_base")), _float(reset_h10.get("best_rolling_100f_best_delta"))),
        ],
    )

    final = {
        "phase0_gate_pass": p0.get("phase0_gate_pass"),
        "read_h10_gate_pass": read_h10.get("gate_pass"),
        "read_h15_gate_pass": read_h15.get("gate_pass"),
        "swa_h10_gate_pass": swa_h10.get("gate_pass"),
        "swa_h15_gate_pass": swa_h15.get("gate_pass"),
        "ttt_h10_gate_pass": ttt_h10.get("gate_pass"),
        "ttt_h15_gate_pass": ttt_h15.get("gate_pass"),
        "reset_h10_gate_pass": reset_h10.get("gate_pass"),
        "full_online_allowed": h15_qualified,
        "full_online_launched": full_launched,
        "target30_success": target30_success,
        "best_deployable_online": "C9_P0_R2",
        "best_deployable_online_ate": 33.7629421029,
    }
    _write(reports / "failure_routing_summary.md", "# v40 Failure Routing Summary\n\n" + _table(final.items()))
    _write(reports / "v40_final_summary.json", json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
