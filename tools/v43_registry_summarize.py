#!/usr/bin/env python3
"""Summarize v43 full-online registries into phase-level audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HISTORICAL_C9_ATE = 33.7629421029


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    val = _float(value)
    return "" if val is None else f"{val:.10f}"


def _best(rows: List[Dict[str, Any]], reference_name: str) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if r.get("status") == "done" and r.get("name") != reference_name]
    if not candidates:
        return None
    return min(candidates, key=lambda r: _float(r.get("ATE_full")) if _float(r.get("ATE_full")) is not None else float("inf"))


def _phase_summary(rows: List[Dict[str, Any]], reference_name: str, historical_c9_ate: float) -> Dict[str, Any]:
    ref = next((r for r in rows if r.get("name") == reference_name), None)
    best = _best(rows, reference_name)
    ref_ate = _float(ref.get("ATE_full")) if ref else None
    best_ate = _float(best.get("ATE_full")) if best else None
    best_delta = None
    if best_ate is not None and ref_ate is not None:
        best_delta = best_ate - ref_ate
    return {
        "rows": len(rows),
        "done_rows": sum(1 for r in rows if r.get("status") == "done"),
        "reference_name": reference_name,
        "reference_ATE_full": ref_ate,
        "historical_c9_ate": historical_c9_ate,
        "best_candidate": best.get("name") if best else None,
        "best_ATE_full": best_ate,
        "best_delta_vs_reference": best_delta,
        "best_delta_vs_historical_c9": (best_ate - historical_c9_ate) if best_ate is not None else None,
        "minimum_progress_pass": bool(best_ate is not None and (best_ate <= 33.3 or (best_delta is not None and best_delta <= -0.5))),
        "stage_success_pass": bool(best_ate is not None and best_ate <= 33.0),
        "strong_success_pass": bool(best_ate is not None and best_ate <= 32.0),
        "target30_success": bool(best_ate is not None and best_ate <= 30.0),
    }


def _component_rows(rows: List[Dict[str, Any]], reference_name: str) -> List[Dict[str, Any]]:
    ref = next((r for r in rows if r.get("name") == reference_name), None)
    if not ref:
        return []
    ref_ate = _float(ref.get("ATE_full"))
    ref_200 = _float(ref.get("segment_200_300_ATE"))
    ref_400 = _float(ref.get("segment_400_600_ATE"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("name") == reference_name or row.get("status") != "done":
            continue
        ate = _float(row.get("ATE_full"))
        s200 = _float(row.get("segment_200_300_ATE"))
        s400 = _float(row.get("segment_400_600_ATE"))
        delta = (ate - ref_ate) if ate is not None and ref_ate is not None else None
        if delta is None:
            cls = "missing"
        elif delta >= 0.5:
            cls = "major_positive_component"
        elif delta >= 0.2:
            cls = "moderate_positive_component"
        elif delta <= -0.2:
            cls = "harmful_or_conflicting_component"
        else:
            cls = "neutral_component"
        out.append(
            {
                "candidate": row.get("name"),
                "ATE_full": ate,
                "ATE_delta_vs_C9": delta,
                "component_class": cls,
                "segment_200_300_delta_vs_C9": (s200 - ref_200) if s200 is not None and ref_200 is not None else None,
                "segment_400_600_delta_vs_C9": (s400 - ref_400) if s400 is not None and ref_400 is not None else None,
                "rolling100_best_delta_vs_C9": _float(row.get("rolling100_best_delta_vs_reference")),
                "rolling100_mean_delta_vs_C9": _float(row.get("rolling100_mean_delta_vs_reference")),
                "rolling100_p90_delta_vs_C9": _float(row.get("rolling100_p90_delta_vs_reference")),
            }
        )
    return out


def _maybe_plot(out_dir: Path, component_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not component_rows:
        return
    names = [str(r["candidate"]) for r in component_rows]
    vals = [_float(r.get("ATE_delta_vs_C9")) or 0.0 for r in component_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    colors = ["#4c78a8" if v >= 0 else "#f58518" for v in vals]
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("ATE delta vs C9 (m)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "component_ate_waterfall.png", dpi=160)
    plt.close(fig)

    seg200 = [_float(r.get("segment_200_300_delta_vs_C9")) or 0.0 for r in component_rows]
    seg400 = [_float(r.get("segment_400_600_delta_vs_C9")) or 0.0 for r in component_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    x = list(range(len(names)))
    ax.bar([i - 0.18 for i in x], seg200, width=0.36, label="[200,300)")
    ax.bar([i + 0.18 for i in x], seg400, width=0.36, label="[400,600)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Segment ATE delta vs C9 (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "component_segment_delta_bar.png", dpi=160)
    plt.close(fig)


def _write_md(path: Path, phase_name: str, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# ACL2 v43 {phase_name} Report",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Name | Status | ATE | Delta vs Ref | [200,300) | [400,600) | Target30 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.get('name')}` | `{row.get('status')}` | {_fmt(row.get('ATE_full'))} | "
            f"{_fmt(row.get('ATE_delta_vs_reference'))} | {_fmt(row.get('segment_200_300_ATE'))} | "
            f"{_fmt(row.get('segment_400_600_ATE'))} | `{row.get('target30_pass')}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--phase-name", required=True)
    parser.add_argument("--reference-name", default="F0")
    parser.add_argument("--historical-c9-ate", type=float, default=HISTORICAL_C9_ATE)
    parser.add_argument("--component-ledger", action="store_true")
    args = parser.parse_args()

    rows = _read_csv(args.registry)
    summary = _phase_summary(rows, args.reference_name, args.historical_c9_ate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / f"{args.phase_name}_summary.json", summary)
    _write_md(args.out_dir / f"{args.phase_name}_report.md", args.phase_name, summary, rows)
    if args.component_ledger:
        comp = _component_rows(rows, args.reference_name)
        _write_csv(args.out_dir / "component_contribution_ate.csv", comp)
        _write_csv(args.out_dir / "component_contribution_segments.csv", comp)
        _write_csv(args.out_dir / "component_contribution_rolling.csv", comp)
        _maybe_plot(args.out_dir, comp)
        notes = [
            "# Component Interaction Notes",
            "",
            "Classification uses leave-one-out ATE delta vs locked C9:",
            "",
            "- `>= +0.5m`: major positive component",
            "- `+0.2m .. +0.5m`: moderate positive component",
            "- `< 0.2m` absolute: neutral component",
            "- `<= -0.2m`: harmful or conflicting component",
            "",
            "| Candidate | ATE delta vs C9 | Class |",
            "|---|---:|---|",
        ]
        for row in comp:
            notes.append(f"| `{row.get('candidate')}` | {_fmt(row.get('ATE_delta_vs_C9'))} | `{row.get('component_class')}` |")
        notes.append("")
        (args.out_dir / "component_interaction_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
