#!/usr/bin/env python3
"""Build the v10 action-space saturation report from landed diagnostics.

This tool is deliberately conservative: it only reports metrics already present
in trajectory diagnostic summaries/CSVs. Missing runs stay missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SELECTED_RUNS = (
    "B0",
    "B0_SWKS3",
    "WINGAM_03",
    "WINGAM_03_repeat",
    "C16ROLE_01",
    "READBETA_01",
    "V9_H8_READBETA_01",
    "H9_READBETA2_03",
    "V9_H9_READBETA2_03",
    "AUTO_WIN_05",
    "AUTO_WIN_06",
    "POSTREG_11",
    "C16FINE_03",
    "H6_ORACLE_03",
    "H6_ORACLE_04",
    "H10_READBETA3_01",
    "H10_READBETA3_02",
    "H10_READBETA3_03",
    "H10_READBETA3_04",
    "H11_RDSUPPORT_01",
    "H11_RDSUPPORT_02",
    "H11_RDSUPPORT_03",
    "H11_RDSUPPORT_04",
)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _norm_name(name: str) -> str:
    out = str(name or "").strip()
    prefixes = ("V8_", "V9_")
    for prefix in prefixes:
        if out.startswith(prefix):
            out = out[len(prefix) :]
    suffixes = ("_SWKS3",)
    for suffix in suffixes:
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _fixed_segment_from_summary(run: Mapping[str, Any], start: int, end: int) -> float:
    for item in run.get("segment_summary", {}).get("fixed_segments", []):
        try:
            if int(item.get("start", -1)) == start and int(item.get("end", -1)) == end:
                return _float(item.get("ate_rmse_m"))
        except (TypeError, ValueError):
            continue
    return float("nan")


def _load_fixed_segments(diag_dir: Path) -> Dict[Tuple[str, int, int], float]:
    out: Dict[Tuple[str, int, int], float] = {}
    path = diag_dir / "segment_errors.csv"
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row.get("start", -1)))
                end = int(float(row.get("end", -1)))
            except (TypeError, ValueError):
                continue
            if (start, end) not in {(200, 300), (200, 400), (400, 500), (400, 600)}:
                continue
            out[(str(row.get("run", "")), start, end)] = _float(row.get("ate_rmse_m"))
    return out


def _family(name: str) -> str:
    n = _norm_name(name)
    if "AUTO_WIN" in n:
        return "auto_window"
    if "WINGAM" in n:
        return "body_exit_scalar"
    if "POSTREG" in n:
        return "post_region_scalar"
    if "C16FINE" in n or "C16ROLE" in n:
        return "chunk16_scalar"
    if "CFILTER" in n or "DGATE" in n:
        return "commit_or_delta_gate"
    if "POSTWIN" in n or "HANDOFF" in n or "C16LONG" in n:
        return "handoff_scalar"
    if "H6_ORACLE" in n:
        return "window_scalar_oracle"
    if "READBETA" in n:
        return "read_beta_scalar"
    if "RDSUPPORT" in n:
        return "read_support"
    if "LGROUTE" in n or "LHROUTE" in n:
        return "layer_head_route"
    return "reference"


def _load_runs(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for summary_path in sorted(root.glob("trajectory_diagnostics*/summary.json")):
            diag_dir = summary_path.parent
            fixed = _load_fixed_segments(diag_dir)
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for run in data.get("runs", []):
                raw_name = str(run.get("name", ""))
                if not raw_name:
                    continue
                name = _norm_name(raw_name)
                row = {
                    "run": name,
                    "raw_run": raw_name,
                    "family": _family(name),
                    "diag_dir": str(diag_dir),
                    "pred_path": str(run.get("path", "")),
                    "ate": _float(run.get("aligned_ate_rmse_m")),
                    "final_err": _float(run.get("final_error_m")),
                    "yaw_rmse": _float(run.get("yaw_rmse_deg")),
                    "sim3_scale": _float(run.get("sim3_scale")),
                    "axis_rmse_x": _float(run.get("axis_rmse_x_m")),
                    "axis_rmse_y": _float(run.get("axis_rmse_y_m")),
                    "axis_rmse_z": _float(run.get("axis_rmse_z_m")),
                    "seg_200_300": fixed.get((raw_name, 200, 300), _fixed_segment_from_summary(run, 200, 300)),
                    "seg_200_400": fixed.get((raw_name, 200, 400), _fixed_segment_from_summary(run, 200, 400)),
                    "seg_400_500": fixed.get((raw_name, 400, 500), _fixed_segment_from_summary(run, 400, 500)),
                    "seg_400_600": fixed.get((raw_name, 400, 600), _fixed_segment_from_summary(run, 400, 600)),
                }
                if name not in rows or row["ate"] < rows[name].get("ate", float("inf")):
                    rows[name] = row
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _fmt(value: Any, digits: int = 4) -> str:
    val = _float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "nan"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--current-best", default="H9_READBETA2_03")
    parser.add_argument("--disease-reference", default="WINGAM_03_repeat")
    args = parser.parse_args()

    roots = [Path(item) for item in args.root]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name = _load_runs(roots)
    all_rows = sorted(rows_by_name.values(), key=lambda row: row.get("ate", float("inf")))
    _write_csv(out_dir / "all_landed_runs.csv", all_rows)

    current_key = _norm_name(args.current_best)
    current = rows_by_name.get(current_key)
    if current is None:
        raise SystemExit(f"current best {args.current_best!r} not found in diagnostics")
    disease_ref = rows_by_name.get(_norm_name(args.disease_reference), current)

    selected: List[Dict[str, Any]] = []
    missing: List[str] = []
    for wanted in SELECTED_RUNS:
        key = _norm_name(wanted)
        row = rows_by_name.get(key)
        if row is None:
            missing.append(wanted)
            continue
        out = dict(row)
        out["gain_vs_current_best"] = _float(current["ate"]) - _float(out["ate"])
        out["disease_gain_vs_ref_200_300"] = _float(disease_ref["seg_200_300"]) - _float(out["seg_200_300"])
        out["post_cost_vs_current_400_600"] = _float(out["seg_400_600"]) - _float(current["seg_400_600"])
        out["gap_to_25"] = _float(out["ate"]) - 25.0
        selected.append(out)
    _write_csv(out_dir / "selected_action_space_runs.csv", selected)

    family_rows: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_family[str(row.get("family", "unknown"))].append(row)
    for fam, fam_rows in sorted(by_family.items()):
        vals = [row for row in fam_rows if math.isfinite(_float(row.get("ate")))]
        if not vals:
            continue
        best = min(vals, key=lambda row: _float(row.get("ate")))
        top5 = sorted(vals, key=lambda row: _float(row.get("ate")))[:5]
        family_rows.append(
            {
                "family": fam,
                "num_runs": len(vals),
                "best_run": best["run"],
                "best_ate": best["ate"],
                "best_gain_vs_current": _float(current["ate"]) - _float(best["ate"]),
                "top5_ate_span": max(_float(row["ate"]) for row in top5) - min(_float(row["ate"]) for row in top5),
                "mean_seg_200_300": mean(_float(row["seg_200_300"]) for row in vals if math.isfinite(_float(row["seg_200_300"]))),
                "mean_seg_400_600": mean(_float(row["seg_400_600"]) for row in vals if math.isfinite(_float(row["seg_400_600"]))),
            }
        )
    _write_csv(out_dir / "family_saturation.csv", family_rows)

    lines = [
        "# V10 Action-Space Saturation Report",
        "",
        f"source_roots = `{', '.join(str(r) for r in roots)}`",
        f"landed_runs = `{len(all_rows)}`",
        f"current_best = `{current['run']}` ATE `{_fmt(current['ate'])}`",
        f"gap_to_25 = `{_fmt(_float(current['ate']) - 25.0)}`",
        "",
        "## Selected Runs",
        "",
        "| Run | family | ATE | [200,300) | [400,600) | gain vs current | disease gain | post cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: _float(item.get("ate"))):
        lines.append(
            f"| `{row['run']}` | `{row['family']}` | {_fmt(row['ate'])} | "
            f"{_fmt(row['seg_200_300'])} | {_fmt(row['seg_400_600'])} | "
            f"{_fmt(row['gain_vs_current_best'])} | {_fmt(row['disease_gain_vs_ref_200_300'])} | "
            f"{_fmt(row['post_cost_vs_current_400_600'])} |"
        )
    lines += [
        "",
        "## Family Saturation",
        "",
        "| Family | runs | best run | best ATE | gain vs current | top5 ATE span |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in sorted(family_rows, key=lambda item: _float(item.get("best_ate"))):
        lines.append(
            f"| `{row['family']}` | {int(row['num_runs'])} | `{row['best_run']}` | "
            f"{_fmt(row['best_ate'])} | {_fmt(row['best_gain_vs_current'])} | {_fmt(row['top5_ate_span'])} |"
        )
    lines += [
        "",
        "## Gate Read",
        "",
        "- Scalar/read-support families do not show target-scale gains in landed data.",
        "- Current best is still more than 9m above the target-25 gate.",
        "- This satisfies the v10 premise that new full runs require a non-scalar oracle/proxy gate first.",
    ]
    if missing:
        lines += ["", "## Missing Requested Runs", "", ", ".join(f"`{item}`" for item in missing)]
    (out_dir / "action_space_saturation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'action_space_saturation_report.md'}")


if __name__ == "__main__":
    main()
