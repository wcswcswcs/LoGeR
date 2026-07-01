#!/usr/bin/env python3
"""Merge v80 downstream probe runner outputs into one audit summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _load_commands(root: Path) -> list[dict[str, Any]]:
    path = root / "commands.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_root"] = str(root)
            rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, str]] = []
    state_rows: list[dict[str, str]] = []
    trajectory_rows: list[dict[str, str]] = []
    run_rows: list[dict[str, str]] = []
    commands: list[dict[str, Any]] = []
    for root in args.roots:
        case_rows.extend(_read_csv(root / "downstream_probe_case_rows.csv"))
        state_rows.extend(_read_csv(root / "downstream_probe_state_rows.csv"))
        trajectory_rows.extend(_read_csv(root / "downstream_probe_trajectory_diff_rows.csv"))
        run_rows.extend(_read_csv(root / "run_status.csv"))
        commands.extend(_load_commands(root))

    _write_csv(args.out_dir / "downstream_probe_case_rows.csv", case_rows)
    _write_csv(args.out_dir / "downstream_probe_state_rows.csv", state_rows)
    _write_csv(args.out_dir / "downstream_probe_trajectory_diff_rows.csv", trajectory_rows)
    _write_csv(args.out_dir / "run_status.csv", run_rows)
    with (args.out_dir / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for row in commands:
            handle.write(json.dumps(_clean(row), ensure_ascii=False, sort_keys=True) + "\n")

    bad_rows = [row for row in case_rows if row.get("group") == "bad_candidate"]
    good_rows = [row for row in case_rows if row.get("group") == "good_counterexample"]
    failed_runs = [row for row in run_rows if row.get("returncode") not in ("0", "", "None")]
    summary = {
        "schema": "acl2_v80_downstream_probe_combined_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "source_roots": [str(root) for root in args.roots],
        "out_dir": str(args.out_dir),
        "case_count": len(case_rows),
        "run_count": len(run_rows),
        "failed_runs": failed_runs,
        "case_rows": case_rows,
        "aggregates": {
            "bad_selected_downstream_max_mean": _mean(
                [_float(row.get("selected_downstream_max_abs_pose_value_diff_vs_LW1")) for row in bad_rows]
            ),
            "bad_control_downstream_max_mean": _mean(
                [_float(row.get("control_downstream_max_abs_pose_value_diff_vs_LW1")) for row in bad_rows]
            ),
            "bad_selected_minus_control_mean": _mean(
                [_float(row.get("selected_minus_control_downstream_max")) for row in bad_rows]
            ),
            "good_selected_downstream_max_mean": _mean(
                [_float(row.get("selected_downstream_max_abs_pose_value_diff_vs_LW1")) for row in good_rows]
            ),
            "good_control_downstream_max_mean": _mean(
                [_float(row.get("control_downstream_max_abs_pose_value_diff_vs_LW1")) for row in good_rows]
            ),
            "good_selected_minus_control_mean": _mean(
                [_float(row.get("selected_minus_control_downstream_max")) for row in good_rows]
            ),
        },
        "outputs": {
            "case_rows_csv": str(args.out_dir / "downstream_probe_case_rows.csv"),
            "state_rows_csv": str(args.out_dir / "downstream_probe_state_rows.csv"),
            "trajectory_diff_rows_csv": str(args.out_dir / "downstream_probe_trajectory_diff_rows.csv"),
            "run_status_csv": str(args.out_dir / "run_status.csv"),
            "commands_jsonl": str(args.out_dir / "commands.jsonl"),
            "summary_json": str(args.out_dir / "downstream_probe_summary.json"),
        },
    }
    (args.out_dir / "downstream_probe_summary.json").write_text(
        json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
