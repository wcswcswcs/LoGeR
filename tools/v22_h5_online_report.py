#!/usr/bin/env python3
"""Aggregate ACL2 v22 H5 full-online explicit scale-state diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v11_metric_registry import _online_row  # noqa: E402


H9_REF = {
    "ate_rmse": 34.1257769401,
    "seg_200_300": 74.409927,
    "seg_400_600": 44.353638,
}
C9_REF = {
    "ate_rmse": 33.7629421029,
    "seg_200_300": 76.102136,
    "seg_400_600": 41.896364,
}


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _read_status(run_dir: Path) -> str:
    path = run_dir / "run_status.txt"
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "DONE " in text:
        return "done"
    if "FAIL " in text:
        return "fail"
    return "incomplete"


def _read_manifest(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "v22_h5_candidate_manifest.yaml"
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_clean(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


def _metric_row(name: str, run_dir: Path, gt_path: Path) -> Dict[str, object]:
    row = _online_row(name, run_dir, gt_path)
    manifest = _read_manifest(run_dir)
    row["candidate_id"] = manifest.get("candidate_id", name)
    row["online_scale_state_mode"] = manifest.get("online_scale_state_mode", "")
    row["run_status"] = _read_status(run_dir)
    row["result_class"] = "full_online_explicit_scale_state_diagnostic"
    row["diagnostic_only"] = True
    row["counts_as_ttt_write"] = False
    row["counts_as_online_ttt_write_success"] = False
    row["counts_as_target25_success"] = False
    row["no_gt_runtime_action"] = True
    row["no_external_trajectory_rewrite"] = True
    row["output_from_online_hmc"] = True
    row["ate_delta_vs_H9"] = _to_float(row.get("ate_rmse")) - H9_REF["ate_rmse"]
    row["ate_delta_vs_C9"] = _to_float(row.get("ate_rmse")) - C9_REF["ate_rmse"]
    row["seg_200_300_delta_vs_H9"] = _to_float(row.get("seg_200_300")) - H9_REF["seg_200_300"]
    row["seg_400_600_delta_vs_H9"] = _to_float(row.get("seg_400_600")) - H9_REF["seg_400_600"]
    row["target25_ate_pass"] = _to_float(row.get("ate_rmse")) <= 25.0
    row["stage_signal_pass"] = (
        _to_float(row["ate_delta_vs_H9"]) <= -3.0
        or _to_float(row["seg_200_300_delta_vs_H9"]) <= -5.0
    )
    row["downstream_proxy_pass"] = (
        not math.isfinite(_to_float(row["seg_400_600_delta_vs_H9"]))
        or _to_float(row["seg_400_600_delta_vs_H9"]) <= 1.0
    )
    row["selector_allowed"] = False
    row["full_online_validation_allowed"] = False
    row["note"] = "full online explicit scale-state diagnostic; not counted as deployable TTT write success"
    return row


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    done_rows = [row for row in rows if row.get("run_status") == "done"]
    best_ate = min(done_rows, key=lambda row: _to_float(row.get("ate_rmse")), default=None)
    best_segment = min(done_rows, key=lambda row: _to_float(row.get("seg_200_300")), default=None)
    any_target25 = any(bool(row.get("target25_ate_pass")) for row in done_rows)
    any_stage_signal = any(bool(row.get("stage_signal_pass")) and bool(row.get("downstream_proxy_pass")) for row in done_rows)
    return {
        "num_runs": len(rows),
        "num_done": len(done_rows),
        "all_done": len(done_rows) == len(rows) and bool(rows),
        "best_ate_run": best_ate.get("run") if best_ate else "",
        "best_ate_candidate": best_ate.get("candidate_id") if best_ate else "",
        "best_ate_rmse": _to_float(best_ate.get("ate_rmse")) if best_ate else float("nan"),
        "best_ate_delta_vs_H9": _to_float(best_ate.get("ate_delta_vs_H9")) if best_ate else float("nan"),
        "best_ate_delta_vs_C9": _to_float(best_ate.get("ate_delta_vs_C9")) if best_ate else float("nan"),
        "best_200_300_run": best_segment.get("run") if best_segment else "",
        "best_200_300_candidate": best_segment.get("candidate_id") if best_segment else "",
        "best_200_300": _to_float(best_segment.get("seg_200_300")) if best_segment else float("nan"),
        "best_200_300_delta_vs_H9": _to_float(best_segment.get("seg_200_300_delta_vs_H9")) if best_segment else float("nan"),
        "any_target25_ate_pass": any_target25,
        "any_stage_signal_pass": any_stage_signal,
        "counts_as_deployable_ttt_write_success": False,
        "selector_started": False,
        "full_online_validation_started_from_selector": False,
    }


def _write_md(path: Path, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    lines = [
        "# ACL2 v22 H5 Online Scale-State Report",
        "",
        "These rows are full-online explicit scale-state diagnostics. They are not counted as deployable TTT write success.",
        "",
        "## Summary",
        "",
        f"- Runs done: `{summary['num_done']}/{summary['num_runs']}`",
        f"- Best ATE run: `{summary['best_ate_run']}`",
        f"- Best ATE: `{summary['best_ate_rmse']}`",
        f"- Best ATE delta vs H9: `{summary['best_ate_delta_vs_H9']}`",
        f"- Best ATE delta vs C9: `{summary['best_ate_delta_vs_C9']}`",
        f"- Best [200,300) run: `{summary['best_200_300_run']}`",
        f"- Best [200,300): `{summary['best_200_300']}`",
        f"- Best [200,300) delta vs H9: `{summary['best_200_300_delta_vs_H9']}`",
        f"- Any Target-25 ATE pass: `{str(summary['any_target25_ate_pass']).lower()}`",
        f"- Counts as deployable TTT write success: `{str(summary['counts_as_deployable_ttt_write_success']).lower()}`",
        "",
        "## Runs",
        "",
        "| Candidate | Status | ATE | Delta vs H9 | Delta vs C9 | [200,300) delta | [400,600) delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{candidate}` | `{status}` | `{ate}` | `{d_h9}` | `{d_c9}` | `{d_200}` | `{d_400}` |".format(
                candidate=row.get("candidate_id", ""),
                status=row.get("run_status", ""),
                ate=row.get("ate_rmse", ""),
                d_h9=row.get("ate_delta_vs_H9", ""),
                d_c9=row.get("ate_delta_vs_C9", ""),
                d_200=row.get("seg_200_300_delta_vs_H9", ""),
                d_400=row.get("seg_400_600_delta_vs_H9", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--run", action="append", default=[], help="NAME=run_dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    gt_path = Path(args.gt)
    rows: List[Dict[str, object]] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be NAME=run_dir, got {item!r}")
        name, path_s = item.split("=", 1)
        rows.append(_metric_row(name, Path(path_s), gt_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    _write_csv(out_dir / "h5_online_scale_registry.csv", rows)
    _write_csv(out_dir / "h5_online_scale_gate_summary.csv", [summary])
    (out_dir / "h5_online_scale_registry.json").write_text(
        json.dumps(_json_clean(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "h5_online_scale_gate_summary.json").write_text(
        json.dumps(_json_clean(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_md(out_dir / "h5_online_scale_report.md", rows, summary)
    print(f"Wrote H5 report to {out_dir}")


if __name__ == "__main__":
    main()
