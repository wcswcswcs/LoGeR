#!/usr/bin/env python3
"""Build a diagnostic v80 seq01 non-GT proxy-controller audit.

The controller is intentionally conservative and offline-only. It materializes
a virtual PhaseE root by choosing either the existing qscale run or native run
per chunk, then leaves the normal evaluator to score that root. The rules use
only run-time style signals from traces/trajectories plus the previously
audited selected-write low-support diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_QSCALE_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
DEFAULT_ALIGNMENT = REPORT_ROOT / "phase9_seq01_error_ttt_semantic_alignment_rediscovery" / (
    "canary_error_ttt_semantic_alignment_rows.csv"
)
DEFAULT_OUT_ROOT = REPORT_ROOT / "phase9_seq01_proxy_controller_v1_canary5"
DEFAULT_RISKBUDGET_OUT_ROOT = REPORT_ROOT / "phase9_seq01_proxy_controller_v1b_riskbudget_canary5"
DEFAULT_CHUNKS = "6,7,8,10,12"
PROFILE_TO_RUN = {
    "conservative": "proxy_controller_v1",
    "risk_budget": "proxy_controller_v1b",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qscale-root", type=Path, default=DEFAULT_QSCALE_ROOT)
    parser.add_argument("--alignment-csv", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--profile", choices=sorted(PROFILE_TO_RUN), default="conservative")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.profile == "risk_budget" and args.out_root == DEFAULT_OUT_ROOT:
        args.out_root = DEFAULT_RISKBUDGET_OUT_ROOT
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _alignment_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk = int(row["chunk"])
        out[chunk] = {
            "selected_runtime_mass": _safe_float(row.get("selected_runtime_mass")),
            "selected_low_support_mass": _safe_float(row.get("selected_low_support_mass")),
            "selected_low_support_enrichment_vs_global": _safe_float(
                row.get("selected_low_support_enrichment_vs_global")
            ),
        }
    return out


def _trace_row(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "merge_state_trace.jsonl")
    selected = [row for row in rows if int(row.get("local_chunk_idx", -1)) == 1]
    return selected[-1] if selected else {}


def _positions(path: Path) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(part) for part in line.split()]
            if len(parts) >= 8:
                out.append((parts[1], parts[2], parts[3]))
            elif len(parts) == 12:
                out.append((parts[3], parts[7], parts[11]))
    return out


def _norm3(vec: tuple[float, float, float]) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _step_lengths(pos: list[tuple[float, float, float]]) -> list[float]:
    return [
        _norm3((pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1], pos[i + 1][2] - pos[i][2]))
        for i in range(len(pos) - 1)
    ]


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    out = num / den
    return out if math.isfinite(out) else None


def _trajectory_features(run_dir: Path, native_dir: Path) -> dict[str, Any]:
    cand_pos = _positions(run_dir / "01.txt")
    native_pos = _positions(native_dir / "01.txt")
    cand_steps = _step_lengths(cand_pos)
    native_steps = _step_lengths(native_pos)

    boundary_idx = 28 if len(cand_steps) > 28 else max(len(cand_steps) // 2, 0)

    def stats(steps: list[float]) -> dict[str, float | None]:
        if not steps:
            return {
                "median_step": None,
                "pre_boundary_median_step": None,
                "boundary_step": None,
                "post_boundary_median_step": None,
                "boundary_over_pre": None,
                "post_over_pre": None,
                "boundary_accel_abs": None,
            }
        pre = steps[max(0, boundary_idx - 10) : boundary_idx]
        post = steps[boundary_idx + 1 : min(len(steps), boundary_idx + 11)]
        pre_med = _median(pre)
        post_med = _median(post)
        boundary = steps[boundary_idx] if boundary_idx < len(steps) else None
        prev_step = steps[boundary_idx - 1] if boundary_idx > 0 and boundary_idx - 1 < len(steps) else None
        accel = abs(boundary - prev_step) if boundary is not None and prev_step is not None else None
        return {
            "median_step": _median(steps),
            "pre_boundary_median_step": pre_med,
            "boundary_step": boundary,
            "post_boundary_median_step": post_med,
            "boundary_over_pre": _ratio(boundary, pre_med),
            "post_over_pre": _ratio(post_med, pre_med),
            "boundary_accel_abs": accel,
        }

    cand = stats(cand_steps)
    native = stats(native_steps)
    return {
        "trajectory_path": str(run_dir / "01.txt"),
        "native_trajectory_path": str(native_dir / "01.txt"),
        "traj_boundary_idx": boundary_idx,
        "traj_candidate_boundary_over_pre": cand["boundary_over_pre"],
        "traj_candidate_post_over_pre": cand["post_over_pre"],
        "traj_candidate_boundary_accel_abs": cand["boundary_accel_abs"],
        "traj_native_boundary_over_pre": native["boundary_over_pre"],
        "traj_native_post_over_pre": native["post_over_pre"],
        "traj_native_boundary_accel_abs": native["boundary_accel_abs"],
        "traj_boundary_over_pre_delta_vs_native": (
            cand["boundary_over_pre"] - native["boundary_over_pre"]
            if cand["boundary_over_pre"] is not None and native["boundary_over_pre"] is not None
            else None
        ),
    }


def _decide(row: dict[str, Any], *, profile: str) -> tuple[str, str]:
    selected_low = _safe_float(row.get("selected_low_support_mass")) or 0.0
    scale = _safe_float(row.get("semantic_merge_scale")) or 1.0
    qobs = _safe_float(row.get("semantic_merge_qscale_observability")) or 0.0
    residual = _safe_float(row.get("semantic_merge_overlap_residual"))
    boundary_over_pre = _safe_float(row.get("traj_candidate_boundary_over_pre")) or 0.0

    if selected_low > 0.0 and scale >= 1.02 and qobs >= 0.50:
        return "keep_qscale", "selected_bad_write_positive_scale_expansion"
    if scale <= 0.98:
        return "use_native", "negative_scale_shrink_veto"
    if qobs < 0.50 and abs(math.log(max(scale, 1e-12))) < 0.02:
        return "use_native", "low_observability_low_gain_hold"
    if (
        profile == "risk_budget"
        and selected_low <= 0.0
        and 1.0 <= scale < 1.02
        and qobs >= 0.50
        and residual is not None
        and residual <= 0.14
    ):
        return "keep_qscale", "risk_budget_marginal_positive_low_residual_keep"
    if selected_low <= 0.0 and 1.0 <= scale < 1.02 and boundary_over_pre >= 2.0:
        return "use_native", "marginal_positive_scale_boundary_motion_hold"
    return "use_native", "no_proxy_keep_condition"


def _replace_symlink(link: Path, target: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or not link.exists():
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve(), target_is_directory=True)
        return
    raise FileExistsError(f"Refusing to replace non-symlink path: {link}")


def _materialize_root(qscale_root: Path, out_root: Path, rows: list[dict[str, Any]], *, proxy_run: str, dry_run: bool) -> None:
    passthrough_runs = [
        "native_no_swa",
        "geometry_only",
        "thingstuff_radio_qscale",
        "thingstuff_radio_qscale_random",
        "thingstuff_radio_qscale_shuffled",
    ]
    for row in rows:
        chunk = int(row["chunk"])
        src_chunk = qscale_root / f"chunk{chunk:02d}"
        dst_chunk = out_root / f"chunk{chunk:02d}"
        for run in passthrough_runs:
            _replace_symlink(dst_chunk / run, src_chunk / run, dry_run=dry_run)
        source_run = "thingstuff_radio_qscale" if row["proxy_action"] == "keep_qscale" else "native_no_swa"
        row["proxy_source_run"] = source_run
        _replace_symlink(dst_chunk / proxy_run, src_chunk / source_run, dry_run=dry_run)


def main() -> None:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    proxy_run = PROFILE_TO_RUN[args.profile]
    alignment = _alignment_by_chunk(args.alignment_csv)

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_dir = args.qscale_root / f"chunk{chunk:02d}"
        qscale_dir = chunk_dir / "thingstuff_radio_qscale"
        native_dir = chunk_dir / "native_no_swa"
        trace = _trace_row(qscale_dir)
        row: dict[str, Any] = {
            "chunk": int(chunk),
            "qscale_run_dir": str(qscale_dir),
            "native_run_dir": str(native_dir),
            "semantic_merge_scale": _safe_float(trace.get("semantic_merge_scale")),
            "semantic_merge_candidate_scale": _safe_float(trace.get("semantic_merge_candidate_scale")),
            "semantic_merge_overlap_residual": _safe_float(trace.get("semantic_merge_overlap_residual")),
            "semantic_merge_qscale_observability": _safe_float(trace.get("semantic_merge_qscale_observability")),
            "semantic_merge_radio_handoff_risk_mean": _safe_float(trace.get("semantic_merge_radio_handoff_risk_mean")),
            "semantic_merge_radio_handoff_stable_mean": _safe_float(trace.get("semantic_merge_radio_handoff_stable_mean")),
        }
        row.update(alignment.get(chunk, {}))
        row.update(_trajectory_features(qscale_dir, native_dir))
        action, reason = _decide(row, profile=args.profile)
        row["proxy_action"] = action
        row["proxy_reason"] = reason
        row["proxy_profile"] = args.profile
        row["diagnostic_only"] = True
        rows.append(row)

    _materialize_root(args.qscale_root, args.out_root, rows, proxy_run=proxy_run, dry_run=args.dry_run)

    keep_chunks = [int(row["chunk"]) for row in rows if row["proxy_action"] == "keep_qscale"]
    native_chunks = [int(row["chunk"]) for row in rows if row["proxy_action"] != "keep_qscale"]
    summary = {
        "schema": "acl2_v80_seq01_proxy_controller_audit_v1",
        "status": "diagnostic_virtual_controller_materialized" if not args.dry_run else "diagnostic_dry_run",
        "v80_goal_achieved": False,
        "diagnostic_only": True,
        "profile": args.profile,
        "qscale_root": str(args.qscale_root),
        "out_root": str(args.out_root),
        "proxy_run": proxy_run,
        "chunks": chunks,
        "keep_qscale_chunks": keep_chunks,
        "use_native_chunks": native_chunks,
        "rules": [
            "keep qscale only for selected-write low-support bad-write evidence with positive scale expansion and qscale observability >=0.50",
            "veto negative scale shrink at scale <=0.98",
            "hold low-observability low-gain updates",
            "risk_budget profile keeps marginal positive low-residual updates with qobs >=0.50 and overlap_residual <=0.14",
            "otherwise hold marginal positive scale updates when trajectory boundary_over_pre >=2.0 and no selected-write low-support evidence",
        ],
        "expected_result": (
            "This proxy is expected to remove qscale overlap harm by falling back to native on risky chunks, "
            "but it is not expected to satisfy the full PhaseE gate on this canary subset."
        ),
        "rows": rows,
    }
    _write_csv(args.out_root / f"{proxy_run}_rows.csv", rows)
    _write_json(args.out_root / f"{proxy_run}_summary.json", summary)
    report = [
        f"# v80 seq01 {proxy_run} audit",
        "",
        f"status: {summary['status']}",
        f"profile: {args.profile}",
        "v80_goal_achieved: false",
        "diagnostic_only: true",
        "",
        f"keep_qscale_chunks: {keep_chunks}",
        f"use_native_chunks: {native_chunks}",
        "",
        "## Rules",
        "",
        *[f"- {item}" for item in summary["rules"]],
        "",
        "## Expected result",
        "",
        str(summary["expected_result"]),
        "",
    ]
    (args.out_root / f"{proxy_run}_report.md").write_text("\n".join(report), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_root={args.out_root}")


if __name__ == "__main__":
    main()
